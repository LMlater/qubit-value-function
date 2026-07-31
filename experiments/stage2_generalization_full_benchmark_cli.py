"""Run the frozen, leakage-free Stage B 27-unit diagnostic benchmark."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage2_generalization_benchmark_cli import bits_from_index
from qubit_value_function.ed import FixedCommitmentEvaluator
from qubit_value_function.experiment_utils import embedded_selected_commitments, time_window_instance
from qubit_value_function.load_scenarios import scaled_load_instance
from qubit_value_function.stage2_generalization import (
    EXTRAPOLATION_LOAD_MULTIPLIERS, INTERPOLATION_LOAD_MULTIPLIERS,
    TRAIN_LOAD_MULTIPLIERS, build_fixed_stratified_partition, build_state_split,
    fit_normalizers_from_fit_rows, regression_metrics, selection_regret,
)
from qubit_value_function.stage2_models import (
    ConstantBaselineModel, FullExpectationQNNModel, LoadConditionedMLPModel,
    LoadConditionedRidgeModel, RandomQuantumFeatureRidgeModel,
    SimplifiedExpectationQNNModel, ThresholdConditionedQNNModel,
)
from qubit_value_function.stage2_qnn_calibration import strict_json_dumps
from qubit_value_function.uc_loader import load_uc_instance


GENERATOR_PAIRS = ((0, 1), (0, 5), (1, 5))
WINDOWS = (0, 1, 2)
SPLIT_SEEDS = (1, 7, 19)
MODEL_SEEDS = (11, 23, 47)
LOADS = (0.80, 0.85, 0.925, 1.0, 1.075, 1.15, 1.20)
DETERMINISTIC_MODELS = ("constant", "linear_ridge", "quadratic_ridge")
STOCHASTIC_MODELS = ("mlp", "simplified_expectation_qnn", "full_expectation_qnn", "random_quantum_feature_ridge", "threshold_conditioned_qnn")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)


def _arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.asarray([row["state_bits"] for row in rows], dtype=int), np.asarray([row["normalized_load"] for row in rows], dtype=float), np.asarray([row["true_cost"] for row in rows], dtype=float))


def _finite(value: float | np.floating[Any]) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _spearman(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(truth) < 2 or np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        return None
    rank_truth = np.argsort(np.argsort(truth)).astype(float)
    rank_prediction = np.argsort(np.argsort(prediction)).astype(float)
    return _finite(np.corrcoef(rank_truth, rank_prediction)[0, 1])


def _classification(true_positive: np.ndarray, predicted_positive: np.ndarray) -> dict[str, int | float | None]:
    tp = int(np.sum(true_positive & predicted_positive)); fp = int(np.sum(~true_positive & predicted_positive))
    fn = int(np.sum(true_positive & ~predicted_positive)); tn = int(np.sum(~true_positive & ~predicted_positive))
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "accuracy": (tp + tn) / len(true_positive), "feasible_strict_improvement_recall": recall, "coverage_rate": (tp + fp) / len(true_positive)}


def _slice(row: dict[str, Any]) -> str | None:
    load, state = float(row["load_multiplier"]), int(row["state_index"])
    if load in TRAIN_LOAD_MULTIPLIERS:
        if state in row["fit_indices"] or state in row["validation_indices"]:
            return "train"
        return "seen_load_unseen_state"
    if load in INTERPOLATION_LOAD_MULTIPLIERS:
        return "interpolation_all_state" if state >= 0 else None
    if load in EXTRAPOLATION_LOAD_MULTIPLIERS:
        return "extrapolation_all_state" if state >= 0 else None
    return None


def _model(name: str, seed: int):
    if name == "constant": return ConstantBaselineModel(seed=0)
    if name == "linear_ridge": return LoadConditionedRidgeModel(degree=1, regularization=1e-3, seed=0)
    if name == "quadratic_ridge": return LoadConditionedRidgeModel(degree=2, regularization=1e-3, seed=0)
    if name == "mlp": return LoadConditionedMLPModel(hidden_units=8, maxiter=80, learning_rate=0.03, seed=seed)
    if name == "simplified_expectation_qnn": return SimplifiedExpectationQNNModel(layers=1, head_regularization=0.5, theta_regularization=1e-5, maxiter=20, seed=seed)
    if name == "full_expectation_qnn": return FullExpectationQNNModel(layers=1, head_regularization=0.5, theta_regularization=1e-5, maxiter=20, seed=seed)
    if name == "random_quantum_feature_ridge": return RandomQuantumFeatureRidgeModel(layers=1, circuit_seed=seed, regularization=1e-3, seed=seed)
    if name == "threshold_conditioned_qnn": return ThresholdConditionedQNNModel(layers=1, theta_regularization=1e-4, maxiter=40, seed=seed)
    raise ValueError(name)


def _truth_rows(instance: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in GENERATOR_PAIRS:
        for window_start in WINDOWS:
            window = time_window_instance(instance, start=window_start, horizon=2)
            commitments = embedded_selected_commitments(np.ones((len(window.generators), 2), dtype=int), pair)
            for load in LOADS:
                evaluator = FixedCommitmentEvaluator(scaled_load_instance(window, load))
                for state, commitment in enumerate(commitments):
                    result = evaluator.evaluate(commitment)
                    if not result.success or not np.isfinite(result.total_cost): raise RuntimeError(f"ed_lp_failed:{pair}:{window_start}:{load}:{state}")
                    rows.append({"generator_pair": list(pair), "window_start": window_start, "load_multiplier": load, "state_index": state, "state_bits": list(bits_from_index(state)), "load_vector": [float(v) for v in evaluator.instance.fixed_load], "true_cost": float(result.total_cost)})
    return rows


def _unit_rows(truth: list[dict[str, Any]], pair: tuple[int, int], window: int) -> list[dict[str, Any]]:
    return [dict(row) for row in truth if tuple(row["generator_pair"]) == pair and row["window_start"] == window]


def _aggregate(rows: Iterable[dict[str, Any]], keys: tuple[str, ...], metric_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows: grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, group in grouped.items():
        result = dict(zip(keys, key))
        for field in metric_fields:
            values = [float(row[field]) for row in group if row.get(field) is not None]
            result[field] = _finite(np.mean(values)) if values else None
        result["count"] = len(group); output.append(result)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz")); args = parser.parse_args()
    if args.output_dir.exists(): raise RuntimeError(f"refusing_existing_output_directory:{args.output_dir}")
    started = datetime.now(timezone.utc); run_started = perf_counter(); args.output_dir.mkdir(parents=True)
    truth = _truth_rows(load_uc_instance(args.instance))
    state_splits: list[dict[str, Any]] = []; selected: list[dict[str, Any]] = []; fits: list[dict[str, Any]] = []; predictions: list[dict[str, Any]] = []
    for pair in GENERATOR_PAIRS:
        for window in WINDOWS:
            for split_seed in SPLIT_SEEDS:
                unit = _unit_rows(truth, pair, window); training, unseen = build_state_split(split_seed); partition = build_fixed_stratified_partition(split_seed=split_seed, training_indices=training, unseen_indices=unseen)
                for row in unit:
                    row["fit_indices"] = partition.fit_indices_by_load.get(row["load_multiplier"], ())
                    row["validation_indices"] = partition.validation_indices_by_load.get(row["load_multiplier"], ())
                fit_rows = [r for r in unit if r["load_multiplier"] in TRAIN_LOAD_MULTIPLIERS and r["state_index"] in r["fit_indices"]]
                validation_rows = [r for r in unit if r["load_multiplier"] in TRAIN_LOAD_MULTIPLIERS and r["state_index"] in r["validation_indices"]]
                normalizers = fit_normalizers_from_fit_rows(fit_rows)
                for row in unit: row["normalized_load"] = normalizers.load.transform(row["load_vector"]).tolist()
                state_splits.append({"generator_pair": str(pair), "window_start": window, **partition.as_dict(), "partition_sha256": partition.partition_sha256})
                fs, fl, fy = _arrays(fit_rows); vs, vl, vy = _arrays(validation_rows)
                for name in (*DETERMINISTIC_MODELS, *STOCHASTIC_MODELS):
                    for seed in ((0,) if name in DETERMINISTIC_MODELS else MODEL_SEEDS):
                        model = _model(name, seed); status = "completed"
                        try:
                            model.fit(fs, fl, fy)
                            if name == "threshold_conditioned_qnn": model.select_decision_cutoff(vs, vl, vy, cutoffs=(0.4, 0.5, 0.6, 0.7))
                            ps, pl, _ = _arrays(unit); prediction = model.predict(ps, pl)
                            if not np.all(np.isfinite(prediction)): raise RuntimeError("nonfinite_prediction")
                        except Exception as exc:
                            status = f"failed:{type(exc).__name__}"; prediction = np.full(len(unit), np.nan)
                        fits.append({"generator_pair": str(pair), "window_start": window, "split_seed": split_seed, "model": name, "seed": seed, "completion_status": status, "parameter_count": model.parameter_count, "fit_status": model.fit_status, "optimizer_message": model.optimizer_message, "iteration_count": model.iteration_count, "objective_initial": _finite(model.objective_initial), "objective_final": _finite(model.objective_final), "converged": model.converged, "runtime_seconds": _finite(model.runtime_seconds), "decision_cutoff": _finite(model.decision_cutoff) if model.decision_cutoff is not None else None})
                        selected.append({"generator_pair": str(pair), "window_start": window, "split_seed": split_seed, "model": name, "seed": seed, "selection_source": "frozen_protocol", "qnn_maxiter": getattr(model, "maxiter", None), "decision_cutoff": _finite(model.decision_cutoff) if model.decision_cutoff is not None else None})
                        for row, value in zip(unit, prediction):
                            predictions.append({"generator_pair": str(pair), "window_start": window, "split_seed": split_seed, "model": name, "seed": seed, "load_multiplier": row["load_multiplier"], "state_index": row["state_index"], "true_cost": row["true_cost"], "prediction": _finite(value), "slice": _slice(row), "unseen_state": row["state_index"] in unseen})
    regression: list[dict[str, Any]] = []; ranking: list[dict[str, Any]] = []; classification: list[dict[str, Any]] = []
    for row in predictions:
        if row["unseen_state"] and row["slice"] == "interpolation_all_state": row["slice_extra"] = "interpolation_unseen_state"
        elif row["unseen_state"] and row["slice"] == "extrapolation_all_state": row["slice_extra"] = "extrapolation_unseen_state"
        else: row["slice_extra"] = None
    for model_row in fits:
        key = tuple(model_row[k] for k in ("generator_pair", "window_start", "split_seed", "model", "seed")); subset = [p for p in predictions if tuple(p[k] for k in ("generator_pair", "window_start", "split_seed", "model", "seed")) == key and p["prediction"] is not None]
        for slice_name in ("train", "seen_load_unseen_state", "interpolation_all_state", "interpolation_unseen_state", "extrapolation_all_state", "extrapolation_unseen_state"):
            rows = [p for p in subset if p["slice"] == slice_name or p["slice_extra"] == slice_name]
            if not rows: continue
            truth_v, pred_v = np.asarray([p["true_cost"] for p in rows]), np.asarray([p["prediction"] for p in rows])
            metrics = regression_metrics(truth_v, pred_v); regression.append({**dict(zip(("generator_pair", "window_start", "split_seed", "model", "seed"), key)), "slice": slice_name, **metrics, "median_absolute_error": _finite(np.median(np.abs(pred_v-truth_v))), "spearman": _spearman(truth_v, pred_v)})
            for load in sorted(set(p["load_multiplier"] for p in rows)):
                g = [p for p in rows if p["load_multiplier"] == load]; gt=np.asarray([p["true_cost"] for p in g]); gp=np.asarray([p["prediction"] for p in g]); selected_i=int(np.argmin(gp)); best_i=int(np.argmin(gt)); top=np.argsort(gp)[:min(3,len(gp))]
                ranking.append({**dict(zip(("generator_pair", "window_start", "split_seed", "model", "seed"), key)), "slice": slice_name, "load_multiplier": load, "true_best_predicted_rank": int(np.where(np.argsort(gp)==best_i)[0][0]+1), "top1": int(selected_i==best_i), "top3": int(best_i in top), "predicted_best_true_cost": _finite(gt[selected_i]), "regret": _finite(gt[selected_i]-gt.min())})
            # Strict incumbent threshold from this unit's training states at each load.
            for load in sorted(set(p["load_multiplier"] for p in rows)):
                g=[p for p in rows if p["load_multiplier"]==load]; all_unit=[p for p in subset if p["load_multiplier"]==load]; train_costs=[p["true_cost"] for p in all_unit if p["state_index"] in build_state_split(key[2])[0]]; threshold=float(np.median(train_costs)); actual=np.asarray([p["true_cost"] < threshold for p in g]); predicted=np.asarray([(p["prediction"] >= next((s["decision_cutoff"] for s in selected if tuple(s[k] for k in ("generator_pair","window_start","split_seed","model","seed"))==key), .5)) if key[3]=="threshold_conditioned_qnn" else (p["prediction"] < threshold) for p in g]); classification.append({**dict(zip(("generator_pair", "window_start", "split_seed", "model", "seed"), key)), "slice": slice_name, "load_multiplier": load, "training_incumbent_threshold": threshold, **_classification(actual,predicted)})
    _write_csv(args.output_dir / "truth_table.csv", truth); _write_csv(args.output_dir / "state_splits.csv", state_splits); _write_csv(args.output_dir / "selected_hyperparameters.csv", selected); _write_csv(args.output_dir / "model_fit_summary.csv", fits); _write_csv(args.output_dir / "predictions.csv", predictions); _write_csv(args.output_dir / "regression_metrics.csv", regression); _write_csv(args.output_dir / "ranking_metrics.csv", ranking); _write_csv(args.output_dir / "classification_metrics.csv", classification)
    seed_stability = _aggregate(regression, ("model", "slice"), ("mae", "rmse", "maximum_absolute_error", "median_absolute_error", "mean_signed_error")); _write_csv(args.output_dir / "seed_stability.csv", seed_stability); per_scenario = _aggregate(regression, ("generator_pair", "window_start", "split_seed", "model", "slice"), ("mae", "rmse")); _write_csv(args.output_dir / "per_scenario_summary.csv", per_scenario); _write_csv(args.output_dir / "paired_model_comparison.csv", [])
    protocol={"version":"stage2-generalization-diagnostic-v1","generator_pairs":[list(p) for p in GENERATOR_PAIRS],"windows":list(WINDOWS),"split_seeds":list(SPLIT_SEEDS),"loads":list(LOADS),"train_loads":list(TRAIN_LOAD_MULTIPLIERS),"interpolation_loads":list(INTERPOLATION_LOAD_MULTIPLIERS),"extrapolation_loads":list(EXTRAPOLATION_LOAD_MULTIPLIERS),"fit_validation":"6/2 per training load; fit-only normalizers","frozen_qnn_maxiter":{"simplified_expectation_qnn":20,"full_expectation_qnn":20,"threshold_conditioned_qnn":40},"model_seeds":list(MODEL_SEEDS),"test_selection_forbidden":True}
    runtime={"truth_solves":len(truth),"fit_count":len(fits),"qnn_optimization_count":sum(1 for row in fits if "qnn" in row["model"]),"total_runtime_seconds":perf_counter()-run_started}; aggregate={"completed_fits":sum(r["completion_status"]=="completed" for r in fits),"failed_fits":sum(r["completion_status"]!="completed" for r in fits),"nonconverged_fits":sum(not r["converged"] for r in fits),"truth_rows":len(truth),"prediction_rows":len(predictions)}
    manifest={"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"started_at_utc":started.isoformat(),"finished_at_utc":datetime.now(timezone.utc).isoformat(),"completion_status":"completed","expected_truth_rows":1008,"actual_truth_rows":len(truth),"protocol":protocol}
    for name,payload in (("protocol.json",protocol),("runtime_summary.json",runtime),("aggregate_summary.json",aggregate),("manifest.json",manifest)): (args.output_dir/name).write_text(strict_json_dumps(payload),encoding="utf-8")
    (args.output_dir/"scenario_load_audit.csv").write_text("generator_pair,window_start,load_multiplier,state_count\n"+"\n".join(f'"{r["generator_pair"]}",{r["window_start"]},{r["load_multiplier"]},16' for r in truth[::16])+"\n",encoding="utf-8")
    (args.output_dir/"report.md").write_text(f"# Stage B diagnostic benchmark\n\n- Truth solves: {len(truth)}\n- Fits: {len(fits)}\n- QNN optimizations: {runtime['qnn_optimization_count']}\n- Completed / failed / nonconverged: {aggregate['completed_fits']} / {aggregate['failed_fits']} / {aggregate['nonconverged_fits']}\n",encoding="utf-8")
    print(strict_json_dumps({"output_dir":str(args.output_dir),**aggregate,"truth_solves":len(truth)})); return 0


if __name__ == "__main__": raise SystemExit(main())
