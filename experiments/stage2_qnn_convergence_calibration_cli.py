"""Run the frozen Stage B QNN convergence calibration on fit/validation only."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage2_generalization_benchmark_cli import (  # noqa: E402
    SMOKE_GENERATOR_PAIR,
    SMOKE_HORIZON,
    SMOKE_WINDOW_START,
    bits_from_index,
)
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.experiment_utils import (  # noqa: E402
    embedded_selected_commitments,
    time_window_instance,
)
from qubit_value_function.load_scenarios import scaled_load_instance  # noqa: E402
from qubit_value_function.stage2_generalization import (  # noqa: E402
    EDLPTruthCache,
    TRAIN_LOAD_MULTIPLIERS,
    build_fixed_stratified_partition,
    build_state_split,
    fit_normalizers_from_fit_rows,
    make_truth_cache_key,
    regression_metrics,
    selection_regret,
)
from qubit_value_function.stage2_models import (  # noqa: E402
    FullExpectationQNNModel,
    SimplifiedExpectationQNNModel,
    ThresholdConditionedQNNModel,
)
from qubit_value_function.stage2_qnn_calibration import (  # noqa: E402
    ITERATION_BUDGETS,
    MODEL_SEEDS,
    calibration_grid,
    select_frozen_config,
    strict_json_dumps,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


EXPECTED_PARTITION_SHA256 = "465b0cc1f3dbd5e857dfda8b027ad9c4d70812d838299d98e8f64accf101e86a"
CALIBRATION_VERSION = "stage2-qnn-convergence-calibration-v2"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray, cutoff: float) -> dict[str, float | int | None]:
    predicted = probabilities >= float(cutoff)
    tp = int(np.sum(predicted & labels)); fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels)); tn = int(np.sum(~predicted & ~labels))
    precision = None if tp + fp == 0 else float(tp / (tp + fp))
    recall = None if tp + fn == 0 else float(tp / (tp + fn))
    f1 = 0.0 if precision is None or recall is None or precision + recall == 0.0 else float(2.0 * precision * recall / (precision + recall))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "accuracy": float((tp + tn) / len(labels))}


def _arrays(rows: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([row["state_bits"] for row in rows], dtype=int),
        np.asarray([row["normalized_load"] for row in rows], dtype=float),
        np.asarray([row["true_cost"] for row in rows], dtype=float),
    )


def build_fit_validation_rows(instance_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Build only the 18 fit and 6 validation rows; no test load is created."""

    source = load_uc_instance(instance_path)
    window = time_window_instance(source, start=SMOKE_WINDOW_START, horizon=SMOKE_HORIZON)
    commitments = embedded_selected_commitments(np.ones((len(window.generators), SMOKE_HORIZON), dtype=int), SMOKE_GENERATOR_PAIR)
    cache = EDLPTruthCache(); rows: list[dict[str, object]] = []
    for multiplier in TRAIN_LOAD_MULTIPLIERS:
        scenario = scaled_load_instance(window, multiplier)
        evaluator = FixedCommitmentEvaluator(scenario)
        for state_index, commitment in enumerate(commitments):
            key = make_truth_cache_key(generator_pair=SMOKE_GENERATOR_PAIR, window_start=SMOKE_WINDOW_START, load_multiplier=multiplier, state_index=state_index)
            def solve(commitment=commitment, evaluator=evaluator) -> float:
                result = evaluator.evaluate(commitment)
                if not result.success or not np.isfinite(result.total_cost):
                    raise RuntimeError(f"ed_lp_failed:{result.message}")
                return float(result.total_cost)
            rows.append({"load_multiplier": float(multiplier), "load_vector": [float(v) for v in scenario.fixed_load], "state_index": state_index, "state_bits": list(bits_from_index(state_index)), "true_cost": cache.resolve(key, solve)})
    training, unseen = build_state_split(1)
    partition = build_fixed_stratified_partition(split_seed=1, training_indices=training, unseen_indices=unseen)
    if partition.partition_sha256 != EXPECTED_PARTITION_SHA256:
        raise RuntimeError(f"unexpected_partition_sha256:{partition.partition_sha256}")
    fit = [row for row in rows if row["state_index"] in partition.fit_indices_by_load[row["load_multiplier"]]]
    validation = [row for row in rows if row["state_index"] in partition.validation_indices_by_load[row["load_multiplier"]]]
    normalizers = fit_normalizers_from_fit_rows(fit)
    for row in rows:
        row["normalized_load"] = normalizers.load.transform(row["load_vector"]).tolist()
    return fit, validation, {"partition": partition.as_dict() | {"partition_sha256": partition.partition_sha256}, "ed_lp_solves": cache.solve_count, "normalizers": {"load": normalizers.load.as_dict(), "target": {"mean": normalizers.target.mean, "scale": normalizers.target.scale}}}


def _model(name: str, maxiter: int, seed: int):
    if name == "simplified_expectation_qnn":
        return SimplifiedExpectationQNNModel(layers=1, head_regularization=0.5, theta_regularization=1e-5, maxiter=maxiter, seed=seed)
    if name == "full_expectation_qnn":
        return FullExpectationQNNModel(layers=1, head_regularization=0.5, theta_regularization=1e-5, maxiter=maxiter, seed=seed)
    if name == "threshold_conditioned_qnn":
        return ThresholdConditionedQNNModel(layers=1, theta_regularization=1e-4, maxiter=maxiter, seed=seed)
    raise ValueError(name)


def run_calibration(fit_rows: Sequence[Mapping[str, object]], validation_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    fit_state, fit_load, fit_cost = _arrays(fit_rows); validation_state, validation_load, validation_cost = _arrays(validation_rows)
    runs: list[dict[str, object]] = []
    for name, maxiter, seed in calibration_grid():
        model = _model(name, maxiter, seed).fit(fit_state, fit_load, fit_cost)
        fit_prediction = model.predict(fit_state, fit_load); validation_prediction = model.predict(validation_state, validation_load)
        record: dict[str, object] = {
            "model": name, "maxiter": maxiter, "seed": seed, "parameter_count": model.parameter_count,
            "iteration_count": model.iteration_count, "fit_status": model.fit_status, "optimizer_message": model.optimizer_message,
            "converged": model.converged, "objective_initial": model.objective_initial, "objective_final": model.objective_final,
            "objective_relative_decrease": (model.objective_initial - model.objective_final) / max(abs(model.objective_initial), 1e-12),
            "runtime_seconds": model.runtime_seconds, "has_nonfinite": False,
            "fit_mae": None, "validation_mae": None, "validation_regret": None,
            "validation_precision": None, "validation_recall": None, "validation_f1": None, "validation_accuracy": None, "validation_cutoff": None,
        }
        if name == "threshold_conditioned_qnn":
            cutoff = model.select_decision_cutoff(validation_state, validation_load, validation_cost, cutoffs=(0.4, 0.5, 0.6, 0.7))
            metrics = _classification_metrics(validation_cost <= model.training_threshold, validation_prediction, cutoff)
            record.update(validation_precision=metrics["precision"], validation_recall=metrics["recall"], validation_f1=metrics["f1"], validation_accuracy=metrics["accuracy"], validation_cutoff=cutoff, training_threshold=model.training_threshold)
        else:
            record.update(fit_mae=regression_metrics(fit_cost, fit_prediction)["mae"], validation_mae=regression_metrics(validation_cost, validation_prediction)["mae"], validation_regret=selection_regret(true_costs=validation_cost, predicted_costs=validation_prediction, group_keys=[row["load_multiplier"] for row in validation_rows])["mean_regret"])
        finite = [value for value in record.values() if isinstance(value, float)]
        record["has_nonfinite"] = not all(np.isfinite(finite))
        if record["has_nonfinite"]:
            raise RuntimeError("nonfinite_calibration_record")
        runs.append(record)
    return runs


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise"); writer.writeheader(); writer.writerows(rows)


def _csv_value(value: str) -> object:
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_completed_calibration_runs(path: Path) -> list[dict[str, object]]:
    """Load an existing completed calibration CSV without rerunning any optimizer."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [{key: _csv_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    if len(rows) != len(calibration_grid()):
        raise ValueError(f"unexpected_calibration_run_count:{len(rows)}")
    return rows


def _selection_protocol() -> dict[str, object]:
    return {
        "selection_version": "stage2-qnn-convergence-qualified-v1",
        "eligibility": {
            "required_model_seeds": list(MODEL_SEEDS),
            "all_seeds_completed": True,
            "all_seeds_converged": True,
            "optimizer_failed_disallowed": True,
            "nonfinite_disallowed": True,
            "validation_metrics_must_be_computable": True,
        },
        "metric_only_selection": "historical diagnostic only; must not authorize a complete pilot",
        "regression_tie_break": ["validation_mae", "validation_regret", "lower_budget", "candidate_order"],
        "threshold_tie_break": ["validation_f1_desc", "validation_recall_desc", "validation_precision_desc", "lower_budget", "candidate_order"],
        "cutoff_candidates": [0.4, 0.5, 0.6, 0.7],
        "cutoff_data": "validation_only",
        "frozen_qualified_budgets": {
            "simplified_expectation_qnn": 20,
            "full_expectation_qnn": 20,
            "threshold_conditioned_qnn": 40,
        },
        "optimizer": {"method": "L-BFGS-B", "ftol": 1e-10, "gtol": 1e-6, "maxls": 20},
        "regularization": {
            "simplified_expectation_qnn": {"head": 0.5, "theta": 1e-5},
            "full_expectation_qnn": {"head": 0.5, "theta": 1e-5},
            "threshold_conditioned_qnn": {"theta": 1e-4},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--summarize-runs", type=Path, help="Re-summarize a completed calibration CSV without running ED/LP or QNN optimization.")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing_existing_output_directory:{args.output_dir}")
    started = datetime.now(timezone.utc)
    source_record: dict[str, object] | None = None
    if args.summarize_runs is None:
        fit, validation, data_summary = build_fit_validation_rows(args.instance)
        runs = run_calibration(fit, validation)
    else:
        fit = []; validation = []
        runs = load_completed_calibration_runs(args.summarize_runs)
        source_record = {
            "completed_calibration_csv": args.summarize_runs.name,
            "completed_calibration_sha256": hashlib.sha256(args.summarize_runs.read_bytes()).hexdigest(),
            "reran_optimizers": False,
        }
        data_summary = {"re_summary_only": True, "source": source_record}
    selected = [select_frozen_config(runs, model=name) for name in ("simplified_expectation_qnn", "full_expectation_qnn", "threshold_conditioned_qnn")]
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "calibration_runs.csv", runs)
    _write_csv(args.output_dir / "per_model_budget_summary.csv", [{"model": item["model"]} | summary for item in selected for summary in item["per_budget"]])
    (args.output_dir / "protocol.json").write_text(strict_json_dumps({"version": CALIBRATION_VERSION, "partition_algorithm": "stage2-fit-validation-v1", "expected_partition_sha256": EXPECTED_PARTITION_SHA256, "train_load_multipliers": list(TRAIN_LOAD_MULTIPLIERS), "iteration_budgets": list(ITERATION_BUDGETS), "model_seeds": list(MODEL_SEEDS), "selection": _selection_protocol()}), encoding="utf-8")
    (args.output_dir / "selected_frozen_config.json").write_text(strict_json_dumps({"selected": selected, "formal_selection": {item["model"]: item["convergence_qualified_selection"] for item in selected}, "metric_only_selection": {item["model"]: item["metric_only_selection"] for item in selected}, "saturation_rule": "report-only; never overrides convergence eligibility"}), encoding="utf-8")
    runtime = {"total_runtime_seconds": float(sum(float(row["runtime_seconds"]) for row in runs)), "run_count": len(runs), "by_model_seconds": {name: float(sum(float(row["runtime_seconds"]) for row in runs if row["model"] == name)) for name in ("simplified_expectation_qnn", "full_expectation_qnn", "threshold_conditioned_qnn")}}
    (args.output_dir / "runtime_summary.json").write_text(strict_json_dumps(runtime), encoding="utf-8")
    finished = datetime.now(timezone.utc)
    manifest = {"git_commit": _git_head(), "working_tree_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(), "python": sys.version, "numpy": np.__version__, "scipy": __import__("scipy").__version__, "development_unit": {"generator_pair": list(SMOKE_GENERATOR_PAIR), "window_start": SMOKE_WINDOW_START, "split_seed": 1, "fit_samples": len(fit), "validation_samples": len(validation)}, "partition_sha256": EXPECTED_PARTITION_SHA256, "model_seeds": list(MODEL_SEEDS), "iteration_budgets": list(ITERATION_BUDGETS), "started_at_utc": started.isoformat(), "finished_at_utc": finished.isoformat(), "completion_status": "completed", "data_summary": data_summary, "source_record": source_record}
    (args.output_dir / "manifest.json").write_text(strict_json_dumps(manifest), encoding="utf-8")
    lines = ["# Stage B QNN convergence calibration", "", "Pure fit/validation development unit; no interpolation, extrapolation, or unseen-state test metric was generated.", "", f"- runs: {len(runs)}", f"- partition SHA-256: `{EXPECTED_PARTITION_SHA256}`", f"- total optimizer runtime: {runtime['total_runtime_seconds']:.3f} s", "", "## Frozen selections", ""]
    lines.extend(f"- {item['model']}: metric-only maxiter {item['metric_only_selection']['maxiter'] if item['metric_only_selection'] else 'n/a'}; convergence-qualified maxiter {item['convergence_qualified_selection']['maxiter'] if item['convergence_qualified_selection'] else 'blocked'}" for item in selected)
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(strict_json_dumps({"output_dir": str(args.output_dir), "run_count": len(runs), "selected": selected, "runtime": runtime}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
