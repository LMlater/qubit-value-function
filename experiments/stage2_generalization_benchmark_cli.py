"""One-unit classical smoke for the Stage B fixed-validation protocol."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.experiment_utils import embedded_selected_commitments, time_window_instance  # noqa: E402
from qubit_value_function.load_scenarios import scaled_load_instance  # noqa: E402
from qubit_value_function.stage2_baselines import (  # noqa: E402
    ConstantRegressor,
    RidgeRegressor,
    fit_constant_regressor,
    fit_ridge_regressor,
)
from qubit_value_function.stage2_generalization import (  # noqa: E402
    EXTRAPOLATION_LOAD_MULTIPLIERS,
    INTERPOLATION_LOAD_MULTIPLIERS,
    TRAIN_LOAD_MULTIPLIERS,
    EDLPTruthCache,
    ModelCandidateScore,
    build_fixed_stratified_partition,
    build_state_split,
    fit_normalizers_from_fit_rows,
    make_truth_cache_key,
    regression_metrics,
    select_best_candidate,
    selection_regret,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


SMOKE_GENERATOR_PAIR = (0, 1)
SMOKE_WINDOW_START = 0
SMOKE_HORIZON = 2
ALL_LOAD_MULTIPLIERS = (
    *EXTRAPOLATION_LOAD_MULTIPLIERS[:1],
    *TRAIN_LOAD_MULTIPLIERS[:1],
    *INTERPOLATION_LOAD_MULTIPLIERS[:1],
    TRAIN_LOAD_MULTIPLIERS[1],
    INTERPOLATION_LOAD_MULTIPLIERS[1],
    TRAIN_LOAD_MULTIPLIERS[2],
    EXTRAPOLATION_LOAD_MULTIPLIERS[1],
)


def bits_from_index(index: int, *, num_bits: int = 4) -> tuple[int, ...]:
    return tuple((int(index) >> offset) & 1 for offset in range(num_bits))


@dataclass(frozen=True)
class ClassicalBaselineSelection:
    selected: ModelCandidateScore
    candidate_scores: tuple[ModelCandidateScore, ...]
    models: Mapping[str, ConstantRegressor | RidgeRegressor]


def _arrays(rows: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray([row["state_bits"] for row in rows], dtype=int)
    loads = np.asarray([row["normalized_load"] for row in rows], dtype=float)
    costs = np.asarray([row["true_cost"] for row in rows], dtype=float)
    return states, loads, costs


def select_classical_baseline(
    fit_rows: Sequence[Mapping[str, object]], validation_rows: Sequence[Mapping[str, object]]
) -> ClassicalBaselineSelection:
    """Fit and select only from fit/validation rows; test labels are not accepted."""

    fit_state, fit_load, fit_cost = _arrays(fit_rows)
    validation_state, validation_load, validation_cost = _arrays(validation_rows)
    models: dict[str, ConstantRegressor | RidgeRegressor] = {
        "constant": fit_constant_regressor(fit_cost),
        "linear_ridge": fit_ridge_regressor(fit_state, fit_load, fit_cost, degree=1, regularization=1e-3),
        "quadratic_ridge": fit_ridge_regressor(fit_state, fit_load, fit_cost, degree=2, regularization=1e-3),
    }
    scores: list[ModelCandidateScore] = []
    for order, (name, model) in enumerate(models.items()):
        prediction = model.predict(validation_state, validation_load)
        scores.append(
            ModelCandidateScore(
                name=name,
                validation_mae=regression_metrics(validation_cost, prediction)["mae"],
                validation_regret=selection_regret(
                    true_costs=validation_cost,
                    predicted_costs=prediction,
                    group_keys=[float(row["load_multiplier"]) for row in validation_rows],
                )["mean_regret"],
                parameter_count=model.parameter_count,
                order=order,
            )
        )
    score_tuple = tuple(scores)
    return ClassicalBaselineSelection(selected=select_best_candidate(score_tuple), candidate_scores=score_tuple, models=models)


def build_smoke_summary(*, instance_path: Path, split_seed: int) -> dict[str, object]:
    """Evaluate one pair/window/split with exactly 7 x 16 cached ED/LP truths."""

    source = load_uc_instance(instance_path)
    window = time_window_instance(source, start=SMOKE_WINDOW_START, horizon=SMOKE_HORIZON)
    commitments = embedded_selected_commitments(
        np.ones((len(window.generators), SMOKE_HORIZON), dtype=int), SMOKE_GENERATOR_PAIR
    )
    cache = EDLPTruthCache()
    rows: list[dict[str, object]] = []
    for multiplier in ALL_LOAD_MULTIPLIERS:
        scenario = scaled_load_instance(window, multiplier)
        evaluator = FixedCommitmentEvaluator(scenario)
        for state_index, commitment in enumerate(commitments):
            key = make_truth_cache_key(
                generator_pair=SMOKE_GENERATOR_PAIR,
                window_start=SMOKE_WINDOW_START,
                load_multiplier=multiplier,
                state_index=state_index,
            )

            def solve(commitment=commitment, multiplier=multiplier, state_index=state_index) -> float:
                result = evaluator.evaluate(commitment)
                if not result.success or not np.isfinite(result.total_cost):
                    raise RuntimeError(f"ed_lp_failed:{multiplier}:{state_index}:{result.message}")
                return float(result.total_cost)

            rows.append(
                {
                    "load_multiplier": float(multiplier),
                    "load_vector": [float(value) for value in scenario.fixed_load],
                    "state_index": int(state_index),
                    "state_bits": list(bits_from_index(state_index)),
                    "true_cost": cache.resolve(key, solve),
                }
            )
    training_indices, unseen_indices = build_state_split(int(split_seed))
    partition = build_fixed_stratified_partition(
        split_seed=int(split_seed), training_indices=training_indices, unseen_indices=unseen_indices
    )
    fit_rows = [
        row for row in rows
        if float(row["load_multiplier"]) in TRAIN_LOAD_MULTIPLIERS
        and int(row["state_index"]) in partition.fit_indices_by_load[float(row["load_multiplier"])]
    ]
    validation_rows = [
        row for row in rows
        if float(row["load_multiplier"]) in TRAIN_LOAD_MULTIPLIERS
        and int(row["state_index"]) in partition.validation_indices_by_load[float(row["load_multiplier"])]
    ]
    normalizers = fit_normalizers_from_fit_rows(fit_rows)
    for row in rows:
        row["normalized_load"] = normalizers.load.transform(row["load_vector"]).tolist()
    selection = select_classical_baseline(fit_rows, validation_rows)
    selected_model = selection.models[selection.selected.name]
    test_rows = [row for row in rows if row not in fit_rows and row not in validation_rows]
    test_state, test_load, test_cost = _arrays(test_rows)
    test_prediction = selected_model.predict(test_state, test_load)
    return {
        "status": "completed_smoke",
        "scope": "one_pair_one_window_one_split_classical_validation_smoke",
        "generator_pair": list(SMOKE_GENERATOR_PAIR),
        "window_start": SMOKE_WINDOW_START,
        "split": partition.as_dict() | {"partition_sha256": partition.partition_sha256},
        "ed_lp_evaluations": cache.solve_count,
        "fit_samples": len(fit_rows),
        "validation_samples": len(validation_rows),
        "test_samples": len(test_rows),
        "normalizers": {
            "load": normalizers.load.as_dict(),
            "target": {"mean": normalizers.target.mean, "scale": normalizers.target.scale},
        },
        "candidate_scores": [score.__dict__ for score in selection.candidate_scores],
        "selected_candidate": selection.selected.__dict__,
        "test_metrics": regression_metrics(test_cost, test_prediction),
        "test_regret": selection_regret(
            true_costs=test_cost,
            predicted_costs=test_prediction,
            group_keys=[float(row["load_multiplier"]) for row in test_rows],
        ),
        "qnn_grid_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        raise RuntimeError("only_smoke_mode_is_implemented")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing_existing_output_directory:{args.output_dir}")
    summary = build_smoke_summary(instance_path=args.instance, split_seed=args.split_seed)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
