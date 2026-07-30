"""Stage-B diagnostic pilot for load-conditioned value models.

This pilot deliberately separates three questions:
1. Can a low-order load-conditioned surrogate generalize to unseen states/loads?
2. Can a conventional data-reuploading expectation QNN improve that regression?
3. Can a threshold-conditioned QNN learn a useful soft improvement predicate?

The QNNs use an exact NumPy statevector. This is not Aer MPS, hardware evidence,
or a Grover integration result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.commitment import is_logic_feasible  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.experiment_utils import (  # noqa: E402
    embedded_selected_commitments,
    time_window_instance,
)
from qubit_value_function.load_conditioned_qnn import (  # noqa: E402
    ExpectationQNNConfig,
    ThresholdQNNConfig,
    fit_expectation_qnn,
    fit_threshold_qnn,
)
from qubit_value_function.load_scenarios import (  # noqa: E402
    fit_load_normalizer,
    scaled_load_instance,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


LOAD_MULTIPLIERS = (0.85, 0.925, 1.0, 1.075, 1.15)
TRAIN_LOAD_MULTIPLIERS = frozenset((0.85, 1.0, 1.15))
TEST_LOAD_MULTIPLIERS = frozenset((0.925, 1.075))
SELECTED_GENERATORS = (0, 5)
WINDOW_START = 0
HORIZON = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage-B load-conditioned QNN pilot; diagnostic, not formal advantage evidence."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--qnn-maxiter", type=int, default=20)
    parser.add_argument("--classifier-maxiter", type=int, default=15)
    parser.add_argument("--smoke", action="store_true")
    return parser


def bits_from_index(index: int, num_bits: int = 4) -> tuple[int, ...]:
    return tuple((int(index) >> offset) & 1 for offset in range(int(num_bits)))


def ensure_new_output_dir(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_existing_output_directory:{path}")
    path.mkdir(parents=True)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([_json_safe(dict(row)) for row in rows])


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def quadratic_features(
    state_bits: Sequence[Sequence[int]],
    normalized_loads: Sequence[Sequence[float]],
) -> np.ndarray:
    rows: list[list[float]] = []
    for raw_bits, raw_load in zip(state_bits, normalized_loads):
        bits = np.asarray(raw_bits, dtype=float)
        load = np.asarray(raw_load, dtype=float)
        row = [1.0, *bits, *load]
        row.extend(bits[i] * bits[j] for i in range(4) for j in range(i + 1, 4))
        row.extend(bits[i] * load[t] for i in range(4) for t in range(2))
        row.extend((load[0] ** 2, load[1] ** 2, load[0] * load[1]))
        rows.append([float(value) for value in row])
    return np.asarray(rows, dtype=float)


def fit_ridge(
    state_bits: np.ndarray,
    normalized_loads: np.ndarray,
    costs: np.ndarray,
    regularization: float = 1e-6,
) -> np.ndarray:
    design = quadratic_features(state_bits, normalized_loads)
    penalty = np.eye(design.shape[1], dtype=float) * float(regularization)
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ costs)


def regression_metrics(costs: Sequence[float], predictions: Sequence[float]) -> dict[str, object]:
    truth = np.asarray(costs, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    error = predicted - truth
    rank = spearmanr(truth, predicted).statistic if len(np.unique(truth)) > 1 else None
    return {
        "count": int(len(truth)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "mean_signed_error": float(np.mean(error)),
        "spearman": None if rank is None or not np.isfinite(rank) else float(rank),
    }


def confusion_metrics(
    labels: Sequence[int | bool], probabilities: Sequence[float], cutoff: float
) -> dict[str, object]:
    truth = np.asarray(labels, dtype=bool)
    predicted = np.asarray(probabilities, dtype=float) >= float(cutoff)
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    return {
        "cutoff": float(cutoff),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "accuracy": float((tp + tn) / len(truth)),
    }


def threshold_regression_metrics(
    rows: Sequence[Mapping[str, object]],
    predictions: Sequence[float],
    training_indices: frozenset[int],
) -> dict[str, object]:
    predicted = np.asarray(predictions, dtype=float)
    outcomes: list[tuple[bool, bool]] = []
    for multiplier in LOAD_MULTIPLIERS:
        positions = [
            position
            for position, row in enumerate(rows)
            if float(row["load_multiplier"]) == multiplier
        ]
        threshold = min(
            float(rows[position]["true_cost"])
            for position in positions
            if int(rows[position]["state_index"]) in training_indices
        )
        for position in positions:
            if int(rows[position]["state_index"]) in training_indices:
                continue
            outcomes.append(
                (
                    bool(predicted[position] < threshold),
                    bool(float(rows[position]["true_cost"]) < threshold),
                )
            )
    tp = sum(marked and improved for marked, improved in outcomes)
    fp = sum(marked and not improved for marked, improved in outcomes)
    fn = sum(not marked and improved for marked, improved in outcomes)
    tn = sum(not marked and not improved for marked, improved in outcomes)
    return {
        "statistical_level": "out_of_training_state_across_all_loads",
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
    }


def main() -> int:
    args = build_parser().parse_args()
    ensure_new_output_dir(args.output_dir)

    source = load_uc_instance(args.instance)
    window = time_window_instance(source, start=WINDOW_START, horizon=HORIZON)
    base_commitment = np.ones((len(window.generators), HORIZON), dtype=int)
    commitments = embedded_selected_commitments(base_commitment, SELECTED_GENERATORS)

    dataset: list[dict[str, object]] = []
    for multiplier in LOAD_MULTIPLIERS:
        scenario = scaled_load_instance(window, multiplier)
        evaluator = FixedCommitmentEvaluator(scenario)
        for state_index, commitment in enumerate(commitments):
            result = evaluator.evaluate(commitment)
            if not result.success or not np.isfinite(result.total_cost):
                raise RuntimeError(
                    f"ed_lp_failed:multiplier={multiplier}:state={state_index}:{result.message}"
                )
            dataset.append(
                {
                    "load_multiplier": float(multiplier),
                    "load_t0": float(scenario.fixed_load[0]),
                    "load_t1": float(scenario.fixed_load[1]),
                    "state_index": int(state_index),
                    "bitstring": "".join(str(value) for value in bits_from_index(state_index)),
                    "hard_logic_feasible": bool(is_logic_feasible(scenario, commitment)),
                    "true_cost": float(result.total_cost),
                }
            )

    rng = np.random.default_rng(int(args.split_seed))
    training_indices = frozenset(
        int(value) for value in sorted(rng.choice(16, size=8, replace=False).tolist())
    )
    training_rows = [
        row
        for row in dataset
        if float(row["load_multiplier"]) in TRAIN_LOAD_MULTIPLIERS
        and int(row["state_index"]) in training_indices
    ]
    normalizer = fit_load_normalizer(
        [[float(row["load_t0"]), float(row["load_t1"])] for row in training_rows]
    )

    def arrays(rows: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = np.asarray(
            [bits_from_index(int(row["state_index"])) for row in rows], dtype=int
        )
        load = np.vstack(
            [normalizer.transform([float(row["load_t0"]), float(row["load_t1"])]) for row in rows]
        )
        cost = np.asarray([float(row["true_cost"]) for row in rows], dtype=float)
        return state, load, cost

    train_state, train_load, train_cost = arrays(training_rows)
    ridge_weights = fit_ridge(train_state, train_load, train_cost)

    qnn_iterations = 2 if args.smoke else int(args.qnn_maxiter)
    classifier_iterations = 2 if args.smoke else int(args.classifier_maxiter)
    regression_fit = fit_expectation_qnn(
        state_bits=train_state,
        normalized_loads=train_load,
        costs=train_cost,
        config=ExpectationQNNConfig(
            layers=2,
            head_regularization=0.5,
            theta_regularization=1e-5,
            maxiter=qnn_iterations,
            seed=int(args.model_seed),
        ),
    )

    groups = {
        "train": training_rows,
        "seen_load_unseen_state": [
            row
            for row in dataset
            if float(row["load_multiplier"]) in TRAIN_LOAD_MULTIPLIERS
            and int(row["state_index"]) not in training_indices
        ],
        "unseen_load_all_state": [
            row
            for row in dataset
            if float(row["load_multiplier"]) in TEST_LOAD_MULTIPLIERS
        ],
        "unseen_load_unseen_state": [
            row
            for row in dataset
            if float(row["load_multiplier"]) in TEST_LOAD_MULTIPLIERS
            and int(row["state_index"]) not in training_indices
        ],
    }

    model_rows: list[dict[str, object]] = []
    for model_name in ("load_conditioned_quadratic_ridge", "expectation_qnn"):
        for group_name, group_rows in groups.items():
            state, load, cost = arrays(group_rows)
            prediction = (
                quadratic_features(state, load) @ ridge_weights
                if model_name == "load_conditioned_quadratic_ridge"
                else regression_fit.model.predict(state, load)
            )
            model_rows.append(
                {"model": model_name, "evaluation_group": group_name, **regression_metrics(cost, prediction)}
            )

    all_state, all_load, all_cost = arrays(dataset)
    ridge_all_prediction = quadratic_features(all_state, all_load) @ ridge_weights
    qnn_all_prediction = regression_fit.model.predict(all_state, all_load)
    regression_threshold = {
        "load_conditioned_quadratic_ridge": threshold_regression_metrics(
            dataset, ridge_all_prediction, training_indices
        ),
        "expectation_qnn": threshold_regression_metrics(
            dataset, qnn_all_prediction, training_indices
        ),
    }

    unseen_top1: dict[str, dict[str, int]] = {}
    for model_name, predictions in (
        ("load_conditioned_quadratic_ridge", ridge_all_prediction),
        ("expectation_qnn", qnn_all_prediction),
    ):
        hits = 0
        for multiplier in TEST_LOAD_MULTIPLIERS:
            positions = [
                position
                for position, row in enumerate(dataset)
                if float(row["load_multiplier"]) == multiplier
            ]
            predicted_best = positions[int(np.argmin(predictions[positions]))]
            true_best = positions[int(np.argmin(all_cost[positions]))]
            hits += int(predicted_best == true_best)
        unseen_top1[model_name] = {"hits": int(hits), "total": len(TEST_LOAD_MULTIPLIERS)}

    threshold_training_samples: list[tuple[Mapping[str, object], float, int]] = []
    training_cost_center = float(train_cost.mean())
    training_cost_scale = float(max(train_cost.std(), 1.0))
    for multiplier in TRAIN_LOAD_MULTIPLIERS:
        scenario_rows = [
            row for row in dataset if float(row["load_multiplier"]) == multiplier
        ]
        scenario_costs = np.asarray(
            [float(row["true_cost"]) for row in scenario_rows], dtype=float
        )
        for threshold in np.quantile(scenario_costs, (0.2, 0.4, 0.6, 0.8)):
            normalized_threshold = (float(threshold) - training_cost_center) / training_cost_scale
            for row in scenario_rows:
                if int(row["state_index"]) in training_indices:
                    threshold_training_samples.append(
                        (row, normalized_threshold, int(float(row["true_cost"]) < threshold))
                    )

    classifier_state = np.asarray(
        [bits_from_index(int(row["state_index"])) for row, _, _ in threshold_training_samples],
        dtype=int,
    )
    classifier_load = np.vstack(
        [
            normalizer.transform([float(row["load_t0"]), float(row["load_t1"])])
            for row, _, _ in threshold_training_samples
        ]
    )
    classifier_threshold = np.asarray(
        [threshold for _, threshold, _ in threshold_training_samples], dtype=float
    )
    classifier_label = np.asarray(
        [label for _, _, label in threshold_training_samples], dtype=int
    )
    classifier_fit = fit_threshold_qnn(
        state_bits=classifier_state,
        normalized_loads=classifier_load,
        normalized_thresholds=classifier_threshold,
        labels=classifier_label,
        config=ThresholdQNNConfig(
            layers=2,
            theta_regularization=1e-4,
            maxiter=classifier_iterations,
            seed=int(args.model_seed),
        ),
    )

    classifier_test: list[tuple[Mapping[str, object], float, int]] = []
    for multiplier in TEST_LOAD_MULTIPLIERS:
        scenario_rows = [
            row for row in dataset if float(row["load_multiplier"]) == multiplier
        ]
        threshold = min(
            float(row["true_cost"])
            for row in scenario_rows
            if int(row["state_index"]) in training_indices
        )
        normalized_threshold = (threshold - training_cost_center) / training_cost_scale
        for row in scenario_rows:
            if int(row["state_index"]) not in training_indices:
                classifier_test.append(
                    (row, normalized_threshold, int(float(row["true_cost"]) < threshold))
                )

    test_state = np.asarray(
        [bits_from_index(int(row["state_index"])) for row, _, _ in classifier_test],
        dtype=int,
    )
    test_load = np.vstack(
        [
            normalizer.transform([float(row["load_t0"]), float(row["load_t1"])])
            for row, _, _ in classifier_test
        ]
    )
    test_threshold = np.asarray([value for _, value, _ in classifier_test], dtype=float)
    test_label = np.asarray([label for _, _, label in classifier_test], dtype=int)
    test_probability = classifier_fit.model.predict_proba(
        test_state, test_load, test_threshold
    )
    classifier_cutoffs = [
        confusion_metrics(test_label, test_probability, cutoff)
        for cutoff in (0.4, 0.5, 0.6, 0.7)
    ]

    for row, ridge_prediction, qnn_prediction in zip(
        dataset, ridge_all_prediction, qnn_all_prediction
    ):
        row["ridge_prediction"] = float(ridge_prediction)
        row["qnn_prediction"] = float(qnn_prediction)
        row["in_training_state_split"] = bool(int(row["state_index"]) in training_indices)
        row["in_training_load_split"] = bool(
            float(row["load_multiplier"]) in TRAIN_LOAD_MULTIPLIERS
        )

    summary = {
        "status": "completed_smoke" if args.smoke else "completed_pilot",
        "scope": "case14_4bit_2period_load_conditioned_diagnostic",
        "selected_generators": list(SELECTED_GENERATORS),
        "window_start": WINDOW_START,
        "horizon": HORIZON,
        "load_multipliers": list(LOAD_MULTIPLIERS),
        "training_load_multipliers": sorted(TRAIN_LOAD_MULTIPLIERS),
        "test_load_multipliers": sorted(TEST_LOAD_MULTIPLIERS),
        "training_state_indices": sorted(training_indices),
        "training_samples": len(training_rows),
        "ed_lp_evaluations": len(dataset),
        "load_normalizer": normalizer.as_dict(),
        "regression_threshold_metrics": regression_threshold,
        "unseen_load_top1": unseen_top1,
        "threshold_qnn_test_samples": len(classifier_test),
        "threshold_qnn_positive_test_samples": int(test_label.sum()),
        "threshold_qnn_cutoff_metrics": classifier_cutoffs,
        "limitations": [
            "exact NumPy statevector, not Aer MPS or quantum hardware",
            "single generator pair and one two-period window",
            "pilot split and hyperparameters are diagnostic, not formal model selection",
            "threshold QNN is not yet integrated into coherent amplitude amplification",
        ],
    }

    write_csv(args.output_dir / "dataset_predictions.csv", dataset)
    write_csv(args.output_dir / "regression_metrics.csv", model_rows)
    write_json(args.output_dir / "expectation_qnn_fit.json", regression_fit.as_dict())
    write_json(args.output_dir / "threshold_qnn_fit.json", classifier_fit.as_dict())
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "manifest.json",
        {
            "script": "experiments/stage2_case14_load_qnn_pilot_cli.py",
            "instance": str(args.instance),
            "split_seed": int(args.split_seed),
            "model_seed": int(args.model_seed),
            "qnn_maxiter": qnn_iterations,
            "classifier_maxiter": classifier_iterations,
            "smoke": bool(args.smoke),
            "no_grover_mps_or_hardware": True,
        },
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
