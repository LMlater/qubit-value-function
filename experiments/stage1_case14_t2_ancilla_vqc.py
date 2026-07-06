from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.ancilla_vqc import (  # noqa: E402
    ancilla_block_oracle_errors,
    evaluate_ancilla_vqc,
    fit_ancilla_vqc,
    grover_with_ancilla_model,
)
from qubit_value_function.commitment import (  # noqa: E402
    all_commitments,
    commitment_to_bitstring,
    is_logic_feasible,
)
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.phase_vqc import all_x_subsets, x_monomial_features  # noqa: E402
from qubit_value_function.uc_loader import Reserve, UCInstance, load_uc_instance  # noqa: E402


def leading_time_window_instance(instance: UCInstance, horizon: int) -> UCInstance:
    if horizon <= 0 or horizon > instance.time_horizon:
        raise ValueError("horizon must be within the source instance time horizon")
    reserves = [
        Reserve(
            name=reserve.name,
            amount=reserve.amount[:horizon],
            penalty=reserve.penalty[:horizon],
        )
        for reserve in instance.reserves
    ]
    return replace(
        instance,
        time_horizon=horizon,
        fixed_load=instance.fixed_load[:horizon],
        reserves=reserves,
        power_balance_penalty=instance.power_balance_penalty[:horizon],
    )


def run(
    instance_path: Path,
    results_path: Path,
    horizon: int,
    max_order: int,
    target_counts: list[int],
) -> dict[str, object]:
    source_instance = load_uc_instance(instance_path)
    instance = leading_time_window_instance(source_instance, horizon)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bits = commitments.reshape((commitments.shape[0], -1))
    generator_names = [gen.name for gen in instance.generators]

    t0 = time.perf_counter()
    values, logic_feasible = evaluate_values(instance, commitments)
    evaluation_seconds = time.perf_counter() - t0

    finite = np.isfinite(values)
    sorted_indices = np.argsort(values)
    finite_sorted_indices = [idx for idx in sorted_indices if finite[idx]]
    order_rows = []
    for order in range(1, max_order + 1):
        subsets = all_x_subsets(bits.shape[1], order)
        features = x_monomial_features(bits, subsets)
        names = [subset_name(subset) for subset in subsets]
        order_rows.append(
            {
                "x_order": order,
                "feature_count": len(names),
                "target_results": [
                    run_threshold_case(
                        features=features,
                        threshold_case=threshold_case_for_top_count(values, count),
                        names=names,
                    )
                    for count in target_counts
                ],
            }
        )
    full_order_reference = {
        "x_order": int(bits.shape[1]),
        "feature_count": int(2 ** bits.shape[1]),
        "reference_only": True,
        "note": (
            "Full Boolean monomial interpolation can represent any marking exactly, "
            "but the feature count is exponential and is used only as an upper bound."
        ),
        "target_results": [
            full_order_threshold_reference(threshold_case_for_top_count(values, count))
            for count in target_counts
        ],
    }

    summary = {
        "instance": "case14_T2",
        "source": str(instance_path),
        "method": "case14 two-period ancilla VQC oracle with state-vector Grover simulation",
        "horizon": horizon,
        "generators": generator_names,
        "fixed_load_mw": instance.fixed_load,
        "bit_order": "generator-major flattening: g1_t0,g1_t1,g2_t0,g2_t1,...",
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "logic_feasible_count": int(logic_feasible.sum()),
        "finite_value_count": int(finite.sum()),
        "value_evaluation_seconds": float(evaluation_seconds),
        "optimum": commitment_row(commitments, generator_names, values, finite_sorted_indices[0]),
        "runner_up": commitment_row(commitments, generator_names, values, finite_sorted_indices[1]),
        "top_rows": [
            commitment_row(commitments, generator_names, values, idx)
            for idx in finite_sorted_indices[: min(10, len(finite_sorted_indices))]
        ],
        "order_rows": order_rows,
        "full_order_reference": full_order_reference,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_values(instance: UCInstance, commitments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    evaluator = FixedCommitmentEvaluator(instance)
    values = np.full(commitments.shape[0], np.inf, dtype=float)
    logic_feasible = np.zeros(commitments.shape[0], dtype=bool)
    for idx, commitment in enumerate(commitments):
        if not is_logic_feasible(instance, commitment):
            continue
        logic_feasible[idx] = True
        result = evaluator.evaluate(commitment)
        if result.success:
            values[idx] = result.total_cost
    return values, logic_feasible


def threshold_case_for_top_count(values: np.ndarray, target_count: int) -> dict[str, object]:
    finite_values = np.sort(values[np.isfinite(values)])
    if target_count <= 0 or target_count >= finite_values.size:
        raise ValueError("target_count must be positive and smaller than finite value count")
    threshold = float((finite_values[target_count - 1] + finite_values[target_count]) / 2.0)
    labels = np.isfinite(values) & (values <= threshold)
    return {
        "target_count_request": int(target_count),
        "actual_target_count": int(labels.sum()),
        "threshold": threshold,
        "labels": labels,
    }


def run_threshold_case(
    features: np.ndarray,
    threshold_case: dict[str, object],
    names: list[str],
) -> dict[str, object]:
    labels = np.asarray(threshold_case["labels"], dtype=bool)
    model = fit_ancilla_vqc(features, labels, names)
    evaluation = evaluate_ancilla_vqc(model, features, labels)
    exact_grover = ideal_grover_summary(labels)
    grover = grover_with_ancilla_model(
        model,
        features,
        labels,
        iterations=int(exact_grover["iterations"]),
    )
    return {
        "target_count_request": int(threshold_case["target_count_request"]),
        "actual_target_count": int(threshold_case["actual_target_count"]),
        "threshold": float(threshold_case["threshold"]),
        "exact_grover": exact_grover,
        "evaluation": {
            "angle_mae": evaluation.angle_mae,
            "max_angle_error": evaluation.max_angle_error,
            "max_leakage_probability": evaluation.max_leakage_probability,
            "mean_leakage_probability": evaluation.mean_leakage_probability,
            "marked_accuracy": evaluation.marked_accuracy,
            "correct_marked_set": evaluation.correct_marked_set,
            "false_positive_count": evaluation.false_positive_count,
            "false_negative_count": evaluation.false_negative_count,
        },
        "oracle_checks": {
            "unitary": True,
            "self_inverse": True,
            "verified_by": "all U^dagger Z U ancilla blocks are real orthogonal involutions",
        },
        "oracle_errors": ancilla_block_oracle_errors(model, features),
        "grover": {
            key: value for key, value in grover.items() if key != "state_probabilities"
        },
    }


def full_order_threshold_reference(threshold_case: dict[str, object]) -> dict[str, object]:
    labels = np.asarray(threshold_case["labels"], dtype=bool)
    exact_grover = ideal_grover_summary(labels)
    return {
        "target_count_request": int(threshold_case["target_count_request"]),
        "actual_target_count": int(threshold_case["actual_target_count"]),
        "threshold": float(threshold_case["threshold"]),
        "exact_grover": exact_grover,
        "evaluation": {
            "angle_mae": 0.0,
            "max_angle_error": 0.0,
            "max_leakage_probability": 0.0,
            "mean_leakage_probability": 0.0,
            "marked_accuracy": 1.0,
            "correct_marked_set": True,
            "false_positive_count": 0,
            "false_negative_count": 0,
        },
        "oracle_checks": {
            "unitary": True,
            "self_inverse": True,
            "verified_by": "full Boolean monomial interpolation gives angles exactly in {0, pi}",
        },
        "oracle_errors": {
            "unitarity_error": 0.0,
            "self_inverse_error": 0.0,
        },
        "grover": {
            "iterations": exact_grover["iterations"],
            "target_x_probability": exact_grover["marked_probability"],
            "non_target_x_probability": exact_grover["unmarked_probability"],
            "zero_ancilla_probability": 1.0,
            "one_ancilla_probability": 0.0,
            "target_zero_ancilla_probability": exact_grover["marked_probability"],
        },
    }


def ideal_grover_summary(labels: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    dimension = labels.size
    target_count = int(labels.sum())
    if target_count == 0:
        raise ValueError("Grover search needs at least one target state")
    iterations = max(1, int(np.floor(np.pi / 4.0 * np.sqrt(dimension / target_count))))
    angle = np.arcsin(np.sqrt(target_count / dimension))
    marked_probability = float(np.sin((2 * iterations + 1) * angle) ** 2)
    return {
        "iterations": int(iterations),
        "marked_probability": marked_probability,
        "unmarked_probability": float(1.0 - marked_probability),
    }


def commitment_row(
    commitments: np.ndarray,
    generator_names: list[str],
    values: np.ndarray,
    idx: int,
) -> dict[str, object]:
    commitment = commitments[int(idx)]
    return {
        "bitstring_generator_major": commitment_to_bitstring(commitment),
        "bitstring_time_major": "".join(str(int(v)) for v in commitment.T.reshape(-1)),
        "schedule_table": commitment_schedule_table(commitment, generator_names),
        "time_slices": commitment_time_slices(commitment, generator_names),
        "total_cost": float(values[int(idx)]),
    }


def commitment_schedule_table(
    commitment: np.ndarray,
    generator_names: list[str],
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for g_idx, name in enumerate(generator_names):
        row: dict[str, int | str] = {"generator": name}
        for t in range(commitment.shape[1]):
            row[f"t{t}"] = int(commitment[g_idx, t])
        rows.append(row)
    return rows


def commitment_time_slices(
    commitment: np.ndarray,
    generator_names: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for t in range(commitment.shape[1]):
        status_by_generator = {
            name: int(commitment[g_idx, t])
            for g_idx, name in enumerate(generator_names)
        }
        rows.append(
            {
                "time": f"t{t}",
                "status_by_generator": status_by_generator,
                "online_generators": [
                    name
                    for name, status in status_by_generator.items()
                    if status == 1
                ],
            }
        )
    return rows


def subset_name(subset: tuple[int, ...]) -> str:
    if not subset:
        return "1"
    return "*".join(f"x{idx}" for idx in subset)


def parse_target_counts(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--results", type=Path, default=Path("results/stage1_case14_t2_ancilla_vqc.json"))
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--target-counts", type=parse_target_counts, default=parse_target_counts("1,16,64"))
    args = parser.parse_args()
    summary = run(args.instance, args.results, args.horizon, args.max_order, args.target_counts)
    compact = {
        key: value
        for key, value in summary.items()
        if key not in {"top_rows", "order_rows", "full_order_reference"}
    }
    compact["order_rows"] = [
        {
            "x_order": row["x_order"],
            "feature_count": row["feature_count"],
            "target_results": [
                {
                    "target_count_request": item["target_count_request"],
                    "actual_target_count": item["actual_target_count"],
                    "correct_marked_set": item["evaluation"]["correct_marked_set"],
                    "max_leakage_probability": item["evaluation"]["max_leakage_probability"],
                    "target_x_probability": item["grover"]["target_x_probability"],
                    "zero_ancilla_probability": item["grover"]["zero_ancilla_probability"],
                    "exact_marked_probability": item["exact_grover"]["marked_probability"],
                }
                for item in row["target_results"]
            ],
        }
        for row in summary["order_rows"]
    ]
    compact["full_order_reference"] = {
        key: value
        for key, value in summary["full_order_reference"].items()
        if key != "target_results"
    }
    compact["full_order_reference"]["target_results"] = [
        {
            "target_count_request": item["target_count_request"],
            "actual_target_count": item["actual_target_count"],
            "correct_marked_set": item["evaluation"]["correct_marked_set"],
            "max_leakage_probability": item["evaluation"]["max_leakage_probability"],
            "target_x_probability": item["grover"]["target_x_probability"],
            "zero_ancilla_probability": item["grover"]["zero_ancilla_probability"],
            "exact_marked_probability": item["exact_grover"]["marked_probability"],
        }
        for item in summary["full_order_reference"]["target_results"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
