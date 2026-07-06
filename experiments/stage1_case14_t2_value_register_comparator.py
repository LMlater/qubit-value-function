from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_t2_ancilla_vqc import (  # noqa: E402
    commitment_row,
    evaluate_values,
    ideal_grover_summary,
    leading_time_window_instance,
    parse_target_counts,
    subset_name,
    threshold_case_for_top_count,
)
from qubit_value_function.commitment import all_commitments  # noqa: E402
from qubit_value_function.phase_vqc import all_x_subsets, x_monomial_features  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402
from qubit_value_function.value_surrogate import (  # noqa: E402
    ScalarValueFunctionEvaluation,
    evaluate_scalar_value_function,
    fit_scalar_value_function,
    quantize_values,
)


def run(
    instance_path: Path,
    results_path: Path,
    horizon: int,
    max_order: int,
    target_counts: list[int],
    register_bits: list[int],
    ridge: float,
    tie_tolerance: float,
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
    value_domain = logic_feasible & finite
    feasible_values = values[value_domain]
    finite_sorted_indices = [idx for idx in np.argsort(values) if finite[idx]]
    threshold_cases = [
        tie_tolerant_threshold_case_for_top_count(values, count, tie_tolerance)
        for count in target_counts
    ]

    order_rows = []
    for order in range(1, max_order + 1):
        subsets = all_x_subsets(bits.shape[1], order)
        features = x_monomial_features(bits, subsets)
        feasible_features = features[value_domain]
        names = [subset_name(subset) for subset in subsets]
        model = fit_scalar_value_function(
            feasible_features,
            feasible_values,
            names,
            ridge=ridge,
        )
        predictions = model.predict(features)
        feasible_predictions = predictions[value_domain]
        value_evaluation = evaluate_scalar_value_function(
            model,
            feasible_features,
            feasible_values,
        )
        order_rows.append(
            {
                "x_order": order,
                "feature_count": len(names),
                "training_domain_count": int(value_domain.sum()),
                "ridge": float(ridge),
                "value_regression": value_evaluation_to_dict(value_evaluation),
                "floating_comparator_results": [
                    run_floating_threshold_case(
                        predictions=predictions,
                        value_domain=value_domain,
                        threshold_case=case,
                    )
                    for case in threshold_cases
                ],
                "fixed_point_register_results": [
                    run_quantized_register_family(
                        predictions=predictions,
                        feasible_predictions=feasible_predictions,
                        value_domain=value_domain,
                        threshold_cases=threshold_cases,
                        bits=count,
                    )
                    for count in register_bits
                ],
            }
        )

    summary = {
        "instance": "case14_T2",
        "source": str(instance_path),
        "method": (
            "scalar value-function surrogate with fixed-point value register "
            "and reversible threshold-comparator simulation"
        ),
        "horizon": horizon,
        "generators": generator_names,
        "fixed_load_mw": instance.fixed_load,
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "logic_feasible_count": int(logic_feasible.sum()),
        "finite_value_count": int(finite.sum()),
        "value_domain_count": int(value_domain.sum()),
        "value_evaluation_seconds": float(evaluation_seconds),
        "target_counts": [int(count) for count in target_counts],
        "register_bits": [int(count) for count in register_bits],
        "tie_tolerance": float(tie_tolerance),
        "oracle_decomposition": {
            "feasibility_oracle": "exact Boolean feasibility register",
            "value_register": (
                "compute a fixed-point approximation of V_theta(x,u) into an "
                "integer value register"
            ),
            "threshold_comparator": "mark if feasible(x) and value_register <= tau_register",
            "uncompute": "reverse value-register and feasibility computations after phase marking",
        },
        "optimum": commitment_row(commitments, generator_names, values, finite_sorted_indices[0]),
        "runner_up": commitment_row(commitments, generator_names, values, finite_sorted_indices[1]),
        "order_rows": order_rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_floating_threshold_case(
    *,
    predictions: np.ndarray,
    value_domain: np.ndarray,
    threshold_case: dict[str, object],
) -> dict[str, object]:
    labels = np.asarray(threshold_case["labels"], dtype=bool)
    predicted_marked = np.zeros(labels.shape, dtype=bool)
    predicted_marked[value_domain] = predictions[value_domain] <= float(threshold_case["threshold"])
    exact_grover = ideal_grover_summary(labels)
    grover = grover_with_predicted_phase_oracle(
        predicted_marked,
        labels,
        iterations=int(exact_grover["iterations"]),
    )
    return {
        "target_count_request": int(threshold_case["target_count_request"]),
        "actual_target_count": int(threshold_case["actual_target_count"]),
        "threshold": float(threshold_case["threshold"]),
        "exact_grover": exact_grover,
        "comparator_evaluation": comparator_evaluation(predicted_marked, labels),
        "grover": grover,
    }


def tie_tolerant_threshold_case_for_top_count(
    values: np.ndarray,
    target_count: int,
    tolerance: float,
) -> dict[str, object]:
    if tolerance < 0.0:
        raise ValueError("tie tolerance must be nonnegative")
    case = threshold_case_for_top_count(values, target_count)
    if tolerance == 0.0:
        return case
    threshold = float(case["threshold"])
    labels = np.isfinite(values) & (values <= threshold + float(tolerance))
    return {
        **case,
        "labels": labels,
        "actual_target_count": int(labels.sum()),
        "tie_tolerance": float(tolerance),
    }


def run_quantized_register_family(
    *,
    predictions: np.ndarray,
    feasible_predictions: np.ndarray,
    value_domain: np.ndarray,
    threshold_cases: list[dict[str, object]],
    bits: int,
) -> dict[str, object]:
    value_min = float(np.min(feasible_predictions))
    value_max = float(np.max(feasible_predictions))
    prediction_register = quantize_values(
        predictions,
        value_min=value_min,
        value_max=value_max,
        bits=bits,
    )
    return {
        "register_bits": int(bits),
        "value_min": value_min,
        "value_max": value_max,
        "results": [
            run_quantized_threshold_case(
                prediction_register=prediction_register,
                value_min=value_min,
                value_max=value_max,
                value_domain=value_domain,
                threshold_case=case,
                bits=bits,
            )
            for case in threshold_cases
        ],
    }


def run_quantized_threshold_case(
    *,
    prediction_register: np.ndarray,
    value_min: float,
    value_max: float,
    value_domain: np.ndarray,
    threshold_case: dict[str, object],
    bits: int,
) -> dict[str, object]:
    labels = np.asarray(threshold_case["labels"], dtype=bool)
    tau_register = int(
        quantize_values(
            np.array([float(threshold_case["threshold"])]),
            value_min=value_min,
            value_max=value_max,
            bits=bits,
        )[0]
    )
    predicted_marked = np.zeros(labels.shape, dtype=bool)
    predicted_marked[value_domain] = prediction_register[value_domain] <= tau_register
    exact_grover = ideal_grover_summary(labels)
    grover = grover_with_predicted_phase_oracle(
        predicted_marked,
        labels,
        iterations=int(exact_grover["iterations"]),
    )
    return {
        "target_count_request": int(threshold_case["target_count_request"]),
        "actual_target_count": int(threshold_case["actual_target_count"]),
        "threshold": float(threshold_case["threshold"]),
        "tau_register": tau_register,
        "comparator_evaluation": comparator_evaluation(predicted_marked, labels),
        "grover": grover,
    }


def comparator_evaluation(predicted_marked: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    predicted_marked = np.asarray(predicted_marked, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    return {
        "marked_accuracy": float(np.mean(predicted_marked == labels)),
        "correct_marked_set": bool(np.array_equal(predicted_marked, labels)),
        "predicted_count": int(predicted_marked.sum()),
        "false_positive_count": int(np.logical_and(predicted_marked, ~labels).sum()),
        "false_negative_count": int(np.logical_and(~predicted_marked, labels).sum()),
    }


def grover_with_predicted_phase_oracle(
    predicted_marked: np.ndarray,
    target_labels: np.ndarray,
    *,
    iterations: int,
) -> dict[str, object]:
    predicted_marked = np.asarray(predicted_marked, dtype=bool)
    target_labels = np.asarray(target_labels, dtype=bool)
    if predicted_marked.shape != target_labels.shape:
        raise ValueError("predicted_marked and target_labels must have matching shape")
    dimension = target_labels.size
    state = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
    for _ in range(iterations):
        state[predicted_marked] *= -1.0
        state = 2.0 * np.mean(state) - state
    probabilities = np.abs(state) ** 2
    return {
        "iterations": int(iterations),
        "target_x_probability": float(probabilities[target_labels].sum()),
        "predicted_marked_probability": float(probabilities[predicted_marked].sum()),
        "non_target_x_probability": float(probabilities[~target_labels].sum()),
        "phase_oracle_unitary": True,
        "phase_oracle_self_inverse": True,
    }


def value_evaluation_to_dict(evaluation: ScalarValueFunctionEvaluation) -> dict[str, object]:
    return {
        "mae": evaluation.mae,
        "rmse": evaluation.rmse,
        "max_abs_error": evaluation.max_abs_error,
        "rank_inversion_count": evaluation.rank_inversion_count,
    }


def parse_register_bits(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_value_register_comparator.json"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument(
        "--target-counts",
        type=parse_target_counts,
        default=parse_target_counts("1,4,8,16,32,48,64"),
    )
    parser.add_argument("--register-bits", type=parse_register_bits, default=parse_register_bits("8,10,12"))
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--tie-tolerance", type=float, default=0.0)
    args = parser.parse_args()
    summary = run(
        args.instance,
        args.results,
        args.horizon,
        args.max_order,
        args.target_counts,
        args.register_bits,
        args.ridge,
        args.tie_tolerance,
    )
    compact = {
        key: value
        for key, value in summary.items()
        if key not in {"order_rows"}
    }
    compact["order_rows"] = [
        {
            "x_order": row["x_order"],
            "feature_count": row["feature_count"],
            "value_regression": row["value_regression"],
            "floating_comparator_results": [
                compact_threshold_result(item)
                for item in row["floating_comparator_results"]
            ],
            "fixed_point_register_results": [
                {
                    "register_bits": family["register_bits"],
                    "results": [
                        compact_threshold_result(item)
                        for item in family["results"]
                    ],
                }
                for family in row["fixed_point_register_results"]
            ],
        }
        for row in summary["order_rows"]
    ]
    print(json.dumps(compact, indent=2))


def compact_threshold_result(item: dict[str, object]) -> dict[str, object]:
    evaluation = item["comparator_evaluation"]
    grover = item["grover"]
    return {
        "target_count_request": item["target_count_request"],
        "actual_target_count": item["actual_target_count"],
        "correct_marked_set": evaluation["correct_marked_set"],
        "predicted_count": evaluation["predicted_count"],
        "false_positive_count": evaluation["false_positive_count"],
        "false_negative_count": evaluation["false_negative_count"],
        "target_x_probability": grover["target_x_probability"],
        "predicted_marked_probability": grover["predicted_marked_probability"],
    }


if __name__ == "__main__":
    main()
