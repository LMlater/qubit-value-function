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
from experiments.stage1_case14_t2_separated_oracle import (  # noqa: E402
    evaluate_separated_oracle,
    evaluation_to_dict,
)
from qubit_value_function.ancilla_vqc import (  # noqa: E402
    ThresholdConditionedAncillaVQCEvaluation,
    ThresholdConditionedAncillaVQCModel,
    evaluate_ancilla_vqc,
    evaluate_threshold_conditioned_ancilla_vqc,
    fit_threshold_conditioned_ancilla_vqc,
    grover_with_explicit_two_ancilla_model,
)
from qubit_value_function.commitment import all_commitments  # noqa: E402
from qubit_value_function.phase_vqc import all_x_subsets, x_monomial_features  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def run(
    instance_path: Path,
    results_path: Path,
    horizon: int,
    max_order: int,
    train_target_counts: list[int],
    holdout_target_counts: list[int],
    tau_degree: int | None,
    boundary_weight: float,
    boundary_bandwidth: float,
    tau_basis: str,
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
    finite_sorted_indices = [idx for idx in np.argsort(values) if finite[idx]]

    train_cases = [threshold_case_for_top_count(values, count) for count in train_target_counts]
    holdout_cases = [threshold_case_for_top_count(values, count) for count in holdout_target_counts]
    train_thresholds = np.asarray([case["threshold"] for case in train_cases], dtype=float)
    if tau_basis == "piecewise_linear":
        resolved_tau_degree = max(0, train_thresholds.size - 1)
    else:
        resolved_tau_degree = max(0, train_thresholds.size - 1) if tau_degree is None else tau_degree

    order_rows = []
    for order in range(1, max_order + 1):
        subsets = all_x_subsets(bits.shape[1], order)
        features = x_monomial_features(bits, subsets)
        feasible_features = features[value_domain]
        names = [subset_name(subset) for subset in subsets]
        labels_matrix = np.column_stack(
            [np.asarray(case["labels"], dtype=bool)[value_domain] for case in train_cases]
        )
        sample_weights = threshold_boundary_weights(
            values[value_domain],
            train_cases,
            boundary_weight=boundary_weight,
            boundary_bandwidth=boundary_bandwidth,
        )
        model = fit_threshold_conditioned_ancilla_vqc(
            feasible_features,
            labels_matrix,
            train_thresholds,
            names,
            tau_degree=resolved_tau_degree,
            sample_weights=sample_weights,
            tau_basis=tau_basis,
        )
        training_family_evaluation = evaluate_threshold_conditioned_ancilla_vqc(
            model,
            feasible_features,
            labels_matrix,
            train_thresholds,
        )
        order_rows.append(
            {
                "x_order": order,
                "x_feature_count": len(names),
                "tau_degree": int(resolved_tau_degree),
                "parameter_count": int(len(names) * (resolved_tau_degree + 1)),
                "training_domain_count": int(value_domain.sum()),
                "value_function_coherence": monotonicity_diagnostics(
                    model,
                    feasible_features,
                    value_domain,
                    [*train_cases, *holdout_cases],
                ),
                "training_family_evaluation": threshold_family_evaluation_to_dict(
                    training_family_evaluation
                ),
                "train_results": [
                    run_threshold_case(
                        model=model,
                        features=features,
                        feasible_features=feasible_features,
                        value_domain=value_domain,
                        threshold_case=case,
                    )
                    for case in train_cases
                ],
                "holdout_results": [
                    run_threshold_case(
                        model=model,
                        features=features,
                        feasible_features=feasible_features,
                        value_domain=value_domain,
                        threshold_case=case,
                    )
                    for case in holdout_cases
                ],
            }
        )

    summary = {
        "instance": "case14_T2",
        "source": str(instance_path),
        "method": (
            "threshold-conditioned ancilla VQC: one U_theta(x,tau) fitted to "
            "multiple value-function sublevel sets"
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
        "train_target_counts": [int(count) for count in train_target_counts],
        "holdout_target_counts": [int(count) for count in holdout_target_counts],
        "train_thresholds": [float(tau) for tau in train_thresholds],
        "tau_degree": int(resolved_tau_degree),
        "tau_basis": tau_basis,
        "training_weights": {
            "method": "rank-distance boundary weighting" if boundary_weight > 0.0 else "uniform",
            "boundary_weight": float(boundary_weight),
            "boundary_bandwidth": float(boundary_bandwidth),
        },
        "oracle_decomposition": {
            "feasibility_oracle": "exact Boolean feasibility register",
            "value_oracle": "x- and tau-conditioned R_y angle model on the value ancilla",
            "target_condition": "logic_feasible(x) and U_theta(x,tau) rotates value ancilla to |1>",
        },
        "optimum": commitment_row(commitments, generator_names, values, finite_sorted_indices[0]),
        "runner_up": commitment_row(commitments, generator_names, values, finite_sorted_indices[1]),
        "order_rows": order_rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_threshold_case(
    *,
    model: ThresholdConditionedAncillaVQCModel,
    features: np.ndarray,
    feasible_features: np.ndarray,
    value_domain: np.ndarray,
    threshold_case: dict[str, object],
) -> dict[str, object]:
    labels = np.asarray(threshold_case["labels"], dtype=bool)
    feasible_labels = labels[value_domain]
    fixed_model = model.fixed_tau_model(float(threshold_case["threshold"]))
    feasible_evaluation = evaluate_ancilla_vqc(
        fixed_model,
        feasible_features,
        feasible_labels,
    )
    combined_evaluation = evaluate_separated_oracle(
        model=fixed_model,
        features=features,
        value_domain=value_domain,
        labels=labels,
    )
    exact_grover = ideal_grover_summary(labels)
    grover = grover_with_explicit_two_ancilla_model(
        fixed_model,
        features,
        value_domain,
        labels,
        iterations=int(exact_grover["iterations"]),
    )
    return {
        "target_count_request": int(threshold_case["target_count_request"]),
        "actual_target_count": int(threshold_case["actual_target_count"]),
        "threshold": float(threshold_case["threshold"]),
        "exact_grover": exact_grover,
        "feasible_domain_evaluation": evaluation_to_dict(feasible_evaluation),
        "combined_oracle_evaluation": combined_evaluation,
        "explicit_grover": {
            key: value for key, value in grover.items() if key != "state_probabilities"
        },
    }


def threshold_family_evaluation_to_dict(
    evaluation: ThresholdConditionedAncillaVQCEvaluation,
) -> dict[str, object]:
    return {
        "marked_accuracy": evaluation.marked_accuracy,
        "correct_marked_sets": evaluation.correct_marked_sets,
        "max_leakage_probability": evaluation.max_leakage_probability,
        "mean_leakage_probability": evaluation.mean_leakage_probability,
        "threshold_summaries": evaluation.threshold_summaries,
    }


def threshold_boundary_weights(
    feasible_values: np.ndarray,
    threshold_cases: list[dict[str, object]],
    *,
    boundary_weight: float,
    boundary_bandwidth: float,
) -> np.ndarray | None:
    if boundary_weight <= 0.0:
        return None
    if boundary_bandwidth <= 0.0:
        raise ValueError("boundary_bandwidth must be positive")
    feasible_values = np.asarray(feasible_values, dtype=float)
    order = np.argsort(feasible_values)
    ranks = np.empty(feasible_values.shape[0], dtype=float)
    ranks[order] = np.arange(feasible_values.shape[0], dtype=float)
    columns = []
    for case in threshold_cases:
        boundary_rank = float(case["actual_target_count"]) - 0.5
        distance = np.abs(ranks - boundary_rank)
        weights = 1.0 + float(boundary_weight) * np.exp(
            -((distance / float(boundary_bandwidth)) ** 2)
        )
        columns.append(weights)
    return np.column_stack(columns)


def monotonicity_diagnostics(
    model: ThresholdConditionedAncillaVQCModel,
    feasible_features: np.ndarray,
    value_domain: np.ndarray,
    threshold_cases: list[dict[str, object]],
) -> dict[str, object]:
    value_domain = np.asarray(value_domain, dtype=bool)
    ordered_cases = sorted(threshold_cases, key=lambda item: float(item["threshold"]))
    predicted = []
    exact = []
    for case in ordered_cases:
        tau = float(case["threshold"])
        fixed_model = model.fixed_tau_model(tau)
        predicted.append(np.cos(fixed_model.angles(feasible_features)) < 0.0)
        labels = np.asarray(case["labels"], dtype=bool)
        exact.append(labels[value_domain])

    predicted_matrix = np.column_stack(predicted)
    exact_matrix = np.column_stack(exact)
    predicted_adjacent_violations = predicted_matrix[:, :-1] & ~predicted_matrix[:, 1:]
    exact_adjacent_violations = exact_matrix[:, :-1] & ~exact_matrix[:, 1:]
    return {
        "thresholds": [float(case["threshold"]) for case in ordered_cases],
        "target_count_requests": [
            int(case["target_count_request"]) for case in ordered_cases
        ],
        "predicted_adjacent_violation_count": int(predicted_adjacent_violations.sum()),
        "predicted_violating_state_count": int(
            np.any(predicted_adjacent_violations, axis=1).sum()
        ),
        "exact_adjacent_violation_count": int(exact_adjacent_violations.sum()),
    }


def optional_tau_degree(raw: str) -> int | None:
    value = int(raw)
    if value < 0:
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_threshold_conditioned_ancilla_vqc.json"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument(
        "--train-target-counts",
        type=parse_target_counts,
        default=parse_target_counts("1,16,64"),
    )
    parser.add_argument(
        "--holdout-target-counts",
        type=parse_target_counts,
        default=parse_target_counts("4,32"),
    )
    parser.add_argument("--tau-degree", type=optional_tau_degree, default=None)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--boundary-bandwidth", type=float, default=8.0)
    parser.add_argument(
        "--tau-basis",
        choices=("polynomial", "piecewise_linear"),
        default="polynomial",
    )
    args = parser.parse_args()
    summary = run(
        args.instance,
        args.results,
        args.horizon,
        args.max_order,
        args.train_target_counts,
        args.holdout_target_counts,
        args.tau_degree,
        args.boundary_weight,
        args.boundary_bandwidth,
        args.tau_basis,
    )
    compact = {
        key: value
        for key, value in summary.items()
        if key not in {"order_rows"}
    }
    compact["order_rows"] = [
        {
            "x_order": row["x_order"],
            "x_feature_count": row["x_feature_count"],
            "parameter_count": row["parameter_count"],
            "training_family_evaluation": {
                key: row["training_family_evaluation"][key]
                for key in (
                    "marked_accuracy",
                    "correct_marked_sets",
                    "max_leakage_probability",
                    "mean_leakage_probability",
                )
            },
            "value_function_coherence": row["value_function_coherence"],
            "train_results": [compact_threshold_result(item) for item in row["train_results"]],
            "holdout_results": [compact_threshold_result(item) for item in row["holdout_results"]],
        }
        for row in summary["order_rows"]
    ]
    print(json.dumps(compact, indent=2))


def compact_threshold_result(item: dict[str, object]) -> dict[str, object]:
    combined = item["combined_oracle_evaluation"]
    grover = item["explicit_grover"]
    return {
        "target_count_request": item["target_count_request"],
        "actual_target_count": item["actual_target_count"],
        "correct_marked_set": combined["correct_marked_set"],
        "false_positive_count": combined["false_positive_count"],
        "false_negative_count": combined["false_negative_count"],
        "max_leakage_probability": combined["max_leakage_probability"],
        "target_x_probability": grover["target_x_probability"],
        "dirty_ancilla_probability": grover["dirty_ancilla_probability"],
        "exact_marked_probability": item["exact_grover"]["marked_probability"],
    }


if __name__ == "__main__":
    main()
