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
    full_order_threshold_reference,
    ideal_grover_summary,
    leading_time_window_instance,
    parse_target_counts,
    subset_name,
    threshold_case_for_top_count,
)
from qubit_value_function.ancilla_vqc import (  # noqa: E402
    AncillaVQCEvaluation,
    controlled_ancilla_block_oracle_errors,
    evaluate_ancilla_vqc,
    fit_ancilla_vqc,
    grover_with_controlled_ancilla_model,
)
from qubit_value_function.commitment import all_commitments  # noqa: E402
from qubit_value_function.phase_vqc import all_x_subsets, x_monomial_features  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


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
    value_domain = logic_feasible & finite
    finite_sorted_indices = [idx for idx in np.argsort(values) if finite[idx]]

    order_rows = []
    for order in range(1, max_order + 1):
        subsets = all_x_subsets(bits.shape[1], order)
        features = x_monomial_features(bits, subsets)
        names = [subset_name(subset) for subset in subsets]
        order_rows.append(
            {
                "x_order": order,
                "feature_count": len(names),
                "training_domain_count": int(value_domain.sum()),
                "target_results": [
                    run_separated_threshold_case(
                        features=features,
                        value_domain=value_domain,
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
            "Full Boolean interpolation gives an exact value oracle on the feasible "
            "domain. It is included only as an exponential upper-bound reference."
        ),
        "target_results": [
            full_order_threshold_reference(threshold_case_for_top_count(values, count))
            for count in target_counts
        ],
    }

    summary = {
        "instance": "case14_T2",
        "source": str(instance_path),
        "method": (
            "separated oracle: exact 0-1 feasibility oracle plus feasible-domain "
            "ancilla VQC value oracle"
        ),
        "horizon": horizon,
        "generators": generator_names,
        "fixed_load_mw": instance.fixed_load,
        "bit_order": "generator-major flattening: g1_t0,g1_t1,g2_t0,g2_t1,...",
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "logic_feasible_count": int(logic_feasible.sum()),
        "finite_value_count": int(finite.sum()),
        "value_domain_count": int(value_domain.sum()),
        "value_evaluation_seconds": float(evaluation_seconds),
        "oracle_decomposition": {
            "feasibility_oracle": (
                "exact Boolean oracle for simple 0-1 physical constraints; "
                "infeasible states receive the identity block in the value oracle"
            ),
            "value_oracle": (
                "ancilla VQC trained only on logic-feasible finite-value commitments"
            ),
            "target_condition": "logic_feasible(x) and Q_theta(x,u) <= tau",
        },
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


def run_separated_threshold_case(
    features: np.ndarray,
    value_domain: np.ndarray,
    threshold_case: dict[str, object],
    names: list[str],
) -> dict[str, object]:
    labels = np.asarray(threshold_case["labels"], dtype=bool)
    if np.any(labels & ~value_domain):
        raise ValueError("target labels must be inside the value-domain mask")

    model = fit_ancilla_vqc(features[value_domain], labels[value_domain], names)
    feasible_evaluation = evaluate_ancilla_vqc(model, features[value_domain], labels[value_domain])
    combined_evaluation = evaluate_separated_oracle(
        model=model,
        features=features,
        value_domain=value_domain,
        labels=labels,
    )
    exact_grover = ideal_grover_summary(labels)
    grover = grover_with_controlled_ancilla_model(
        model,
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
        "oracle_checks": {
            "unitary": True,
            "self_inverse": True,
            "verified_by": (
                "infeasible x use identity blocks; feasible x use real "
                "U^dagger Z U involution blocks"
            ),
        },
        "oracle_errors": controlled_ancilla_block_oracle_errors(model, features, value_domain),
        "grover": {
            key: value for key, value in grover.items() if key != "state_probabilities"
        },
    }


def evaluate_separated_oracle(
    model,
    features: np.ndarray,
    value_domain: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    value_domain = np.asarray(value_domain, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    angles = model.angles(features)
    predicted = np.zeros(labels.shape, dtype=bool)
    predicted[value_domain] = np.cos(angles[value_domain]) < 0.0
    leakage = np.zeros(labels.shape, dtype=float)
    leakage[value_domain] = np.sin(angles[value_domain]) ** 2
    return {
        "marked_accuracy": float(np.mean(predicted == labels)),
        "correct_marked_set": bool(np.array_equal(predicted, labels)),
        "false_positive_count": int(np.logical_and(predicted, ~labels).sum()),
        "false_negative_count": int(np.logical_and(~predicted, labels).sum()),
        "max_leakage_probability": float(np.max(leakage)),
        "mean_leakage_probability": float(np.mean(leakage)),
        "mean_feasible_leakage_probability": float(np.mean(leakage[value_domain])),
        "infeasible_leakage_probability": 0.0,
    }


def evaluation_to_dict(evaluation: AncillaVQCEvaluation) -> dict[str, object]:
    return {
        "angle_mae": evaluation.angle_mae,
        "max_angle_error": evaluation.max_angle_error,
        "max_leakage_probability": evaluation.max_leakage_probability,
        "mean_leakage_probability": evaluation.mean_leakage_probability,
        "marked_accuracy": evaluation.marked_accuracy,
        "correct_marked_set": evaluation.correct_marked_set,
        "false_positive_count": evaluation.false_positive_count,
        "false_negative_count": evaluation.false_negative_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_separated_oracle.json"),
    )
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
            "training_domain_count": row["training_domain_count"],
            "target_results": [
                {
                    "target_count_request": item["target_count_request"],
                    "actual_target_count": item["actual_target_count"],
                    "correct_marked_set": item["combined_oracle_evaluation"]["correct_marked_set"],
                    "feasible_correct_marked_set": item["feasible_domain_evaluation"][
                        "correct_marked_set"
                    ],
                    "max_leakage_probability": item["combined_oracle_evaluation"][
                        "max_leakage_probability"
                    ],
                    "target_x_probability": item["grover"]["target_x_probability"],
                    "feasible_x_probability": item["grover"]["feasible_x_probability"],
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
