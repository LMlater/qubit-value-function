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
from experiments.stage1_case14_t2_separated_oracle import (  # noqa: E402
    evaluate_separated_oracle,
    evaluation_to_dict,
)
from qubit_value_function.ancilla_vqc import (  # noqa: E402
    controlled_ancilla_block_oracle_errors,
    evaluate_ancilla_vqc,
    fit_ancilla_vqc,
    grover_with_controlled_ancilla_model,
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
                    run_explicit_threshold_case(
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
            "Full Boolean interpolation can make the explicit two-ancilla oracle exact, "
            "but the feature count is exponential."
        ),
        "target_results": [
            full_order_threshold_reference(threshold_case_for_top_count(values, count))
            for count in target_counts
        ],
    }

    summary = {
        "instance": "case14_T2",
        "source": str(instance_path),
        "method": "explicit two-ancilla oracle: A_f -> U_theta -> CCZ(f,a) -> U_theta^dagger -> A_f^dagger",
        "horizon": horizon,
        "generators": generator_names,
        "fixed_load_mw": instance.fixed_load,
        "registers": {
            "x": "commitment register",
            "f": "feasibility ancilla, computed by exact Boolean physical-constraint oracle",
            "a": "value ancilla, rotated by the VQC value-function threshold module",
        },
        "bit_order": "generator-major flattening: g1_t0,g1_t1,g2_t0,g2_t1,...",
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "logic_feasible_count": int(logic_feasible.sum()),
        "finite_value_count": int(finite.sum()),
        "value_domain_count": int(value_domain.sum()),
        "value_evaluation_seconds": float(evaluation_seconds),
        "oracle_sequence": [
            "A_f |x>|0>_f = |x>|f(x)>_f",
            "U_theta rotates |a> according to theta(x,u,tau)",
            "CCZ(f,a) flips phase only when f=1 and a=1",
            "U_theta^dagger uncomputes the value ancilla approximately",
            "A_f^dagger uncomputes the feasibility ancilla exactly",
        ],
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


def run_explicit_threshold_case(
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
    explicit_grover = grover_with_explicit_two_ancilla_model(
        model,
        features,
        value_domain,
        labels,
        iterations=int(exact_grover["iterations"]),
    )
    equivalent_grover = grover_with_controlled_ancilla_model(
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
                "A_f and A_f^dagger are the same reversible XOR map; "
                "U_theta and U_theta^dagger are inverse rotations; CCZ is self-inverse"
            ),
        },
        "oracle_errors": controlled_ancilla_block_oracle_errors(model, features, value_domain),
        "explicit_grover": {
            key: value for key, value in explicit_grover.items() if key != "state_probabilities"
        },
        "equivalent_controlled_grover": {
            key: value for key, value in equivalent_grover.items() if key != "state_probabilities"
        },
        "explicit_vs_equivalent_abs_diff": grover_abs_differences(
            explicit_grover,
            equivalent_grover,
        ),
    }


def grover_abs_differences(
    explicit_grover: dict[str, object],
    equivalent_grover: dict[str, object],
) -> dict[str, float]:
    key_pairs = {
        "target_x_probability": ("target_x_probability", "target_x_probability"),
        "non_target_x_probability": ("non_target_x_probability", "non_target_x_probability"),
        "feasible_x_probability": ("feasible_x_probability", "feasible_x_probability"),
        "infeasible_x_probability": ("infeasible_x_probability", "infeasible_x_probability"),
        "one_value_ancilla_probability": ("one_value_ancilla_probability", "one_ancilla_probability"),
        "target_clean_probability": ("target_clean_probability", "target_zero_ancilla_probability"),
    }
    return {
        name: float(abs(float(explicit_grover[left]) - float(equivalent_grover[right])))
        for name, (left, right) in key_pairs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_explicit_two_ancilla_oracle.json"),
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
                    "max_leakage_probability": item["combined_oracle_evaluation"][
                        "max_leakage_probability"
                    ],
                    "target_x_probability": item["explicit_grover"]["target_x_probability"],
                    "one_feasibility_probability": item["explicit_grover"][
                        "one_feasibility_probability"
                    ],
                    "one_value_ancilla_probability": item["explicit_grover"][
                        "one_value_ancilla_probability"
                    ],
                    "clean_ancilla_probability": item["explicit_grover"][
                        "clean_ancilla_probability"
                    ],
                    "max_explicit_equivalent_diff": max(
                        item["explicit_vs_equivalent_abs_diff"].values()
                    ),
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
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
