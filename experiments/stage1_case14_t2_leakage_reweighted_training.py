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
    evaluate_ancilla_vqc,
    fit_ancilla_vqc,
    fit_leakage_reweighted_ancilla_vqc,
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
    alphas: tuple[float, ...],
    iterations: int,
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
        feasible_features = features[value_domain]
        names = [subset_name(subset) for subset in subsets]
        order_rows.append(
            {
                "x_order": order,
                "feature_count": len(names),
                "training_domain_count": int(value_domain.sum()),
                "target_results": [
                    run_training_case(
                        features=features,
                        feasible_features=feasible_features,
                        value_domain=value_domain,
                        threshold_case=threshold_case_for_top_count(values, count),
                        names=names,
                        alphas=alphas,
                        iterations=iterations,
                    )
                    for count in target_counts
                ],
            }
        )

    summary = {
        "instance": "case14_T2",
        "source": str(instance_path),
        "method": "leakage-reweighted training for explicit two-ancilla value oracle",
        "horizon": horizon,
        "generators": generator_names,
        "fixed_load_mw": instance.fixed_load,
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "logic_feasible_count": int(logic_feasible.sum()),
        "finite_value_count": int(finite.sum()),
        "value_domain_count": int(value_domain.sum()),
        "value_evaluation_seconds": float(evaluation_seconds),
        "training": {
            "baseline": "ordinary least-squares fit to angles 0/pi",
            "reweighted": (
                "iteratively reweighted least squares, with larger weights on "
                "states having larger sin(theta)^2 leakage"
            ),
            "alphas": [float(alpha) for alpha in alphas],
            "iterations_per_alpha": int(iterations),
        },
        "optimum": commitment_row(commitments, generator_names, values, finite_sorted_indices[0]),
        "runner_up": commitment_row(commitments, generator_names, values, finite_sorted_indices[1]),
        "order_rows": order_rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_training_case(
    features: np.ndarray,
    feasible_features: np.ndarray,
    value_domain: np.ndarray,
    threshold_case: dict[str, object],
    names: list[str],
    alphas: tuple[float, ...],
    iterations: int,
) -> dict[str, object]:
    labels = np.asarray(threshold_case["labels"], dtype=bool)
    feasible_labels = labels[value_domain]
    if np.any(labels & ~value_domain):
        raise ValueError("target labels must be inside the value-domain mask")

    baseline_model = fit_ancilla_vqc(feasible_features, feasible_labels, names)
    training = fit_leakage_reweighted_ancilla_vqc(
        feasible_features,
        feasible_labels,
        names,
        alphas=alphas,
        iterations=iterations,
    )
    exact_grover = ideal_grover_summary(labels)
    baseline_grover = grover_with_explicit_two_ancilla_model(
        baseline_model,
        features,
        value_domain,
        labels,
        iterations=int(exact_grover["iterations"]),
    )
    reweighted_grover = grover_with_explicit_two_ancilla_model(
        training.model,
        features,
        value_domain,
        labels,
        iterations=int(exact_grover["iterations"]),
    )
    baseline_feasible_evaluation = evaluate_ancilla_vqc(
        baseline_model,
        feasible_features,
        feasible_labels,
    )
    baseline_combined = evaluate_separated_oracle(
        model=baseline_model,
        features=features,
        value_domain=value_domain,
        labels=labels,
    )
    reweighted_combined = evaluate_separated_oracle(
        model=training.model,
        features=features,
        value_domain=value_domain,
        labels=labels,
    )
    return {
        "target_count_request": int(threshold_case["target_count_request"]),
        "actual_target_count": int(threshold_case["actual_target_count"]),
        "threshold": float(threshold_case["threshold"]),
        "exact_grover": exact_grover,
        "baseline": {
            "feasible_domain_evaluation": evaluation_to_dict(baseline_feasible_evaluation),
            "combined_oracle_evaluation": baseline_combined,
            "explicit_grover": {
                key: value for key, value in baseline_grover.items() if key != "state_probabilities"
            },
        },
        "reweighted": {
            "selected_alpha": training.selected_alpha,
            "selected_iteration": training.selected_iteration,
            "feasible_domain_evaluation": evaluation_to_dict(training.final_evaluation),
            "combined_oracle_evaluation": reweighted_combined,
            "explicit_grover": {
                key: value for key, value in reweighted_grover.items() if key != "state_probabilities"
            },
            "history": training.history,
        },
        "improvement": {
            "max_leakage_reduction": float(
                baseline_combined["max_leakage_probability"]
                - reweighted_combined["max_leakage_probability"]
            ),
            "target_probability_change": float(
                reweighted_grover["target_x_probability"]
                - baseline_grover["target_x_probability"]
            ),
            "dirty_ancilla_probability_change": float(
                reweighted_grover["dirty_ancilla_probability"]
                - baseline_grover["dirty_ancilla_probability"]
            ),
        },
    }


def parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_leakage_reweighted_training.json"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--target-counts", type=parse_target_counts, default=parse_target_counts("1,16,64"))
    parser.add_argument("--alphas", type=parse_float_tuple, default=parse_float_tuple("5,20,100"))
    parser.add_argument("--iterations", type=int, default=6)
    args = parser.parse_args()
    summary = run(
        args.instance,
        args.results,
        args.horizon,
        args.max_order,
        args.target_counts,
        args.alphas,
        args.iterations,
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
            "target_results": [
                {
                    "target_count_request": item["target_count_request"],
                    "actual_target_count": item["actual_target_count"],
                    "baseline_max_leakage": item["baseline"]["combined_oracle_evaluation"][
                        "max_leakage_probability"
                    ],
                    "reweighted_max_leakage": item["reweighted"]["combined_oracle_evaluation"][
                        "max_leakage_probability"
                    ],
                    "baseline_target_probability": item["baseline"]["explicit_grover"][
                        "target_x_probability"
                    ],
                    "reweighted_target_probability": item["reweighted"]["explicit_grover"][
                        "target_x_probability"
                    ],
                    "selected_alpha": item["reweighted"]["selected_alpha"],
                    "selected_iteration": item["reweighted"]["selected_iteration"],
                    "target_probability_change": item["improvement"][
                        "target_probability_change"
                    ],
                }
                for item in row["target_results"]
            ],
        }
        for row in summary["order_rows"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
