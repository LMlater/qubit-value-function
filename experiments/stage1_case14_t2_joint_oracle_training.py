from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
from experiments.stage1_case14_t2_leakage_reweighted_training import parse_float_tuple  # noqa: E402
from experiments.stage1_case14_t2_separated_oracle import (  # noqa: E402
    evaluate_separated_oracle,
    evaluation_to_dict,
)
from qubit_value_function.ancilla_vqc import (  # noqa: E402
    AncillaVQCModel,
    evaluate_ancilla_vqc,
    fit_ancilla_vqc,
    grover_with_explicit_two_ancilla_model,
)
from qubit_value_function.commitment import all_commitments  # noqa: E402
from qubit_value_function.phase_vqc import all_x_subsets, x_monomial_features  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


@dataclass(frozen=True)
class CandidateModel:
    model: AncillaVQCModel
    method: str
    alpha: float | None
    iteration: int


def run(
    instance_path: Path,
    results_path: Path,
    horizon: int,
    max_order: int,
    target_counts: list[int],
    alphas: tuple[float, ...],
    iterations: int,
    weights: dict[str, float],
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
                "candidate_count_per_threshold": 1 + len(alphas) * iterations,
                "target_results": [
                    run_joint_case(
                        features=features,
                        feasible_features=feasible_features,
                        value_domain=value_domain,
                        threshold_case=threshold_case_for_top_count(values, count),
                        names=names,
                        alphas=alphas,
                        iterations=iterations,
                        weights=weights,
                    )
                    for count in target_counts
                ],
            }
        )

    summary = {
        "instance": "case14_T2",
        "source": str(instance_path),
        "method": "joint-score selection over ordinary and leakage-reweighted VQC candidates",
        "horizon": horizon,
        "generators": generator_names,
        "fixed_load_mw": instance.fixed_load,
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "logic_feasible_count": int(logic_feasible.sum()),
        "finite_value_count": int(finite.sum()),
        "value_domain_count": int(value_domain.sum()),
        "value_evaluation_seconds": float(evaluation_seconds),
        "candidate_generation": {
            "ordinary": "ordinary least-squares fit to angles 0/pi",
            "reweighted": "intermediate models from leakage-reweighted least squares",
            "alphas": [float(alpha) for alpha in alphas],
            "iterations_per_alpha": int(iterations),
        },
        "joint_score": {
            "formula": (
                "target_probability - w_max*max_leakage - w_mean*mean_leakage "
                "- w_mark*mark_error_count"
            ),
            "weights": weights,
        },
        "optimum": commitment_row(commitments, generator_names, values, finite_sorted_indices[0]),
        "runner_up": commitment_row(commitments, generator_names, values, finite_sorted_indices[1]),
        "order_rows": order_rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_joint_case(
    features: np.ndarray,
    feasible_features: np.ndarray,
    value_domain: np.ndarray,
    threshold_case: dict[str, object],
    names: list[str],
    alphas: tuple[float, ...],
    iterations: int,
    weights: dict[str, float],
) -> dict[str, object]:
    labels = np.asarray(threshold_case["labels"], dtype=bool)
    feasible_labels = labels[value_domain]
    if np.any(labels & ~value_domain):
        raise ValueError("target labels must be inside the value-domain mask")

    exact_grover = ideal_grover_summary(labels)
    candidates = generate_candidates(
        feasible_features,
        feasible_labels,
        names,
        alphas=alphas,
        iterations=iterations,
    )
    evaluated = [
        evaluate_candidate(
            candidate,
            features=features,
            feasible_features=feasible_features,
            value_domain=value_domain,
            labels=labels,
            feasible_labels=feasible_labels,
            grover_iterations=int(exact_grover["iterations"]),
            weights=weights,
        )
        for candidate in candidates
    ]
    baseline = evaluated[0]
    leakage_only = min(evaluated, key=leakage_selection_key)
    joint = max(evaluated, key=lambda item: item["score"])
    return {
        "target_count_request": int(threshold_case["target_count_request"]),
        "actual_target_count": int(threshold_case["actual_target_count"]),
        "threshold": float(threshold_case["threshold"]),
        "exact_grover": exact_grover,
        "baseline": compact_candidate(baseline),
        "leakage_only": compact_candidate(leakage_only),
        "joint_score": compact_candidate(joint),
        "improvement_vs_baseline": {
            "joint_target_probability_change": float(
                joint["explicit_grover"]["target_x_probability"]
                - baseline["explicit_grover"]["target_x_probability"]
            ),
            "joint_max_leakage_change": float(
                joint["combined_oracle_evaluation"]["max_leakage_probability"]
                - baseline["combined_oracle_evaluation"]["max_leakage_probability"]
            ),
            "leakage_only_target_probability_change": float(
                leakage_only["explicit_grover"]["target_x_probability"]
                - baseline["explicit_grover"]["target_x_probability"]
            ),
            "leakage_only_max_leakage_change": float(
                leakage_only["combined_oracle_evaluation"]["max_leakage_probability"]
                - baseline["combined_oracle_evaluation"]["max_leakage_probability"]
            ),
        },
        "top_score_candidates": [
            compact_candidate(candidate)
            for candidate in sorted(evaluated, key=lambda item: item["score"], reverse=True)[:5]
        ],
    }


def generate_candidates(
    features: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    *,
    alphas: tuple[float, ...],
    iterations: int,
) -> list[CandidateModel]:
    target_angles = np.pi * labels.astype(float)
    candidates = [
        CandidateModel(
            model=fit_ancilla_vqc(features, labels, names),
            method="ordinary_lstsq",
            alpha=None,
            iteration=0,
        )
    ]
    for alpha in alphas:
        weights = np.ones(features.shape[0], dtype=float)
        for iteration in range(1, iterations + 1):
            weighted_features = features * np.sqrt(weights)[:, None]
            weighted_targets = target_angles * np.sqrt(weights)
            coefficients, *_ = np.linalg.lstsq(weighted_features, weighted_targets, rcond=None)
            candidates.append(
                CandidateModel(
                    model=AncillaVQCModel(names=list(names), coefficients=np.asarray(coefficients)),
                    method="leakage_reweighted_lstsq",
                    alpha=float(alpha),
                    iteration=iteration,
                )
            )
            leakage = np.sin(features @ coefficients) ** 2
            max_leakage = float(np.max(leakage))
            if max_leakage <= 1e-15:
                weights = np.ones_like(weights)
            else:
                weights = 1.0 + float(alpha) * (leakage / max_leakage) ** 2
    return candidates


def evaluate_candidate(
    candidate: CandidateModel,
    *,
    features: np.ndarray,
    feasible_features: np.ndarray,
    value_domain: np.ndarray,
    labels: np.ndarray,
    feasible_labels: np.ndarray,
    grover_iterations: int,
    weights: dict[str, float],
) -> dict[str, object]:
    feasible_evaluation = evaluate_ancilla_vqc(
        candidate.model,
        feasible_features,
        feasible_labels,
    )
    combined_evaluation = evaluate_separated_oracle(
        model=candidate.model,
        features=features,
        value_domain=value_domain,
        labels=labels,
    )
    grover = grover_with_explicit_two_ancilla_model(
        candidate.model,
        features,
        value_domain,
        labels,
        iterations=grover_iterations,
    )
    score = joint_oracle_score(
        target_probability=float(grover["target_x_probability"]),
        max_leakage=float(combined_evaluation["max_leakage_probability"]),
        mean_leakage=float(combined_evaluation["mean_feasible_leakage_probability"]),
        mark_error_count=int(combined_evaluation["false_positive_count"])
        + int(combined_evaluation["false_negative_count"]),
        weights=weights,
    )
    return {
        "method": candidate.method,
        "alpha": candidate.alpha,
        "iteration": candidate.iteration,
        "score": score,
        "feasible_domain_evaluation": evaluation_to_dict(feasible_evaluation),
        "combined_oracle_evaluation": combined_evaluation,
        "explicit_grover": {
            key: value for key, value in grover.items() if key != "state_probabilities"
        },
    }


def joint_oracle_score(
    *,
    target_probability: float,
    max_leakage: float,
    mean_leakage: float,
    mark_error_count: int,
    weights: dict[str, float],
) -> float:
    return float(
        target_probability
        - weights["max_leakage"] * max_leakage
        - weights["mean_leakage"] * mean_leakage
        - weights["mark_error"] * mark_error_count
    )


def leakage_selection_key(candidate: dict[str, object]) -> tuple[bool, float, float]:
    evaluation = candidate["combined_oracle_evaluation"]
    return (
        not bool(evaluation["correct_marked_set"]),
        float(evaluation["max_leakage_probability"]),
        float(evaluation["mean_feasible_leakage_probability"]),
    )


def compact_candidate(candidate: dict[str, object]) -> dict[str, object]:
    combined = candidate["combined_oracle_evaluation"]
    grover = candidate["explicit_grover"]
    return {
        "method": candidate["method"],
        "alpha": candidate["alpha"],
        "iteration": candidate["iteration"],
        "score": candidate["score"],
        "correct_marked_set": combined["correct_marked_set"],
        "false_positive_count": combined["false_positive_count"],
        "false_negative_count": combined["false_negative_count"],
        "max_leakage_probability": combined["max_leakage_probability"],
        "mean_feasible_leakage_probability": combined["mean_feasible_leakage_probability"],
        "target_x_probability": grover["target_x_probability"],
        "dirty_ancilla_probability": grover["dirty_ancilla_probability"],
        "one_feasibility_probability": grover["one_feasibility_probability"],
        "one_value_ancilla_probability": grover["one_value_ancilla_probability"],
    }


def parse_weights(raw: str) -> dict[str, float]:
    parts = parse_float_tuple(raw)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("weights must be max_leakage,mean_leakage,mark_error")
    return {
        "max_leakage": float(parts[0]),
        "mean_leakage": float(parts[1]),
        "mark_error": float(parts[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_joint_oracle_training.json"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--target-counts", type=parse_target_counts, default=parse_target_counts("1,16,64"))
    parser.add_argument("--alphas", type=parse_float_tuple, default=parse_float_tuple("1,5,20,100"))
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--weights", type=parse_weights, default=parse_weights("0.35,0.05,10.0"))
    args = parser.parse_args()
    summary = run(
        args.instance,
        args.results,
        args.horizon,
        args.max_order,
        args.target_counts,
        args.alphas,
        args.iterations,
        args.weights,
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
            "candidate_count_per_threshold": row["candidate_count_per_threshold"],
            "target_results": [
                {
                    "target_count_request": item["target_count_request"],
                    "actual_target_count": item["actual_target_count"],
                    "baseline": {
                        key: item["baseline"][key]
                        for key in (
                            "max_leakage_probability",
                            "target_x_probability",
                            "score",
                        )
                    },
                    "leakage_only": {
                        key: item["leakage_only"][key]
                        for key in (
                            "max_leakage_probability",
                            "target_x_probability",
                            "score",
                            "alpha",
                            "iteration",
                        )
                    },
                    "joint_score": {
                        key: item["joint_score"][key]
                        for key in (
                            "max_leakage_probability",
                            "target_x_probability",
                            "score",
                            "alpha",
                            "iteration",
                        )
                    },
                }
                for item in row["target_results"]
            ],
        }
        for row in summary["order_rows"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
