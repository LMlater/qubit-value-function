from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_single_period import single_period_instance  # noqa: E402
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator, startup_cost  # noqa: E402
from qubit_value_function.feature_phase import evaluate_feature_phase_model, fit_feature_phase_model  # noqa: E402
from qubit_value_function.oracle import grover_search_probabilities, grover_with_oracle_matrix, phase_oracle_errors, verify_phase_oracle  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


DEFAULT_CANDIDATE_COUNTS = [4, 8, 12, 16, 24, 32]


def run(
    instance_path: Path,
    results_path: Path,
    period: int,
    candidate_counts: list[int],
) -> dict[str, object]:
    instance = single_period_instance(load_uc_instance(instance_path), period)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bits = commitments.reshape((commitments.shape[0], -1))
    bitstrings = [commitment_to_bitstring(commitment) for commitment in commitments]
    exact_values = np.array([FixedCommitmentEvaluator(instance).evaluate(c).total_cost for c in commitments])
    surrogate_values = np.array([surrogate_commitment_cost(instance, c) for c in commitments])
    exact_order = np.argsort(exact_values)
    surrogate_order = np.argsort(surrogate_values)
    features, feature_names = hierarchical_features(instance, commitments, surrogate_values)

    experiments = []
    for count in candidate_counts:
        candidate = np.zeros(bits.shape[0], dtype=bool)
        candidate[surrogate_order[:count]] = True
        feature_model = fit_feature_phase_model(features, candidate, feature_names)
        feature_evaluation = evaluate_feature_phase_model(feature_model, features, candidate)
        feature_oracle = feature_model.oracle_matrix(features)
        feature_grover = grover_with_oracle_matrix(feature_oracle, candidate)
        exact_candidate_grover = grover_search_probabilities(candidate)

        candidate_indices = np.where(candidate)[0]
        candidate_exact_values = exact_values[candidate_indices]
        experiments.append(
            {
                "candidate_count": int(candidate.sum()),
                "surrogate_threshold": float(np.max(surrogate_values[candidate])),
                "contains_exact_best": bool(candidate[exact_order[0]]),
                "top5_recall": recall_at_k(candidate, exact_order, 5),
                "top10_recall": recall_at_k(candidate, exact_order, 10),
                "candidate_best_exact_bitstring": bitstrings[int(candidate_indices[np.argmin(candidate_exact_values)])],
                "candidate_best_exact_cost": float(np.min(candidate_exact_values)),
                "candidate_mean_exact_cost": float(np.mean(candidate_exact_values)),
                "candidate_worst_exact_cost": float(np.max(candidate_exact_values)),
                "exact_candidate_grover": {
                    key: value
                    for key, value in exact_candidate_grover.items()
                    if key != "probabilities"
                },
                "feature_phase_oracle": {
                    "evaluation": feature_evaluation,
                    "oracle_checks": verify_phase_oracle(feature_oracle, atol=1e-8),
                    "oracle_errors": phase_oracle_errors(feature_oracle),
                    "target_probability": feature_grover["target_probability"],
                    "non_target_probability": feature_grover["non_target_probability"],
                    "iterations": feature_grover["iterations"],
                },
            }
        )

    rank_rows = []
    for idx in exact_order[:16]:
        rank_rows.append(
            {
                "bitstring": bitstrings[idx],
                "exact_cost": float(exact_values[idx]),
                "surrogate_cost": float(surrogate_values[idx]),
                "surrogate_rank": int(np.where(surrogate_order == idx)[0][0]) + 1,
            }
        )

    summary = {
        "instance": str(instance_path),
        "method": "case14 hierarchical coarse oracle diagnostic",
        "note": (
            "This is a diagnostic experiment. It tests whether a physical surrogate can "
            "coarsely narrow the search region before exact value-function phase marking."
        ),
        "period": period,
        "fixed_load_mw": instance.fixed_load,
        "reserve_requirement_mw": float(sum(reserve.amount[0] for reserve in instance.reserves)),
        "generators": [gen.name for gen in instance.generators],
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "exact_best": {
            "bitstring": bitstrings[int(exact_order[0])],
            "exact_cost": float(exact_values[exact_order[0]]),
            "surrogate_rank": int(np.where(surrogate_order == exact_order[0])[0][0]) + 1,
            "surrogate_cost": float(surrogate_values[exact_order[0]]),
        },
        "surrogate_best": {
            "bitstring": bitstrings[int(surrogate_order[0])],
            "exact_cost": float(exact_values[surrogate_order[0]]),
            "surrogate_cost": float(surrogate_values[surrogate_order[0]]),
        },
        "spearman_like_rank_correlation": rank_correlation(exact_values, surrogate_values),
        "feature_names": feature_names,
        "experiments": experiments,
        "top_exact_rows": rank_rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def surrogate_commitment_cost(instance, commitment: np.ndarray) -> float:
    online = commitment[:, 0].astype(bool)
    load = instance.fixed_load[0]
    reserve_requirement = sum(reserve.amount[0] for reserve in instance.reserves)
    startup = startup_cost(instance, commitment)
    base_mw = 0.0
    base_cost = 0.0
    reserve_capacity = 0.0
    segments: list[tuple[float, float]] = []
    for is_online, gen in zip(online, instance.generators):
        if not is_online:
            continue
        base_mw += gen.p_min
        base_cost += gen.cost_usd[0]
        if gen.reserve_eligibility:
            reserve_capacity += gen.p_max
        for idx in range(1, len(gen.cost_mw)):
            width = gen.cost_mw[idx] - gen.cost_mw[idx - 1]
            if width <= 0:
                continue
            slope = (gen.cost_usd[idx] - gen.cost_usd[idx - 1]) / width
            segments.append((slope, width))

    remaining = load - base_mw
    dispatch_cost = base_cost
    if remaining > 0:
        for slope, width in sorted(segments, key=lambda item: item[0]):
            used = min(remaining, width)
            dispatch_cost += slope * used
            remaining -= used
            if remaining <= 1e-9:
                break
    shortage = max(remaining, 0.0)
    surplus = max(-remaining, 0.0)
    reserve_shortfall = max(reserve_requirement - reserve_capacity, 0.0)
    penalty = instance.power_balance_penalty[0] * (shortage + surplus)
    reserve_penalty = sum(reserve.penalty[0] for reserve in instance.reserves) * reserve_shortfall
    return float(startup + dispatch_cost + penalty + reserve_penalty)


def hierarchical_features(instance, commitments: np.ndarray, surrogate_values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    bits = commitments.reshape((commitments.shape[0], -1)).astype(float)
    pmax = np.array([gen.p_max for gen in instance.generators], dtype=float)
    pmin = np.array([gen.p_min for gen in instance.generators], dtype=float)
    startup = np.array([
        startup_cost(instance, _single_generator_commitment(len(instance.generators), idx))
        for idx in range(len(instance.generators))
    ])
    load = instance.fixed_load[0]
    reserve_requirement = sum(reserve.amount[0] for reserve in instance.reserves)
    reserve_cap = bits @ np.array(
        [gen.p_max if gen.reserve_eligibility else 0.0 for gen in instance.generators],
        dtype=float,
    )
    cap = bits @ pmax
    min_gen = bits @ pmin
    start = bits @ startup
    margin = cap - load
    reserve_margin = reserve_cap - reserve_requirement
    unit_count = bits.sum(axis=1)
    raw = np.column_stack(
        [
            np.ones(bits.shape[0]),
            unit_count,
            cap,
            min_gen,
            start,
            margin,
            reserve_margin,
            margin**2,
            reserve_margin**2,
            surrogate_values,
            surrogate_values**2,
        ]
    )
    names = [
        "1",
        "unit_count",
        "pmax_capacity",
        "pmin_generation",
        "startup_cost",
        "capacity_margin",
        "reserve_margin",
        "capacity_margin_sq",
        "reserve_margin_sq",
        "surrogate_cost",
        "surrogate_cost_sq",
    ]
    features = raw.copy()
    for col in range(1, features.shape[1]):
        scale = np.max(np.abs(features[:, col]))
        if scale > 0:
            features[:, col] = features[:, col] / scale
    return features, names


def recall_at_k(candidate: np.ndarray, exact_order: np.ndarray, k: int) -> float:
    top = exact_order[: min(k, exact_order.size)]
    return float(np.mean(candidate[top]))


def rank_correlation(first_values: np.ndarray, second_values: np.ndarray) -> float:
    first_rank = np.empty_like(first_values, dtype=float)
    second_rank = np.empty_like(second_values, dtype=float)
    first_rank[np.argsort(first_values)] = np.arange(first_values.size)
    second_rank[np.argsort(second_values)] = np.arange(second_values.size)
    if np.std(first_rank) == 0 or np.std(second_rank) == 0:
        return 0.0
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _single_generator_commitment(num_generators: int, generator_idx: int) -> np.ndarray:
    commitment = np.zeros((num_generators, 1), dtype=int)
    commitment[generator_idx, 0] = 1
    return commitment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--results", type=Path, default=Path("results/stage1_case14_hierarchical_oracle.json"))
    parser.add_argument("--period", type=int, default=0)
    parser.add_argument("--candidate-counts", type=int, nargs="*", default=DEFAULT_CANDIDATE_COUNTS)
    args = parser.parse_args()
    summary = run(args.instance, args.results, args.period, args.candidate_counts)
    print(json.dumps({k: v for k, v in summary.items() if k != "top_exact_rows"}, indent=2))


if __name__ == "__main__":
    main()
