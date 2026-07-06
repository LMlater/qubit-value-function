from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_hamming_phase import threshold_for_top_count  # noqa: E402
from experiments.stage1_case14_single_period import single_period_instance  # noqa: E402
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.feature_phase import (  # noqa: E402
    evaluate_feature_phase_model,
    fit_feature_phase_model,
    forward_select_phase_features,
    selected_feature_matrix,
)
from qubit_value_function.oracle import grover_search_probabilities, grover_with_oracle_matrix, phase_oracle_errors, verify_phase_oracle  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


DEFAULT_TARGET_COUNTS = [1, 2, 5, 10]


def run(
    instance_path: Path,
    results_path: Path,
    period: int,
    target_counts: list[int],
    max_terms: int,
    min_support_ratio: float,
) -> dict[str, object]:
    instance = single_period_instance(load_uc_instance(instance_path), period)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bits = commitments.reshape((commitments.shape[0], -1))
    bitstrings = [commitment_to_bitstring(commitment) for commitment in commitments]
    values = np.array([FixedCommitmentEvaluator(instance).evaluate(c).total_cost for c in commitments])
    generator_names = [gen.name for gen in instance.generators]

    experiments = []
    for target_count in target_counts:
        tau = threshold_for_top_count(values, target_count)
        labels = values <= tau
        target_indices = np.where(labels)[0]
        exact_grover = grover_search_probabilities(labels)

        open_features, open_names, open_bundles = frequent_open_bundle_features(
            bits,
            target_indices,
            generator_names,
            min_support_ratio=min_support_ratio,
        )
        open_result = greedy_bundle_result(open_features, open_names, labels, max_terms, exact_grover)

        pattern_features, pattern_names, pattern_bundles = exact_pattern_features(
            bits,
            target_indices,
            generator_names,
        )
        pattern_model = fit_feature_phase_model(pattern_features, labels, pattern_names)
        pattern_eval = evaluate_feature_phase_model(pattern_model, pattern_features, labels)
        pattern_oracle = pattern_model.oracle_matrix(pattern_features)
        pattern_grover = grover_with_oracle_matrix(pattern_oracle, labels)
        pattern_result = {
            "feature_count": len(pattern_names),
            "feature_names": pattern_names,
            "bundles": pattern_bundles,
            "evaluation": pattern_eval,
            "oracle_checks": verify_phase_oracle(pattern_oracle, atol=1e-8),
            "oracle_errors": phase_oracle_errors(pattern_oracle),
            "target_probability": pattern_grover["target_probability"],
            "exact_target_probability": exact_grover["marked_probability"],
            "iterations": pattern_grover["iterations"],
        }

        experiments.append(
            {
                "target_count": int(labels.sum()),
                "requested_target_count": target_count,
                "threshold": tau,
                "exact_grover": {
                    key: value for key, value in exact_grover.items() if key != "probabilities"
                },
                "open_bundle_candidate_count": len(open_names),
                "open_bundle_result": open_result,
                "pattern_bundle_result": pattern_result,
            }
        )

    rows = sorted(
        [
            {"bitstring": bitstring, "total_cost": float(value)}
            for bitstring, value in zip(bitstrings, values)
        ],
        key=lambda item: item["total_cost"],
    )
    summary = {
        "instance": str(instance_path),
        "method": "case14 frequent open-bundle and signed-pattern phase features",
        "period": period,
        "fixed_load_mw": instance.fixed_load,
        "generators": generator_names,
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "target_counts": target_counts,
        "max_terms": max_terms,
        "min_support_ratio": min_support_ratio,
        "optimum": rows[0],
        "runner_up": rows[1],
        "experiments": experiments,
        "rows": rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def frequent_open_bundle_features(
    bits: np.ndarray,
    target_indices: np.ndarray,
    generator_names: list[str],
    *,
    min_support_ratio: float,
) -> tuple[np.ndarray, list[str], list[dict[str, object]]]:
    target_bits = bits[target_indices]
    min_support = max(1, int(np.ceil(len(target_indices) * min_support_ratio)))
    bundles: list[tuple[int, ...]] = [()]
    for size in range(1, bits.shape[1] + 1):
        for subset in combinations(range(bits.shape[1]), size):
            support = int(np.sum(np.all(target_bits[:, subset] == 1, axis=1)))
            if support >= min_support:
                bundles.append(subset)
    features = np.column_stack([open_bundle_column(bits, bundle) for bundle in bundles])
    names = [bundle_name(bundle, (), generator_names) for bundle in bundles]
    metadata = [
        {
            "on": [generator_names[idx] for idx in bundle],
            "off": [],
            "order": len(bundle),
        }
        for bundle in bundles
    ]
    return features, names, metadata


def exact_pattern_features(
    bits: np.ndarray,
    target_indices: np.ndarray,
    generator_names: list[str],
) -> tuple[np.ndarray, list[str], list[dict[str, object]]]:
    patterns = []
    seen = set()
    for idx in target_indices:
        on = tuple(np.where(bits[idx] == 1)[0])
        off = tuple(np.where(bits[idx] == 0)[0])
        key = (on, off)
        if key not in seen:
            seen.add(key)
            patterns.append(key)
    features = np.column_stack([signed_bundle_column(bits, on, off) for on, off in patterns])
    names = [bundle_name(on, off, generator_names) for on, off in patterns]
    metadata = [
        {
            "on": [generator_names[idx] for idx in on],
            "off": [generator_names[idx] for idx in off],
            "order": len(on) + len(off),
        }
        for on, off in patterns
    ]
    return features, names, metadata


def greedy_bundle_result(
    features: np.ndarray,
    names: list[str],
    labels: np.ndarray,
    max_terms: int,
    exact_grover: dict[str, object],
) -> dict[str, object] | None:
    if features.size == 0:
        return None
    history = forward_select_phase_features(
        features,
        labels,
        names,
        max_terms=min(max_terms, features.shape[1]),
        mandatory=[0],
    )
    exact_probability = float(exact_grover["marked_probability"])
    final = None
    first_success = None
    for entry in history:
        selected = entry["selected_indices"]
        selected_features, selected_names = selected_feature_matrix(features, names, selected)
        model = fit_feature_phase_model(selected_features, labels, selected_names)
        oracle = model.oracle_matrix(selected_features)
        grover = grover_with_oracle_matrix(oracle, labels)
        row = {
            "term_count": entry["term_count"],
            "selected_names": selected_names,
            "evaluation": entry["evaluation"],
            "oracle_checks": verify_phase_oracle(oracle, atol=1e-8),
            "oracle_errors": phase_oracle_errors(oracle),
            "target_probability": grover["target_probability"],
            "exact_target_probability": exact_probability,
            "iterations": grover["iterations"],
        }
        final = row
        if (
            first_success is None
            and row["evaluation"]["correct_marked_set"]
            and row["oracle_errors"]["self_inverse_error"] < 1e-8
            and row["target_probability"] >= exact_probability - 1e-8
        ):
            first_success = row
            break
    return {
        "first_success": compact_row(first_success),
        "final_attempt": compact_row(final),
        "candidate_count": features.shape[1],
    }


def compact_row(row: dict[str, object] | None) -> dict[str, object] | None:
    if row is None:
        return None
    return {
        "term_count": row["term_count"],
        "selected_names": row["selected_names"],
        "correct_marked_set": row["evaluation"]["correct_marked_set"],
        "max_phase_factor_error": row["evaluation"]["max_phase_factor_error"],
        "self_inverse_error": row["oracle_errors"]["self_inverse_error"],
        "target_probability": row["target_probability"],
        "exact_target_probability": row["exact_target_probability"],
    }


def open_bundle_column(bits: np.ndarray, on: tuple[int, ...]) -> np.ndarray:
    if not on:
        return np.ones(bits.shape[0])
    return np.prod(bits[:, on], axis=1)


def signed_bundle_column(bits: np.ndarray, on: tuple[int, ...], off: tuple[int, ...]) -> np.ndarray:
    column = np.ones(bits.shape[0])
    if on:
        column *= np.prod(bits[:, on], axis=1)
    if off:
        column *= np.prod(1 - bits[:, off], axis=1)
    return column


def bundle_name(on: tuple[int, ...], off: tuple[int, ...], generator_names: list[str]) -> str:
    if not on and not off:
        return "1"
    parts = []
    if on:
        parts.append("ON[" + ",".join(generator_names[idx] for idx in on) + "]")
    if off:
        parts.append("OFF[" + ",".join(generator_names[idx] for idx in off) + "]")
    return ";".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--results", type=Path, default=Path("results/stage1_case14_bundle_phase.json"))
    parser.add_argument("--period", type=int, default=0)
    parser.add_argument("--target-counts", type=int, nargs="*", default=DEFAULT_TARGET_COUNTS)
    parser.add_argument("--max-terms", type=int, default=20)
    parser.add_argument("--min-support-ratio", type=float, default=0.6)
    args = parser.parse_args()
    summary = run(
        args.instance,
        args.results,
        args.period,
        args.target_counts,
        args.max_terms,
        args.min_support_ratio,
    )
    compact = {
        key: value
        for key, value in summary.items()
        if key not in {"rows", "experiments"}
    }
    compact["experiments"] = [
        {
            "target_count": exp["target_count"],
            "open_bundle_result": exp["open_bundle_result"],
            "pattern_bundle_summary": {
                "feature_count": exp["pattern_bundle_result"]["feature_count"],
                "correct_marked_set": exp["pattern_bundle_result"]["evaluation"]["correct_marked_set"],
                "max_phase_factor_error": exp["pattern_bundle_result"]["evaluation"]["max_phase_factor_error"],
                "self_inverse_error": exp["pattern_bundle_result"]["oracle_errors"]["self_inverse_error"],
                "target_probability": exp["pattern_bundle_result"]["target_probability"],
                "exact_target_probability": exp["pattern_bundle_result"]["exact_target_probability"],
            },
        }
        for exp in summary["experiments"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
