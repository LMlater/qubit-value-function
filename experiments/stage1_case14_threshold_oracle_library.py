from __future__ import annotations

import argparse
import json
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
    fit_feature_phase_model,
    forward_select_phase_features,
    selected_feature_matrix,
)
from qubit_value_function.oracle import (  # noqa: E402
    grover_search_probabilities,
    grover_with_oracle_matrix,
    phase_oracle_errors,
    verify_phase_oracle,
)
from qubit_value_function.phase_vqc import all_x_subsets, x_monomial_features  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


DEFAULT_TARGET_COUNTS = [1, 2, 5, 10, 20, 32]


def monomial_feature_matrix(bits: np.ndarray) -> tuple[np.ndarray, list[str], tuple[tuple[int, ...], ...]]:
    subsets = all_x_subsets(bits.shape[1], bits.shape[1])
    features = x_monomial_features(bits, subsets)
    names = [_subset_name(subset) for subset in subsets]
    return features, names, subsets


def run(
    instance_path: Path,
    results_path: Path,
    period: int,
    max_terms: int,
    target_counts: list[int],
) -> dict[str, object]:
    instance = single_period_instance(load_uc_instance(instance_path), period)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bits = commitments.reshape((commitments.shape[0], -1))
    bitstrings = [commitment_to_bitstring(commitment) for commitment in commitments]
    values = np.array([FixedCommitmentEvaluator(instance).evaluate(c).total_cost for c in commitments])
    features, names, subsets = monomial_feature_matrix(bits)

    library = []
    for target_count in target_counts:
        tau = threshold_for_top_count(values, target_count)
        labels = values <= tau
        exact_grover = grover_search_probabilities(labels)
        history = forward_select_phase_features(
            features,
            labels,
            names,
            max_terms=max_terms,
            mandatory=[0],
        )
        first_success = None
        final_row = None
        for entry in history:
            selected = entry["selected_indices"]
            selected_features, selected_names = selected_feature_matrix(features, names, selected)
            model = fit_feature_phase_model(selected_features, labels, selected_names)
            oracle = model.oracle_matrix(selected_features)
            grover = grover_with_oracle_matrix(oracle, labels)
            row = {
                "term_count": entry["term_count"],
                "selected_names": selected_names,
                "selected_orders": [len(subsets[idx]) for idx in selected],
                "max_selected_order": int(max(len(subsets[idx]) for idx in selected)),
                "evaluation": entry["evaluation"],
                "oracle_checks": verify_phase_oracle(oracle, atol=1e-8),
                "oracle_errors": phase_oracle_errors(oracle),
                "target_probability": grover["target_probability"],
                "non_target_probability": grover["non_target_probability"],
                "iterations": grover["iterations"],
            }
            final_row = row
            if first_success is None and _is_success(row, float(exact_grover["marked_probability"])):
                first_success = row
                break
        library.append(
            {
                "target_count": int(labels.sum()),
                "requested_target_count": target_count,
                "threshold": tau,
                "exact_grover": {
                    key: value for key, value in exact_grover.items() if key != "probabilities"
                },
                "first_success": _compact_success(first_success),
                "final_attempt": _compact_success(final_row),
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
        "method": "case14 fixed-threshold sparse phase oracle library",
        "period": period,
        "fixed_load_mw": instance.fixed_load,
        "generators": [gen.name for gen in instance.generators],
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "max_terms": max_terms,
        "target_counts": target_counts,
        "optimum": rows[0],
        "runner_up": rows[1],
        "library": library,
        "rows": rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _is_success(row: dict[str, object], exact_probability: float) -> bool:
    return bool(
        row["evaluation"]["correct_marked_set"]
        and row["oracle_errors"]["self_inverse_error"] < 1e-8
        and row["target_probability"] >= exact_probability - 1e-8
    )


def _compact_success(row: dict[str, object] | None) -> dict[str, object] | None:
    if row is None:
        return None
    return {
        "term_count": row["term_count"],
        "max_selected_order": row["max_selected_order"],
        "selected_names": row["selected_names"],
        "selected_orders": row["selected_orders"],
        "correct_marked_set": row["evaluation"]["correct_marked_set"],
        "max_phase_factor_error": row["evaluation"]["max_phase_factor_error"],
        "self_inverse_error": row["oracle_errors"]["self_inverse_error"],
        "target_probability": row["target_probability"],
    }


def _subset_name(subset: tuple[int, ...]) -> str:
    if not subset:
        return "1"
    return "*".join(f"x{i}" for i in subset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--results", type=Path, default=Path("results/stage1_case14_threshold_oracle_library.json"))
    parser.add_argument("--period", type=int, default=0)
    parser.add_argument("--max-terms", type=int, default=32)
    parser.add_argument("--target-counts", type=int, nargs="*", default=DEFAULT_TARGET_COUNTS)
    args = parser.parse_args()
    summary = run(args.instance, args.results, args.period, args.max_terms, args.target_counts)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
