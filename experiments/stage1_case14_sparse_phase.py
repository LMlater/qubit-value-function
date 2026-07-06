from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_single_period import single_period_instance, threshold_grid  # noqa: E402
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.feature_phase import (  # noqa: E402
    fit_feature_phase_model,
    forward_select_phase_features,
    selected_feature_matrix,
)
from qubit_value_function.oracle import grover_with_oracle_matrix, phase_oracle_errors, verify_phase_oracle  # noqa: E402
from qubit_value_function.phase_vqc import all_x_subsets, x_monomial_features  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def monomial_feature_matrix(bits: np.ndarray) -> tuple[np.ndarray, list[str]]:
    subsets = all_x_subsets(bits.shape[1], bits.shape[1])
    features = x_monomial_features(bits, subsets)
    names = [_subset_name(subset) for subset in subsets]
    return features, names


def run(instance_path: Path, results_path: Path, period: int, max_terms: int) -> dict[str, object]:
    instance = single_period_instance(load_uc_instance(instance_path), period)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bits = commitments.reshape((commitments.shape[0], -1))
    bitstrings = [commitment_to_bitstring(commitment) for commitment in commitments]
    values = np.array([FixedCommitmentEvaluator(instance).evaluate(c).total_cost for c in commitments])
    _, _, tau = threshold_grid(values)
    labels = values <= tau
    features, names = monomial_feature_matrix(bits)
    history = forward_select_phase_features(
        features,
        labels,
        names,
        max_terms=max_terms,
        mandatory=[0],
    )

    grover_rows = []
    for entry in history:
        selected = entry["selected_indices"]
        selected_features, selected_names = selected_feature_matrix(features, names, selected)
        model = fit_feature_phase_model(selected_features, labels, selected_names)
        oracle = model.oracle_matrix(selected_features)
        grover = grover_with_oracle_matrix(oracle, labels)
        grover_rows.append(
            {
                "term_count": entry["term_count"],
                "selected_names": entry["selected_names"],
                "coefficients": entry["coefficients"],
                "evaluation": entry["evaluation"],
                "oracle_checks": verify_phase_oracle(oracle, atol=1e-8),
                "oracle_errors": phase_oracle_errors(oracle),
                "target_probability": grover["target_probability"],
                "non_target_probability": grover["non_target_probability"],
                "iterations": grover["iterations"],
            }
        )

    rows = sorted(
        [
            {"bitstring": bitstring, "total_cost": float(value), "target": bool(label)}
            for bitstring, value, label in zip(bitstrings, values, labels)
        ],
        key=lambda item: item["total_cost"],
    )
    summary = {
        "instance": str(instance_path),
        "method": "case14 fixed-threshold greedy sparse monomial phase selection",
        "period": period,
        "fixed_load_mw": instance.fixed_load,
        "generators": [gen.name for gen in instance.generators],
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "threshold": float(tau),
        "optimum": rows[0],
        "runner_up": rows[1],
        "max_terms": max_terms,
        "first_success": _first_success(grover_rows),
        "grover_rows": grover_rows,
        "rows": rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _subset_name(subset: tuple[int, ...]) -> str:
    if not subset:
        return "1"
    return "*".join(f"x{i}" for i in subset)


def _first_success(grover_rows: list[dict[str, object]]) -> dict[str, object] | None:
    for row in grover_rows:
        if (
            row["evaluation"]["correct_marked_set"]
            and row["target_probability"] > 0.9
            and row["oracle_errors"]["self_inverse_error"] < 1e-8
        ):
            return {
                "term_count": row["term_count"],
                "selected_names": row["selected_names"],
                "target_probability": row["target_probability"],
                "self_inverse_error": row["oracle_errors"]["self_inverse_error"],
                "max_phase_factor_error": row["evaluation"]["max_phase_factor_error"],
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--results", type=Path, default=Path("results/stage1_case14_sparse_phase.json"))
    parser.add_argument("--period", type=int, default=0)
    parser.add_argument("--max-terms", type=int, default=12)
    args = parser.parse_args()
    summary = run(args.instance, args.results, args.period, args.max_terms)
    compact = {
        key: value
        for key, value in summary.items()
        if key not in {"rows", "grover_rows"}
    }
    compact["grover_rows"] = [
        {
            "term_count": row["term_count"],
            "correct": row["evaluation"]["correct_marked_set"],
            "max_phase_factor_error": row["evaluation"]["max_phase_factor_error"],
            "self_inverse_error": row["oracle_errors"]["self_inverse_error"],
            "target_probability": row["target_probability"],
            "selected_names": row["selected_names"],
            "coefficients": row["coefficients"],
        }
        for row in summary["grover_rows"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
