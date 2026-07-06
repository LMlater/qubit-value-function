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
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.feature_phase import evaluate_feature_phase_model, fit_feature_phase_model  # noqa: E402
from qubit_value_function.oracle import grover_with_oracle_matrix, phase_oracle_errors, verify_phase_oracle  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


DEFAULT_TARGET_COUNTS = [1, 2, 5, 10, 20, 32]


def hamming_features(bits: np.ndarray, center: np.ndarray, degree: int) -> tuple[np.ndarray, list[str]]:
    distances = np.sum(bits != center.reshape(1, -1), axis=1).astype(float)
    max_distance = max(float(bits.shape[1]), 1.0)
    normalized = distances / max_distance
    features = np.column_stack([normalized**power for power in range(degree + 1)])
    names = ["1"] + [f"(dH/{bits.shape[1]})^{power}" for power in range(1, degree + 1)]
    return features, names


def threshold_for_top_count(values: np.ndarray, target_count: int) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    if target_count <= 0 or target_count > sorted_values.size:
        raise ValueError("target_count is out of range")
    if target_count == sorted_values.size:
        return float(sorted_values[-1])
    return float((sorted_values[target_count - 1] + sorted_values[target_count]) / 2.0)


def run(
    instance_path: Path,
    results_path: Path,
    period: int,
    max_degree: int,
    target_counts: list[int],
) -> dict[str, object]:
    instance = single_period_instance(load_uc_instance(instance_path), period)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bits = commitments.reshape((commitments.shape[0], -1))
    bitstrings = [commitment_to_bitstring(commitment) for commitment in commitments]
    values = np.array([FixedCommitmentEvaluator(instance).evaluate(c).total_cost for c in commitments])
    optimum_idx = int(np.argmin(values))
    center = bits[optimum_idx]

    experiments = []
    for target_count in target_counts:
        tau = threshold_for_top_count(values, target_count)
        labels = values <= tau
        degree_rows = []
        for degree in range(1, max_degree + 1):
            features, names = hamming_features(bits, center, degree)
            model = fit_feature_phase_model(features, labels, names)
            evaluation = evaluate_feature_phase_model(model, features, labels)
            oracle = model.oracle_matrix(features)
            grover = grover_with_oracle_matrix(oracle, labels)
            degree_rows.append(
                {
                    "degree": degree,
                    "feature_count": len(names),
                    "evaluation": evaluation,
                    "oracle_checks": verify_phase_oracle(oracle, atol=1e-8),
                    "oracle_errors": phase_oracle_errors(oracle),
                    "target_probability": grover["target_probability"],
                    "non_target_probability": grover["non_target_probability"],
                    "iterations": grover["iterations"],
                    "coefficients": [float(v) for v in model.coefficients],
                }
            )
        experiments.append(
            {
                "target_count": int(labels.sum()),
                "requested_target_count": target_count,
                "threshold": tau,
                "best_degree": _best_degree(degree_rows),
                "degree_rows": degree_rows,
            }
        )

    rows = sorted(
        [
            {
                "bitstring": bitstring,
                "total_cost": float(value),
                "hamming_to_optimum": int(np.sum(bits[idx] != center)),
            }
            for idx, (bitstring, value) in enumerate(zip(bitstrings, values))
        ],
        key=lambda item: item["total_cost"],
    )
    summary = {
        "instance": str(instance_path),
        "method": "case14 Hamming-distance phase features around the optimum",
        "period": period,
        "fixed_load_mw": instance.fixed_load,
        "generators": [gen.name for gen in instance.generators],
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "optimum": rows[0],
        "runner_up": rows[1],
        "center_bitstring": bitstrings[optimum_idx],
        "max_degree": max_degree,
        "experiments": experiments,
        "rows": rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _best_degree(degree_rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [
        row
        for row in degree_rows
        if row["evaluation"]["correct_marked_set"]
        and row["oracle_errors"]["self_inverse_error"] < 1e-8
        and row["target_probability"] > 0.9
    ]
    if successful:
        row = min(successful, key=lambda item: item["degree"])
    else:
        row = max(degree_rows, key=lambda item: item["target_probability"])
    return {
        "degree": row["degree"],
        "correct_marked_set": row["evaluation"]["correct_marked_set"],
        "max_phase_factor_error": row["evaluation"]["max_phase_factor_error"],
        "self_inverse_error": row["oracle_errors"]["self_inverse_error"],
        "target_probability": row["target_probability"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--results", type=Path, default=Path("results/stage1_case14_hamming_phase.json"))
    parser.add_argument("--period", type=int, default=0)
    parser.add_argument("--max-degree", type=int, default=6)
    parser.add_argument("--target-counts", type=int, nargs="*", default=DEFAULT_TARGET_COUNTS)
    args = parser.parse_args()
    summary = run(args.instance, args.results, args.period, args.max_degree, args.target_counts)
    compact = {
        key: value
        for key, value in summary.items()
        if key not in {"rows", "experiments"}
    }
    compact["experiments"] = [
        {
            "target_count": exp["target_count"],
            "threshold": exp["threshold"],
            "best_degree": exp["best_degree"],
        }
        for exp in summary["experiments"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
