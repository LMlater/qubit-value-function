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
from qubit_value_function.ancilla_vqc import (  # noqa: E402
    ancilla_oracle_errors,
    evaluate_ancilla_vqc,
    fit_ancilla_vqc,
    grover_with_ancilla_oracle,
    verify_ancilla_oracle,
)
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.oracle import grover_search_probabilities  # noqa: E402
from qubit_value_function.phase_vqc import all_x_subsets, x_monomial_features  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def run(results_path: Path) -> dict[str, object]:
    aelmp = run_instance(
        instance_path=Path("data/aelmp_simple.json.gz"),
        instance_name="aelmp_simple",
        period=None,
        threshold_mode="between_best_two",
    )
    case14 = run_instance(
        instance_path=Path("data/case14.json.gz"),
        instance_name="case14_period0",
        period=0,
        threshold_mode="between_best_two",
    )
    summary = {
        "method": "ancilla-based VQC phase oracle O = U^dagger Z_a U",
        "note": (
            "The oracle is always unitary and self-inverse on x plus ancilla. "
            "It behaves like a Grover phase oracle only when the trained ancilla "
            "rotation angles are close to 0 for non-target states and pi for target states."
        ),
        "experiments": [aelmp, case14],
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_instance(
    instance_path: Path,
    instance_name: str,
    period: int | None,
    threshold_mode: str,
) -> dict[str, object]:
    instance = load_uc_instance(instance_path)
    if period is not None:
        instance = single_period_instance(instance, period)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bits = commitments.reshape((commitments.shape[0], -1))
    bitstrings = [commitment_to_bitstring(commitment) for commitment in commitments]
    values = np.array([FixedCommitmentEvaluator(instance).evaluate(c).total_cost for c in commitments])
    threshold = choose_threshold(values, threshold_mode)
    labels = values <= threshold
    exact_grover = grover_search_probabilities(labels)
    order_rows = []
    for order in range(1, bits.shape[1] + 1):
        subsets = all_x_subsets(bits.shape[1], order)
        features = x_monomial_features(bits, subsets)
        names = [subset_name(subset) for subset in subsets]
        model = fit_ancilla_vqc(features, labels, names)
        evaluation = evaluate_ancilla_vqc(model, features, labels)
        oracle = model.oracle_matrix(features)
        grover = grover_with_ancilla_oracle(
            oracle,
            labels,
            iterations=int(exact_grover["iterations"]),
        )
        order_rows.append(
            {
                "x_order": order,
                "feature_count": len(names),
                "evaluation": {
                    "angle_mae": evaluation.angle_mae,
                    "max_angle_error": evaluation.max_angle_error,
                    "max_leakage_probability": evaluation.max_leakage_probability,
                    "mean_leakage_probability": evaluation.mean_leakage_probability,
                    "marked_accuracy": evaluation.marked_accuracy,
                    "correct_marked_set": evaluation.correct_marked_set,
                    "false_positive_count": evaluation.false_positive_count,
                    "false_negative_count": evaluation.false_negative_count,
                },
                "oracle_checks": verify_ancilla_oracle(oracle, atol=1e-8),
                "oracle_errors": ancilla_oracle_errors(oracle),
                "grover": {
                    key: value for key, value in grover.items() if key != "state_probabilities"
                },
            }
        )

    sorted_indices = np.argsort(values)
    return {
        "instance": instance_name,
        "source": str(instance_path),
        "period": period,
        "generators": [gen.name for gen in instance.generators],
        "fixed_load_mw": instance.fixed_load,
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "threshold": threshold,
        "target_count": int(labels.sum()),
        "optimum": {
            "bitstring": bitstrings[int(sorted_indices[0])],
            "total_cost": float(values[sorted_indices[0]]),
        },
        "runner_up": {
            "bitstring": bitstrings[int(sorted_indices[1])],
            "total_cost": float(values[sorted_indices[1]]),
        },
        "exact_grover": {
            key: value for key, value in exact_grover.items() if key != "probabilities"
        },
        "order_rows": order_rows,
    }


def choose_threshold(values: np.ndarray, mode: str) -> float:
    if mode == "between_best_two":
        unique = np.unique(np.sort(values))
        return float((unique[0] + unique[1]) / 2.0)
    if mode.startswith("top"):
        return threshold_for_top_count(values, int(mode[3:]))
    raise ValueError(f"unsupported threshold mode: {mode}")


def subset_name(subset: tuple[int, ...]) -> str:
    if not subset:
        return "1"
    return "*".join(f"x{idx}" for idx in subset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/stage1_ancilla_vqc_oracle.json"))
    args = parser.parse_args()
    summary = run(args.results)
    compact = {
        "method": summary["method"],
        "experiments": [
            {
                key: value
                for key, value in exp.items()
                if key != "order_rows"
            }
            | {
                "order_rows": [
                    {
                        "x_order": row["x_order"],
                        "feature_count": row["feature_count"],
                        "correct_marked_set": row["evaluation"]["correct_marked_set"],
                        "max_leakage_probability": row["evaluation"]["max_leakage_probability"],
                        "target_x_probability": row["grover"]["target_x_probability"],
                        "zero_ancilla_probability": row["grover"]["zero_ancilla_probability"],
                    }
                    for row in exp["order_rows"]
                ]
            }
            for exp in summary["experiments"]
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
