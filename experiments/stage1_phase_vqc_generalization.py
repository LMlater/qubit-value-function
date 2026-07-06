from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.oracle import grover_with_oracle_matrix, phase_oracle_errors, verify_phase_oracle  # noqa: E402
from qubit_value_function.phase_vqc import evaluate_threshold_phase_vqc, train_threshold_phase_vqc  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


TRAIN_THRESHOLDS = [60000.0, 150000.0, 250000.0]
TEST_THRESHOLDS = [45060.0, 90000.0, 120000.0, 200000.0]


def run(instance_path: Path, results_path: Path) -> dict[str, object]:
    instance = load_uc_instance(instance_path)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bits = commitments.reshape((commitments.shape[0], -1))
    bitstrings = [commitment_to_bitstring(commitment) for commitment in commitments]
    evaluator = FixedCommitmentEvaluator(instance)
    values = np.array([evaluator.evaluate(commitment).total_cost for commitment in commitments])

    training = train_threshold_phase_vqc(
        bits,
        values,
        TRAIN_THRESHOLDS,
        x_order=bits.shape[1],
        tau_degree=len(TRAIN_THRESHOLDS) - 1,
    )
    train_eval = evaluate_threshold_phase_vqc(training.model, bits, values, TRAIN_THRESHOLDS)
    test_eval = evaluate_threshold_phase_vqc(training.model, bits, values, TEST_THRESHOLDS)

    grover_results = []
    for tau in TEST_THRESHOLDS:
        exact_marked = values <= tau
        if not exact_marked.any():
            continue
        oracle = training.model.oracle_matrix(bits, tau)
        grover = grover_with_oracle_matrix(oracle, exact_marked)
        grover_results.append(
            {
                "threshold": float(tau),
                "target_count": int(exact_marked.sum()),
                "oracle_checks": verify_phase_oracle(oracle, atol=1e-8),
                "oracle_errors": phase_oracle_errors(oracle),
                "target_probability": grover["target_probability"],
                "non_target_probability": grover["non_target_probability"],
                "iterations": grover["iterations"],
            }
        )

    rows = []
    for bitstring, value in zip(bitstrings, values):
        rows.append({"bitstring": bitstring, "total_cost": float(value)})

    summary = {
        "instance": str(instance_path),
        "method": "threshold generalization for diagonal phase VQC",
        "note": (
            "The model is trained on a few thresholds and evaluated on unseen thresholds. "
            "This checks whether tau-conditioning generalizes beyond direct interpolation points."
        ),
        "fixed_load_mw": instance.fixed_load,
        "generators": [gen.name for gen in instance.generators],
        "train_thresholds": TRAIN_THRESHOLDS,
        "test_thresholds": TEST_THRESHOLDS,
        "train_evaluation": _evaluation_dict(train_eval),
        "test_evaluation": _evaluation_dict(test_eval),
        "grover_on_test_thresholds": grover_results,
        "rows": rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _evaluation_dict(evaluation) -> dict[str, object]:
    return {
        "marked_accuracy": evaluation.marked_accuracy,
        "correct_marked_sets": evaluation.correct_marked_sets,
        "max_phase_error": evaluation.max_phase_error,
        "max_phase_factor_error": evaluation.max_phase_factor_error,
        "threshold_summaries": evaluation.threshold_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/aelmp_simple.json.gz"))
    parser.add_argument("--results", type=Path, default=Path("results/stage1_phase_vqc_generalization.json"))
    args = parser.parse_args()
    summary = run(args.instance, args.results)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
