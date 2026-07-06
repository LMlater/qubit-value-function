from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.oracle import grover_with_oracle_matrix, phase_oracle_errors, phase_oracle_matrix, verify_phase_oracle  # noqa: E402
from qubit_value_function.phase_vqc import evaluate_threshold_phase_vqc, train_threshold_phase_vqc  # noqa: E402
from qubit_value_function.uc_loader import Reserve, UCInstance, load_uc_instance  # noqa: E402


def single_period_instance(instance: UCInstance, period: int) -> UCInstance:
    if period < 0 or period >= instance.time_horizon:
        raise ValueError("period is out of range")
    reserves = [
        Reserve(
            name=reserve.name,
            amount=[reserve.amount[period]],
            penalty=[reserve.penalty[period]],
        )
        for reserve in instance.reserves
    ]
    return replace(
        instance,
        time_horizon=1,
        fixed_load=[instance.fixed_load[period]],
        reserves=reserves,
        power_balance_penalty=[instance.power_balance_penalty[period]],
    )


def threshold_grid(values: np.ndarray) -> tuple[list[float], list[float], float]:
    unique = np.unique(np.sort(values))
    if unique.size < 4:
        raise ValueError("need at least four unique values to build thresholds")
    optimum_threshold = float((unique[0] + unique[1]) / 2.0)
    train_percentiles = [10, 25, 40, 60, 75, 90]
    test_percentiles = [15, 33, 50, 67, 85]
    train = [float(np.percentile(values, pct)) for pct in train_percentiles]
    test = [float(np.percentile(values, pct)) for pct in test_percentiles]
    train = sorted(set(train + [optimum_threshold]))
    test = sorted(set(test))
    return train, test, optimum_threshold


def run(instance_path: Path, results_path: Path, period: int, max_order: int) -> dict[str, object]:
    full_instance = load_uc_instance(instance_path)
    instance = single_period_instance(full_instance, period)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bits = commitments.reshape((commitments.shape[0], -1))
    bitstrings = [commitment_to_bitstring(commitment) for commitment in commitments]
    evaluator = FixedCommitmentEvaluator(instance)
    values = np.array([evaluator.evaluate(commitment).total_cost for commitment in commitments])
    train_thresholds, test_thresholds, grover_threshold = threshold_grid(values)
    exact_marked = values <= grover_threshold
    exact_oracle = phase_oracle_matrix(exact_marked)
    exact_grover = grover_with_oracle_matrix(exact_oracle, exact_marked)

    fixed_threshold_order_results = []
    for order in range(1, max_order + 1):
        training = train_threshold_phase_vqc(
            bits,
            values,
            [grover_threshold],
            x_order=order,
            tau_degree=0,
        )
        evaluation = evaluate_threshold_phase_vqc(training.model, bits, values, [grover_threshold])
        oracle = training.model.oracle_matrix(bits, grover_threshold)
        vqc_marked = training.model.marked_by_phase(bits, grover_threshold)
        grover = grover_with_oracle_matrix(
            oracle,
            exact_marked,
            iterations=int(exact_grover["iterations"]),
        )
        fixed_threshold_order_results.append(
            {
                "x_order": order,
                "num_gate_terms": len(training.model.gate_terms()),
                "parameters": int(len(training.model.gate_terms())),
                "evaluation": _evaluation_dict(evaluation),
                "grover_threshold_evaluation": {
                    "threshold": grover_threshold,
                    "target_count": int(exact_marked.sum()),
                    "predicted_count": int(vqc_marked.sum()),
                    "correct_marked_set": bool(np.array_equal(exact_marked, vqc_marked)),
                    "oracle_checks": verify_phase_oracle(oracle, atol=1e-8),
                    "oracle_errors": phase_oracle_errors(oracle),
                    "target_probability": grover["target_probability"],
                    "non_target_probability": grover["non_target_probability"],
                    "iterations": grover["iterations"],
                },
            }
        )

    order_results = []
    for order in range(1, max_order + 1):
        training = train_threshold_phase_vqc(
            bits,
            values,
            train_thresholds,
            x_order=order,
            tau_degree=len(train_thresholds) - 1,
        )
        train_eval = evaluate_threshold_phase_vqc(training.model, bits, values, train_thresholds)
        test_eval = evaluate_threshold_phase_vqc(training.model, bits, values, test_thresholds)
        oracle = training.model.oracle_matrix(bits, grover_threshold)
        vqc_marked = training.model.marked_by_phase(bits, grover_threshold)
        grover = grover_with_oracle_matrix(
            oracle,
            exact_marked,
            iterations=int(exact_grover["iterations"]),
        )
        order_results.append(
            {
                "x_order": order,
                "num_gate_terms": len(training.model.gate_terms()),
                "parameters": int(len(training.model.gate_terms()) * len(train_thresholds)),
                "train_evaluation": _evaluation_dict(train_eval),
                "test_evaluation": _evaluation_dict(test_eval),
                "grover_threshold_evaluation": {
                    "threshold": grover_threshold,
                    "target_count": int(exact_marked.sum()),
                    "predicted_count": int(vqc_marked.sum()),
                    "correct_marked_set": bool(np.array_equal(exact_marked, vqc_marked)),
                    "oracle_checks": verify_phase_oracle(oracle, atol=1e-8),
                    "oracle_errors": phase_oracle_errors(oracle),
                    "target_probability": grover["target_probability"],
                    "non_target_probability": grover["non_target_probability"],
                    "iterations": grover["iterations"],
                },
            }
        )

    sorted_indices = np.argsort(values)
    rows = [
        {
            "bitstring": bitstrings[idx],
            "total_cost": float(values[idx]),
        }
        for idx in sorted_indices
    ]
    summary = {
        "instance": str(instance_path),
        "method": "case14 single-period diagonal phase VQC order comparison",
        "period": period,
        "fixed_load_mw": instance.fixed_load,
        "generators": [gen.name for gen in instance.generators],
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "train_thresholds": train_thresholds,
        "test_thresholds": test_thresholds,
        "grover_threshold": grover_threshold,
        "optimum": rows[0],
        "runner_up": rows[1],
        "exact_grover": {
            key: value for key, value in exact_grover.items() if key != "probabilities"
        },
        "fixed_threshold_order_results": fixed_threshold_order_results,
        "order_results": order_results,
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
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--results", type=Path, default=Path("results/stage1_case14_single_period.json"))
    parser.add_argument("--period", type=int, default=0)
    parser.add_argument("--max-order", type=int, default=6)
    args = parser.parse_args()
    summary = run(args.instance, args.results, args.period, args.max_order)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows"}}, indent=2))


if __name__ == "__main__":
    main()
