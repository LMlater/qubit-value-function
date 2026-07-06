from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.commitment import (  # noqa: E402
    all_commitments,
    commitment_to_bitstring,
    is_logic_feasible,
)
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.oracle import (  # noqa: E402
    grover_search_probabilities,
    phase_oracle_matrix,
    verify_phase_oracle,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402
from qubit_value_function.vqc import train_vqc_value_function  # noqa: E402


def run(instance_path: Path, results_path: Path, steps: int, seed: int) -> dict[str, object]:
    instance = load_uc_instance(instance_path)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    evaluator = FixedCommitmentEvaluator(instance)

    exact_rows = []
    for commitment in commitments:
        result = evaluator.evaluate(commitment)
        feasible = is_logic_feasible(instance, commitment) and result.success
        exact_rows.append(
            {
                "bitstring": commitment_to_bitstring(commitment),
                "logic_feasible": bool(feasible),
                "total_cost": result.total_cost,
                "dispatch_cost": result.dispatch_cost,
                "startup_cost": result.startup_cost,
                "balance_penalty": result.balance_penalty,
                "reserve_penalty": result.reserve_penalty,
            }
        )

    finite_costs = np.array([row["total_cost"] for row in exact_rows], dtype=float)
    bit_matrix = np.array(
        [[int(bit) for bit in row["bitstring"]] for row in exact_rows],
        dtype=int,
    )
    vqc_result = train_vqc_value_function(bit_matrix, finite_costs, steps=steps, seed=seed)

    order = np.argsort(finite_costs)
    optimum_idx = int(order[0])
    runner_up_idx = int(order[1])
    exact_threshold = float((finite_costs[optimum_idx] + finite_costs[runner_up_idx]) / 2.0)
    predicted_costs = vqc_result.predictions
    marked_exact = finite_costs <= exact_threshold
    marked_predicted = predicted_costs <= exact_threshold
    oracle = phase_oracle_matrix(marked_predicted)
    grover = grover_search_probabilities(marked_predicted)

    for row, pred in zip(exact_rows, predicted_costs):
        row["vqc_prediction"] = float(pred)
        row["vqc_error"] = float(pred - row["total_cost"])
        row["marked_by_exact_threshold"] = bool(row["total_cost"] <= exact_threshold)
        row["marked_by_vqc_oracle"] = bool(pred <= exact_threshold)

    summary = {
        "instance": str(instance_path),
        "source": "https://axavier.org/UnitCommitment.jl/0.3/instances/test/aelmp_simple.json.gz",
        "method": "measurement-readout VQC value-function baseline",
        "note": (
            "This baseline fits V_d(x) through measured probability features and a "
            "classical linear head. It is useful for value-function learnability, "
            "but it is not itself a reversible Grover oracle."
        ),
        "time_horizon": instance.time_horizon,
        "generators": [gen.name for gen in instance.generators],
        "fixed_load_mw": instance.fixed_load,
        "num_commitments": len(exact_rows),
        "optimum": {
            "bitstring": exact_rows[optimum_idx]["bitstring"],
            "total_cost": float(finite_costs[optimum_idx]),
        },
        "runner_up": {
            "bitstring": exact_rows[runner_up_idx]["bitstring"],
            "total_cost": float(finite_costs[runner_up_idx]),
        },
        "threshold": exact_threshold,
        "vqc_metrics": {
            "mae": vqc_result.mae,
            "rmse": vqc_result.rmse,
            "max_abs_error": vqc_result.max_abs_error,
            "normalized_loss": vqc_result.normalized_loss,
            "steps": vqc_result.steps,
            "seed": vqc_result.seed,
            "correct_marked_set": bool(np.array_equal(marked_exact, marked_predicted)),
        },
        "oracle_checks": verify_phase_oracle(oracle),
        "grover": {
            key: value
            for key, value in grover.items()
            if key != "probabilities"
        },
        "rows": exact_rows,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance",
        type=Path,
        default=Path("data/aelmp_simple.json.gz"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_aelmp_simple.json"),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    summary = run(args.instance, args.results, args.steps, args.seed)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
