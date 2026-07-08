from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.diagnostics import random_baseline_probabilities  # noqa: E402
from qubit_value_function.experiment_utils import (  # noqa: E402
    embedded_selected_commitments,
    leading_time_window_instance,
    parse_indices,
    sanitize_for_strict_json,
    write_strict_json,
)
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.gate_level_oracle import bitstring_from_index  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def hidden_reference_distribution(
    *,
    instance_path: Path,
    selected_generator_indices: tuple[int, ...],
) -> dict[str, object]:
    source = load_uc_instance(instance_path)
    instance = leading_time_window_instance(source, 2)
    base_commitment = np.ones((len(instance.generators), instance.time_horizon), dtype=int)
    embedded_commitments = embedded_selected_commitments(base_commitment, selected_generator_indices)
    evaluator = FixedCommitmentEvaluator(instance)
    values = []
    for commitment in embedded_commitments:
        result = evaluator.evaluate(commitment)
        values.append(float(result.total_cost) if result.success else float("inf"))
    values_array = np.asarray(values, dtype=float)
    hidden_best = int(np.nanargmin(values_array))
    num_bits = len(selected_generator_indices) * 2
    return {
        "selected_generators": [int(index) for index in selected_generator_indices],
        "num_search_qubits": int(num_bits),
        "dimension": int(values_array.size),
        "hidden_best_index": int(hidden_best),
        "hidden_best_bitstring": bitstring_from_index(hidden_best, num_bits),
        "hidden_best_true_cost": float(values_array[hidden_best]),
        "values": values_array,
    }


def random_budget_baseline(
    *,
    instance_path: Path,
    selected_generator_indices: tuple[int, ...],
    draws: int,
) -> dict[str, object]:
    distribution = hidden_reference_distribution(
        instance_path=instance_path,
        selected_generator_indices=selected_generator_indices,
    )
    probabilities = random_baseline_probabilities(
        values=distribution["values"],
        hidden_best_true_cost=float(distribution["hidden_best_true_cost"]),
        draws=int(draws),
    )
    return {
        **{key: value for key, value in distribution.items() if key != "values"},
        "search_verification_budget": int(draws),
        **probabilities,
        "diagnostic_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--selected-generators", type=parse_indices, default=(0, 5))
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = random_budget_baseline(
        instance_path=args.instance,
        selected_generator_indices=args.selected_generators,
        draws=args.draws,
    )
    if args.output_json is not None:
        write_strict_json(args.output_json, payload)
    print(json.dumps(sanitize_for_strict_json(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
