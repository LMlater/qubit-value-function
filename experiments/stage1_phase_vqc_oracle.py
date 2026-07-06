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
from qubit_value_function.oracle import (  # noqa: E402
    grover_search_probabilities,
    grover_with_oracle_matrix,
    phase_oracle_errors,
    phase_oracle_matrix,
    verify_phase_oracle,
)
from qubit_value_function.phase_vqc import train_threshold_phase_vqc  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


DEFAULT_THRESHOLDS = [45060.0, 60000.0, 90000.0, 150000.0, 250000.0]


def run(
    instance_path: Path,
    results_path: Path,
    grover_threshold: float,
    thresholds: list[float],
) -> dict[str, object]:
    instance = load_uc_instance(instance_path)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bit_matrix = commitments.reshape((commitments.shape[0], -1))
    bitstrings = [commitment_to_bitstring(commitment) for commitment in commitments]
    evaluator = FixedCommitmentEvaluator(instance)
    values = np.array([evaluator.evaluate(commitment).total_cost for commitment in commitments])

    training = train_threshold_phase_vqc(
        bit_matrix,
        values,
        thresholds,
        x_order=bit_matrix.shape[1],
        tau_degree=len(thresholds) - 1,
    )
    model = training.model

    exact_marked = values <= grover_threshold
    vqc_marked = model.marked_by_phase(bit_matrix, grover_threshold)
    exact_oracle = phase_oracle_matrix(exact_marked)
    vqc_oracle = model.oracle_matrix(bit_matrix, grover_threshold)
    exact_grover = grover_search_probabilities(exact_marked)
    vqc_grover = grover_with_oracle_matrix(
        vqc_oracle,
        exact_marked,
        iterations=int(exact_grover["iterations"]),
    )

    rows = []
    diagonal = model.diagonal(bit_matrix, grover_threshold)
    phases = model.phase(bit_matrix, grover_threshold)
    for bitstring, value, exact, predicted, phase, factor in zip(
        bitstrings,
        values,
        exact_marked,
        vqc_marked,
        phases,
        diagonal,
    ):
        rows.append(
            {
                "bitstring": bitstring,
                "total_cost": float(value),
                "exact_marked": bool(exact),
                "vqc_marked": bool(predicted),
                "phase_radians": float(phase),
                "phase_factor_real": float(np.real(factor)),
                "phase_factor_imag": float(np.imag(factor)),
            }
        )

    summary = {
        "instance": str(instance_path),
        "source": "https://axavier.org/UnitCommitment.jl/0.3/instances/test/aelmp_simple.json.gz",
        "method": "threshold-conditioned diagonal phase VQC oracle",
        "note": (
            "The VQC is a diagonal unitary and does not use measurement readout. "
            "It learns the value-function sublevel sets V_d(x) <= tau."
        ),
        "time_horizon": instance.time_horizon,
        "generators": [gen.name for gen in instance.generators],
        "fixed_load_mw": instance.fixed_load,
        "num_commitments": int(len(values)),
        "thresholds": [float(tau) for tau in thresholds],
        "grover_threshold": float(grover_threshold),
        "phase_vqc_metrics": {
            "x_order": int(bit_matrix.shape[1]),
            "tau_degree": int(len(thresholds) - 1),
            "num_gate_terms": len(model.gate_terms()),
            "max_phase_error": training.max_phase_error,
            "max_phase_factor_error": training.max_phase_factor_error,
            "correct_training_marked_sets": training.correct_marked_sets,
            "residual_norm": training.residual_norm,
            "correct_grover_threshold_marked_set": bool(np.array_equal(exact_marked, vqc_marked)),
        },
        "exact_oracle_checks": verify_phase_oracle(exact_oracle),
        "phase_vqc_oracle_checks": verify_phase_oracle(vqc_oracle, atol=1e-8),
        "phase_vqc_oracle_errors": phase_oracle_errors(vqc_oracle),
        "exact_grover": {
            key: value
            for key, value in exact_grover.items()
            if key != "probabilities"
        },
        "phase_vqc_grover": {
            key: value
            for key, value in vqc_grover.items()
            if key != "probabilities"
        },
        "gate_terms": model.gate_terms(),
        "rows": rows,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/aelmp_simple.json.gz"))
    parser.add_argument("--results", type=Path, default=Path("results/stage1_phase_vqc_oracle.json"))
    parser.add_argument("--grover-threshold", type=float, default=45060.0)
    parser.add_argument("--thresholds", type=float, nargs="*", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    summary = run(args.instance, args.results, args.grover_threshold, args.thresholds)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "gate_terms"}}, indent=2))


if __name__ == "__main__":
    main()
