from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_t2_ancilla_vqc import (  # noqa: E402
    commitment_row,
    evaluate_values,
    leading_time_window_instance,
)
from experiments.stage1_case14_t2_gate_level_grover_oracle import (  # noqa: E402
    embedded_selected_commitments,
)
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.gate_level_oracle import (  # noqa: E402
    GateLevelAffinePieceSpec,
    GateLevelMaxAffineOracleSpec,
    bitstring_from_index,
    build_max_affine_grover_circuit,
    build_max_affine_phase_oracle_circuit,
    circuit_resource_summary,
    simulate_max_affine_grover,
    simulate_max_affine_phase_oracle,
)
from qubit_value_function.uc_loader import UCInstance, load_uc_instance  # noqa: E402


def run(
    instance_path: Path,
    results_path: Path,
    horizon: int,
    selected_generator_indices: tuple[int, ...],
) -> dict[str, object]:
    if horizon != 2:
        raise ValueError("this max-affine gate-level prototype is restricted to T=2")
    if len(selected_generator_indices) != 2:
        raise ValueError("the current max-affine prototype uses two selected generators")

    source_instance = load_uc_instance(instance_path)
    instance = leading_time_window_instance(source_instance, horizon)
    generator_names = [gen.name for gen in instance.generators]
    commitments = all_commitments(len(instance.generators), instance.time_horizon)

    value_start = time.perf_counter()
    values, logic_feasible = evaluate_values(instance, commitments)
    value_seconds = time.perf_counter() - value_start
    finite = np.isfinite(values)
    finite_sorted_indices = [int(index) for index in np.argsort(values) if finite[index]]
    full_optimum_index = finite_sorted_indices[0]
    full_optimum_commitment = commitments[full_optimum_index]

    spec = case14_t2_gate_level_max_affine_spec(instance, selected_generator_indices)
    embedded_commitments = embedded_selected_commitments(
        full_optimum_commitment,
        selected_generator_indices,
    )
    embedded_values, embedded_logic_feasible = evaluate_values(instance, embedded_commitments)
    embedded_finite = np.isfinite(embedded_values)
    embedded_best_index = int(np.nanargmin(embedded_values))

    phase_start = time.perf_counter()
    phase_probe = simulate_max_affine_phase_oracle(spec)
    phase_seconds = time.perf_counter() - phase_start

    grover_start = time.perf_counter()
    grover_result = simulate_max_affine_grover(spec)
    grover_seconds = time.perf_counter() - grover_start

    phase_circuit = build_max_affine_phase_oracle_circuit(spec)
    grover_circuit = build_max_affine_grover_circuit(spec, grover_result.iterations)
    max_values = spec.values_for_all_x()
    piece_values = spec.piece_values_for_all_x()
    marked_indices = [int(index) for index in np.flatnonzero(spec.marked_mask())]

    summary = {
        "instance": f"case14_T{horizon}",
        "source": str(instance_path),
        "method": "T=2 selected-subregister Qiskit gate-level max-affine value oracle",
        "scope": {
            "full_case14_bits": int(commitments.reshape((commitments.shape[0], -1)).shape[1]),
            "gate_level_x_bits": int(spec.num_x_qubits),
            "piece_count": int(spec.piece_count),
            "note": (
                "This is a two-piece max-affine gate-level prototype on a T=2 selected "
                "generator subregister. It proves the max-affine threshold oracle structure, "
                "but does not yet synthesize the full 12-bit, 207-feature, 32-piece T=2 model."
            ),
        },
        "fixed_load_mw": instance.fixed_load,
        "generators": generator_names,
        "selected_generators": [
            generator_names[index] for index in selected_generator_indices
        ],
        "bit_order": "selected generator-major order: g_i_t0,g_i_t1,...",
        "max_affine_oracle": spec_to_dict(spec),
        "oracle_decomposition": {
            "piece_registers": "compute each integer affine piece L_r(x) into its own value register",
            "piece_comparators": "compare each L_r(x) <= tau and store the result in a flag qubit",
            "max_threshold_logic": "max_r L_r(x) <= tau is implemented as AND_r[L_r(x) <= tau]",
            "phase_mark": "multi-controlled phase on the piece-comparison flags",
            "uncompute": "inverse comparators and inverse adders restore all non-x registers",
            "diffuser": "standard Grover diffuser acts only on the x register",
        },
        "full_t2_reference": {
            "logic_feasible_count": int(logic_feasible.sum()),
            "finite_value_count": int(finite.sum()),
            "value_evaluation_seconds": float(value_seconds),
            "optimum": commitment_row(
                commitments,
                generator_names,
                values,
                full_optimum_index,
            ),
        },
        "embedded_subspace_validation": {
            "base_commitment": commitment_to_bitstring(full_optimum_commitment),
            "base_note": "Non-selected generators are fixed to the exhaustive T=2 optimum for validation only.",
            "subspace_size": int(embedded_commitments.shape[0]),
            "logic_feasible_count": int(embedded_logic_feasible.sum()),
            "finite_value_count": int(embedded_finite.sum()),
            "best_selected_bitstring": bitstring_from_index(
                embedded_best_index,
                spec.num_x_qubits,
            ),
            "best_full_bitstring": commitment_to_bitstring(
                embedded_commitments[embedded_best_index]
            ),
            "best_true_cost": _finite_or_none(embedded_values[embedded_best_index]),
            "marked_selected_bitstrings": [
                bitstring_from_index(index, spec.num_x_qubits)
                for index in marked_indices
            ],
            "marked_full_bitstrings": [
                commitment_to_bitstring(embedded_commitments[index])
                for index in marked_indices
            ],
        },
        "phase_oracle_check": {
            "marked_count": int(phase_probe.marked_mask.sum()),
            "aux_zero_probability": float(phase_probe.aux_zero_probability),
            "max_phase_error": float(phase_probe.max_phase_error),
            "simulation_seconds": float(phase_seconds),
        },
        "grover_result": {
            "iterations": int(grover_result.iterations),
            "marked_probability": float(grover_result.marked_probability),
            "unmarked_probability": float(grover_result.unmarked_probability),
            "aux_zero_probability": float(grover_result.aux_zero_probability),
            "argmax_selected_bitstring": bitstring_from_index(
                int(np.argmax(grover_result.x_probabilities)),
                spec.num_x_qubits,
            ),
            "simulation_seconds": float(grover_seconds),
            "top_probability_rows": top_probability_rows(
                grover_result.x_probabilities,
                max_values,
                piece_values,
                spec.marked_mask(),
                embedded_commitments,
                embedded_values,
                spec.num_x_qubits,
            ),
        },
        "resources": {
            "phase_oracle": circuit_resource_summary(phase_circuit, decompose_reps=3),
            "grover_circuit": circuit_resource_summary(grover_circuit, decompose_reps=2),
        },
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def case14_t2_gate_level_max_affine_spec(
    instance: UCInstance,
    selected_generator_indices: tuple[int, ...],
) -> GateLevelMaxAffineOracleSpec:
    if instance.time_horizon != 2:
        raise ValueError("this prototype assumes T=2")
    if len(selected_generator_indices) != 2:
        raise ValueError("this prototype expects two selected generators")
    first = instance.generators[selected_generator_indices[0]]
    second = instance.generators[selected_generator_indices[1]]
    labels = (
        f"{first.name}_t0",
        f"{first.name}_t1",
        f"{second.name}_t0",
        f"{second.name}_t1",
    )
    pieces = (
        GateLevelAffinePieceSpec(
            weights=(1, 1, 1, 0),
            inverted_bit_indices=(0, 1),
            name="L0_first_on_second_t0_off",
        ),
        GateLevelAffinePieceSpec(
            weights=(0, 0, 1, 1),
            inverted_bit_indices=(),
            name="L1_second_off_both_periods",
        ),
    )
    return GateLevelMaxAffineOracleSpec(
        pieces=pieces,
        threshold=0,
        bit_labels=labels,
        name="case14_t2_two_piece_max_affine_oracle",
    )


def top_probability_rows(
    probabilities: np.ndarray,
    max_values: np.ndarray,
    piece_values: np.ndarray,
    marked: np.ndarray,
    embedded_commitments: np.ndarray,
    embedded_values: np.ndarray,
    num_bits: int,
    limit: int = 10,
) -> list[dict[str, object]]:
    rows = []
    for state_index in np.argsort(probabilities)[::-1][:limit]:
        rows.append(
            {
                "selected_bitstring": bitstring_from_index(int(state_index), num_bits),
                "probability": float(probabilities[int(state_index)]),
                "piece_values": [
                    int(value) for value in piece_values[int(state_index)].tolist()
                ],
                "max_affine_value": int(max_values[int(state_index)]),
                "marked": bool(marked[int(state_index)]),
                "embedded_full_bitstring": commitment_to_bitstring(
                    embedded_commitments[int(state_index)]
                ),
                "embedded_true_cost": _finite_or_none(
                    embedded_values[int(state_index)]
                ),
            }
        )
    return rows


def spec_to_dict(spec: GateLevelMaxAffineOracleSpec) -> dict[str, object]:
    return {
        "value_formula": "V_hat(x) = max_r L_r(x)",
        "threshold_tau": int(spec.threshold),
        "bit_labels": list(spec.bit_labels),
        "pieces": [
            {
                "name": piece.name,
                "weights": [int(weight) for weight in piece.weights],
                "inverted_bit_indices": [
                    int(index) for index in piece.inverted_bit_indices
                ],
                "terms": piece_terms(piece, spec.bit_labels),
            }
            for piece in spec.pieces
        ],
    }


def piece_terms(
    piece: GateLevelAffinePieceSpec,
    bit_labels: tuple[str, ...],
) -> list[str]:
    inverted = set(piece.inverted_bit_indices)
    terms = []
    for bit_index, (label, weight) in enumerate(zip(bit_labels, piece.weights)):
        if weight == 0:
            continue
        if bit_index in inverted:
            terms.append(f"{weight}*(1-{label})")
        else:
            terms.append(f"{weight}*{label}")
    return terms


def parse_indices(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _finite_or_none(value: float) -> float | None:
    if np.isfinite(value):
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_gate_level_max_affine_oracle.json"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--selected-generators", type=parse_indices, default=(0, 1))
    args = parser.parse_args()

    summary = run(
        args.instance,
        args.results,
        args.horizon,
        args.selected_generators,
    )
    compact = {
        "instance": summary["instance"],
        "method": summary["method"],
        "scope": summary["scope"],
        "selected_generators": summary["selected_generators"],
        "phase_oracle_check": summary["phase_oracle_check"],
        "grover_result": {
            key: value
            for key, value in summary["grover_result"].items()
            if key != "top_probability_rows"
        },
        "embedded_subspace_validation": summary["embedded_subspace_validation"],
        "resources": {
            "phase_qubits": summary["resources"]["phase_oracle"]["num_qubits"],
            "grover_qubits": summary["resources"]["grover_circuit"]["num_qubits"],
            "grover_depth": summary["resources"]["grover_circuit"]["depth"],
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
