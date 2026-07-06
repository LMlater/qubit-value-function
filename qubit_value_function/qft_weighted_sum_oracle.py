from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IntegerComparator, QFT
from qiskit.quantum_info import Statevector

from .gate_level_oracle import (
    GateLevelAffinePieceSpec,
    GateLevelGroverResult,
    GateLevelMaxAffineOracleSpec,
    PhaseOracleProbe,
    _append_phase_on_flags,
    _append_x_register_diffuser,
    bitstring_from_index,
    circuit_resource_summary,
    optimal_grover_iterations,
    x_marginal_probabilities,
)


@dataclass
class _QFTPieceWorkspace:
    value_qubits: list[Any]
    flag_qubit: Any
    comparator_ancillas: list[Any]
    compute_gate: Any
    comparator: IntegerComparator
    piece: GateLevelAffinePieceSpec


@dataclass
class _QFTMaxAffineWorkspace:
    circuit: QuantumCircuit
    x_qubits: list[Any]
    pieces: list[_QFTPieceWorkspace]


def build_qft_max_affine_phase_oracle_circuit(
    spec: GateLevelMaxAffineOracleSpec,
) -> QuantumCircuit:
    workspace = _new_qft_max_affine_workspace(spec)
    _append_qft_max_affine_threshold_phase_oracle(workspace)
    return workspace.circuit


def build_qft_max_affine_grover_circuit(
    spec: GateLevelMaxAffineOracleSpec,
    iterations: int | None = None,
) -> QuantumCircuit:
    marked_count = int(spec.marked_mask().sum())
    if marked_count == 0:
        raise ValueError("Grover circuit needs at least one marked state")
    if iterations is None:
        iterations = optimal_grover_iterations(2**spec.num_x_qubits, marked_count)
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")

    workspace = _new_qft_max_affine_workspace(spec)
    workspace.circuit.h(workspace.x_qubits)
    for _ in range(iterations):
        _append_qft_max_affine_threshold_phase_oracle(workspace)
        _append_x_register_diffuser(workspace.circuit, workspace.x_qubits)
    return workspace.circuit


def simulate_qft_max_affine_phase_oracle(
    spec: GateLevelMaxAffineOracleSpec,
) -> PhaseOracleProbe:
    workspace = _new_qft_max_affine_workspace(spec)
    workspace.circuit.h(workspace.x_qubits)
    _append_qft_max_affine_threshold_phase_oracle(workspace)
    statevector = Statevector.from_instruction(workspace.circuit)
    probabilities = statevector.probabilities()
    x_dimension = 2**spec.num_x_qubits
    initial_amplitude = 1.0 / np.sqrt(x_dimension)
    phase_signs = np.zeros(x_dimension, dtype=float)
    for state_index in range(x_dimension):
        phase_signs[state_index] = float(np.real(statevector.data[state_index] / initial_amplitude))
    marked = spec.marked_mask()
    expected = np.where(marked, -1.0, 1.0)
    _, aux_zero_probability = x_marginal_probabilities(probabilities, spec.num_x_qubits)
    return PhaseOracleProbe(
        phase_signs=phase_signs,
        marked_mask=marked,
        aux_zero_probability=float(aux_zero_probability),
        max_phase_error=float(np.max(np.abs(phase_signs - expected))),
    )


def simulate_qft_max_affine_grover(
    spec: GateLevelMaxAffineOracleSpec,
    iterations: int | None = None,
) -> GateLevelGroverResult:
    marked = spec.marked_mask()
    marked_count = int(marked.sum())
    if marked_count == 0:
        raise ValueError("Grover simulation needs at least one marked state")
    if iterations is None:
        iterations = optimal_grover_iterations(2**spec.num_x_qubits, marked_count)
    circuit = build_qft_max_affine_grover_circuit(spec, iterations)
    statevector = Statevector.from_instruction(circuit)
    x_probabilities, aux_zero_probability = x_marginal_probabilities(
        statevector.probabilities(),
        spec.num_x_qubits,
    )
    marked_probability = float(x_probabilities[marked].sum())
    return GateLevelGroverResult(
        iterations=int(iterations),
        x_probabilities=x_probabilities,
        marked_mask=marked,
        marked_probability=marked_probability,
        unmarked_probability=float(1.0 - marked_probability),
        aux_zero_probability=float(aux_zero_probability),
    )


def qft_weighted_sum_resource_summary(
    spec: GateLevelMaxAffineOracleSpec,
    *,
    decompose_reps: int = 3,
) -> dict[str, object]:
    phase = build_qft_max_affine_phase_oracle_circuit(spec)
    grover = build_qft_max_affine_grover_circuit(spec)
    return {
        "phase_oracle": circuit_resource_summary(phase, decompose_reps=decompose_reps),
        "grover_circuit": circuit_resource_summary(grover, decompose_reps=decompose_reps),
    }


def _new_qft_max_affine_workspace(spec: GateLevelMaxAffineOracleSpec) -> _QFTMaxAffineWorkspace:
    x_register = QuantumRegister(spec.num_x_qubits, "x")
    registers = [x_register]
    piece_workspaces: list[_QFTPieceWorkspace] = []
    for piece_index, piece in enumerate(spec.pieces):
        value_qubit_count = _value_qubit_count(piece)
        value_register = QuantumRegister(value_qubit_count, f"qft_v{piece_index}")
        flag_register = QuantumRegister(1, f"qft_flag{piece_index}")
        compare_value = _piece_compare_value(spec.threshold, piece, value_qubit_count)
        comparator = IntegerComparator(value_qubit_count, compare_value, geq=False)
        comparator_ancilla_count = comparator.num_qubits - comparator.num_state_qubits - 1
        comparator_register = _optional_register(comparator_ancilla_count, f"qft_cmp{piece_index}")

        registers.append(value_register)
        registers.append(flag_register)
        if comparator_register is not None:
            registers.append(comparator_register)

        piece_workspaces.append(
            _QFTPieceWorkspace(
                value_qubits=list(value_register),
                flag_qubit=flag_register[0],
                comparator_ancillas=list(comparator_register) if comparator_register is not None else [],
                compute_gate=_qft_weighted_sum_gate(piece, value_qubit_count),
                comparator=comparator,
                piece=piece,
            )
        )

    return _QFTMaxAffineWorkspace(
        circuit=QuantumCircuit(*registers, name=f"qft_{spec.name}"),
        x_qubits=list(x_register),
        pieces=piece_workspaces,
    )


def _append_qft_max_affine_threshold_phase_oracle(
    workspace: _QFTMaxAffineWorkspace,
) -> None:
    for piece_workspace in workspace.pieces:
        workspace.circuit.append(
            piece_workspace.compute_gate,
            workspace.x_qubits + piece_workspace.value_qubits,
        )
        workspace.circuit.append(
            piece_workspace.comparator.to_gate(),
            piece_workspace.value_qubits
            + [piece_workspace.flag_qubit]
            + piece_workspace.comparator_ancillas,
        )

    _append_phase_on_flags(
        workspace.circuit,
        [piece_workspace.flag_qubit for piece_workspace in workspace.pieces],
    )

    for piece_workspace in reversed(workspace.pieces):
        workspace.circuit.append(
            piece_workspace.comparator.to_gate().inverse(),
            piece_workspace.value_qubits
            + [piece_workspace.flag_qubit]
            + piece_workspace.comparator_ancillas,
        )
        workspace.circuit.append(
            piece_workspace.compute_gate.inverse(),
            workspace.x_qubits + piece_workspace.value_qubits,
        )


def _qft_weighted_sum_gate(
    piece: GateLevelAffinePieceSpec,
    value_qubit_count: int,
) -> Any:
    x_register = QuantumRegister(piece.num_x_qubits, "x")
    value_register = QuantumRegister(value_qubit_count, "qft_v")
    circuit = QuantumCircuit(x_register, value_register, name=f"qft_sum_{piece.name}")

    for bit_index in piece.inverted_bit_indices:
        circuit.x(x_register[bit_index])
    circuit.append(QFT(value_qubit_count, do_swaps=False).to_gate(), value_register)
    for bit_index, weight in enumerate(piece.weights):
        if weight == 0:
            continue
        for value_index in range(value_qubit_count):
            angle = 2.0 * np.pi * int(weight) / float(2 ** (value_index + 1))
            circuit.cp(angle, x_register[bit_index], value_register[value_index])
    circuit.append(QFT(value_qubit_count, do_swaps=False).inverse().to_gate(), value_register)
    for bit_index in piece.inverted_bit_indices:
        circuit.x(x_register[bit_index])
    return circuit.to_gate()


def _value_qubit_count(piece: GateLevelAffinePieceSpec) -> int:
    return max(1, int(np.ceil(np.log2(int(sum(piece.weights)) + 1))))


def _piece_compare_value(
    threshold: int,
    piece: GateLevelAffinePieceSpec,
    value_qubit_count: int,
) -> int:
    raw_threshold = int(threshold) - int(piece.bias)
    compare_value = raw_threshold + 1
    return min(max(compare_value, 0), 2**value_qubit_count)


def _optional_register(size: int, name: str) -> QuantumRegister | None:
    if size <= 0:
        return None
    return QuantumRegister(size, name)
