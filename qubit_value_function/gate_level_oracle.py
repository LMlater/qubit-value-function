from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IntegerComparator, WeightedAdder
from qiskit.quantum_info import Statevector


@dataclass(frozen=True)
class GateLevelAffineOracleSpec:
    """Integer affine value-register oracle specification.

    The represented value is
    sum_i weights[i] * z_i, where z_i is x_i or 1 - x_i for inverted inputs.
    The phase oracle marks states with value <= threshold.
    """

    weights: tuple[int, ...]
    threshold: int
    inverted_bit_indices: tuple[int, ...] = ()
    bit_labels: tuple[str, ...] = ()
    name: str = "affine_threshold_oracle"

    def __post_init__(self) -> None:
        weights = tuple(int(weight) for weight in self.weights)
        if not weights:
            raise ValueError("at least one weight is required")
        if any(weight < 0 for weight in weights):
            raise ValueError("weights must be nonnegative integers")
        threshold = int(self.threshold)
        if threshold < 0:
            raise ValueError("threshold must be nonnegative")
        inverted = tuple(sorted({int(index) for index in self.inverted_bit_indices}))
        if any(index < 0 or index >= len(weights) for index in inverted):
            raise ValueError("inverted bit indices must be within the weight vector")
        labels = tuple(self.bit_labels) if self.bit_labels else tuple(
            f"x{index}" for index in range(len(weights))
        )
        if len(labels) != len(weights):
            raise ValueError("bit_labels must have the same length as weights")

        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "inverted_bit_indices", inverted)
        object.__setattr__(self, "bit_labels", labels)

    @property
    def num_x_qubits(self) -> int:
        return len(self.weights)

    @property
    def max_value(self) -> int:
        return int(sum(self.weights))

    def values_for_all_x(self) -> np.ndarray:
        values = np.zeros(2**self.num_x_qubits, dtype=int)
        inverted = set(self.inverted_bit_indices)
        for state_index in range(values.size):
            total = 0
            for bit_index, weight in enumerate(self.weights):
                bit_value = (state_index >> bit_index) & 1
                term_value = 1 - bit_value if bit_index in inverted else bit_value
                total += weight * term_value
            values[state_index] = total
        return values

    def marked_mask(self) -> np.ndarray:
        return self.values_for_all_x() <= self.threshold


@dataclass(frozen=True)
class GateLevelAffinePieceSpec:
    weights: tuple[int, ...]
    inverted_bit_indices: tuple[int, ...] = ()
    name: str = "piece"
    bias: int = 0

    def __post_init__(self) -> None:
        weights = tuple(int(weight) for weight in self.weights)
        if not weights:
            raise ValueError("at least one weight is required")
        if any(weight < 0 for weight in weights):
            raise ValueError("weights must be nonnegative integers")
        inverted = tuple(sorted({int(index) for index in self.inverted_bit_indices}))
        if any(index < 0 or index >= len(weights) for index in inverted):
            raise ValueError("inverted bit indices must be within the weight vector")

        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "inverted_bit_indices", inverted)
        object.__setattr__(self, "bias", int(self.bias))

    @property
    def num_x_qubits(self) -> int:
        return len(self.weights)

    def values_for_all_x(self) -> np.ndarray:
        values = np.zeros(2**self.num_x_qubits, dtype=int)
        inverted = set(self.inverted_bit_indices)
        for state_index in range(values.size):
            total = int(self.bias)
            for bit_index, weight in enumerate(self.weights):
                bit_value = (state_index >> bit_index) & 1
                term_value = 1 - bit_value if bit_index in inverted else bit_value
                total += weight * term_value
            values[state_index] = total
        return values


@dataclass(frozen=True)
class GateLevelMaxAffineOracleSpec:
    pieces: tuple[GateLevelAffinePieceSpec, ...]
    threshold: int
    bit_labels: tuple[str, ...] = ()
    name: str = "max_affine_threshold_oracle"

    def __post_init__(self) -> None:
        pieces = tuple(self.pieces)
        if not pieces:
            raise ValueError("at least one affine piece is required")
        num_x_qubits = pieces[0].num_x_qubits
        if any(piece.num_x_qubits != num_x_qubits for piece in pieces):
            raise ValueError("all affine pieces must use the same x-register size")
        threshold = int(self.threshold)
        if threshold < 0:
            raise ValueError("threshold must be nonnegative")
        labels = tuple(self.bit_labels) if self.bit_labels else tuple(
            f"x{index}" for index in range(num_x_qubits)
        )
        if len(labels) != num_x_qubits:
            raise ValueError("bit_labels must match the x-register size")

        object.__setattr__(self, "pieces", pieces)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "bit_labels", labels)

    @property
    def num_x_qubits(self) -> int:
        return self.pieces[0].num_x_qubits

    @property
    def piece_count(self) -> int:
        return len(self.pieces)

    def piece_values_for_all_x(self) -> np.ndarray:
        return np.column_stack([piece.values_for_all_x() for piece in self.pieces])

    def values_for_all_x(self) -> np.ndarray:
        return np.max(self.piece_values_for_all_x(), axis=1)

    def marked_mask(self) -> np.ndarray:
        return self.values_for_all_x() <= self.threshold


@dataclass(frozen=True)
class PhaseOracleProbe:
    phase_signs: np.ndarray
    marked_mask: np.ndarray
    aux_zero_probability: float
    max_phase_error: float


@dataclass(frozen=True)
class GateLevelGroverResult:
    iterations: int
    x_probabilities: np.ndarray
    marked_mask: np.ndarray
    marked_probability: float
    unmarked_probability: float
    aux_zero_probability: float


@dataclass
class _OracleWorkspace:
    circuit: QuantumCircuit
    x_qubits: list[Any]
    value_qubits: list[Any]
    carry_qubits: list[Any]
    control_qubits: list[Any]
    flag_qubit: Any
    comparator_ancillas: list[Any]
    adder: WeightedAdder
    comparator: IntegerComparator


@dataclass
class _PieceWorkspace:
    value_qubits: list[Any]
    carry_qubits: list[Any]
    control_qubits: list[Any]
    flag_qubit: Any
    comparator_ancillas: list[Any]
    adder: WeightedAdder
    comparator: IntegerComparator
    piece: GateLevelAffinePieceSpec


@dataclass
class _MaxAffineWorkspace:
    circuit: QuantumCircuit
    x_qubits: list[Any]
    pieces: list[_PieceWorkspace]


def build_affine_phase_oracle_circuit(spec: GateLevelAffineOracleSpec) -> QuantumCircuit:
    """Build U_f^dagger Z_c U_f for value <= threshold."""

    workspace = _new_workspace(spec)
    _append_affine_threshold_phase_oracle(workspace, spec)
    return workspace.circuit


def build_max_affine_phase_oracle_circuit(spec: GateLevelMaxAffineOracleSpec) -> QuantumCircuit:
    """Build a threshold-equivalent max-affine phase oracle.

    The circuit marks max_r L_r(x) <= tau by computing and comparing every
    affine piece, applying a multi-controlled phase on all comparison flags,
    then uncomputing the comparisons and piece values.
    """

    workspace = _new_max_affine_workspace(spec)
    _append_max_affine_threshold_phase_oracle(workspace, spec)
    return workspace.circuit


def build_affine_grover_circuit(
    spec: GateLevelAffineOracleSpec,
    iterations: int | None = None,
) -> QuantumCircuit:
    marked_count = int(spec.marked_mask().sum())
    if marked_count == 0:
        raise ValueError("Grover circuit needs at least one marked state")
    if iterations is None:
        iterations = optimal_grover_iterations(2**spec.num_x_qubits, marked_count)
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")

    workspace = _new_workspace(spec)
    workspace.circuit.h(workspace.x_qubits)
    for _ in range(iterations):
        _append_affine_threshold_phase_oracle(workspace, spec)
        _append_x_register_diffuser(workspace.circuit, workspace.x_qubits)
    return workspace.circuit


def build_max_affine_grover_circuit(
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

    workspace = _new_max_affine_workspace(spec)
    workspace.circuit.h(workspace.x_qubits)
    for _ in range(iterations):
        _append_max_affine_threshold_phase_oracle(workspace, spec)
        _append_x_register_diffuser(workspace.circuit, workspace.x_qubits)
    return workspace.circuit


def simulate_affine_phase_oracle(spec: GateLevelAffineOracleSpec) -> PhaseOracleProbe:
    workspace = _new_workspace(spec)
    workspace.circuit.h(workspace.x_qubits)
    _append_affine_threshold_phase_oracle(workspace, spec)
    statevector = Statevector.from_instruction(workspace.circuit)
    probabilities = statevector.probabilities()
    x_dimension = 2**spec.num_x_qubits
    initial_amplitude = 1.0 / np.sqrt(x_dimension)
    phase_signs = np.zeros(x_dimension, dtype=float)
    for state_index in range(x_dimension):
        phase_signs[state_index] = float(np.real(statevector.data[state_index] / initial_amplitude))
    marked = spec.marked_mask()
    expected = np.where(marked, -1.0, 1.0)
    aux_zero_probability = _aux_zero_probability(probabilities, spec.num_x_qubits)
    return PhaseOracleProbe(
        phase_signs=phase_signs,
        marked_mask=marked,
        aux_zero_probability=float(aux_zero_probability),
        max_phase_error=float(np.max(np.abs(phase_signs - expected))),
    )


def simulate_max_affine_phase_oracle(spec: GateLevelMaxAffineOracleSpec) -> PhaseOracleProbe:
    workspace = _new_max_affine_workspace(spec)
    workspace.circuit.h(workspace.x_qubits)
    _append_max_affine_threshold_phase_oracle(workspace, spec)
    statevector = Statevector.from_instruction(workspace.circuit)
    probabilities = statevector.probabilities()
    x_dimension = 2**spec.num_x_qubits
    initial_amplitude = 1.0 / np.sqrt(x_dimension)
    phase_signs = np.zeros(x_dimension, dtype=float)
    for state_index in range(x_dimension):
        phase_signs[state_index] = float(np.real(statevector.data[state_index] / initial_amplitude))
    marked = spec.marked_mask()
    expected = np.where(marked, -1.0, 1.0)
    aux_zero_probability = _aux_zero_probability(probabilities, spec.num_x_qubits)
    return PhaseOracleProbe(
        phase_signs=phase_signs,
        marked_mask=marked,
        aux_zero_probability=float(aux_zero_probability),
        max_phase_error=float(np.max(np.abs(phase_signs - expected))),
    )


def simulate_affine_grover(
    spec: GateLevelAffineOracleSpec,
    iterations: int | None = None,
) -> GateLevelGroverResult:
    marked = spec.marked_mask()
    marked_count = int(marked.sum())
    if marked_count == 0:
        raise ValueError("Grover simulation needs at least one marked state")
    if iterations is None:
        iterations = optimal_grover_iterations(2**spec.num_x_qubits, marked_count)
    circuit = build_affine_grover_circuit(spec, iterations)
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


def simulate_max_affine_grover(
    spec: GateLevelMaxAffineOracleSpec,
    iterations: int | None = None,
) -> GateLevelGroverResult:
    marked = spec.marked_mask()
    marked_count = int(marked.sum())
    if marked_count == 0:
        raise ValueError("Grover simulation needs at least one marked state")
    if iterations is None:
        iterations = optimal_grover_iterations(2**spec.num_x_qubits, marked_count)
    circuit = build_max_affine_grover_circuit(spec, iterations)
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


def x_marginal_probabilities(
    basis_probabilities: np.ndarray,
    num_x_qubits: int,
) -> tuple[np.ndarray, float]:
    basis_probabilities = np.asarray(basis_probabilities, dtype=float)
    x_dimension = 2**num_x_qubits
    if basis_probabilities.size % x_dimension != 0:
        raise ValueError("basis probability vector is incompatible with num_x_qubits")

    x_mask = x_dimension - 1
    x_probabilities = np.zeros(x_dimension, dtype=float)
    aux_zero_probability = 0.0
    for basis_index, probability in enumerate(basis_probabilities):
        x_index = basis_index & x_mask
        x_probabilities[x_index] += probability
        if basis_index >> num_x_qubits == 0:
            aux_zero_probability += probability
    return x_probabilities, aux_zero_probability


def circuit_resource_summary(
    circuit: QuantumCircuit,
    *,
    decompose_reps: int = 3,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "num_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "operations": _operation_counts(circuit),
    }
    if decompose_reps > 0:
        decomposed = circuit.decompose(reps=decompose_reps)
        summary["decomposed_depth"] = int(decomposed.depth())
        summary["decomposed_operations"] = _operation_counts(decomposed)
    return summary


def bitstring_from_index(state_index: int, num_bits: int) -> str:
    return "".join(str((int(state_index) >> bit_index) & 1) for bit_index in range(num_bits))


def optimal_grover_iterations(dimension: int, marked_count: int) -> int:
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("dimension must be a positive power of two")
    if marked_count <= 0:
        return 0
    if marked_count > dimension:
        raise ValueError("marked_count cannot exceed dimension")
    return max(1, int(np.floor(np.pi / 4.0 * np.sqrt(dimension / marked_count))))


def _new_workspace(spec: GateLevelAffineOracleSpec) -> _OracleWorkspace:
    adder = WeightedAdder(spec.num_x_qubits, list(spec.weights))
    compare_value = spec.threshold + 1
    if compare_value > 2**adder.num_sum_qubits:
        raise ValueError("threshold is outside the weighted-adder sum register range")
    comparator = IntegerComparator(adder.num_sum_qubits, compare_value, geq=False)

    x_register = QuantumRegister(spec.num_x_qubits, "x")
    value_register = QuantumRegister(adder.num_sum_qubits, "v")
    carry_register = _optional_register(adder.num_carry_qubits, "carry")
    control_register = _optional_register(adder.num_control_qubits, "ctrl")
    flag_register = QuantumRegister(1, "flag")
    comparator_ancilla_count = comparator.num_qubits - comparator.num_state_qubits - 1
    comparator_register = _optional_register(comparator_ancilla_count, "cmp")

    registers = [x_register, value_register]
    if carry_register is not None:
        registers.append(carry_register)
    if control_register is not None:
        registers.append(control_register)
    registers.append(flag_register)
    if comparator_register is not None:
        registers.append(comparator_register)

    return _OracleWorkspace(
        circuit=QuantumCircuit(*registers, name=spec.name),
        x_qubits=list(x_register),
        value_qubits=list(value_register),
        carry_qubits=list(carry_register) if carry_register is not None else [],
        control_qubits=list(control_register) if control_register is not None else [],
        flag_qubit=flag_register[0],
        comparator_ancillas=list(comparator_register) if comparator_register is not None else [],
        adder=adder,
        comparator=comparator,
    )


def _new_max_affine_workspace(spec: GateLevelMaxAffineOracleSpec) -> _MaxAffineWorkspace:
    x_register = QuantumRegister(spec.num_x_qubits, "x")
    registers = [x_register]
    piece_workspaces: list[_PieceWorkspace] = []

    for piece_index, piece in enumerate(spec.pieces):
        adder = WeightedAdder(spec.num_x_qubits, list(piece.weights))
        compare_value = _piece_compare_value(spec.threshold, piece, adder.num_sum_qubits)
        if compare_value > 2**adder.num_sum_qubits:
            raise ValueError("threshold is outside a weighted-adder sum register range")
        comparator = IntegerComparator(adder.num_sum_qubits, compare_value, geq=False)
        value_register = QuantumRegister(adder.num_sum_qubits, f"v{piece_index}")
        carry_register = _optional_register(adder.num_carry_qubits, f"carry{piece_index}")
        control_register = _optional_register(adder.num_control_qubits, f"ctrl{piece_index}")
        flag_register = QuantumRegister(1, f"flag{piece_index}")
        comparator_ancilla_count = comparator.num_qubits - comparator.num_state_qubits - 1
        comparator_register = _optional_register(comparator_ancilla_count, f"cmp{piece_index}")

        registers.append(value_register)
        if carry_register is not None:
            registers.append(carry_register)
        if control_register is not None:
            registers.append(control_register)
        registers.append(flag_register)
        if comparator_register is not None:
            registers.append(comparator_register)

        piece_workspaces.append(
            _PieceWorkspace(
                value_qubits=list(value_register),
                carry_qubits=list(carry_register) if carry_register is not None else [],
                control_qubits=list(control_register) if control_register is not None else [],
                flag_qubit=flag_register[0],
                comparator_ancillas=list(comparator_register) if comparator_register is not None else [],
                adder=adder,
                comparator=comparator,
                piece=piece,
            )
        )

    return _MaxAffineWorkspace(
        circuit=QuantumCircuit(*registers, name=spec.name),
        x_qubits=list(x_register),
        pieces=piece_workspaces,
    )


def _append_affine_threshold_phase_oracle(
    workspace: _OracleWorkspace,
    spec: GateLevelAffineOracleSpec,
) -> None:
    for bit_index in spec.inverted_bit_indices:
        workspace.circuit.x(workspace.x_qubits[bit_index])
    workspace.circuit.append(
        workspace.adder.to_gate(),
        workspace.x_qubits
        + workspace.value_qubits
        + workspace.carry_qubits
        + workspace.control_qubits,
    )
    workspace.circuit.append(
        workspace.comparator.to_gate(),
        workspace.value_qubits + [workspace.flag_qubit] + workspace.comparator_ancillas,
    )
    workspace.circuit.z(workspace.flag_qubit)
    workspace.circuit.append(
        workspace.comparator.to_gate().inverse(),
        workspace.value_qubits + [workspace.flag_qubit] + workspace.comparator_ancillas,
    )
    workspace.circuit.append(
        workspace.adder.to_gate().inverse(),
        workspace.x_qubits
        + workspace.value_qubits
        + workspace.carry_qubits
        + workspace.control_qubits,
    )
    for bit_index in spec.inverted_bit_indices:
        workspace.circuit.x(workspace.x_qubits[bit_index])


def _append_max_affine_threshold_phase_oracle(
    workspace: _MaxAffineWorkspace,
    spec: GateLevelMaxAffineOracleSpec,
) -> None:
    for piece_workspace in workspace.pieces:
        _append_piece_value_compute(workspace.circuit, workspace.x_qubits, piece_workspace)
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
        _append_piece_value_uncompute(workspace.circuit, workspace.x_qubits, piece_workspace)


def _append_piece_value_compute(
    circuit: QuantumCircuit,
    x_qubits: list[Any],
    piece_workspace: _PieceWorkspace,
) -> None:
    for bit_index in piece_workspace.piece.inverted_bit_indices:
        circuit.x(x_qubits[bit_index])
    circuit.append(
        piece_workspace.adder.to_gate(),
        x_qubits
        + piece_workspace.value_qubits
        + piece_workspace.carry_qubits
        + piece_workspace.control_qubits,
    )
    for bit_index in piece_workspace.piece.inverted_bit_indices:
        circuit.x(x_qubits[bit_index])


def _append_piece_value_uncompute(
    circuit: QuantumCircuit,
    x_qubits: list[Any],
    piece_workspace: _PieceWorkspace,
) -> None:
    for bit_index in piece_workspace.piece.inverted_bit_indices:
        circuit.x(x_qubits[bit_index])
    circuit.append(
        piece_workspace.adder.to_gate().inverse(),
        x_qubits
        + piece_workspace.value_qubits
        + piece_workspace.carry_qubits
        + piece_workspace.control_qubits,
    )
    for bit_index in piece_workspace.piece.inverted_bit_indices:
        circuit.x(x_qubits[bit_index])


def _append_phase_on_flags(circuit: QuantumCircuit, flag_qubits: list[Any]) -> None:
    if not flag_qubits:
        raise ValueError("at least one flag qubit is required")
    if len(flag_qubits) == 1:
        circuit.z(flag_qubits[0])
        return
    target = flag_qubits[-1]
    controls = flag_qubits[:-1]
    circuit.h(target)
    circuit.mcx(controls, target)
    circuit.h(target)


def _append_x_register_diffuser(circuit: QuantumCircuit, x_qubits: list[Any]) -> None:
    circuit.h(x_qubits)
    circuit.x(x_qubits)
    if len(x_qubits) == 1:
        circuit.z(x_qubits[0])
    else:
        target = x_qubits[-1]
        controls = x_qubits[:-1]
        circuit.h(target)
        circuit.mcx(controls, target)
        circuit.h(target)
    circuit.x(x_qubits)
    circuit.h(x_qubits)


def _optional_register(size: int, name: str) -> QuantumRegister | None:
    if size <= 0:
        return None
    return QuantumRegister(size, name)


def _piece_compare_value(
    threshold: int,
    piece: GateLevelAffinePieceSpec,
    num_sum_qubits: int,
) -> int:
    raw_threshold = int(threshold) - int(piece.bias)
    compare_value = raw_threshold + 1
    return min(max(compare_value, 0), 2**num_sum_qubits)


def _aux_zero_probability(probabilities: np.ndarray, num_x_qubits: int) -> float:
    _, aux_zero_probability = x_marginal_probabilities(probabilities, num_x_qubits)
    return aux_zero_probability


def _operation_counts(circuit: QuantumCircuit) -> dict[str, int]:
    return {str(name): int(count) for name, count in circuit.count_ops().items()}

