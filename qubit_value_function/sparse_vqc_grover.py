from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

try:
    from qiskit_aer import AerSimulator
except Exception:  # pragma: no cover - only when Aer is unavailable
    AerSimulator = None

from .coherent_phase_value import (
    QuantizedSparseValueModel,
    build_sparse_vqc_threshold_phase_oracle,
    estimate_statevector_memory_gb,
)
from .gate_level_oracle import bitstring_from_index


@dataclass(frozen=True)
class OrdinaryGroverValidationPlan:
    """Small-instance validation plan; marked-state enumeration is not a builder input."""

    marked_indices: tuple[int, ...]
    marked_count: int
    dimension: int
    iterations: int
    initial_marked_probability: float


@dataclass(frozen=True)
class SparseGroverStatevectorProbe:
    iterations: int
    x_probabilities: np.ndarray
    marked_indices: tuple[int, ...]
    marked_probability: float
    auxiliary_zero_probability: float


@dataclass(frozen=True)
class SparseGroverMPSResult:
    shots: int
    seed: int
    raw_counts: dict[str, int]
    x_counts: dict[str, int]
    x_probabilities: np.ndarray
    auxiliary_zero_probability: float
    total_qubits: int
    estimated_statevector_memory_gb: float
    elapsed_seconds: float


@dataclass(frozen=True)
class MeasuredCandidate:
    index: int
    bitstring: str
    count: int
    probability: float
    was_observed: bool


def build_sparse_vqc_grover_circuit(
    model: QuantizedSparseValueModel,
    *,
    encoded_threshold: int,
    iterations: int,
) -> QuantumCircuit:
    """Build ordinary Grover using the coherent sparse-VQC threshold oracle.

    ``iterations`` is explicit so this scalable circuit builder never enumerates
    the search space to infer a marked-state count.
    """

    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("iterations 不能为负数")

    oracle = build_sparse_vqc_threshold_phase_oracle(
        model,
        encoded_real_threshold=int(encoded_threshold),
        strict=True,
    )
    oracle_gate = oracle.to_gate(label="sparse_vqc_threshold")
    circuit = QuantumCircuit(oracle.num_qubits, name="sparse_vqc_ordinary_grover")
    x_qubits = list(circuit.qubits[: model.num_x_qubits])
    all_qubits = list(circuit.qubits)

    circuit.h(x_qubits)
    for _ in range(iterations):
        circuit.append(oracle_gate, all_qubits)
        append_search_register_diffuser(circuit, x_qubits)
    return circuit


def append_search_register_diffuser(
    circuit: QuantumCircuit,
    x_qubits: Sequence[object],
) -> None:
    """Append the standard inversion-about-the-mean only on the search register."""

    qubits = list(x_qubits)
    if not qubits:
        raise ValueError("search register 至少需要一个 qubit")
    circuit.h(qubits)
    circuit.x(qubits)
    if len(qubits) == 1:
        circuit.z(qubits[0])
    else:
        target = qubits[-1]
        controls = qubits[:-1]
        circuit.h(target)
        circuit.mcx(controls, target)
        circuit.h(target)
    circuit.x(qubits)
    circuit.h(qubits)


def ordinary_grover_validation_plan(
    model: QuantizedSparseValueModel,
    *,
    encoded_threshold: int,
) -> OrdinaryGroverValidationPlan:
    """Enumerate only for small-instance validation and iteration selection."""

    dimension = 2 ** int(model.num_x_qubits)
    marked = tuple(
        index
        for index in range(dimension)
        if model.is_marked(
            _bits_from_index(index, model.num_x_qubits),
            int(encoded_threshold),
            strict=True,
        )
    )
    marked_count = len(marked)
    iterations = (
        0
        if marked_count == 0
        else int(np.floor(np.pi / 4.0 * np.sqrt(float(dimension) / float(marked_count))))
    )
    return OrdinaryGroverValidationPlan(
        marked_indices=marked,
        marked_count=int(marked_count),
        dimension=int(dimension),
        iterations=int(iterations),
        initial_marked_probability=float(marked_count / dimension),
    )


def direct_float_marked_indices_for_validation(
    model: QuantizedSparseValueModel,
    *,
    predict_cost,
    encoded_threshold: int,
) -> tuple[int, ...]:
    """Diagnostic-only marked set from rounding the complete floating prediction."""

    return tuple(
        index
        for index in range(2 ** model.num_x_qubits)
        if model.fixed_point_config.encode(
            float(predict_cost(bitstring_from_index(index, model.num_x_qubits)))
        )
        < int(encoded_threshold)
    )


def marked_semantics_diagnostics(
    sparse_marked_indices: Sequence[int],
    direct_float_marked_indices: Sequence[int],
) -> dict[str, object]:
    sparse = {int(index) for index in sparse_marked_indices}
    direct = {int(index) for index in direct_float_marked_indices}
    return {
        "sparse_integer_marked_indices": sorted(sparse),
        "direct_rounded_float_marked_indices": sorted(direct),
        "marked_set_intersection": sorted(sparse & direct),
        "marked_set_symmetric_difference": sorted(sparse ^ direct),
        "integer_only_marked_indices": sorted(sparse - direct),
        "direct_float_only_marked_indices": sorted(direct - sparse),
        "classification_disagreement_count": int(len(sparse ^ direct)),
    }


def simulate_sparse_vqc_grover_statevector(
    model: QuantizedSparseValueModel,
    *,
    encoded_threshold: int,
    iterations: int,
) -> SparseGroverStatevectorProbe:
    """Exact small-instance validation of the full ordinary-Grover circuit."""

    plan = ordinary_grover_validation_plan(model, encoded_threshold=encoded_threshold)
    circuit = build_sparse_vqc_grover_circuit(
        model,
        encoded_threshold=encoded_threshold,
        iterations=iterations,
    )
    probabilities = Statevector.from_instruction(circuit).probabilities()
    x_probabilities, auxiliary_zero_probability = _x_marginal_probabilities(
        probabilities,
        num_x_qubits=model.num_x_qubits,
    )
    marked_probability = float(
        sum(float(x_probabilities[index]) for index in plan.marked_indices)
    )
    return SparseGroverStatevectorProbe(
        iterations=int(iterations),
        x_probabilities=x_probabilities,
        marked_indices=plan.marked_indices,
        marked_probability=marked_probability,
        auxiliary_zero_probability=float(auxiliary_zero_probability),
    )


def execute_sparse_vqc_grover_mps(
    circuit: QuantumCircuit,
    *,
    num_x_qubits: int,
    shots: int = 4096,
    seed: int = 0,
) -> SparseGroverMPSResult:
    """Execute the complete circuit with Aer MPS and aggregate actual measurements."""

    if AerSimulator is None:
        raise RuntimeError("MPS 模拟需要安装 qiskit-aer")
    num_x_qubits = int(num_x_qubits)
    shots = int(shots)
    seed = int(seed)
    if num_x_qubits <= 0 or num_x_qubits > circuit.num_qubits:
        raise ValueError("num_x_qubits 与量子电路不兼容")
    if shots <= 0:
        raise ValueError("shots 必须为正数")

    total_qubits = int(circuit.num_qubits)
    measured = circuit.copy()
    classical = ClassicalRegister(total_qubits, "measure")
    measured.add_register(classical)
    measured.measure(measured.qubits, classical)

    backend = AerSimulator(method="matrix_product_state")
    started = perf_counter()
    compiled = transpile(
        measured,
        backend,
        optimization_level=1,
        seed_transpiler=seed,
    )
    result = backend.run(
        compiled,
        shots=shots,
        seed_simulator=seed,
    ).result()
    elapsed = perf_counter() - started
    returned_counts = result.get_counts(compiled)

    x_dimension = 2**num_x_qubits
    x_mask = x_dimension - 1
    x_index_counts = np.zeros(x_dimension, dtype=int)
    auxiliary_zero_count = 0
    compact_raw_counts: dict[str, int] = {}
    for raw_bitstring, count in returned_counts.items():
        compact = str(raw_bitstring).replace(" ", "")
        compact_raw_counts[compact] = compact_raw_counts.get(compact, 0) + int(count)
        full_index = int(compact, 2)
        x_index = full_index & x_mask
        x_index_counts[x_index] += int(count)
        if full_index >> num_x_qubits == 0:
            auxiliary_zero_count += int(count)

    actual_shots = int(x_index_counts.sum())
    if actual_shots <= 0:
        raise RuntimeError("MPS 模拟没有返回有效测量结果")
    x_probabilities = x_index_counts.astype(float) / float(actual_shots)
    x_counts = {
        bitstring_from_index(index, num_x_qubits): int(count)
        for index, count in enumerate(x_index_counts)
        if int(count) > 0
    }
    return SparseGroverMPSResult(
        shots=actual_shots,
        seed=seed,
        raw_counts=compact_raw_counts,
        x_counts=x_counts,
        x_probabilities=x_probabilities,
        auxiliary_zero_probability=float(auxiliary_zero_count / actual_shots),
        total_qubits=total_qubits,
        estimated_statevector_memory_gb=estimate_statevector_memory_gb(total_qubits),
        elapsed_seconds=float(elapsed),
    )


def select_measured_candidate(
    x_counts: Mapping[str, int],
    *,
    num_x_qubits: int,
    allowed_indices: Sequence[int],
    observed_indices: Sequence[int] = (),
) -> MeasuredCandidate | None:
    """Select only from actual nonzero counts, preferring an unobserved allowed state."""

    allowed = {int(index) for index in allowed_indices}
    observed = {int(index) for index in observed_indices}
    total = int(sum(int(count) for count in x_counts.values()))
    if total <= 0:
        return None

    rows: list[tuple[int, str, int]] = []
    for bitstring, raw_count in x_counts.items():
        count = int(raw_count)
        if count <= 0:
            continue
        index = _index_from_project_bitstring(bitstring, num_x_qubits)
        if index in allowed:
            rows.append((index, str(bitstring), count))
    if not rows:
        return None

    unobserved = [row for row in rows if row[0] not in observed]
    pool = unobserved if unobserved else rows
    index, bitstring, count = max(pool, key=lambda row: (row[2], -row[0]))
    return MeasuredCandidate(
        index=int(index),
        bitstring=bitstring,
        count=int(count),
        probability=float(count / total),
        was_observed=bool(index in observed),
    )


def _x_marginal_probabilities(
    basis_probabilities: np.ndarray,
    *,
    num_x_qubits: int,
) -> tuple[np.ndarray, float]:
    probabilities = np.asarray(basis_probabilities, dtype=float)
    dimension = 2 ** int(num_x_qubits)
    if probabilities.size % dimension != 0:
        raise ValueError("概率向量与 search register 不兼容")
    x_mask = dimension - 1
    x_probabilities = np.zeros(dimension, dtype=float)
    auxiliary_zero_probability = 0.0
    for basis_index, probability in enumerate(probabilities):
        x_index = int(basis_index) & x_mask
        x_probabilities[x_index] += float(probability)
        if int(basis_index) >> int(num_x_qubits) == 0:
            auxiliary_zero_probability += float(probability)
    return x_probabilities, float(auxiliary_zero_probability)


def _index_from_project_bitstring(bitstring: str, num_bits: int) -> int:
    compact = str(bitstring).replace(" ", "")
    if len(compact) != int(num_bits) or set(compact) - {"0", "1"}:
        raise ValueError("x_counts 中的 bitstring 格式不正确")
    return int(sum(int(bit) << index for index, bit in enumerate(compact)))


def _bits_from_index(index: int, num_qubits: int) -> tuple[int, ...]:
    return tuple((int(index) >> qubit) & 1 for qubit in range(int(num_qubits)))
