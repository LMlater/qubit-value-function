from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseGate
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


@dataclass(frozen=True)
class PhaseFeature:
    """A local Boolean monomial encoded by a diagonal phase gate."""

    kind: str
    qubits: tuple[int, ...]
    label: str

    def __post_init__(self) -> None:
        qubits = tuple(int(index) for index in self.qubits)
        if self.kind not in {"linear", "pair"}:
            raise ValueError("kind 必须是 linear 或 pair")
        expected_size = 1 if self.kind == "linear" else 2
        if len(qubits) != expected_size:
            raise ValueError("feature 的 qubit 数与 kind 不一致")
        if any(index < 0 for index in qubits):
            raise ValueError("feature qubit 索引不能为负数")
        if len(set(qubits)) != len(qubits):
            raise ValueError("pair feature 不能重复使用同一 qubit")
        object.__setattr__(self, "qubits", qubits)

    def evaluate(self, bits: Sequence[int]) -> float:
        result = 1
        for index in self.qubits:
            result *= int(bits[index])
        return float(result)


@dataclass(frozen=True)
class SparsePhaseVQC:
    """Sparse diagonal phase model.

    Parameters are stored in cycles rather than radians. For a bit vector x,
    the circuit phase is

        exp(2πi * (phase_intercept + Σ_j phase_weights[j] f_j(x))).

    The unwrapped phase is mapped affinely back to the learned cost.
    """

    num_generators: int
    num_periods: int
    features: tuple[PhaseFeature, ...]
    phase_intercept: float
    phase_weights: tuple[float, ...]
    cost_center: float
    cost_scale: float
    phase_center: float = 0.25
    phase_scale: float = 0.20
    generator_edges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        num_generators = int(self.num_generators)
        num_periods = int(self.num_periods)
        if num_generators <= 0 or num_periods <= 0:
            raise ValueError("num_generators 和 num_periods 必须为正数")
        weights = tuple(float(value) for value in self.phase_weights)
        features = tuple(self.features)
        if len(weights) != len(features):
            raise ValueError("phase_weights 数量必须与 features 一致")
        if not np.isfinite(self.phase_intercept) or any(
            not np.isfinite(value) for value in weights
        ):
            raise ValueError("phase parameters 必须全部有限")
        if not np.isfinite(self.cost_center) or not np.isfinite(self.phase_center):
            raise ValueError("cost_center 和 phase_center 必须有限")
        if not np.isfinite(self.cost_scale) or float(self.cost_scale) <= 0.0:
            raise ValueError("cost_scale 必须为有限正数")
        if not np.isfinite(self.phase_scale) or float(self.phase_scale) <= 0.0:
            raise ValueError("phase_scale 必须为有限正数")
        if any(
            index >= num_generators * num_periods
            for feature in features
            for index in feature.qubits
        ):
            raise ValueError("feature qubit 索引超出模型范围")
        object.__setattr__(self, "num_generators", num_generators)
        object.__setattr__(self, "num_periods", num_periods)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "phase_weights", weights)
        object.__setattr__(
            self,
            "generator_edges",
            tuple((int(a), int(b)) for a, b in self.generator_edges),
        )

    @property
    def num_qubits(self) -> int:
        return self.num_generators * self.num_periods

    @property
    def num_parameters(self) -> int:
        return 1 + len(self.phase_weights)

    def feature_values(self, bits: Sequence[int] | str) -> np.ndarray:
        row = _coerce_bits(bits, self.num_qubits)
        return np.array([feature.evaluate(row) for feature in self.features], dtype=float)

    def unwrapped_phase(self, bits: Sequence[int] | str) -> float:
        return float(
            self.phase_intercept
            + np.dot(np.asarray(self.phase_weights, dtype=float), self.feature_values(bits))
        )

    def wrapped_phase(self, bits: Sequence[int] | str) -> float:
        return float(np.mod(self.unwrapped_phase(bits), 1.0))

    def phase_factor(self, bits: Sequence[int] | str) -> complex:
        return complex(np.exp(2.0j * np.pi * self.unwrapped_phase(bits)))

    def predict_cost(self, bits: Sequence[int] | str) -> float:
        normalized = (self.unwrapped_phase(bits) - self.phase_center) / self.phase_scale
        return float(self.cost_center + self.cost_scale * normalized)

    def predict_costs(self, bitstrings: Sequence[Sequence[int] | str]) -> np.ndarray:
        return np.array([self.predict_cost(bits) for bits in bitstrings], dtype=float)

    def as_dict(self) -> dict[str, object]:
        return {
            "num_generators": int(self.num_generators),
            "num_periods": int(self.num_periods),
            "num_qubits": int(self.num_qubits),
            "num_parameters": int(self.num_parameters),
            "phase_intercept_cycles": float(self.phase_intercept),
            "phase_weights_cycles": [float(value) for value in self.phase_weights],
            "cost_center": float(self.cost_center),
            "cost_scale": float(self.cost_scale),
            "phase_center": float(self.phase_center),
            "phase_scale": float(self.phase_scale),
            "generator_edges": [[int(a), int(b)] for a, b in self.generator_edges],
            "features": [
                {
                    "kind": feature.kind,
                    "qubits": [int(index) for index in feature.qubits],
                    "label": feature.label,
                    "weight_cycles": float(weight),
                }
                for feature, weight in zip(self.features, self.phase_weights)
            ],
        }


@dataclass(frozen=True)
class SparsePhaseFitResult:
    model: SparsePhaseVQC
    initial_loss: float
    final_loss: float
    optimizer_success: bool
    optimizer_message: str
    iterations: int
    function_evaluations: int
    initial_parameters: tuple[float, ...]
    final_parameters: tuple[float, ...]
    target_phases: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "initial_loss": float(self.initial_loss),
            "final_loss": float(self.final_loss),
            "optimizer_success": bool(self.optimizer_success),
            "optimizer_message": self.optimizer_message,
            "iterations": int(self.iterations),
            "function_evaluations": int(self.function_evaluations),
            "initial_parameters": [float(value) for value in self.initial_parameters],
            "final_parameters": [float(value) for value in self.final_parameters],
            "target_phases": [float(value) for value in self.target_phases],
        }


def default_generator_edges(num_generators: int) -> tuple[tuple[int, int], ...]:
    """Use a nearest-neighbour generator chain as the default sparse graph."""

    num_generators = int(num_generators)
    if num_generators <= 0:
        raise ValueError("num_generators 必须为正数")
    return tuple((index, index + 1) for index in range(num_generators - 1))


def build_local_phase_features(
    num_generators: int,
    num_periods: int,
    *,
    generator_edges: Iterable[tuple[int, int]] | None = None,
) -> tuple[PhaseFeature, ...]:
    """Build O(GT + G(T-1) + |E|T) local phase features."""

    num_generators = int(num_generators)
    num_periods = int(num_periods)
    if num_generators <= 0 or num_periods <= 0:
        raise ValueError("num_generators 和 num_periods 必须为正数")

    raw_edges = (
        default_generator_edges(num_generators)
        if generator_edges is None
        else tuple((int(a), int(b)) for a, b in generator_edges)
    )
    normalized_edges: list[tuple[int, int]] = []
    for a, b in raw_edges:
        if a == b or a < 0 or b < 0 or a >= num_generators or b >= num_generators:
            raise ValueError("generator edge 超出范围或包含自环")
        edge = (min(a, b), max(a, b))
        if edge not in normalized_edges:
            normalized_edges.append(edge)

    features: list[PhaseFeature] = []
    for generator in range(num_generators):
        for period in range(num_periods):
            qubit = generator * num_periods + period
            features.append(
                PhaseFeature(
                    kind="linear",
                    qubits=(qubit,),
                    label=f"g{generator}_t{period}",
                )
            )

    for generator in range(num_generators):
        for period in range(num_periods - 1):
            first = generator * num_periods + period
            second = first + 1
            features.append(
                PhaseFeature(
                    kind="pair",
                    qubits=(first, second),
                    label=f"temporal_g{generator}_t{period}_t{period + 1}",
                )
            )

    for first_generator, second_generator in normalized_edges:
        for period in range(num_periods):
            first = first_generator * num_periods + period
            second = second_generator * num_periods + period
            features.append(
                PhaseFeature(
                    kind="pair",
                    qubits=(first, second),
                    label=f"spatial_g{first_generator}_g{second_generator}_t{period}",
                )
            )
    return tuple(features)


def feature_matrix(
    bitstrings: Sequence[Sequence[int] | str],
    features: Sequence[PhaseFeature],
    *,
    num_qubits: int,
) -> np.ndarray:
    rows = [_coerce_bits(bits, int(num_qubits)) for bits in bitstrings]
    return np.array(
        [[feature.evaluate(bits) for feature in features] for bits in rows],
        dtype=float,
    )


def fit_sparse_phase_vqc(
    *,
    bitstrings: Sequence[Sequence[int] | str],
    costs: Sequence[float],
    num_generators: int,
    num_periods: int,
    generator_edges: Iterable[tuple[int, int]] | None = None,
    seed: int = 0,
    regularization: float = 1e-4,
    maxiter: int = 300,
    phase_center: float = 0.25,
    phase_scale: float = 0.20,
) -> SparsePhaseFitResult:
    """Train a sparse diagonal VQC using an exact phase-expectation loss.

    The optimized objective is the analytic expectation-equivalent form of a
    noiseless Hadamard test. Circuit-level Statevector validation is provided
    separately by ``hadamard_phase_expectation``.
    """

    num_generators = int(num_generators)
    num_periods = int(num_periods)
    num_qubits = num_generators * num_periods
    if len(bitstrings) == 0:
        raise ValueError("至少需要一个训练样本")
    if len(bitstrings) != len(costs):
        raise ValueError("bitstrings 与 costs 长度必须一致")
    if regularization < 0.0 or not np.isfinite(regularization):
        raise ValueError("regularization 必须为有限非负数")
    if int(maxiter) <= 0:
        raise ValueError("maxiter 必须为正数")
    if not 0.0 < phase_center < 1.0:
        raise ValueError("phase_center 必须位于 (0, 1)")
    if not 0.0 < phase_scale < min(phase_center, 1.0 - phase_center):
        raise ValueError("phase_scale 必须保证训练相位远离模 1 边界")

    provided_edges = (
        None
        if generator_edges is None
        else tuple((int(a), int(b)) for a, b in generator_edges)
    )
    features = build_local_phase_features(
        num_generators,
        num_periods,
        generator_edges=provided_edges,
    )
    feature_values = feature_matrix(bitstrings, features, num_qubits=num_qubits)
    design = np.column_stack([np.ones(len(bitstrings), dtype=float), feature_values])
    targets = np.asarray(costs, dtype=float)
    if not np.all(np.isfinite(targets)):
        raise ValueError("训练 costs 必须全部有限")

    cost_center = float(np.mean(targets))
    cost_scale = float(max(np.max(np.abs(targets - cost_center)), 1.0))
    target_phases = phase_center + phase_scale * (targets - cost_center) / cost_scale

    rng = np.random.default_rng(int(seed))
    initial = np.zeros(design.shape[1], dtype=float)
    initial[0] = phase_center + float(rng.normal(0.0, 0.01))
    if initial.size > 1:
        initial[1:] = rng.normal(0.0, 0.02, size=initial.size - 1)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        predicted = design @ parameters
        phase_difference = predicted - target_phases
        angles = 2.0 * np.pi * phase_difference
        data_loss = float(np.mean(2.0 - 2.0 * np.cos(angles)))
        regularization_loss = float(
            regularization * np.dot(parameters[1:], parameters[1:])
        )
        gradient = design.T @ (4.0 * np.pi * np.sin(angles)) / float(design.shape[0])
        if parameters.size > 1:
            gradient[1:] += 2.0 * regularization * parameters[1:]
        return data_loss + regularization_loss, np.asarray(gradient, dtype=float)

    initial_loss = float(objective(initial)[0])
    result = minimize(
        fun=lambda parameters: objective(parameters)[0],
        x0=initial,
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
        bounds=[(-0.25, 0.75)] + [(-0.5, 0.5)] * len(features),
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8},
    )
    final_parameters = np.asarray(result.x, dtype=float)
    final_loss = float(objective(final_parameters)[0])
    edges = (
        default_generator_edges(num_generators)
        if provided_edges is None
        else tuple((min(a, b), max(a, b)) for a, b in provided_edges)
    )
    model = SparsePhaseVQC(
        num_generators=num_generators,
        num_periods=num_periods,
        features=features,
        phase_intercept=float(final_parameters[0]),
        phase_weights=tuple(float(value) for value in final_parameters[1:]),
        cost_center=cost_center,
        cost_scale=cost_scale,
        phase_center=float(phase_center),
        phase_scale=float(phase_scale),
        generator_edges=tuple(dict.fromkeys(edges)),
    )
    return SparsePhaseFitResult(
        model=model,
        initial_loss=initial_loss,
        final_loss=final_loss,
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        function_evaluations=int(getattr(result, "nfev", 0)),
        initial_parameters=tuple(float(value) for value in initial),
        final_parameters=tuple(float(value) for value in final_parameters),
        target_phases=tuple(float(value) for value in target_phases),
    )


def build_phase_circuit(
    model: SparsePhaseVQC,
    bits: Sequence[int] | str | None = None,
    *,
    include_intercept: bool = True,
) -> QuantumCircuit:
    """Build U_theta and optionally prepare one computational-basis input."""

    circuit = QuantumCircuit(model.num_qubits, name="sparse_phase_vqc")
    if bits is not None:
        row = _coerce_bits(bits, model.num_qubits)
        for index, value in enumerate(row):
            if value:
                circuit.x(index)
    if include_intercept:
        circuit.global_phase += 2.0 * np.pi * model.phase_intercept
    _append_phase_terms(circuit, model, qubit_offset=0)
    return circuit


def build_hadamard_readout_circuit(
    model: SparsePhaseVQC,
    bits: Sequence[int] | str,
) -> QuantumCircuit:
    """Prepare a noiseless Hadamard-test state for the model phase."""

    row = _coerce_bits(bits, model.num_qubits)
    circuit = QuantumCircuit(model.num_qubits + 1, name="phase_hadamard_readout")
    readout = 0
    for index, value in enumerate(row):
        if value:
            circuit.x(index + 1)
    circuit.h(readout)
    circuit.p(2.0 * np.pi * model.phase_intercept, readout)
    for feature, weight in zip(model.features, model.phase_weights):
        angle = 2.0 * np.pi * float(weight)
        if feature.kind == "linear":
            circuit.cp(angle, readout, feature.qubits[0] + 1)
        else:
            controlled_phase = PhaseGate(angle).control(2)
            circuit.append(
                controlled_phase,
                [readout, feature.qubits[0] + 1, feature.qubits[1] + 1],
            )
    return circuit


def basis_phase_factor_from_statevector(
    model: SparsePhaseVQC,
    bits: Sequence[int] | str,
) -> complex:
    row = _coerce_bits(bits, model.num_qubits)
    circuit = build_phase_circuit(model, row, include_intercept=True)
    statevector = Statevector.from_instruction(circuit)
    basis_index = sum(int(value) << index for index, value in enumerate(row))
    amplitude = complex(statevector.data[basis_index])
    magnitude = abs(amplitude)
    if magnitude == 0.0:
        raise RuntimeError("basis phase circuit returned zero amplitude")
    return amplitude / magnitude


def hadamard_phase_expectation(
    model: SparsePhaseVQC,
    bits: Sequence[int] | str,
) -> complex:
    """Return <X> + i<Y> of the readout ancilla from a Statevector."""

    row = _coerce_bits(bits, model.num_qubits)
    circuit = build_hadamard_readout_circuit(model, row)
    statevector = Statevector.from_instruction(circuit)
    base_index = sum(int(value) << (index + 1) for index, value in enumerate(row))
    amplitude_zero = complex(statevector.data[base_index])
    amplitude_one = complex(statevector.data[base_index | 1])
    return complex(2.0 * np.conjugate(amplitude_zero) * amplitude_one)


def _append_phase_terms(
    circuit: QuantumCircuit,
    model: SparsePhaseVQC,
    *,
    qubit_offset: int,
) -> None:
    for feature, weight in zip(model.features, model.phase_weights):
        angle = 2.0 * np.pi * float(weight)
        if feature.kind == "linear":
            circuit.p(angle, feature.qubits[0] + qubit_offset)
        else:
            circuit.cp(
                angle,
                feature.qubits[0] + qubit_offset,
                feature.qubits[1] + qubit_offset,
            )


def _coerce_bits(bits: Sequence[int] | str, num_qubits: int) -> tuple[int, ...]:
    if isinstance(bits, str):
        row = tuple(int(value) for value in bits)
    else:
        row = tuple(int(value) for value in bits)
    if len(row) != int(num_qubits):
        raise ValueError("bit vector 长度与模型 qubit 数不一致")
    if any(value not in (0, 1) for value in row):
        raise ValueError("bit vector 必须是二进制")
    return row
