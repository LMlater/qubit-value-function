from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IntegerComparator, PhaseGate, QFTGate
from qiskit.quantum_info import Statevector

from .fixed_point_oracle import FixedPointConfig
from .sparse_phase_vqc import PhaseFeature, SparsePhaseVQC


@dataclass(frozen=True)
class QuantizedSparseValueModel:
    """Fixed-point sparse value model derived from a trained diagonal phase VQC."""

    num_generators: int
    num_periods: int
    features: tuple[PhaseFeature, ...]
    fixed_point_config: FixedPointConfig
    real_intercept: float
    real_weights: tuple[float, ...]
    integer_intercept: int
    integer_weights: tuple[int, ...]
    coefficient_quantization_errors: tuple[float, ...]
    lower_bound: int
    upper_bound: int
    value_shift: int
    shifted_upper_bound: int
    num_value_qubits: int
    source_phase_model_metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        num_generators = int(self.num_generators)
        num_periods = int(self.num_periods)
        features = tuple(self.features)
        real_weights = tuple(float(value) for value in self.real_weights)
        integer_weights = tuple(int(value) for value in self.integer_weights)
        errors = tuple(float(value) for value in self.coefficient_quantization_errors)
        if num_generators <= 0 or num_periods <= 0:
            raise ValueError("num_generators 和 num_periods 必须为正数")
        if not (len(features) == len(real_weights) == len(integer_weights)):
            raise ValueError("features、real_weights 和 integer_weights 长度必须一致")
        if len(errors) != 1 + len(integer_weights):
            raise ValueError("量化误差必须包含截距和全部权重")
        if not np.isfinite(self.real_intercept) or any(
            not np.isfinite(value) for value in real_weights + errors
        ):
            raise ValueError("真实系数和量化误差必须全部有限")
        if any(
            index < 0 or index >= num_generators * num_periods
            for feature in features
            for index in feature.qubits
        ):
            raise ValueError("feature qubit 索引超出模型范围")

        expected_lower, expected_upper = conservative_integer_bounds(
            int(self.integer_intercept), integer_weights
        )
        if int(self.lower_bound) != expected_lower or int(self.upper_bound) != expected_upper:
            raise ValueError("lower_bound 或 upper_bound 与稀疏整数系数不一致")
        if int(self.value_shift) != -expected_lower:
            raise ValueError("value_shift 必须等于 -lower_bound")
        if int(self.shifted_upper_bound) != expected_upper + int(self.value_shift):
            raise ValueError("shifted_upper_bound 与 bounds/value_shift 不一致")
        if int(self.shifted_upper_bound) < 0:
            raise ValueError("shifted_upper_bound 不能为负数")
        if int(self.num_value_qubits) <= 0:
            raise ValueError("num_value_qubits 必须为正数")
        if int(self.shifted_upper_bound) >= 2 ** int(self.num_value_qubits):
            raise ValueError("值寄存器宽度不足，会发生模溢出")

        object.__setattr__(self, "num_generators", num_generators)
        object.__setattr__(self, "num_periods", num_periods)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "real_weights", real_weights)
        object.__setattr__(self, "integer_weights", integer_weights)
        object.__setattr__(self, "coefficient_quantization_errors", errors)
        object.__setattr__(self, "integer_intercept", int(self.integer_intercept))
        object.__setattr__(self, "lower_bound", int(self.lower_bound))
        object.__setattr__(self, "upper_bound", int(self.upper_bound))
        object.__setattr__(self, "value_shift", int(self.value_shift))
        object.__setattr__(self, "shifted_upper_bound", int(self.shifted_upper_bound))
        object.__setattr__(self, "num_value_qubits", int(self.num_value_qubits))

    @property
    def num_x_qubits(self) -> int:
        return int(self.num_generators * self.num_periods)

    @property
    def phase_modulus(self) -> int:
        return 2 ** int(self.num_value_qubits)

    @property
    def shifted_intercept(self) -> int:
        return int(self.integer_intercept + self.value_shift)

    def feature_values(self, bits: Sequence[int] | str) -> np.ndarray:
        row = _coerce_bits(bits, self.num_x_qubits)
        return np.array([feature.evaluate(row) for feature in self.features], dtype=int)

    def integer_value(self, bits: Sequence[int] | str) -> int:
        return int(
            self.integer_intercept
            + np.dot(np.asarray(self.integer_weights, dtype=int), self.feature_values(bits))
        )

    def shifted_integer_value(self, bits: Sequence[int] | str) -> int:
        value = int(self.integer_value(bits) + self.value_shift)
        if value < 0 or value >= self.phase_modulus:
            raise ValueError("当前输入的 shifted integer value 超出值寄存器范围")
        return value

    def decoded_value(self, bits: Sequence[int] | str) -> float:
        return self.fixed_point_config.decode(self.integer_value(bits))

    def shifted_compare_value(self, encoded_threshold: int, *, strict: bool = True) -> int:
        compare_value = int(encoded_threshold) + int(self.value_shift)
        if not strict:
            compare_value += 1
        return min(max(compare_value, 0), self.phase_modulus)

    def is_marked(
        self,
        bits: Sequence[int] | str,
        encoded_threshold: int,
        *,
        strict: bool = True,
    ) -> bool:
        value = self.integer_value(bits)
        if strict:
            return bool(value < int(encoded_threshold))
        return bool(value <= int(encoded_threshold))

    def as_dict(self) -> dict[str, object]:
        return {
            "num_generators": int(self.num_generators),
            "num_periods": int(self.num_periods),
            "num_x_qubits": int(self.num_x_qubits),
            "num_value_qubits": int(self.num_value_qubits),
            "fixed_point": {
                "fractional_bits": int(self.fixed_point_config.fractional_bits),
                "scale": int(self.fixed_point_config.scale),
                "cost_unit": float(self.fixed_point_config.unit),
                "quantum": float(self.fixed_point_config.quantum),
                "rounding": self.fixed_point_config.rounding,
            },
            "real_intercept": float(self.real_intercept),
            "real_weights": [float(value) for value in self.real_weights],
            "integer_intercept": int(self.integer_intercept),
            "integer_weights": [int(value) for value in self.integer_weights],
            "coefficient_quantization_errors": [
                float(value) for value in self.coefficient_quantization_errors
            ],
            "lower_bound": int(self.lower_bound),
            "upper_bound": int(self.upper_bound),
            "value_shift": int(self.value_shift),
            "shifted_upper_bound": int(self.shifted_upper_bound),
            "phase_modulus": int(self.phase_modulus),
            "features": [
                {
                    "kind": feature.kind,
                    "qubits": [int(index) for index in feature.qubits],
                    "label": feature.label,
                    "real_weight": float(real_weight),
                    "integer_weight": int(integer_weight),
                }
                for feature, real_weight, integer_weight in zip(
                    self.features, self.real_weights, self.integer_weights
                )
            ],
            "source_phase_model_metadata": self.source_phase_model_metadata,
        }


@dataclass(frozen=True)
class BasisValueCodeProbe:
    expected_code: int
    most_likely_code: int
    correct_code_probability: float
    value_probabilities: np.ndarray


@dataclass(frozen=True)
class PhaseToValueSuperpositionProbe:
    pairing_probability: float
    inverse_auxiliary_zero_probability: float


@dataclass(frozen=True)
class SparseThresholdOracleProbe:
    phase_signs: np.ndarray
    marked_mask: np.ndarray
    auxiliary_zero_probability: float
    max_phase_error: float


def conservative_integer_bounds(
    integer_intercept: int,
    integer_weights: Sequence[int],
) -> tuple[int, int]:
    """Return safe Boolean-feature bounds without enumerating input states."""

    weights = tuple(int(value) for value in integer_weights)
    lower = int(integer_intercept) + sum(min(0, value) for value in weights)
    upper = int(integer_intercept) + sum(max(0, value) for value in weights)
    return int(lower), int(upper)


def quantize_sparse_phase_model(
    phase_model: SparsePhaseVQC,
    config: FixedPointConfig,
) -> QuantizedSparseValueModel:
    """Convert a trained phase model to a signed fixed-point sparse value model."""

    scale_factor = float(phase_model.cost_scale) / float(phase_model.phase_scale)
    real_intercept = float(
        phase_model.cost_center
        + scale_factor * (phase_model.phase_intercept - phase_model.phase_center)
    )
    real_weights = tuple(scale_factor * float(value) for value in phase_model.phase_weights)
    integer_intercept = int(config.encode(real_intercept))
    integer_weights = tuple(int(config.encode(value)) for value in real_weights)
    lower_bound, upper_bound = conservative_integer_bounds(
        integer_intercept, integer_weights
    )
    value_shift = int(-lower_bound)
    shifted_upper_bound = int(upper_bound + value_shift)
    num_value_qubits = max(1, int(shifted_upper_bound).bit_length())
    coefficient_errors = (
        float(config.decode(integer_intercept) - real_intercept),
        *(
            float(config.decode(value) - real)
            for value, real in zip(integer_weights, real_weights)
        ),
    )
    model = QuantizedSparseValueModel(
        num_generators=phase_model.num_generators,
        num_periods=phase_model.num_periods,
        features=phase_model.features,
        fixed_point_config=config,
        real_intercept=real_intercept,
        real_weights=real_weights,
        integer_intercept=integer_intercept,
        integer_weights=integer_weights,
        coefficient_quantization_errors=tuple(coefficient_errors),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        value_shift=value_shift,
        shifted_upper_bound=shifted_upper_bound,
        num_value_qubits=num_value_qubits,
        source_phase_model_metadata=phase_model.as_dict(),
    )
    if model.lower_bound + model.value_shift < 0:
        raise ValueError("value_shift 未能消除负的保守下界")
    if model.shifted_upper_bound >= model.phase_modulus:
        raise ValueError("值寄存器宽度不足，无法避免 phase wrap")
    return model


def build_integer_value_phase_circuit(model: QuantizedSparseValueModel) -> QuantumCircuit:
    """Build U|x> = exp(2πi shifted_value(x)/2^m)|x>."""

    circuit = QuantumCircuit(model.num_x_qubits, name="integer_sparse_value_phase")
    angle_unit = 2.0 * np.pi / float(model.phase_modulus)
    circuit.global_phase += angle_unit * float(model.shifted_intercept)
    _append_uncontrolled_integer_phase_terms(
        circuit,
        model,
        angle_multiplier=angle_unit,
        x_qubits=list(circuit.qubits),
    )
    return circuit


def build_phase_to_value_circuit(model: QuantizedSparseValueModel) -> QuantumCircuit:
    """Coherently map |x>|0> to |x>|shifted_integer_value(x)>."""

    x_register = QuantumRegister(model.num_x_qubits, "x")
    value_register = QuantumRegister(model.num_value_qubits, "value")
    circuit = QuantumCircuit(x_register, value_register, name="phase_to_value")
    _append_phase_to_value_body(
        circuit,
        model,
        x_qubits=list(x_register),
        value_qubits=list(value_register),
    )
    return circuit


def append_phase_to_value(
    circuit: QuantumCircuit,
    model: QuantizedSparseValueModel,
    x_qubits: Sequence[Any],
    value_qubits: Sequence[Any],
) -> None:
    gate = build_phase_to_value_circuit(model).to_gate(label="phase_to_value")
    circuit.append(gate, list(x_qubits) + list(value_qubits))


def append_inverse_phase_to_value(
    circuit: QuantumCircuit,
    model: QuantizedSparseValueModel,
    x_qubits: Sequence[Any],
    value_qubits: Sequence[Any],
) -> None:
    gate = build_phase_to_value_circuit(model).to_gate(label="phase_to_value")
    circuit.append(gate.inverse(), list(x_qubits) + list(value_qubits))


def build_sparse_vqc_threshold_phase_oracle(
    model: QuantizedSparseValueModel,
    *,
    encoded_real_threshold: int,
    strict: bool = True,
) -> QuantumCircuit:
    """Build phase-to-value, integer comparison, phase marking, and uncompute."""

    compare_value = model.shifted_compare_value(encoded_real_threshold, strict=strict)
    comparator = IntegerComparator(model.num_value_qubits, compare_value, geq=False)
    comparator_ancilla_count = comparator.num_qubits - comparator.num_state_qubits - 1

    x_register = QuantumRegister(model.num_x_qubits, "x")
    value_register = QuantumRegister(model.num_value_qubits, "value")
    flag_register = QuantumRegister(1, "flag")
    comparator_register = (
        QuantumRegister(comparator_ancilla_count, "cmp")
        if comparator_ancilla_count > 0
        else None
    )
    registers = [x_register, value_register, flag_register]
    if comparator_register is not None:
        registers.append(comparator_register)
    circuit = QuantumCircuit(*registers, name="sparse_vqc_threshold_oracle")

    x_qubits = list(x_register)
    value_qubits = list(value_register)
    comparator_ancillas = (
        list(comparator_register) if comparator_register is not None else []
    )
    compute_gate = build_phase_to_value_circuit(model).to_gate(label="phase_to_value")
    comparator_gate = comparator.to_gate()
    circuit.append(compute_gate, x_qubits + value_qubits)
    circuit.append(
        comparator_gate,
        value_qubits + [flag_register[0]] + comparator_ancillas,
    )
    circuit.z(flag_register[0])
    circuit.append(
        comparator_gate.inverse(),
        value_qubits + [flag_register[0]] + comparator_ancillas,
    )
    circuit.append(compute_gate.inverse(), x_qubits + value_qubits)
    return circuit


def basis_value_code_probe(
    model: QuantizedSparseValueModel,
    bits: Sequence[int] | str,
) -> BasisValueCodeProbe:
    row = _coerce_bits(bits, model.num_x_qubits)
    compute_gate = build_phase_to_value_circuit(model).to_gate(label="phase_to_value")
    circuit = QuantumCircuit(model.num_x_qubits + model.num_value_qubits)
    for index, bit in enumerate(row):
        if bit:
            circuit.x(index)
    circuit.append(compute_gate, list(circuit.qubits))
    probabilities = Statevector.from_instruction(circuit).probabilities()
    value_probabilities = _value_marginal_probabilities(
        probabilities,
        num_x_qubits=model.num_x_qubits,
        num_value_qubits=model.num_value_qubits,
    )
    expected = model.shifted_integer_value(row)
    return BasisValueCodeProbe(
        expected_code=int(expected),
        most_likely_code=int(np.argmax(value_probabilities)),
        correct_code_probability=float(value_probabilities[expected]),
        value_probabilities=value_probabilities,
    )


def phase_to_value_superposition_probe(
    model: QuantizedSparseValueModel,
) -> PhaseToValueSuperpositionProbe:
    compute = build_phase_to_value_circuit(model).to_gate(label="phase_to_value")
    circuit = QuantumCircuit(model.num_x_qubits + model.num_value_qubits)
    circuit.h(list(circuit.qubits[: model.num_x_qubits]))
    circuit.append(compute, list(circuit.qubits))
    probabilities = Statevector.from_instruction(circuit).probabilities()
    pairing_probability = 0.0
    for x_index in range(2 ** model.num_x_qubits):
        bits = _bits_from_index(x_index, model.num_x_qubits)
        value = model.shifted_integer_value(bits)
        pairing_probability += float(
            probabilities[x_index | (int(value) << model.num_x_qubits)]
        )

    uncompute = QuantumCircuit(model.num_x_qubits + model.num_value_qubits)
    uncompute.h(list(uncompute.qubits[: model.num_x_qubits]))
    uncompute.append(compute, list(uncompute.qubits))
    uncompute.append(compute.inverse(), list(uncompute.qubits))
    uncompute_probabilities = Statevector.from_instruction(uncompute).probabilities()
    x_dimension = 2 ** model.num_x_qubits
    return PhaseToValueSuperpositionProbe(
        pairing_probability=float(pairing_probability),
        inverse_auxiliary_zero_probability=float(
            np.sum(uncompute_probabilities[:x_dimension])
        ),
    )


def simulate_sparse_vqc_threshold_phase_oracle(
    model: QuantizedSparseValueModel,
    *,
    encoded_real_threshold: int,
    strict: bool = True,
) -> SparseThresholdOracleProbe:
    oracle = build_sparse_vqc_threshold_phase_oracle(
        model,
        encoded_real_threshold=encoded_real_threshold,
        strict=strict,
    )
    circuit = QuantumCircuit(oracle.num_qubits)
    circuit.h(list(circuit.qubits[: model.num_x_qubits]))
    circuit.append(oracle.to_gate(), list(circuit.qubits))
    statevector = Statevector.from_instruction(circuit)
    probabilities = statevector.probabilities()
    dimension = 2 ** model.num_x_qubits
    initial_amplitude = 1.0 / np.sqrt(float(dimension))
    phase_signs = np.array(
        [statevector.data[index] / initial_amplitude for index in range(dimension)],
        dtype=complex,
    )
    marked = np.array(
        [
            model.is_marked(
                _bits_from_index(index, model.num_x_qubits),
                encoded_real_threshold,
                strict=strict,
            )
            for index in range(dimension)
        ],
        dtype=bool,
    )
    expected = np.where(marked, -1.0 + 0.0j, 1.0 + 0.0j)
    return SparseThresholdOracleProbe(
        phase_signs=phase_signs,
        marked_mask=marked,
        auxiliary_zero_probability=float(np.sum(probabilities[:dimension])),
        max_phase_error=float(np.max(np.abs(phase_signs - expected))),
    )


def estimate_statevector_memory_gb(num_qubits: int) -> float:
    if int(num_qubits) < 0:
        raise ValueError("num_qubits 不能为负数")
    return float((2 ** int(num_qubits)) * 16 / (1024**3))


def _append_phase_to_value_body(
    circuit: QuantumCircuit,
    model: QuantizedSparseValueModel,
    *,
    x_qubits: Sequence[Any],
    value_qubits: Sequence[Any],
) -> None:
    if len(x_qubits) != model.num_x_qubits:
        raise ValueError("x_qubits 数量与模型不一致")
    if len(value_qubits) != model.num_value_qubits:
        raise ValueError("value_qubits 数量与模型不一致")

    circuit.h(list(value_qubits))
    modulus = float(model.phase_modulus)
    for power_index, evaluation_qubit in enumerate(value_qubits):
        power = 2 ** int(power_index)
        angle_unit = 2.0 * np.pi * float(power) / modulus
        circuit.p(angle_unit * float(model.shifted_intercept), evaluation_qubit)
        _append_controlled_integer_phase_terms(
            circuit,
            model,
            evaluation_qubit=evaluation_qubit,
            x_qubits=x_qubits,
            angle_multiplier=angle_unit,
        )
    circuit.append(QFTGate(model.num_value_qubits).inverse(), list(value_qubits))


def _append_uncontrolled_integer_phase_terms(
    circuit: QuantumCircuit,
    model: QuantizedSparseValueModel,
    *,
    angle_multiplier: float,
    x_qubits: Sequence[Any],
) -> None:
    for feature, weight in zip(model.features, model.integer_weights):
        angle = float(angle_multiplier) * float(weight)
        if feature.kind == "linear":
            circuit.p(angle, x_qubits[feature.qubits[0]])
        else:
            circuit.cp(
                angle,
                x_qubits[feature.qubits[0]],
                x_qubits[feature.qubits[1]],
            )


def _append_controlled_integer_phase_terms(
    circuit: QuantumCircuit,
    model: QuantizedSparseValueModel,
    *,
    evaluation_qubit: Any,
    x_qubits: Sequence[Any],
    angle_multiplier: float,
) -> None:
    for feature, weight in zip(model.features, model.integer_weights):
        angle = float(angle_multiplier) * float(weight)
        if feature.kind == "linear":
            circuit.cp(angle, evaluation_qubit, x_qubits[feature.qubits[0]])
        else:
            controlled_phase = PhaseGate(angle).control(2)
            circuit.append(
                controlled_phase,
                [
                    evaluation_qubit,
                    x_qubits[feature.qubits[0]],
                    x_qubits[feature.qubits[1]],
                ],
            )


def _value_marginal_probabilities(
    basis_probabilities: np.ndarray,
    *,
    num_x_qubits: int,
    num_value_qubits: int,
) -> np.ndarray:
    probabilities = np.asarray(basis_probabilities, dtype=float)
    expected_size = 2 ** (int(num_x_qubits) + int(num_value_qubits))
    if probabilities.size != expected_size:
        raise ValueError("概率向量尺寸与 x/value 寄存器不一致")
    value_mask = 2 ** int(num_value_qubits) - 1
    result = np.zeros(2 ** int(num_value_qubits), dtype=float)
    for basis_index, probability in enumerate(probabilities):
        value_index = (int(basis_index) >> int(num_x_qubits)) & value_mask
        result[value_index] += float(probability)
    return result


def _coerce_bits(bits: Sequence[int] | str, num_qubits: int) -> tuple[int, ...]:
    row = tuple(int(value) for value in bits)
    if len(row) != int(num_qubits):
        raise ValueError("bit vector 长度与模型 qubit 数不一致")
    if any(value not in (0, 1) for value in row):
        raise ValueError("bit vector 必须是二进制")
    return row


def _bits_from_index(index: int, num_qubits: int) -> tuple[int, ...]:
    return tuple((int(index) >> qubit) & 1 for qubit in range(int(num_qubits)))
