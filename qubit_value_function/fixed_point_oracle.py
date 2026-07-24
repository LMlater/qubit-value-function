from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IntegerComparator, WeightedAdder
from qiskit.quantum_info import Statevector


@dataclass(frozen=True)
class FixedPointConfig:
    """统一的成本定点数编码配置。

    real_value 先除以 ``unit``，再乘以 ``2**fractional_bits`` 并取整。
    例如 unit=1000、fractional_bits=2 时，一个整数码表示 250 美元。
    """

    fractional_bits: int = 2
    unit: float = 1000.0
    rounding: str = "nearest"

    def __post_init__(self) -> None:
        if self.fractional_bits < 0:
            raise ValueError("fractional_bits 不能为负数")
        if not np.isfinite(self.unit) or self.unit <= 0.0:
            raise ValueError("unit 必须为有限正数")
        if self.rounding not in {"nearest", "floor", "ceil"}:
            raise ValueError("rounding 必须为 nearest、floor 或 ceil")

    @property
    def scale(self) -> int:
        return 2**int(self.fractional_bits)

    @property
    def quantum(self) -> float:
        """一个整数码对应的真实成本间隔。"""

        return float(self.unit) / float(self.scale)

    @property
    def max_abs_rounding_error(self) -> float:
        if self.rounding == "nearest":
            return 0.5 * self.quantum
        return self.quantum

    def encode(self, value: float) -> int:
        if not np.isfinite(value):
            raise ValueError("待编码成本必须为有限数")
        scaled = float(value) * float(self.scale) / float(self.unit)
        if self.rounding == "nearest":
            return int(np.rint(scaled))
        if self.rounding == "floor":
            return int(np.floor(scaled))
        return int(np.ceil(scaled))

    def decode(self, encoded: int) -> float:
        return float(int(encoded)) * self.quantum


@dataclass(frozen=True)
class FixedPointAffineSpec:
    """可由 WeightedAdder 计算的定点仿射成本模型。

    原模型为 ``intercept + sum(coefficients[i] * x[i])``。负系数通过
    ``-a*x = -a + a*(1-x)`` 转换为非负权重和反相输入，从而兼容
    Qiskit ``WeightedAdder``。
    """

    config: FixedPointConfig
    real_intercept: float
    real_coefficients: tuple[float, ...]
    encoded_offset: int
    weights: tuple[int, ...]
    inverted_bit_indices: tuple[int, ...]
    bit_labels: tuple[str, ...] = ()
    name: str = "fixed_point_affine_cost"

    def __post_init__(self) -> None:
        if not self.real_coefficients:
            raise ValueError("real_coefficients 至少需要一个系数")
        if len(self.weights) != len(self.real_coefficients):
            raise ValueError("weights 与 real_coefficients 长度必须一致")
        if any(int(weight) < 0 for weight in self.weights):
            raise ValueError("WeightedAdder 的 weights 必须为非负整数")
        inverted = tuple(sorted({int(index) for index in self.inverted_bit_indices}))
        if any(index < 0 or index >= len(self.weights) for index in inverted):
            raise ValueError("inverted_bit_indices 超出输入 bit 范围")
        labels = self.bit_labels or tuple(f"x{index}" for index in range(len(self.weights)))
        if len(labels) != len(self.weights):
            raise ValueError("bit_labels 与输入 bit 数量必须一致")
        object.__setattr__(self, "weights", tuple(int(weight) for weight in self.weights))
        object.__setattr__(self, "inverted_bit_indices", inverted)
        object.__setattr__(self, "bit_labels", tuple(labels))

    @classmethod
    def from_real_coefficients(
        cls,
        *,
        config: FixedPointConfig,
        intercept: float,
        coefficients: list[float] | tuple[float, ...] | np.ndarray,
        bit_labels: tuple[str, ...] = (),
        name: str = "fixed_point_affine_cost",
    ) -> "FixedPointAffineSpec":
        coefficient_array = np.asarray(coefficients, dtype=float)
        if coefficient_array.ndim != 1 or coefficient_array.size == 0:
            raise ValueError("coefficients 必须是一维非空数组")
        if not np.all(np.isfinite(coefficient_array)) or not np.isfinite(intercept):
            raise ValueError("仿射模型系数必须为有限数")

        encoded_intercept = config.encode(float(intercept))
        encoded_coefficients = [config.encode(float(value)) for value in coefficient_array]
        encoded_offset = int(encoded_intercept)
        weights: list[int] = []
        inverted: list[int] = []
        for index, coefficient in enumerate(encoded_coefficients):
            if coefficient < 0:
                encoded_offset += int(coefficient)
                weights.append(int(-coefficient))
                inverted.append(index)
            else:
                weights.append(int(coefficient))

        return cls(
            config=config,
            real_intercept=float(intercept),
            real_coefficients=tuple(float(value) for value in coefficient_array),
            encoded_offset=encoded_offset,
            weights=tuple(weights),
            inverted_bit_indices=tuple(inverted),
            bit_labels=bit_labels,
            name=name,
        )

    @property
    def num_x_qubits(self) -> int:
        return len(self.weights)

    @property
    def max_weighted_sum(self) -> int:
        return int(sum(self.weights))

    def weighted_sum_for_index(self, state_index: int) -> int:
        inverted = set(self.inverted_bit_indices)
        total = 0
        for bit_index, weight in enumerate(self.weights):
            bit = (int(state_index) >> bit_index) & 1
            term = 1 - bit if bit_index in inverted else bit
            total += int(weight) * int(term)
        return int(total)

    def encoded_cost_for_index(self, state_index: int) -> int:
        return int(self.encoded_offset + self.weighted_sum_for_index(state_index))

    def decoded_cost_for_index(self, state_index: int) -> float:
        return self.config.decode(self.encoded_cost_for_index(state_index))

    def direct_encoded_cost_for_index(self, state_index: int) -> int:
        bits = np.array(
            [(int(state_index) >> bit_index) & 1 for bit_index in range(self.num_x_qubits)],
            dtype=float,
        )
        value = self.real_intercept + float(np.dot(np.asarray(self.real_coefficients), bits))
        return self.config.encode(value)

    def shifted_threshold(self, real_threshold: float, *, strict: bool = True) -> int:
        encoded_threshold = self.config.encode(real_threshold)
        if strict:
            encoded_threshold -= 1
        return int(encoded_threshold - self.encoded_offset)

    def marked_mask(self, real_threshold: float, *, strict: bool = True) -> np.ndarray:
        threshold = self.config.encode(real_threshold)
        values = np.array(
            [self.encoded_cost_for_index(index) for index in range(2**self.num_x_qubits)],
            dtype=int,
        )
        return values < threshold if strict else values <= threshold


@dataclass(frozen=True)
class FixedPointOracleProbe:
    phase_signs: np.ndarray
    marked_mask: np.ndarray
    x_probabilities: np.ndarray
    auxiliary_zero_probability: float
    max_phase_error: float


@dataclass(frozen=True)
class FixedPointGroverResult:
    iterations: int
    x_probabilities: np.ndarray
    marked_mask: np.ndarray
    marked_probability: float
    auxiliary_zero_probability: float


@dataclass
class _Workspace:
    circuit: QuantumCircuit
    x_qubits: list[Any]
    value_qubits: list[Any]
    carry_qubits: list[Any]
    control_qubits: list[Any]
    flag_qubit: Any
    comparator_ancillas: list[Any]
    adder: WeightedAdder
    comparator: IntegerComparator


def build_fixed_point_phase_oracle_circuit(
    spec: FixedPointAffineSpec,
    *,
    real_threshold: float,
    strict: bool = True,
) -> QuantumCircuit:
    workspace = _new_workspace(spec, real_threshold=real_threshold, strict=strict)
    _append_phase_oracle(workspace, spec)
    return workspace.circuit


def build_fixed_point_grover_circuit(
    spec: FixedPointAffineSpec,
    *,
    real_threshold: float,
    iterations: int | None = None,
    strict: bool = True,
) -> QuantumCircuit:
    marked = spec.marked_mask(real_threshold, strict=strict)
    marked_count = int(marked.sum())
    if marked_count == 0:
        raise ValueError("当前固定点阈值下没有 marked state，无法构造 Grover 电路")
    if iterations is None:
        iterations = optimal_grover_iterations(2**spec.num_x_qubits, marked_count)
    if int(iterations) < 0:
        raise ValueError("iterations 不能为负数")

    workspace = _new_workspace(spec, real_threshold=real_threshold, strict=strict)
    workspace.circuit.h(workspace.x_qubits)
    for _ in range(int(iterations)):
        _append_phase_oracle(workspace, spec)
        _append_x_register_diffuser(workspace.circuit, workspace.x_qubits)
    return workspace.circuit


def simulate_fixed_point_phase_oracle(
    spec: FixedPointAffineSpec,
    *,
    real_threshold: float,
    strict: bool = True,
) -> FixedPointOracleProbe:
    workspace = _new_workspace(spec, real_threshold=real_threshold, strict=strict)
    workspace.circuit.h(workspace.x_qubits)
    _append_phase_oracle(workspace, spec)
    statevector = Statevector.from_instruction(workspace.circuit)
    probabilities = statevector.probabilities()
    x_dimension = 2**spec.num_x_qubits
    initial_amplitude = 1.0 / np.sqrt(x_dimension)
    phase_signs = np.zeros(x_dimension, dtype=float)
    for state_index in range(x_dimension):
        phase_signs[state_index] = float(np.real(statevector.data[state_index] / initial_amplitude))
    marked = spec.marked_mask(real_threshold, strict=strict)
    expected = np.where(marked, -1.0, 1.0)
    x_probabilities, auxiliary_zero_probability = x_marginal_probabilities(
        probabilities,
        spec.num_x_qubits,
    )
    return FixedPointOracleProbe(
        phase_signs=phase_signs,
        marked_mask=marked,
        x_probabilities=x_probabilities,
        auxiliary_zero_probability=float(auxiliary_zero_probability),
        max_phase_error=float(np.max(np.abs(phase_signs - expected))),
    )


def simulate_fixed_point_grover(
    spec: FixedPointAffineSpec,
    *,
    real_threshold: float,
    iterations: int | None = None,
    strict: bool = True,
) -> FixedPointGroverResult:
    marked = spec.marked_mask(real_threshold, strict=strict)
    marked_count = int(marked.sum())
    if marked_count == 0:
        raise ValueError("当前固定点阈值下没有 marked state，无法执行 Grover 模拟")
    if iterations is None:
        iterations = optimal_grover_iterations(2**spec.num_x_qubits, marked_count)
    circuit = build_fixed_point_grover_circuit(
        spec,
        real_threshold=real_threshold,
        iterations=int(iterations),
        strict=strict,
    )
    statevector = Statevector.from_instruction(circuit)
    x_probabilities, auxiliary_zero_probability = x_marginal_probabilities(
        statevector.probabilities(),
        spec.num_x_qubits,
    )
    return FixedPointGroverResult(
        iterations=int(iterations),
        x_probabilities=x_probabilities,
        marked_mask=marked,
        marked_probability=float(x_probabilities[marked].sum()),
        auxiliary_zero_probability=float(auxiliary_zero_probability),
    )


def fit_affine_cost_model(
    *,
    bitstrings: list[str] | tuple[str, ...],
    costs: list[float] | tuple[float, ...] | np.ndarray,
    ridge: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """使用单一 ridge 最小二乘方法拟合成本仿射模型。"""

    if not bitstrings:
        raise ValueError("bitstrings 不能为空")
    num_bits = len(bitstrings[0])
    if num_bits <= 0:
        raise ValueError("bitstring 至少需要一个 bit")
    if any(len(bitstring) != num_bits for bitstring in bitstrings):
        raise ValueError("所有 bitstring 的长度必须一致")
    if any(set(bitstring) - {"0", "1"} for bitstring in bitstrings):
        raise ValueError("bitstring 只能包含字符 0 和 1")
    values = np.asarray(costs, dtype=float)
    if values.ndim != 1 or values.size != len(bitstrings):
        raise ValueError("costs 与 bitstrings 的样本数量必须一致")
    if not np.all(np.isfinite(values)):
        raise ValueError("costs 必须全部为有限数")
    if ridge < 0.0:
        raise ValueError("ridge 不能为负数")

    features = np.array(
        [[1.0] + [float(bit) for bit in bitstring] for bitstring in bitstrings],
        dtype=float,
    )
    regularizer = np.eye(features.shape[1], dtype=float) * float(ridge)
    regularizer[0, 0] = 0.0
    solution = np.linalg.solve(features.T @ features + regularizer, features.T @ values)
    return float(solution[0]), np.asarray(solution[1:], dtype=float)


def x_marginal_probabilities(
    basis_probabilities: np.ndarray,
    num_x_qubits: int,
) -> tuple[np.ndarray, float]:
    basis_probabilities = np.asarray(basis_probabilities, dtype=float)
    x_dimension = 2**int(num_x_qubits)
    if basis_probabilities.size % x_dimension != 0:
        raise ValueError("概率向量与 num_x_qubits 不兼容")
    x_mask = x_dimension - 1
    x_probabilities = np.zeros(x_dimension, dtype=float)
    auxiliary_zero_probability = 0.0
    for basis_index, probability in enumerate(basis_probabilities):
        x_index = int(basis_index) & x_mask
        x_probabilities[x_index] += float(probability)
        if int(basis_index) >> int(num_x_qubits) == 0:
            auxiliary_zero_probability += float(probability)
    return x_probabilities, float(auxiliary_zero_probability)


def optimal_grover_iterations(dimension: int, marked_count: int) -> int:
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("dimension 必须为 2 的正整数次幂")
    if marked_count <= 0:
        return 0
    if marked_count > dimension:
        raise ValueError("marked_count 不能大于 dimension")
    return max(1, int(np.floor(np.pi / 4.0 * np.sqrt(dimension / marked_count))))


def _new_workspace(
    spec: FixedPointAffineSpec,
    *,
    real_threshold: float,
    strict: bool,
) -> _Workspace:
    adder = WeightedAdder(spec.num_x_qubits, list(spec.weights))
    shifted_threshold = spec.shifted_threshold(real_threshold, strict=strict)
    compare_value = min(max(int(shifted_threshold) + 1, 0), 2**adder.num_sum_qubits)
    comparator = IntegerComparator(adder.num_sum_qubits, compare_value, geq=False)

    x_register = QuantumRegister(spec.num_x_qubits, "x")
    value_register = QuantumRegister(adder.num_sum_qubits, "value")
    carry_register = _optional_register(adder.num_carry_qubits, "carry")
    control_register = _optional_register(adder.num_control_qubits, "control")
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

    return _Workspace(
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


def _append_phase_oracle(workspace: _Workspace, spec: FixedPointAffineSpec) -> None:
    # 负系数对应的输入先取反，使 WeightedAdder 只处理非负整数权重。
    for bit_index in spec.inverted_bit_indices:
        workspace.circuit.x(workspace.x_qubits[bit_index])
    workspace.circuit.append(
        workspace.adder.to_gate(),
        workspace.x_qubits
        + workspace.value_qubits
        + workspace.carry_qubits
        + workspace.control_qubits,
    )
    for bit_index in spec.inverted_bit_indices:
        workspace.circuit.x(workspace.x_qubits[bit_index])

    workspace.circuit.append(
        workspace.comparator.to_gate(),
        workspace.value_qubits + [workspace.flag_qubit] + workspace.comparator_ancillas,
    )
    workspace.circuit.z(workspace.flag_qubit)
    workspace.circuit.append(
        workspace.comparator.to_gate().inverse(),
        workspace.value_qubits + [workspace.flag_qubit] + workspace.comparator_ancillas,
    )

    for bit_index in spec.inverted_bit_indices:
        workspace.circuit.x(workspace.x_qubits[bit_index])
    workspace.circuit.append(
        workspace.adder.to_gate().inverse(),
        workspace.x_qubits
        + workspace.value_qubits
        + workspace.carry_qubits
        + workspace.control_qubits,
    )
    for bit_index in spec.inverted_bit_indices:
        workspace.circuit.x(workspace.x_qubits[bit_index])


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
    if int(size) <= 0:
        return None
    return QuantumRegister(int(size), name)
