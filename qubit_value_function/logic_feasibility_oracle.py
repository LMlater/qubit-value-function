from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IntegerComparator
from qiskit.quantum_info import Statevector

from .coherent_phase_value import QuantizedSparseValueModel, build_phase_to_value_circuit
from .uc_loader import UCInstance


@dataclass(frozen=True, order=True)
class BooleanLiteral:
    """One required value of a search-register qubit inside a forbidden pattern."""

    qubit: int
    value: int

    def __post_init__(self) -> None:
        qubit = int(self.qubit)
        value = int(self.value)
        if qubit < 0:
            raise ValueError("literal qubit 不能为负数")
        if value not in (0, 1):
            raise ValueError("literal value 必须为 0 或 1")
        object.__setattr__(self, "qubit", qubit)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class ForbiddenBooleanPattern:
    """A local Boolean assignment that violates one exact UC logic rule."""

    literals: tuple[BooleanLiteral, ...]
    label: str

    def __post_init__(self) -> None:
        literals = tuple(sorted(self.literals))
        if not literals:
            raise ValueError("forbidden pattern 至少需要一个 variable literal")
        qubits = [literal.qubit for literal in literals]
        if len(qubits) != len(set(qubits)):
            raise ValueError("forbidden pattern 不能重复使用同一 qubit")
        object.__setattr__(self, "literals", literals)
        object.__setattr__(self, "label", str(self.label))

    def matches(self, bits: Sequence[int]) -> bool:
        return all(int(bits[literal.qubit]) == literal.value for literal in self.literals)


@dataclass(frozen=True)
class LogicFeasibilitySpec:
    """Selected-subregister compilation of exact Boolean UC logic constraints.

    The specification represents infeasibility as a disjunction of local forbidden
    patterns.  It is therefore O(number of UC logic clauses), not a table with one
    entry per commitment state.
    """

    num_generators: int
    num_periods: int
    selected_generator_indices: tuple[int, ...]
    patterns: tuple[ForbiddenBooleanPattern, ...]
    always_infeasible: bool = False
    source_constraint_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        num_generators = int(self.num_generators)
        num_periods = int(self.num_periods)
        selected = tuple(int(index) for index in self.selected_generator_indices)
        patterns = tuple(self.patterns)
        if num_generators <= 0 or num_periods <= 0:
            raise ValueError("num_generators 和 num_periods 必须为正数")
        if not selected:
            raise ValueError("至少需要一个 selected generator")
        if len(selected) != len(set(selected)):
            raise ValueError("selected_generator_indices 不能重复")
        if any(index < 0 or index >= num_generators for index in selected):
            raise ValueError("selected generator index 超出范围")
        if any(
            literal.qubit >= len(selected) * num_periods
            for pattern in patterns
            for literal in pattern.literals
        ):
            raise ValueError("forbidden pattern qubit 超出 selected search register")
        object.__setattr__(self, "num_generators", num_generators)
        object.__setattr__(self, "num_periods", num_periods)
        object.__setattr__(self, "selected_generator_indices", selected)
        object.__setattr__(self, "patterns", patterns)
        object.__setattr__(self, "always_infeasible", bool(self.always_infeasible))
        object.__setattr__(
            self,
            "source_constraint_counts",
            tuple((str(name), int(count)) for name, count in self.source_constraint_counts),
        )

    @property
    def num_x_qubits(self) -> int:
        return int(len(self.selected_generator_indices) * self.num_periods)

    @property
    def num_violation_qubits(self) -> int:
        return int(len(self.patterns))

    def is_feasible(self, bits: Sequence[int] | str) -> bool:
        row = _coerce_bits(bits, self.num_x_qubits)
        if self.always_infeasible:
            return False
        return not any(pattern.matches(row) for pattern in self.patterns)

    def as_dict(self) -> dict[str, object]:
        return {
            "num_generators": int(self.num_generators),
            "num_periods": int(self.num_periods),
            "selected_generator_indices": [
                int(index) for index in self.selected_generator_indices
            ],
            "num_x_qubits": int(self.num_x_qubits),
            "num_forbidden_patterns": int(len(self.patterns)),
            "always_infeasible": bool(self.always_infeasible),
            "source_constraint_counts": {
                name: int(count) for name, count in self.source_constraint_counts
            },
            "patterns": [
                {
                    "label": pattern.label,
                    "literals": [
                        {"qubit": int(literal.qubit), "value": int(literal.value)}
                        for literal in pattern.literals
                    ],
                }
                for pattern in self.patterns
            ],
            "uses_full_state_lookup": False,
            "encoded_constraints": [
                "must_run",
                "initial_remaining_minimum_uptime",
                "initial_remaining_minimum_downtime",
                "minimum_uptime_after_startup",
                "minimum_downtime_after_shutdown",
            ],
        }


@dataclass(frozen=True)
class LogicFeasibilityOracleProbe:
    phase_signs: np.ndarray
    feasible_mask: np.ndarray
    auxiliary_zero_probability: float
    max_phase_error: float


@dataclass(frozen=True)
class JointFeasibleBetterOracleProbe:
    phase_signs: np.ndarray
    feasible_mask: np.ndarray
    better_mask: np.ndarray
    marked_mask: np.ndarray
    auxiliary_zero_probability: float
    max_phase_error: float


def compile_logic_feasibility_spec(
    instance: UCInstance,
    *,
    selected_generator_indices: Sequence[int],
    base_commitment: np.ndarray,
) -> LogicFeasibilitySpec:
    """Compile ``is_logic_feasible`` semantics into local forbidden patterns.

    Variables belonging to unselected generators are substituted from
    ``base_commitment``.  A fully constant violated rule sets
    ``always_infeasible=True``; impossible constant patterns are discarded.
    """

    selected = tuple(int(index) for index in selected_generator_indices)
    horizon = int(instance.time_horizon)
    base = np.asarray(base_commitment, dtype=int)
    if base.shape != (len(instance.generators), horizon):
        raise ValueError("base_commitment shape 与 UC instance 不一致")
    if np.any((base != 0) & (base != 1)):
        raise ValueError("base_commitment 必须是二进制")
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("selected_generator_indices 必须非空且不能重复")
    if any(index < 0 or index >= len(instance.generators) for index in selected):
        raise ValueError("selected generator index 超出范围")

    local_by_global = {generator: local for local, generator in enumerate(selected)}
    pattern_rows: list[ForbiddenBooleanPattern] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    always_infeasible = False
    counts: dict[str, int] = {}

    def add_pattern(
        literals: Iterable[tuple[int, int, int]],
        *,
        family: str,
        label: str,
    ) -> None:
        nonlocal always_infeasible
        required_by_qubit: dict[int, int] = {}
        for generator, period, required_value in literals:
            generator = int(generator)
            period = int(period)
            required_value = int(required_value)
            if required_value not in (0, 1):
                raise ValueError("constraint literal 必须要求 0 或 1")
            if generator in local_by_global:
                qubit = local_by_global[generator] * horizon + period
                previous = required_by_qubit.get(qubit)
                if previous is not None and previous != required_value:
                    return
                required_by_qubit[qubit] = required_value
            elif int(base[generator, period]) != required_value:
                return

        counts[family] = counts.get(family, 0) + 1
        if not required_by_qubit:
            always_infeasible = True
            return
        key = tuple(sorted(required_by_qubit.items()))
        if key in seen:
            return
        seen.add(key)
        pattern_rows.append(
            ForbiddenBooleanPattern(
                literals=tuple(
                    BooleanLiteral(qubit=qubit, value=value) for qubit, value in key
                ),
                label=label,
            )
        )

    for generator_index, generator in enumerate(instance.generators):
        if generator.must_run:
            for period in range(horizon):
                add_pattern(
                    ((generator_index, period, 0),),
                    family="must_run",
                    label=f"must_run_g{generator_index}_t{period}",
                )

        if generator.initial_status < 0:
            remaining_down = max(generator.min_downtime + generator.initial_status, 0)
            for period in range(min(remaining_down, horizon)):
                add_pattern(
                    ((generator_index, period, 1),),
                    family="initial_remaining_minimum_downtime",
                    label=f"initial_down_g{generator_index}_t{period}",
                )
        elif generator.initial_status > 0:
            remaining_up = max(generator.min_uptime - generator.initial_status, 0)
            for period in range(min(remaining_up, horizon)):
                add_pattern(
                    ((generator_index, period, 0),),
                    family="initial_remaining_minimum_uptime",
                    label=f"initial_up_g{generator_index}_t{period}",
                )

        previous_initial = 1 if generator.initial_status > 0 else 0
        for period in range(horizon):
            previous_literal = (
                None if period == 0 else (generator_index, period - 1)
            )
            up_end = min(horizon, period + int(generator.min_uptime))
            for future in range(period + 1, up_end):
                literals: list[tuple[int, int, int]] = []
                if previous_literal is None:
                    if previous_initial != 0:
                        continue
                else:
                    literals.append((*previous_literal, 0))
                literals.extend(
                    [
                        (generator_index, period, 1),
                        (generator_index, future, 0),
                    ]
                )
                add_pattern(
                    literals,
                    family="minimum_uptime_after_startup",
                    label=f"min_up_g{generator_index}_start{period}_future{future}",
                )

            down_end = min(horizon, period + int(generator.min_downtime))
            for future in range(period + 1, down_end):
                literals = []
                if previous_literal is None:
                    if previous_initial != 1:
                        continue
                else:
                    literals.append((*previous_literal, 1))
                literals.extend(
                    [
                        (generator_index, period, 0),
                        (generator_index, future, 1),
                    ]
                )
                add_pattern(
                    literals,
                    family="minimum_downtime_after_shutdown",
                    label=f"min_down_g{generator_index}_stop{period}_future{future}",
                )

    return LogicFeasibilitySpec(
        num_generators=len(instance.generators),
        num_periods=horizon,
        selected_generator_indices=selected,
        patterns=tuple(pattern_rows),
        always_infeasible=always_infeasible,
        source_constraint_counts=tuple(sorted(counts.items())),
    )


def build_logic_feasibility_phase_oracle(
    spec: LogicFeasibilitySpec,
) -> QuantumCircuit:
    """Build a hard phase oracle that marks exactly the compiled feasible states."""

    x_register = QuantumRegister(spec.num_x_qubits, "x")
    violation_register = (
        QuantumRegister(spec.num_violation_qubits, "logic_violation")
        if spec.num_violation_qubits > 0
        else None
    )
    registers = [x_register]
    if violation_register is not None:
        registers.append(violation_register)
    circuit = QuantumCircuit(*registers, name="uc_logic_feasibility_oracle")
    violation_qubits = list(violation_register) if violation_register is not None else []

    append_logic_violation_compute(circuit, spec, list(x_register), violation_qubits)
    if not spec.always_infeasible:
        if not violation_qubits:
            circuit.global_phase += np.pi
        else:
            circuit.x(violation_qubits)
            _append_all_ones_phase(circuit, violation_qubits)
            circuit.x(violation_qubits)
    append_inverse_logic_violation_compute(circuit, spec, list(x_register), violation_qubits)
    return circuit


def build_joint_feasible_better_phase_oracle(
    model: QuantizedSparseValueModel,
    *,
    encoded_threshold: int,
    feasibility_spec: LogicFeasibilitySpec,
) -> QuantumCircuit:
    """Mark ``logic feasible AND sparse integer cost < threshold`` coherently."""

    _validate_model_and_spec(model, feasibility_spec)
    compare_value = model.shifted_compare_value(int(encoded_threshold), strict=True)
    comparator = IntegerComparator(model.num_value_qubits, compare_value, geq=False)
    comparator_ancilla_count = comparator.num_qubits - comparator.num_state_qubits - 1

    x_register = QuantumRegister(model.num_x_qubits, "x")
    value_register = QuantumRegister(model.num_value_qubits, "value")
    better_register = QuantumRegister(1, "better")
    comparator_register = (
        QuantumRegister(comparator_ancilla_count, "cmp")
        if comparator_ancilla_count > 0
        else None
    )
    violation_register = (
        QuantumRegister(feasibility_spec.num_violation_qubits, "logic_violation")
        if feasibility_spec.num_violation_qubits > 0
        else None
    )
    registers = [x_register, value_register, better_register]
    if comparator_register is not None:
        registers.append(comparator_register)
    if violation_register is not None:
        registers.append(violation_register)
    circuit = QuantumCircuit(*registers, name="feasible_sparse_vqc_threshold_oracle")

    if feasibility_spec.always_infeasible:
        return circuit

    x_qubits = list(x_register)
    value_qubits = list(value_register)
    cmp_qubits = list(comparator_register) if comparator_register is not None else []
    violation_qubits = list(violation_register) if violation_register is not None else []
    value_gate = build_phase_to_value_circuit(model).to_gate(label="phase_to_value")
    comparator_gate = comparator.to_gate(label="better_than_threshold")

    circuit.append(value_gate, x_qubits + value_qubits)
    circuit.append(
        comparator_gate,
        value_qubits + [better_register[0]] + cmp_qubits,
    )
    append_logic_violation_compute(circuit, feasibility_spec, x_qubits, violation_qubits)
    if violation_qubits:
        circuit.x(violation_qubits)
    _append_all_ones_phase(circuit, [better_register[0]] + violation_qubits)
    if violation_qubits:
        circuit.x(violation_qubits)
    append_inverse_logic_violation_compute(
        circuit, feasibility_spec, x_qubits, violation_qubits
    )
    circuit.append(
        comparator_gate.inverse(),
        value_qubits + [better_register[0]] + cmp_qubits,
    )
    circuit.append(value_gate.inverse(), x_qubits + value_qubits)
    return circuit


def append_logic_violation_compute(
    circuit: QuantumCircuit,
    spec: LogicFeasibilitySpec,
    x_qubits: Sequence[Any],
    violation_qubits: Sequence[Any],
) -> None:
    """Compute one violation ancilla for every local forbidden Boolean pattern."""

    if len(x_qubits) != spec.num_x_qubits:
        raise ValueError("x_qubits 数量与 feasibility spec 不一致")
    if len(violation_qubits) != spec.num_violation_qubits:
        raise ValueError("violation_qubits 数量与 feasibility spec 不一致")
    for pattern, target in zip(spec.patterns, violation_qubits):
        _append_pattern_toggle(circuit, pattern, x_qubits, target)


def append_inverse_logic_violation_compute(
    circuit: QuantumCircuit,
    spec: LogicFeasibilitySpec,
    x_qubits: Sequence[Any],
    violation_qubits: Sequence[Any],
) -> None:
    """Uncompute every violation ancilla back to zero."""

    if len(x_qubits) != spec.num_x_qubits:
        raise ValueError("x_qubits 数量与 feasibility spec 不一致")
    if len(violation_qubits) != spec.num_violation_qubits:
        raise ValueError("violation_qubits 数量与 feasibility spec 不一致")
    for pattern, target in reversed(list(zip(spec.patterns, violation_qubits))):
        _append_pattern_toggle(circuit, pattern, x_qubits, target)


def simulate_logic_feasibility_phase_oracle(
    spec: LogicFeasibilitySpec,
) -> LogicFeasibilityOracleProbe:
    circuit = QuantumCircuit(build_logic_feasibility_phase_oracle(spec).num_qubits)
    circuit.h(list(circuit.qubits[: spec.num_x_qubits]))
    circuit.compose(build_logic_feasibility_phase_oracle(spec), inplace=True)
    statevector = Statevector.from_instruction(circuit)
    dimension = 2**spec.num_x_qubits
    initial_amplitude = 1.0 / np.sqrt(dimension)
    phase_signs = np.array(
        [float(np.real(statevector.data[index] / initial_amplitude)) for index in range(dimension)],
        dtype=float,
    )
    feasible = np.array(
        [spec.is_feasible(_bits_from_index(index, spec.num_x_qubits)) for index in range(dimension)],
        dtype=bool,
    )
    expected = np.where(feasible, -1.0, 1.0)
    probabilities = statevector.probabilities()
    auxiliary_zero = float(np.sum(probabilities[:dimension]))
    return LogicFeasibilityOracleProbe(
        phase_signs=phase_signs,
        feasible_mask=feasible,
        auxiliary_zero_probability=auxiliary_zero,
        max_phase_error=float(np.max(np.abs(phase_signs - expected))),
    )


def simulate_joint_feasible_better_phase_oracle(
    model: QuantizedSparseValueModel,
    *,
    encoded_threshold: int,
    feasibility_spec: LogicFeasibilitySpec,
) -> JointFeasibleBetterOracleProbe:
    oracle = build_joint_feasible_better_phase_oracle(
        model,
        encoded_threshold=encoded_threshold,
        feasibility_spec=feasibility_spec,
    )
    circuit = QuantumCircuit(oracle.num_qubits)
    circuit.h(list(circuit.qubits[: model.num_x_qubits]))
    circuit.compose(oracle, inplace=True)
    statevector = Statevector.from_instruction(circuit)
    dimension = 2**model.num_x_qubits
    initial_amplitude = 1.0 / np.sqrt(dimension)
    phase_signs = np.array(
        [float(np.real(statevector.data[index] / initial_amplitude)) for index in range(dimension)],
        dtype=float,
    )
    feasible = np.array(
        [
            feasibility_spec.is_feasible(_bits_from_index(index, model.num_x_qubits))
            for index in range(dimension)
        ],
        dtype=bool,
    )
    better = np.array(
        [
            model.is_marked(
                _bits_from_index(index, model.num_x_qubits),
                int(encoded_threshold),
                strict=True,
            )
            for index in range(dimension)
        ],
        dtype=bool,
    )
    marked = feasible & better
    expected = np.where(marked, -1.0, 1.0)
    auxiliary_zero = float(np.sum(statevector.probabilities()[:dimension]))
    return JointFeasibleBetterOracleProbe(
        phase_signs=phase_signs,
        feasible_mask=feasible,
        better_mask=better,
        marked_mask=marked,
        auxiliary_zero_probability=auxiliary_zero,
        max_phase_error=float(np.max(np.abs(phase_signs - expected))),
    )


def _append_pattern_toggle(
    circuit: QuantumCircuit,
    pattern: ForbiddenBooleanPattern,
    x_qubits: Sequence[Any],
    target: Any,
) -> None:
    zero_controls = [x_qubits[literal.qubit] for literal in pattern.literals if literal.value == 0]
    controls = [x_qubits[literal.qubit] for literal in pattern.literals]
    if zero_controls:
        circuit.x(zero_controls)
    if len(controls) == 1:
        circuit.cx(controls[0], target)
    else:
        circuit.mcx(controls, target)
    if zero_controls:
        circuit.x(zero_controls)


def _append_all_ones_phase(circuit: QuantumCircuit, qubits: Sequence[Any]) -> None:
    rows = list(qubits)
    if not rows:
        circuit.global_phase += np.pi
    elif len(rows) == 1:
        circuit.z(rows[0])
    else:
        target = rows[-1]
        circuit.h(target)
        circuit.mcx(rows[:-1], target)
        circuit.h(target)


def _validate_model_and_spec(
    model: QuantizedSparseValueModel,
    spec: LogicFeasibilitySpec,
) -> None:
    if model.num_x_qubits != spec.num_x_qubits:
        raise ValueError("value model 与 feasibility spec 的 search register 不一致")
    if model.num_generators != len(spec.selected_generator_indices):
        raise ValueError("value model generator 数必须等于 selected generator 数")
    if model.num_periods != spec.num_periods:
        raise ValueError("value model period 数与 feasibility spec 不一致")


def _coerce_bits(bits: Sequence[int] | str, expected: int) -> tuple[int, ...]:
    if isinstance(bits, str):
        row = tuple(int(value) for value in bits)
    else:
        row = tuple(int(value) for value in bits)
    if len(row) != int(expected) or any(value not in (0, 1) for value in row):
        raise ValueError("bits 必须是长度正确的二进制序列")
    return row


def _bits_from_index(index: int, num_qubits: int) -> tuple[int, ...]:
    return tuple((int(index) >> qubit) & 1 for qubit in range(int(num_qubits)))
