from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from .commitment import is_logic_feasible
from .experiment_utils import time_window_instance
from .logic_feasibility_oracle import (
    LogicFeasibilitySpec,
    compile_logic_feasibility_spec,
)
from .uc_loader import UCInstance


@dataclass(frozen=True)
class LogicSubspaceScanRow:
    window_start: int
    horizon: int
    selected_generator_indices: tuple[int, ...]
    selected_generator_names: tuple[str, ...]
    num_x_qubits: int
    always_infeasible: bool
    num_forbidden_patterns: int
    source_constraint_counts: tuple[tuple[str, int], ...]
    retained_pattern_counts: tuple[tuple[str, int], ...]
    logic_feasible_indices: tuple[int, ...]
    logic_infeasible_indices: tuple[int, ...]
    base_commitment: tuple[tuple[int, ...], ...]
    classical_spec_agree: bool

    @property
    def dimension(self) -> int:
        return 2 ** int(self.num_x_qubits)

    @property
    def logic_feasible_count(self) -> int:
        return int(len(self.logic_feasible_indices))

    @property
    def logic_infeasible_count(self) -> int:
        return int(len(self.logic_infeasible_indices))

    @property
    def feasible_ratio(self) -> float:
        return float(self.logic_feasible_count / self.dimension)

    @property
    def has_active_filter(self) -> bool:
        return bool(
            not self.always_infeasible
            and self.num_forbidden_patterns > 0
            and self.logic_feasible_count > 0
            and self.logic_infeasible_count > 0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "window_start": int(self.window_start),
            "horizon": int(self.horizon),
            "selected_generator_indices": [
                int(index) for index in self.selected_generator_indices
            ],
            "selected_generator_names": list(self.selected_generator_names),
            "num_x_qubits": int(self.num_x_qubits),
            "dimension": int(self.dimension),
            "always_infeasible": bool(self.always_infeasible),
            "num_forbidden_patterns": int(self.num_forbidden_patterns),
            "source_constraint_counts": {
                name: int(count) for name, count in self.source_constraint_counts
            },
            "retained_pattern_counts": {
                name: int(count) for name, count in self.retained_pattern_counts
            },
            "logic_feasible_count": int(self.logic_feasible_count),
            "logic_infeasible_count": int(self.logic_infeasible_count),
            "feasible_ratio": float(self.feasible_ratio),
            "logic_feasible_indices": [
                int(index) for index in self.logic_feasible_indices
            ],
            "logic_infeasible_indices": [
                int(index) for index in self.logic_infeasible_indices
            ],
            "base_commitment": [list(row) for row in self.base_commitment],
            "classical_spec_agree": bool(self.classical_spec_agree),
            "has_active_filter": bool(self.has_active_filter),
            "selection_uses_cost_or_hidden_optimum": False,
        }


def build_logic_safe_base_commitment(instance: UCInstance) -> np.ndarray:
    """Build a deterministic high-capacity commitment satisfying Boolean UC logic.

    Unselected generators remain online whenever their initial minimum-down
    obligation permits it. This uses no load, cost, ED/LP, or hidden optimum.
    """

    horizon = int(instance.time_horizon)
    base = np.zeros((len(instance.generators), horizon), dtype=int)
    for generator_index, generator in enumerate(instance.generators):
        if generator.must_run:
            if generator.initial_status < 0:
                remaining_down = max(
                    int(generator.min_downtime) + int(generator.initial_status),
                    0,
                )
                if remaining_down > 0:
                    raise ValueError(
                        f"{generator.name}: must-run 与初始剩余 minimum downtime 冲突"
                    )
            base[generator_index, :] = 1
            continue

        if generator.initial_status > 0:
            base[generator_index, :] = 1
            continue

        remaining_down = (
            max(int(generator.min_downtime) + int(generator.initial_status), 0)
            if generator.initial_status < 0
            else 0
        )
        start = min(remaining_down, horizon)
        if start < horizon:
            base[generator_index, start:] = 1

    if not is_logic_feasible(instance, base):
        raise RuntimeError("无法构造满足 is_logic_feasible 的确定性 base commitment")
    return base


def scan_logic_feasibility_subspaces(
    source: UCInstance,
    *,
    horizons: Iterable[int] = (2, 3),
    selected_generator_count: int = 2,
    window_start: int = 0,
) -> tuple[LogicSubspaceScanRow, ...]:
    """Enumerate small selected subspaces for logic diagnostics only.

    The scan never evaluates cost, ED/LP, a VQC, BBHT, or a hidden optimum.
    Enumeration is limited to the selected Boolean subspace and is not used by
    the Grover/BBHT builders.
    """

    selected_generator_count = int(selected_generator_count)
    if selected_generator_count <= 0:
        raise ValueError("selected_generator_count 必须为正整数")
    rows: list[LogicSubspaceScanRow] = []
    for raw_horizon in horizons:
        horizon = int(raw_horizon)
        if horizon <= 0:
            raise ValueError("horizon 必须为正整数")
        if int(window_start) + horizon > source.time_horizon:
            continue
        instance = time_window_instance(
            source,
            start=int(window_start),
            horizon=horizon,
        )
        base = build_logic_safe_base_commitment(instance)
        for selected in combinations(
            range(len(instance.generators)),
            selected_generator_count,
        ):
            spec = compile_logic_feasibility_spec(
                instance,
                selected_generator_indices=selected,
                base_commitment=base,
            )
            feasible: list[int] = []
            infeasible: list[int] = []
            classical_agree = True
            dimension = 2 ** int(spec.num_x_qubits)
            for index in range(dimension):
                bits = _bits_from_index(index, spec.num_x_qubits)
                spec_feasible = bool(spec.is_feasible(bits))
                commitment = _embed_bits(base, selected, bits, horizon)
                classical_feasible = bool(is_logic_feasible(instance, commitment))
                classical_agree = classical_agree and spec_feasible == classical_feasible
                (feasible if spec_feasible else infeasible).append(int(index))
            retained_counts: dict[str, int] = {}
            for pattern in spec.patterns:
                family = _pattern_family(pattern.label)
                retained_counts[family] = retained_counts.get(family, 0) + 1
            rows.append(
                LogicSubspaceScanRow(
                    window_start=int(window_start),
                    horizon=horizon,
                    selected_generator_indices=tuple(int(index) for index in selected),
                    selected_generator_names=tuple(
                        instance.generators[index].name for index in selected
                    ),
                    num_x_qubits=int(spec.num_x_qubits),
                    always_infeasible=bool(spec.always_infeasible),
                    num_forbidden_patterns=int(len(spec.patterns)),
                    source_constraint_counts=tuple(spec.source_constraint_counts),
                    retained_pattern_counts=tuple(sorted(retained_counts.items())),
                    logic_feasible_indices=tuple(feasible),
                    logic_infeasible_indices=tuple(infeasible),
                    base_commitment=tuple(
                        tuple(int(value) for value in row) for row in base
                    ),
                    classical_spec_agree=bool(classical_agree),
                )
            )
    return tuple(rows)


def select_active_logic_subspace(
    rows: Sequence[LogicSubspaceScanRow],
    *,
    minimum_feasible_states: int = 8,
    preferred_feasible_ratio: tuple[float, float] = (0.25, 0.75),
) -> LogicSubspaceScanRow:
    """Select an active subspace using logic-only deterministic criteria."""

    minimum_feasible_states = int(minimum_feasible_states)
    if minimum_feasible_states <= 0:
        raise ValueError("minimum_feasible_states 必须为正整数")
    lower, upper = (float(value) for value in preferred_feasible_ratio)
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("preferred_feasible_ratio 必须位于 [0, 1]")

    candidates = [
        row
        for row in rows
        if row.has_active_filter
        and row.classical_spec_agree
        and row.logic_feasible_count >= minimum_feasible_states
    ]
    if not candidates:
        raise RuntimeError(
            "扫描未找到同时包含可行/不可行状态且训练样本充足的 active logic subspace"
        )

    def rank(row: LogicSubspaceScanRow) -> tuple[object, ...]:
        preferred = lower <= row.feasible_ratio <= upper
        return (
            0 if preferred else 1,
            -int(row.logic_infeasible_count),
            int(row.num_forbidden_patterns),
            int(row.horizon),
            tuple(int(index) for index in row.selected_generator_indices),
        )

    return min(candidates, key=rank)


def _embed_bits(
    base: np.ndarray,
    selected_generator_indices: Sequence[int],
    bits: Sequence[int],
    horizon: int,
) -> np.ndarray:
    commitment = np.asarray(base, dtype=int).copy()
    for local_generator, global_generator in enumerate(selected_generator_indices):
        for period in range(int(horizon)):
            qubit = local_generator * int(horizon) + period
            commitment[int(global_generator), period] = int(bits[qubit])
    return commitment


def _pattern_family(label: str) -> str:
    value = str(label)
    for prefix, family in (
        ("must_run_", "must_run"),
        ("initial_down_", "initial_remaining_minimum_downtime"),
        ("initial_up_", "initial_remaining_minimum_uptime"),
        ("min_up_", "minimum_uptime_after_startup"),
        ("min_down_", "minimum_downtime_after_shutdown"),
    ):
        if value.startswith(prefix):
            return family
    return "other"


def _bits_from_index(index: int, num_bits: int) -> tuple[int, ...]:
    return tuple((int(index) >> bit) & 1 for bit in range(int(num_bits)))
