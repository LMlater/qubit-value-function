from __future__ import annotations

from pathlib import Path

import pytest

from experiments.stage2_case14_active_logic_margin_sweep import (
    _resolve_selected_subspace,
    _validate_margins,
    _window_execution_plan,
)
from qubit_value_function.logic_subspace_scan import (
    scan_logic_feasibility_subspaces,
)
from qubit_value_function.uc_loader import UCInstance


def test_window_execution_plan_records_invalid_windows_explicitly() -> None:
    executed, skipped = _window_execution_plan(
        source_horizon=4,
        horizon=3,
        requested_window_starts=(0, 1, 2, -1),
    )
    assert executed == (0, 1)
    assert [row["window_start"] for row in skipped] == [2, -1]
    assert "exceeds source horizon" in skipped[0]["reason"]
    assert "nonnegative" in skipped[1]["reason"]


def test_selected_generators_and_horizon_are_an_atomic_contract() -> None:
    with pytest.raises(ValueError, match="同时提供或同时缺省"):
        _resolve_selected_subspace(
            None,
            scan_horizons=(2, 3),
            scan_window_start=0,
            selected_generator_indices=(0, 1),
            horizon=None,
            train_sample_count=8,
            max_scan_qubits=12,
        )
    with pytest.raises(ValueError, match="同时提供或同时缺省"):
        _resolve_selected_subspace(
            None,
            scan_horizons=(2, 3),
            scan_window_start=0,
            selected_generator_indices=None,
            horizon=3,
            train_sample_count=8,
            max_scan_qubits=12,
        )


def test_margin_contract_rejects_negative_and_duplicate_values() -> None:
    assert _validate_margins((0, 1, 2, 4, 8)) == (0, 1, 2, 4, 8)
    with pytest.raises(ValueError, match="非负"):
        _validate_margins((0, -1))
    with pytest.raises(ValueError, match="不能重复"):
        _validate_margins((0, 1, 1))


def test_logic_scan_qubit_guard_precedes_exponential_enumeration() -> None:
    source = UCInstance(
        path=Path("synthetic"),
        version="test",
        time_horizon=3,
        generators=[],
        fixed_load=[0.0, 0.0, 0.0],
        reserves=[],
        power_balance_penalty=[1.0, 1.0, 1.0],
    )
    with pytest.raises(ValueError, match="max_scan_qubits"):
        scan_logic_feasibility_subspaces(
            source,
            horizons=(3,),
            selected_generator_count=5,
            window_start=0,
            max_scan_qubits=12,
        )
