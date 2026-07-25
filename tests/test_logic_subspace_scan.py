from __future__ import annotations

from pathlib import Path

from qubit_value_function.logic_subspace_scan import (
    build_logic_safe_base_commitment,
    scan_logic_feasibility_subspaces,
    select_active_logic_subspace,
)
from qubit_value_function.uc_loader import Generator, UCInstance


def _generator(name: str, *, initial_status: int, min_uptime: int = 1, min_downtime: int = 1, must_run: bool = False) -> Generator:
    return Generator(
        name=name,
        bus="b1",
        cost_mw=[0.0, 100.0],
        cost_usd=[0.0, 1000.0],
        startup_delays=[1],
        startup_costs=[10.0],
        initial_status=initial_status,
        initial_power=0.0,
        min_uptime=min_uptime,
        min_downtime=min_downtime,
        ramp_up=None,
        ramp_down=None,
        must_run=must_run,
        reserve_eligibility=(),
    )


def _instance() -> UCInstance:
    return UCInstance(
        path=Path("synthetic.json"),
        version="test",
        time_horizon=3,
        generators=[
            _generator("g0", initial_status=0, min_uptime=2),
            _generator("g1", initial_status=1, min_downtime=2),
            _generator("g2", initial_status=0),
        ],
        fixed_load=[100.0, 100.0, 100.0],
        reserves=[],
        power_balance_penalty=[1000.0, 1000.0, 1000.0],
    )


def test_logic_safe_base_and_scan_match_classical_semantics() -> None:
    instance = _instance()
    base = build_logic_safe_base_commitment(instance)
    assert base.shape == (3, 3)
    rows = scan_logic_feasibility_subspaces(
        instance,
        horizons=(2, 3),
        selected_generator_count=2,
        window_start=0,
    )
    assert len(rows) == 6
    assert all(row.classical_spec_agree for row in rows)
    assert any(row.has_active_filter for row in rows)


def test_active_subspace_selection_is_logic_only_and_deterministic() -> None:
    rows = scan_logic_feasibility_subspaces(
        _instance(),
        horizons=(2, 3),
        selected_generator_count=2,
        window_start=0,
    )
    selected_a = select_active_logic_subspace(rows, minimum_feasible_states=2)
    selected_b = select_active_logic_subspace(rows, minimum_feasible_states=2)
    assert selected_a == selected_b
    assert selected_a.has_active_filter
    assert selected_a.logic_feasible_count >= 2
    assert selected_a.logic_infeasible_count > 0
    assert selected_a.as_dict()["selection_uses_cost_or_hidden_optimum"] is False
