from __future__ import annotations

from pathlib import Path

from qubit_value_function.logic_subspace_scan import (
    scan_logic_feasibility_subspaces,
    select_active_logic_subspace,
)
from qubit_value_function.uc_loader import load_uc_instance


def test_case14_scan_finds_active_logic_subspace_without_cost_information() -> None:
    source = load_uc_instance(Path("data/case14.json.gz"))
    rows = scan_logic_feasibility_subspaces(
        source,
        horizons=(2, 3),
        selected_generator_count=2,
        window_start=0,
    )
    assert rows
    assert all(row.classical_spec_agree for row in rows)
    selected = select_active_logic_subspace(rows, minimum_feasible_states=8)
    assert selected.has_active_filter
    assert selected.logic_feasible_count >= 8
    assert selected.logic_infeasible_count > 0
    assert selected.num_forbidden_patterns > 0
