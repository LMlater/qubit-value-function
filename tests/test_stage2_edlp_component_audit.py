from __future__ import annotations

import json
from pathlib import Path

import pytest

from qubit_value_function.stage2_edlp_component_audit import (
    FORMAL_KEY_FIELDS,
    component_cost_sum,
    join_by_formal_key,
    require_hard_logic_agreement,
    strict_json_dumps,
    validate_audit_paths,
    validate_formal_truth_rows,
    within_cost_tolerance,
)


def _row(*, state_index: int, true_cost: float = 10.0) -> dict[str, object]:
    return {
        "generator_pair": "[0, 1]", "window_start": 0, "load_multiplier": 0.8,
        "state_index": state_index, "state_bits": "[0, 0, 0, 0]",
        "load_vector": "[1.0, 1.0]", "true_cost": true_cost,
    }


def test_formal_truth_validation_requires_unique_protocol_keys() -> None:
    rows = [_row(state_index=0), _row(state_index=1)]
    validate_formal_truth_rows(rows, expected_count=2)
    with pytest.raises(ValueError, match="duplicate_formal_key"):
        validate_formal_truth_rows([_row(state_index=0), _row(state_index=0)], expected_count=2)
    assert FORMAL_KEY_FIELDS == ("generator_pair", "window_start", "load_multiplier", "state_index")


def test_join_is_keyed_not_row_order_dependent() -> None:
    formal = [_row(state_index=0), _row(state_index=1)]
    supplemental = [
        {**_row(state_index=1), "hard_logic_feasible": True},
        {**_row(state_index=0), "hard_logic_feasible": False},
    ]
    joined = join_by_formal_key(formal, supplemental)
    assert [row["hard_logic_feasible"] for row in joined] == [False, True]


def test_component_sum_and_fixed_cost_tolerance_gate() -> None:
    assert component_cost_sum(dispatch_cost=1.0, startup_cost=2.0, balance_penalty=3.0, reserve_penalty=4.0) == 10.0
    assert within_cost_tolerance(formal_true_cost=1_000_000.0, recomputed_total_cost=1_000_000.0005)
    assert not within_cost_tolerance(formal_true_cost=10.0, recomputed_total_cost=10.00001)


def test_hard_logic_disagreement_blocks_downstream_attribution() -> None:
    with pytest.raises(RuntimeError, match="hard_logic_disagreement"):
        require_hard_logic_agreement([
            {"hard_logic_feasible": True, "independent_hard_logic_feasible": False},
        ])


def test_strict_json_is_parseable_and_rejects_nonfinite_values(tmp_path: Path) -> None:
    payload = strict_json_dumps({"ok": 1.0, "none": None})
    assert json.loads(payload) == {"none": None, "ok": 1.0}
    with pytest.raises(ValueError):
        strict_json_dumps({"bad": float("nan")})


def test_component_audit_refuses_to_write_into_formal_benchmark_directory(tmp_path: Path) -> None:
    formal = tmp_path / "formal"; output = tmp_path / "output"
    formal.mkdir()
    validate_audit_paths(formal, output)
    with pytest.raises(ValueError, match="formal_benchmark_read_only"):
        validate_audit_paths(formal, formal / "attempted_write")
