from __future__ import annotations

import numpy as np
import pytest

from qubit_value_function.stage2_error_attribution import (
    PENALTY_DOMINANT_RATIO,
    aggregate_seed_rows_by_unit,
    bootstrap_paired_interval,
    feasible_improvement_labels,
    join_formal_and_components,
    margin_targets_from_fit_rows,
    residual_targets_from_fit_rows,
    reserve_penalty_group_label,
)
from qubit_value_function.stage2_controlled_alternatives import QuadraticResidualModel


def _keyed_row(*, state: int, value: float) -> dict[str, object]:
    return {"generator_pair": "[0, 1]", "window_start": 0, "load_multiplier": 1.0, "state_index": state, "value": value}


def test_component_join_is_complete_and_independent_of_input_row_order() -> None:
    formal = [_keyed_row(state=0, value=1.0), _keyed_row(state=1, value=2.0)]
    components = [{**_keyed_row(state=1, value=20.0), "hard_logic_feasible": True}, {**_keyed_row(state=0, value=10.0), "hard_logic_feasible": False}]
    joined = join_formal_and_components(formal, components)
    assert [row["hard_logic_feasible"] for row in joined] == [False, True]
    with pytest.raises(ValueError, match="component_join_missing"):
        join_formal_and_components(formal, components[:1])


def test_residual_and_margin_targets_are_constructed_only_from_fit_values() -> None:
    fit_truth = np.asarray([10.0, 12.0]); fit_base = np.asarray([8.0, 13.0])
    assert np.array_equal(residual_targets_from_fit_rows(fit_truth, fit_base), np.asarray([2.0, -1.0]))
    margins, tau = margin_targets_from_fit_rows(loads=[0.85, 0.85], true_costs=fit_truth)
    assert tau == {0.85: 10.0}
    assert np.array_equal(margins, np.asarray([0.0, -2.0]))


def test_feasible_strict_improvement_denominator_excludes_hard_logic_infeasible_rows() -> None:
    labels = feasible_improvement_labels(true_costs=[9.0, 8.0], thresholds=[10.0, 10.0], hard_logic_feasible=[True, False], edlp_success=[True, True])
    assert np.array_equal(labels, np.asarray([True, False]))


def test_unit_seed_aggregation_and_bootstrap_are_deterministic() -> None:
    rows = [
        {"generator_pair": "p", "window_start": 0, "split_seed": 1, "model": "m", "slice": "x", "seed": 11, "mae": 1.0},
        {"generator_pair": "p", "window_start": 0, "split_seed": 1, "model": "m", "slice": "x", "seed": 23, "mae": 3.0},
    ]
    aggregate = aggregate_seed_rows_by_unit(rows, metric="mae")
    assert aggregate[0]["mae"] == 2.0 and aggregate[0]["seed_count"] == 2
    first = bootstrap_paired_interval([1.0, -1.0, 2.0], seed=7, draws=200)
    assert first == bootstrap_paired_interval([1.0, -1.0, 2.0], seed=7, draws=200)


def test_reserve_penalty_without_nonzero_rows_is_explicitly_not_applicable() -> None:
    assert PENALTY_DOMINANT_RATIO == 0.5
    assert reserve_penalty_group_label([0.0, 0.0]) == "N/A: no nonzero reserve-penalty samples"


def test_residual_model_constructs_targets_from_fit_costs_only() -> None:
    states = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=int)
    loads = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    model = QuadraticResidualModel(correction="linear", seed=11).fit(states, loads, [1.0, 2.0, 3.0, 5.0])
    assert np.isfinite(model.fit_residual_mae)
    assert np.all(np.isfinite(model.predict(states, loads)))
