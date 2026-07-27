from __future__ import annotations

import inspect

import pytest

from qubit_value_function.candidate_acceptance_loop import ExactCandidateEvaluation
from qubit_value_function.sparse_vqc_bbht import (
    select_initial_incumbent,
    select_training_initial_incumbent,
)


def _record(cost: float | None, *, success: bool = True) -> ExactCandidateEvaluation:
    return ExactCandidateEvaluation(
        success=success,
        total_cost=cost,
        message="ok" if success else "failed",
        source="training_exact_cache",
    )


def test_first_training_is_default_and_preserves_original_first_finite_choice() -> None:
    indices = (5, 3, 1)
    cache = {5: _record(9.0), 3: _record(2.0), 1: _record(1.0)}
    selection = select_training_initial_incumbent(indices, cache)
    assert selection.policy == "first_training"
    assert selection.index == 5
    assert selection.true_cost == 9.0
    assert selection.index == select_initial_incumbent(indices, cache, policy="first")


def test_best_training_uses_only_finite_cached_training_labels_and_preserves_tie_order() -> None:
    indices = (5, 3, 1, 2)
    cache = {5: _record(9.0), 3: _record(2.0), 1: _record(2.0), 2: _record(None, success=False)}
    before = dict(cache)
    selection = select_training_initial_incumbent(indices, cache, policy="best_training")
    assert selection.index == 3
    assert selection.true_cost == 2.0
    assert selection.best_training_candidate_indices == (3, 1)
    assert selection.best_training_tie_count == 2
    assert selection.best_training_tie_break_rule == "first_in_training_indices_order"
    assert cache == before
    assert all(float(record.total_cost) >= selection.true_cost for record in cache.values() if record.success and record.total_cost is not None)


def test_best_training_rejects_missing_or_nonfinite_training_labels_without_rng_or_global_truth() -> None:
    with pytest.raises(ValueError, match="finite"):
        select_training_initial_incumbent((0, 1), {0: _record(None, success=False)}, policy="best_training")
    source = inspect.getsource(select_training_initial_incumbent)
    assert "default_rng" not in source
    assert "global" not in source.lower()
    assert "validation" not in source.lower()


def test_initial_incumbent_policy_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="initial_incumbent_policy"):
        select_training_initial_incumbent((0,), {0: _record(1.0)}, policy="random")
