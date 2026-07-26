from __future__ import annotations

import inspect

import pytest

from qubit_value_function.candidate_acceptance_loop import (
    CandidateProposal,
    ClosedLoopBudgets,
    ExactCandidateEvaluation,
    accept_candidate_proposal,
    create_closed_loop_state,
)
from qubit_value_function.sparse_vqc_bbht import run_sparse_vqc_bbht


def _record(
    cost: float | None,
    *,
    success: bool = True,
    source: str = "new_ed_lp_call",
    lp_solve: bool = True,
) -> ExactCandidateEvaluation:
    return ExactCandidateEvaluation(
        success=success,
        total_cost=cost,
        message="ok" if success else "failed",
        source=source,
        lp_solve_performed=lp_solve,
    )


def _state(
    *,
    training_indices: tuple[int, ...] = (3,),
    allow_zero: bool = False,
):
    return create_closed_loop_state(
        initial_incumbent_index=3,
        initial_exact_cache={3: _record(3.0, source="training_exact_cache")},
        training_indices=training_indices,
        num_x_qubits=2,
        encode_true_cost=lambda value: int(value),
        hard_logic_is_feasible=(
            (lambda bits: True) if allow_zero else (lambda bits: bits != (0, 0))
        ),
        budgets=ClosedLoopBudgets(
            max_proposals=8,
            max_new_exact_evaluations=4,
            max_actual_ed_lp_solves=4,
            max_threshold_updates=4,
            max_consecutive_nonimproving_marked=4,
            max_same_encoded_threshold_updates=3,
            max_auxiliary_syndrome_rejections=2,
        ),
    )


def _proposal(index: int, *, auxiliary_accepted: bool = True) -> CandidateProposal:
    return CandidateProposal(
        candidate_index=index,
        source_method="bbht",
        auxiliary_accepted=auxiliary_accepted,
        surrogate_integer_cost=index,
        grover_iterations=1,
        oracle_calls=1,
    )


def test_only_strict_true_cost_improvement_updates_incumbent_and_threshold() -> None:
    state = _state(allow_zero=True)
    equal = accept_candidate_proposal(
        state,
        _proposal(2),
        evaluate_candidate=lambda index: _record(3.0),
    )
    predicted_only = accept_candidate_proposal(
        state,
        _proposal(1),
        evaluate_candidate=lambda index: _record(4.0),
    )
    improved = accept_candidate_proposal(
        state,
        _proposal(0),
        evaluate_candidate=lambda index: _record(2.0),
    )

    assert equal.accepted_update is False
    assert predicted_only.accepted_update is False
    assert state.incumbent_index == 0
    assert state.incumbent_true_cost == 2.0
    assert state.encoded_threshold == 2
    assert improved.threshold_before == 3
    assert improved.threshold_after == 2


def test_cache_confirmation_skips_new_edlp_and_uncached_attempt_counts_actual_solve() -> None:
    state = _state()
    state.exact_cache[2] = _record(2.0, source="training_exact_cache")
    cached = accept_candidate_proposal(
        state,
        _proposal(2),
        evaluate_candidate=lambda index: pytest.fail("cache hit must not evaluate"),
    )
    uncached = accept_candidate_proposal(
        state,
        _proposal(1),
        evaluate_candidate=lambda index: _record(1.0),
    )

    assert cached.is_cache_hit is True
    assert cached.cache_confirmed_improvement is True
    assert cached.new_edlp_solve_performed is False
    assert uncached.new_exact_evaluation_attempted is True
    assert uncached.new_edlp_solve_performed is True
    assert uncached.new_edlp_confirmed_improvement is True
    assert uncached.trace_fields()["new_exact_evaluation_attempted"] is True
    assert uncached.trace_fields()["new_edlp_solve_performed"] is True
    assert state.new_exact_evaluation_attempts == 1
    assert state.actual_ed_lp_solves == 1


def test_every_proposal_is_charged_even_when_auxiliary_unmarked_or_logic_rejected() -> None:
    state = _state()
    auxiliary = accept_candidate_proposal(
        state,
        _proposal(1, auxiliary_accepted=False),
        evaluate_candidate=lambda index: pytest.fail("auxiliary rejection must skip evaluation"),
    )
    logic = accept_candidate_proposal(
        state,
        _proposal(0),
        evaluate_candidate=lambda index: pytest.fail("logic rejection must skip evaluation"),
    )
    unmarked = accept_candidate_proposal(
        state,
        CandidateProposal(
            candidate_index=3,
            source_method="bbht",
            auxiliary_accepted=True,
            surrogate_integer_cost=3,
        ),
        evaluate_candidate=lambda index: pytest.fail("unmarked proposal must skip evaluation"),
    )

    assert state.proposals_used == 3
    assert auxiliary.rejection_reason == "auxiliary_syndrome_rejected"
    assert logic.rejection_reason == "hard_logic_infeasible"
    assert unmarked.rejection_reason == "surrogate_unmarked"


def test_nontraining_first_improvement_is_recorded_once_and_repeats_remain_charged() -> None:
    state = _state(training_indices=(3,))
    first = accept_candidate_proposal(
        state,
        _proposal(2),
        evaluate_candidate=lambda index: _record(2.0),
    )
    repeated = accept_candidate_proposal(
        state,
        _proposal(2),
        evaluate_candidate=lambda index: pytest.fail("repeat must use cache"),
    )

    assert first.is_training_state is False
    assert first.nontraining_true_improvement is True
    assert first.first_nontraining_improvement is True
    assert repeated.is_repeated_candidate is True
    assert repeated.is_unique_candidate is False
    assert repeated.first_nontraining_improvement is False
    assert state.proposals_used == 2


def test_failed_exact_evaluation_is_charged_without_updating_incumbent() -> None:
    state = _state()
    decision = accept_candidate_proposal(
        state,
        _proposal(2),
        evaluate_candidate=lambda index: _record(
            None,
            success=False,
            source="ed_lp_failed",
            lp_solve=True,
        ),
    )

    assert decision.accepted_update is False
    assert decision.new_exact_evaluation_attempted is True
    assert decision.new_edlp_solve_performed is True
    assert state.incumbent_index == 3
    assert state.encoded_threshold == 3


def test_shared_acceptance_api_has_no_validation_landscape_input() -> None:
    assert "validation" not in inspect.signature(accept_candidate_proposal).parameters


def test_bbht_delegates_incumbent_mutation_to_shared_acceptance_loop() -> None:
    source = inspect.getsource(run_sparse_vqc_bbht)
    assert "accept_candidate_proposal(" in source
    assert "state.incumbent_true_cost =" not in source
    assert "state.encoded_threshold =" not in source
