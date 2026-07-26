from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from qubit_value_function.candidate_acceptance_loop import (
    ClosedLoopBudgets,
    ExactCandidateEvaluation,
)
from qubit_value_function.closed_loop_random_baselines import (
    run_full_space_random,
    run_logic_rejection_random,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig


@dataclass(frozen=True)
class _Model:
    values: tuple[int, int, int, int]
    num_x_qubits: int = 2
    fixed_point_config: FixedPointConfig = FixedPointConfig(fractional_bits=0, unit=1.0)

    def integer_value(self, bits: tuple[int, ...]) -> int:
        index = sum(int(bit) << offset for offset, bit in enumerate(bits))
        return int(self.values[index])


def _record(
    cost: float | None,
    *,
    success: bool = True,
    source: str = "new_ed_lp_call",
    lp_solve: bool = True,
    precheck: bool = False,
) -> ExactCandidateEvaluation:
    return ExactCandidateEvaluation(
        success=success,
        total_cost=cost,
        message="ok" if success else "rejected",
        source=source,
        lp_solve_performed=lp_solve,
        logic_precheck_rejected=precheck,
    )


def _budgets(*, proposals: int = 1) -> ClosedLoopBudgets:
    return ClosedLoopBudgets(
        max_proposals=proposals,
        max_new_exact_evaluations=4,
        max_actual_ed_lp_solves=4,
        max_threshold_updates=4,
        max_consecutive_nonimproving_marked=4,
        max_same_encoded_threshold_updates=3,
        max_auxiliary_syndrome_rejections=2,
    )


def _seed_for(indices: tuple[int, ...]) -> int:
    for seed in range(10_000):
        draws = tuple(int(value) for value in np.random.default_rng(seed).integers(0, 4, size=len(indices)))
        if draws == indices:
            return seed
    raise AssertionError("failed to find deterministic seed")


def _run_full(model: _Model, *, seed: int, proposals: int = 1, evaluator=None):
    return run_full_space_random(
        model,
        initial_incumbent_index=3,
        initial_exact_cache={3: _record(3.0, source="training_exact_cache")},
        training_indices=(3,),
        evaluate_candidate=evaluator or (lambda index: _record(2.0)),
        budgets=_budgets(proposals=proposals),
        seed=seed,
        hard_logic_is_feasible=lambda bits: bits != (0, 0),
    )


def test_full_space_random_accepts_surrogate_unmarked_true_improvement() -> None:
    result = _run_full(_Model((9, 9, 9, 9)), seed=_seed_for((1,)))

    assert result.final_incumbent_index == 1
    assert result.final_incumbent_true_cost == 2.0
    assert result.trial_trace[0]["surrogate_better"] is False
    assert result.trial_trace[0]["admission_passed"] is True
    assert result.trial_trace[0]["threshold_updated"] is True


def test_logic_rejection_random_skips_infeasible_candidate_but_charges_proposal() -> None:
    calls: list[int] = []
    result = run_logic_rejection_random(
        _Model((9, 9, 9, 9)),
        initial_incumbent_index=3,
        initial_exact_cache={3: _record(3.0, source="training_exact_cache")},
        training_indices=(3,),
        evaluate_candidate=lambda index: calls.append(index) or _record(2.0),
        budgets=_budgets(),
        seed=_seed_for((0,)),
        hard_logic_is_feasible=lambda bits: bits != (0, 0),
    )

    assert calls == []
    assert result.proposals_used == 1
    assert result.new_exact_evaluation_attempts == 0
    assert result.trial_trace[0]["admission_rejection_reason"] == "hard_logic_infeasible"


def test_logic_rejection_random_accepts_surrogate_unmarked_true_improvement() -> None:
    result = run_logic_rejection_random(
        _Model((9, 9, 9, 9)),
        initial_incumbent_index=3,
        initial_exact_cache={3: _record(3.0, source="training_exact_cache")},
        training_indices=(3,),
        evaluate_candidate=lambda index: _record(2.0),
        budgets=_budgets(),
        seed=_seed_for((1,)),
        hard_logic_is_feasible=lambda bits: bits != (0, 0),
    )

    assert result.final_incumbent_index == 1
    assert result.trial_trace[0]["surrogate_better"] is False
    assert result.trial_trace[0]["admission_passed"] is True
    assert result.trial_trace[0]["threshold_updated"] is True


def test_full_space_random_sends_infeasible_candidate_to_exact_precheck() -> None:
    result = _run_full(
        _Model((9, 9, 9, 9)),
        seed=_seed_for((0,)),
        evaluator=lambda index: _record(
            None,
            success=False,
            source="logic_precheck",
            lp_solve=False,
            precheck=True,
        ),
    )

    assert result.new_exact_evaluation_attempts == 1
    assert result.actual_ed_lp_solves == 0
    assert result.logic_precheck_rejections == 1
    assert result.trial_trace[0]["admission_passed"] is True


def test_random_candidate_sequence_is_seed_reproducible_and_proxy_independent() -> None:
    seed = _seed_for((2, 1, 2))
    low_proxy = _run_full(_Model((0, 1, 2, 3)), seed=seed, proposals=3)
    high_proxy = _run_full(_Model((9, 9, 9, 9)), seed=seed, proposals=3)
    repeat = _run_full(_Model((0, 1, 2, 3)), seed=seed, proposals=3)

    first_sequence = [row["candidate_index"] for row in low_proxy.trial_trace]
    assert first_sequence == [row["candidate_index"] for row in high_proxy.trial_trace]
    assert first_sequence == [row["candidate_index"] for row in repeat.trial_trace]
    assert all(row["oracle_calls_added"] == 0 for row in low_proxy.trial_trace)
    assert all(row["grover_iterations"] == 0 for row in low_proxy.trial_trace)

    low_logic = run_logic_rejection_random(
        _Model((0, 1, 2, 3)),
        initial_incumbent_index=3,
        initial_exact_cache={3: _record(3.0, source="training_exact_cache")},
        training_indices=(3,),
        evaluate_candidate=lambda index: _record(3.0),
        budgets=_budgets(proposals=3),
        seed=seed,
        hard_logic_is_feasible=lambda bits: True,
    )
    high_logic = run_logic_rejection_random(
        _Model((9, 9, 9, 9)),
        initial_incumbent_index=3,
        initial_exact_cache={3: _record(3.0, source="training_exact_cache")},
        training_indices=(3,),
        evaluate_candidate=lambda index: _record(3.0),
        budgets=_budgets(proposals=3),
        seed=seed,
        hard_logic_is_feasible=lambda bits: True,
    )
    assert [row["candidate_index"] for row in low_logic.trial_trace] == [
        row["candidate_index"] for row in high_logic.trial_trace
    ]


def test_random_repeats_are_not_free_and_equal_true_cost_does_not_update() -> None:
    seed = _seed_for((1, 1))
    result = _run_full(
        _Model((9, 9, 9, 9)),
        seed=seed,
        proposals=2,
        evaluator=lambda index: _record(3.0),
    )

    assert result.proposals_used == 2
    assert result.trial_trace[1]["is_repeated_candidate"] is True
    assert result.final_incumbent_index == 3
    assert result.threshold_updates == 0


def test_random_runner_has_no_validation_landscape_parameter() -> None:
    assert "validation" not in run_full_space_random.__annotations__
    assert "validation" not in run_logic_rejection_random.__annotations__


def test_random_runs_keep_independent_threshold_histories() -> None:
    first = _run_full(_Model((9, 9, 9, 9)), seed=_seed_for((1,)))
    second = _run_full(
        _Model((9, 9, 9, 9)),
        seed=_seed_for((2,)),
        evaluator=lambda index: _record(3.0),
    )

    assert first.final_encoded_threshold == 2
    assert second.final_encoded_threshold == 3
    assert len(first.threshold_history) == 2
    assert len(second.threshold_history) == 1
