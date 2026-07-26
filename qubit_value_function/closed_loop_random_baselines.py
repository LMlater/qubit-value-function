from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .candidate_acceptance_loop import (
    CandidateAdmissionPolicy,
    CandidateProposal,
    ClosedLoopBudgets,
    ExactCandidateEvaluation,
    ExactEvaluator,
    FULL_SPACE_RANDOM_ADMISSION_POLICY,
    HardLogicEvaluator,
    LOGIC_REJECTION_RANDOM_ADMISSION_POLICY,
    accept_candidate_proposal,
    create_closed_loop_state,
)
from .coherent_phase_value import QuantizedSparseValueModel


@dataclass(frozen=True)
class ClosedLoopRandomResult:
    method: str
    seed: int
    initial_incumbent_index: int
    initial_incumbent_true_cost: float
    final_incumbent_index: int
    final_incumbent_true_cost: float
    final_encoded_threshold: int
    stop_reason: str
    trial_trace: tuple[dict[str, object], ...]
    threshold_history: tuple[dict[str, object], ...]
    exact_cache: dict[int, ExactCandidateEvaluation]
    proposals_used: int
    new_exact_evaluation_attempts: int
    actual_ed_lp_solves: int
    logic_precheck_rejections: int
    cached_exact_lookups: int
    threshold_updates: int
    admission_policy: CandidateAdmissionPolicy
    budgets: ClosedLoopBudgets

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "seed": int(self.seed),
            "initial_incumbent_index": int(self.initial_incumbent_index),
            "initial_incumbent_true_cost": float(self.initial_incumbent_true_cost),
            "final_incumbent_index": int(self.final_incumbent_index),
            "final_incumbent_true_cost": float(self.final_incumbent_true_cost),
            "final_encoded_threshold": int(self.final_encoded_threshold),
            "stop_reason": self.stop_reason,
            "trial_trace": list(self.trial_trace),
            "threshold_history": list(self.threshold_history),
            "exact_cache": {
                str(index): record.as_dict()
                for index, record in sorted(self.exact_cache.items())
            },
            "counters": {
                "proposals_used": int(self.proposals_used),
                "new_exact_evaluation_attempts": int(self.new_exact_evaluation_attempts),
                "actual_ed_lp_solves": int(self.actual_ed_lp_solves),
                "logic_precheck_rejections": int(self.logic_precheck_rejections),
                "cached_exact_lookups": int(self.cached_exact_lookups),
                "threshold_updates": int(self.threshold_updates),
            },
            "admission_policy": {
                "name": self.admission_policy.name,
                "require_auxiliary_accepted": self.admission_policy.require_auxiliary_accepted,
                "require_hard_logic_feasible": self.admission_policy.require_hard_logic_feasible,
                "require_surrogate_better": self.admission_policy.require_surrogate_better,
            },
            "quantum_resources": {
                "applicable": False,
                "oracle_calls": 0,
                "grover_iterations": 0,
            },
        }


def run_full_space_random(
    model: QuantizedSparseValueModel,
    *,
    initial_incumbent_index: int,
    initial_exact_cache: Mapping[int, ExactCandidateEvaluation],
    training_indices: Sequence[int],
    evaluate_candidate: ExactEvaluator,
    budgets: ClosedLoopBudgets,
    seed: int,
    hard_logic_is_feasible: HardLogicEvaluator | None = None,
) -> ClosedLoopRandomResult:
    return _run_random_closed_loop(
        model,
        method="full_space_random",
        admission_policy=FULL_SPACE_RANDOM_ADMISSION_POLICY,
        initial_incumbent_index=initial_incumbent_index,
        initial_exact_cache=initial_exact_cache,
        training_indices=training_indices,
        evaluate_candidate=evaluate_candidate,
        budgets=budgets,
        seed=seed,
        hard_logic_is_feasible=hard_logic_is_feasible,
    )


def run_logic_rejection_random(
    model: QuantizedSparseValueModel,
    *,
    initial_incumbent_index: int,
    initial_exact_cache: Mapping[int, ExactCandidateEvaluation],
    training_indices: Sequence[int],
    evaluate_candidate: ExactEvaluator,
    budgets: ClosedLoopBudgets,
    seed: int,
    hard_logic_is_feasible: HardLogicEvaluator,
) -> ClosedLoopRandomResult:
    return _run_random_closed_loop(
        model,
        method="logic_rejection_random",
        admission_policy=LOGIC_REJECTION_RANDOM_ADMISSION_POLICY,
        initial_incumbent_index=initial_incumbent_index,
        initial_exact_cache=initial_exact_cache,
        training_indices=training_indices,
        evaluate_candidate=evaluate_candidate,
        budgets=budgets,
        seed=seed,
        hard_logic_is_feasible=hard_logic_is_feasible,
    )


def _run_random_closed_loop(
    model: QuantizedSparseValueModel,
    *,
    method: str,
    admission_policy: CandidateAdmissionPolicy,
    initial_incumbent_index: int,
    initial_exact_cache: Mapping[int, ExactCandidateEvaluation],
    training_indices: Sequence[int],
    evaluate_candidate: ExactEvaluator,
    budgets: ClosedLoopBudgets,
    seed: int,
    hard_logic_is_feasible: HardLogicEvaluator | None,
) -> ClosedLoopRandomResult:
    """Uniform with-replacement candidate generation; no landscape is accepted."""

    state = create_closed_loop_state(
        initial_incumbent_index=initial_incumbent_index,
        initial_exact_cache=initial_exact_cache,
        training_indices=training_indices,
        num_x_qubits=model.num_x_qubits,
        encode_true_cost=model.fixed_point_config.encode,
        hard_logic_is_feasible=hard_logic_is_feasible,
        budgets=budgets,
        admission_policy=admission_policy,
    )
    rng = np.random.default_rng(int(seed))
    dimension = 2 ** int(model.num_x_qubits)
    trace: list[dict[str, object]] = []
    stop_reason: str | None = None

    while stop_reason is None:
        if state.proposals_used >= int(budgets.max_proposals):
            stop_reason = "max_proposals_reached"
            break
        candidate_index = int(rng.integers(0, dimension))
        bits = _bits_from_index(candidate_index, model.num_x_qubits)
        surrogate_integer_cost = int(model.integer_value(bits))
        incumbent_before = int(state.incumbent_index)
        decision = accept_candidate_proposal(
            state,
            CandidateProposal(
                candidate_index=candidate_index,
                source_method=method,
                auxiliary_accepted=True,
                surrogate_integer_cost=surrogate_integer_cost,
                grover_iterations=0,
                oracle_calls=0,
            ),
            evaluate_candidate=evaluate_candidate,
        )
        stop_reason = _random_stop_reason(decision.stop_reason)
        trace.append(
            {
                "proposal_number": int(state.proposals_used),
                "candidate_index": candidate_index,
                "candidate_bitstring": _bitstring_from_index(
                    candidate_index, model.num_x_qubits
                ),
                "candidate_source_method": method,
                "surrogate_integer_cost": surrogate_integer_cost,
                "grover_iterations": 0,
                "oracle_calls_added": 0,
                "quantum_resources": {
                    "applicable": False,
                    "oracle_calls": 0,
                    "grover_iterations": 0,
                },
                "incumbent_index_before": incumbent_before,
                "incumbent_index_after": int(state.incumbent_index),
                **decision.trace_fields(),
                "stop_reason_after_proposal": stop_reason,
            }
        )

    return ClosedLoopRandomResult(
        method=method,
        seed=int(seed),
        initial_incumbent_index=int(state.initial_incumbent_index),
        initial_incumbent_true_cost=float(state.initial_incumbent_true_cost),
        final_incumbent_index=int(state.incumbent_index),
        final_incumbent_true_cost=float(state.incumbent_true_cost),
        final_encoded_threshold=int(state.encoded_threshold),
        stop_reason=str(stop_reason),
        trial_trace=tuple(trace),
        threshold_history=tuple(state.threshold_history),
        exact_cache=state.exact_cache,
        proposals_used=int(state.proposals_used),
        new_exact_evaluation_attempts=int(state.new_exact_evaluation_attempts),
        actual_ed_lp_solves=int(state.actual_ed_lp_solves),
        logic_precheck_rejections=int(state.logic_precheck_rejections),
        cached_exact_lookups=int(state.cached_exact_lookups),
        threshold_updates=int(state.threshold_updates),
        admission_policy=admission_policy,
        budgets=budgets,
    )


def _random_stop_reason(stop_reason: str | None) -> str | None:
    if stop_reason == "max_new_ed_lp_calls_reached":
        return "max_new_exact_evaluations_reached"
    return stop_reason


def _bits_from_index(index: int, num_qubits: int) -> tuple[int, ...]:
    return tuple((int(index) >> qubit) & 1 for qubit in range(int(num_qubits)))


def _bitstring_from_index(index: int, num_qubits: int) -> str:
    return "".join(str(bit) for bit in _bits_from_index(index, num_qubits))
