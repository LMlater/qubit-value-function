from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Callable, Mapping, Sequence

import numpy as np

from .candidate_acceptance_loop import (
    CandidateProposal,
    CandidateAdmissionPolicy,
    ClosedLoopBudgets,
    ExactCandidateEvaluation,
    ExactEvaluator,
    JOINT_BBHT_ADMISSION_POLICY,
    accept_candidate_proposal,
    create_closed_loop_state,
)
from .coherent_phase_value import QuantizedSparseValueModel
from .gate_level_oracle import circuit_resource_summary
from .logic_feasibility_oracle import LogicFeasibilitySpec
from .sparse_vqc_grover import (
    build_sparse_vqc_grover_circuit,
    execute_sparse_vqc_grover_mps,
)


INITIALIZATION_POLICIES = ("first", "random", "best-training")
STOP_REASONS = (
    "no_hard_logic_feasible_state_by_compiled_constraints",
    "no_surrogate_marked_state_by_conservative_lower_bound",
    "max_trials_reached",
    "max_oracle_calls_reached",
    "max_new_ed_lp_calls_reached",
    "max_actual_ed_lp_solves_reached",
    "max_threshold_updates_reached",
    "max_consecutive_nonimproving_marked_reached",
    "same_encoded_threshold_update_limit",
    "max_auxiliary_syndrome_rejections_reached",
)


@dataclass(frozen=True)
class BBHTConfig:
    """Budgets, random-window policy, and auxiliary-syndrome acceptance rule."""

    lambda_factor: float = 1.2
    max_trials: int = 64
    max_oracle_calls: int = 128
    max_new_ed_lp_calls: int = 16
    max_threshold_updates: int = 8
    max_consecutive_nonimproving_marked: int = 16
    max_same_encoded_threshold_updates: int = 3
    max_auxiliary_syndrome_rejections: int = 8
    shots_per_trial: int = 1
    minimum_auxiliary_zero_probability: float = 1.0 - 1e-12
    seed: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.lambda_factor) or float(self.lambda_factor) <= 1.0:
            raise ValueError("lambda_factor 必须是大于 1 的有限数")
        for name in (
            "max_trials", "max_oracle_calls", "max_new_ed_lp_calls",
            "max_threshold_updates", "max_consecutive_nonimproving_marked",
            "max_same_encoded_threshold_updates", "max_auxiliary_syndrome_rejections",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if int(self.shots_per_trial) != 1:
            raise ValueError("正式 BBHT 循环要求 shots_per_trial=1；多 shots 仅用于普通 Grover 诊断")
        minimum = float(self.minimum_auxiliary_zero_probability)
        if not np.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            raise ValueError("minimum_auxiliary_zero_probability 必须位于 [0, 1]")


@dataclass(frozen=True)
class BBHTTrialExecution:
    measured_index: int
    measured_bitstring: str
    measured_count: int
    measured_probability: float
    shots: int
    seed: int
    raw_counts: dict[str, int]
    x_counts: dict[str, int]
    auxiliary_zero_probability: float
    total_qubits: int
    estimated_statevector_memory_gb: float
    elapsed_seconds: float
    circuit_resources: dict[str, object]


@dataclass(frozen=True)
class SparseVQCBBHTResult:
    method: str
    initial_incumbent_index: int
    initial_incumbent_true_cost: float
    final_incumbent_index: int
    final_incumbent_true_cost: float
    final_encoded_threshold: int
    stop_reason: str
    trial_trace: tuple[dict[str, object], ...]
    threshold_history: tuple[dict[str, object], ...]
    exact_cache: dict[int, ExactCandidateEvaluation]
    marked_candidate_hits: dict[int, int]
    repeated_candidate_hits: dict[int, int]
    circuit_executions: int
    total_shots: int
    total_oracle_calls: int
    total_diffuser_calls: int
    new_exact_evaluation_attempts: int
    actual_ed_lp_solves: int
    logic_precheck_rejections: int
    new_ed_lp_calls: int
    cached_exact_lookups: int
    auxiliary_syndrome_rejections: int
    threshold_updates: int
    same_encoded_threshold_updates: int
    max_m_reached: int
    bbht_m_cap: int
    uses_hard_feasibility_oracle: bool
    admission_policy: CandidateAdmissionPolicy
    config: BBHTConfig

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "initial_incumbent_index": int(self.initial_incumbent_index),
            "initial_incumbent_true_cost": float(self.initial_incumbent_true_cost),
            "final_incumbent_index": int(self.final_incumbent_index),
            "final_incumbent_true_cost": float(self.final_incumbent_true_cost),
            "final_encoded_threshold": int(self.final_encoded_threshold),
            "stop_reason": self.stop_reason,
            "trial_trace": list(self.trial_trace),
            "threshold_history": list(self.threshold_history),
            "exact_cache": {str(index): record.as_dict() for index, record in sorted(self.exact_cache.items())},
            "marked_candidate_hits": {str(index): int(count) for index, count in sorted(self.marked_candidate_hits.items())},
            "repeated_candidate_hits": {str(index): int(count) for index, count in sorted(self.repeated_candidate_hits.items())},
            "counters": {
                "circuit_executions": int(self.circuit_executions),
                "total_shots": int(self.total_shots),
                "total_oracle_calls": int(self.total_oracle_calls),
                "total_diffuser_calls": int(self.total_diffuser_calls),
                "new_exact_evaluation_attempts": int(self.new_exact_evaluation_attempts),
                "actual_ed_lp_solves": int(self.actual_ed_lp_solves),
                "logic_precheck_rejections": int(self.logic_precheck_rejections),
                "new_ed_lp_calls": int(self.new_ed_lp_calls),
                "cached_exact_lookups": int(self.cached_exact_lookups),
                "auxiliary_syndrome_rejections": int(self.auxiliary_syndrome_rejections),
                "threshold_updates": int(self.threshold_updates),
                "same_encoded_threshold_updates": int(self.same_encoded_threshold_updates),
            },
            "max_m_reached": int(self.max_m_reached),
            "bbht_m_cap": int(self.bbht_m_cap),
            "config": self.config.__dict__.copy(),
            "uses_marked_count": False,
            "uses_validation_enumeration": False,
            "uses_hard_feasibility_oracle": bool(self.uses_hard_feasibility_oracle),
            "admission_policy": {
                "name": self.admission_policy.name,
                "require_auxiliary_accepted": self.admission_policy.require_auxiliary_accepted,
                "require_hard_logic_feasible": self.admission_policy.require_hard_logic_feasible,
                "require_surrogate_better": self.admission_policy.require_surrogate_better,
            },
            "threshold_updates_require_true_ed_lp_improvement": True,
            "legacy_new_ed_lp_calls_semantics": "alias of new_exact_evaluation_attempts; use actual_ed_lp_solves for physical LP-solve count",
        }


TrialExecutor = Callable[[QuantizedSparseValueModel, int, int, int, int], BBHTTrialExecution]


def select_initial_incumbent(
    training_indices: list[int] | tuple[int, ...],
    exact_cache: Mapping[int, ExactCandidateEvaluation],
    *, policy: str = "first", seed: int = 0,
) -> int:
    indices = [int(index) for index in training_indices]
    if not indices:
        raise ValueError("training_indices 不能为空")
    if policy not in INITIALIZATION_POLICIES:
        raise ValueError(f"initialization_policy 必须是 {INITIALIZATION_POLICIES} 之一")
    for index in indices:
        record = exact_cache.get(index)
        if record is None or not record.success or record.total_cost is None:
            raise ValueError("每个训练索引都必须具有成功的 exact ED/LP cache")
    if policy == "first":
        return indices[0]
    if policy == "random":
        return int(np.random.default_rng(int(seed)).choice(np.asarray(indices, dtype=int)))
    return int(min(indices, key=lambda index: float(exact_cache[index].total_cost)))


def sample_bbht_iterations(rng: np.random.Generator, m: int) -> int:
    if int(m) <= 0:
        raise ValueError("BBHT m 必须为正整数")
    return int(rng.integers(0, int(m)))


def grow_bbht_window(m: int, *, lambda_factor: float, m_cap: int) -> int:
    if int(m) <= 0 or int(m_cap) <= 0:
        raise ValueError("m 和 m_cap 必须为正整数")
    if not np.isfinite(lambda_factor) or float(lambda_factor) <= 1.0:
        raise ValueError("lambda_factor 必须是大于 1 的有限数")
    return int(min(int(m_cap), max(int(m) + 1, ceil(float(lambda_factor) * int(m)))))


def execute_bbht_trial_mps(
    model: QuantizedSparseValueModel, encoded_threshold: int, iterations: int,
    shots: int, seed: int, *, feasibility_spec: LogicFeasibilitySpec | None = None,
) -> BBHTTrialExecution:
    if int(shots) != 1:
        raise ValueError("BBHT trial 必须使用单 shot")
    circuit = build_sparse_vqc_grover_circuit(
        model, encoded_threshold=int(encoded_threshold), iterations=int(iterations),
        feasibility_spec=feasibility_spec,
    )
    execution = execute_sparse_vqc_grover_mps(circuit, num_x_qubits=model.num_x_qubits, shots=1, seed=int(seed))
    measured_index, measured_bitstring, measured_count = _select_actual_measurement(execution.x_counts, model.num_x_qubits)
    return BBHTTrialExecution(
        measured_index=int(measured_index), measured_bitstring=measured_bitstring,
        measured_count=int(measured_count), measured_probability=float(measured_count / execution.shots),
        shots=int(execution.shots), seed=int(seed), raw_counts=dict(execution.raw_counts),
        x_counts=dict(execution.x_counts), auxiliary_zero_probability=float(execution.auxiliary_zero_probability),
        total_qubits=int(execution.total_qubits),
        estimated_statevector_memory_gb=float(execution.estimated_statevector_memory_gb),
        elapsed_seconds=float(execution.elapsed_seconds),
        circuit_resources=circuit_resource_summary(circuit, decompose_reps=1),
    )


def run_sparse_vqc_bbht(
    model: QuantizedSparseValueModel,
    *,
    initial_incumbent_index: int,
    initial_exact_cache: Mapping[int, ExactCandidateEvaluation],
    evaluate_candidate: ExactEvaluator,
    config: BBHTConfig = BBHTConfig(),
    trial_executor: TrialExecutor | None = None,
    feasibility_spec: LogicFeasibilitySpec | None = None,
    training_indices: Sequence[int] | None = None,
    method: str = "joint_bbht",
    admission_policy: CandidateAdmissionPolicy = JOINT_BBHT_ADMISSION_POLICY,
) -> SparseVQCBBHTResult:
    """Run BBHT; all candidate acceptance occurs in the shared closed loop."""

    initial_index = int(initial_incumbent_index)
    if feasibility_spec is not None:
        if feasibility_spec.num_x_qubits != model.num_x_qubits:
            raise ValueError("feasibility spec 与 value model 的 search register 不一致")
        if not feasibility_spec.always_infeasible and not feasibility_spec.is_feasible(_bits_from_index(initial_index, model.num_x_qubits)):
            raise ValueError("初始 incumbent 不满足 hard logic-feasibility spec")
    state = create_closed_loop_state(
        initial_incumbent_index=initial_index,
        initial_exact_cache=initial_exact_cache,
        training_indices=(tuple(initial_exact_cache) if training_indices is None else training_indices),
        num_x_qubits=model.num_x_qubits,
        encode_true_cost=model.fixed_point_config.encode,
        hard_logic_is_feasible=(None if feasibility_spec is None else feasibility_spec.is_feasible),
        budgets=ClosedLoopBudgets(
            max_proposals=config.max_trials,
            max_new_exact_evaluations=config.max_new_ed_lp_calls,
            max_actual_ed_lp_solves=None,
            max_threshold_updates=config.max_threshold_updates,
            max_consecutive_nonimproving_marked=config.max_consecutive_nonimproving_marked,
            max_same_encoded_threshold_updates=config.max_same_encoded_threshold_updates,
            max_auxiliary_syndrome_rejections=config.max_auxiliary_syndrome_rejections,
        ),
        admission_policy=admission_policy,
    )
    if trial_executor is None:
        def executor(value_model: QuantizedSparseValueModel, threshold: int, iterations: int, shots: int, seed: int) -> BBHTTrialExecution:
            return execute_bbht_trial_mps(value_model, threshold, iterations, shots, seed, feasibility_spec=feasibility_spec)
    else:
        executor = trial_executor

    rng = np.random.default_rng(int(config.seed))
    dimension = 2 ** int(model.num_x_qubits)
    m_cap = max(1, int(ceil(sqrt(float(dimension)))))
    m = 1
    max_m_reached = 1
    circuit_executions = 0
    total_shots = 0
    total_oracle_calls = 0
    trace: list[dict[str, object]] = []
    stop_reason: str | None = (
        "no_hard_logic_feasible_state_by_compiled_constraints"
        if feasibility_spec is not None and feasibility_spec.always_infeasible else None
    )

    while stop_reason is None:
        if int(model.lower_bound) >= int(state.encoded_threshold):
            stop_reason = "no_surrogate_marked_state_by_conservative_lower_bound"
            break
        if circuit_executions >= int(config.max_trials):
            stop_reason = "max_trials_reached"
            break
        if total_oracle_calls >= int(config.max_oracle_calls):
            stop_reason = "max_oracle_calls_reached"
            break
        m_before = int(m)
        sampled_iterations = sample_bbht_iterations(rng, m_before)
        if total_oracle_calls + sampled_iterations > int(config.max_oracle_calls):
            stop_reason = "max_oracle_calls_reached"
            break
        execution_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        execution = executor(model, int(state.encoded_threshold), sampled_iterations, int(config.shots_per_trial), execution_seed)
        circuit_executions += 1
        total_shots += int(execution.shots)
        total_oracle_calls += int(sampled_iterations)
        candidate_index = int(execution.measured_index)
        if candidate_index < 0 or candidate_index >= dimension:
            raise RuntimeError("trial executor 返回的 measured_index 超出搜索空间")
        surrogate_integer_cost = int(model.integer_value(_bits_from_index(candidate_index, model.num_x_qubits)))
        auxiliary_accepted = bool(float(execution.auxiliary_zero_probability) >= float(config.minimum_auxiliary_zero_probability))
        incumbent_before = int(state.incumbent_index)
        decision = accept_candidate_proposal(
            state,
            CandidateProposal(
                candidate_index=candidate_index,
                source_method=method,
                auxiliary_accepted=auxiliary_accepted,
                surrogate_integer_cost=surrogate_integer_cost,
                grover_iterations=sampled_iterations,
                oracle_calls=sampled_iterations,
            ),
            evaluate_candidate=evaluate_candidate,
        )
        if decision.reset_window:
            m_after = 1
        elif decision.should_grow_window:
            m_after = grow_bbht_window(m_before, lambda_factor=config.lambda_factor, m_cap=m_cap)
        else:
            m_after = m_before
        m = int(m_after)
        max_m_reached = max(max_m_reached, m)
        stop_reason = decision.stop_reason
        shared_trace = decision.trace_fields()
        shared_trace["auxiliary_accepted"] = auxiliary_accepted
        trace.append({
            "trial_number": circuit_executions, "m_before": m_before,
            "sampled_grover_iterations": sampled_iterations, "m_after": m_after,
            "execution_seed": execution_seed, "shots": int(execution.shots),
            "oracle_calls_added": sampled_iterations, "measured_index": candidate_index,
            "candidate_source_method": method,
            "measured_bitstring": execution.measured_bitstring,
            "measured_count": int(execution.measured_count),
            "measured_probability": float(execution.measured_probability),
            "raw_counts": execution.raw_counts, "x_counts": execution.x_counts,
            "auxiliary_zero_probability": float(execution.auxiliary_zero_probability),
            "surrogate_integer_cost": surrogate_integer_cost,
            "incumbent_index_before": incumbent_before,
            "incumbent_index_after": int(state.incumbent_index),
            "consecutive_nonimproving_marked": int(state.consecutive_nonimproving_marked),
            "auxiliary_syndrome_rejections_total": int(state.auxiliary_syndrome_rejections),
            "elapsed_seconds": float(execution.elapsed_seconds),
            "total_qubits": int(execution.total_qubits),
            "estimated_statevector_memory_gb": float(execution.estimated_statevector_memory_gb),
            "circuit_resources": execution.circuit_resources,
            **shared_trace,
            "stop_reason_after_trial": stop_reason,
        })

    if stop_reason not in STOP_REASONS:
        raise RuntimeError("BBHT 结束时缺少受支持的 stop_reason")
    return SparseVQCBBHTResult(
        method=str(method),
        initial_incumbent_index=state.initial_incumbent_index,
        initial_incumbent_true_cost=state.initial_incumbent_true_cost,
        final_incumbent_index=state.incumbent_index,
        final_incumbent_true_cost=state.incumbent_true_cost,
        final_encoded_threshold=state.encoded_threshold,
        stop_reason=stop_reason, trial_trace=tuple(trace),
        threshold_history=tuple(state.threshold_history), exact_cache=state.exact_cache,
        marked_candidate_hits=state.marked_candidate_hits,
        repeated_candidate_hits=state.repeated_candidate_hits,
        circuit_executions=circuit_executions, total_shots=total_shots,
        total_oracle_calls=total_oracle_calls, total_diffuser_calls=total_oracle_calls,
        new_exact_evaluation_attempts=state.new_exact_evaluation_attempts,
        actual_ed_lp_solves=state.actual_ed_lp_solves,
        logic_precheck_rejections=state.logic_precheck_rejections,
        new_ed_lp_calls=state.new_exact_evaluation_attempts,
        cached_exact_lookups=state.cached_exact_lookups,
        auxiliary_syndrome_rejections=state.auxiliary_syndrome_rejections,
        threshold_updates=state.threshold_updates,
        same_encoded_threshold_updates=state.same_encoded_threshold_updates,
        max_m_reached=max_m_reached, bbht_m_cap=m_cap,
        uses_hard_feasibility_oracle=bool(feasibility_spec is not None),
        admission_policy=admission_policy,
        config=config,
    )


def _select_actual_measurement(x_counts: Mapping[str, int], num_x_qubits: int) -> tuple[int, str, int]:
    rows: list[tuple[int, str, int]] = []
    for bitstring, raw_count in x_counts.items():
        count = int(raw_count)
        compact = str(bitstring).replace(" ", "")
        if count <= 0:
            continue
        if len(compact) != int(num_x_qubits) or set(compact) - {"0", "1"}:
            raise ValueError("x_counts 中的 bitstring 格式不正确")
        index = int(sum(int(bit) << offset for offset, bit in enumerate(compact)))
        rows.append((index, compact, count))
    if not rows:
        raise RuntimeError("BBHT trial 没有返回实际搜索寄存器测量")
    return max(rows, key=lambda row: (row[2], -row[0]))


def _bits_from_index(index: int, num_qubits: int) -> tuple[int, ...]:
    return tuple((int(index) >> qubit) & 1 for qubit in range(int(num_qubits)))
