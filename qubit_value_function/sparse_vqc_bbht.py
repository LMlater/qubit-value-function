from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Callable, Mapping

import numpy as np

from .coherent_phase_value import QuantizedSparseValueModel
from .gate_level_oracle import bitstring_from_index, circuit_resource_summary
from .sparse_vqc_grover import (
    build_sparse_vqc_grover_circuit,
    execute_sparse_vqc_grover_mps,
)


INITIALIZATION_POLICIES = ("first", "random", "best-training")
STOP_REASONS = (
    "no_surrogate_marked_state_by_conservative_lower_bound",
    "max_trials_reached",
    "max_oracle_calls_reached",
    "max_new_ed_lp_calls_reached",
    "max_threshold_updates_reached",
    "max_consecutive_nonimproving_marked_reached",
    "same_encoded_threshold_update_limit",
)


@dataclass(frozen=True)
class BBHTConfig:
    """Budgets and random-window policy for one adaptive BBHT run."""

    lambda_factor: float = 1.2
    max_trials: int = 64
    max_oracle_calls: int = 128
    max_new_ed_lp_calls: int = 16
    max_threshold_updates: int = 8
    max_consecutive_nonimproving_marked: int = 16
    max_same_encoded_threshold_updates: int = 3
    shots_per_trial: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.lambda_factor) or float(self.lambda_factor) <= 1.0:
            raise ValueError("lambda_factor 必须是大于 1 的有限数")
        for name in (
            "max_trials",
            "max_oracle_calls",
            "max_new_ed_lp_calls",
            "max_threshold_updates",
            "max_consecutive_nonimproving_marked",
            "max_same_encoded_threshold_updates",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if int(self.shots_per_trial) != 1:
            raise ValueError("正式 BBHT 循环要求 shots_per_trial=1；多 shots 仅用于普通 Grover 诊断")


@dataclass(frozen=True)
class ExactCandidateEvaluation:
    """Exact ED/LP record cached by commitment index."""

    success: bool
    total_cost: float | None
    message: str
    source: str

    def __post_init__(self) -> None:
        if self.success:
            if self.total_cost is None or not np.isfinite(float(self.total_cost)):
                raise ValueError("成功的 exact evaluation 必须包含有限 total_cost")
        elif self.total_cost is not None and not np.isfinite(float(self.total_cost)):
            raise ValueError("失败 evaluation 的 total_cost 必须为 None 或有限数")

    def as_dict(self) -> dict[str, object]:
        return {
            "success": bool(self.success),
            "total_cost": float(self.total_cost) if self.total_cost is not None else None,
            "message": str(self.message),
            "source": str(self.source),
        }


@dataclass(frozen=True)
class BBHTTrialExecution:
    """One actually executed MPS circuit and its measured search candidate."""

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
    initial_incumbent_index: int
    initial_incumbent_true_cost: float
    final_incumbent_index: int
    final_incumbent_true_cost: float
    final_encoded_threshold: int
    stop_reason: str
    trial_trace: tuple[dict[str, object], ...]
    threshold_history: tuple[dict[str, object], ...]
    exact_cache: dict[int, ExactCandidateEvaluation]
    repeated_candidate_hits: dict[int, int]
    circuit_executions: int
    total_shots: int
    total_oracle_calls: int
    total_diffuser_calls: int
    new_ed_lp_calls: int
    cached_exact_lookups: int
    threshold_updates: int
    same_encoded_threshold_updates: int
    max_m_reached: int
    bbht_m_cap: int
    config: BBHTConfig

    def as_dict(self) -> dict[str, object]:
        return {
            "initial_incumbent_index": int(self.initial_incumbent_index),
            "initial_incumbent_true_cost": float(self.initial_incumbent_true_cost),
            "final_incumbent_index": int(self.final_incumbent_index),
            "final_incumbent_true_cost": float(self.final_incumbent_true_cost),
            "final_encoded_threshold": int(self.final_encoded_threshold),
            "stop_reason": self.stop_reason,
            "trial_trace": list(self.trial_trace),
            "threshold_history": list(self.threshold_history),
            "exact_cache": {
                str(index): record.as_dict() for index, record in sorted(self.exact_cache.items())
            },
            "repeated_candidate_hits": {
                str(index): int(count)
                for index, count in sorted(self.repeated_candidate_hits.items())
            },
            "counters": {
                "circuit_executions": int(self.circuit_executions),
                "total_shots": int(self.total_shots),
                "total_oracle_calls": int(self.total_oracle_calls),
                "total_diffuser_calls": int(self.total_diffuser_calls),
                "new_ed_lp_calls": int(self.new_ed_lp_calls),
                "cached_exact_lookups": int(self.cached_exact_lookups),
                "threshold_updates": int(self.threshold_updates),
                "same_encoded_threshold_updates": int(
                    self.same_encoded_threshold_updates
                ),
            },
            "max_m_reached": int(self.max_m_reached),
            "bbht_m_cap": int(self.bbht_m_cap),
            "config": {
                "lambda_factor": float(self.config.lambda_factor),
                "max_trials": int(self.config.max_trials),
                "max_oracle_calls": int(self.config.max_oracle_calls),
                "max_new_ed_lp_calls": int(self.config.max_new_ed_lp_calls),
                "max_threshold_updates": int(self.config.max_threshold_updates),
                "max_consecutive_nonimproving_marked": int(
                    self.config.max_consecutive_nonimproving_marked
                ),
                "max_same_encoded_threshold_updates": int(
                    self.config.max_same_encoded_threshold_updates
                ),
                "shots_per_trial": int(self.config.shots_per_trial),
                "seed": int(self.config.seed),
            },
            "uses_marked_count": False,
            "uses_validation_enumeration": False,
            "threshold_updates_require_true_ed_lp_improvement": True,
        }


TrialExecutor = Callable[
    [QuantizedSparseValueModel, int, int, int, int],
    BBHTTrialExecution,
]
ExactEvaluator = Callable[[int], ExactCandidateEvaluation]


def select_initial_incumbent(
    training_indices: list[int] | tuple[int, ...],
    exact_cache: Mapping[int, ExactCandidateEvaluation],
    *,
    policy: str = "first",
    seed: int = 0,
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
        rng = np.random.default_rng(int(seed))
        return int(rng.choice(np.asarray(indices, dtype=int)))
    return int(min(indices, key=lambda index: float(exact_cache[index].total_cost)))


def sample_bbht_iterations(rng: np.random.Generator, m: int) -> int:
    m = int(m)
    if m <= 0:
        raise ValueError("BBHT m 必须为正整数")
    return int(rng.integers(0, m))


def grow_bbht_window(m: int, *, lambda_factor: float, m_cap: int) -> int:
    m = int(m)
    m_cap = int(m_cap)
    if m <= 0 or m_cap <= 0:
        raise ValueError("m 和 m_cap 必须为正整数")
    if not np.isfinite(lambda_factor) or float(lambda_factor) <= 1.0:
        raise ValueError("lambda_factor 必须是大于 1 的有限数")
    return int(min(m_cap, max(m + 1, ceil(float(lambda_factor) * m))))


def execute_bbht_trial_mps(
    model: QuantizedSparseValueModel,
    encoded_threshold: int,
    iterations: int,
    shots: int,
    seed: int,
) -> BBHTTrialExecution:
    """Build and execute one actual BBHT trial; no marked-count input is used."""

    if int(shots) != 1:
        raise ValueError("BBHT trial 必须使用单 shot")
    circuit = build_sparse_vqc_grover_circuit(
        model,
        encoded_threshold=int(encoded_threshold),
        iterations=int(iterations),
    )
    execution = execute_sparse_vqc_grover_mps(
        circuit,
        num_x_qubits=model.num_x_qubits,
        shots=1,
        seed=int(seed),
    )
    measured_index, measured_bitstring, measured_count = _select_actual_measurement(
        execution.x_counts,
        model.num_x_qubits,
    )
    return BBHTTrialExecution(
        measured_index=int(measured_index),
        measured_bitstring=measured_bitstring,
        measured_count=int(measured_count),
        measured_probability=float(measured_count / execution.shots),
        shots=int(execution.shots),
        seed=int(seed),
        raw_counts=dict(execution.raw_counts),
        x_counts=dict(execution.x_counts),
        auxiliary_zero_probability=float(execution.auxiliary_zero_probability),
        total_qubits=int(execution.total_qubits),
        estimated_statevector_memory_gb=float(
            execution.estimated_statevector_memory_gb
        ),
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
) -> SparseVQCBBHTResult:
    """Run adaptive BBHT without marked-count enumeration.

    A measured candidate updates the incumbent only when its cached or newly
    evaluated exact ED/LP cost is strictly lower than the current true cost.
    """

    cache = {int(index): record for index, record in initial_exact_cache.items()}
    initial_incumbent_index = int(initial_incumbent_index)
    initial_record = cache.get(initial_incumbent_index)
    if initial_record is None or not initial_record.success or initial_record.total_cost is None:
        raise ValueError("初始 incumbent 必须存在于成功的 exact cache 中")

    executor = trial_executor or execute_bbht_trial_mps
    rng = np.random.default_rng(int(config.seed))
    dimension = 2 ** int(model.num_x_qubits)
    m_cap = max(1, int(ceil(sqrt(float(dimension)))))
    m = 1
    max_m_reached = 1

    incumbent_index = initial_incumbent_index
    incumbent_true_cost = float(initial_record.total_cost)
    encoded_threshold = int(model.fixed_point_config.encode(incumbent_true_cost))
    initial_true_cost = incumbent_true_cost

    threshold_history: list[dict[str, object]] = [
        {
            "update_number": 0,
            "incumbent_index": int(incumbent_index),
            "incumbent_bitstring": bitstring_from_index(
                incumbent_index, model.num_x_qubits
            ),
            "true_threshold": float(incumbent_true_cost),
            "encoded_threshold": int(encoded_threshold),
            "source": "initial_exact_cache",
        }
    ]
    trace: list[dict[str, object]] = []
    repeated_hits: dict[int, int] = {}
    circuit_executions = 0
    total_shots = 0
    total_oracle_calls = 0
    new_ed_lp_calls = 0
    cached_exact_lookups = 0
    threshold_updates = 0
    same_encoded_threshold_updates = 0
    consecutive_nonimproving_marked = 0
    stop_reason: str | None = None

    while stop_reason is None:
        if int(model.lower_bound) >= int(encoded_threshold):
            stop_reason = "no_surrogate_marked_state_by_conservative_lower_bound"
            break
        if circuit_executions >= int(config.max_trials):
            stop_reason = "max_trials_reached"
            break
        if total_oracle_calls >= int(config.max_oracle_calls):
            stop_reason = "max_oracle_calls_reached"
            break
        if threshold_updates >= int(config.max_threshold_updates):
            stop_reason = "max_threshold_updates_reached"
            break

        m_before = int(m)
        sampled_iterations = sample_bbht_iterations(rng, m_before)
        if total_oracle_calls + sampled_iterations > int(config.max_oracle_calls):
            stop_reason = "max_oracle_calls_reached"
            break
        execution_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        execution = executor(
            model,
            int(encoded_threshold),
            int(sampled_iterations),
            int(config.shots_per_trial),
            execution_seed,
        )
        circuit_executions += 1
        total_shots += int(execution.shots)
        total_oracle_calls += int(sampled_iterations)

        candidate_index = int(execution.measured_index)
        if candidate_index < 0 or candidate_index >= dimension:
            raise RuntimeError("trial executor 返回的 measured_index 超出搜索空间")
        bits = _bits_from_index(candidate_index, model.num_x_qubits)
        surrogate_integer_cost = int(model.integer_value(bits))
        surrogate_marked = bool(surrogate_integer_cost < int(encoded_threshold))
        threshold_before = float(incumbent_true_cost)
        encoded_before = int(encoded_threshold)
        incumbent_before = int(incumbent_index)
        m_after = m_before
        verification_source: str | None = None
        exact_record: ExactCandidateEvaluation | None = None
        exact_cost: float | None = None
        true_improvement = False
        encoded_threshold_changed = False
        quantization_stagnation = False
        trial_status = "measured_unmarked"
        ed_lp_calls_added = 0

        if not surrogate_marked:
            m_after = grow_bbht_window(
                m_before,
                lambda_factor=config.lambda_factor,
                m_cap=m_cap,
            )
        else:
            repeated_hits[candidate_index] = repeated_hits.get(candidate_index, 0) + 1
            if candidate_index in cache:
                cached_exact_lookups += 1
                exact_record = cache[candidate_index]
                verification_source = str(exact_record.source)
            else:
                if new_ed_lp_calls >= int(config.max_new_ed_lp_calls):
                    trial_status = "new_ed_lp_budget_exhausted"
                    stop_reason = "max_new_ed_lp_calls_reached"
                else:
                    try:
                        exact_record = evaluate_candidate(candidate_index)
                    except Exception as exc:  # pragma: no cover - defensive integration path
                        exact_record = ExactCandidateEvaluation(
                            success=False,
                            total_cost=None,
                            message=f"exact evaluator raised: {exc}",
                            source="new_ed_lp_call",
                        )
                    if not isinstance(exact_record, ExactCandidateEvaluation):
                        raise TypeError("evaluate_candidate 必须返回 ExactCandidateEvaluation")
                    cache[candidate_index] = exact_record
                    new_ed_lp_calls += 1
                    ed_lp_calls_added = 1
                    verification_source = str(exact_record.source)

            if exact_record is not None:
                if exact_record.success and exact_record.total_cost is not None:
                    exact_cost = float(exact_record.total_cost)
                    true_improvement = bool(exact_cost < incumbent_true_cost)
                if true_improvement:
                    incumbent_index = candidate_index
                    incumbent_true_cost = float(exact_cost)
                    encoded_threshold = int(
                        model.fixed_point_config.encode(incumbent_true_cost)
                    )
                    encoded_threshold_changed = bool(encoded_threshold != encoded_before)
                    quantization_stagnation = not encoded_threshold_changed
                    threshold_updates += 1
                    if encoded_threshold_changed:
                        same_encoded_threshold_updates = 0
                    else:
                        same_encoded_threshold_updates += 1
                    consecutive_nonimproving_marked = 0
                    m_after = 1
                    trial_status = "true_incumbent_improved"
                    threshold_history.append(
                        {
                            "update_number": int(threshold_updates),
                            "trial_number": int(circuit_executions),
                            "incumbent_index": int(incumbent_index),
                            "incumbent_bitstring": bitstring_from_index(
                                incumbent_index, model.num_x_qubits
                            ),
                            "true_threshold": float(incumbent_true_cost),
                            "encoded_threshold": int(encoded_threshold),
                            "encoded_threshold_changed": bool(
                                encoded_threshold_changed
                            ),
                            "verification_source": verification_source,
                        }
                    )
                    if (
                        same_encoded_threshold_updates
                        >= int(config.max_same_encoded_threshold_updates)
                    ):
                        stop_reason = "same_encoded_threshold_update_limit"
                    elif threshold_updates >= int(config.max_threshold_updates):
                        stop_reason = "max_threshold_updates_reached"
                else:
                    consecutive_nonimproving_marked += 1
                    m_after = grow_bbht_window(
                        m_before,
                        lambda_factor=config.lambda_factor,
                        m_cap=m_cap,
                    )
                    if exact_record.success:
                        trial_status = "marked_but_not_true_improvement"
                    else:
                        trial_status = "marked_exact_evaluation_failed"
                    if (
                        consecutive_nonimproving_marked
                        >= int(config.max_consecutive_nonimproving_marked)
                    ):
                        stop_reason = "max_consecutive_nonimproving_marked_reached"

        m = int(m_after)
        max_m_reached = max(max_m_reached, m)
        trace.append(
            {
                "trial_number": int(circuit_executions),
                "m_before": int(m_before),
                "sampled_grover_iterations": int(sampled_iterations),
                "m_after": int(m_after),
                "execution_seed": int(execution_seed),
                "shots": int(execution.shots),
                "oracle_calls_added": int(sampled_iterations),
                "measured_index": int(candidate_index),
                "measured_bitstring": execution.measured_bitstring,
                "measured_count": int(execution.measured_count),
                "measured_probability": float(execution.measured_probability),
                "raw_counts": execution.raw_counts,
                "x_counts": execution.x_counts,
                "auxiliary_zero_probability": float(
                    execution.auxiliary_zero_probability
                ),
                "surrogate_integer_cost": int(surrogate_integer_cost),
                "surrogate_marked": bool(surrogate_marked),
                "candidate_status": trial_status,
                "candidate_cache_hit": bool(
                    surrogate_marked and candidate_index in initial_exact_cache
                    or surrogate_marked and ed_lp_calls_added == 0 and exact_record is not None
                ),
                "candidate_repeat_count": int(repeated_hits.get(candidate_index, 0)),
                "verification_source": verification_source,
                "exact_evaluation": (
                    exact_record.as_dict() if exact_record is not None else None
                ),
                "exact_cost": exact_cost,
                "ed_lp_calls_added": int(ed_lp_calls_added),
                "true_improvement": bool(true_improvement),
                "incumbent_index_before": int(incumbent_before),
                "incumbent_index_after": int(incumbent_index),
                "true_threshold_before": float(threshold_before),
                "true_threshold_after": float(incumbent_true_cost),
                "encoded_threshold_before": int(encoded_before),
                "encoded_threshold_after": int(encoded_threshold),
                "encoded_threshold_changed": bool(encoded_threshold_changed),
                "quantization_stagnation": bool(quantization_stagnation),
                "threshold_updated": bool(true_improvement),
                "consecutive_nonimproving_marked": int(
                    consecutive_nonimproving_marked
                ),
                "elapsed_seconds": float(execution.elapsed_seconds),
                "total_qubits": int(execution.total_qubits),
                "estimated_statevector_memory_gb": float(
                    execution.estimated_statevector_memory_gb
                ),
                "circuit_resources": execution.circuit_resources,
                "stop_reason_after_trial": stop_reason,
            }
        )

    if stop_reason not in STOP_REASONS:
        raise RuntimeError("BBHT 结束时缺少受支持的 stop_reason")
    return SparseVQCBBHTResult(
        initial_incumbent_index=int(initial_incumbent_index),
        initial_incumbent_true_cost=float(initial_true_cost),
        final_incumbent_index=int(incumbent_index),
        final_incumbent_true_cost=float(incumbent_true_cost),
        final_encoded_threshold=int(encoded_threshold),
        stop_reason=stop_reason,
        trial_trace=tuple(trace),
        threshold_history=tuple(threshold_history),
        exact_cache=cache,
        repeated_candidate_hits=repeated_hits,
        circuit_executions=int(circuit_executions),
        total_shots=int(total_shots),
        total_oracle_calls=int(total_oracle_calls),
        total_diffuser_calls=int(total_oracle_calls),
        new_ed_lp_calls=int(new_ed_lp_calls),
        cached_exact_lookups=int(cached_exact_lookups),
        threshold_updates=int(threshold_updates),
        same_encoded_threshold_updates=int(same_encoded_threshold_updates),
        max_m_reached=int(max_m_reached),
        bbht_m_cap=int(m_cap),
        config=config,
    )


def _select_actual_measurement(
    x_counts: Mapping[str, int],
    num_x_qubits: int,
) -> tuple[int, str, int]:
    rows: list[tuple[int, str, int]] = []
    for bitstring, raw_count in x_counts.items():
        count = int(raw_count)
        if count <= 0:
            continue
        compact = str(bitstring).replace(" ", "")
        if len(compact) != int(num_x_qubits) or set(compact) - {"0", "1"}:
            raise ValueError("x_counts 中的 bitstring 格式不正确")
        index = int(sum(int(bit) << offset for offset, bit in enumerate(compact)))
        rows.append((index, compact, count))
    if not rows:
        raise RuntimeError("BBHT trial 没有返回实际搜索寄存器测量")
    return max(rows, key=lambda row: (row[2], -row[0]))


def _bits_from_index(index: int, num_qubits: int) -> tuple[int, ...]:
    return tuple((int(index) >> qubit) & 1 for qubit in range(int(num_qubits)))
