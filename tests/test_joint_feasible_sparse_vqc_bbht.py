from __future__ import annotations

from qubit_value_function.coherent_phase_value import (
    QuantizedSparseValueModel,
    conservative_integer_bounds,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig
from qubit_value_function.logic_feasibility_oracle import (
    BooleanLiteral,
    ForbiddenBooleanPattern,
    LogicFeasibilitySpec,
)
from qubit_value_function.sparse_phase_vqc import build_local_phase_features
from qubit_value_function.sparse_vqc_bbht import (
    BBHTConfig,
    BBHTTrialExecution,
    ExactCandidateEvaluation,
    run_sparse_vqc_bbht,
)


def _model() -> QuantizedSparseValueModel:
    features = build_local_phase_features(1, 2, generator_edges=())
    integer_weights = (1, 2) + (0,) * (len(features) - 2)
    lower, upper = conservative_integer_bounds(0, integer_weights)
    return QuantizedSparseValueModel(
        num_generators=1,
        num_periods=2,
        features=features,
        fixed_point_config=FixedPointConfig(fractional_bits=0, unit=1.0),
        real_intercept=0.0,
        real_weights=tuple(float(value) for value in integer_weights),
        integer_intercept=0,
        integer_weights=integer_weights,
        coefficient_quantization_errors=(0.0,) * (1 + len(integer_weights)),
        lower_bound=lower,
        upper_bound=upper,
        value_shift=-lower,
        shifted_upper_bound=upper - lower,
        num_value_qubits=max(1, int(upper - lower).bit_length()),
    )


def _spec() -> LogicFeasibilitySpec:
    return LogicFeasibilitySpec(
        num_generators=1,
        num_periods=2,
        selected_generator_indices=(0,),
        patterns=(
            ForbiddenBooleanPattern(
                literals=(BooleanLiteral(0, 0),),
                label="x0_must_be_one",
            ),
        ),
    )


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


def _bitstring(index: int) -> str:
    return "".join(str((int(index) >> bit) & 1) for bit in range(2))


class Executor:
    def __init__(self, rows: list[tuple[int, float]]) -> None:
        self.rows = list(rows)

    def __call__(self, model, threshold, iterations, shots, seed):
        index, auxiliary_zero = self.rows.pop(0)
        bitstring = _bitstring(index)
        return BBHTTrialExecution(
            measured_index=index,
            measured_bitstring=bitstring,
            measured_count=1,
            measured_probability=1.0,
            shots=1,
            seed=int(seed),
            raw_counts={"0": 1},
            x_counts={bitstring: 1},
            auxiliary_zero_probability=float(auxiliary_zero),
            total_qubits=model.num_x_qubits + model.num_value_qubits,
            estimated_statevector_memory_gb=0.0,
            elapsed_seconds=0.0,
            circuit_resources={"depth": int(iterations)},
        )


def _cache() -> dict[int, ExactCandidateEvaluation]:
    return {3: _record(3.0, source="training_exact_cache", lp_solve=True)}


def test_hard_infeasible_measurement_skips_exact_evaluation_then_feasible_updates() -> None:
    evaluated: list[int] = []

    def evaluate(index: int) -> ExactCandidateEvaluation:
        evaluated.append(index)
        return _record(1.0)

    result = run_sparse_vqc_bbht(
        _model(),
        initial_incumbent_index=3,
        initial_exact_cache=_cache(),
        evaluate_candidate=evaluate,
        feasibility_spec=_spec(),
        config=BBHTConfig(max_threshold_updates=1, seed=0),
        trial_executor=Executor([(0, 1.0), (1, 1.0)]),
    )
    assert evaluated == [1]
    assert result.trial_trace[0]["candidate_status"] == "measured_hard_logic_infeasible"
    assert result.trial_trace[0]["hard_logic_feasible"] is False
    assert result.trial_trace[0]["new_exact_evaluation_attempts_added"] == 0
    assert result.final_incumbent_index == 1
    assert result.threshold_updates == 1
    assert result.actual_ed_lp_solves == 1
    assert result.uses_hard_feasibility_oracle is True


def test_nonzero_auxiliary_syndrome_is_rejected_without_candidate_validation() -> None:
    result = run_sparse_vqc_bbht(
        _model(),
        initial_incumbent_index=3,
        initial_exact_cache=_cache(),
        evaluate_candidate=lambda index: (_ for _ in ()).throw(
            AssertionError("auxiliary syndrome must skip exact validation")
        ),
        feasibility_spec=_spec(),
        config=BBHTConfig(max_trials=1, seed=0),
        trial_executor=Executor([(1, 0.0)]),
    )
    assert result.stop_reason == "max_trials_reached"
    assert result.auxiliary_syndrome_rejections == 1
    assert result.trial_trace[0]["candidate_status"] == "auxiliary_syndrome_rejected"
    assert result.trial_trace[0]["m_after"] == result.trial_trace[0]["m_before"]
    assert result.new_exact_evaluation_attempts == 0


def test_exact_attempt_lp_solve_and_logic_precheck_counts_are_separate() -> None:
    result = run_sparse_vqc_bbht(
        _model(),
        initial_incumbent_index=3,
        initial_exact_cache=_cache(),
        evaluate_candidate=lambda index: _record(
            None,
            success=False,
            source="logic_infeasible_precheck",
            lp_solve=False,
            precheck=True,
        ),
        feasibility_spec=_spec(),
        config=BBHTConfig(max_consecutive_nonimproving_marked=1, seed=0),
        trial_executor=Executor([(1, 1.0)]),
    )
    assert result.stop_reason == "max_consecutive_nonimproving_marked_reached"
    assert result.new_exact_evaluation_attempts == 1
    assert result.new_ed_lp_calls == 1
    assert result.actual_ed_lp_solves == 0
    assert result.logic_precheck_rejections == 1
    assert result.trial_trace[0]["candidate_status"] == "marked_logic_precheck_rejected"


def test_repeated_hit_counter_excludes_first_marked_occurrence() -> None:
    result = run_sparse_vqc_bbht(
        _model(),
        initial_incumbent_index=3,
        initial_exact_cache={
            **_cache(),
            1: _record(4.0, source="training_exact_cache", lp_solve=True),
        },
        evaluate_candidate=lambda index: (_ for _ in ()).throw(
            AssertionError("cached candidate must not be re-evaluated")
        ),
        feasibility_spec=_spec(),
        config=BBHTConfig(max_consecutive_nonimproving_marked=2, seed=0),
        trial_executor=Executor([(1, 1.0), (1, 1.0)]),
    )
    assert result.marked_candidate_hits == {1: 2}
    assert result.repeated_candidate_hits == {1: 1}
    assert result.trial_trace[0]["candidate_repeat_count"] == 0
    assert result.trial_trace[1]["candidate_repeat_count"] == 1
