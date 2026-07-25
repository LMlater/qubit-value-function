from __future__ import annotations

from qubit_value_function.coherent_phase_value import (
    QuantizedSparseValueModel,
    conservative_integer_bounds,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig
from qubit_value_function.sparse_phase_vqc import build_local_phase_features
from qubit_value_function.sparse_vqc_bbht import (
    BBHTConfig,
    BBHTTrialExecution,
    ExactCandidateEvaluation,
    run_sparse_vqc_bbht,
)


def _model() -> QuantizedSparseValueModel:
    features = build_local_phase_features(2, 2)
    integer_weights = (1, 2, 4, 8) + (0,) * (len(features) - 4)
    lower, upper = conservative_integer_bounds(0, integer_weights)
    return QuantizedSparseValueModel(
        num_generators=2,
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


def _record(cost: float) -> ExactCandidateEvaluation:
    return ExactCandidateEvaluation(True, float(cost), "ok", "new_ed_lp_call")


def _bitstring(index: int) -> str:
    return "".join(str((int(index) >> bit) & 1) for bit in range(4))


class Executor:
    def __init__(self, indices: list[int]) -> None:
        self.indices = list(indices)
        self.calls = 0

    def __call__(self, model, threshold, iterations, shots, seed):
        self.calls += 1
        index = self.indices.pop(0)
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
            auxiliary_zero_probability=1.0,
            total_qubits=model.num_x_qubits + model.num_value_qubits,
            estimated_statevector_memory_gb=0.0,
            elapsed_seconds=0.0,
            circuit_resources={"depth": int(iterations)},
        )


def _initial_cache() -> dict[int, ExactCandidateEvaluation]:
    return {3: ExactCandidateEvaluation(True, 3.0, "ok", "training_exact_cache")}


def test_max_oracle_calls_stops_after_executed_random_iterations_reach_budget() -> None:
    executor = Executor([4, 4])
    result = run_sparse_vqc_bbht(
        _model(),
        initial_incumbent_index=3,
        initial_exact_cache=_initial_cache(),
        evaluate_candidate=lambda index: _record(10.0),
        config=BBHTConfig(max_oracle_calls=1, seed=0),
        trial_executor=executor,
    )
    assert result.stop_reason == "max_oracle_calls_reached"
    assert result.total_oracle_calls == 1
    assert result.circuit_executions == 2


def test_new_ed_lp_budget_stops_before_second_uncached_marked_evaluation() -> None:
    executor = Executor([2, 1])
    calls: list[int] = []

    def evaluate(index: int) -> ExactCandidateEvaluation:
        calls.append(index)
        return _record(4.0)

    result = run_sparse_vqc_bbht(
        _model(),
        initial_incumbent_index=3,
        initial_exact_cache=_initial_cache(),
        evaluate_candidate=evaluate,
        config=BBHTConfig(max_new_ed_lp_calls=1, seed=0),
        trial_executor=executor,
    )
    assert result.stop_reason == "max_new_ed_lp_calls_reached"
    assert result.new_ed_lp_calls == 1
    assert calls == [2]
    assert result.trial_trace[-1]["candidate_status"] == "new_ed_lp_budget_exhausted"


def test_consecutive_nonimproving_marked_budget_stops_repeated_false_positives() -> None:
    result = run_sparse_vqc_bbht(
        _model(),
        initial_incumbent_index=3,
        initial_exact_cache=_initial_cache(),
        evaluate_candidate=lambda index: _record(4.0),
        config=BBHTConfig(max_consecutive_nonimproving_marked=2, seed=0),
        trial_executor=Executor([2, 1]),
    )
    assert result.stop_reason == "max_consecutive_nonimproving_marked_reached"
    assert result.threshold_updates == 0
    assert result.trial_trace[-1]["consecutive_nonimproving_marked"] == 2


def test_threshold_update_budget_stops_immediately_after_allowed_improvement() -> None:
    result = run_sparse_vqc_bbht(
        _model(),
        initial_incumbent_index=3,
        initial_exact_cache=_initial_cache(),
        evaluate_candidate=lambda index: _record(2.0),
        config=BBHTConfig(max_threshold_updates=1, seed=0),
        trial_executor=Executor([2]),
    )
    assert result.stop_reason == "max_threshold_updates_reached"
    assert result.threshold_updates == 1
    assert result.final_incumbent_index == 2
    assert result.trial_trace[0]["threshold_updated"] is True
