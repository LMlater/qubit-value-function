from __future__ import annotations

import inspect

import numpy as np
import pytest

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
    execute_bbht_trial_mps,
    grow_bbht_window,
    run_sparse_vqc_bbht,
    sample_bbht_iterations,
    select_initial_incumbent,
)
from qubit_value_function.sparse_vqc_grover import AerSimulator


def _binary_value_model(
    num_generators: int = 2,
    num_periods: int = 2,
    *,
    config: FixedPointConfig | None = None,
) -> QuantizedSparseValueModel:
    features = build_local_phase_features(num_generators, num_periods)
    num_x_qubits = num_generators * num_periods
    linear_weights = tuple(2**index for index in range(num_x_qubits))
    integer_weights = linear_weights + (0,) * (len(features) - num_x_qubits)
    lower, upper = conservative_integer_bounds(0, integer_weights)
    return QuantizedSparseValueModel(
        num_generators=num_generators,
        num_periods=num_periods,
        features=features,
        fixed_point_config=config or FixedPointConfig(fractional_bits=0, unit=1.0),
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


def _evaluation(cost: float, source: str = "new_ed_lp_call") -> ExactCandidateEvaluation:
    return ExactCandidateEvaluation(
        success=True,
        total_cost=float(cost),
        message="ok",
        source=source,
    )


def _bitstring(index: int, num_bits: int = 4) -> str:
    return "".join(str((int(index) >> bit) & 1) for bit in range(num_bits))


class SequenceExecutor:
    def __init__(self, indices: list[int], num_bits: int = 4) -> None:
        self.indices = list(indices)
        self.num_bits = int(num_bits)
        self.calls: list[tuple[int, int, int]] = []

    def __call__(self, model, threshold, iterations, shots, seed):
        assert shots == 1
        index = self.indices.pop(0)
        bitstring = _bitstring(index, self.num_bits)
        self.calls.append((int(threshold), int(iterations), int(seed)))
        return BBHTTrialExecution(
            measured_index=index,
            measured_bitstring=bitstring,
            measured_count=1,
            measured_probability=1.0,
            shots=1,
            seed=int(seed),
            raw_counts={"0" * (model.num_value_qubits + model.num_x_qubits): 1},
            x_counts={bitstring: 1},
            auxiliary_zero_probability=1.0,
            total_qubits=model.num_x_qubits + model.num_value_qubits,
            estimated_statevector_memory_gb=0.0,
            elapsed_seconds=0.0,
            circuit_resources={"num_qubits": model.num_x_qubits + model.num_value_qubits},
        )


def test_bbht_sampling_window_and_single_shot_configuration() -> None:
    rng = np.random.default_rng(11)
    samples = [sample_bbht_iterations(rng, 4) for _ in range(50)]
    assert all(0 <= sample < 4 for sample in samples)
    assert grow_bbht_window(1, lambda_factor=1.2, m_cap=4) == 2
    assert grow_bbht_window(2, lambda_factor=1.2, m_cap=4) == 3
    assert grow_bbht_window(4, lambda_factor=1.2, m_cap=4) == 4
    with pytest.raises(ValueError, match="shots_per_trial=1"):
        BBHTConfig(shots_per_trial=2)


def test_initial_incumbent_policies_use_only_exact_training_cache() -> None:
    cache = {0: _evaluation(30.0, "training_exact_cache"), 3: _evaluation(20.0, "training_exact_cache")}
    assert select_initial_incumbent((0, 3), cache, policy="first") == 0
    assert select_initial_incumbent((0, 3), cache, policy="best-training") == 3
    assert select_initial_incumbent((0, 3), cache, policy="random", seed=5) in (0, 3)


def test_true_ed_improvements_repeatedly_shrink_threshold_until_lower_bound_stop() -> None:
    model = _binary_value_model()
    executor = SequenceExecutor([2, 1, 0])
    exact_costs = {2: 2.0, 1: 1.0, 0: 0.0}
    result = run_sparse_vqc_bbht(
        model,
        initial_incumbent_index=3,
        initial_exact_cache={3: _evaluation(3.0, "training_exact_cache")},
        evaluate_candidate=lambda index: _evaluation(exact_costs[index]),
        config=BBHTConfig(seed=7),
        trial_executor=executor,
    )
    assert result.final_incumbent_index == 0
    assert result.final_incumbent_true_cost == 0.0
    assert result.final_encoded_threshold == 0
    assert result.threshold_updates == 3
    assert [row["encoded_threshold"] for row in result.threshold_history] == [3, 2, 1, 0]
    assert result.stop_reason == "no_surrogate_marked_state_by_conservative_lower_bound"
    assert all(row["threshold_updated"] for row in result.trial_trace)


def test_marked_false_positive_does_not_update_threshold_and_grows_m() -> None:
    model = _binary_value_model()
    executor = SequenceExecutor([2, 1, 0])
    exact_costs = {2: 4.0, 1: 2.0, 0: 0.0}
    result = run_sparse_vqc_bbht(
        model,
        initial_incumbent_index=3,
        initial_exact_cache={3: _evaluation(3.0, "training_exact_cache")},
        evaluate_candidate=lambda index: _evaluation(exact_costs[index]),
        config=BBHTConfig(seed=3),
        trial_executor=executor,
    )
    first = result.trial_trace[0]
    assert first["surrogate_marked"] is True
    assert first["true_improvement"] is False
    assert first["threshold_updated"] is False
    assert first["encoded_threshold_before"] == first["encoded_threshold_after"] == 3
    assert first["m_after"] > first["m_before"]
    assert result.final_incumbent_index == 0


def test_cached_exact_candidate_updates_without_new_ed_lp_call() -> None:
    model = _binary_value_model()
    result = run_sparse_vqc_bbht(
        model,
        initial_incumbent_index=3,
        initial_exact_cache={
            3: _evaluation(3.0, "training_exact_cache"),
            0: _evaluation(0.0, "training_exact_cache"),
        },
        evaluate_candidate=lambda index: pytest.fail("cached candidate must not call ED/LP"),
        config=BBHTConfig(seed=1),
        trial_executor=SequenceExecutor([0]),
    )
    assert result.final_incumbent_index == 0
    assert result.new_ed_lp_calls == 0
    assert result.cached_exact_lookups == 1
    assert result.trial_trace[0]["verification_source"] == "training_exact_cache"


def test_unmarked_measurement_skips_ed_lp_and_trial_budget_stops() -> None:
    model = _binary_value_model()
    result = run_sparse_vqc_bbht(
        model,
        initial_incumbent_index=3,
        initial_exact_cache={3: _evaluation(3.0, "training_exact_cache")},
        evaluate_candidate=lambda index: pytest.fail("unmarked state must not call ED/LP"),
        config=BBHTConfig(max_trials=1, seed=2),
        trial_executor=SequenceExecutor([4]),
    )
    assert result.stop_reason == "max_trials_reached"
    assert result.new_ed_lp_calls == 0
    assert result.trial_trace[0]["candidate_status"] == "measured_unmarked"
    assert result.trial_trace[0]["m_after"] == 2


def test_same_encoded_threshold_improvements_trigger_quantization_stagnation_stop() -> None:
    model = _binary_value_model(config=FixedPointConfig(fractional_bits=0, unit=1.0))
    executor = SequenceExecutor([2, 1])
    costs = {2: 3.2, 1: 3.1}
    result = run_sparse_vqc_bbht(
        model,
        initial_incumbent_index=3,
        initial_exact_cache={3: _evaluation(3.4, "training_exact_cache")},
        evaluate_candidate=lambda index: _evaluation(costs[index]),
        config=BBHTConfig(max_same_encoded_threshold_updates=2, seed=4),
        trial_executor=executor,
    )
    assert result.final_incumbent_index == 1
    assert result.final_encoded_threshold == 3
    assert result.same_encoded_threshold_updates == 2
    assert result.stop_reason == "same_encoded_threshold_update_limit"
    assert all(row["quantization_stagnation"] for row in result.trial_trace)


def test_conservative_lower_bound_stops_before_any_circuit_execution() -> None:
    model = _binary_value_model()
    result = run_sparse_vqc_bbht(
        model,
        initial_incumbent_index=0,
        initial_exact_cache={0: _evaluation(0.0, "training_exact_cache")},
        evaluate_candidate=lambda index: pytest.fail("no evaluation expected"),
        config=BBHTConfig(),
        trial_executor=lambda *args: pytest.fail("no circuit execution expected"),
    )
    assert result.stop_reason == "no_surrogate_marked_state_by_conservative_lower_bound"
    assert result.circuit_executions == 0
    assert result.total_oracle_calls == 0


def test_same_seed_preserves_bbht_trace_after_shared_acceptance_refactor() -> None:
    def run_once():
        return run_sparse_vqc_bbht(
            _binary_value_model(),
            initial_incumbent_index=3,
            initial_exact_cache={3: _evaluation(3.0, "training_exact_cache")},
            evaluate_candidate=lambda index: _evaluation({2: 2.0, 1: 1.0}[index]),
            config=BBHTConfig(max_threshold_updates=2, seed=19),
            trial_executor=SequenceExecutor([2, 1]),
            training_indices=(3,),
        )

    first = run_once()
    second = run_once()
    assert first.trial_trace == second.trial_trace
    assert first.threshold_history == second.threshold_history
    assert first.final_incumbent_index == second.final_incumbent_index == 1


def test_bbht_core_does_not_call_marked_count_validation_or_enumerate_states() -> None:
    source = inspect.getsource(run_sparse_vqc_bbht)
    assert "ordinary_grover_validation_plan" not in source
    assert "direct_float_marked_indices" not in source
    assert "range(2" not in source
    for generators, periods in ((3, 2), (3, 3)):
        model = _binary_value_model(generators, periods)
        result = run_sparse_vqc_bbht(
            model,
            initial_incumbent_index=0,
            initial_exact_cache={0: _evaluation(0.0, "training_exact_cache")},
            evaluate_candidate=lambda index: pytest.fail("no evaluation expected"),
            config=BBHTConfig(),
            trial_executor=lambda *args: pytest.fail("no execution expected"),
        )
        assert result.circuit_executions == 0


@pytest.mark.skipif(AerSimulator is None, reason="qiskit-aer is not installed")
def test_one_shot_bbht_mps_trial_returns_actual_measurement_and_zero_auxiliaries() -> None:
    model = _binary_value_model()
    execution = execute_bbht_trial_mps(
        model,
        encoded_threshold=3,
        iterations=1,
        shots=1,
        seed=23,
    )
    assert execution.shots == 1
    assert sum(execution.raw_counts.values()) == 1
    assert sum(execution.x_counts.values()) == 1
    assert execution.measured_bitstring in execution.x_counts
    assert execution.measured_count == 1
    assert execution.measured_probability == 1.0
    assert execution.auxiliary_zero_probability > 1.0 - 1e-12
