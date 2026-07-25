from __future__ import annotations

import pytest

from qubit_value_function.coherent_phase_value import (
    QuantizedSparseValueModel,
    conservative_integer_bounds,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig
from qubit_value_function.logic_feasibility_oracle import LogicFeasibilitySpec
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
    return ExactCandidateEvaluation(
        success=True,
        total_cost=float(cost),
        message="ok",
        source="exact",
    )


def _trial(index: int, *, timing: bool = False) -> BBHTTrialExecution:
    bitstring = "".join(str((index >> bit) & 1) for bit in range(4))
    return BBHTTrialExecution(
        measured_index=index,
        measured_bitstring=bitstring,
        measured_count=1,
        measured_probability=1.0,
        shots=1,
        seed=0,
        raw_counts={"0": 1},
        x_counts={bitstring: 1},
        auxiliary_zero_probability=1.0,
        total_qubits=8,
        estimated_statevector_memory_gb=0.0,
        elapsed_seconds=0.5 if timing else 0.0,
        circuit_resources={"depth": 1},
        circuit_build_seconds=0.1 if timing else 0.0,
        transpile_seconds=0.2 if timing else 0.0,
        backend_run_seconds=0.3 if timing else 0.0,
        total_trial_seconds=0.6 if timing else 0.0,
    )


def test_always_infeasible_preflight_requires_no_incumbent_or_exact_cache() -> None:
    spec = LogicFeasibilitySpec(
        num_generators=2,
        num_periods=2,
        selected_generator_indices=(0, 1),
        patterns=(),
        always_infeasible=True,
    )
    result = run_sparse_vqc_bbht(
        _model(),
        feasibility_spec=spec,
        trial_executor=lambda *args: pytest.fail("no circuit expected"),
    )
    assert result.stop_reason == "no_hard_logic_feasible_state_by_compiled_constraints"
    assert result.initial_incumbent_index is None
    assert result.final_incumbent_index is None
    assert result.final_encoded_threshold is None
    assert result.threshold_history == ()
    assert result.preflight_hard_logic_space_empty is True
    assert result.preflight_circuit_skipped is True
    assert result.as_dict()["preflight"]["reason"] == "compiled_constraints"


def test_surrogate_integer_margin_widens_only_oracle_threshold() -> None:
    seen_thresholds: list[int] = []
    evaluated: list[int] = []

    def executor(model, threshold, iterations, shots, seed):
        seen_thresholds.append(int(threshold))
        return _trial(4)

    def evaluate(index: int) -> ExactCandidateEvaluation:
        evaluated.append(index)
        return _record(4.0)

    result = run_sparse_vqc_bbht(
        _model(),
        initial_incumbent_index=3,
        initial_exact_cache={3: _record(3.0)},
        evaluate_candidate=evaluate,
        config=BBHTConfig(
            surrogate_integer_margin=2,
            max_consecutive_nonimproving_marked=1,
            seed=0,
        ),
        trial_executor=executor,
    )
    assert seen_thresholds == [5]
    assert evaluated == [4]
    assert result.final_encoded_threshold == 3
    assert result.final_effective_oracle_threshold == 5
    trace = result.trial_trace[0]
    assert trace["encoded_threshold_before"] == 3
    assert trace["effective_oracle_threshold_before"] == 5
    assert trace["surrogate_integer_cost"] == 4
    assert trace["surrogate_better"] is True
    assert trace["true_improvement"] is False


def test_zero_margin_preserves_previous_unmarked_semantics() -> None:
    result = run_sparse_vqc_bbht(
        _model(),
        initial_incumbent_index=3,
        initial_exact_cache={3: _record(3.0)},
        evaluate_candidate=lambda index: pytest.fail("unmarked candidate"),
        config=BBHTConfig(max_trials=1, surrogate_integer_margin=0, seed=0),
        trial_executor=lambda *args: _trial(4),
    )
    assert result.trial_trace[0]["surrogate_better"] is False
    assert result.new_exact_evaluation_attempts == 0


def test_split_timing_is_preserved_in_trial_trace() -> None:
    result = run_sparse_vqc_bbht(
        _model(),
        initial_incumbent_index=3,
        initial_exact_cache={3: _record(3.0)},
        evaluate_candidate=lambda index: pytest.fail("unmarked candidate"),
        config=BBHTConfig(max_trials=1, seed=0),
        trial_executor=lambda *args: _trial(4, timing=True),
    )
    trace = result.trial_trace[0]
    assert trace["circuit_build_seconds"] == pytest.approx(0.1)
    assert trace["transpile_seconds"] == pytest.approx(0.2)
    assert trace["backend_run_seconds"] == pytest.approx(0.3)
    assert trace["total_trial_seconds"] == pytest.approx(0.6)


def test_negative_surrogate_margin_is_rejected() -> None:
    with pytest.raises(ValueError, match="surrogate_integer_margin"):
        BBHTConfig(surrogate_integer_margin=-1)
