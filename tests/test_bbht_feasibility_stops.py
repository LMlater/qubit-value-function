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
    linear = (1, 2, 4, 8)
    weights = linear + (0,) * (len(features) - len(linear))
    lower, upper = conservative_integer_bounds(0, weights)
    return QuantizedSparseValueModel(
        num_generators=2,
        num_periods=2,
        features=features,
        fixed_point_config=FixedPointConfig(fractional_bits=0, unit=1.0),
        real_intercept=0.0,
        real_weights=tuple(float(value) for value in weights),
        integer_intercept=0,
        integer_weights=weights,
        coefficient_quantization_errors=(0.0,) * (1 + len(weights)),
        lower_bound=lower,
        upper_bound=upper,
        value_shift=-lower,
        shifted_upper_bound=upper - lower,
        num_value_qubits=max(1, int(upper - lower).bit_length()),
    )


def _evaluation(cost: float) -> ExactCandidateEvaluation:
    return ExactCandidateEvaluation(
        success=True,
        total_cost=float(cost),
        message="ok",
        source="training_exact_cache",
    )


def _trial(*, auxiliary_zero_probability: float) -> BBHTTrialExecution:
    return BBHTTrialExecution(
        measured_index=0,
        measured_bitstring="0000",
        measured_count=1,
        measured_probability=1.0,
        shots=1,
        seed=0,
        raw_counts={"0000": 1},
        x_counts={"0000": 1},
        auxiliary_zero_probability=float(auxiliary_zero_probability),
        total_qubits=4,
        estimated_statevector_memory_gb=0.0,
        elapsed_seconds=0.0,
        circuit_resources={"num_qubits": 4, "depth": 0},
    )


def test_always_infeasible_spec_stops_before_any_circuit_or_evaluation() -> None:
    model = _model()
    spec = LogicFeasibilitySpec(
        num_generators=2,
        num_periods=2,
        selected_generator_indices=(0, 1),
        patterns=(),
        always_infeasible=True,
    )
    result = run_sparse_vqc_bbht(
        model,
        initial_incumbent_index=3,
        initial_exact_cache={3: _evaluation(3.0)},
        evaluate_candidate=lambda index: pytest.fail("no exact evaluation expected"),
        config=BBHTConfig(),
        trial_executor=lambda *args: pytest.fail("no circuit execution expected"),
        feasibility_spec=spec,
    )
    assert result.stop_reason == "no_hard_logic_feasible_state_by_compiled_constraints"
    assert result.circuit_executions == 0
    assert result.total_shots == 0
    assert result.total_oracle_calls == 0
    assert result.new_exact_evaluation_attempts == 0
    assert result.actual_ed_lp_solves == 0
    assert result.trial_trace == ()


def test_auxiliary_syndrome_has_independent_budget_and_keeps_m_fixed() -> None:
    model = _model()
    calls = 0

    def executor(*args):
        nonlocal calls
        calls += 1
        return _trial(auxiliary_zero_probability=0.0)

    result = run_sparse_vqc_bbht(
        model,
        initial_incumbent_index=3,
        initial_exact_cache={3: _evaluation(3.0)},
        evaluate_candidate=lambda index: pytest.fail("syndrome candidate must be rejected"),
        config=BBHTConfig(
            max_trials=10,
            max_auxiliary_syndrome_rejections=2,
            minimum_auxiliary_zero_probability=1.0,
            seed=7,
        ),
        trial_executor=executor,
    )
    assert calls == 2
    assert result.stop_reason == "max_auxiliary_syndrome_rejections_reached"
    assert result.auxiliary_syndrome_rejections == 2
    assert result.circuit_executions == 2
    assert result.new_exact_evaluation_attempts == 0
    assert result.cached_exact_lookups == 0
    assert all(row["candidate_status"] == "auxiliary_syndrome_rejected" for row in result.trial_trace)
    assert all(row["m_after"] == row["m_before"] for row in result.trial_trace)
    assert result.trial_trace[-1]["stop_reason_after_trial"] == (
        "max_auxiliary_syndrome_rejections_reached"
    )
