from __future__ import annotations

from qubit_value_function.candidate_acceptance_loop import ExactCandidateEvaluation
from qubit_value_function.candidate_acceptance_loop import ClosedLoopBudgets
from qubit_value_function.closed_loop_scenario import ClosedLoopScenario, run_closed_loop_method
from qubit_value_function.coherent_phase_value import (
    QuantizedSparseValueModel,
    conservative_integer_bounds,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig
from qubit_value_function.sparse_phase_vqc import build_local_phase_features
from qubit_value_function.sparse_vqc_bbht import BBHTConfig, BBHTTrialExecution, run_sparse_vqc_bbht


def _model() -> QuantizedSparseValueModel:
    features = build_local_phase_features(2, 2)
    weights = (1, 2, 4, 8) + (0,) * (len(features) - 4)
    lower, upper = conservative_integer_bounds(0, weights)
    return QuantizedSparseValueModel(
        num_generators=2, num_periods=2, features=features,
        fixed_point_config=FixedPointConfig(fractional_bits=0, unit=1.0),
        real_intercept=0.0, real_weights=tuple(float(value) for value in weights),
        integer_intercept=0, integer_weights=weights,
        coefficient_quantization_errors=(0.0,) * (1 + len(weights)),
        lower_bound=lower, upper_bound=upper, value_shift=-lower,
        shifted_upper_bound=upper - lower, num_value_qubits=4,
    )


class _Executor:
    def __init__(self, indices: list[int]) -> None:
        self.indices = list(indices)

    def __call__(self, model, threshold, iterations, shots, seed):
        index = self.indices.pop(0)
        bitstring = "".join(str((index >> offset) & 1) for offset in range(model.num_x_qubits))
        return BBHTTrialExecution(
            measured_index=index, measured_bitstring=bitstring, measured_count=1,
            measured_probability=1.0, shots=1, seed=seed, raw_counts={bitstring: 1},
            x_counts={bitstring: 1}, auxiliary_zero_probability=1.0,
            total_qubits=8, estimated_statevector_memory_gb=0.0,
            elapsed_seconds=0.0, circuit_resources={},
        )


def _run(persist: bool):
    return run_sparse_vqc_bbht(
        _model(), initial_incumbent_index=3,
        initial_exact_cache={3: ExactCandidateEvaluation(True, 3.0, "ok", "training_exact_cache")},
        evaluate_candidate=lambda index: ExactCandidateEvaluation(True, float(index), "ok", "new_ed_lp_call"),
        config=BBHTConfig(max_threshold_updates=2, seed=19),
        trial_executor=_Executor([2, 1]), training_indices=(3,),
        persist_dynamic_oracle_metadata=persist,
        hard_logic_metadata=lambda bits: bits[0] == 0,
    )


def test_metadata_toggle_preserves_bbht_decisions_cache_and_rng_trace() -> None:
    off = _run(False)
    on = _run(True)
    for off_row, on_row in zip(off.trial_trace, on.trial_trace):
        for field in (
            "sampled_grover_iterations", "measured_index", "admission_passed",
            "candidate_cache_hit", "true_improvement", "accepted_update",
            "encoded_threshold_before", "encoded_threshold_after", "stop_reason_after_trial",
        ):
            assert on_row[field] == off_row[field]
        assert on_row["candidate_admitted"] == off_row["admission_passed"]
        assert on_row["cache_hit"] == off_row["candidate_cache_hit"]
        assert on_row["new_ed_lp_solve"] == off_row["new_edlp_solve_performed"]
        assert on_row["candidate_true_cost"] == off_row["exact_cost"]
    assert on.exact_cache == off.exact_cache
    assert on.total_oracle_calls == off.total_oracle_calls
    assert on.actual_ed_lp_solves == off.actual_ed_lp_solves
    assert on.final_incumbent_index == off.final_incumbent_index
    assert on.final_incumbent_true_cost == off.final_incumbent_true_cost
    assert all("joint_marked_count_before" in row for row in on.trial_trace)
    assert all("joint_marked_count_before" not in row for row in off.trial_trace)


def test_closed_loop_envelope_exposes_single_scenario_snapshot_only_when_enabled() -> None:
    model = _model()
    scenario = ClosedLoopScenario(
        scenario_id="case14-g0g5-w2-s0", generator_pair=(0, 5), window_start=2,
        horizon=2, training_seed=0, training_indices=(3,), training_labels=((3, 3.0),),
        value_model=model, initial_incumbent_index=3,
        initial_exact_cache={3: ExactCandidateEvaluation(True, 3.0, "ok", "training_exact_cache")},
        evaluate_candidate=lambda index: ExactCandidateEvaluation(True, float(index), "ok", "new_ed_lp_call"),
        hard_logic_is_feasible=lambda bits: bits[0] == 0,
        commitments=tuple(range(16)),
        budgets=ClosedLoopBudgets(2, 2, 2, 2, 2, 2, 2),
        bbht_config=BBHTConfig(max_trials=2, max_threshold_updates=2, seed=0),
        reproducibility_metadata={"fixture": True},
    )
    envelope = run_closed_loop_method(
        scenario, "cost_only_bbht", run_seed=4,
        trial_executor=_Executor([2, 1]), persist_dynamic_oracle_metadata=True,
    )
    snapshot = envelope["quantized_model_snapshot"]
    assert snapshot["scenario_id"] == scenario.scenario_id
    assert len(snapshot["state_proxy_table"]) == 16
