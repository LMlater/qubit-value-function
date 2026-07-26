from __future__ import annotations

from dataclasses import dataclass

import pytest

from qubit_value_function.candidate_acceptance_loop import (
    ClosedLoopBudgets,
    ExactCandidateEvaluation,
)
from qubit_value_function.closed_loop_scenario import (
    ClosedLoopScenario,
    run_closed_loop_method,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig
from qubit_value_function.sparse_vqc_bbht import BBHTConfig, BBHTTrialExecution


@dataclass(frozen=True)
class _Model:
    values: tuple[int, int, int, int] = (0, 1, 2, 3)
    num_x_qubits: int = 2
    lower_bound: int = 0
    fixed_point_config: FixedPointConfig = FixedPointConfig(fractional_bits=0, unit=1.0)

    def integer_value(self, bits: tuple[int, ...]) -> int:
        return self.values[sum(int(bit) << offset for offset, bit in enumerate(bits))]


def _record(cost: float, source: str = "training_exact_cache") -> ExactCandidateEvaluation:
    return ExactCandidateEvaluation(True, cost, "ok", source, lp_solve_performed=True)


def _scenario() -> ClosedLoopScenario:
    def evaluate(index: int) -> ExactCandidateEvaluation:
        if int(index) == 0:
            return ExactCandidateEvaluation(
                False, None, "logic rejected", "logic_precheck", False, True
            )
        return _record(float(index), "new_ed_lp_call")

    return ClosedLoopScenario(
        scenario_id="fake-g0g1-w0-s0",
        generator_pair=(0, 1),
        window_start=0,
        horizon=2,
        training_seed=0,
        training_indices=(3,),
        training_labels=((3, 3.0),),
        value_model=_Model(),
        initial_incumbent_index=3,
        initial_exact_cache={3: _record(3.0)},
        evaluate_candidate=evaluate,
        hard_logic_is_feasible=lambda bits: bits != (0, 0),
        commitments=((0, 0), (1, 0), (0, 1), (1, 1)),
        budgets=ClosedLoopBudgets(3, 3, 3, 3, 3, 3, 3),
        bbht_config=BBHTConfig(max_trials=3, max_new_ed_lp_calls=3, seed=0),
        reproducibility_metadata={"fixture": True},
    )


def _trial(index: int) -> BBHTTrialExecution:
    bitstring = "".join(str((index >> bit) & 1) for bit in range(2))
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
        total_qubits=3,
        estimated_statevector_memory_gb=0.0,
        elapsed_seconds=0.0,
        circuit_resources={},
    )


def test_six_methods_share_frozen_setup_but_receive_independent_cache_and_state() -> None:
    scenario = _scenario()
    methods = (
        "joint_bbht", "cost_only_bbht", "full_space_random", "logic_rejection_random",
        "direct_logic_feasible_random", "classical_joint_marked_random",
    )
    results = {
        method: run_closed_loop_method(
            scenario,
            method,
            run_seed=7,
            trial_executor=lambda *args, index=1: _trial(index),
        )
        for method in methods
    }

    assert all(result["scenario"]["training_indices"] == [3] for result in results.values())
    assert all(result["scenario"]["initial_incumbent_index"] == 3 for result in results.values())
    assert scenario.initial_exact_cache == {3: _record(3.0)}
    with pytest.raises(TypeError):
        scenario.initial_exact_cache[1] = _record(1.0)
    assert results["full_space_random"]["result"].exact_cache is not scenario.initial_exact_cache
    assert results["joint_bbht"]["method_role"] == "quantum_method"
    assert results["direct_logic_feasible_random"]["diagnostic_only"] is True
    assert results["classical_joint_marked_random"]["diagnostic_only"] is True


def test_cost_only_uses_cost_oracle_and_classical_joint_marked_recomputes_threshold_set() -> None:
    scenario = _scenario()
    cost_only = run_closed_loop_method(
        scenario, "cost_only_bbht", run_seed=2, trial_executor=lambda *args: _trial(0)
    )["result"]
    marked = run_closed_loop_method(
        scenario, "classical_joint_marked_random", run_seed=2
    )["result"]

    assert cost_only.uses_hard_feasibility_oracle is False
    assert cost_only.admission_policy.name == "cost_only_bbht"
    assert cost_only.logic_precheck_rejections == 1
    assert marked.stop_reason in {"max_proposals_reached", "no_classical_joint_marked_state"}
    assert all(row["quantum_resources"]["applicable"] is False for row in marked.trial_trace)


def test_direct_logic_feasible_never_draws_infeasible_and_dispatcher_rejects_unknown_method() -> None:
    scenario = _scenario()
    result = run_closed_loop_method(scenario, "direct_logic_feasible_random", run_seed=4)["result"]

    assert all(row["hard_logic_feasible"] is True for row in result.trial_trace)
    with pytest.raises(ValueError, match="不支持"):
        run_closed_loop_method(scenario, "unknown", run_seed=0)
