from __future__ import annotations

import pytest

from qubit_value_function.closed_loop_oracle_grover_audit import OracleGroverAuditError, _group, derive_trial_rows


def _run(trace):
    return {
        "run_id": "run0", "method": "joint_bbht",
        "scenario": {"scenario_id": "case14-g0g5-w2-s0"},
        "run_spec": {"run_seed": 0},
        "result": {"threshold_history": [{"encoded_threshold": 8}], "trial_trace": trace},
    }


def _trace(number, *, threshold=8, cost=3, feasible=True, marked=True, iterations=1, changed=False):
    return {"trial_number": number, "encoded_threshold_before": threshold, "true_threshold_before": 10.0,
            "sampled_grover_iterations": iterations, "oracle_calls_added": iterations,
            "measured_index": 2, "measured_bitstring": "0100", "surrogate_integer_cost": cost,
            "hard_logic_feasible": feasible, "auxiliary_accepted": True, "joint_marked": marked,
            "cost_marked": cost < threshold, "admission_passed": marked, "candidate_cache_hit": False,
            "actual_ed_lp_solves_added": 0, "true_improvement": False, "accepted_update": False,
            "encoded_threshold_changed": changed, "stop_reason_after_trial": None}


def _landscape():
    return {"scenario_id": "case14-g0g5-w2-s0", "initial_joint_marked_indices": [1, 2],
            "initial_cost_marked_indices": [1, 2, 3], "initial_joint_marked_count": 2,
            "initial_surrogate_cost_marked_count": 3, "real_improvement_indices": [1, 2, 4]}


def test_initial_threshold_rows_recover_marked_count_and_ideal_probability() -> None:
    rows = derive_trial_rows(_run([_trace(1)]), _landscape())
    row = rows[0]
    assert row["joint_marked_count"] == 2
    assert row["cost_marked_count"] == 3
    assert row["measured_joint_marked_hit"] is True
    assert row["valid_joint_marked_hit"] is True
    assert row["random_joint_hit_probability"] == 2 / 16
    assert row["ideal_joint_grover_hit_probability"] > row["random_joint_hit_probability"]


def test_changed_threshold_is_unavailable_not_guessed() -> None:
    rows = derive_trial_rows(_run([_trace(1), _trace(2, threshold=5, cost=4, marked=True)]), _landscape())
    assert rows[0]["joint_marked_count"] == 2
    assert rows[1]["joint_marked_count"] is None
    assert rows[1]["marked_set_recovery_status"] == "unavailable_quantized_vqc_table_not_persisted"


def test_trace_oracle_semantic_mismatch_is_an_error() -> None:
    with pytest.raises(OracleGroverAuditError, match="joint_marked"):
        derive_trial_rows(_run([_trace(1, cost=9, marked=True)]), _landscape())


def test_legacy_grover_field_is_used_only_as_fallback() -> None:
    trace = _trace(1); trace.pop("sampled_grover_iterations"); trace["grover_iterations"] = 3
    row = derive_trial_rows(_run([trace]), _landscape())[0]
    assert row["sampled_grover_iterations"] == 3


def test_k_group_averages_uniform_baseline_across_different_marked_counts() -> None:
    grouped = _group([
        {"sampled_grover_iterations": 1, "joint_marked_count": 1, "measured_joint_marked_hit": True, "random_joint_hit_probability": 1 / 16, "ideal_joint_grover_hit_probability": 0.5, "candidate_admitted": True, "true_improvement": True},
        {"sampled_grover_iterations": 1, "joint_marked_count": 3, "measured_joint_marked_hit": False, "random_joint_hit_probability": 3 / 16, "ideal_joint_grover_hit_probability": 0.7, "candidate_admitted": False, "true_improvement": False},
    ], ("sampled_grover_iterations",))
    assert grouped[0]["uniform_baseline"] == 2 / 16
    assert grouped[0]["ideal_grover_hit_probability"] == 0.6
