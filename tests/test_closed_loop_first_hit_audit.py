from __future__ import annotations

import json
from pathlib import Path

import pytest

from qubit_value_function.closed_loop_first_hit_audit import (
    FirstHitAuditError,
    audit_formal_first_hits,
    derive_first_hit_run,
)


def _validation(*, optima=(2,), training=(0,), initial=0, optimum_cost=5.0):
    return {
        "scenario_id": "case14-g0g5-w2-s0",
        "true_global_optimum_indices": list(optima),
        "true_global_optimum_cost": optimum_cost,
        "initial_incumbent_index": initial,
        "initial_incumbent_is_global_optimum": initial in optima,
        "initial_true_improvement_exists": initial not in optima,
    }


def _row(number, index, **extra):
    row = {
        "trial_number": number,
        "measured_index": index,
        "sampled_grover_iterations": 2,
        "oracle_calls_added": 2,
        "actual_ed_lp_solves_added": 0,
        "new_exact_evaluation_attempts_added": 0,
        "candidate_cache_hit": False,
        "threshold_updated": False,
        "admission_passed": False,
        "accepted_update": False,
        "incumbent_index_before": 0,
        "incumbent_index_after": 0,
        "exact_evaluation": None,
    }
    row.update(extra)
    return row


def _run(trace, *, training=(0,), run_seed=0, stop_reason="max_trials_reached"):
    return {
        "run_id": f"case14_g0-g5_w2_train0_joint_bbht_run{run_seed}",
        "method": "joint_bbht",
        "run_spec": {"scenario_id": "case14-g0g5-w2-s0", "generator_pair": [0, 5], "window_start": 2, "training_seed": 0, "run_seed": run_seed},
        "scenario": {"scenario_id": "case14-g0g5-w2-s0", "training_indices": list(training)},
        "result": {
            "initial_incumbent_index": 0,
            "initial_incumbent_true_cost": 10.0,
            "final_incumbent_index": trace[-1].get("incumbent_index_after", 0),
            "final_incumbent_true_cost": 5.0 if any(row.get("accepted_update") for row in trace) else 10.0,
            "stop_reason": stop_reason,
            "trial_trace": trace,
        },
    }


def test_first_trial_new_edlp_nontraining_optimum_and_tail_are_separated() -> None:
    trace = [
        _row(1, 2, admission_passed=True, accepted_update=True, incumbent_index_after=2,
             actual_ed_lp_solves_added=1, new_exact_evaluation_attempts_added=1, threshold_updated=True,
             exact_evaluation={"success": True, "total_cost": 5.0, "source": "new_ed_lp_call", "lp_solve_performed": True}),
        _row(2, 1, sampled_grover_iterations=4, oracle_calls_added=4),
    ]
    row = derive_first_hit_run(_run(trace), _validation())
    assert row["first_measured_optimum_trial"] == 1
    assert row["first_admitted_optimum_trial"] == 1
    assert row["first_confirmed_optimum_trial"] == 1
    assert row["first_accepted_optimum_trial"] == 1
    assert row["first_accepted_optimum_is_nontraining"] is True
    assert row["first_accepted_optimum_source"] == "new_ed_lp_call"
    assert row["trials_to_first_accepted_optimum"] == 1
    assert row["grover_iterations_to_first_accepted_optimum"] == 2
    assert row["tail_trials_after_first_optimum"] == 1
    assert row["tail_grover_iterations_after_first_optimum"] == 4
    assert row["post_measurement_predicate_checks_to_first_accepted_optimum"] == 1
    assert row["combined_predicate_query_reference_to_first_accepted_optimum"] == 3
    assert row["diffuser_calls_to_first_accepted_optimum"] is None


def test_measured_optimum_can_be_rejected_before_cached_admitted_acceptance() -> None:
    trace = [
        _row(1, 2),
        _row(2, 2, admission_passed=True, accepted_update=True, incumbent_index_after=2,
             candidate_cache_hit=True, threshold_updated=True,
             exact_evaluation={"success": True, "total_cost": 5.0, "source": "initial_training_cache", "lp_solve_performed": False}),
    ]
    row = derive_first_hit_run(_run(trace, training=(0, 2)), _validation(training=(0, 2)))
    assert row["first_measured_optimum_trial"] == 1
    assert row["first_admitted_optimum_trial"] == 2
    assert row["first_confirmed_optimum_trial"] == 2
    assert row["first_accepted_optimum_trial"] == 2
    assert row["first_accepted_optimum_is_nontraining"] is False
    assert row["first_accepted_optimum_source"] == "training_exact_cache"


def test_multiple_optima_and_legacy_grover_field_are_supported() -> None:
    trace = [_row(1, 3, admission_passed=True, accepted_update=True, incumbent_index_after=3,
                  exact_evaluation={"success": True, "total_cost": 5.0, "source": "cached", "lp_solve_performed": False})]
    trace[0].pop("sampled_grover_iterations")
    trace[0]["grover_iterations"] = 7
    row = derive_first_hit_run(_run(trace), _validation(optima=(2, 3)))
    assert row["all_global_optima_outside_training"] is True
    assert row["any_global_optimum_outside_training"] is True
    assert row["grover_iterations_to_first_accepted_optimum"] == 7
    assert row["first_accepted_optimum_source"] == "cached_nontraining_exact"


def test_no_hit_and_marked_empty_stop_use_null_first_hit_fields() -> None:
    row = derive_first_hit_run(_run([_row(1, 1)], stop_reason="no_surrogate_marked_state_by_conservative_lower_bound"), _validation())
    assert row["first_accepted_optimum_trial"] is None
    assert row["trials_to_first_accepted_optimum"] is None
    assert row["stopped_without_optimum"] is True
    assert row["marked_empty_or_lower_bound_stop"] is True


def test_missing_validation_optimum_is_explicitly_rejected() -> None:
    with pytest.raises(FirstHitAuditError):
        derive_first_hit_run(_run([_row(1, 1)]), {"scenario_id": "case14-g0g5-w2-s0"})


def test_audit_writes_deterministic_strict_json_without_overwriting(tmp_path: Path) -> None:
    formal = tmp_path / "formal"; completed = formal / "runs" / "completed"; validation = formal / "validation" / "scenarios"
    completed.mkdir(parents=True); validation.mkdir(parents=True)
    run = _run([_row(1, 2, admission_passed=True, accepted_update=True, incumbent_index_after=2,
                    exact_evaluation={"success": True, "total_cost": 5.0, "source": "new_ed_lp_call", "lp_solve_performed": True})])
    (completed / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (validation / "case14-g0g5-w2-s0.json").write_text(json.dumps(_validation()), encoding="utf-8")
    first_output = formal / "audit_one"
    first = audit_formal_first_hits(formal, output_dir=first_output)
    assert first["validation_checks"]["joint_bbht_run_count"]["actual"] == 1
    payload = (first_output / "joint_bbht_first_hit_summary.json").read_text(encoding="utf-8")
    assert "NaN" not in payload and "Infinity" not in payload
    second_output = formal / "audit_two"
    audit_formal_first_hits(formal, output_dir=second_output)
    assert payload == (second_output / "joint_bbht_first_hit_summary.json").read_text(encoding="utf-8")
    with pytest.raises(FirstHitAuditError):
        audit_formal_first_hits(formal, output_dir=first_output)
