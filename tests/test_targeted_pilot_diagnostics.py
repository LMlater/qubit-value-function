from __future__ import annotations

import pytest

from qubit_value_function.targeted_pilot_diagnostics import (
    classify_quantum_trial,
    grover_theoretical_probability,
    initial_marked_counts,
    summarize_runs,
)


def test_grover_probability_uses_the_actual_oracle_marked_count() -> None:
    assert grover_theoretical_probability(search_space_size=16, marked_count=1, iterations=0) == pytest.approx(1 / 16)
    assert grover_theoretical_probability(search_space_size=16, marked_count=1, iterations=1) == pytest.approx(121 / 256)
    assert grover_theoretical_probability(search_space_size=16, marked_count=3, iterations=0) == pytest.approx(3 / 16)
    assert grover_theoretical_probability(search_space_size=16, marked_count=3, iterations=1) == pytest.approx(243 / 256)
    assert grover_theoretical_probability(search_space_size=16, marked_count=0, iterations=3) == 0.0
    assert grover_theoretical_probability(search_space_size=16, marked_count=16, iterations=3) == 1.0


def test_initial_counts_distinguish_training_and_nontraining_joint_marks() -> None:
    snapshot = {
        "search_space_size": 4,
        "training_indices": [0, 1],
        "state_proxy_table": [
            {"state_index": 0, "integer_vqc_value": 3, "hard_logic_feasible": True},
            {"state_index": 1, "integer_vqc_value": 4, "hard_logic_feasible": True},
            {"state_index": 2, "integer_vqc_value": 2, "hard_logic_feasible": True},
            {"state_index": 3, "integer_vqc_value": 1, "hard_logic_feasible": False},
        ],
    }
    assert initial_marked_counts(snapshot, encoded_threshold=4) == {
        "initial_cost_marked_count": 3,
        "initial_joint_marked_count": 2,
        "initial_nontraining_cost_marked_count": 2,
        "initial_nontraining_joint_marked_count": 1,
        "initial_training_joint_marked_count": 1,
        "initial_cost_marked_indices": [0, 2, 3],
        "initial_joint_marked_indices": [0, 2],
    }


@pytest.mark.parametrize(
    ("method", "row", "expected"),
    [
        ("joint_bbht", {"measured_joint_marked": False}, "unmarked_measurement"),
        ("joint_bbht", {"measured_joint_marked": True, "cache_source": "training_cache"}, "marked_training_cache"),
        ("joint_bbht", {"measured_joint_marked": True, "cache_source": "search_cache"}, "marked_search_cache"),
        ("joint_bbht", {"measured_joint_marked": True, "new_ed_lp_solve": True}, "marked_new_edlp_no_improvement"),
        ("joint_bbht", {"measured_joint_marked": True, "new_ed_lp_solve": True, "true_strict_improvement": True, "improvement_is_nontraining": True}, "marked_new_edlp_nontraining_improvement"),
        ("cost_only_bbht", {"measured_cost_marked": False}, "unmarked_measurement"),
        ("cost_only_bbht", {"measured_cost_marked": True, "hard_logic_feasible_measured": False}, "cost_marked_hard_logic_infeasible"),
        ("cost_only_bbht", {"measured_cost_marked": True, "hard_logic_feasible_measured": True, "cache_source": "training_cache"}, "marked_training_cache"),
        ("cost_only_bbht", {"measured_cost_marked": True, "hard_logic_feasible_measured": True, "new_ed_lp_solve": True}, "marked_new_edlp_no_improvement"),
        ("cost_only_bbht", {"measured_cost_marked": True, "hard_logic_feasible_measured": True, "new_ed_lp_solve": True, "true_strict_improvement": True, "improvement_is_nontraining": True}, "marked_new_edlp_nontraining_improvement"),
    ],
)
def test_quantum_outcome_categories_are_exhaustive_for_persisted_predicates(
    method: str, row: dict[str, object], expected: str
) -> None:
    assert classify_quantum_trial(method=method, row=row) == expected


def test_run_summary_reports_successes_and_run_level_edlp_statistics() -> None:
    runs = [
        {
            "run_id": "a", "method": "joint_bbht", "scenario_id": "s",
            "search_stage_new_edlp": 0, "training_edlp": 8,
            "trials": [{"trial_index": 2, "measured_index": 7, "true_strict_improvement": True, "improvement_is_nontraining": True, "new_ed_lp_solve": True}],
            "validation": {"nontraining_global_optimum_success": False},
        },
        {
            "run_id": "b", "method": "joint_bbht", "scenario_id": "s",
            "search_stage_new_edlp": 2, "training_edlp": 8,
            "trials": [{"trial_index": 1, "measured_index": 9, "true_strict_improvement": True, "improvement_is_nontraining": True, "cache_source": "search_cache"}],
            "validation": {"nontraining_global_optimum_success": True, "successful_state_index": 9},
        },
        {"run_id": "c", "method": "joint_bbht", "scenario_id": "s", "search_stage_new_edlp": 4, "training_edlp": 8, "trials": [], "validation": {}},
    ]
    summary = summarize_runs(runs)[0]
    assert summary["runs"] == 3
    assert summary["runs_with_nontraining_strict_improvement"] == 2
    assert summary["nontraining_strict_improvement_probability"] == pytest.approx(2 / 3)
    assert summary["runs_with_nontraining_global_optimum"] == 1
    assert summary["nontraining_global_optimum_probability"] == pytest.approx(1 / 3)
    assert summary["total_search_stage_new_edlp"] == 6
    assert summary["mean_search_stage_new_edlp_per_run"] == pytest.approx(2.0)
    assert summary["median_search_stage_new_edlp_per_run"] == pytest.approx(2.0)
    assert summary["q25_search_stage_new_edlp_per_run"] == pytest.approx(1.0)
    assert summary["q75_search_stage_new_edlp_per_run"] == pytest.approx(3.0)
    assert summary["standard_deviation_search_stage_new_edlp_per_run"] == pytest.approx(2.0)
    assert summary["total_training_plus_search_edlp"] == 30
    assert summary["first_nontraining_improvement_trial"] == {"a": 2, "b": 1}
    assert summary["first_nontraining_improvement_new_edlp_count"] == {"a": 1, "b": 0}
