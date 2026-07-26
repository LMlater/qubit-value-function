from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from qubit_value_function.candidate_acceptance_loop import ExactCandidateEvaluation
from qubit_value_function.closed_loop_validation import (
    ValidationConsistencyError,
    edlp_budget_curve,
    trace_metrics,
    summarize_validated,
    validate_run_against_landscape,
    validate_scenario_landscape,
    validate_source_batch,
)


@dataclass(frozen=True)
class _Model:
    num_x_qubits: int = 2
    def integer_value(self, bits): return (4, 2, 1, 3)[sum(bit << i for i, bit in enumerate(bits))]
    def encode_true_cost(self, cost): return int(cost)


@dataclass
class _Scenario:
    scenario_id: str = "case14-g0g1-w0-s0"
    generator_pair: tuple[int, int] = (0, 1)
    window_start: int = 0
    training_seed: int = 0
    initial_incumbent_index: int = 0
    training_indices: tuple[int, ...] = (0,)
    value_model: object = _Model()
    initial_exact_cache: dict[int, ExactCandidateEvaluation] = None
    calls: list[int] = None
    def __post_init__(self):
        self.initial_exact_cache = {0: ExactCandidateEvaluation(True, 10.0, "train", "training_exact_cache")}
        self.calls = []
    def hard_logic_is_feasible(self, bits): return tuple(bits) != (1, 1)
    def evaluate_candidate(self, index):
        self.calls.append(index)
        return ExactCandidateEvaluation(True, {1: 8.0, 2: 9.0}[index], "new", "new_ed_lp_call", True)


def _run_payload() -> dict[str, object]:
    return {
        "run_id": "run0", "fingerprint": "fp", "method": "logic_rejection_random",
        "method_role": "random_baseline", "diagnostic_only": False,
        "scenario": {"scenario_id": "case14-g0g1-w0-s0"},
        "result": {
            "initial_incumbent_index": 0, "initial_incumbent_true_cost": 10.0,
            "final_incumbent_index": 1, "final_incumbent_true_cost": 8.0,
            "threshold_history": [], "exact_cache": {"1": {"success": True, "total_cost": 8.0, "message": "cached", "source": "new_ed_lp_call", "lp_solve_performed": True, "logic_precheck_rejected": False}},
            "trial_trace": [
                {"candidate_index": 3, "is_unique_candidate": True, "is_repeated_candidate": False, "admission_rejection_reason": "hard_logic_infeasible", "hard_logic_feasible": False, "actual_ed_lp_solves_added": 0, "true_threshold_after": 10.0, "grover_iterations": 0, "oracle_calls_added": 0},
                {"candidate_index": 1, "is_unique_candidate": True, "is_repeated_candidate": False, "candidate_cache_hit": False, "new_edlp_confirmed_improvement": True, "nontraining_true_improvement": True, "new_exact_evaluation_attempts_added": 1, "actual_ed_lp_solves_added": 1, "true_threshold_after": 8.0, "grover_iterations": 2, "oracle_calls_added": 2, "elapsed_seconds": 0.5, "total_qubits": 7, "circuit_resources": {"depth": 11}},
            ],
            "counters": {"threshold_updates": 1}, "stop_reason": "done",
        },
    }


def test_validation_landscape_scans_all_states_and_never_calls_lp_for_logic_infeasible() -> None:
    scenario = _Scenario()
    landscape = validate_scenario_landscape(scenario, [_run_payload()])
    assert landscape["search_space_dimension"] == 4
    assert landscape["hard_logic_feasible_indices"] == [0, 1, 2]
    assert landscape["exact_successful_indices"] == [0, 1, 2]
    assert scenario.calls == [2]  # 1 is reused from a run cache; 3 never invokes ED/LP.
    assert landscape["true_global_optimum_index"] == 1
    assert landscape["initial_true_improvement_exists"] is True
    assert landscape["real_improvement_exists_but_joint_marked_empty"] is False


def test_validation_rejects_conflicting_reused_true_costs() -> None:
    conflicting = _run_payload()
    conflicting["result"]["exact_cache"]["1"]["total_cost"] = 7.0
    with pytest.raises(ValidationConsistencyError):
        validate_scenario_landscape(_Scenario(), [_run_payload(), conflicting])


def test_trace_metrics_separate_admission_and_exact_logic_rejections_and_budget_curve_is_observed_only() -> None:
    result = _run_payload()["result"]
    metrics = trace_metrics(result, wrapper_elapsed_seconds=1.0)
    assert metrics["proposal_events"] == 2
    assert metrics["admission_hard_logic_rejections"] == 1
    assert metrics["exact_logic_precheck_rejections"] == 0
    assert metrics["total_logic_rejections"] == 1
    assert metrics["mps_trial_elapsed_sum"] == 0.5
    assert metrics["maximum_qubits"] == 7 and metrics["maximum_circuit_depth"] == 11
    curve = edlp_budget_curve(result)
    assert curve[0] == 10.0 and curve[1] == 8.0 and 2 not in curve


def test_run_validation_uses_tolerance_for_global_flag_and_marks_nontraining_hit() -> None:
    landscape = validate_scenario_landscape(_Scenario(), [_run_payload()])
    derived = validate_run_against_landscape(_run_payload(), landscape, abs_tol=1e-6, rel_tol=1e-9)
    assert derived["reached_global_optimum"] is True
    assert derived["nontraining_global_optimum_hit"] is True
    assert derived["final_rank_among_successful_states"] == 1


def test_source_validation_builds_each_scenario_once_and_never_rewrites_source_run(tmp_path: Path) -> None:
    source = tmp_path / "source"; completed = source / "runs" / "completed"; completed.mkdir(parents=True)
    runs = []
    for run_id in ("run0", "run1"):
        run = _run_payload()
        run.update({"run_id": run_id, "status": "completed", "code": {"head": "search"}})
        run["run_spec"] = {"scenario_id": "case14-g0g1-w0-s0", "budget_config": {"max_trials": 2}, "fixed_point_config": {"fractional_bits": 0}}
        runs.append(run)
        (completed / f"{run_id}.json").write_text(json.dumps(run), encoding="utf-8")
    manifest = {"expected_head": "search", "config_fingerprint": "batch-fp", "budget_config": {"max_trials": 2}, "fixed_point_config": {"fractional_bits": 0}, "planned_fingerprints": {run["run_id"]: "fp" for run in runs}}
    (source / "batch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    original = (completed / "run0.json").read_text(encoding="utf-8")
    builds = 0
    def build(_spec):
        nonlocal builds
        builds += 1
        return _Scenario()
    result = validate_source_batch(source, scenario_builder=build, validation_code_sha="validation", expected_search_head="search")
    assert result["scenarios"] == 1 and result["runs"] == 2 and builds == 1
    assert (completed / "run0.json").read_text(encoding="utf-8") == original
    assert (source / "validation" / "scenarios" / "case14-g0g1-w0-s0.json").exists()
    summary = summarize_validated(source)
    assert summary["validated_runs"] == 2
    assert (source / "summaries_validated" / "edlp_budget_curves.csv").exists()
    assert validate_source_batch(source, scenario_builder=build, validation_code_sha="validation", expected_search_head="search", resume=True)["runs"] == 2
    assert builds == 1
    scenario_path = source / "validation" / "scenarios" / "case14-g0g1-w0-s0.json"
    corrupt = json.loads(scenario_path.read_text(encoding="utf-8")); corrupt["source_batch_fingerprint"] = "wrong"
    scenario_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValidationConsistencyError):
        validate_source_batch(source, scenario_builder=build, validation_code_sha="validation", expected_search_head="search", resume=True)
