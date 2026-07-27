"""Read-only integrity audit for the completed targeted candidate pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


class SelectedSplitAuditError(RuntimeError):
    """Persisted targeted-pilot artifacts do not satisfy their stated invariants."""


def _load(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectedSplitAuditError(f"pilot_json_unavailable:{path}") from error
    if not isinstance(payload, dict):
        raise SelectedSplitAuditError(f"pilot_json_not_object:{path}")
    return payload


def audit_targeted_best_training_candidate_pilot(pilot_dir: Path) -> dict[str, object]:
    """Read completed JSON only; never writes, evaluates ED/LP, or invokes a search routine."""

    pilot_dir = Path(pilot_dir)
    completed = sorted((pilot_dir / "runs" / "completed").glob("*.json"))
    summary = _load(pilot_dir / "summaries" / "targeted_pilot_summary.json")
    if len(completed) != 80:
        raise SelectedSplitAuditError("pilot_completed_run_count_is_not_80")
    per_scenario: dict[str, list[dict[str, object]]] = {}
    successful_indices: dict[str, set[int]] = {}
    successful_costs: dict[str, set[float]] = {}
    for path in completed:
        payload = _load(path)
        scenario = dict(payload.get("scenario", {}))
        result = dict(payload.get("result", {}))
        scenario_id = str(scenario.get("scenario_id", ""))
        training = {int(index) for index in scenario.get("training_indices", [])}
        trace = [dict(row) for row in result.get("trial_trace", [])]
        improvements = [
            row for row in trace
            if bool(row.get("true_strict_improvement")) and not bool(row.get("candidate_in_training_set"))
        ]
        if not improvements:
            raise SelectedSplitAuditError(f"pilot_missing_outside_training_strict_improvement:{path.name}")
        for row in improvements:
            index = int(row["measured_index"])
            if index in training:
                raise SelectedSplitAuditError(f"pilot_improvement_in_training_indices:{path.name}")
            successful_indices.setdefault(scenario_id, set()).add(index)
            successful_costs.setdefault(scenario_id, set()).add(float(row["candidate_true_cost"]))
        if int(dict(result.get("counters", {})).get("actual_ed_lp_solves", -1)) != 1:
            raise SelectedSplitAuditError(f"pilot_search_stage_new_edlp_is_not_one:{path.name}")
        for row in trace:
            if bool(row.get("incumbent_updated")) and not bool(row.get("true_strict_improvement")):
                raise SelectedSplitAuditError(f"pilot_incumbent_update_not_strict:{path.name}")
            if bool(row.get("threshold_updated")) and not bool(row.get("true_strict_improvement")):
                raise SelectedSplitAuditError(f"pilot_threshold_update_not_strict:{path.name}")
        per_scenario.setdefault(scenario_id, []).append(payload)
    if len(per_scenario) != 4 or any(len(group) != 20 for group in per_scenario.values()):
        raise SelectedSplitAuditError("pilot_scenario_run_distribution_is_not_four_times_twenty")
    if any(len(indices) != 1 for indices in successful_indices.values()):
        raise SelectedSplitAuditError("pilot_improved_state_not_consistent_within_scenario")
    if any(len(costs) != 1 for costs in successful_costs.values()):
        raise SelectedSplitAuditError("pilot_improved_cost_not_consistent_within_scenario")
    if int(summary.get("planned_runs", -1)) != 80:
        raise SelectedSplitAuditError("pilot_summary_planned_runs_mismatch")
    if int(summary.get("completed_runs", -1)) != len(completed):
        raise SelectedSplitAuditError("pilot_summary_completed_runs_mismatch")
    if int(summary.get("outside_training_strict_improvement_run_count", -1)) != len(completed):
        raise SelectedSplitAuditError("pilot_summary_strict_improvement_runs_mismatch")
    return {
        "scope": "read-only audit of persisted targeted-pilot JSON; no algorithm execution",
        "completed_runs": len(completed), "scenarios": sorted(per_scenario), "runs_per_scenario": 20,
        "all_runs_have_outside_training_strict_improvement": True,
        "all_runs_have_one_search_stage_new_edlp": True,
        "scenario_improved_state_and_cost": {
            scenario_id: {"state_index": next(iter(successful_indices[scenario_id])), "true_cost": next(iter(successful_costs[scenario_id]))}
            for scenario_id in sorted(per_scenario)
        },
        "summary_consistent": True,
    }
