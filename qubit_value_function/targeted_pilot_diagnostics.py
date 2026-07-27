"""Pure, read-only helpers for the targeted best-training pilot."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import median, stdev
from typing import Mapping, Sequence


QUANTUM_METHODS = frozenset(("joint_bbht", "cost_only_bbht"))


def grover_theoretical_probability(
    *, search_space_size: int, marked_count: int, iterations: int
) -> float:
    """Return the standard Grover probability for the supplied oracle set."""

    size = int(search_space_size)
    marked = int(marked_count)
    if size <= 0 or not 0 <= marked <= size or int(iterations) < 0:
        raise ValueError("invalid Grover probability parameters")
    if marked == 0:
        return 0.0
    if marked == size:
        return 1.0
    theta = math.asin(math.sqrt(marked / size))
    return math.sin((2 * int(iterations) + 1) * theta) ** 2


def initial_marked_counts(
    snapshot: Mapping[str, object], *, encoded_threshold: int
) -> dict[str, object]:
    """Derive initial sets using only persisted surrogate and training facts."""

    training = {int(item) for item in snapshot["training_indices"]}  # type: ignore[index]
    table = list(snapshot["state_proxy_table"])  # type: ignore[index]
    cost = sorted(
        int(row["state_index"])
        for row in table
        if int(row["integer_vqc_value"]) < int(encoded_threshold)
    )
    feasible = {
        int(row["state_index"])
        for row in table
        if bool(row["hard_logic_feasible"])
    }
    joint = [index for index in cost if index in feasible]
    nontraining_cost = [index for index in cost if index not in training]
    nontraining_joint = [index for index in joint if index not in training]
    training_joint = [index for index in joint if index in training]
    return {
        "initial_cost_marked_count": len(cost),
        "initial_joint_marked_count": len(joint),
        "initial_nontraining_cost_marked_count": len(nontraining_cost),
        "initial_nontraining_joint_marked_count": len(nontraining_joint),
        "initial_training_joint_marked_count": len(training_joint),
        "initial_cost_marked_indices": cost,
        "initial_joint_marked_indices": joint,
    }


def classify_quantum_trial(*, method: str, row: Mapping[str, object]) -> str:
    """Classify one persisted quantum trial without changing it."""

    if method == "joint_bbht":
        if not bool(row.get("measured_joint_marked")):
            return "unmarked_measurement"
    elif method == "cost_only_bbht":
        if not bool(row.get("measured_cost_marked")):
            return "unmarked_measurement"
        if not bool(row.get("hard_logic_feasible_measured")):
            return "cost_marked_hard_logic_infeasible"
    else:
        raise ValueError(f"not a quantum pilot method: {method}")
    source = str(row.get("cache_source", "none"))
    if source == "training_cache":
        return "marked_training_cache"
    if source == "search_cache":
        return "marked_search_cache"
    if bool(row.get("new_ed_lp_solve")):
        if bool(row.get("true_strict_improvement")) and bool(row.get("improvement_is_nontraining")):
            return "marked_new_edlp_nontraining_improvement"
        return "marked_new_edlp_no_improvement"
    return "marked_other_persisted_outcome"


def _percentile(values: Sequence[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _first_nontraining_improvement(
    trials: Sequence[Mapping[str, object]],
) -> tuple[int, int, int] | None:
    new_edlp = 0
    for fallback_index, row in enumerate(trials, start=1):
        if bool(row.get("new_ed_lp_solve")):
            new_edlp += 1
        if bool(row.get("true_strict_improvement")) and bool(row.get("improvement_is_nontraining")):
            return int(row.get("trial_index", fallback_index)), new_edlp, int(row["measured_index"])
    return None


def summarize_runs(runs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Summarize persisted run records at run granularity, never trial totals."""

    groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for run in runs:
        groups[(str(run["scenario_id"]), str(run["method"]))].append(run)
    summaries: list[dict[str, object]] = []
    for (scenario_id, method), group in sorted(groups.items()):
        search_counts = [int(run.get("search_stage_new_edlp", 0)) for run in group]
        training = [run.get("training_edlp") for run in group]
        has_training = all(value is not None for value in training)
        firsts = {
            str(run["run_id"]): first
            for run in group
            if (first := _first_nontraining_improvement(list(run.get("trials", [])))) is not None
        }
        global_successes = [
            run for run in group
            if bool(dict(run.get("validation", {})).get("nontraining_global_optimum_success"))
        ]
        result: dict[str, object] = {
            "scenario_id": scenario_id,
            "method": method,
            "runs": len(group),
            "runs_with_nontraining_strict_improvement": len(firsts),
            "nontraining_strict_improvement_probability": len(firsts) / len(group) if group else None,
            "runs_with_nontraining_global_optimum": len(global_successes),
            "nontraining_global_optimum_probability": len(global_successes) / len(group) if group else None,
            "total_search_stage_new_edlp": sum(search_counts),
            "mean_search_stage_new_edlp_per_run": sum(search_counts) / len(search_counts) if search_counts else None,
            "median_search_stage_new_edlp_per_run": median(search_counts) if search_counts else None,
            "q25_search_stage_new_edlp_per_run": _percentile(search_counts, 0.25),
            "q75_search_stage_new_edlp_per_run": _percentile(search_counts, 0.75),
            "standard_deviation_search_stage_new_edlp_per_run": stdev(search_counts) if len(search_counts) >= 2 else None,
            "first_nontraining_improvement_trial": {run_id: value[0] for run_id, value in firsts.items()},
            "first_nontraining_improvement_new_edlp_count": {run_id: value[1] for run_id, value in firsts.items()},
            "successful_state_index": {run_id: value[2] for run_id, value in firsts.items()},
            "successful_state_is_global_optimum": {
                str(run["run_id"]): bool(dict(run.get("validation", {})).get("nontraining_global_optimum_success"))
                for run in group if str(run["run_id"]) in firsts
            },
        }
        if has_training:
            training_counts = [int(value) for value in training]
            result["total_training_edlp"] = sum(training_counts)
            result["total_training_plus_search_edlp"] = sum(training_counts) + sum(search_counts)
        summaries.append(result)
    return summaries
