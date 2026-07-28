"""Frozen-snapshot selected-split best-training benchmark wiring."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_targeted_best_training_pilot_cli import (  # noqa: E402
    _load_snapshot,
    _snapshot_threshold,
    augment_targeted_result_schema,
    ensure_new_output_dir,
    restore_snapshot_scenario,
)
from qubit_value_function.closed_loop_batch import (  # noqa: E402
    DEFAULT_GENERATOR_PAIRS,
    METHODS,
    MPS_METHODS,
    ClosedLoopBatchExecutor,
    RunSpec,
    atomic_write_json,
    build_run_specs,
)
from qubit_value_function.closed_loop_scenario import run_closed_loop_method  # noqa: E402
from qubit_value_function.targeted_pilot_diagnostics import initial_marked_counts  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


FORMAL_RUN_SEEDS = tuple(range(5))
FORMAL_BUDGET = {
    "lambda_factor": 1.2,
    "max_trials": 24,
    "max_oracle_calls": 48,
    "max_new_ed_lp_calls": 8,
    "max_threshold_updates": 3,
}


class SelectedSplitBenchmarkError(RuntimeError):
    """The fixed selected-split benchmark cannot safely be started."""


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen selected-split best-training benchmark; not formal or quantum-advantage evidence."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--expected-head")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser


def _scenario_facts(snapshot: Mapping[str, object]) -> dict[str, object]:
    training_indices = tuple(int(index) for index in snapshot["training_indices"])
    labels = [dict(row) for row in snapshot["training_labels"]]
    if len(training_indices) != 8 or len(labels) != 8:
        raise SelectedSplitBenchmarkError("snapshot_training_sample_count_is_not_eight")
    if {int(row["state_index"]) for row in labels} != set(training_indices):
        raise SelectedSplitBenchmarkError("snapshot_training_labels_do_not_match_indices")
    if sorted(int(index) for index in snapshot["initial_cache_indices"]) != sorted(training_indices):
        raise SelectedSplitBenchmarkError("snapshot_initial_cache_is_not_training_cache")
    label_costs = {int(row["state_index"]): float(row["true_cost"]) for row in labels}
    incumbent = int(snapshot["initial_incumbent_index"])
    if incumbent not in label_costs or label_costs[incumbent] != min(label_costs.values()):
        raise SelectedSplitBenchmarkError("snapshot_initial_incumbent_is_not_best_training")
    threshold = _snapshot_threshold(snapshot)
    counts = initial_marked_counts(snapshot, encoded_threshold=threshold)
    nontraining_joint = [
        int(index) for index in counts["initial_joint_marked_indices"] if int(index) not in set(training_indices)
    ]
    return {
        "scenario_id": str(snapshot["scenario_id"]),
        "training_data_seed": int(snapshot["training_data_seed"]),
        "model_seed": int(snapshot["model_seed"]),
        "training_indices": list(training_indices),
        "training_indices_hash": str(snapshot["training_indices_hash"]),
        "training_sample_count": len(training_indices),
        "initial_incumbent_index": incumbent,
        "initial_incumbent_true_cost": label_costs[incumbent],
        "initial_encoded_threshold": threshold,
        "initial_marked_counts": counts,
        "initial_nontraining_joint_marked_indices": nontraining_joint,
        "oracle_reachability_stratum": "reachable" if nontraining_joint else "unreachable",
    }


def _validate_manifest_shape(manifest: Mapping[str, object]) -> None:
    if tuple(manifest.get("methods", ())) != METHODS:
        raise SelectedSplitBenchmarkError("formal_method_list_mismatch")
    if tuple(int(seed) for seed in manifest.get("run_seeds", ())) != FORMAL_RUN_SEEDS:
        raise SelectedSplitBenchmarkError("selected_split_requires_run_seeds_0_through_4")
    if str(manifest.get("initial_incumbent_policy")) != "best_training":
        raise SelectedSplitBenchmarkError("selected_split_requires_best_training_initialization")
    if dict(manifest.get("bbht_config", {})) != FORMAL_BUDGET:
        raise SelectedSplitBenchmarkError("formal_budget_config_mismatch")
    scenarios = list(manifest.get("scenarios", []))
    expected = {(pair, window) for pair in DEFAULT_GENERATOR_PAIRS for window in range(3)}
    actual = {(tuple(int(value) for value in item["generator_pair"]), int(item["window_start"])) for item in scenarios}
    if len(scenarios) != 12 or actual != expected:
        raise SelectedSplitBenchmarkError("selected_split_requires_all_twelve_base_scenarios")


def _git_value(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def build_runtime_provenance(
    *,
    actual_execution_head: str,
    manifest_declared_code_sha: str,
    externally_expected_head: str | None,
    working_tree_tracked_clean: bool,
) -> dict[str, object]:
    """Keep runtime identity separate from immutable manifest provenance."""

    if externally_expected_head is not None and externally_expected_head != actual_execution_head:
        raise SelectedSplitBenchmarkError("expected_head_mismatch")
    return {
        "actual_execution_head": str(actual_execution_head),
        "manifest_declared_code_sha": str(manifest_declared_code_sha),
        "externally_expected_head": externally_expected_head,
        "head_match": externally_expected_head == actual_execution_head if externally_expected_head is not None else None,
        "working_tree_tracked_clean": bool(working_tree_tracked_clean),
    }


def preflight_selected_split_manifest(
    manifest: Mapping[str, object], *, runtime_provenance: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Validate only persisted snapshot/frozen formal metadata; never search or evaluate ED/LP."""

    _validate_manifest_shape(manifest)
    source_path = Path(str(manifest["methods_source"]))
    try:
        formal_source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectedSplitBenchmarkError(f"formal_methods_source_unavailable:{source_path}") from error
    if set(formal_source.get("methods", ())) != set(METHODS):
        raise SelectedSplitBenchmarkError("formal_methods_source_mismatch")
    if dict(formal_source.get("budget_config", {})) != FORMAL_BUDGET:
        raise SelectedSplitBenchmarkError("formal_budget_source_mismatch")
    rows: list[dict[str, object]] = []
    for item in manifest["scenarios"]:
        snapshot = _load_snapshot(Path(str(item["snapshot"])))
        if str(snapshot["scenario_id"]) != str(item["scenario_id"]):
            raise SelectedSplitBenchmarkError("snapshot_scenario_id_mismatch")
        if tuple(int(value) for value in snapshot["generator_pair"]) != tuple(int(value) for value in item["generator_pair"]):
            raise SelectedSplitBenchmarkError("snapshot_generator_pair_mismatch")
        if int(snapshot["window_start"]) != int(item["window_start"]):
            raise SelectedSplitBenchmarkError("snapshot_window_mismatch")
        facts = _scenario_facts(snapshot)
        if facts["training_data_seed"] != 1 or facts["model_seed"] != 0:
            raise SelectedSplitBenchmarkError("selected_split_snapshot_seed_mismatch")
        rows.append({"snapshot": str(item["snapshot"]), **facts})
    reachable = sum(row["oracle_reachability_stratum"] == "reachable" for row in rows)
    report = {
        "selection_status": "eligible",
        "formal": False,
        "global_optimum_used_in_preflight": False,
        "methods": list(METHODS),
        "run_seeds": list(FORMAL_RUN_SEEDS),
        "planned_runs": 360,
        "scenarios": rows,
        "oracle_reachability_counts": {"reachable": reachable, "unreachable": len(rows) - reachable},
    }
    return {**report, **dict(runtime_provenance or {})}


def build_selected_split_run_specs(manifest: Mapping[str, object], *, expected_head: str) -> tuple[RunSpec, ...]:
    _validate_manifest_shape(manifest)
    specs = build_run_specs(
        batch_id=str(manifest["experiment_name"]),
        preset="selected_split_best_training",
        generator_pairs=DEFAULT_GENERATOR_PAIRS,
        windows=(0, 1, 2),
        training_seeds=(1,),
        methods=METHODS,
        run_seeds=FORMAL_RUN_SEEDS,
        budget_config=FORMAL_BUDGET,
        fixed_point_config={"fractional_bits": 2, "cost_unit": 1000.0, "rounding": "nearest"},
        initialization_policy="best_training",
        expected_code_sha=expected_head,
    )
    if len(specs) != 360 or len({spec.run_id for spec in specs}) != 360:
        raise SelectedSplitBenchmarkError("selected_split_run_plan_is_not_360_unique_runs")
    return specs


def _counter(result: Mapping[str, object], *names: str) -> int:
    counters = result.get("counters", {})
    return next((int(counters[name]) for name in names if isinstance(counters, Mapping) and isinstance(counters.get(name), (int, float))), 0)


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"total": 0.0, "mean": None, "median": None, "q25": None, "q75": None, "standard_deviation": None}
    array = np.asarray(values, dtype=float)
    return {
        "total": float(array.sum()), "mean": float(array.mean()), "median": float(np.quantile(array, 0.5)),
        "q25": float(np.quantile(array, 0.25)), "q75": float(np.quantile(array, 0.75)),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else None,
    }


def compute_selected_split_summary(output_dir: Path, *, planned_runs: int = 360) -> dict[str, object]:
    """Summarize completed JSON only; this function never invokes a search method."""

    rows: list[dict[str, object]] = []
    for path in sorted((Path(output_dir) / "runs" / "completed").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result, scenario = dict(payload["result"]), dict(payload["scenario"])
        trace = [dict(row) for row in result.get("trial_trace", [])]
        initial, final = float(result["initial_incumbent_true_cost"]), float(result["final_incumbent_true_cost"])
        strict = any(bool(row.get("true_strict_improvement")) for row in trace)
        outside_strict = any(bool(row.get("true_strict_improvement")) and not bool(row.get("candidate_in_training_set")) for row in trace)
        rows.append({
            "scenario_id": str(scenario["scenario_id"]), "method": str(payload["method"]),
            "oracle_reachability_stratum": str(result["oracle_reachability_stratum"]),
            "strict": strict, "outside_strict": outside_strict, "initial": initial, "final": final,
            "proposals": _counter(result, "proposals_used", "circuit_executions"),
            "oracle_calls": _counter(result, "total_oracle_calls"),
            "mps_executions": _counter(result, "circuit_executions") if payload["method"] in MPS_METHODS else 0,
            "cache_hits": _counter(result, "cached_exact_lookups"),
            "cache_misses": _counter(result, "new_exact_evaluation_attempts"),
            "search_edlp": _counter(result, "actual_ed_lp_solves"),
            "stop_reason": str(result.get("stop_reason")),
        })
    def grouped(fields: Sequence[str]) -> list[dict[str, object]]:
        buckets: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            buckets[tuple(row[field] for field in fields)].append(row)
        output: list[dict[str, object]] = []
        for key, group in sorted(buckets.items(), key=lambda item: tuple(str(x) for x in item[0])):
            record = dict(zip(fields, key))
            record.update({
                "runs": len(group), "strict_improvement_runs": sum(row["strict"] for row in group),
                "strict_improvement_rate": sum(row["strict"] for row in group) / len(group),
                "outside_training_strict_improvement_runs": sum(row["outside_strict"] for row in group),
                "outside_training_strict_improvement_rate": sum(row["outside_strict"] for row in group) / len(group),
                "final_true_cost": _percentiles([row["final"] for row in group]),
                "absolute_improvement": _percentiles([row["initial"] - row["final"] for row in group]),
                "relative_improvement": _percentiles([(row["initial"] - row["final"]) / row["initial"] for row in group]),
                "search_stage_new_edlp": _percentiles([row["search_edlp"] for row in group]),
                "proposals": sum(row["proposals"] for row in group), "oracle_calls": sum(row["oracle_calls"] for row in group),
                "mps_executions": sum(row["mps_executions"] for row in group), "cache_hits": sum(row["cache_hits"] for row in group),
                "cache_misses": sum(row["cache_misses"] for row in group),
                "stop_reasons": {reason: sum(row["stop_reason"] == reason for row in group) for reason in sorted({row["stop_reason"] for row in group})},
            })
            output.append(record)
        return output
    failed = list((Path(output_dir) / "runs" / "failed").glob("*.json"))
    summary = {
        "scope": "selected-split descriptive baseline; no claim of end-to-end quantum advantage",
        "planned_runs": planned_runs, "completed_runs": len(rows), "failed_runs": len(failed),
        "global_optimum_posthoc_status": "not_assessed",
        "overall": grouped(())[0] if rows else {"runs": 0},
        "by_method": grouped(("method",)), "by_scenario": grouped(("scenario_id",)),
        "by_reachability_stratum": grouped(("oracle_reachability_stratum",)),
        "by_method_and_reachability_stratum": grouped(("method", "oracle_reachability_stratum")),
    }
    return summary


def summarize_selected_split_batch(output_dir: Path, *, planned_runs: int = 360) -> dict[str, object]:
    """Persist the unchanged selected-split summary schema after pure recomputation."""

    summary = compute_selected_split_summary(output_dir, planned_runs=planned_runs)
    atomic_write_json(Path(output_dir) / "summaries" / "selected_split_summary.json", summary)
    return summary


def main() -> int:
    args = build_argument_parser().parse_args()
    manifest = json.loads(args.config.read_text(encoding="utf-8"))
    actual_execution_head = _git_value("rev-parse", "HEAD")
    tracked_clean = all(line[:2] == "??" for line in _git_value("status", "--porcelain").splitlines())
    runtime_provenance = build_runtime_provenance(
        actual_execution_head=actual_execution_head,
        manifest_declared_code_sha=str(manifest["expected_code_sha"]),
        externally_expected_head=args.expected_head,
        working_tree_tracked_clean=tracked_clean,
    )
    report = preflight_selected_split_manifest(manifest, runtime_provenance=runtime_provenance)
    specs = build_selected_split_run_specs(manifest, expected_head=actual_execution_head)
    if args.preflight:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    ensure_new_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True)
    atomic_write_json(args.output_dir / "selected_split_execution_manifest.json", {**report, "source_manifest": str(args.config)})
    snapshots = {str(item["scenario_id"]): _load_snapshot(Path(str(item["snapshot"]))) for item in manifest["scenarios"]}
    snapshot_paths = {str(item["scenario_id"]): Path(str(item["snapshot"])) for item in manifest["scenarios"]}
    facts = {scenario_id: _scenario_facts(snapshot) for scenario_id, snapshot in snapshots.items()}
    source = load_uc_instance(args.instance)
    def scenario_builder(spec: RunSpec):
        return restore_snapshot_scenario(snapshot=snapshots[spec.scenario_id], snapshot_path=snapshot_paths[spec.scenario_id], source=source, bbht_config=FORMAL_BUDGET)
    def method_runner(scenario, method: str, run_seed: int):
        envelope = run_closed_loop_method(scenario, method, run_seed=run_seed, persist_dynamic_oracle_metadata=True)
        fact = facts[scenario.scenario_id]
        result = augment_targeted_result_schema(dict(envelope["result_schema"]), training_indices=scenario.training_indices, initial_nontraining_joint_marked_indices=fact["initial_nontraining_joint_marked_indices"])
        envelope["result_schema"] = {**result, **fact, "global_optimum_posthoc_status": "not_assessed"}
        envelope["quantized_model_snapshot"] = snapshots[scenario.scenario_id]
        return envelope
    executor = ClosedLoopBatchExecutor(
        output_dir=args.output_dir,
        code={"execution": "frozen_selected_split_best_training", "head": actual_execution_head, **runtime_provenance},
        scenario_builder=scenario_builder,
        method_runner=method_runner,
        workers=1,
    )
    counts = executor.execute(specs, continue_on_error=True)
    summary = summarize_selected_split_batch(args.output_dir, planned_runs=len(specs))
    print(json.dumps({"preflight": report, "counts": counts, "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
