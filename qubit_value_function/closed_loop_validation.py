"""阶段 A 搜索完成后的只读真值景观验证与增强汇总。

这里绝不调用 ``run_closed_loop_method``。验证只重建冻结场景、复用已落盘
exact cache，并仅对尚未观察到的硬逻辑可行状态调用同一个 exact evaluator。
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Sequence

from .candidate_acceptance_loop import ExactCandidateEvaluation
from .closed_loop_batch import atomic_write_json


VALIDATION_SCHEMA_VERSION = "stage-a-closed-loop-validation-v1"


class ValidationConsistencyError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bits(index: int, width: int) -> tuple[int, ...]:
    return tuple((int(index) >> bit) & 1 for bit in range(width))


def _record_dict(record: ExactCandidateEvaluation | Mapping[str, object], source: str) -> dict[str, object]:
    if isinstance(record, ExactCandidateEvaluation):
        payload = record.as_dict()
    else:
        payload = dict(record)
    return {
        "success": bool(payload.get("success", False)),
        "total_cost": (float(payload["total_cost"]) if payload.get("total_cost") is not None else None),
        "message": str(payload.get("message", "")),
        "source": source,
        "lp_solve_performed": bool(payload.get("lp_solve_performed", False)),
        "logic_precheck_rejected": bool(payload.get("logic_precheck_rejected", False)),
    }


def _same_record(left: Mapping[str, object], right: Mapping[str, object], abs_tol: float, rel_tol: float) -> bool:
    if bool(left["success"]) != bool(right["success"]):
        return False
    if not left["success"]:
        return True
    return math.isclose(float(left["total_cost"]), float(right["total_cost"]), abs_tol=abs_tol, rel_tol=rel_tol)


def _collect_reusable_cache(
    scenario: object,
    runs: Sequence[Mapping[str, object]],
    *, abs_tol: float, rel_tol: float,
) -> dict[int, dict[str, object]]:
    cache: dict[int, dict[str, object]] = {}

    def add(index: int, record: ExactCandidateEvaluation | Mapping[str, object], source: str) -> None:
        candidate = _record_dict(record, source)
        old = cache.get(int(index))
        if old is not None and not _same_record(old, candidate, abs_tol, rel_tol):
            raise ValidationConsistencyError(f"index {index} 的已观察 exact cache 真值不一致")
        cache[int(index)] = old or candidate

    for index, record in getattr(scenario, "initial_exact_cache").items():
        add(int(index), record, "initial_training_cache")
    for run in runs:
        result = run.get("result", {})
        if not isinstance(result, Mapping):
            raise ValidationConsistencyError("completed run 缺少 result")
        exact_cache = result.get("exact_cache", {})
        if not isinstance(exact_cache, Mapping):
            raise ValidationConsistencyError("completed run exact_cache 无效")
        for raw_index, record in exact_cache.items():
            if not isinstance(record, Mapping):
                raise ValidationConsistencyError("exact_cache record 无效")
            add(int(raw_index), record, "reused_search_cache")
    return cache


def _encode(model: object, cost: float) -> int:
    if hasattr(model, "encode_true_cost"):
        return int(model.encode_true_cost(cost))
    return int(model.fixed_point_config.encode(cost))


def validate_scenario_landscape(
    scenario: object,
    runs: Sequence[Mapping[str, object]],
    *, abs_tol: float = 1e-6, rel_tol: float = 1e-9,
) -> dict[str, object]:
    """扫描完整有限空间；所有写入都由调用者落入独立 validation 目录。"""

    started = perf_counter()
    model = scenario.value_model
    width = int(model.num_x_qubits)
    dimension = 2 ** width
    cache = _collect_reusable_cache(scenario, runs, abs_tol=abs_tol, rel_tol=rel_tol)
    rows: list[dict[str, object]] = []
    feasible: list[int] = []
    successful: list[int] = []
    failed: list[int] = []
    reused = 0
    new_solves = 0
    for index in range(dimension):
        bits = _bits(index, width)
        logic_ok = bool(scenario.hard_logic_is_feasible(bits))
        row: dict[str, object] = {"index": index, "hard_logic_feasible": logic_ok}
        if not logic_ok:
            row.update({"exact_evaluated": False, "validation_source": "logic_infeasible", "exact_record": None})
            rows.append(row)
            continue
        feasible.append(index)
        record = cache.get(index)
        if record is not None:
            reused += 1
            row.update({"exact_evaluated": True, "validation_source": record["source"], "exact_record": record})
        else:
            exact = scenario.evaluate_candidate(index)
            record = _record_dict(exact, "new_validation_ed_lp")
            row.update({"exact_evaluated": True, "validation_source": "new_validation_ed_lp", "exact_record": record})
            if record["lp_solve_performed"]:
                new_solves += 1
        if record["success"]:
            successful.append(index)
        else:
            failed.append(index)
        rows.append(row)
    if not successful:
        raise ValidationConsistencyError("真值景观中没有 exact-successful 状态")
    costs = {int(row["index"]): float(row["exact_record"]["total_cost"]) for row in rows if row.get("exact_record") and row["exact_record"]["success"]}
    optimum = min(costs.values())
    optimum_indices = sorted(index for index, cost in costs.items() if math.isclose(cost, optimum, abs_tol=abs_tol, rel_tol=rel_tol))
    initial_index = int(scenario.initial_incumbent_index)
    initial_cost = float(cache[initial_index]["total_cost"])
    encoded = _encode(model, initial_cost)
    cost_marked = [index for index in range(dimension) if int(model.integer_value(_bits(index, width))) < encoded]
    joint_marked = [index for index in cost_marked if index in feasible]
    real_improvements = sorted(index for index, cost in costs.items() if cost < initial_cost and not math.isclose(cost, initial_cost, abs_tol=abs_tol, rel_tol=rel_tol))
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "scenario_id": str(scenario.scenario_id),
        "generator_pair": list(scenario.generator_pair), "window_start": int(scenario.window_start), "training_seed": int(scenario.training_seed),
        "search_space_dimension": dimension,
        "hard_logic_feasible_indices": feasible, "exact_successful_indices": successful, "exact_failed_indices": failed,
        "true_cost_by_index": {str(index): cost for index, cost in sorted(costs.items())},
        "true_global_optimum_index": optimum_indices[0], "true_global_optimum_cost": optimum,
        "true_global_optimum_indices": optimum_indices,
        "comparison_tolerance": {"abs_tol": abs_tol, "rel_tol": rel_tol},
        "initial_incumbent_index": initial_index, "initial_incumbent_true_cost": initial_cost,
        "initial_incumbent_is_global_optimum": initial_index in optimum_indices,
        "initial_optimality_gap": initial_cost - optimum,
        "initial_true_improvement_exists": bool(real_improvements),
        "initial_surrogate_cost_marked_count": len(cost_marked), "initial_joint_marked_count": len(joint_marked),
        "initial_cost_marked_indices": cost_marked, "initial_joint_marked_indices": joint_marked,
        "real_improvement_indices": real_improvements,
        "real_improvement_exists_but_cost_marked_empty": bool(real_improvements and not cost_marked),
        "real_improvement_exists_but_joint_marked_empty": bool(real_improvements and not joint_marked),
        "reused_cache_records": reused, "new_validation_ed_lp_solves": new_solves,
        "state_rows": rows, "validation_elapsed_seconds": perf_counter() - started,
    }


def trace_metrics(result: Mapping[str, object], *, wrapper_elapsed_seconds: float | None = None) -> dict[str, object]:
    trace = result.get("trial_trace", [])
    if not isinstance(trace, list):
        trace = []
    count = lambda key: sum(bool(row.get(key, False)) for row in trace if isinstance(row, Mapping))
    admission_logic = sum(row.get("admission_rejection_reason") == "hard_logic_infeasible" for row in trace if isinstance(row, Mapping))
    exact_logic = sum(bool(row.get("logic_precheck_rejections_added", 0)) for row in trace if isinstance(row, Mapping))
    depths = [int(row.get("circuit_resources", {}).get("depth", 0)) for row in trace if isinstance(row, Mapping) and isinstance(row.get("circuit_resources"), Mapping)]
    qubits = [int(row.get("total_qubits", 0)) for row in trace if isinstance(row, Mapping) and row.get("total_qubits") is not None]
    return {
        "proposal_events": len(trace), "unique_candidate_events": count("is_unique_candidate"), "repeated_candidate_events": count("is_repeated_candidate"),
        "cache_hits": count("candidate_cache_hit"), "cache_confirmed_improvements": count("cache_confirmed_improvement"),
        "new_edlp_confirmed_improvements": count("new_edlp_confirmed_improvement"), "nontraining_true_improvements": count("nontraining_true_improvement"),
        "first_nontraining_improvement": count("first_nontraining_improvement"),
        "admission_hard_logic_rejections": admission_logic, "exact_logic_precheck_rejections": exact_logic,
        "total_logic_rejections": admission_logic + exact_logic, "auxiliary_syndrome_rejections": sum(row.get("admission_rejection_reason") == "auxiliary_syndrome_rejected" for row in trace if isinstance(row, Mapping)),
        "surrogate_unmarked_rejections": sum(row.get("admission_rejection_reason") == "surrogate_unmarked" for row in trace if isinstance(row, Mapping)),
        "cost_marked_events": count("cost_marked"), "joint_marked_events": count("joint_marked"),
        "actual_ed_lp_solves": sum(int(row.get("actual_ed_lp_solves_added", 0)) for row in trace if isinstance(row, Mapping)),
        "new_exact_evaluation_attempts": sum(int(row.get("new_exact_evaluation_attempts_added", 0)) for row in trace if isinstance(row, Mapping)),
        "oracle_calls": sum(int(row.get("oracle_calls_added", 0)) for row in trace if isinstance(row, Mapping)),
        "grover_iterations": sum(int(row.get("grover_iterations", 0)) for row in trace if isinstance(row, Mapping)),
        "mps_circuit_executions": sum(bool(row.get("quantum_resources", {}).get("applicable", False)) for row in trace if isinstance(row, Mapping) and isinstance(row.get("quantum_resources"), Mapping)),
        "mps_trial_elapsed_sum": sum(float(row.get("elapsed_seconds", 0.0)) for row in trace if isinstance(row, Mapping)),
        "method_wrapper_elapsed": wrapper_elapsed_seconds, "maximum_qubits": max(qubits, default=0), "maximum_circuit_depth": max(depths, default=0),
    }


def edlp_budget_curve(result: Mapping[str, object]) -> dict[int, float]:
    curve = {0: float(result["initial_incumbent_true_cost"])}
    solves = 0
    for row in result.get("trial_trace", []):
        if not isinstance(row, Mapping):
            continue
        after = row.get("true_threshold_after")
        if after is None:
            continue
        solves += int(row.get("actual_ed_lp_solves_added", 0))
        curve[solves] = float(after)
    return curve


def validate_run_against_landscape(run: Mapping[str, object], landscape: Mapping[str, object], *, abs_tol: float, rel_tol: float) -> dict[str, object]:
    result = run["result"]
    assert isinstance(result, Mapping)
    costs = {int(key): float(value) for key, value in landscape["true_cost_by_index"].items()}
    optimum = float(landscape["true_global_optimum_cost"])
    final_index, final_cost = int(result["final_incumbent_index"]), float(result["final_incumbent_true_cost"])
    initial_cost = float(result["initial_incumbent_true_cost"])
    rank = 1 + sum(cost < final_cost and not math.isclose(cost, final_cost, abs_tol=abs_tol, rel_tol=rel_tol) for cost in costs.values())
    trace = result.get("trial_trace", [])
    reached_cache = any(row.get("accepted_update") and row.get("candidate_index") == final_index and row.get("candidate_cache_hit") for row in trace if isinstance(row, Mapping))
    reached_new = any(row.get("accepted_update") and row.get("candidate_index") == final_index and row.get("new_edlp_confirmed_improvement") for row in trace if isinstance(row, Mapping))
    training = set(run.get("scenario", {}).get("training_indices", [])) if isinstance(run.get("scenario"), Mapping) else set()
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION, "run_id": run["run_id"], "fingerprint": run["fingerprint"],
        "scenario_id": landscape["scenario_id"], "method": run.get("method"), "method_role": run.get("method_role"), "diagnostic_only": bool(run.get("diagnostic_only", False)),
        "initial_incumbent_is_global_optimum": int(result["initial_incumbent_index"]) in landscape["true_global_optimum_indices"],
        "initial_optimality_gap": initial_cost - optimum, "initial_true_improvement_exists": landscape["initial_true_improvement_exists"],
        "final_incumbent_index": final_index, "final_incumbent_true_cost": final_cost,
        "final_is_global_optimum": math.isclose(final_cost, optimum, abs_tol=abs_tol, rel_tol=rel_tol), "final_optimality_gap": final_cost - optimum,
        "final_rank_among_successful_states": rank, "reached_global_optimum": final_index in landscape["true_global_optimum_indices"],
        "reached_global_optimum_through_cache": reached_cache, "reached_global_optimum_through_new_edlp": reached_new,
        "nontraining_global_optimum_hit": final_index in landscape["true_global_optimum_indices"] and final_index not in training,
        "true_cost_decrease": final_cost < initial_cost and not math.isclose(final_cost, initial_cost, abs_tol=abs_tol, rel_tol=rel_tol),
        "true_cost_decrease_amount": initial_cost - final_cost, "threshold_update_count": int(result.get("counters", {}).get("threshold_updates", 0)),
        "trace_metrics": trace_metrics(result, wrapper_elapsed_seconds=run.get("timing", {}).get("elapsed_seconds") if isinstance(run.get("timing"), Mapping) else None),
        "edlp_budget_curve": {str(key): value for key, value in edlp_budget_curve(result).items()},
    }


def _read_json(path: Path) -> dict[str, object]:
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error: raise ValidationConsistencyError(f"无法读取 {path}: {error}") from error
    if not isinstance(value, dict): raise ValidationConsistencyError(f"{path} 不是 JSON object")
    return value


def _validate_source(manifest: Mapping[str, object], runs: Sequence[Mapping[str, object]], expected_search_head: str | None) -> str:
    head = str(manifest.get("expected_head", ""))
    if expected_search_head is not None and head != expected_search_head: raise ValidationConsistencyError("source manifest search HEAD 不匹配")
    fingerprints = manifest.get("planned_fingerprints", {})
    if not isinstance(fingerprints, Mapping): raise ValidationConsistencyError("source manifest 缺少 planned_fingerprints")
    for run in runs:
        run_id = run.get("run_id")
        if run.get("status") != "completed" or not isinstance(run_id, str): raise ValidationConsistencyError("source run 不是 completed")
        if run.get("fingerprint") != fingerprints.get(run_id): raise ValidationConsistencyError("source run fingerprint 不匹配")
        if not isinstance(run.get("code"), Mapping) or run["code"].get("head") != head: raise ValidationConsistencyError("source run search code SHA 不一致")
        spec, scenario = run.get("run_spec"), run.get("scenario")
        if not isinstance(spec, Mapping) or not isinstance(scenario, Mapping) or spec.get("scenario_id") != scenario.get("scenario_id"): raise ValidationConsistencyError("source run scenario_id 不一致")
        if spec.get("budget_config") != manifest.get("budget_config") or spec.get("fixed_point_config") != manifest.get("fixed_point_config"): raise ValidationConsistencyError("source run budget/fixed-point 不一致")
    return head


def validate_source_batch(
    source_output_dir: Path, *, scenario_builder: Callable[[Mapping[str, object]], object], validation_code_sha: str,
    expected_search_head: str | None = None, resume: bool = False, abs_tol: float = 1e-6, rel_tol: float = 1e-9,
) -> dict[str, object]:
    source_output_dir = Path(source_output_dir)
    manifest = _read_json(source_output_dir / "batch_manifest.json")
    runs = [_read_json(path) for path in sorted((source_output_dir / "runs" / "completed").glob("*.json"))]
    search_head = _validate_source(manifest, runs, expected_search_head)
    validation_root = source_output_dir / "validation"
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in runs: grouped.setdefault(str(run["scenario"]["scenario_id"]), []).append(run)
    scenarios_done = runs_done = 0
    for scenario_id, group in grouped.items():
        scenario_path = validation_root / "scenarios" / f"{scenario_id}.json"
        if resume and scenario_path.exists():
            landscape = _read_json(scenario_path)
            if (
                landscape.get("schema_version") != VALIDATION_SCHEMA_VERSION
                or landscape.get("scenario_id") != scenario_id
                or landscape.get("search_code_sha") != search_head
                or landscape.get("source_batch_fingerprint") != manifest.get("config_fingerprint")
            ):
                raise ValidationConsistencyError("已有 validation scenario 的来源身份不一致")
        else:
            scenario = scenario_builder(group[0]["run_spec"])
            if getattr(scenario, "scenario_id", None) != scenario_id: raise ValidationConsistencyError("重建场景 scenario_id 不匹配")
            landscape = validate_scenario_landscape(scenario, group, abs_tol=abs_tol, rel_tol=rel_tol)
            landscape.update({"search_code_sha": search_head, "validation_code_sha": validation_code_sha, "source_batch_fingerprint": manifest.get("config_fingerprint"), "source_manifest_path": str(source_output_dir / "batch_manifest.json"), "validation_time": _utc_now()})
            atomic_write_json(scenario_path, landscape)
        scenarios_done += 1
        for run in group:
            path = validation_root / "runs" / f"{run['run_id']}.json"
            if resume and path.exists():
                old = _read_json(path)
                if (
                    old.get("schema_version") != VALIDATION_SCHEMA_VERSION
                    or old.get("fingerprint") != run.get("fingerprint")
                    or old.get("scenario_id") != scenario_id
                    or old.get("search_code_sha") != search_head
                    or old.get("source_batch_fingerprint") != manifest.get("config_fingerprint")
                ):
                    raise ValidationConsistencyError("已有 validation run 的来源身份冲突")
            else:
                derived = validate_run_against_landscape(run, landscape, abs_tol=abs_tol, rel_tol=rel_tol)
                derived.update({"search_code_sha": search_head, "validation_code_sha": validation_code_sha, "source_batch_fingerprint": manifest.get("config_fingerprint"), "source_manifest_path": str(source_output_dir / "batch_manifest.json"), "validation_time": _utc_now()})
                atomic_write_json(path, derived)
            runs_done += 1
    return {"scenarios": scenarios_done, "runs": runs_done, "search_code_sha": search_head, "validation_code_sha": validation_code_sha}


def summarize_validated(source_output_dir: Path) -> dict[str, object]:
    root = Path(source_output_dir); rows = [_read_json(path) for path in sorted((root / "validation" / "runs").glob("*.json"))]
    output = root / "summaries_validated"; output.mkdir(parents=True, exist_ok=True)
    flat = []
    for row in rows:
        metrics = row.get("trace_metrics", {})
        flat.append({"run_id": row["run_id"], "scenario_id": row["scenario_id"], "method": row.get("method"), "method_role": row.get("method_role"), "diagnostic_only": row.get("diagnostic_only"), "reached_global_optimum": row.get("reached_global_optimum"), "final_optimality_gap": row.get("final_optimality_gap"), "true_cost_decrease": row.get("true_cost_decrease"), "actual_ed_lp_solves": metrics.get("actual_ed_lp_solves", 0), "proposals": metrics.get("proposal_events", 0)})
    def write_csv(path: Path, data: list[dict[str, object]]) -> None:
        if not data: path.write_text("\n", encoding="utf-8"); return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    write_csv(output / "validated_run_index.csv", flat)
    by_method: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in flat: by_method.setdefault((row["method"], row["method_role"], row["diagnostic_only"]), []).append(row)
    method_rows = [{"method": key[0], "method_role": key[1], "diagnostic_only": key[2], "runs": len(group), "global_optimum_hits": sum(bool(r["reached_global_optimum"]) for r in group), "true_cost_decreases": sum(bool(r["true_cost_decrease"]) for r in group)} for key, group in sorted(by_method.items(), key=str)]
    write_csv(output / "validated_method_summary.csv", method_rows); write_csv(output / "validated_scenario_summary.csv", flat)
    curves = [{"run_id": row["run_id"], "actual_edlp_solves": key, "best_true_cost": value} for row in rows for key, value in row.get("edlp_budget_curve", {}).items()]
    write_csv(output / "edlp_budget_curves.csv", curves)
    summary = {"schema_version": VALIDATION_SCHEMA_VERSION, "validated_runs": len(rows), "formal_runs": sum(not bool(r.get("diagnostic_only")) for r in rows), "diagnostic_only_runs": sum(bool(r.get("diagnostic_only")) for r in rows), "method_summary": method_rows}
    atomic_write_json(output / "validated_summary.json", summary)
    return summary
