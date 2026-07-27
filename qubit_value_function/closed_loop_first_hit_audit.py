"""只读提取 formal joint-BBHT trace 的首次真实全局最优命中代价。

本模块绝不导入场景构建器、VQC、Aer 或精确求解器；它只读取已落盘的
completed JSON 与 validation landscape，并把新审计文件写入独立目录。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


AUDIT_SCHEMA_VERSION = "stage-a-closed-loop-first-hit-audit-v1"


class FirstHitAuditError(RuntimeError):
    """输入结果不完整，或输出目录不能安全创建时抛出。"""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FirstHitAuditError(f"无法读取 JSON：{path}: {error}") from error
    if not isinstance(value, dict):
        raise FirstHitAuditError(f"JSON 根对象必须为 object：{path}")
    return value


def _trace(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = result.get("trial_trace", [])
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise FirstHitAuditError("completed run 的 trial_trace 缺失或无效")
    return sorted(value, key=lambda row: int(row.get("trial_number", 0)))


def _number(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _grover(row: Mapping[str, Any]) -> int:
    # 新字段优先；老结果仅在新字段不存在时才回退。
    return _number(row, "sampled_grover_iterations") if "sampled_grover_iterations" in row else _number(row, "grover_iterations")


def _close(left: object, right: object, *, abs_tol: float, rel_tol: float) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(float(left), float(right), abs_tol=abs_tol, rel_tol=rel_tol)


def _event_is_confirmed_optimum(row: Mapping[str, Any], optimum: float, *, abs_tol: float, rel_tol: float) -> bool:
    exact = row.get("exact_evaluation")
    return isinstance(exact, Mapping) and bool(exact.get("success")) and _close(exact.get("total_cost"), optimum, abs_tol=abs_tol, rel_tol=rel_tol)


def _source_for_accepted(row: Mapping[str, Any], index: int, training: set[int]) -> str:
    if index in training:
        return "training_exact_cache"
    exact = row.get("exact_evaluation")
    source = exact.get("source") if isinstance(exact, Mapping) else None
    if bool(row.get("new_edlp_solve_performed")) or source == "new_ed_lp_call":
        return "new_ed_lp_call"
    if bool(row.get("candidate_cache_hit")) or source in {"training_exact_cache", "initial_training_cache", "cached"}:
        return "cached_nontraining_exact"
    return "unavailable"


def _sum(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(_number(row, key) for row in rows)


def _diffuser(rows: Sequence[Mapping[str, Any]]) -> int | None:
    # 现有 trace 未保存这个原生计数；不能以 Grover 迭代数替代。
    if not rows or not all("diffuser_calls_added" in row for row in rows):
        return None
    return _sum(rows, "diffuser_calls_added")


def _trial_or_none(row: Mapping[str, Any] | None) -> int | None:
    return int(row["trial_number"]) if row is not None and isinstance(row.get("trial_number"), (int, float)) else None


def _index_or_none(row: Mapping[str, Any] | None) -> int | None:
    return int(row["measured_index"]) if row is not None and isinstance(row.get("measured_index"), (int, float)) else None


def derive_first_hit_run(run: Mapping[str, Any], landscape: Mapping[str, Any]) -> dict[str, Any]:
    """从一个 joint_bbht run 与其已验证 landscape 无求解地恢复首次命中。"""
    required = ("true_global_optimum_indices", "true_global_optimum_cost", "initial_true_improvement_exists")
    if any(key not in landscape for key in required):
        raise FirstHitAuditError("validation landscape 缺少全局最优或初始改善字段")
    result = run.get("result")
    scenario = run.get("scenario")
    spec = run.get("run_spec")
    if not isinstance(result, Mapping) or not isinstance(scenario, Mapping) or not isinstance(spec, Mapping):
        raise FirstHitAuditError("completed run 缺少 result/scenario/run_spec")
    optima_raw = landscape["true_global_optimum_indices"]
    if not isinstance(optima_raw, list) or not optima_raw:
        raise FirstHitAuditError("validation landscape 的 true_global_optimum_indices 无效")
    optima = {int(value) for value in optima_raw}
    optimum = float(landscape["true_global_optimum_cost"])
    tolerance = landscape.get("comparison_tolerance", {})
    abs_tol = float(tolerance.get("abs_tol", 1e-6)) if isinstance(tolerance, Mapping) else 1e-6
    rel_tol = float(tolerance.get("rel_tol", 1e-9)) if isinstance(tolerance, Mapping) else 1e-9
    training = {int(value) for value in scenario.get("training_indices", [])}
    trace = _trace(result)
    measured = next((row for row in trace if _index_or_none(row) in optima), None)
    admitted = next((row for row in trace if _index_or_none(row) in optima and bool(row.get("admission_passed"))), None)
    confirmed = next((row for row in trace if _index_or_none(row) in optima and _event_is_confirmed_optimum(row, optimum, abs_tol=abs_tol, rel_tol=rel_tol)), None)
    accepted = next((
        row for row in trace
        if _index_or_none(row) in optima
        and bool(row.get("accepted_update"))
        and row.get("incumbent_index_after") == _index_or_none(row)
        and _event_is_confirmed_optimum(row, optimum, abs_tol=abs_tol, rel_tol=rel_tol)
    ), None)
    through = trace[: trace.index(accepted) + 1] if accepted is not None else []
    after = trace[trace.index(accepted) + 1 :] if accepted is not None else []
    accepted_index = _index_or_none(accepted)
    final_index = result.get("final_incumbent_index")
    reached = isinstance(final_index, (int, float)) and int(final_index) in optima
    stop_reason = result.get("stop_reason")
    marked_empty = stop_reason == "no_surrogate_marked_state_by_conservative_lower_bound"
    row: dict[str, Any] = {
        "run_id": run.get("run_id"), "scenario_id": scenario.get("scenario_id"),
        "generator_pair": "-".join(str(x) for x in spec.get("generator_pair", [])),
        "window_start": spec.get("window_start"), "training_seed": spec.get("training_seed"), "run_seed": spec.get("run_seed"), "method": run.get("method"),
        "training_indices": sorted(training), "initial_incumbent_index": result.get("initial_incumbent_index", landscape.get("initial_incumbent_index")),
        "initial_incumbent_true_cost": result.get("initial_incumbent_true_cost", landscape.get("initial_incumbent_true_cost")),
        "global_optimum_indices": sorted(optima), "global_optimum_true_cost": optimum,
        "initial_is_global_optimum": bool(landscape.get("initial_incumbent_is_global_optimum")),
        "initial_has_true_improvement": bool(landscape["initial_true_improvement_exists"]),
        "all_global_optima_outside_training": optima.isdisjoint(training), "any_global_optimum_outside_training": bool(optima - training),
        "reached_global_optimum": reached, "first_accepted_optimum_index": accepted_index,
        "first_accepted_optimum_is_nontraining": (accepted_index not in training) if accepted_index is not None else None,
        "first_accepted_optimum_source": _source_for_accepted(accepted, accepted_index, training) if accepted is not None and accepted_index is not None else "unavailable",
        "first_measured_optimum_trial": _trial_or_none(measured), "first_measured_optimum_index": _index_or_none(measured),
        "first_admitted_optimum_trial": _trial_or_none(admitted), "first_confirmed_optimum_trial": _trial_or_none(confirmed),
        "first_accepted_optimum_trial": _trial_or_none(accepted),
        "trials_to_first_accepted_optimum": len(through) if accepted is not None else None,
        "grover_iterations_to_first_accepted_optimum": sum(_grover(item) for item in through) if accepted is not None else None,
        "oracle_calls_to_first_accepted_optimum": _sum(through, "oracle_calls_added") if accepted is not None else None,
        "diffuser_calls_to_first_accepted_optimum": _diffuser(through) if accepted is not None else None,
        "mps_executions_to_first_accepted_optimum": len(through) if accepted is not None else None,
        "post_measurement_predicate_checks_to_first_accepted_optimum": len(through) if accepted is not None else None,
        "combined_predicate_query_reference_to_first_accepted_optimum": (
            _sum(through, "oracle_calls_added") + len(through) if accepted is not None else None
        ),
        "actual_ed_lp_solves_to_first_accepted_optimum": _sum(through, "actual_ed_lp_solves_added") if accepted is not None else None,
        "new_exact_evaluation_attempts_to_first_accepted_optimum": _sum(through, "new_exact_evaluation_attempts_added") if accepted is not None else None,
        "cached_exact_lookups_to_first_accepted_optimum": sum(bool(item.get("candidate_cache_hit")) for item in through) if accepted is not None else None,
        "threshold_updates_to_first_accepted_optimum": sum(bool(item.get("threshold_updated")) for item in through) if accepted is not None else None,
        "total_trials": len(trace), "total_grover_iterations": sum(_grover(item) for item in trace),
        "total_oracle_calls": _sum(trace, "oracle_calls_added"), "total_actual_ed_lp_solves": _sum(trace, "actual_ed_lp_solves_added"),
        "tail_trials_after_first_optimum": len(after) if accepted is not None else None,
        "tail_grover_iterations_after_first_optimum": sum(_grover(item) for item in after) if accepted is not None else None,
        "tail_oracle_calls_after_first_optimum": _sum(after, "oracle_calls_added") if accepted is not None else None,
        "tail_actual_ed_lp_solves_after_first_optimum": _sum(after, "actual_ed_lp_solves_added") if accepted is not None else None,
        "stop_reason": stop_reason, "optimum_found_before_stop": accepted is not None,
        "stopped_without_optimum": accepted is None, "marked_empty_or_lower_bound_stop": marked_empty,
    }
    return row


def _stats(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return {"mean": statistics.mean(values) if values else None, "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None, "denominator": len(values)}


def _scenario_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows: groups.setdefault(str(row["scenario_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for scenario_id, group in sorted(groups.items()):
        successful = [row for row in group if row["first_accepted_optimum_trial"] is not None]
        stop = {reason: sum(row["stop_reason"] == reason for row in group) for reason in sorted({str(row["stop_reason"]) for row in group})}
        output.append({"scenario_id": scenario_id, "runs": len(group), "reached_global_optimum_runs": sum(bool(row["reached_global_optimum"]) for row in group),
                       "nontraining_global_optimum_runs": sum(row["first_accepted_optimum_is_nontraining"] is True for row in group),
                       "first_hit_successful_runs": len(successful), "first_hit_trial": _stats(successful, "trials_to_first_accepted_optimum"),
                       "first_hit_grover_iterations": _stats(successful, "grover_iterations_to_first_accepted_optimum"),
                       "first_hit_oracle_calls": _stats(successful, "oracle_calls_to_first_accepted_optimum"),
                       "first_hit_new_ed_lp": _stats(successful, "actual_ed_lp_solves_to_first_accepted_optimum"),
                       "tail_trials": _stats(successful, "tail_trials_after_first_optimum"), "tail_grover_iterations": _stats(successful, "tail_grover_iterations_after_first_optimum"),
                       "stop_reason_distribution": stop})
    return output


def _validation_checks(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    initial = [row for row in rows if row["initial_has_true_improvement"]]
    reached = [row for row in rows if row["reached_global_optimum"]]
    successful_initial = [row for row in initial if row["reached_global_optimum"]]
    actual = {"joint_bbht_run_count": len(rows), "initial_true_improvement_runs": len(initial), "final_global_optimum_runs": len(reached),
              "initial_improvement_and_final_global_optimum_runs": len(successful_initial), "total_mps_circuit_executions": sum(int(row["total_trials"]) for row in rows),
              "total_grover_oracle_calls": sum(int(row["total_oracle_calls"]) for row in rows), "total_actual_ed_lp_solves": sum(int(row["total_actual_ed_lp_solves"]) for row in rows),
              "g1g3_w0_runs": sum(row["generator_pair"] == "1-3" and row["window_start"] == 0 for row in rows),
              "g1g3_w0_marked_empty_stops": sum(row["generator_pair"] == "1-3" and row["window_start"] == 0 and row["marked_empty_or_lower_bound_stop"] for row in rows),
              "g1g3_w0_first_optimum_hits": sum(row["generator_pair"] == "1-3" and row["window_start"] == 0 and row["first_accepted_optimum_trial"] is not None for row in rows)}
    expected = {"joint_bbht_run_count": 180, "initial_true_improvement_runs": 105, "final_global_optimum_runs": 162,
                "initial_improvement_and_final_global_optimum_runs": 87, "total_mps_circuit_executions": 2532, "total_grover_oracle_calls": 3161,
                "total_actual_ed_lp_solves": 38, "g1g3_w0_runs": 15, "g1g3_w0_marked_empty_stops": 15, "g1g3_w0_first_optimum_hits": 0}
    return {key: {"expected": expected[key], "actual": actual[key], "matches": expected[key] == actual[key]} for key in expected}


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["first_accepted_optimum_trial"] is not None]
    nontraining = [row for row in successful if row["first_accepted_optimum_is_nontraining"] is True]
    nohit = [row for row in rows if row["first_accepted_optimum_trial"] is None]
    return {"schema_version": AUDIT_SCHEMA_VERSION, "scope": "existing formal completed JSON + existing validation only; no search, VQC, Aer/MPS, ED/LP, or enumeration was executed",
            "query_counting_note": "total_oracle_calls is the recorded amplitude-amplification phase-oracle count. post_measurement_predicate_checks is reported separately; their sum is only a comparison reference, not native total_oracle_calls semantics.",
            "diffuser_note": "unavailable unless every trace row records diffuser_calls_added; it is not inferred from Grover iterations.",
            "all_joint_runs": len(rows), "initial_true_improvement_runs": sum(bool(row["initial_has_true_improvement"]) for row in rows),
            "final_global_optimum_runs": sum(bool(row["reached_global_optimum"]) for row in rows), "first_accepted_global_optimum_runs": len(successful),
            "first_accepted_nontraining_global_optimum_runs": len(nontraining), "first_accepted_optimum_from_new_ed_lp_call_runs": sum(row["first_accepted_optimum_source"] == "new_ed_lp_call" for row in successful),
            "first_hit_success_metrics": {"trials": _stats(successful, "trials_to_first_accepted_optimum"), "grover_iterations": _stats(successful, "grover_iterations_to_first_accepted_optimum"), "amplitude_amplification_oracle_calls": _stats(successful, "oracle_calls_to_first_accepted_optimum"), "actual_ed_lp_solves": _stats(successful, "actual_ed_lp_solves_to_first_accepted_optimum"), "tail_trials": _stats(successful, "tail_trials_after_first_optimum"), "tail_grover_iterations": _stats(successful, "tail_grover_iterations_after_first_optimum")},
            "first_hit_nontraining_metrics": {"runs": len(nontraining), "trials": _stats(nontraining, "trials_to_first_accepted_optimum"), "grover_iterations": _stats(nontraining, "grover_iterations_to_first_accepted_optimum"), "amplitude_amplification_oracle_calls": _stats(nontraining, "oracle_calls_to_first_accepted_optimum"), "actual_ed_lp_solves": _stats(nontraining, "actual_ed_lp_solves_to_first_accepted_optimum")},
            "stopped_without_first_optimum": {reason: sum(row["stop_reason"] == reason for row in nohit) for reason in sorted({str(row["stop_reason"]) for row in nohit})},
            "validation_checks": _validation_checks(rows)}


def _csv_value(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) if isinstance(value, (list, dict)) else value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows: writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _representatives(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    wanted = {"case14-g0g5-w2-s0", "case14-g0g5-w1-s0", "case14-g0g5-w1-s1", "case14-g0g5-w1-s2", "case14-g0g1-w0-s0", "case14-g1g3-w0-s0", "case14-g1g3-w0-s1", "case14-g1g3-w0-s2"}
    fields = ("scenario_id", "run_seed", "first_accepted_optimum_trial", "grover_iterations_to_first_accepted_optimum", "actual_ed_lp_solves_to_first_accepted_optimum", "first_accepted_optimum_is_nontraining", "tail_trials_after_first_optimum", "stop_reason")
    return [{field: row[field] for field in fields} for row in rows if row["scenario_id"] in wanted]


def _report(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    checks = summary["validation_checks"]
    nontraining = summary["first_hit_nontraining_metrics"]
    missing = summary["stopped_without_first_optimum"]
    return "\n".join([
        "# Formal joint-BBHT first-hit audit", "", "## 范围", "", str(summary["scope"]), "",
        "主指标是首次 `accepted_update` 且经已保存 exact record 确认为真实全局最优的 trial；measured、admitted、confirmed 字段单列，未把它们混为一谈。", "",
        "## 结果", "", f"- joint runs：{summary['all_joint_runs']}；首次 accepted 全局最优：{summary['first_accepted_global_optimum_runs']}。",
        f"- 首次 accepted 的训练集外真实全局最优：{nontraining['runs']}。", f"- 训练集外首次命中的 trial 统计：{json.dumps(nontraining['trials'], ensure_ascii=False)}。",
        f"- 训练集外首次命中的 Grover 迭代统计：{json.dumps(nontraining['grover_iterations'], ensure_ascii=False)}。",
        f"- 未首次命中的停止原因：{json.dumps(missing, ensure_ascii=False, sort_keys=True)}。", "",
        "## 计数口径", "", str(summary["query_counting_note"]), "", str(summary["diffuser_note"]), "",
        "## 一致性校验", "", *[f"- {key}: expected={value['expected']}, actual={value['actual']}, matches={value['matches']}" for key, value in checks.items()], "",
        "## 解释边界", "", "这些是已保存 formal trace 的描述性 first-hit 统计。它们可用于评估候选准入与真实 ED/LP 筛选的记录表现，但不构成量子优势、平方级加速或端到端加速的证明。",
    ]) + "\n"


def audit_formal_first_hits(formal_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    """只读审计 formal 结果，并在一个此前不存在的目录写入六个派生文件。"""
    formal_dir = Path(formal_dir)
    completed_dir = formal_dir / "runs" / "completed"
    validation_dir = formal_dir / "validation" / "scenarios"
    paths = sorted(completed_dir.glob("*.json"))
    if not paths:
        raise FirstHitAuditError("未找到 formal completed JSON")
    rows: list[dict[str, Any]] = []
    for path in paths:
        run = _load(path)
        if run.get("method") != "joint_bbht":
            continue
        scenario = run.get("scenario")
        if not isinstance(scenario, Mapping) or not isinstance(scenario.get("scenario_id"), str):
            raise FirstHitAuditError(f"缺少 scenario_id：{path}")
        landscape = _load(validation_dir / f"{scenario['scenario_id']}.json")
        rows.append(derive_first_hit_run(run, landscape))
    rows.sort(key=lambda row: str(row["run_id"]))
    scenarios = _scenario_rows(rows)
    summary = _summary(rows)
    destination = Path(output_dir) if output_dir is not None else formal_dir / "first_hit_audit"
    if destination.exists():
        raise FirstHitAuditError(f"输出目录已经存在，拒绝覆盖：{destination}")
    destination.mkdir(parents=True)
    _write_csv(destination / "joint_bbht_first_hit_runs.csv", rows)
    _write_json(destination / "joint_bbht_first_hit_runs.json", rows)
    _write_csv(destination / "joint_bbht_first_hit_scenarios.csv", scenarios)
    _write_json(destination / "joint_bbht_first_hit_summary.json", summary)
    _write_csv(destination / "representative_first_hit_table.csv", _representatives(rows))
    (destination / "first_hit_audit_report.md").write_text(_report(summary, rows), encoding="utf-8")
    return summary
