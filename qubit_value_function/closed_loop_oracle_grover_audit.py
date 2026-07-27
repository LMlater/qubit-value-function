"""从已保存 formal trace 做只读的 joint-oracle/Grover 审计。

不构造场景、VQC、量子电路或精确求解器。若落盘数据没有保存量化 VQC 的
16 状态值表，动态阈值阶段的 marked set/count 一律标为 unavailable，不猜测。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


class OracleGroverAuditError(RuntimeError):
    pass


DIMENSION = 16
UNAVAILABLE = "unavailable_quantized_vqc_table_not_persisted"


def _load(path: Path) -> dict[str, Any]:
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error: raise OracleGroverAuditError(f"无法读取 {path}: {error}") from error
    if not isinstance(value, dict): raise OracleGroverAuditError(f"JSON 根对象必须为 object: {path}")
    return value


def _int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _iterations(row: Mapping[str, Any]) -> int:
    return _int(row, "sampled_grover_iterations") if "sampled_grover_iterations" in row else _int(row, "grover_iterations")


def _ideal(marked: int | None, iterations: int) -> float | None:
    if marked is None: return None
    if marked == 0: return 0.0
    return math.sin((2 * iterations + 1) * math.asin(math.sqrt(marked / DIMENSION))) ** 2


def derive_trial_rows(run: Mapping[str, Any], landscape: Mapping[str, Any]) -> list[dict[str, Any]]:
    """严格核对 trace oracle 谓词，并恢复仅初始阈值阶段可证的 marked count。"""
    result, scenario, spec = run.get("result"), run.get("scenario"), run.get("run_spec")
    if not isinstance(result, Mapping) or not isinstance(scenario, Mapping) or not isinstance(spec, Mapping):
        raise OracleGroverAuditError("run 缺少 result/scenario/run_spec")
    trace = result.get("trial_trace")
    if not isinstance(trace, list) or not all(isinstance(item, Mapping) for item in trace): raise OracleGroverAuditError("trial_trace 无效")
    history = result.get("threshold_history")
    if not isinstance(history, list) or not history or not isinstance(history[0], Mapping) or "encoded_threshold" not in history[0]:
        raise OracleGroverAuditError("threshold_history 缺少初始 encoded threshold")
    initial_threshold = int(history[0]["encoded_threshold"])
    required = ("initial_joint_marked_count", "initial_surrogate_cost_marked_count")
    if any(key not in landscape for key in required): raise OracleGroverAuditError("validation 缺少初始 marked count")
    rows: list[dict[str, Any]] = []
    for item in sorted(trace, key=lambda row: _int(row, "trial_number")):
        threshold = int(item["encoded_threshold_before"])
        cost, feasible = int(item["surrogate_integer_cost"]), bool(item["hard_logic_feasible"])
        expected_cost = cost < threshold
        expected_joint = expected_cost and feasible
        if bool(item.get("joint_marked")) != expected_joint:
            raise OracleGroverAuditError(
                f"joint_marked 与 surrogate_integer_cost < encoded_threshold_before AND hard_logic_feasible 不一致: {run.get('run_id')} trial {_int(item, 'trial_number')}"
            )
        if bool(item.get("cost_marked")) != expected_cost:
            raise OracleGroverAuditError(f"cost_marked 与严格 comparator 语义不一致: {run.get('run_id')} trial {_int(item, 'trial_number')}")
        initial_phase = threshold == initial_threshold
        marked_count = int(landscape["initial_joint_marked_count"]) if initial_phase else None
        cost_count = int(landscape["initial_surrogate_cost_marked_count"]) if initial_phase else None
        k = _iterations(item)
        hit = bool(item.get("joint_marked"))
        rows.append({
            "run_id": run.get("run_id"), "scenario_id": scenario.get("scenario_id"), "run_seed": spec.get("run_seed"),
            "trial_number": _int(item, "trial_number"), "encoded_threshold_before_trial": threshold,
            "true_threshold_before_trial": item.get("true_threshold_before"), "sampled_grover_iterations": k,
            "oracle_calls_added": _int(item, "oracle_calls_added"), "measured_index": item.get("measured_index"),
            "measured_bitstring": item.get("measured_bitstring"), "surrogate_integer_cost": cost,
            "hard_logic_feasible": feasible, "auxiliary_accepted": bool(item.get("auxiliary_accepted")),
            "cost_marked_count": cost_count, "joint_marked_count": marked_count,
            "marked_set_recovery_status": "available_from_validation_initial_threshold" if initial_phase else UNAVAILABLE,
            "measured_cost_marked_hit": expected_cost, "measured_joint_marked_hit": hit,
            "valid_joint_marked_hit": hit and bool(item.get("auxiliary_accepted")),
            "random_joint_hit_probability": (marked_count / DIMENSION) if marked_count is not None else None,
            "ideal_joint_grover_hit_probability": _ideal(marked_count, k),
            "hit_probability_gain_over_uniform_reference": (None if marked_count is None else (1.0 if hit else 0.0) - marked_count / DIMENSION),
            "candidate_admitted": bool(item.get("admission_passed")), "cache_hit": bool(item.get("candidate_cache_hit")),
            "exact_evaluated": isinstance(item.get("exact_evaluation"), Mapping),
            "exact_evaluation_success": bool(item.get("exact_evaluation", {}).get("success")) if isinstance(item.get("exact_evaluation"), Mapping) else None,
            "actual_ed_lp_solve": _int(item, "actual_ed_lp_solves_added"), "true_improvement": bool(item.get("true_improvement")),
            "incumbent_updated": bool(item.get("accepted_update")), "stop_reason_after_trial": item.get("stop_reason_after_trial"),
        })
    return rows


def _wilson(hits: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0: return None
    p = hits / total; denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return [centre - radius, centre + radius]


def _group(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows: buckets.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items(), key=lambda pair: tuple(str(x) for x in pair[0])):
        count = len(group); hits = sum(bool(row["measured_joint_marked_hit"]) for row in group)
        uniform = statistics.mean(float(row["random_joint_hit_probability"]) for row in group)
        ideal = statistics.mean(float(row["ideal_joint_grover_hit_probability"]) for row in group)
        output.append({**dict(zip(keys, key)), "trials": count, "marked_hits": hits, "empirical_hit_rate": hits / count,
                       "uniform_baseline": uniform, "ideal_grover_hit_probability": ideal,
                       "empirical_minus_uniform": hits / count - uniform,
                       "empirical_minus_ideal": hits / count - ideal,
                       "wilson_95_percent_ci": _wilson(hits, count), "mean_grover_iterations": statistics.mean(int(row["sampled_grover_iterations"]) for row in group),
                       "admitted_candidate_rate": sum(bool(row["candidate_admitted"]) for row in group) / count,
                       "true_improvement_rate": sum(bool(row["true_improvement"]) for row in group) / count})
    return output


def _first_hit_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows: groups.setdefault((str(row["run_id"]), int(row["encoded_threshold_before_trial"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (run_id, threshold), group in sorted(groups.items()):
        group = sorted(group, key=lambda row: int(row["trial_number"]))
        hit = next((row for row in group if row["measured_joint_marked_hit"]), None)
        admitted = next((row for row in group if row["candidate_admitted"]), None)
        improvement = next((row for row in group if row["true_improvement"]), None)
        result.append({"run_id": run_id, "scenario_id": group[0]["scenario_id"], "run_seed": group[0]["run_seed"],
                       "encoded_threshold_before_trial": threshold, "joint_marked_count": group[0]["joint_marked_count"],
                       "first_joint_marked_hit_trial": hit["trial_number"] if hit else None,
                       "oracle_calls_to_first_joint_marked_hit": sum(int(row["oracle_calls_added"]) for row in group[:group.index(hit)+1]) if hit else None,
                       "circuit_executions_to_first_joint_marked_hit": group.index(hit)+1 if hit else None,
                       "predicate_checks_to_first_joint_marked_hit": group.index(hit)+1 if hit else None,
                       "first_admitted_trial": admitted["trial_number"] if admitted else None, "first_true_improvement_trial": improvement["trial_number"] if improvement else None})
    return result


def _summary(rows: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if row["joint_marked_count"] is not None]
    nonempty = [row for row in known if int(row["joint_marked_count"]) > 0]
    empty = [row for row in known if int(row["joint_marked_count"]) == 0]
    # 精度只能以 trace 中确实执行 exact evaluation 的 marked trial 为条件；未保存则不扩张分母。
    marked = [row for row in rows if row["measured_joint_marked_hit"]]
    exact_marked = [row for row in marked if row["exact_evaluated"] and row["exact_evaluation_success"]]
    initial_recall = []
    for land in validation:
        improving = set(land.get("real_improvement_indices", [])); joint = set(land.get("initial_joint_marked_indices", []))
        if improving: initial_recall.append(len(improving & joint) / len(improving))
    pooled_recall_numerator = sum(len(set(land.get("real_improvement_indices", [])) & set(land.get("initial_joint_marked_indices", []))) for land in validation)
    pooled_recall_denominator = sum(len(set(land.get("real_improvement_indices", []))) for land in validation)
    observed_nonempty = sum(bool(row["measured_joint_marked_hit"]) for row in nonempty)
    expected_uniform = sum(float(row["random_joint_hit_probability"]) for row in nonempty)
    return {"scope": "read-only existing formal completed JSON plus validation; no VQC/Aer/MPS/ED-LP/enumeration executed",
            "data_limit": "Dynamic threshold marked counts are unavailable because formal artifacts do not persist the 16-state quantized VQC value table per scenario. Only the initial-threshold phase has a validation-persisted marked set.",
            "trace_semantics": {"trials": len(rows), "joint_predicate_consistent_trials": len(rows), "observed_joint_marked_hits_all_trials": len(marked), "observed_joint_marked_hit_rate_all_trials_not_conditioned_on_M": len(marked) / len(rows) if rows else None,
                                "valid_auxiliary_trials": sum(bool(row["auxiliary_accepted"]) for row in rows), "valid_joint_marked_hits": sum(bool(row["valid_joint_marked_hit"]) for row in rows)},
            "recoverable_initial_threshold_phase": {"trials_with_known_M": len(known), "marked_nonempty_trials": len(nonempty), "marked_empty_trials": len(empty),
                "observed_joint_marked_hits": observed_nonempty,
                "observed_joint_marked_hit_rate_on_nonempty_trials": (observed_nonempty / len(nonempty)) if nonempty else None,
                "expected_uniform_hits": expected_uniform,
                "expected_uniform_hit_rate": (expected_uniform / len(nonempty)) if nonempty else None,
                "observed_to_uniform_hit_ratio": (observed_nonempty / expected_uniform) if expected_uniform else None},
            "unavailable_dynamic_threshold_trials": len(rows) - len(known),
            "vqc_marked_precision_trace_observable": {"definition": "P(true ED/LP improvement | measured joint-marked and exactly evaluated)", "exactly_evaluated_marked_trials": len(exact_marked), "true_improving_marked_trials": sum(bool(row["true_improvement"]) for row in exact_marked), "precision": sum(bool(row["true_improvement"]) for row in exact_marked) / len(exact_marked) if exact_marked else None},
            "vqc_marked_recall_initial_threshold_scenarios": {"definition": "initial threshold only; dynamic threshold recall unavailable", "scenario_denominator": len(initial_recall), "mean_recall": statistics.mean(initial_recall) if initial_recall else None, "pooled_recall": pooled_recall_numerator / pooled_recall_denominator if pooled_recall_denominator else None, "true_improving_states": pooled_recall_denominator, "joint_marked_true_improving_states": pooled_recall_numerator},
            "groups_by_M": _group(nonempty, ("joint_marked_count",)), "groups_by_M_k": _group(nonempty, ("joint_marked_count", "sampled_grover_iterations")),
            "groups_by_k": _group(nonempty, ("sampled_grover_iterations",)), "first_joint_marked_hit_by_threshold_phase": _first_hit_rows(rows)}


def _write_json(path: Path, data: Any) -> None: path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def audit_oracle_grover(formal_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    formal_dir = Path(formal_dir); completed = formal_dir / "runs" / "completed"; landscapes = formal_dir / "validation" / "scenarios"
    output = Path(output_dir) if output_dir else formal_dir / "oracle_grover_audit"
    if output.exists(): raise OracleGroverAuditError(f"输出目录已存在，拒绝覆盖：{output}")
    rows: list[dict[str, Any]] = []; used: dict[str, dict[str, Any]] = {}
    for path in sorted(completed.glob("*.json")):
        run = _load(path)
        if run.get("method") != "joint_bbht": continue
        scenario = run.get("scenario")
        if not isinstance(scenario, Mapping) or not isinstance(scenario.get("scenario_id"), str): raise OracleGroverAuditError(f"缺 scenario_id: {path}")
        sid = str(scenario["scenario_id"]); land = _load(landscapes / f"{sid}.json"); used[sid] = land
        rows.extend(derive_trial_rows(run, land))
    rows.sort(key=lambda row: (str(row["run_id"]), int(row["trial_number"])))
    summary = _summary(rows, list(used.values()))
    output.mkdir(parents=True); _write_csv(output / "joint_bbht_trial_oracle_grover_audit.csv", rows); _write_json(output / "joint_bbht_trial_oracle_grover_audit.json", rows)
    _write_json(output / "oracle_grover_summary.json", summary); _write_csv(output / "oracle_grover_groups_by_M_k.csv", summary["groups_by_M_k"]); _write_csv(output / "first_joint_marked_hit.csv", summary["first_joint_marked_hit_by_threshold_phase"])
    return summary
