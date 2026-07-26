"""只读汇总阶段 A 闭环批量 JSON 结果。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .closed_loop_batch import BATCH_SCHEMA_VERSION, atomic_write_json


class SummaryValidationError(RuntimeError):
    """已有批量结果不完整或相互矛盾。"""


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryValidationError(f"无法读取 {path}: {error}") from error
    if not isinstance(value, dict):
        raise SummaryValidationError(f"{path} 不是 JSON object")
    return value


def _completed_rows(output_dir: Path, expected_fingerprints: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in sorted((output_dir / "runs" / "completed").glob("*.json")):
        payload = _load(path)
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or run_id in seen:
            raise SummaryValidationError(f"completed 结果存在重复或无效 run_id: {path}")
        if payload.get("schema_version") != BATCH_SCHEMA_VERSION or payload.get("status") != "completed":
            raise SummaryValidationError(f"completed 结果 schema/status 无效: {path}")
        if expected_fingerprints.get(run_id) != payload.get("fingerprint"):
            raise SummaryValidationError(f"completed 结果 fingerprint 与 manifest 不一致: {path}")
        if not isinstance(payload.get("result"), Mapping):
            raise SummaryValidationError(f"completed 结果缺少 JSON result: {path}")
        seen.add(run_id)
        rows.append(payload)
    return rows


def _failed_ids(
    output_dir: Path,
    expected_fingerprints: Mapping[str, object],
    completed_ids: set[str],
) -> tuple[set[str], set[str]]:
    failed: set[str] = set()
    recovered: set[str] = set()
    for path in sorted((output_dir / "runs" / "failed").glob("*.json")):
        payload = _load(path)
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or run_id in failed:
            raise SummaryValidationError(f"failed 结果存在重复或无效 run_id: {path}")
        if payload.get("schema_version") != BATCH_SCHEMA_VERSION or payload.get("status") != "failed":
            raise SummaryValidationError(f"failed 结果 schema/status 无效: {path}")
        if expected_fingerprints.get(run_id) != payload.get("fingerprint"):
            raise SummaryValidationError(f"failed 结果 fingerprint 与 manifest 不一致: {path}")
        if run_id in completed_ids:
            recovered.add(run_id)
        else:
            failed.add(run_id)
    return failed, recovered


def _counter(result: Mapping[str, object], name: str) -> int:
    counters = result.get("counters", {})
    if not isinstance(counters, Mapping):
        return 0
    aliases = {
        "proposals": ("proposals_used", "circuit_executions"),
        "threshold_updates": ("threshold_updates",),
        "new_exact_evaluations": ("new_exact_evaluation_attempts",),
        "actual_ed_lp_solves": ("actual_ed_lp_solves",),
        "cache_hits": ("cached_exact_lookups",),
        "logic_rejections": ("logic_precheck_rejections",),
        "oracle_calls": ("total_oracle_calls",),
    }
    for key in aliases[name]:
        value = counters.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _run_row(payload: Mapping[str, object]) -> dict[str, object]:
    result = payload["result"]
    assert isinstance(result, Mapping)
    spec = payload.get("run_spec", {})
    assert isinstance(spec, Mapping)
    scenario = payload.get("scenario", {})
    assert isinstance(scenario, Mapping)
    initial = result.get("initial_incumbent_true_cost")
    final = result.get("final_incumbent_true_cost")
    return {
        "run_id": payload["run_id"],
        "method": payload.get("method"),
        "method_role": payload.get("method_role"),
        "diagnostic_only": payload.get("diagnostic_only"),
        "generator_pair": "-".join(str(value) for value in spec.get("generator_pair", [])),
        "window_start": spec.get("window_start"),
        "training_seed": spec.get("training_seed"),
        "run_seed": spec.get("run_seed"),
        "initial_true_cost": initial,
        "final_true_cost": final,
        "true_cost_improved": bool(isinstance(initial, (int, float)) and isinstance(final, (int, float)) and final < initial),
        "proposals": _counter(result, "proposals"),
        "threshold_updates": _counter(result, "threshold_updates"),
        "new_exact_evaluations": _counter(result, "new_exact_evaluations"),
        "actual_ed_lp_solves": _counter(result, "actual_ed_lp_solves"),
        "cache_hits": _counter(result, "cache_hits"),
        "logic_rejections": _counter(result, "logic_rejections"),
        "oracle_calls": _counter(result, "oracle_calls"),
        "stop_reason": result.get("stop_reason"),
        "scenario_id": scenario.get("scenario_id"),
        "global_optimum": "unavailable",
        "optimality_gap": "unavailable",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_summary(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output: list[dict[str, object]] = []
    for key, grouped in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        summary = dict(zip(fields, key))
        summary.update({
            "runs": len(grouped),
            "true_cost_improvement_runs": sum(bool(row["true_cost_improved"]) for row in grouped),
            "mean_final_true_cost": (
                sum(float(row["final_true_cost"]) for row in grouped if isinstance(row["final_true_cost"], (int, float)))
                / max(1, sum(isinstance(row["final_true_cost"], (int, float)) for row in grouped))
            ),
            "proposals": sum(int(row["proposals"]) for row in grouped),
            "threshold_updates": sum(int(row["threshold_updates"]) for row in grouped),
            "new_exact_evaluations": sum(int(row["new_exact_evaluations"]) for row in grouped),
            "actual_ed_lp_solves": sum(int(row["actual_ed_lp_solves"]) for row in grouped),
            "cache_hits": sum(int(row["cache_hits"]) for row in grouped),
            "logic_rejections": sum(int(row["logic_rejections"]) for row in grouped),
            "oracle_calls": sum(int(row["oracle_calls"]) for row in grouped),
        })
        output.append(summary)
    return output


def summarize_batch(output_dir: Path) -> dict[str, object]:
    """只读取落盘 JSON；绝不导入、构建或训练场景。"""

    output_dir = Path(output_dir)
    manifest = _load(output_dir / "batch_manifest.json")
    if manifest.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise SummaryValidationError("manifest schema_version 不兼容")
    expected_fingerprints = manifest.get("planned_fingerprints")
    planned_ids = manifest.get("planned_run_ids")
    if not isinstance(expected_fingerprints, Mapping) or not isinstance(planned_ids, list):
        raise SummaryValidationError("manifest 缺少计划 fingerprint/run_id 清单")
    completed = _completed_rows(output_dir, expected_fingerprints)
    completed_ids = {str(payload["run_id"]) for payload in completed}
    failed, recovered = _failed_ids(output_dir, expected_fingerprints, completed_ids)
    missing = sorted(set(str(value) for value in planned_ids) - completed_ids - failed)
    run_rows = [_run_row(payload) for payload in completed]
    method_rows = _group_summary(run_rows, ("method", "method_role", "diagnostic_only"))
    scenario_rows = _group_summary(run_rows, ("generator_pair", "window_start", "training_seed", "method", "method_role", "diagnostic_only"))
    summary = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "batch_id": manifest.get("batch_id"),
        "planned_runs": len(planned_ids),
        "completed_runs": len(completed),
        "failed_runs": len(failed),
        "recovered_failed_runs": len(recovered),
        "missing_runs": len(missing),
        "missing_run_ids": missing,
        "diagnostic_only_runs": sum(bool(row["diagnostic_only"]) for row in run_rows),
        "formal_runs": sum(not bool(row["diagnostic_only"]) for row in run_rows),
        "global_optimum": "unavailable (未执行 validation-only 全景后处理)",
        "optimality_gap": "unavailable (未执行 validation-only 全景后处理)",
        "statistical_claims": "not computed; 此汇总仅提供描述性计数和原始配对行",
        "method_summary": method_rows,
        "scenario_summary": scenario_rows,
    }
    summaries = output_dir / "summaries"
    _write_csv(summaries / "run_index.csv", run_rows)
    _write_csv(summaries / "method_summary.csv", method_rows)
    _write_csv(summaries / "scenario_summary.csv", scenario_rows)
    atomic_write_json(summaries / "summary.json", summary)
    return summary
