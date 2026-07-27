"""Read-only post-processing audit for persisted best-training smoke outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.post_best_training_diagnostics import (  # noqa: E402
    ensure_json_finite,
    quantization_diagnostic,
    trial_row,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only post best-training diagnostic")
    parser.add_argument("--input-dir", type=Path, default=Path("results/stage1_best_training_smoke_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage1_post_best_training_audit_v1"))
    parser.add_argument("--fractional-bits", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--snapshot-dir", type=Path, action="append", default=[])
    return parser


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    ensure_json_finite(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _snapshot_locations(args: argparse.Namespace) -> tuple[Path, ...]:
    defaults = (
        Path(args.input_dir) / "scenario_snapshots",
        Path("results/stage1_dynamic_oracle_metadata_smoke_v3/scenario_snapshots"),
        Path("results/stage1_dynamic_oracle_metadata_smoke_v2/scenario_snapshots"),
        Path("results/stage1_dynamic_oracle_metadata_smoke/scenario_snapshots"),
    )
    return tuple(Path(item) for item in (*args.snapshot_dir, *defaults))


def _load_snapshot(scenario_id: str, locations: tuple[Path, ...]) -> tuple[dict[str, object] | None, str | None]:
    for directory in locations:
        path = directory / f"{scenario_id}.json"
        if path.is_file():
            return _read_json(path), str(path)
    return None, None


def _pretrial_row(*, scenario: Mapping[str, object], policy: str, method: str, search: Mapping[str, object], initial: Mapping[str, object]) -> dict[str, object]:
    return {
        "scenario_id": scenario["scenario_id"], "method": method,
        "initial_incumbent_policy": policy, "run_seed": scenario["run_seed"],
        "pre_trial_stop": True, "stop_reason": search["stop_reason"],
        "initial_encoded_threshold": initial["initial_encoded_threshold"],
        "initial_cost_marked_count": initial["initial_cost_marked_count"],
        "initial_joint_marked_count": initial["initial_joint_marked_count"],
    }


def _summary(rows: list[dict[str, object]], pretrial: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, object, object], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["scenario_id"], row["initial_incumbent_policy"], row["method"]), []).append(row)
    output: list[dict[str, object]] = []
    for key, items in groups.items():
        actual = len(items)
        marked = sum(bool(item["measured_joint_marked"]) for item in items)
        output.append({
            "scenario_id": key[0], "initial_incumbent_policy": key[1], "method": key[2],
            "total_actual_grover_trials": actual,
            "m_zero_trial_count": sum(int(item["joint_marked_count_before"]) == 0 for item in items),
            "m_positive_trial_count": sum(int(item["joint_marked_count_before"]) > 0 for item in items),
            "measured_marked_count": marked,
            "theoretical_expected_marked_hits": sum(float(item["grover_theoretical_marked_probability"] or 0.0) for item in items),
            "uniform_expected_marked_hits": sum(float(item["uniform_marked_probability"] or 0.0) for item in items),
            "unmarked_count": sum(item["classification"] == "unmarked_measurement" for item in items),
            "marked_training_cache_count": sum(item["classification"] == "marked_training_cache" for item in items),
            "marked_search_cache_count": sum(item["classification"] == "marked_search_cache" for item in items),
            "new_ed_lp_count": sum(bool(item.get("new_ed_lp_solve")) for item in items),
            "true_improvement_count": sum(bool(item.get("true_strict_improvement")) for item in items),
            "threshold_update_count": sum(bool(item.get("threshold_updated")) for item in items),
            "trials_after_last_threshold_update": int(items[-1]["trials_since_last_threshold_update"] or 0),
            "final_stop_reason": items[-1].get("stop_reason_after_trial"),
        })
    for item in pretrial:
        output.append({**item, "total_actual_grover_trials": 0})
    return output


def _report_markdown(summary: list[dict[str, object]], sensitivity: list[dict[str, object]], unavailable: list[str]) -> str:
    g0 = [row for row in summary if row.get("scenario_id") == "case14-g0g5-w2-s0" and row.get("initial_incumbent_policy") == "best_training"]
    g0_text = json.dumps(g0[0], ensure_ascii=False) if g0 else "未找到"
    w0 = [row for row in sensitivity if row.get("scenario_id") == "case14-g1g3-w0-s0" and row.get("fractional_bits") == 2]
    w0_text = json.dumps(w0[0], ensure_ascii=False) if w0 else "快照不可用"
    return "\n".join([
        "# Post best-training read-only diagnostics", "",
        "本报告只读取已持久化 smoke trace 与量化模型快照；未执行 ED/LP、训练、RNG 或量子线路。", "",
        "## 直接回答", "",
        f"- g0g5-w2-s0 best-training：{g0_text}",
        "- M=1 时 k=0 的理论 marked 概率为 1/16；k=1 为 sin(3 asin(sqrt(1/16)))²。实际三次单 shot 不能用于性能结论。",
        "- 无新 ED/LP 的原因由逐 trial 分类给出：未标记测量不会准入；标记训练 cache 是已知状态，不触发新的精确求解。",
        f"- g1g3-w0-s0 fractional_bits=2：{w0_text}",
        "- 未修改停止规则或在线 fractional bits；是否应调整只能由 sensitivity 输出决定。", "",
        "## 数据可用性", "",
        "以下场景缺少任何持久化量化模型快照，故未重训或猜测：" + (", ".join(unavailable) if unavailable else "无"),
    ]) + "\n"


def main() -> int:
    args = build_argument_parser().parse_args()
    if set(args.fractional_bits) != {2, 3, 4, 5}:
        raise ValueError("this audit requires exactly fractional bits 2 3 4 5")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite audit output: {output_dir}")
    report_path = Path(args.input_dir) / "best_training_smoke_report.json"
    smoke = _read_json(report_path)
    output_dir.mkdir(parents=True)
    trajectories: list[dict[str, object]] = []
    pretrial: list[dict[str, object]] = []
    scenario_diagnostics: list[dict[str, object]] = []
    sensitivity: list[dict[str, object]] = []
    unavailable: list[str] = []
    locations = _snapshot_locations(args)
    for scenario in smoke["scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        policies = scenario["policies"]
        for policy, record in policies.items():
            search = record["search"]
            trace = list(search["trial_trace"])
            if trace:
                trajectories.extend(
                    trial_row(scenario_id=scenario_id, method="joint_bbht", policy=str(policy), run_seed=int(scenario["run_seed"]), row=row, search_space_size=16)
                    for row in trace
                )
            else:
                pretrial.append(_pretrial_row(scenario=scenario, policy=str(policy), method="joint_bbht", search=search, initial=record["initial"]))
        snapshot, snapshot_path = _load_snapshot(scenario_id, locations)
        if snapshot is None:
            unavailable.append(scenario_id)
            scenario_diagnostics.append({"scenario_id": scenario_id, "snapshot_status": "snapshot_unavailable"})
            continue
        best = policies["best_training"]
        threshold = float(best["initial"]["initial_incumbent_true_cost"])
        diagnostics = [quantization_diagnostic(snapshot, true_threshold=threshold, fractional_bits=bits) for bits in args.fractional_bits]
        sensitivity.extend(diagnostics)
        scenario_diagnostics.append({"scenario_id": scenario_id, "snapshot_status": "available", "snapshot_path": snapshot_path, "true_threshold": threshold, "quantization": diagnostics})
    breakdown = _summary(trajectories, pretrial)
    _write_json(output_dir / "summary.json", {"trial_summary": breakdown, "pre_trial_stops": pretrial, "snapshot_unavailable": unavailable})
    _write_json(output_dir / "trial_trajectory.json", trajectories)
    _write_json(output_dir / "quantization_sensitivity.json", sensitivity)
    _write_json(output_dir / "scenario_diagnostics.json", scenario_diagnostics)
    _write_csv(output_dir / "trial_trajectory.csv", trajectories)
    _write_csv(output_dir / "no_progress_breakdown.csv", breakdown)
    _write_csv(output_dir / "quantization_sensitivity.csv", sensitivity)
    (output_dir / "report.md").write_text(_report_markdown(breakdown, sensitivity, unavailable), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "actual_trials": len(trajectories), "snapshot_unavailable": unavailable}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
