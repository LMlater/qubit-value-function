"""Supplement official Stage B truth rows with reproducible ED/LP components."""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.commitment import is_logic_feasible
from qubit_value_function.ed import FixedCommitmentEvaluator
from qubit_value_function.experiment_utils import embedded_selected_commitments, time_window_instance
from qubit_value_function.load_scenarios import scaled_load_instance
from qubit_value_function.logic_feasibility_oracle import compile_logic_feasibility_spec
from qubit_value_function.stage2_edlp_component_audit import (
    ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, component_cost_sum, cost_consistency_record,
    join_by_formal_key, require_hard_logic_agreement, strict_json_dumps,
    validate_audit_paths, validate_formal_truth_rows,
)
from qubit_value_function.uc_loader import load_uc_instance


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _pair(value: object) -> tuple[int, int]:
    values = ast.literal_eval(str(value))
    return tuple(int(item) for item in values)  # type: ignore[return-value]


def _commitment_cache(source: Any, pair: tuple[int, int], window_start: int) -> tuple[Any, np.ndarray, Any]:
    window = time_window_instance(source, start=window_start, horizon=2)
    base = np.ones((len(window.generators), 2), dtype=int)
    commitments = embedded_selected_commitments(base, pair)
    specification = compile_logic_feasibility_spec(window, selected_generator_indices=pair, base_commitment=base)
    return window, commitments, specification


def audit_rows(formal_rows: list[dict[str, object]], *, instance_path: Path) -> tuple[list[dict[str, object]], int]:
    """Re-evaluate exactly the formal rows, keyed rather than CSV-positioned."""

    validate_formal_truth_rows(formal_rows)
    source = load_uc_instance(instance_path)
    contexts: dict[tuple[tuple[int, int], int], tuple[Any, np.ndarray, Any]] = {}
    evaluators: dict[tuple[tuple[int, int], int, float], FixedCommitmentEvaluator] = {}
    output: list[dict[str, object]] = []
    for formal in formal_rows:
        pair, window_start, multiplier, state_index = _pair(formal["generator_pair"]), int(formal["window_start"]), float(formal["load_multiplier"]), int(formal["state_index"])
        context_key = (pair, window_start)
        if context_key not in contexts:
            contexts[context_key] = _commitment_cache(source, pair, window_start)
        window, commitments, specification = contexts[context_key]
        evaluator_key = (pair, window_start, multiplier)
        if evaluator_key not in evaluators:
            evaluators[evaluator_key] = FixedCommitmentEvaluator(scaled_load_instance(window, multiplier))
        commitment = commitments[state_index]
        production_logic = bool(is_logic_feasible(window, commitment))
        independent_logic = bool(specification.is_feasible(tuple((state_index >> bit) & 1 for bit in range(4))))
        result = evaluators[evaluator_key].evaluate(commitment)
        success = bool(result.success and np.isfinite(result.total_cost))
        dispatch = float(result.dispatch_cost) if success else None
        startup = float(result.startup_cost) if success else None
        balance = float(result.balance_penalty) if success else None
        reserve = float(result.reserve_penalty) if success else None
        total = component_cost_sum(dispatch_cost=dispatch, startup_cost=startup, balance_penalty=balance, reserve_penalty=reserve) if success else None
        output.append({
            "generator_pair": formal["generator_pair"], "window_start": window_start, "load_multiplier": multiplier,
            "state_index": state_index, "true_cost": float(formal["true_cost"]),
            "hard_logic_feasible": production_logic, "independent_hard_logic_feasible": independent_logic,
            "hard_logic_consistent": production_logic == independent_logic, "edlp_success": success,
            "dispatch_cost": dispatch, "startup_cost": startup, "balance_penalty": balance,
            "reserve_penalty": reserve, "recomputed_total_cost": total, "evaluator_message": str(result.message),
        })
    require_hard_logic_agreement(output)
    return output, len(output)


def _summary(rows: list[dict[str, object]], consistency: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if bool(row["edlp_success"])]
    def stats(field: str) -> dict[str, float]:
        values = np.asarray([float(row[field]) for row in successful], dtype=float)
        return {"mean": float(values.mean()), "median": float(np.median(values)), "minimum": float(values.min()), "maximum": float(values.max()), "sum": float(values.sum())}
    return {
        "row_count": len(rows), "edlp_success_count": len(successful),
        "hard_logic_feasible_count": sum(bool(row["hard_logic_feasible"]) for row in rows),
        "hard_logic_consistency_count": sum(bool(row["hard_logic_consistent"]) for row in rows),
        "cost_consistency_count": sum(bool(row["within_tolerance"]) for row in consistency),
        "components": {field: stats(field) for field in ("dispatch_cost", "startup_cost", "balance_penalty", "reserve_penalty", "recomputed_total_cost")},
        "maximum_absolute_difference": float(max(float(row["absolute_difference"] or 0.0) for row in consistency)),
        "maximum_relative_difference": float(max(float(row["relative_difference"] or 0.0) for row in consistency)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    args = parser.parse_args(); validate_audit_paths(args.formal_benchmark_dir, args.output_dir)
    if args.output_dir.exists(): raise RuntimeError(f"refusing_existing_output_directory:{args.output_dir}")
    truth_path = args.formal_benchmark_dir / "truth_table.csv"; formal_rows = _read_csv(truth_path); validate_formal_truth_rows(formal_rows)
    started = datetime.now(timezone.utc); components, calls = audit_rows(formal_rows, instance_path=args.instance)
    consistency = [cost_consistency_record(formal, component) for formal, component in zip(formal_rows, components)]
    if not all(bool(row["within_tolerance"]) for row in consistency):
        status = "cost_consistency_failed"
    else:
        status = "completed_cost_consistent"
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "component_truth_table.csv", join_by_formal_key(formal_rows, components))
    _write_csv(args.output_dir / "cost_consistency_audit.csv", consistency)
    _write_csv(args.output_dir / "hard_logic_consistency.csv", [{key: row[key] for key in ("generator_pair", "window_start", "load_multiplier", "state_index", "hard_logic_feasible", "independent_hard_logic_feasible", "hard_logic_consistent")} for row in components])
    summary = _summary(components, consistency)
    protocol = {"version": "stage2-edlp-component-audit-v1", "scope": "exactly formal truth_table keys", "cost_tolerance": {"absolute": ABSOLUTE_TOLERANCE, "relative": RELATIVE_TOLERANCE}, "formal_benchmark_is_read_only": True, "hard_logic": {"production": "qubit_value_function.commitment.is_logic_feasible", "independent": "qubit_value_function.logic_feasibility_oracle.compile_logic_feasibility_spec"}}
    manifest = {"formal_benchmark_dir": args.formal_benchmark_dir.as_posix(), "formal_truth_table_sha256": _sha256(truth_path), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "working_tree_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(), "python": sys.version, "numpy": np.__version__, "scipy": __import__("scipy").__version__, "case14_source_sha256": _sha256(args.instance), "evaluator": "qubit_value_function.ed.FixedCommitmentEvaluator", "hard_logic": protocol["hard_logic"], "tolerance": protocol["cost_tolerance"], "started_at_utc": started.isoformat(), "finished_at_utc": datetime.now(timezone.utc).isoformat(), "edlp_calls": calls, "completion_status": status}
    for name, payload in (("protocol.json", protocol), ("aggregate_component_summary.json", summary), ("manifest.json", manifest)):
        (args.output_dir / name).write_text(strict_json_dumps(payload), encoding="utf-8")
    (args.output_dir / "report.md").write_text("# Supplemental Stage B ED/LP component audit\n\nThis audit supplements missing hard-logic labels and ED/LP cost components. It does not modify the formal benchmark, its predictions, metrics, or conclusions. It may be joined to formal predictions only when all total-cost consistency checks pass. Reports must cite the formal benchmark and this supplemental audit separately.\n\n" + f"- completion: `{status}`\n- ED/LP calls: {calls}\n- cost-consistent rows: {summary['cost_consistency_count']}/{len(consistency)}\n", encoding="utf-8")
    print(strict_json_dumps({"output_dir": str(args.output_dir), "completion_status": status, **summary})); return 0


if __name__ == "__main__": raise SystemExit(main())
