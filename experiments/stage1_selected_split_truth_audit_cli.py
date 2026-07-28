"""Enumerate frozen selected-split states and obtain posthoc ED/LP truth labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_targeted_best_training_pilot_cli import _load_snapshot
from qubit_value_function.ed import FixedCommitmentEvaluator
from qubit_value_function.experiment_utils import embedded_selected_commitments, time_window_instance
from qubit_value_function.stage1_evidence import classify_unreachable, confusion_metrics, index_bitstring, subspace_optimum
from qubit_value_function.uc_loader import load_uc_instance


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _completed_final_indices(benchmark_dir: Path) -> dict[str, dict[str, Any]]:
    """Read completed selected-split results only; never reruns a method."""
    output: dict[str, dict[str, Any]] = {}
    for path in (benchmark_dir / "runs" / "completed").glob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        scenario = str(item["scenario"]["scenario_id"])
        result = item["result"]
        key = f"{scenario}:{item['method']}"
        candidate = {"method": item["method"], "final_index": result.get("final_incumbent_index"),
                     "final_true_cost": result.get("final_incumbent_true_cost"), "run_id": item.get("run_id")}
        old = output.get(key)
        if old is None or float(candidate["final_true_cost"]) < float(old["final_true_cost"]):
            output[key] = candidate
    return output


def run_audit(*, benchmark_dir: Path, instance_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output_dir}")
    started = datetime.now(timezone.utc).isoformat()
    execution = json.loads((benchmark_dir / "selected_split_execution_manifest.json").read_text(encoding="utf-8"))
    source = load_uc_instance(instance_path)
    head = _git_head()
    generated = datetime.now(timezone.utc).isoformat()
    final_results = _completed_final_indices(benchmark_dir)
    rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    optimum_rows: list[dict[str, Any]] = []
    for entry in execution["scenarios"]:
        snapshot_path = ROOT / Path(entry["snapshot"])
        snapshot = _load_snapshot(snapshot_path)
        scenario_id = str(snapshot["scenario_id"])
        training = {int(value) for value in snapshot["training_indices"]}
        labels = {int(item["state_index"]): float(item["true_cost"]) for item in snapshot["training_labels"]}
        table = {int(item["state_index"]): dict(item) for item in snapshot["state_proxy_table"]}
        instance = time_window_instance(source, start=int(snapshot["window_start"]), horizon=2)
        commitments = embedded_selected_commitments(np.ones((len(instance.generators), 2), dtype=int), tuple(int(x) for x in snapshot["generator_pair"]))
        evaluator = FixedCommitmentEvaluator(instance)
        threshold = int(snapshot["best_training_encoded_threshold"])
        scenario_rows: list[dict[str, Any]] = []
        for index in range(16):
            proxy = table[index]
            in_training = index in training
            if in_training:
                success, cost, source_name, result_path = True, labels[index], "training_snapshot_cache", str(snapshot_path)
                message = "persisted training label"
            else:
                answer = evaluator.evaluate(commitments[index])
                success = bool(answer.success and np.isfinite(answer.total_cost))
                cost = float(answer.total_cost) if success else None
                source_name, result_path, message = "new_ed_lp_call", None, str(answer.message)
            cost_marked = int(proxy["integer_vqc_value"]) < threshold
            joint_marked = bool(cost_marked and proxy["hard_logic_feasible"])
            row = {"scenario_id": scenario_id, "state_index": index, "bitstring": index_bitstring(index),
                   "in_training_set": in_training, "in_nontraining_set": not in_training, "hard_logic_feasible": bool(proxy["hard_logic_feasible"]),
                   "real_vqc_prediction": float(proxy["real_vqc_prediction"]), "predicted_fixed_point_cost": int(proxy["integer_vqc_value"]),
                   "value_register_integer": int(proxy["integer_vqc_value"]), "encoded_threshold": threshold, "cost_oracle_marked": cost_marked,
                   "joint_oracle_marked": joint_marked, "edlp_success": success, "true_cost": cost, "edlp_source": source_name,
                   "source_result_path": result_path, "edlp_message": message, "best_training_true_cost": float(snapshot["initial_incumbent_true_cost"]),
                   "true_improvement": bool(success and cost < float(snapshot["initial_incumbent_true_cost"])), "current_code_head": head,
                   "fixed_point_config": {"cost_unit": snapshot["cost_unit"], "fractional_bits": snapshot["fractional_bits"], "rounding": snapshot["quantization_mode"]},
                   "vqc_model_id": f"{scenario_id}:model_seed={snapshot['model_seed']}", "generated_at": generated}
            rows.append(row); scenario_rows.append(row)
        nontraining = [row for row in scenario_rows if row["in_nontraining_set"]]
        truth = [bool(row["true_improvement"]) for row in nontraining]
        for oracle in ("cost", "joint"):
            metrics = confusion_metrics([bool(row[f"{oracle}_oracle_marked"]) for row in nontraining], truth)
            confusion_rows.append({"scenario_id": scenario_id, "oracle": oracle, **metrics})
        reachable = any(bool(row["joint_oracle_marked"]) for row in nontraining)
        if not reachable:
            labels_out = classify_unreachable(true_improvement=truth, cost_marked=[bool(row["cost_oracle_marked"]) for row in nontraining], joint_marked=[bool(row["joint_oracle_marked"]) for row in nontraining], logic_feasible=[bool(row["hard_logic_feasible"]) for row in nontraining])
            classification_rows.append({"scenario_id": scenario_id, "labels": labels_out, "nontraining_true_improvement_count": sum(truth),
                                        "cost_false_negative_true_improvement_count": sum(t and not r["cost_oracle_marked"] for t, r in zip(truth, nontraining)),
                                        "hard_logic_excluded_true_improvement_count": sum(t and r["cost_oracle_marked"] and not r["hard_logic_feasible"] for t, r in zip(truth, nontraining))})
        candidates = [int(value) for value in snapshot["initial_nontraining_joint_marked_indices"]]
        discovered = candidates[0] if len(candidates) == 1 else None
        optimum = subspace_optimum([row["true_cost"] for row in scenario_rows], discovered_index=discovered)
        method_results = [value for key, value in final_results.items() if key.startswith(scenario_id + ":")]
        optimum_rows.append({"scenario_id": scenario_id, "best_training_index": int(snapshot["initial_incumbent_index"]), "best_training_true_cost": float(snapshot["initial_incumbent_true_cost"]),
                             "initial_nontraining_joint_marked_indices": candidates, **optimum, "best_completed_method_results": method_results})
    nontraining_rows = [row for row in rows if row["in_nontraining_set"]]
    aggregate_metrics = {oracle: confusion_metrics([bool(row[f"{oracle}_oracle_marked"]) for row in nontraining_rows], [bool(row["true_improvement"]) for row in nontraining_rows]) for oracle in ("cost", "joint")}
    summary = {"scope": "posthoc selected-split 16-state truth audit; truth is not used for online selection", "scenario_count": 12,
               "state_rows": len(rows), "nontraining_state_rows": len(nontraining_rows), "new_edlp_calls": sum(row["edlp_source"] == "new_ed_lp_call" for row in rows),
               "nontraining_true_improvement_states": sum(bool(row["true_improvement"]) for row in nontraining_rows), "aggregate_oracle_metrics": aggregate_metrics,
               "joint_unreachable_scenarios": len(classification_rows), "positive_initial_joint_scenarios": sum(bool(row["initial_nontraining_joint_marked_indices"]) for row in optimum_rows),
               "completion_status": "completed", "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat()}
    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "scenario_state_truth_table.csv", rows); _write_json(output_dir / "scenario_state_truth_table.json", rows)
    _write_csv(output_dir / "scenario_oracle_confusion_matrix.csv", confusion_rows); _write_csv(output_dir / "scenario_oracle_metrics.csv", confusion_rows)
    _write_csv(output_dir / "scenario_unreachable_classification.csv", classification_rows); _write_csv(output_dir / "scenario_subspace_optimum_audit.csv", optimum_rows)
    _write_json(output_dir / "summary.json", summary)
    manifest = {"current_code_head": head, "python": sys.version, "input_benchmark_dir": str(benchmark_dir), "input_instance": str(instance_path),
                "input_hashes": {"execution_manifest": _sha256(benchmark_dir / "selected_split_execution_manifest.json"), "instance": _sha256(instance_path)},
                "parameters": {"dimension": 16, "training_count": 8, "strict_truth_comparison": "true_cost < best_training_true_cost"},
                "started_at": started, "completed_at": summary["completed_at"], "completion_status": "completed"}
    _write_json(output_dir / "manifest.json", manifest)
    report = "# Selected-split truth audit\n\n" + f"- 12 frozen scenarios; {summary['new_edlp_calls']} new nontraining ED/LP evaluations.\n" + f"- Cost metrics: {json.dumps(aggregate_metrics['cost'])}\n- Joint metrics: {json.dumps(aggregate_metrics['joint'])}\n" + "\nThis is a 16-state selected-subspace posthoc audit, not a global UC optimum claim.\n"
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(benchmark_dir=args.benchmark_dir, instance_path=args.instance, output_dir=args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
