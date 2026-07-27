"""Small, isolated behaviour-invariance smoke for dynamic oracle metadata.

It deliberately never invokes the formal/pilot/baseline batch CLIs and refuses
to overwrite an output directory.  Each selected method is run twice from one
frozen scenario: metadata off, then metadata on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_2x2_joint_feasible_sparse_vqc_bbht import build_case14_closed_loop_scenario
from qubit_value_function.closed_loop_batch import atomic_write_json
from qubit_value_function.closed_loop_metadata import canonical_sha256
from qubit_value_function.closed_loop_scenario import run_closed_loop_method
from qubit_value_function.fixed_point_oracle import FixedPointConfig
from qubit_value_function.sparse_vqc_bbht import BBHTConfig
from qubit_value_function.uc_loader import load_uc_instance


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage A dynamic oracle metadata behaviour-invariance smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage1_dynamic_oracle_metadata_smoke"))
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--train-sample-count", type=int, default=8)
    parser.add_argument("--fractional-bits", type=int, default=2)
    parser.add_argument("--cost-unit", type=float, default=1000.0)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--max-oracle-calls", type=int, default=4)
    parser.add_argument("--max-new-ed-lp-calls", type=int, default=2)
    parser.add_argument("--max-threshold-updates", type=int, default=1)
    return parser


def _trial_signature(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "sampled_grover_iterations": row.get("sampled_grover_iterations", row.get("grover_iterations")),
        "measured_index": row.get("measured_index", row.get("candidate_index")),
        "candidate_admitted": row.get("admission_passed"),
        "cache_hit": row.get("candidate_cache_hit"),
        "new_ed_lp_solve": row.get("new_edlp_solve_performed"),
        "candidate_true_cost": row.get("exact_cost"),
        "true_strict_improvement": row.get("true_improvement"),
        "incumbent_updated": row.get("accepted_update"),
        "true_threshold_before": row.get("true_threshold_before"),
        "true_threshold_after": row.get("true_threshold_after"),
        "encoded_threshold_before": row.get("encoded_threshold_before"),
        "encoded_threshold_after": row.get("encoded_threshold_after"),
        "stop_reason_after_trial": row.get("stop_reason_after_trial", row.get("stop_reason_after_proposal")),
    }


def assert_behavior_invariance(off_result: Mapping[str, object], on_result: Mapping[str, object]) -> dict[str, object]:
    off_trace = list(off_result.get("trial_trace", []))
    on_trace = list(on_result.get("trial_trace", []))
    if len(off_trace) != len(on_trace):
        raise AssertionError("metadata off/on produced different trial counts")
    signatures = []
    for trial_number, (off_row, on_row) in enumerate(zip(off_trace, on_trace), start=1):
        if not isinstance(off_row, Mapping) or not isinstance(on_row, Mapping):
            raise AssertionError("trial trace must contain JSON mappings")
        off_signature = _trial_signature(off_row)
        on_signature = _trial_signature(on_row)
        if off_signature != on_signature:
            raise AssertionError(f"metadata off/on mismatch at trial {trial_number}: {off_signature} != {on_signature}")
        signatures.append(off_signature)
    fields = (
        "final_incumbent_index", "final_incumbent_true_cost", "final_encoded_threshold",
        "stop_reason", "exact_cache",
    )
    for field in fields:
        if off_result.get(field) != on_result.get(field):
            raise AssertionError(f"metadata off/on final-state mismatch: {field}")
    off_counters = off_result.get("counters", {})
    on_counters = on_result.get("counters", {})
    for field in ("total_oracle_calls", "actual_ed_lp_solves", "new_exact_evaluation_attempts", "cached_exact_lookups"):
        if off_counters.get(field) != on_counters.get(field):
            raise AssertionError(f"metadata off/on counter mismatch: {field}")
    return {"trials": signatures, "counters": dict(on_counters)}


def _persist_snapshot(output_dir: Path, snapshot: Mapping[str, object]) -> dict[str, object]:
    scenario_id = str(snapshot["scenario_id"])
    relative = Path("scenario_snapshots") / f"{scenario_id}.json"
    path = output_dir / relative
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(snapshot):
            raise AssertionError("same smoke scenario produced inconsistent quantized snapshots")
    else:
        atomic_write_json(path, dict(snapshot))
    return {"path": relative.as_posix(), "sha256": canonical_sha256(snapshot), "version": snapshot["quantized_model_snapshot_version"]}


def main() -> int:
    args = build_argument_parser().parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing smoke output: {output_dir}")
    output_dir.mkdir(parents=True)
    source = load_uc_instance(args.instance)
    fixed = FixedPointConfig(fractional_bits=args.fractional_bits, unit=args.cost_unit, rounding="nearest")
    config = BBHTConfig(
        max_trials=args.max_trials,
        max_oracle_calls=args.max_oracle_calls,
        max_new_ed_lp_calls=args.max_new_ed_lp_calls,
        max_threshold_updates=args.max_threshold_updates,
        seed=0,
    )
    scenario_methods = (
        ((0, 5), 2, 0, ("joint_bbht", "cost_only_bbht", "full_space_random")),
        ((1, 3), 0, 0, ("joint_bbht",)),
    )
    rows: list[dict[str, object]] = []
    scenario_builds: list[dict[str, object]] = []
    for generator_pair, window_start, training_seed, methods in scenario_methods:
        scenario = build_case14_closed_loop_scenario(
            source=source,
            window_start=window_start,
            selected_generator_indices=generator_pair,
            train_sample_count=args.train_sample_count,
            fixed_point=fixed,
            initialization_policy="first",
            seed=training_seed,
            regularization=1e-4,
            maxiter=args.maxiter,
            bbht_config=config,
        )
        scenario_builds.append(
            {
                "scenario_id": scenario.scenario_id,
                "training_initial_cache_size": len(scenario.initial_exact_cache),
                "training_initial_actual_ed_lp_solves": int(
                    scenario.reproducibility_metadata.get("training_actual_ed_lp_solves", 0)
                ),
            }
        )
        for method in methods:
            off = run_closed_loop_method(scenario, method, run_seed=args.run_seed, persist_dynamic_oracle_metadata=False)
            on = run_closed_loop_method(scenario, method, run_seed=args.run_seed, persist_dynamic_oracle_metadata=True)
            off_result = dict(off["result_schema"])
            on_result = dict(on["result_schema"])
            comparison = assert_behavior_invariance(off_result, on_result)
            snapshot_reference = _persist_snapshot(output_dir, on["quantized_model_snapshot"])
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "method": method,
                    "run_seed": int(args.run_seed),
                    "metadata_invariance_passed": True,
                    "snapshot": snapshot_reference,
                    "comparison": comparison,
                    "metadata_off_counters": dict(off_result.get("counters", {})),
                    "metadata_on_counters": dict(on_result.get("counters", {})),
                    "metadata_on_result": on_result,
                }
            )
    report = {
        "result_schema_version": "stage1_closed_loop_dynamic_oracle_v1",
        "scope": "small independent behaviour-invariance smoke; no formal or pilot batch",
        "config": {
            **{key: value for key, value in vars(args).items() if key not in {"output_dir", "instance"}},
            "output_dir": str(args.output_dir),
            "instance": str(args.instance),
        },
        "runs": rows,
        "scenario_builds": scenario_builds,
        "total_method_runs": len(rows),
        "quantum_trials_per_mode": sum(
            len(row["comparison"]["trials"])
            for row in rows if row["method"] in {"joint_bbht", "cost_only_bbht"}
        ),
        "total_quantum_trials_across_off_on": 2 * sum(
            len(row["comparison"]["trials"])
            for row in rows if row["method"] in {"joint_bbht", "cost_only_bbht"}
        ),
        "actual_ed_lp_solves_per_mode": sum(
            int(row["comparison"]["counters"].get("actual_ed_lp_solves", 0)) for row in rows
        ),
        "total_actual_ed_lp_solves_across_off_on": 2 * sum(
            int(row["comparison"]["counters"].get("actual_ed_lp_solves", 0)) for row in rows
        ),
        "total_training_initial_ed_lp_solves": sum(
            int(row["training_initial_actual_ed_lp_solves"]) for row in scenario_builds
        ),
    }
    report["total_actual_ed_lp_solves_including_training"] = (
        int(report["total_actual_ed_lp_solves_across_off_on"])
        + int(report["total_training_initial_ed_lp_solves"])
    )
    atomic_write_json(output_dir / "dynamic_oracle_metadata_smoke_report.json", report)
    (output_dir / "README.md").write_text(
        "# Dynamic oracle metadata smoke\n\n"
        "Each selected frozen scenario/method ran once with metadata off and once with it on. "
        "The JSON report records an exact comparison of decision, cache, threshold, and budget fields.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "total_method_runs": len(rows), "total_quantum_trials_across_off_on": report["total_quantum_trials_across_off_on"], "total_actual_ed_lp_solves_including_training": report["total_actual_ed_lp_solves_including_training"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
