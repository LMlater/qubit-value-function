"""Build best-training snapshots only; never start Grover, MPS, or search."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_2x2_joint_feasible_sparse_vqc_bbht import build_case14_closed_loop_scenario  # noqa: E402
from qubit_value_function.closed_loop_metadata import build_quantized_model_snapshot  # noqa: E402
from qubit_value_function.fixed_point_oracle import FixedPointConfig  # noqa: E402
from qubit_value_function.sparse_vqc_bbht import BBHTConfig  # noqa: E402
from qubit_value_function.targeted_pilot_diagnostics import build_candidate_discovery_snapshot  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


class CandidateDiscoveryError(RuntimeError):
    pass


def ensure_new_output_dir(path: Path) -> None:
    if path.exists():
        raise CandidateDiscoveryError(f"refusing_existing_output_directory:{path}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Snapshot-only best-training candidate discovery; no Grover/MPS/search/global optimum.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = build_argument_parser().parse_args()
    ensure_new_output_dir(args.output_dir)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = load_uc_instance(args.instance)
    fixed = FixedPointConfig(fractional_bits=2, unit=1000.0, rounding="nearest")
    args.output_dir.mkdir(parents=True)
    snapshots = args.output_dir / "scenario_snapshots"
    snapshots.mkdir()
    rows: list[dict[str, object]] = []
    for item in config["scenarios"]:
        scenario = build_case14_closed_loop_scenario(
            source=source, window_start=int(item["window_start"]), selected_generator_indices=tuple(item["generator_pair"]),
            train_sample_count=8, fixed_point=fixed, initialization_policy="best_training", seed=int(item["training_seed"]),
            regularization=1e-4, maxiter=300, bbht_config=BBHTConfig(seed=0),
        )
        raw = build_quantized_model_snapshot(
            scenario_id=scenario.scenario_id, generator_pair=scenario.generator_pair, window_start=scenario.window_start,
            training_seed=scenario.training_seed, model=scenario.value_model, hard_logic_is_feasible=scenario.hard_logic_is_feasible,
            training_indices=scenario.training_indices, training_labels=scenario.training_labels, initial_incumbent_index=scenario.initial_incumbent_index,
            initial_incumbent_true_cost=float(scenario.initial_exact_cache[scenario.initial_incumbent_index].total_cost), initial_cache_indices=tuple(scenario.initial_exact_cache),
        )
        snapshot = build_candidate_discovery_snapshot(raw)
        path = snapshots / f"{scenario.scenario_id}.json"
        _write_json(path, snapshot)
        rows.append({"scenario_id": scenario.scenario_id, "generator_pair": list(scenario.generator_pair), "window_start": scenario.window_start,
            "training_seed": scenario.training_seed, "initial_cost_marked_count": snapshot["initial_cost_marked_count"], "initial_joint_marked_count": snapshot["initial_joint_marked_count"],
            "initial_training_joint_marked_count": snapshot["initial_training_joint_marked_count"], "initial_nontraining_cost_marked_count": snapshot["initial_nontraining_cost_marked_count"],
            "initial_nontraining_joint_marked_count": snapshot["initial_nontraining_joint_marked_count"], "nontraining_joint_marked_indices": snapshot["initial_nontraining_joint_marked_indices"],
            "has_nontraining_joint_marked_candidate": snapshot["has_nontraining_joint_marked_candidate"], "training_edlp_cache_hits": 0,
            "training_edlp_cache_misses": scenario.reproducibility_metadata["training_actual_ed_lp_solves"], "actual_training_edlp_solves": scenario.reproducibility_metadata["training_actual_ed_lp_solves"], "snapshot_path": str(path), "status": "ok"})
    _write_json(args.output_dir / "candidate_summary.json", rows)
    with (args.output_dir / "candidate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    _write_json(args.output_dir / "candidate_discovery_manifest.json", {**config, "snapshot_only": True, "no_grover_mps_or_search": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
