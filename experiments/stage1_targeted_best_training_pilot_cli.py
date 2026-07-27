"""Artifact-only preflight for the targeted best-training pilot.

This command deliberately performs no ED/LP solve, VQC training, random draw,
or global-optimum lookup.  A run is permitted only after this preflight proves
that a manifest has a nontraining joint-marked candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.targeted_pilot_diagnostics import initial_marked_counts  # noqa: E402


PRIMARY_METHODS = (
    "joint_bbht",
    "cost_only_bbht",
    "full_space_random",
    "logic_rejection_random",
)


class PilotPreflightError(RuntimeError):
    """The persisted artifacts cannot safely support this targeted pilot."""


def _csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("seed list must not be empty")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("seed list must not contain duplicates")
    return values


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only best-training targeted pilot preflight.  It never overwrites "
            "results, does not prove quantum advantage, and Aer MPS is classical simulation."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-seeds", type=_csv_ints)
    parser.add_argument("--seed-range", type=int, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--methods", type=lambda raw: tuple(item.strip() for item in raw.split(",") if item.strip()))
    parser.add_argument("--preflight", action="store_true")
    return parser


def _snapshot_threshold(snapshot: Mapping[str, object]) -> int:
    labels = list(snapshot.get("training_labels", []))
    if not labels:
        raise PilotPreflightError("snapshot_missing_training_labels")
    costs = [
        float(item["true_cost"]) if isinstance(item, Mapping) else float(item)
        for item in labels
    ]
    if "encoded_cost_scale" in snapshot:
        scale = int(snapshot["encoded_cost_scale"])
        unit = float(snapshot["cost_unit"])
    else:
        model = dict(snapshot.get("quantized_model", {}))
        fixed = dict(model.get("fixed_point", {}))
        scale = int(fixed["scale"])
        unit = float(fixed["cost_unit"])
    return int(np.rint(min(costs) * scale / unit))


def _load_snapshot(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise PilotPreflightError(f"snapshot_unavailable:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "state_proxy_table" not in payload:
        raise PilotPreflightError(f"snapshot_unsupported_schema:{path}")
    return payload


def preflight_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    """Derive scenario eligibility solely from declared saved artifacts."""

    scenarios = list(manifest.get("scenarios", []))
    if not scenarios:
        raise PilotPreflightError("manifest_has_no_scenarios")
    rows: list[dict[str, object]] = []
    for entry in scenarios:
        item = dict(entry)
        role = str(item["role"])
        if role not in {
            "grover_calibration_training_marked",
            "nontraining_improvement_candidate",
            "joint_marked_empty_control",
        }:
            raise PilotPreflightError(f"unsupported_scenario_role:{role}")
        snapshot = _load_snapshot(Path(str(item["snapshot"])))
        threshold = _snapshot_threshold(snapshot)
        counts = initial_marked_counts(snapshot, encoded_threshold=threshold)
        rows.append(
            {
                "scenario_id": str(item["scenario_id"]),
                "role": role,
                "snapshot": str(item["snapshot"]),
                "best_training_encoded_threshold": threshold,
                **counts,
                "has_initial_nontraining_joint_marked": bool(
                    counts["initial_nontraining_joint_marked_count"]
                ),
            }
        )
    eligible = [
        row for row in rows
        if row["role"] == "nontraining_improvement_candidate"
        and int(row["initial_nontraining_joint_marked_count"]) > 0
    ]
    if not eligible:
        raise PilotPreflightError("no_nontraining_joint_marked_candidate")
    return {
        "selection_status": "eligible",
        "targeted_diagnostic_selection": True,
        "formal": False,
        "diagnostic_only": True,
        "scenarios": rows,
    }


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.run_seeds is not None and args.seed_range is not None:
        raise PilotPreflightError("choose --run-seeds or --seed-range, not both")
    manifest = json.loads(args.config.read_text(encoding="utf-8"))
    report = preflight_manifest(manifest)
    seeds = (
        tuple(args.run_seeds)
        if args.run_seeds is not None
        else tuple(range(args.seed_range[0], args.seed_range[1] + 1))
        if args.seed_range is not None
        else tuple(range(20))
    )
    methods = tuple(args.methods) if args.methods is not None else PRIMARY_METHODS
    if set(methods) - set(PRIMARY_METHODS):
        raise PilotPreflightError("unsupported_primary_method")
    plan = {**report, "run_seeds": list(seeds), "methods": list(methods), "output_dir": str(args.output_dir)}
    if args.preflight:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.output_dir.exists():
        raise PilotPreflightError(f"refusing_existing_output_directory:{args.output_dir}")
    raise PilotPreflightError(
        "execution_not_started: preflight is implemented; a manifest with restored frozen scenarios is required before scheduling"
    )


if __name__ == "__main__":
    raise SystemExit(main())
