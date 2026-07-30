"""Build a minimal, frozen Stage 1 evidence fixture from verified full results.

This is an explicit evidence-maintenance command.  Default pytest never calls
it and never reads the full historical results directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping


EXTRACTOR_VERSION = "stage1-evidence-fixture-v1"
WHITELIST_VERSION = "stage1-evidence-fields-v1"

FORMAL_MANIFEST = Path("stage1_closed_loop_baseline_formal/batch_manifest.json")
SNAPSHOT_ROOT = Path("stage1_best_training_candidate_discovery_rewired_seed1_20260727_212932/scenario_snapshots")
TARGETED_ROOT = Path("stage1_targeted_best_training_candidate_pilot_20260727_221524")
SELECTED_ROOT = Path("stage1_best_training_selected_split_benchmark_fixed_20260727_231759")

EXPECTED_SOURCE_SHA256 = {
    FORMAL_MANIFEST.as_posix(): "2f225f75fb000a216ea3835dee6b231588bbdcf589010d9f89335c6408ffac03",
    (TARGETED_ROOT / "summaries/targeted_pilot_summary.json").as_posix(): "3e6f8a9b6963ea3a87f17e19cbda1d6fa807a3744694959b8728d7406fef469e",
    (SELECTED_ROOT / "batch_manifest.json").as_posix(): "5cd8732eeef15870145fecdd3208e27cecc7071641851b1f9b8ea0e5db1f787b",
    (SELECTED_ROOT / "summaries/selected_split_summary.json").as_posix(): "0257f91af4a747e2504881ae6e58bc0d323a726671aecd19de9f73c31c54e1ac",
}


class FixtureBuildError(RuntimeError):
    """The full historical evidence cannot safely be converted to a fixture."""


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureBuildError(f"source_json_unavailable:{path}") from error
    if not isinstance(payload, dict):
        raise FixtureBuildError(f"source_json_not_object:{path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(mapping: Mapping[str, Any], key: str, *, context: str) -> Any:
    if key not in mapping:
        raise FixtureBuildError(f"missing_required_field:{context}:{key}")
    return mapping[key]


def _pick(mapping: Mapping[str, Any], keys: tuple[str, ...], *, context: str) -> dict[str, Any]:
    return {key: _require(mapping, key, context=context) for key in keys}


def _project_formal_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _pick(payload, ("methods", "budget_config"), context="formal_manifest")


def _project_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    state_proxy = [
        _pick(dict(row), ("state_index", "integer_vqc_value", "hard_logic_feasible"), context="snapshot_state_proxy")
        for row in _require(payload, "state_proxy_table", context="snapshot")
    ]
    labels = [
        _pick(dict(row), ("state_index", "true_cost"), context="snapshot_training_label")
        for row in _require(payload, "training_labels", context="snapshot")
    ]
    return {
        **_pick(
            payload,
            (
                "best_training_encoded_threshold",
                "cost_unit",
                "encoded_cost_scale",
                "generator_pair",
                "initial_cache_indices",
                "initial_incumbent_index",
                "model_seed",
                "scenario_id",
                "training_data_seed",
                "training_indices",
                "training_indices_hash",
                "window_start",
            ),
            context="snapshot",
        ),
        "state_proxy_table": state_proxy,
        "training_labels": labels,
    }


def _project_targeted_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _pick(
        payload,
        ("planned_runs", "completed_runs", "outside_training_strict_improvement_run_count"),
        context="targeted_summary",
    )


def _project_targeted_run(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario = dict(_require(payload, "scenario", context="targeted_run"))
    result = dict(_require(payload, "result", context="targeted_run"))
    counters = dict(_require(result, "counters", context="targeted_run_result"))
    trace = [
        _pick(
            dict(row),
            (
                "true_strict_improvement",
                "candidate_in_training_set",
                "measured_index",
                "candidate_true_cost",
                "incumbent_updated",
                "threshold_updated",
            ),
            context="targeted_trial_trace",
        )
        for row in _require(result, "trial_trace", context="targeted_run_result")
    ]
    return {
        "scenario": _pick(scenario, ("scenario_id", "training_indices"), context="targeted_scenario"),
        "result": {
            "counters": _pick(counters, ("actual_ed_lp_solves",), context="targeted_counters"),
            "trial_trace": trace,
        },
    }


def _project_selected_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    environment = dict(_require(payload, "environment", context="selected_manifest"))
    counts = dict(_require(payload, "counts", context="selected_manifest"))
    return {
        "counts": _pick(counts, ("completed", "failed"), context="selected_manifest_counts"),
        "environment": _pick(environment, ("head",), context="selected_manifest_environment"),
        "expected_head": _require(payload, "expected_head", context="selected_manifest"),
    }


def _project_selected_run(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(_require(payload, "result", context="selected_run"))
    counters = dict(_require(result, "counters", context="selected_run_result"))
    trace = []
    for row in _require(result, "trial_trace", context="selected_run_result"):
        source_row = dict(row)
        projected_row = _pick(source_row, ("true_strict_improvement",), context="selected_trial_trace")
        if "candidate_in_training_set" in source_row:
            projected_row["candidate_in_training_set"] = source_row["candidate_in_training_set"]
        trace.append(projected_row)
    counter_names = (
        "proposals_used",
        "circuit_executions",
        "total_oracle_calls",
        "cached_exact_lookups",
        "new_exact_evaluation_attempts",
        "actual_ed_lp_solves",
    )
    return {
        "method": _require(payload, "method", context="selected_run"),
        "run_spec": _pick(dict(_require(payload, "run_spec", context="selected_run")), ("run_seed",), context="selected_run_spec"),
        "scenario": _pick(dict(_require(payload, "scenario", context="selected_run")), ("scenario_id",), context="selected_scenario"),
        "result": {
            **_pick(
                result,
                (
                    "initial_incumbent_true_cost",
                    "final_incumbent_true_cost",
                    "oracle_reachability_stratum",
                    "stop_reason",
                ),
                context="selected_run_result",
            ),
            "counters": {name: counters[name] for name in counter_names if name in counters},
            "trial_trace": trace,
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _collect_paths(source_root: Path) -> list[tuple[Path, Callable[[Mapping[str, Any]], dict[str, Any]]]]:
    snapshots = sorted((source_root / SNAPSHOT_ROOT).glob("*.json"))
    targeted_runs = sorted((source_root / TARGETED_ROOT / "runs/completed").glob("*.json"))
    selected_runs = sorted((source_root / SELECTED_ROOT / "runs/completed").glob("*.json"))
    if len(snapshots) != 12:
        raise FixtureBuildError(f"snapshot_count_is_not_12:{len(snapshots)}")
    if len(targeted_runs) != 80:
        raise FixtureBuildError(f"targeted_completed_run_count_is_not_80:{len(targeted_runs)}")
    if len(selected_runs) != 360:
        raise FixtureBuildError(f"selected_completed_run_count_is_not_360:{len(selected_runs)}")
    projections: list[tuple[Path, Callable[[Mapping[str, Any]], dict[str, Any]]]] = [
        (FORMAL_MANIFEST, _project_formal_manifest),
        *((path.relative_to(source_root), _project_snapshot) for path in snapshots),
        (TARGETED_ROOT / "summaries/targeted_pilot_summary.json", _project_targeted_summary),
        *((path.relative_to(source_root), _project_targeted_run) for path in targeted_runs),
        (SELECTED_ROOT / "batch_manifest.json", _project_selected_manifest),
        (SELECTED_ROOT / "summaries/selected_split_summary.json", lambda payload: dict(payload)),
        *((path.relative_to(source_root), _project_selected_run) for path in selected_runs),
    ]
    for path, _projection in projections:
        if not (source_root / path).is_file():
            raise FixtureBuildError(f"expected_source_file_missing:{path.as_posix()}")
    return projections


def build_fixture(*, source_root: Path, output_dir: Path, generated_at: str) -> dict[str, Any]:
    """Create the fixture once; neither output nor a replacement destination may exist."""

    source_root, output_dir = Path(source_root), Path(output_dir)
    if output_dir.exists():
        raise FixtureBuildError(f"fixture_output_directory_already_exists:{output_dir}")
    if not source_root.is_dir():
        raise FixtureBuildError(f"source_results_root_unavailable:{source_root}")
    projections = _collect_paths(source_root)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        source_sha256: dict[str, str] = {}
        source_files: list[dict[str, Any]] = []
        files: dict[str, dict[str, Any]] = {}
        for relative, projection in projections:
            source_path = source_root / relative
            source_hash = _sha256(source_path)
            expected = EXPECTED_SOURCE_SHA256.get(relative.as_posix())
            if expected is not None and source_hash != expected:
                raise FixtureBuildError(f"canonical_source_sha256_mismatch:{relative.as_posix()}:{source_hash}")
            projected = projection(_load(source_path))
            destination = temporary / relative
            _write_json(destination, projected)
            fixture_relative = relative.as_posix()
            source_sha256[fixture_relative] = source_hash
            source_files.append({"path": fixture_relative, "sha256": source_hash, "bytes": source_path.stat().st_size})
            files[fixture_relative] = {"sha256": _sha256(destination), "bytes": destination.stat().st_size}
        source_files.sort(key=lambda item: str(item["path"]))
        fixture_bytes = sum(int(item["bytes"]) for item in files.values())
        source_bytes = sum(int(item["bytes"]) for item in source_files)
        manifest = {
            "schema_version": 1,
            "fixture_purpose": "Minimal frozen mirror for four Stage 1 historical evidence-integration tests.",
            "fixture_is_not_complete_formal_results": True,
            "source_result_directories": sorted({str(path.parts[0]) for path, _projection in projections}),
            "source_paths_are_relative_to": "the supplied full historical results root",
            "source_sha256": dict(sorted(source_sha256.items())),
            "source_files": source_files,
            "files": dict(sorted(files.items())),
            "fixture_file_count": len(files),
            "source_total_bytes": source_bytes,
            "fixture_total_bytes": fixture_bytes,
            "reduction_ratio": 1.0 - (fixture_bytes / source_bytes) if source_bytes else 0.0,
            "field_whitelist_version": WHITELIST_VERSION,
            "extraction_rule": "explicit per-artifact whitelist in build_stage1_evidence_fixtures.py",
            "extractor_version": EXTRACTOR_VERSION,
            "generated_at": generated_at,
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-at", required=True, help="Stable ISO-8601 timestamp; part of reproducible input.")
    args = parser.parse_args()
    manifest = build_fixture(
        source_root=args.source_results_root,
        output_dir=args.output_dir,
        generated_at=str(args.generated_at),
    )
    print(json.dumps({"fixture_file_count": manifest["fixture_file_count"], "fixture_total_bytes": manifest["fixture_total_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
