"""Read-only provenance correction sidecar for selected-split benchmark results."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping

from qubit_value_function.closed_loop_batch import METHODS, atomic_write_json


class ProvenanceAuditError(RuntimeError):
    """The immutable benchmark result set is incomplete or internally inconsistent."""


def _load(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceAuditError(f"audit_json_unavailable:{path}") from error
    if not isinstance(payload, dict):
        raise ProvenanceAuditError(f"audit_json_not_object:{path}")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_hashes(root: Path) -> tuple[dict[str, str], str]:
    hashes = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    canonical = json.dumps(hashes, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashes, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_selected_split_provenance(
    *,
    result_dir: Path,
    actual_execution_head: str,
    output_dir: Path,
    audit_code_head: str,
) -> dict[str, object]:
    """Verify persisted JSON and write a correction sidecar without touching ``result_dir``."""

    result_dir, output_dir = Path(result_dir), Path(output_dir)
    if output_dir.exists():
        raise ProvenanceAuditError("audit_output_directory_exists")
    batch_manifest = _load(result_dir / "batch_manifest.json")
    original_summary = _load(result_dir / "summaries" / "selected_split_summary.json")
    completed_paths = sorted((result_dir / "runs" / "completed").glob("*.json"))
    failed_paths = sorted((result_dir / "runs" / "failed").glob("*.json"))
    counts = dict(batch_manifest.get("counts", {}))
    if int(counts.get("completed", -1)) != 360 or int(counts.get("failed", -1)) != 0:
        raise ProvenanceAuditError("batch_manifest_counts_not_360_completed_zero_failed")
    if len(completed_paths) != 360 or failed_paths:
        raise ProvenanceAuditError("completed_or_failed_file_count_mismatch")
    methods: Counter[str] = Counter()
    combinations: set[tuple[str, str, int]] = set()
    for path in completed_paths:
        payload = _load(path)
        spec = dict(payload.get("run_spec", {}))
        method = str(payload.get("method", spec.get("method", "")))
        scenario_id = str(dict(payload.get("scenario", {})).get("scenario_id", spec.get("scenario_id", "")))
        run_seed = int(spec["run_seed"])
        key = (scenario_id, method, run_seed)
        if key in combinations:
            raise ProvenanceAuditError("duplicate_scenario_method_seed_combination")
        combinations.add(key)
        methods[method] += 1
    if set(methods) != set(METHODS) or any(methods[method] != 60 for method in METHODS):
        raise ProvenanceAuditError("per_method_completed_counts_not_sixty")
    if len(combinations) != 360:
        raise ProvenanceAuditError("scenario_method_seed_combinations_not_360_unique")
    from experiments.stage1_best_training_selected_split_benchmark_cli import compute_selected_split_summary

    recomputed_summary = compute_selected_split_summary(result_dir, planned_runs=360)
    if recomputed_summary != original_summary:
        raise ProvenanceAuditError("recomputed_summary_mismatch")
    file_hashes, target_hash = _file_hashes(result_dir)
    environment = dict(batch_manifest.get("environment", {}))
    originally_recorded_head = str(environment.get("head", batch_manifest.get("expected_head", "")))
    correction = {
        "target_result_directory": str(result_dir),
        "target_result_directory_hash": target_hash,
        "actual_execution_head": str(actual_execution_head),
        "originally_recorded_head": originally_recorded_head,
        "manifest_declared_code_sha": str(batch_manifest.get("expected_head", "")),
        "mismatch_reason": "provenance_metadata_mismatch" if originally_recorded_head != actual_execution_head else None,
        "completed_runs": len(completed_paths),
        "failed_runs": len(failed_paths),
        "per_method_completed_counts": dict(sorted(methods.items())),
        "recomputed_summary_matches": True,
        "numerical_results_modified": False,
        "original_results_modified": False,
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "audit_code_head": str(audit_code_head),
        "status": "corrected_by_sidecar",
    }
    hashes_payload = {
        "target_result_directory": str(result_dir),
        "target_result_directory_hash": target_hash,
        "batch_manifest_sha256": file_hashes["batch_manifest.json"],
        "summary_sha256": file_hashes["summaries/selected_split_summary.json"],
        "completed_run_sha256": {path.relative_to(result_dir).as_posix(): file_hashes[path.relative_to(result_dir).as_posix()] for path in completed_paths},
        "all_file_sha256": file_hashes,
    }
    atomic_write_json(output_dir / "provenance_correction.json", correction)
    atomic_write_json(output_dir / "file_hashes.json", hashes_payload)
    atomic_write_json(output_dir / "recomputed_summary.json", recomputed_summary)
    return correction
