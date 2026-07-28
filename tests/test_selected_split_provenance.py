from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from experiments.stage1_best_training_selected_split_benchmark_cli import (
    SelectedSplitBenchmarkError,
    build_runtime_provenance,
    main as benchmark_main,
)
from qubit_value_function.closed_loop_batch import ClosedLoopBatchExecutor, build_run_specs
from qubit_value_function.selected_split_provenance_audit import (
    ProvenanceAuditError,
    audit_selected_split_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "stage1_best_training_selected_split_benchmark_fixed_20260727_231759"
ACTUAL_HEAD = "6d530a353c523bcf72659fe27d26ceb4fb22504e"
DECLARED_HEAD = "c959bbdc021e1236f82b2ce958b95568bc38b98f"


def test_runtime_provenance_separates_actual_manifest_and_external_heads() -> None:
    provenance = build_runtime_provenance(
        actual_execution_head=ACTUAL_HEAD,
        manifest_declared_code_sha=DECLARED_HEAD,
        externally_expected_head=ACTUAL_HEAD,
        working_tree_tracked_clean=True,
    )

    assert provenance == {
        "actual_execution_head": ACTUAL_HEAD,
        "manifest_declared_code_sha": DECLARED_HEAD,
        "externally_expected_head": ACTUAL_HEAD,
        "head_match": True,
        "working_tree_tracked_clean": True,
    }


def test_runtime_provenance_rejects_external_head_mismatch() -> None:
    with pytest.raises(SelectedSplitBenchmarkError, match="expected_head_mismatch"):
        build_runtime_provenance(
            actual_execution_head=ACTUAL_HEAD,
            manifest_declared_code_sha=DECLARED_HEAD,
            externally_expected_head="different",
            working_tree_tracked_clean=True,
        )


def test_batch_manifest_exposes_runtime_provenance_without_changing_code_head() -> None:
    spec = build_run_specs(
        batch_id="provenance", preset="custom", generator_pairs=((0, 1),), windows=(0,), training_seeds=(1,),
        methods=("full_space_random",), run_seeds=(0,), budget_config={}, fixed_point_config={},
        initialization_policy="best_training", expected_code_sha=ACTUAL_HEAD,
    )[0]
    executor = ClosedLoopBatchExecutor(
        output_dir=ROOT / "ignored", scenario_builder=lambda _spec: None, method_runner=lambda *_args: {}, workers=1,
        code={"head": ACTUAL_HEAD, "actual_execution_head": ACTUAL_HEAD, "manifest_declared_code_sha": DECLARED_HEAD,
              "externally_expected_head": ACTUAL_HEAD, "head_match": True, "working_tree_tracked_clean": True},
    )

    manifest = executor._manifest((spec,), {"completed": 0, "skipped": 0, "failed": 0, "pending": 1}, interrupted=False, started_at="now")

    assert manifest["expected_head"] == ACTUAL_HEAD
    assert manifest["actual_execution_head"] == ACTUAL_HEAD
    assert manifest["manifest_declared_code_sha"] == DECLARED_HEAD
    assert manifest["head_match"] is True


def test_cli_expected_head_mismatch_fails_before_output_creation(monkeypatch) -> None:
    output = ROOT / "results" / "selected_split_expected_head_mismatch_test"
    assert not output.exists()
    monkeypatch.setattr(
        sys,
        "argv",
        ["selected", "--config", str(ROOT / "experiments" / "configs" / "stage1_best_training_selected_split_benchmark.json"),
         "--output-dir", str(output), "--expected-head", "not-the-current-head", "--preflight"],
    )

    with pytest.raises(SelectedSplitBenchmarkError, match="expected_head_mismatch"):
        benchmark_main()

    assert not output.exists()


def test_read_only_provenance_audit_recomputes_complete_result_set(monkeypatch) -> None:
    import qubit_value_function.selected_split_provenance_audit as audit_module

    written: dict[str, object] = {}
    monkeypatch.setattr(audit_module, "atomic_write_json", lambda path, payload: written.setdefault(Path(path).name, payload))

    report = audit_selected_split_provenance(
        result_dir=RESULT_DIR,
        actual_execution_head=ACTUAL_HEAD,
        output_dir=ROOT / "results" / "unused_provenance_audit_test_output",
        audit_code_head=ACTUAL_HEAD,
    )

    assert report["completed_runs"] == 360
    assert report["failed_runs"] == 0
    assert report["per_method_completed_counts"] == {
        "classical_joint_marked_random": 60,
        "cost_only_bbht": 60,
        "direct_logic_feasible_random": 60,
        "full_space_random": 60,
        "joint_bbht": 60,
        "logic_rejection_random": 60,
    }
    assert report["recomputed_summary_matches"] is True
    assert report["original_results_modified"] is False
    assert report["status"] == "corrected_by_sidecar"
    assert set(written) == {"provenance_correction.json", "file_hashes.json", "recomputed_summary.json"}


def test_provenance_audit_rejects_existing_output_directory(monkeypatch) -> None:
    import qubit_value_function.selected_split_provenance_audit as audit_module

    monkeypatch.setattr(audit_module.Path, "exists", lambda _self: True)
    with pytest.raises(ProvenanceAuditError, match="audit_output_directory_exists"):
        audit_selected_split_provenance(
            result_dir=RESULT_DIR,
            actual_execution_head=ACTUAL_HEAD,
            output_dir=ROOT / "results" / "existing",
            audit_code_head=ACTUAL_HEAD,
        )
