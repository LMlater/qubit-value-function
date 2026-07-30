from __future__ import annotations

import json
from pathlib import Path

import pytest

import qubit_value_function.selected_split_best_training_audit as audit_module

from experiments.stage1_best_training_selected_split_benchmark_cli import (
    SelectedSplitBenchmarkError,
    build_selected_split_run_specs,
    preflight_selected_split_manifest,
)
from qubit_value_function.selected_split_best_training_audit import (
    SelectedSplitAuditError,
    audit_targeted_best_training_candidate_pilot,
)
from tests.stage1_evidence_fixture import (
    load_selected_split_fixture_manifest,
    stage1_evidence_fixture_root,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = stage1_evidence_fixture_root()
TARGETED_PILOT_DIR = FIXTURE_ROOT / "stage1_targeted_best_training_candidate_pilot_20260727_221524"


def test_selected_split_manifest_reuses_twelve_seed_one_snapshots_and_formal_methods() -> None:
    manifest = load_selected_split_fixture_manifest()

    report = preflight_selected_split_manifest(manifest)
    specs = build_selected_split_run_specs(manifest, expected_head="head")

    assert len(report["scenarios"]) == 12
    assert {row["training_data_seed"] for row in report["scenarios"]} == {1}
    assert {row["model_seed"] for row in report["scenarios"]} == {0}
    assert all(row["training_sample_count"] == 8 for row in report["scenarios"])
    assert report["oracle_reachability_counts"] == {"reachable": 4, "unreachable": 8}
    assert tuple(manifest["methods"]) == (
        "joint_bbht", "cost_only_bbht", "full_space_random", "logic_rejection_random",
        "direct_logic_feasible_random", "classical_joint_marked_random",
    )
    assert len(specs) == 360
    assert {spec.run_seed for spec in specs} == set(range(5))
    assert {spec.initialization_policy for spec in specs} == {"best_training"}


def test_selected_split_rejects_changed_formal_method_list() -> None:
    manifest = load_selected_split_fixture_manifest()
    manifest["methods"] = manifest["methods"][:-1]

    with pytest.raises(SelectedSplitBenchmarkError, match="formal_method_list_mismatch"):
        build_selected_split_run_specs(manifest, expected_head="head")


def test_read_only_targeted_pilot_audit_accepts_existing_eighty_run_case_study() -> None:
    report = audit_targeted_best_training_candidate_pilot(
        TARGETED_PILOT_DIR
    )

    assert report["completed_runs"] == 80
    assert report["runs_per_scenario"] == 20
    assert report["all_runs_have_outside_training_strict_improvement"] is True
    assert report["all_runs_have_one_search_stage_new_edlp"] is True


def test_read_only_targeted_pilot_audit_detects_summary_inconsistency(monkeypatch) -> None:
    real_load = audit_module._load

    def inconsistent_summary(path: Path) -> dict[str, object]:
        payload = real_load(path)
        if path.name == "targeted_pilot_summary.json":
            return {**payload, "completed_runs": 79}
        return payload

    monkeypatch.setattr(audit_module, "_load", inconsistent_summary)
    with pytest.raises(SelectedSplitAuditError, match="pilot_summary_completed_runs_mismatch"):
        audit_targeted_best_training_candidate_pilot(
            TARGETED_PILOT_DIR
        )
