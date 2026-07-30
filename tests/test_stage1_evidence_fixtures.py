from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_stage1_evidence_fixtures import _project_selected_run
from tests.stage1_evidence_fixture import stage1_evidence_fixture_root


def test_stage1_evidence_fixture_manifest_is_required() -> None:
    root = stage1_evidence_fixture_root()

    assert (root / "manifest.json").is_file()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fixture_file_count"] == 456
    assert len(manifest["files"]) == 456
    assert len(manifest["source_files"]) == 456
    assert manifest["field_whitelist_version"] == "stage1-evidence-fields-v1"
    assert manifest["source_sha256"]["stage1_closed_loop_baseline_formal/batch_manifest.json"] == (
        "2f225f75fb000a216ea3835dee6b231588bbdcf589010d9f89335c6408ffac03"
    )
    assert manifest["source_sha256"]["stage1_targeted_best_training_candidate_pilot_20260727_221524/summaries/targeted_pilot_summary.json"] == (
        "3e6f8a9b6963ea3a87f17e19cbda1d6fa807a3744694959b8728d7406fef469e"
    )
    assert manifest["source_sha256"]["stage1_best_training_selected_split_benchmark_fixed_20260727_231759/batch_manifest.json"] == (
        "5cd8732eeef15870145fecdd3208e27cecc7071641851b1f9b8ea0e5db1f787b"
    )
    assert manifest["source_sha256"]["stage1_best_training_selected_split_benchmark_fixed_20260727_231759/summaries/selected_split_summary.json"] == (
        "0257f91af4a747e2504881ae6e58bc0d323a726671aecd19de9f73c31c54e1ac"
    )
    for relative_path in manifest["source_sha256"]:
        assert not Path(relative_path).is_absolute()
        assert ":" not in relative_path
    for relative_path, metadata in manifest["files"].items():
        assert not Path(relative_path).is_absolute()
        assert ":" not in relative_path
        content = (root / relative_path).read_bytes()
        assert len(content) == metadata["bytes"]
        assert hashlib.sha256(content).hexdigest() == metadata["sha256"]


def test_selected_run_projection_preserves_missing_training_membership_as_false() -> None:
    payload = {
        "method": "full_space_random",
        "run_spec": {"run_seed": 0},
        "scenario": {"scenario_id": "case14-g0g1-w0-s1"},
        "result": {
            "initial_incumbent_true_cost": 10.0,
            "final_incumbent_true_cost": 9.0,
            "oracle_reachability_stratum": "reachable",
            "stop_reason": "complete",
            "counters": {},
            "trial_trace": [{"true_strict_improvement": True}],
        },
    }

    projected = _project_selected_run(payload)

    assert projected["result"]["trial_trace"] == [{"true_strict_improvement": True}]
