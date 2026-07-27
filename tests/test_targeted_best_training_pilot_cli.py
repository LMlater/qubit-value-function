from __future__ import annotations

import json

import pytest

from experiments.stage1_targeted_best_training_pilot_cli import (
    PilotPreflightError,
    build_argument_parser,
    preflight_manifest,
)


def _snapshot(*, nontraining_joint: bool) -> dict[str, object]:
    return {
        "scenario_id": "candidate", "search_space_size": 4,
        "training_indices": [0],
        "training_labels": [{"state_index": 0, "true_cost": 1.0}],
        "cost_unit": 1.0, "encoded_cost_scale": 1,
        "state_proxy_table": [
            {"state_index": 0, "integer_vqc_value": 0, "hard_logic_feasible": True},
            {"state_index": 1, "integer_vqc_value": -1 if nontraining_joint else 2, "hard_logic_feasible": True},
            {"state_index": 2, "integer_vqc_value": 2, "hard_logic_feasible": True},
            {"state_index": 3, "integer_vqc_value": 2, "hard_logic_feasible": False},
        ],
    }


def test_cli_defaults_to_twenty_seeds_and_requires_output_dir() -> None:
    parser = build_argument_parser()
    args = parser.parse_args(["--config", "pilot.json", "--output-dir", "fresh"])
    assert args.run_seeds is None
    assert args.seed_range is None
    assert args.output_dir.name == "fresh"


def test_preflight_requires_a_nontraining_joint_marked_candidate(monkeypatch) -> None:
    monkeypatch.setattr(
        "experiments.stage1_targeted_best_training_pilot_cli._load_snapshot",
        lambda _path: _snapshot(nontraining_joint=False),
    )
    manifest = {
        "scenarios": [
            {"scenario_id": "calibration", "role": "grover_calibration_training_marked", "snapshot": "control.json"},
            {"scenario_id": "empty", "role": "joint_marked_empty_control", "snapshot": "control.json"},
            {"scenario_id": "candidate", "role": "nontraining_improvement_candidate", "snapshot": "candidate.json"},
        ]
    }
    with pytest.raises(PilotPreflightError, match="no_nontraining_joint_marked_candidate"):
        preflight_manifest(manifest)


def test_preflight_persists_roles_and_artifact_derived_counts(monkeypatch) -> None:
    monkeypatch.setattr(
        "experiments.stage1_targeted_best_training_pilot_cli._load_snapshot",
        lambda path: _snapshot(nontraining_joint=path.name == "candidate.json"),
    )
    manifest = {
        "scenarios": [
            {"scenario_id": "calibration", "role": "grover_calibration_training_marked", "snapshot": "calibration.json"},
            {"scenario_id": "empty", "role": "joint_marked_empty_control", "snapshot": "control.json"},
            {"scenario_id": "candidate", "role": "nontraining_improvement_candidate", "snapshot": "candidate.json"},
        ]
    }
    report = preflight_manifest(manifest)
    candidate_row = next(row for row in report["scenarios"] if row["role"] == "nontraining_improvement_candidate")
    assert candidate_row["initial_nontraining_joint_marked_count"] == 1
    assert candidate_row["has_initial_nontraining_joint_marked"] is True
    assert report["selection_status"] == "eligible"
