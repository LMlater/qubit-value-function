from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.stage1_targeted_best_training_pilot_cli import (
    PilotPreflightError,
    augment_targeted_result_schema,
    build_argument_parser,
    build_targeted_run_specs,
    ensure_new_output_dir,
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


def test_fixed_candidate_plan_has_four_scenarios_and_eighty_joint_bbht_runs() -> None:
    manifest = {
        "experiment_name": "targeted", "methods": ["joint_bbht"],
        "run_seeds": list(range(20)), "initial_incumbent_policy": "best_training",
        "bbht_config": {
            "lambda_factor": 1.2, "max_trials": 64, "max_oracle_calls": 128,
            "max_new_ed_lp_calls": 16, "max_threshold_updates": 8,
            "max_consecutive_nonimproving_marked": 16,
            "max_same_encoded_threshold_updates": 3,
            "max_auxiliary_syndrome_rejections": 8,
        },
        "scenarios": [
            {"scenario_id": "case14-g0g1-w2-s1", "generator_pair": [0, 1], "window_start": 2, "training_data_seed": 1, "role": "nontraining_improvement_candidate"},
            {"scenario_id": "case14-g0g5-w0-s1", "generator_pair": [0, 5], "window_start": 0, "training_data_seed": 1, "role": "nontraining_improvement_candidate"},
            {"scenario_id": "case14-g0g5-w2-s1", "generator_pair": [0, 5], "window_start": 2, "training_data_seed": 1, "role": "nontraining_improvement_candidate"},
            {"scenario_id": "case14-g1g5-w0-s1", "generator_pair": [1, 5], "window_start": 0, "training_data_seed": 1, "role": "nontraining_improvement_candidate"},
        ],
    }

    specs = build_targeted_run_specs(manifest, expected_head="head")

    assert len(specs) == 80
    assert {spec.method for spec in specs} == {"joint_bbht"}
    assert {spec.run_seed for spec in specs} == set(range(20))
    assert {spec.scenario_id for spec in specs} == {
        "case14-g0g1-w2-s1", "case14-g0g5-w0-s1", "case14-g0g5-w2-s1", "case14-g1g5-w0-s1",
    }


def test_trial_augmentation_records_training_and_nontraining_first_events() -> None:
    result = {
        "trial_trace": [
            {"trial_number": 1, "measured_index": 2, "candidate_in_training_set": True, "measured_joint_marked": True, "cache_hit": True, "new_ed_lp_solve": False, "true_strict_improvement": False},
            {"trial_number": 2, "measured_index": 3, "candidate_in_training_set": False, "measured_joint_marked": True, "cache_hit": False, "new_ed_lp_solve": True, "true_strict_improvement": True},
        ],
    }

    augmented = augment_targeted_result_schema(
        result, training_indices=(2,), initial_nontraining_joint_marked_indices=(3,)
    )

    assert augmented["trial_trace"][0]["sampled_initial_nontraining_joint_marked_candidate"] is False
    assert augmented["trial_trace"][1]["sampled_initial_nontraining_joint_marked_candidate"] is True
    assert augmented["targeted_pilot_events"] == {
        "first_outside_training_evaluation_trial": 2,
        "first_outside_training_strict_improvement_trial": 2,
    }


def test_new_output_directory_is_required() -> None:
    class ExistingPath:
        def exists(self) -> bool:
            return True

        def __str__(self) -> str:
            return "existing"

    with pytest.raises(PilotPreflightError, match="refusing_existing_output_directory"):
        ensure_new_output_dir(ExistingPath())


def test_fixed_candidate_manifest_is_four_snapshot_groups_times_twenty_seeds() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "experiments" / "configs" / "stage1_targeted_best_training_candidate_pilot.json").read_text(encoding="utf-8")
    )

    specs = build_targeted_run_specs(manifest, expected_head="head")

    assert manifest["fixed_candidate_manifest"] is True
    assert manifest["initial_incumbent_policy"] == "best_training"
    assert len(manifest["scenarios"]) == 4
    assert len(specs) == 80
    assert all(spec.training_seed == 1 and spec.method == "joint_bbht" for spec in specs)


@pytest.mark.parametrize(
    "method",
    (
        "full_space_random",
        "logic_rejection_random",
        "direct_logic_feasible_random",
        "classical_joint_marked_random",
    ),
)
def test_trial_augmentation_normalizes_classical_candidate_index_schema(method: str) -> None:
    augmented = augment_targeted_result_schema(
        {
            "trial_trace": [{
                "proposal_number": 1,
                "candidate_source_method": method,
                "candidate_index": 3,
                "candidate_cache_hit": False,
                "new_edlp_solve_performed": True,
                "true_improvement": True,
            }]
        },
        training_indices=(2,),
        initial_nontraining_joint_marked_indices=(3,),
    )

    row = augmented["trial_trace"][0]
    assert row["normalized_candidate_index"] == 3
    assert row["sampled_in_training_set"] is False
    assert row["sampled_initial_nontraining_joint_marked_candidate"] is True
    assert row["cache_hit"] is False
    assert row["new_ed_lp_solve"] is True
    assert row["true_strict_improvement"] is True
    assert augmented["targeted_pilot_events"] == {
        "first_outside_training_evaluation_trial": 1,
        "first_outside_training_strict_improvement_trial": 1,
    }


def test_trial_augmentation_preserves_bbht_measured_index_schema() -> None:
    augmented = augment_targeted_result_schema(
        {"trial_trace": [{"trial_number": 1, "measured_index": 2, "cache_hit": True, "new_ed_lp_solve": False, "true_strict_improvement": False}]},
        training_indices=(2,),
        initial_nontraining_joint_marked_indices=(3,),
    )

    row = augmented["trial_trace"][0]
    assert row["measured_index"] == 2
    assert "candidate_index" not in row
    assert row["normalized_candidate_index"] == 2
    assert row["sampled_in_training_set"] is True


def test_trial_augmentation_accepts_trial_without_candidate_state() -> None:
    augmented = augment_targeted_result_schema(
        {"trial_trace": [{"trial_number": 1, "candidate_status": "no_joint_marked_state"}]},
        training_indices=(2,),
        initial_nontraining_joint_marked_indices=(3,),
    )

    row = augmented["trial_trace"][0]
    assert row["normalized_candidate_index"] is None
    assert row["sampled_in_training_set"] is None
    assert row["sampled_initial_nontraining_joint_marked_candidate"] is False
    assert augmented["targeted_pilot_events"] == {
        "first_outside_training_evaluation_trial": None,
        "first_outside_training_strict_improvement_trial": None,
    }


def test_trial_augmentation_preserves_empty_classical_joint_marked_trace() -> None:
    augmented = augment_targeted_result_schema(
        {"method": "classical_joint_marked_random", "trial_trace": []},
        training_indices=(2,),
        initial_nontraining_joint_marked_indices=(),
    )

    assert augmented["trial_trace"] == []
    assert augmented["targeted_pilot_events"] == {
        "first_outside_training_evaluation_trial": None,
        "first_outside_training_strict_improvement_trial": None,
    }
