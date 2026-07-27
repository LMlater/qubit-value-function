from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.stage1_best_training_candidate_discovery_cli import (
    CandidateDiscoveryError,
    candidate_summary_row,
    candidate_discovery_run_settings,
    ensure_new_output_dir,
)


def test_candidate_discovery_refuses_existing_output_dir() -> None:
    class ExistingPath:
        def exists(self) -> bool:
            return True

        def __str__(self) -> str:
            return "existing"

    existing = ExistingPath()
    with pytest.raises(CandidateDiscoveryError, match="refusing_existing_output_directory"):
        ensure_new_output_dir(existing)


def test_candidate_run_settings_keep_model_seed_separate_from_training_data_seed() -> None:
    settings = candidate_discovery_run_settings(
        {"training_index_policy": "seeded_without_replacement_v1", "model_seed": 0},
        {
            "generator_pair": [0, 5],
            "window_start": 1,
            "training_data_seed": 2,
            "frozen_seed0_training_indices": [0, 3, 5, 6, 9, 10, 12, 15],
        },
    )

    assert settings == {
        "base_scenario_id": "case14-g0g5-w1",
        "training_data_seed": 2,
        "training_index_policy": "seeded_without_replacement_v1",
        "model_seed": 0,
        "frozen_seed0_training_indices": (0, 3, 5, 6, 9, 10, 12, 15),
    }


def test_seed_manifests_define_twelve_distinct_scenarios_and_a_fixed_model_seed() -> None:
    root = Path(__file__).resolve().parents[1]
    for filename, expected_seed in (
        ("stage1_best_training_candidate_discovery.json", 0),
        ("stage1_best_training_candidate_discovery_seed1.json", 1),
        ("stage1_best_training_candidate_discovery_seed2.json", 2),
    ):
        config = json.loads((root / "experiments" / "configs" / filename).read_text(encoding="utf-8"))
        settings = [candidate_discovery_run_settings(config, item) for item in config["scenarios"]]
        assert len(settings) == 12
        assert len({setting["base_scenario_id"] for setting in settings}) == 12
        assert {setting["training_data_seed"] for setting in settings} == {expected_seed}
        assert {setting["model_seed"] for setting in settings} == {0}


def test_candidate_summary_records_the_actual_split_and_hash() -> None:
    scenario = SimpleNamespace(
        scenario_id="case14-g0g5-w1-s2",
        generator_pair=(0, 5),
        window_start=1,
        training_indices=(8, 7, 6, 5, 4, 3, 2, 1),
        reproducibility_metadata={"training_actual_ed_lp_solves": 8},
    )
    snapshot = {
        "training_data_seed": 2,
        "training_index_policy": "seeded_without_replacement_v1",
        "model_seed": 0,
        "initial_cost_marked_count": 1,
        "initial_joint_marked_count": 1,
        "initial_training_joint_marked_count": 0,
        "initial_nontraining_cost_marked_count": 1,
        "initial_nontraining_joint_marked_count": 1,
        "initial_nontraining_joint_marked_indices": [9],
        "has_nontraining_joint_marked_candidate": True,
    }

    row = candidate_summary_row(
        scenario=scenario, snapshot=snapshot, base_scenario_id="case14-g0g5-w1"
    )

    assert row["actual_training_indices"] == [8, 7, 6, 5, 4, 3, 2, 1]
    assert row["training_indices_hash"] == "fbac8caf13e220231ebf0a08ca627a140c8dfd235f53c75482863ba8308ebc2e"
    assert row["training_data_seed"] == 2
    assert row["model_seed"] == 0
