from __future__ import annotations

from qubit_value_function.fixed_oracle_multishot_diagnostic import validate_training_truth_metadata


def test_random_truth_blind_metadata_is_explicit_and_posthoc_only() -> None:
    metadata = validate_training_truth_metadata({"training_selection_protocol": "random_truth_blind", "global_truth_used_online": False})
    assert metadata == {
        "training_selection_protocol": "random_truth_blind",
        "controlled_optimum_holdout": False,
        "global_truth_used_online": False,
        "global_truth_used_for_posthoc_validation": True,
    }
