from __future__ import annotations

import math

import pytest

from qubit_value_function.fixed_oracle_multishot_diagnostic import (
    FixedOracleDiagnosticError,
    ideal_grover_probability,
    strict_joint_marked_indices,
    validate_training_truth_metadata,
    wilson_interval,
)


def test_ideal_grover_curve_covers_uniform_amplification_and_overrotation() -> None:
    assert ideal_grover_probability(3, 0) == pytest.approx(3 / 16)
    assert ideal_grover_probability(3, 1) == pytest.approx(0.94921875)
    assert ideal_grover_probability(3, 3) < 1e-3


def test_strict_joint_marking_excludes_equal_threshold_and_infeasible_states() -> None:
    assert strict_joint_marked_indices([4, 5, 3, 1], [True, True, False, True], 5) == (0, 3)


def test_wilson_interval_is_valid_and_contains_observation() -> None:
    low, high = wilson_interval(43, 45)
    assert 0 <= low <= 43 / 45 <= high <= 1


def test_training_truth_metadata_rejects_online_global_truth_and_holdout_without_label() -> None:
    with pytest.raises(FixedOracleDiagnosticError):
        validate_training_truth_metadata({"training_selection_protocol": "random_truth_blind", "global_truth_used_online": True})
    with pytest.raises(FixedOracleDiagnosticError):
        validate_training_truth_metadata({"training_selection_protocol": "controlled_optimum_holdout", "global_truth_used_online": False})
    valid = validate_training_truth_metadata({"training_selection_protocol": "controlled_optimum_holdout", "controlled_optimum_holdout": True, "global_truth_used_online": False})
    assert valid["global_truth_used_for_posthoc_validation"] is True
