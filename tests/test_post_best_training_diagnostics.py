from __future__ import annotations

import pytest

from qubit_value_function.post_best_training_diagnostics import (
    classify_trial,
    ensure_json_finite,
    grover_marked_probability,
    quantization_diagnostic,
    trial_row,
)


def _snapshot() -> dict[str, object]:
    return {
        "scenario_id": "fixture", "cost_unit": 1.0, "quantization_mode": "nearest",
        "fractional_bits": 2, "real_intercept": 1.0, "real_weights": [0.5],
        "state_proxy_table": [
            {"state_index": 0, "feature_vector": [0], "integer_vqc_value": 4, "real_vqc_prediction": 1.0, "hard_logic_feasible": True},
            {"state_index": 1, "feature_vector": [1], "integer_vqc_value": 6, "real_vqc_prediction": 1.5, "hard_logic_feasible": True},
        ],
    }


def test_grover_probability_matches_uniform_at_k_zero_and_handles_empty() -> None:
    zero = grover_marked_probability(marked_count=0, search_space_size=16, iterations=0)
    assert zero["grover_theoretical_marked_probability"] is None
    assert zero["marked_probability_gain"] is None
    result = grover_marked_probability(marked_count=3, search_space_size=16, iterations=0)
    assert result["grover_theoretical_marked_probability"] == pytest.approx(3 / 16)
    assert result["marked_probability_gain"] == pytest.approx(1.0)


def test_trial_row_checks_marked_membership_and_classifies_exclusively() -> None:
    row = {
        "joint_marked_indices_before": [2], "joint_marked_count_before": 1,
        "measured_index": 2, "measured_joint_marked": True,
        "sampled_grover_iterations": 1, "cache_hit": True,
        "cache_source": "training_cache", "trial_number": 1,
    }
    payload = trial_row(scenario_id="s", method="joint_bbht", policy="best_training", run_seed=0, row=row, search_space_size=16)
    assert payload["classification"] == "marked_training_cache"
    assert payload["measured_matches_theoretical_marked_set"] is True
    assert classify_trial({"measured_joint_marked": False}) == "unmarked_measurement"
    row["measured_joint_marked"] = False
    with pytest.raises(AssertionError):
        trial_row(scenario_id="s", method="joint_bbht", policy="best_training", run_seed=0, row=row, search_space_size=16)


def test_quantization_reconstruction_uses_strict_comparator_and_is_finite() -> None:
    result = quantization_diagnostic(_snapshot(), true_threshold=1.5, fractional_bits=2)
    assert result["encoded_threshold"] == 6
    assert result["joint_marked_indices"] == [0]
    assert result["real_joint_marked_count"] == 1
    ensure_json_finite(result)
    with pytest.raises(ValueError):
        ensure_json_finite({"bad": float("nan")})
