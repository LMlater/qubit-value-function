from __future__ import annotations

from qubit_value_function.closed_loop_metadata import (
    DynamicOracleMetadataRecorder,
    build_quantized_model_snapshot,
)
from qubit_value_function.coherent_phase_value import (
    QuantizedSparseValueModel,
    conservative_integer_bounds,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig
from qubit_value_function.sparse_phase_vqc import build_local_phase_features


def _model() -> QuantizedSparseValueModel:
    features = build_local_phase_features(2, 2)
    weights = (1, 2, 4, 8) + (0,) * (len(features) - 4)
    lower, upper = conservative_integer_bounds(0, weights)
    return QuantizedSparseValueModel(
        num_generators=2,
        num_periods=2,
        features=features,
        fixed_point_config=FixedPointConfig(fractional_bits=0, unit=1.0),
        real_intercept=0.0,
        real_weights=tuple(float(value) for value in weights),
        integer_intercept=0,
        integer_weights=weights,
        coefficient_quantization_errors=(0.0,) * (1 + len(weights)),
        lower_bound=lower,
        upper_bound=upper,
        value_shift=-lower,
        shifted_upper_bound=upper - lower,
        num_value_qubits=4,
    )


def test_snapshot_persists_complete_16_state_quantized_model_without_truth() -> None:
    model = _model()
    snapshot = build_quantized_model_snapshot(
        scenario_id="case14-g0g5-w2-s0",
        generator_pair=(0, 5),
        window_start=2,
        training_seed=0,
        model=model,
        hard_logic_is_feasible=lambda bits: bits[0] == 0 or bits[1] == 1,
        training_indices=(1, 3),
        training_labels=((1, 10.0), (3, 8.0)),
        initial_incumbent_index=3,
        initial_incumbent_true_cost=8.0,
        initial_cache_indices=(1, 3),
    )

    rows = snapshot["state_proxy_table"]
    assert snapshot["quantized_model_snapshot_version"] == "v1"
    assert snapshot["search_space_size"] == 16
    assert len(rows) == 16
    assert [row["integer_vqc_value"] for row in rows] == list(range(16))
    assert rows[3]["in_training_set"] is True
    assert rows[2]["initially_in_cache"] is False
    assert all("true_cost" not in row for row in rows)


def test_trial_metadata_uses_strict_marked_sets_and_encoded_threshold_stages() -> None:
    recorder = DynamicOracleMetadataRecorder(
        _model(),
        hard_logic_is_feasible=lambda bits: bits[0] == 0,
        initial_true_threshold=3.4,
        initial_encoded_threshold=3,
        initial_cache_indices=(3,),
    )

    before = recorder.before_trial(threshold_updates_before_trial=0)
    assert before["cost_marked_indices_before"] == [0, 1, 2]
    assert before["joint_marked_indices_before"] == [0, 2]
    assert before["cost_marked_count_before"] == 3
    assert before["joint_marked_count_before"] == 2
    assert before["threshold_stage_id"] == 0

    same_encoded = recorder.after_trial(
        true_threshold_after_trial=3.1,
        encoded_threshold_after_trial=3,
        true_strict_improvement=True,
        encoded_threshold_changed=False,
        stop_reason_after_trial=None,
    )
    assert same_encoded["threshold_stage_id_after_trial"] == 0
    assert same_encoded["true_threshold_update_without_encoded_change"] is True

    recorder.before_trial(threshold_updates_before_trial=1)
    changed = recorder.after_trial(
        true_threshold_after_trial=2.0,
        encoded_threshold_after_trial=2,
        true_strict_improvement=True,
        encoded_threshold_changed=True,
        stop_reason_after_trial="max_threshold_updates_reached",
    )
    assert changed["threshold_stage_id_after_trial"] == 1
    assert changed["stop_reason_after_trial"] == "max_threshold_updates_reached"


def test_cache_source_is_derived_from_existing_cache_hit_without_a_cache_lookup() -> None:
    recorder = DynamicOracleMetadataRecorder(
        _model(), hard_logic_is_feasible=lambda bits: True,
        initial_true_threshold=3.0, initial_encoded_threshold=3,
        initial_cache_indices=(3,),
    )
    before = recorder.before_trial(threshold_updates_before_trial=0)
    common = {
        "before": before, "auxiliary_accepted": True, "candidate_admitted": True,
        "new_ed_lp_solve": False, "exact_evaluation_success": True,
        "candidate_true_cost": 1.0, "true_strict_improvement": True,
        "incumbent_updated": True,
    }
    assert recorder.measured_metadata(measured_index=3, cache_hit=True, **common)["cache_source"] == "training_cache"
    assert recorder.measured_metadata(measured_index=1, cache_hit=True, **common)["cache_source"] == "search_cache"
    assert recorder.measured_metadata(measured_index=1, cache_hit=False, **common)["cache_source"] == "none"
