"""Read-only audit metadata for the Stage A closed-loop methods.

The helpers in this module must never be used to make an online decision.  They
only re-evaluate the already frozen quantized surrogate and hard-logic predicate
for the 2x2 (16-state) Stage A search space, and consume no random numbers.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Mapping, Sequence

import numpy as np

from .coherent_phase_value import QuantizedSparseValueModel


QUANTIZED_MODEL_SNAPSHOT_VERSION = "v1"


def bits_from_index(index: int, num_x_qubits: int) -> tuple[int, ...]:
    return tuple((int(index) >> offset) & 1 for offset in range(int(num_x_qubits)))


def bitstring_from_index(index: int, num_x_qubits: int) -> str:
    return "".join(str(bit) for bit in bits_from_index(index, num_x_qubits))


def canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_quantized_model_snapshot(
    *,
    scenario_id: str,
    generator_pair: tuple[int, int],
    window_start: int,
    training_seed: int,
    model: QuantizedSparseValueModel,
    hard_logic_is_feasible: Callable[[Sequence[int]], bool],
    training_indices: Sequence[int],
    training_labels: Sequence[tuple[int, float]],
    initial_incumbent_index: int,
    initial_incumbent_true_cost: float,
    initial_cache_indices: Sequence[int],
) -> dict[str, object]:
    """Serialize only the frozen surrogate and original training/cache facts."""

    dimension = 2 ** int(model.num_x_qubits)
    if dimension != 16:
        raise ValueError("Stage A dynamic metadata requires exactly 16 search states")
    training = frozenset(int(index) for index in training_indices)
    initial_cache = frozenset(int(index) for index in initial_cache_indices)
    model_dict = model.as_dict()
    rows: list[dict[str, object]] = []
    for index in range(dimension):
        bits = bits_from_index(index, model.num_x_qubits)
        feature_vector = [int(value) for value in model.feature_values(bits)]
        real_prediction = float(
            model.real_intercept
            + np.dot(np.asarray(model.real_weights, dtype=float), np.asarray(feature_vector, dtype=float))
        )
        rows.append(
            {
                "state_index": int(index),
                "bitstring": bitstring_from_index(index, model.num_x_qubits),
                "feature_vector": feature_vector,
                "real_vqc_prediction": real_prediction,
                "integer_vqc_value": int(model.integer_value(bits)),
                "hard_logic_feasible": bool(hard_logic_is_feasible(bits)),
                "in_training_set": bool(index in training),
                "initially_in_cache": bool(index in initial_cache),
            }
        )
    fixed = model_dict["fixed_point"]
    return {
        "quantized_model_snapshot_version": QUANTIZED_MODEL_SNAPSHOT_VERSION,
        "scenario_id": str(scenario_id),
        "generator_pair": [int(value) for value in generator_pair],
        "window_start": int(window_start),
        "training_seed": int(training_seed),
        "num_search_qubits": int(model.num_x_qubits),
        "search_space_size": int(dimension),
        "training_indices": [int(index) for index in training_indices],
        "training_bitstrings": [bitstring_from_index(index, model.num_x_qubits) for index in training_indices],
        "training_labels": [
            {"state_index": int(index), "true_cost": float(cost)}
            for index, cost in training_labels
        ],
        "training_selection_protocol": "persisted_scenario_training_indices",
        "initial_incumbent_index": int(initial_incumbent_index),
        "initial_incumbent_true_cost": float(initial_incumbent_true_cost),
        "initial_cache_indices": sorted(int(index) for index in initial_cache),
        "feature_schema_version": "sparse_phase_vqc_features_v1",
        "feature_names": [str(item["label"]) for item in model_dict["features"]],
        "feature_definition": model_dict["features"],
        "real_intercept": float(model.real_intercept),
        "real_weights": [float(value) for value in model.real_weights],
        "integer_intercept": int(model.integer_intercept),
        "integer_weights": [int(value) for value in model.integer_weights],
        "cost_unit": float(fixed["cost_unit"]),
        "fractional_bits": int(fixed["fractional_bits"]),
        "quantization_mode": str(fixed["rounding"]),
        "encoded_cost_scale": int(fixed["scale"]),
        "integer_lower_bound": int(model.lower_bound),
        "integer_upper_bound": int(model.upper_bound),
        "state_proxy_table": rows,
    }


class DynamicOracleMetadataRecorder:
    """Side-channel recorder for the state before/after each quantum trial."""

    def __init__(
        self,
        model: QuantizedSparseValueModel,
        *,
        hard_logic_is_feasible: Callable[[Sequence[int]], bool],
        initial_true_threshold: float,
        initial_encoded_threshold: int,
        initial_cache_indices: Sequence[int],
        initial_incumbent_policy: str = "first_training",
    ) -> None:
        self._model = model
        self._hard_logic_is_feasible = hard_logic_is_feasible
        self._true_threshold = float(initial_true_threshold)
        self._encoded_threshold = int(initial_encoded_threshold)
        self._initial_cache_indices = frozenset(int(index) for index in initial_cache_indices)
        self._initial_incumbent_policy = str(initial_incumbent_policy)
        self._stage_id = 0
        self._trial_in_stage = 0
        self._trials_since_encoded_threshold_change = 0
        self._states = tuple(
            (
                int(index),
                bits_from_index(index, model.num_x_qubits),
                int(model.integer_value(bits_from_index(index, model.num_x_qubits))),
                bool(hard_logic_is_feasible(bits_from_index(index, model.num_x_qubits))),
            )
            for index in range(2 ** int(model.num_x_qubits))
        )

    def before_trial(self, *, threshold_updates_before_trial: int) -> dict[str, object]:
        cost_marked = [index for index, _bits, value, _feasible in self._states if value < self._encoded_threshold]
        joint_marked = [index for index, _bits, value, feasible in self._states if value < self._encoded_threshold and feasible]
        if joint_marked != [index for index in cost_marked if self._states[index][3]]:
            raise AssertionError("joint marked set must be the feasible intersection of cost marked")
        self._trial_in_stage += 1
        self._trials_since_encoded_threshold_change += 1
        return {
            "threshold_stage_id": int(self._stage_id),
            "trial_index_within_threshold_stage": int(self._trial_in_stage),
            "threshold_updates_before_trial": int(threshold_updates_before_trial),
            "trials_since_last_threshold_update": int(self._trials_since_encoded_threshold_change),
            "true_threshold_before_trial": float(self._true_threshold),
            "encoded_threshold_before_trial": int(self._encoded_threshold),
            "cost_marked_indices_before": cost_marked,
            "cost_marked_count_before": int(len(cost_marked)),
            "joint_marked_indices_before": joint_marked,
            "joint_marked_count_before": int(len(joint_marked)),
        }

    def after_trial(
        self,
        *,
        true_threshold_after_trial: float,
        encoded_threshold_after_trial: int,
        true_strict_improvement: bool,
        encoded_threshold_changed: bool,
        stop_reason_after_trial: str | None,
    ) -> dict[str, object]:
        encoded_after = int(encoded_threshold_after_trial)
        changed = bool(encoded_threshold_changed)
        if changed != (encoded_after != self._encoded_threshold):
            raise AssertionError("encoded threshold change metadata disagrees with threshold values")
        true_without_encoded = bool(true_strict_improvement and not changed)
        self._true_threshold = float(true_threshold_after_trial)
        self._encoded_threshold = encoded_after
        if changed:
            self._stage_id += 1
            self._trial_in_stage = 0
            self._trials_since_encoded_threshold_change = 0
        return {
            "true_threshold_after_trial": float(true_threshold_after_trial),
            "encoded_threshold_after_trial": encoded_after,
            "threshold_stage_id_after_trial": int(self._stage_id),
            "true_threshold_update_without_encoded_change": true_without_encoded,
            "stop_reason_after_trial": stop_reason_after_trial,
        }

    def measured_metadata(
        self,
        *,
        measured_index: int,
        before: Mapping[str, object],
        auxiliary_accepted: bool,
        candidate_admitted: bool,
        cache_hit: bool,
        new_ed_lp_solve: bool,
        exact_evaluation_success: bool | None,
        candidate_true_cost: float | None,
        true_strict_improvement: bool,
        incumbent_updated: bool,
    ) -> dict[str, object]:
        index = int(measured_index)
        _index, bits, value, feasible = self._states[index]
        threshold = int(before["encoded_threshold_before_trial"])
        cost_marked = bool(value < threshold)
        joint_marked = bool(cost_marked and feasible)
        cost_indices = list(before["cost_marked_indices_before"])
        joint_indices = list(before["joint_marked_indices_before"])
        if bool(index in cost_indices) != cost_marked or bool(index in joint_indices) != joint_marked:
            raise AssertionError("marked-set membership must match strict measured predicates")
        if int(before["cost_marked_count_before"]) != len(cost_indices):
            raise AssertionError("cost marked count must equal list length")
        if int(before["joint_marked_count_before"]) != len(joint_indices):
            raise AssertionError("joint marked count must equal list length")
        cache_source = "none"
        if cache_hit:
            cache_source = "training_cache" if index in self._initial_cache_indices else "search_cache"
        candidate_in_training_set = bool(index in self._initial_cache_indices)
        improvement_is_nontraining = bool(
            true_strict_improvement and not candidate_in_training_set
        )
        if (
            self._initial_incumbent_policy == "best_training"
            and bool(true_strict_improvement)
            and not improvement_is_nontraining
        ):
            raise AssertionError(
                "best_training strict improvement must come from outside the training cache"
            )
        return {
            "measured_cost_marked": cost_marked,
            "measured_joint_marked": joint_marked,
            "auxiliary_accepted": bool(auxiliary_accepted),
            "hard_logic_feasible_measured": bool(feasible),
            "measured_integer_vqc_value": int(value),
            "candidate_admitted": bool(candidate_admitted),
            "cache_hit": bool(cache_hit),
            "cache_source": cache_source,
            "candidate_in_training_set": candidate_in_training_set,
            "improvement_is_nontraining": improvement_is_nontraining,
            "new_ed_lp_solve": bool(new_ed_lp_solve),
            "exact_evaluation_success": exact_evaluation_success,
            "candidate_true_cost": candidate_true_cost,
            "true_strict_improvement": bool(true_strict_improvement),
            "incumbent_updated": bool(incumbent_updated),
        }
