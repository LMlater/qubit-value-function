"""Pure post-processing helpers for persisted best-training smoke artifacts."""

from __future__ import annotations

from math import asin, isfinite, sin, sqrt
from typing import Mapping, Sequence

from .coherent_phase_value import conservative_integer_bounds
from .fixed_point_oracle import FixedPointConfig


def grover_marked_probability(*, marked_count: int, search_space_size: int, iterations: int) -> dict[str, float | None]:
    """Return the standard fixed-oracle Grover marked probability, without sampling."""

    marked = int(marked_count)
    size = int(search_space_size)
    if size <= 0 or marked < 0 or marked > size or int(iterations) < 0:
        raise ValueError("invalid Grover probability inputs")
    if marked == 0:
        return {
            "uniform_marked_probability": 0.0,
            "grover_theoretical_marked_probability": None,
            "marked_probability_gain": None,
        }
    theta = asin(sqrt(marked / size))
    probability = sin((2 * int(iterations) + 1) * theta) ** 2
    uniform = marked / size
    return {
        "uniform_marked_probability": uniform,
        "grover_theoretical_marked_probability": probability,
        "marked_probability_gain": probability / uniform,
    }


def classify_trial(row: Mapping[str, object]) -> str:
    """Assign exactly one post-hoc no-progress category to a persisted trial."""

    if not bool(row.get("measured_joint_marked")):
        return "unmarked_measurement"
    if bool(row.get("cache_hit")):
        return "marked_training_cache" if row.get("cache_source") == "training_cache" else "marked_search_cache"
    if bool(row.get("new_ed_lp_solve")):
        return "marked_new_edlp_improvement" if bool(row.get("true_strict_improvement")) else "marked_new_edlp_no_improvement"
    return "admitted_other"


def trial_row(
    *, scenario_id: str, method: str, policy: str, run_seed: int, row: Mapping[str, object], search_space_size: int,
) -> dict[str, object]:
    joint_indices = [int(item) for item in row.get("joint_marked_indices_before", [])]
    measured = int(row["measured_index"])
    recorded_marked = bool(row["measured_joint_marked"])
    derived_marked = measured in joint_indices
    if recorded_marked != derived_marked:
        raise AssertionError("measured_joint_marked disagrees with persisted joint marked set")
    probability = grover_marked_probability(
        marked_count=int(row["joint_marked_count_before"]),
        search_space_size=int(search_space_size),
        iterations=int(row["sampled_grover_iterations"]),
    )
    fields = (
        "threshold_stage_id", "trial_number", "true_threshold_before_trial",
        "encoded_threshold_before_trial", "cost_marked_count_before",
        "joint_marked_count_before", "sampled_grover_iterations", "measured_index",
        "measured_integer_vqc_value", "measured_joint_marked",
        "hard_logic_feasible_measured", "candidate_admitted", "cache_hit",
        "cache_source", "new_ed_lp_solve", "candidate_true_cost",
        "true_strict_improvement", "threshold_updated", "stop_reason_after_trial",
        "trials_since_last_threshold_update",
    )
    payload = {key: row.get(key) for key in fields}
    payload.update({
        "scenario_id": scenario_id,
        "method": method,
        "initial_incumbent_policy": policy,
        "run_seed": int(run_seed),
        "trial_index": int(row["trial_number"]),
        "measured_matches_theoretical_marked_set": True,
        "classification": classify_trial(row),
        **probability,
    })
    return payload


def quantization_diagnostic(
    snapshot: Mapping[str, object], *, true_threshold: float, fractional_bits: int) -> dict[str, object]:
    """Reapply coefficient quantization and strict integer comparison to a snapshot."""

    rows = list(snapshot["state_proxy_table"])
    config = FixedPointConfig(
        fractional_bits=int(fractional_bits),
        unit=float(snapshot["cost_unit"]),
        rounding=str(snapshot["quantization_mode"]),
    )
    real_intercept = float(snapshot["real_intercept"])
    real_weights = [float(value) for value in snapshot["real_weights"]]
    integer_intercept = config.encode(real_intercept)
    integer_weights = [config.encode(value) for value in real_weights]
    lower, upper = conservative_integer_bounds(integer_intercept, integer_weights)
    encoded_threshold = config.encode(float(true_threshold))
    integer_values = {
        int(row["state_index"]): int(integer_intercept + sum(
            weight * int(feature) for weight, feature in zip(integer_weights, row["feature_vector"])
        ))
        for row in rows
    }
    real_cost = [int(row["state_index"]) for row in rows if float(row["real_vqc_prediction"]) < float(true_threshold)]
    real_joint = [index for index in real_cost if bool(rows[index]["hard_logic_feasible"])]
    cost_marked = [index for index, value in integer_values.items() if value < encoded_threshold]
    joint_marked = [index for index in cost_marked if bool(rows[index]["hard_logic_feasible"])]
    current_bits = int(snapshot["fractional_bits"])
    if int(fractional_bits) == current_bits:
        persisted_values = {int(row["state_index"]): int(row["integer_vqc_value"]) for row in rows}
        if integer_values != persisted_values:
            raise AssertionError("fractional_bits=2 reconstruction differs from persisted integer values")
    real_set, integer_set = set(real_joint), set(joint_marked)
    return {
        "scenario_id": str(snapshot["scenario_id"]),
        "fractional_bits": int(fractional_bits),
        "lsb": float(config.quantum),
        "encoded_threshold": int(encoded_threshold),
        "integer_lower_bound": int(lower),
        "integer_upper_bound": int(upper),
        "required_value_register_width": max(1, int(upper - lower).bit_length()),
        "real_cost_marked_count": len(real_cost),
        "real_joint_marked_count": len(real_joint),
        "cost_marked_count": len(cost_marked),
        "joint_marked_count": len(joint_marked),
        "marked_indices": cost_marked,
        "joint_marked_indices": joint_marked,
        "recall_vs_real_joint": len(real_set & integer_set) / len(real_set) if real_set else None,
        "precision_vs_real_joint": len(real_set & integer_set) / len(integer_set) if integer_set else None,
        "equality_to_threshold_state_count": sum(value == encoded_threshold for value in integer_values.values()),
        "quantization_induced_false_negative_indices": sorted(real_set - integer_set),
        "quantization_induced_false_positive_indices": sorted(integer_set - real_set),
        "real_margins": [
            {"state_index": int(row["state_index"]), "real_margin": float(true_threshold - float(row["real_vqc_prediction"]))}
            for row in rows
        ],
    }


def ensure_json_finite(value: object) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("diagnostic JSON may not contain NaN or Infinity")
    if isinstance(value, Mapping):
        for item in value.values():
            ensure_json_finite(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            ensure_json_finite(item)
