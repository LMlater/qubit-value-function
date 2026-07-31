"""Leakage-free joins, attribution summaries, and paired statistics for Stage B."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


PENALTY_DOMINANT_RATIO = 0.5
KEY_FIELDS = ("generator_pair", "window_start", "load_multiplier", "state_index")
UNIT_FIELDS = ("generator_pair", "window_start", "split_seed", "model", "slice")


def _key(row: Mapping[str, object]) -> tuple[str, int, float, int]:
    return (str(row["generator_pair"]), int(row["window_start"]), float(row["load_multiplier"]), int(row["state_index"]))


def join_formal_and_components(formal_rows: Sequence[Mapping[str, object]], component_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    components = {_key(row): row for row in component_rows}
    if len(components) != len(component_rows):
        raise ValueError("component_join_duplicate_key")
    missing = [_key(row) for row in formal_rows if _key(row) not in components]
    if missing:
        raise ValueError(f"component_join_missing:{len(missing)}")
    return [{**dict(row), **{name: value for name, value in components[_key(row)].items() if name not in KEY_FIELDS}} for row in formal_rows]


def residual_targets_from_fit_rows(true_costs: Sequence[float], base_predictions: Sequence[float]) -> np.ndarray:
    truth, base = np.asarray(true_costs, dtype=float), np.asarray(base_predictions, dtype=float)
    if truth.shape != base.shape or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(base)):
        raise ValueError("fit_residual_inputs_must_be_finite_and_aligned")
    return truth - base


def margin_targets_from_fit_rows(*, loads: Sequence[float], true_costs: Sequence[float]) -> tuple[np.ndarray, dict[float, float]]:
    load_values, truth = np.asarray(loads, dtype=float), np.asarray(true_costs, dtype=float)
    if load_values.shape != truth.shape or not len(truth) or not np.all(np.isfinite(truth)):
        raise ValueError("fit_margin_inputs_must_be_finite_and_aligned")
    tau = {float(load): float(np.min(truth[load_values == load])) for load in sorted(set(load_values.tolist()))}
    return np.asarray([tau[float(load)] - cost for load, cost in zip(load_values, truth)], dtype=float), tau


def feasible_improvement_labels(*, true_costs: Sequence[float], thresholds: Sequence[float], hard_logic_feasible: Sequence[bool], edlp_success: Sequence[bool]) -> np.ndarray:
    costs, cutoff = np.asarray(true_costs, dtype=float), np.asarray(thresholds, dtype=float)
    logic, success = np.asarray(hard_logic_feasible, dtype=bool), np.asarray(edlp_success, dtype=bool)
    if costs.shape != cutoff.shape or costs.shape != logic.shape or costs.shape != success.shape:
        raise ValueError("feasible_improvement_inputs_must_align")
    return logic & success & (costs < cutoff)


def aggregate_seed_rows_by_unit(rows: Sequence[Mapping[str, object]], *, metric: str) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in UNIT_FIELDS)].append(float(row[metric]))
    return [dict(zip(UNIT_FIELDS, key)) | {metric: float(np.mean(values)), "seed_count": len(values)} for key, values in sorted(grouped.items(), key=lambda item: repr(item[0]))]


def bootstrap_paired_interval(differences: Sequence[float], *, seed: int = 20260731, draws: int = 10_000) -> dict[str, float]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("paired_differences_must_be_nonempty_and_finite")
    rng = np.random.default_rng(seed); means = np.mean(rng.choice(values, size=(int(draws), len(values)), replace=True), axis=1)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "ci95_low": float(np.quantile(means, 0.025)), "ci95_high": float(np.quantile(means, 0.975)), "unit_count": int(len(values))}


def reserve_penalty_group_label(values: Sequence[float]) -> str:
    return "N/A: no nonzero reserve-penalty samples" if not any(float(value) != 0.0 for value in values) else "available"


def error_components(true_costs: Sequence[float], predictions: Sequence[float]) -> dict[str, float]:
    truth, predicted = np.asarray(true_costs, dtype=float), np.asarray(predictions, dtype=float)
    error = predicted - truth; absolute = np.abs(error)
    return {"mae": float(absolute.mean()), "mean_signed_error": float(error.mean()), "median_absolute_error": float(np.median(absolute)), "maximum_absolute_error": float(absolute.max())}


def win_tie_loss(differences: Sequence[float], *, tolerance: float = 1e-9) -> dict[str, int]:
    values = np.asarray(differences, dtype=float)
    return {"win": int(np.sum(values < -tolerance)), "tie": int(np.sum(np.abs(values) <= tolerance)), "loss": int(np.sum(values > tolerance))}
