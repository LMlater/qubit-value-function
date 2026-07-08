from __future__ import annotations

from math import comb

import numpy as np

from .experiment_utils import finite_or_none


def gap_metrics(final_true_cost: float | None, hidden_best_true_cost: float | None) -> dict[str, object]:
    if final_true_cost is None or hidden_best_true_cost is None:
        return _null_gap_metrics()
    final = float(final_true_cost)
    hidden = float(hidden_best_true_cost)
    if not np.isfinite(final) or not np.isfinite(hidden):
        return _null_gap_metrics()
    absolute_gap = final - hidden
    relative_gap = absolute_gap / max(abs(hidden), 1e-12)
    return {
        "absolute_gap_to_hidden_best": finite_or_none(absolute_gap),
        "relative_gap_to_hidden_best": finite_or_none(relative_gap),
        "success_within_1_percent": bool(relative_gap <= 0.01),
        "success_within_3_percent": bool(relative_gap <= 0.03),
        "success_within_5_percent": bool(relative_gap <= 0.05),
    }


def _null_gap_metrics() -> dict[str, object]:
    return {
        "absolute_gap_to_hidden_best": None,
        "relative_gap_to_hidden_best": None,
        "success_within_1_percent": False,
        "success_within_3_percent": False,
        "success_within_5_percent": False,
    }


def hypergeometric_hit_probability(*, dimension: int, num_good_states: int, draws: int) -> float:
    dimension = int(dimension)
    num_good_states = max(0, min(int(num_good_states), dimension))
    draws = max(0, min(int(draws), dimension))
    if dimension <= 0 or draws <= 0 or num_good_states <= 0:
        return 0.0
    if draws >= dimension or num_good_states >= dimension:
        return 1.0
    bad_states = dimension - num_good_states
    if draws > bad_states:
        return 1.0
    return float(1.0 - comb(bad_states, draws) / comb(dimension, draws))


def random_baseline_probabilities(
    *,
    values: np.ndarray,
    hidden_best_true_cost: float,
    draws: int,
) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    hidden = float(hidden_best_true_cost)
    finite = np.isfinite(values)
    dimension = int(values.size)
    exact_good = int(np.logical_and(finite, np.isclose(values, hidden, rtol=0.0, atol=1e-9)).sum())
    rel_gap = (values - hidden) / max(abs(hidden), 1e-12)
    eps = 1e-12
    within_1 = int(np.logical_and(finite, rel_gap <= 0.01 + eps).sum())
    within_3 = int(np.logical_and(finite, rel_gap <= 0.03 + eps).sum())
    within_5 = int(np.logical_and(finite, rel_gap <= 0.05 + eps).sum())
    return {
        "random_exact_success_probability": hypergeometric_hit_probability(
            dimension=dimension,
            num_good_states=exact_good,
            draws=draws,
        ),
        "random_success_within_1_percent_probability": hypergeometric_hit_probability(
            dimension=dimension,
            num_good_states=within_1,
            draws=draws,
        ),
        "random_success_within_3_percent_probability": hypergeometric_hit_probability(
            dimension=dimension,
            num_good_states=within_3,
            draws=draws,
        ),
        "random_success_within_5_percent_probability": hypergeometric_hit_probability(
            dimension=dimension,
            num_good_states=within_5,
            draws=draws,
        ),
    }
