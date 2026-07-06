from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroverMinimumRound:
    round_index: int
    incumbent_index_before: int
    incumbent_true_value_before: float
    incumbent_predicted_value_before: float
    marked_count: int
    iterations: int
    marked_probability: float
    sampled_index: int
    sampled_is_marked: bool
    sampled_is_feasible: bool
    sampled_true_value: float
    sampled_predicted_value: float
    accepted: bool
    incumbent_index_after: int
    incumbent_true_value_after: float
    incumbent_predicted_value_after: float


@dataclass(frozen=True)
class GroverMinimumRun:
    initial_index: int
    best_index: int
    best_true_value: float
    best_predicted_value: float
    optimum_index: int
    optimum_true_value: float
    success: bool
    oracle_calls: int
    rounds: list[GroverMinimumRound]


def optimal_grover_iterations(dimension: int, marked_count: int) -> int:
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("dimension must be a positive power of two")
    if marked_count <= 0:
        return 0
    if marked_count > dimension:
        raise ValueError("marked_count cannot exceed dimension")
    return max(1, int(np.floor(np.pi / 4.0 * np.sqrt(dimension / marked_count))))


def grover_threshold_probabilities(marked: np.ndarray, iterations: int | None = None) -> np.ndarray:
    """Return the state probabilities after Grover iterations for a marked set.

    The simulation keeps the state vector only. This is the state-space analogue
    of applying the value-register oracle, phase marking, uncomputing the
    auxiliary registers, and applying the standard diffuser on the x register.
    """

    marked = np.asarray(marked, dtype=bool)
    dimension = marked.size
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("number of states must be a power of two")
    marked_count = int(marked.sum())
    if marked_count == 0:
        return np.full(dimension, 1.0 / dimension, dtype=float)
    if iterations is None:
        iterations = optimal_grover_iterations(dimension, marked_count)
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")

    state = np.ones(dimension, dtype=float) / np.sqrt(dimension)
    for _ in range(iterations):
        state[marked] *= -1.0
        state = 2.0 * np.mean(state) - state
    probabilities = state * state
    return probabilities / probabilities.sum()


def run_grover_minimum_finding(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    feasible: np.ndarray,
    *,
    initial_index: int | None = None,
    max_rounds: int = 20,
    seed: int = 0,
    strict_tolerance: float = 1e-9,
) -> GroverMinimumRun:
    """Simulate threshold-iterated Grover minimum finding.

    The oracle threshold is the incumbent predicted value, so a round marks
    feasible states whose surrogate value is strictly below the incumbent
    surrogate value. A sampled state is accepted only when its true value
    improves the incumbent. Setting ``predicted_values=true_values`` gives the
    exact-value baseline.
    """

    true_values = np.asarray(true_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    if true_values.shape != predicted_values.shape or true_values.shape != feasible.shape:
        raise ValueError("true_values, predicted_values, and feasible must have the same shape")
    dimension = true_values.size
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("number of states must be a power of two")
    finite_feasible = feasible & np.isfinite(true_values) & np.isfinite(predicted_values)
    feasible_indices = np.flatnonzero(finite_feasible)
    if feasible_indices.size == 0:
        raise ValueError("at least one finite feasible state is required")
    if max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative")

    rng = np.random.default_rng(seed)
    if initial_index is None:
        initial_index = int(rng.choice(feasible_indices))
    if not finite_feasible[int(initial_index)]:
        raise ValueError("initial_index must refer to a finite feasible state")

    optimum_index = int(feasible_indices[np.argmin(true_values[feasible_indices])])
    incumbent = int(initial_index)
    rounds: list[GroverMinimumRound] = []
    oracle_calls = 0

    for round_index in range(max_rounds):
        threshold = float(predicted_values[incumbent])
        marked = finite_feasible & (predicted_values < threshold - strict_tolerance)
        marked_count = int(marked.sum())
        if marked_count == 0:
            break

        iterations = optimal_grover_iterations(dimension, marked_count)
        probabilities = grover_threshold_probabilities(marked, iterations)
        sampled = int(rng.choice(dimension, p=probabilities))
        oracle_calls += iterations
        accepted = bool(
            finite_feasible[sampled]
            and true_values[sampled] < true_values[incumbent] - strict_tolerance
        )
        before = incumbent
        if accepted:
            incumbent = sampled
        rounds.append(
            GroverMinimumRound(
                round_index=round_index,
                incumbent_index_before=before,
                incumbent_true_value_before=float(true_values[before]),
                incumbent_predicted_value_before=float(predicted_values[before]),
                marked_count=marked_count,
                iterations=iterations,
                marked_probability=float(probabilities[marked].sum()),
                sampled_index=sampled,
                sampled_is_marked=bool(marked[sampled]),
                sampled_is_feasible=bool(finite_feasible[sampled]),
                sampled_true_value=float(true_values[sampled]) if np.isfinite(true_values[sampled]) else float("inf"),
                sampled_predicted_value=(
                    float(predicted_values[sampled]) if np.isfinite(predicted_values[sampled]) else float("inf")
                ),
                accepted=accepted,
                incumbent_index_after=incumbent,
                incumbent_true_value_after=float(true_values[incumbent]),
                incumbent_predicted_value_after=float(predicted_values[incumbent]),
            )
        )
        if incumbent == optimum_index:
            break

    return GroverMinimumRun(
        initial_index=int(initial_index),
        best_index=incumbent,
        best_true_value=float(true_values[incumbent]),
        best_predicted_value=float(predicted_values[incumbent]),
        optimum_index=optimum_index,
        optimum_true_value=float(true_values[optimum_index]),
        success=bool(incumbent == optimum_index),
        oracle_calls=int(oracle_calls),
        rounds=rounds,
    )


def summarize_minimum_finding_runs(runs: list[GroverMinimumRun]) -> dict[str, object]:
    if not runs:
        raise ValueError("at least one run is required")
    gaps = np.array([run.best_true_value - run.optimum_true_value for run in runs], dtype=float)
    oracle_calls = np.array([run.oracle_calls for run in runs], dtype=float)
    round_counts = np.array([len(run.rounds) for run in runs], dtype=float)
    return {
        "trial_count": len(runs),
        "success_count": int(sum(run.success for run in runs)),
        "success_rate": float(np.mean([run.success for run in runs])),
        "mean_true_optimality_gap": float(np.mean(gaps)),
        "median_true_optimality_gap": float(np.median(gaps)),
        "max_true_optimality_gap": float(np.max(gaps)),
        "mean_oracle_calls": float(np.mean(oracle_calls)),
        "median_oracle_calls": float(np.median(oracle_calls)),
        "mean_rounds": float(np.mean(round_counts)),
        "median_rounds": float(np.median(round_counts)),
    }
