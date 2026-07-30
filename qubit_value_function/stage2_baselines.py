"""Small deterministic classical baselines for the Stage B protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _arrays(state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(state_bits, dtype=float)
    loads = np.asarray(normalized_loads, dtype=float)
    if states.ndim != 2 or states.shape[1] != 4 or loads.shape != (len(states), 2):
        raise ValueError("state_bits must have shape (n, 4) and normalized_loads shape (n, 2)")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(loads)):
        raise ValueError("baseline inputs must be finite")
    return states, loads


def design_matrix(state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]], *, degree: int) -> np.ndarray:
    states, loads = _arrays(state_bits, normalized_loads)
    if degree not in {1, 2}:
        raise ValueError("degree must be one or two")
    rows = [np.ones(len(states)), *[states[:, index] for index in range(4)], loads[:, 0], loads[:, 1]]
    if degree == 2:
        combined = np.column_stack((states, loads))
        rows.extend(combined[:, left] * combined[:, right] for left in range(6) for right in range(left, 6))
    return np.column_stack(rows)


@dataclass(frozen=True)
class ConstantRegressor:
    value: float
    parameter_count: int = 1

    def predict(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]]) -> np.ndarray:
        states, _loads = _arrays(state_bits, normalized_loads)
        return np.full(len(states), self.value, dtype=float)


@dataclass(frozen=True)
class RidgeRegressor:
    degree: int
    weights: tuple[float, ...]
    parameter_count: int

    def predict(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]]) -> np.ndarray:
        return design_matrix(state_bits, normalized_loads, degree=self.degree) @ np.asarray(self.weights, dtype=float)


def fit_constant_regressor(costs: Sequence[float]) -> ConstantRegressor:
    targets = np.asarray(costs, dtype=float)
    if targets.ndim != 1 or not len(targets) or not np.all(np.isfinite(targets)):
        raise ValueError("costs must be non-empty and finite")
    return ConstantRegressor(value=float(targets.mean()))


def fit_ridge_regressor(
    state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]], costs: Sequence[float], *, degree: int, regularization: float
) -> RidgeRegressor:
    design = design_matrix(state_bits, normalized_loads, degree=degree)
    targets = np.asarray(costs, dtype=float)
    if targets.shape != (len(design),) or not np.all(np.isfinite(targets)):
        raise ValueError("costs must be finite and aligned with fit rows")
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    penalty = np.eye(design.shape[1], dtype=float) * float(regularization)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    if not np.all(np.isfinite(weights)):
        raise ValueError("ridge fit produced non-finite weights")
    return RidgeRegressor(degree=degree, weights=tuple(float(value) for value in weights), parameter_count=len(weights))
