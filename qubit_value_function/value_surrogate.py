from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScalarValueFunctionModel:
    names: list[str]
    coefficients: np.ndarray
    value_min: float
    value_range: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        scaled = np.asarray(features, dtype=float) @ self.coefficients
        return self.value_min + self.value_range * scaled


@dataclass(frozen=True)
class ScalarValueFunctionEvaluation:
    mae: float
    rmse: float
    max_abs_error: float
    rank_inversion_count: int


def fit_scalar_value_function(
    features: np.ndarray,
    values: np.ndarray,
    names: list[str],
    *,
    ridge: float = 0.0,
) -> ScalarValueFunctionModel:
    features = np.asarray(features, dtype=float)
    values = np.asarray(values, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must have shape (states, features)")
    if values.shape != (features.shape[0],):
        raise ValueError("values must have one entry per feature row")
    if not np.all(np.isfinite(values)):
        raise ValueError("fit_scalar_value_function requires finite values")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")

    value_min = float(values.min())
    value_range = float(values.max() - value_min)
    if value_range <= 0.0:
        raise ValueError("values must not be constant")
    targets = (values - value_min) / value_range
    if ridge == 0.0:
        coefficients, *_ = np.linalg.lstsq(features, targets, rcond=None)
    else:
        lhs = features.T @ features + float(ridge) * np.eye(features.shape[1])
        rhs = features.T @ targets
        coefficients = np.linalg.solve(lhs, rhs)
    return ScalarValueFunctionModel(
        names=list(names),
        coefficients=np.asarray(coefficients, dtype=float),
        value_min=value_min,
        value_range=value_range,
    )


def evaluate_scalar_value_function(
    model: ScalarValueFunctionModel,
    features: np.ndarray,
    values: np.ndarray,
) -> ScalarValueFunctionEvaluation:
    values = np.asarray(values, dtype=float)
    predictions = model.predict(features)
    errors = predictions - values
    return ScalarValueFunctionEvaluation(
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        max_abs_error=float(np.max(np.abs(errors))),
        rank_inversion_count=count_rank_inversions(values, predictions),
    )


def count_rank_inversions(values: np.ndarray, predictions: np.ndarray) -> int:
    values = np.asarray(values, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if values.shape != predictions.shape:
        raise ValueError("values and predictions must have the same shape")
    count = 0
    for idx in range(values.size):
        exact_less = values[idx] < values[idx + 1 :]
        predicted_greater = predictions[idx] > predictions[idx + 1 :]
        count += int(np.logical_and(exact_less, predicted_greater).sum())
        exact_greater = values[idx] > values[idx + 1 :]
        predicted_less = predictions[idx] < predictions[idx + 1 :]
        count += int(np.logical_and(exact_greater, predicted_less).sum())
    return count


def quantize_values(
    values: np.ndarray,
    *,
    value_min: float,
    value_max: float,
    bits: int,
) -> np.ndarray:
    if bits <= 0:
        raise ValueError("bits must be positive")
    if value_max <= value_min:
        raise ValueError("value_max must be greater than value_min")
    levels = 2**bits - 1
    scaled = (np.asarray(values, dtype=float) - value_min) / (value_max - value_min)
    return np.rint(np.clip(scaled, 0.0, 1.0) * levels).astype(int)
