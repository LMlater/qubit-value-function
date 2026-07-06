from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .value_surrogate import ScalarValueFunctionEvaluation, count_rank_inversions


@dataclass(frozen=True)
class MaxAffineValueFunctionModel:
    names: list[str]
    coefficients: np.ndarray
    value_min: float
    value_range: float
    initialization: str
    anchor_indices: list[int]

    def scaled_piece_values(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        return features @ self.coefficients.T

    def predict_scaled(self, features: np.ndarray) -> np.ndarray:
        return np.max(self.scaled_piece_values(features), axis=1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.value_min + self.value_range * self.predict_scaled(features)


@dataclass(frozen=True)
class MaxAffineFitDiagnostics:
    requested_piece_count: int
    actual_piece_count: int
    selected_anchor_indices: list[int]
    selected_anchor_values: list[float]
    selected_anchor_residuals: list[float]
    lower_bound_violations: int
    max_lower_bound_violation: float


def fit_max_affine_value_function(
    features: np.ndarray,
    values: np.ndarray,
    names: list[str],
    *,
    piece_count: int,
    candidate_count: int | None = None,
    initialization: str = "floor",
    cut_tolerance: float = 1e-9,
    ridge: float = 0.0,
    sample_weights: np.ndarray | None = None,
    anchor_weights: np.ndarray | None = None,
    candidate_order: np.ndarray | None = None,
) -> tuple[MaxAffineValueFunctionModel, MaxAffineFitDiagnostics]:
    """Fit a max-affine surrogate ``max_r b_r + theta_r^T f(x)``.

    The additional pieces are supporting-cut-style affine functions constrained
    to lie below the observed normalized values on the training domain. This
    keeps each added piece compatible with a convex lower-envelope view while
    still allowing a least-squares affine piece as a practical initialization.
    """

    features = np.asarray(features, dtype=float)
    values = np.asarray(values, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must have shape (states, features)")
    if values.shape != (features.shape[0],):
        raise ValueError("values must have one entry per feature row")
    if piece_count <= 0:
        raise ValueError("piece_count must be positive")
    if candidate_count is not None and candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if cut_tolerance < 0.0:
        raise ValueError("cut_tolerance must be nonnegative")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    if not np.all(np.isfinite(values)):
        raise ValueError("fit_max_affine_value_function requires finite values")
    sample_weights = _validate_optional_weights(
        sample_weights,
        values.shape[0],
        name="sample_weights",
    )
    anchor_weights = _validate_optional_weights(
        anchor_weights,
        values.shape[0],
        name="anchor_weights",
    )
    if candidate_order is None:
        candidate_order = np.argsort(values)
    else:
        candidate_order = _validate_candidate_order(candidate_order, values.shape[0])

    value_min = float(values.min())
    value_range = float(values.max() - value_min)
    if value_range <= 0.0:
        raise ValueError("values must not be constant")
    targets = (values - value_min) / value_range

    pieces: list[np.ndarray] = []
    if initialization == "floor":
        pieces.append(np.zeros(features.shape[1], dtype=float))
    elif initialization == "least_squares":
        pieces.append(
            _least_squares_piece(
                features,
                targets,
                ridge,
                sample_weights=sample_weights,
            )
        )
    else:
        raise ValueError("initialization must be 'floor' or 'least_squares'")

    if candidate_count is not None:
        candidate_order = candidate_order[: min(candidate_count, candidate_order.size)]
    selected: list[int] = []
    selected_residuals: list[float] = []
    selected_values: list[float] = []

    while len(pieces) < piece_count:
        current = np.max(features @ np.vstack(pieces).T, axis=1)
        residuals = targets - current
        anchor = _select_anchor(
            candidate_order,
            residuals,
            set(selected),
            anchor_weights=anchor_weights,
        )
        if anchor is None:
            break
        cut = _supporting_cut(
            features,
            targets,
            int(anchor),
            cut_tolerance=cut_tolerance,
        )
        pieces.append(cut)
        selected.append(int(anchor))
        selected_residuals.append(float(residuals[anchor]))
        selected_values.append(float(values[anchor]))

    coefficients = np.vstack(pieces)
    model = MaxAffineValueFunctionModel(
        names=list(names),
        coefficients=coefficients,
        value_min=value_min,
        value_range=value_range,
        initialization=initialization,
        anchor_indices=selected,
    )
    scaled_predictions = model.predict_scaled(features)
    violations = scaled_predictions - targets
    positive_violations = violations > max(cut_tolerance * 10.0, 1e-10)
    diagnostics = MaxAffineFitDiagnostics(
        requested_piece_count=int(piece_count),
        actual_piece_count=int(coefficients.shape[0]),
        selected_anchor_indices=selected,
        selected_anchor_values=selected_values,
        selected_anchor_residuals=selected_residuals,
        lower_bound_violations=int(positive_violations.sum()),
        max_lower_bound_violation=float(np.max(np.maximum(violations, 0.0))),
    )
    return model, diagnostics


def evaluate_max_affine_value_function(
    model: MaxAffineValueFunctionModel,
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


def max_affine_gate_counts(feature_count: int, piece_count: int) -> dict[str, int]:
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    if piece_count <= 0:
        raise ValueError("piece_count must be positive")
    return {
        "affine_accumulators": int(piece_count),
        "affine_weighted_additions": int(feature_count * piece_count),
        "max_comparators": int(max(piece_count - 1, 0)),
        "threshold_comparators": 1,
    }


def _least_squares_piece(
    features: np.ndarray,
    targets: np.ndarray,
    ridge: float,
    *,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    if sample_weights is None:
        weighted_features = features
        weighted_targets = targets
    else:
        sqrt_weights = np.sqrt(sample_weights)[:, None]
        weighted_features = features * sqrt_weights
        weighted_targets = targets * sqrt_weights[:, 0]
    if ridge == 0.0:
        coefficients, *_ = np.linalg.lstsq(weighted_features, weighted_targets, rcond=None)
        return np.asarray(coefficients, dtype=float)
    lhs = weighted_features.T @ weighted_features + float(ridge) * np.eye(features.shape[1])
    rhs = weighted_features.T @ weighted_targets
    return np.linalg.solve(lhs, rhs)


def _select_anchor(
    candidate_order: np.ndarray,
    residuals: np.ndarray,
    selected: set[int],
    *,
    anchor_weights: np.ndarray | None = None,
) -> int | None:
    best_idx: int | None = None
    best_score = -np.inf
    for idx in candidate_order:
        idx = int(idx)
        if idx in selected:
            continue
        residual = max(float(residuals[idx]), 0.0)
        weight = 1.0 if anchor_weights is None else float(anchor_weights[idx])
        score = residual * weight
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx


def _validate_optional_weights(
    weights: np.ndarray | None,
    size: int,
    *,
    name: str,
) -> np.ndarray | None:
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (size,):
        raise ValueError(f"{name} must have one entry per training row")
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} must be finite")
    if np.any(weights <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return weights


def _validate_candidate_order(candidate_order: np.ndarray, size: int) -> np.ndarray:
    candidate_order = np.asarray(candidate_order, dtype=int)
    if candidate_order.ndim != 1:
        raise ValueError("candidate_order must be one-dimensional")
    if candidate_order.size == 0:
        raise ValueError("candidate_order must not be empty")
    if np.any(candidate_order < 0) or np.any(candidate_order >= size):
        raise ValueError("candidate_order contains out-of-range indices")
    deduplicated: list[int] = []
    seen: set[int] = set()
    for idx in candidate_order:
        idx = int(idx)
        if idx in seen:
            continue
        deduplicated.append(idx)
        seen.add(idx)
    if len(deduplicated) != size:
        for idx in range(size):
            if idx not in seen:
                deduplicated.append(idx)
    return np.asarray(deduplicated, dtype=int)


def _supporting_cut(
    features: np.ndarray,
    targets: np.ndarray,
    anchor_idx: int,
    *,
    cut_tolerance: float,
) -> np.ndarray:
    objective = -features[anchor_idx]
    result = linprog(
        c=objective,
        A_ub=features,
        b_ub=targets + float(cut_tolerance),
        bounds=[(None, None)] * features.shape[1],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"supporting cut LP failed: {result.message}")
    return np.asarray(result.x, dtype=float)
