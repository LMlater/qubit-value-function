from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phase_vqc import tau_polynomial_features, wrapped_phase_error


@dataclass(frozen=True)
class AncillaVQCModel:
    names: list[str]
    coefficients: np.ndarray

    def angles(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=float) @ self.coefficients

    def unitary(self, features: np.ndarray) -> np.ndarray:
        angles = self.angles(features)
        blocks = [_ry(angle) for angle in angles]
        return _block_diag(blocks)

    def oracle_matrix(self, features: np.ndarray) -> np.ndarray:
        unitary = self.unitary(features)
        z_ancilla = np.tile(np.diag([1.0, -1.0]), (features.shape[0], 1, 1))
        z_matrix = _block_diag(list(z_ancilla))
        return unitary.T.conj() @ z_matrix @ unitary


@dataclass(frozen=True)
class AncillaVQCEvaluation:
    angle_mae: float
    max_angle_error: float
    max_leakage_probability: float
    mean_leakage_probability: float
    marked_accuracy: float
    correct_marked_set: bool
    false_positive_count: int
    false_negative_count: int


@dataclass(frozen=True)
class ThresholdConditionedAncillaVQCModel:
    names: list[str]
    coefficients: np.ndarray
    tau_degree: int
    tau_min: float
    tau_max: float
    tau_basis: str = "polynomial"
    tau_knots: np.ndarray | None = None

    def angles(self, features: np.ndarray, tau: float) -> np.ndarray:
        tau_features = _tau_features(
            tau,
            self.tau_degree,
            self.tau_min,
            self.tau_max,
            self.tau_basis,
            self.tau_knots,
        )
        return np.asarray(features, dtype=float) @ self.coefficients @ tau_features

    def fixed_tau_model(self, tau: float) -> AncillaVQCModel:
        tau_features = _tau_features(
            tau,
            self.tau_degree,
            self.tau_min,
            self.tau_max,
            self.tau_basis,
            self.tau_knots,
        )
        coefficients = self.coefficients @ tau_features
        return AncillaVQCModel(names=list(self.names), coefficients=np.asarray(coefficients, dtype=float))


@dataclass(frozen=True)
class ThresholdConditionedAncillaVQCEvaluation:
    thresholds: np.ndarray
    marked_accuracy: float
    correct_marked_sets: bool
    max_leakage_probability: float
    mean_leakage_probability: float
    threshold_summaries: list[dict[str, object]]


@dataclass(frozen=True)
class LeakageReweightedTrainingResult:
    model: AncillaVQCModel
    initial_evaluation: AncillaVQCEvaluation
    final_evaluation: AncillaVQCEvaluation
    selected_alpha: float | None
    selected_iteration: int
    history: list[dict[str, object]]


def fit_ancilla_vqc(features: np.ndarray, labels: np.ndarray, names: list[str]) -> AncillaVQCModel:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    target_angles = np.pi * labels.astype(float)
    coefficients, *_ = np.linalg.lstsq(features, target_angles, rcond=None)
    return AncillaVQCModel(names=list(names), coefficients=np.asarray(coefficients, dtype=float))


def fit_threshold_conditioned_ancilla_vqc(
    features: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float] | np.ndarray,
    names: list[str],
    *,
    tau_degree: int | None = None,
    sample_weights: np.ndarray | None = None,
    tau_basis: str = "polynomial",
) -> ThresholdConditionedAncillaVQCModel:
    """Fit one ancilla VQC for a family of threshold comparisons."""

    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    thresholds = np.asarray(thresholds, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must have shape (states, features)")
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("at least one threshold is required")
    if labels.shape != (features.shape[0], thresholds.size):
        raise ValueError("labels must have shape (states, thresholds)")
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=float)
        if sample_weights.shape != labels.shape:
            raise ValueError("sample_weights must have shape (states, thresholds)")
        if np.any(sample_weights < 0.0):
            raise ValueError("sample_weights must be nonnegative")
    if tau_basis not in {"polynomial", "piecewise_linear"}:
        raise ValueError("tau_basis must be 'polynomial' or 'piecewise_linear'")
    if tau_basis == "piecewise_linear" and (
        thresholds.size < 2 or np.any(np.diff(thresholds) <= 0.0)
    ):
        raise ValueError("piecewise_linear tau_basis requires increasing thresholds")
    if tau_basis == "piecewise_linear":
        tau_degree = int(thresholds.size - 1)
    elif tau_degree is None:
        tau_degree = max(0, thresholds.size - 1)

    tau_min = float(np.min(thresholds))
    tau_max = float(np.max(thresholds))
    design = _threshold_conditioned_design(
        features,
        thresholds,
        tau_degree,
        tau_min,
        tau_max,
        tau_basis,
    )
    target_angles = np.pi * labels.astype(float).reshape(-1)
    if sample_weights is not None:
        sqrt_weights = np.sqrt(sample_weights.reshape(-1))
        design = design * sqrt_weights[:, None]
        target_angles = target_angles * sqrt_weights
    coefficients, *_ = np.linalg.lstsq(design, target_angles, rcond=None)
    coefficient_matrix = np.asarray(coefficients, dtype=float).reshape(
        features.shape[1],
        _tau_feature_count(tau_degree, thresholds, tau_basis),
    )
    return ThresholdConditionedAncillaVQCModel(
        names=list(names),
        coefficients=coefficient_matrix,
        tau_degree=int(tau_degree),
        tau_min=tau_min,
        tau_max=tau_max,
        tau_basis=tau_basis,
        tau_knots=np.asarray(thresholds, dtype=float) if tau_basis == "piecewise_linear" else None,
    )


def evaluate_threshold_conditioned_ancilla_vqc(
    model: ThresholdConditionedAncillaVQCModel,
    features: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float] | np.ndarray,
) -> ThresholdConditionedAncillaVQCEvaluation:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    thresholds = np.asarray(thresholds, dtype=float)
    if labels.shape != (features.shape[0], thresholds.size):
        raise ValueError("labels must have shape (states, thresholds)")

    summaries = []
    predicted_columns = []
    leakage_columns = []
    for idx, tau in enumerate(thresholds):
        fixed_model = model.fixed_tau_model(float(tau))
        evaluation = evaluate_ancilla_vqc(fixed_model, features, labels[:, idx])
        angles = fixed_model.angles(features)
        predicted = np.cos(angles) < 0.0
        leakage = np.sin(angles) ** 2
        predicted_columns.append(predicted)
        leakage_columns.append(leakage)
        summaries.append(
            {
                "threshold": float(tau),
                "target_count": int(labels[:, idx].sum()),
                "predicted_count": int(predicted.sum()),
                "marked_accuracy": evaluation.marked_accuracy,
                "correct_marked_set": evaluation.correct_marked_set,
                "false_positive_count": evaluation.false_positive_count,
                "false_negative_count": evaluation.false_negative_count,
                "max_leakage_probability": evaluation.max_leakage_probability,
                "mean_leakage_probability": evaluation.mean_leakage_probability,
            }
        )

    predicted_matrix = np.column_stack(predicted_columns)
    leakage_matrix = np.column_stack(leakage_columns)
    return ThresholdConditionedAncillaVQCEvaluation(
        thresholds=thresholds,
        marked_accuracy=float(np.mean(predicted_matrix == labels)),
        correct_marked_sets=bool(np.array_equal(predicted_matrix, labels)),
        max_leakage_probability=float(np.max(leakage_matrix)),
        mean_leakage_probability=float(np.mean(leakage_matrix)),
        threshold_summaries=summaries,
    )


def fit_leakage_reweighted_ancilla_vqc(
    features: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    *,
    alphas: tuple[float, ...] = (5.0, 20.0, 100.0),
    iterations: int = 6,
) -> LeakageReweightedTrainingResult:
    """Fit an ancilla VQC while iteratively emphasizing high-leakage states."""

    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    target_angles = np.pi * labels.astype(float)
    base_model = fit_ancilla_vqc(features, labels, names)
    initial_evaluation = evaluate_ancilla_vqc(base_model, features, labels)
    best_model = base_model
    best_evaluation = initial_evaluation
    best_alpha: float | None = None
    best_iteration = 0
    history: list[dict[str, object]] = [
        {
            "method": "ordinary_lstsq",
            "alpha": None,
            "iteration": 0,
            "correct_marked_set": initial_evaluation.correct_marked_set,
            "max_leakage_probability": initial_evaluation.max_leakage_probability,
            "mean_leakage_probability": initial_evaluation.mean_leakage_probability,
        }
    ]

    for alpha in alphas:
        weights = np.ones(features.shape[0], dtype=float)
        for iteration in range(1, iterations + 1):
            weighted_features = features * np.sqrt(weights)[:, None]
            weighted_targets = target_angles * np.sqrt(weights)
            coefficients, *_ = np.linalg.lstsq(weighted_features, weighted_targets, rcond=None)
            model = AncillaVQCModel(names=list(names), coefficients=np.asarray(coefficients, dtype=float))
            evaluation = evaluate_ancilla_vqc(model, features, labels)
            history.append(
                {
                    "method": "leakage_reweighted_lstsq",
                    "alpha": float(alpha),
                    "iteration": iteration,
                    "correct_marked_set": evaluation.correct_marked_set,
                    "max_leakage_probability": evaluation.max_leakage_probability,
                    "mean_leakage_probability": evaluation.mean_leakage_probability,
                }
            )
            if _is_better_leakage_evaluation(evaluation, best_evaluation):
                best_model = model
                best_evaluation = evaluation
                best_alpha = float(alpha)
                best_iteration = iteration

            leakage = np.sin(features @ coefficients) ** 2
            max_leakage = float(np.max(leakage))
            if max_leakage <= 1e-15:
                weights = np.ones_like(weights)
            else:
                weights = 1.0 + float(alpha) * (leakage / max_leakage) ** 2

    return LeakageReweightedTrainingResult(
        model=best_model,
        initial_evaluation=initial_evaluation,
        final_evaluation=best_evaluation,
        selected_alpha=best_alpha,
        selected_iteration=best_iteration,
        history=history,
    )


def _threshold_conditioned_design(
    features: np.ndarray,
    thresholds: np.ndarray,
    tau_degree: int,
    tau_min: float,
    tau_max: float,
    tau_basis: str,
) -> np.ndarray:
    tau_features = _tau_feature_matrix(thresholds, tau_degree, tau_min, tau_max, tau_basis)
    design = features[:, None, :, None] * tau_features[None, :, None, :]
    return design.reshape(
        features.shape[0] * thresholds.size,
        features.shape[1] * tau_features.shape[1],
    )


def _tau_feature_count(
    tau_degree: int,
    thresholds: np.ndarray,
    tau_basis: str,
) -> int:
    if tau_basis == "piecewise_linear":
        return int(thresholds.size)
    return int(tau_degree + 1)


def _tau_feature_matrix(
    thresholds: np.ndarray,
    tau_degree: int,
    tau_min: float,
    tau_max: float,
    tau_basis: str,
) -> np.ndarray:
    if tau_basis == "piecewise_linear":
        return np.eye(thresholds.size, dtype=float)
    return np.vstack(
        [
            tau_polynomial_features(float(tau), tau_degree, tau_min, tau_max)
            for tau in thresholds
        ]
    )


def _tau_features(
    tau: float,
    tau_degree: int,
    tau_min: float,
    tau_max: float,
    tau_basis: str,
    tau_knots: np.ndarray | None,
) -> np.ndarray:
    if tau_basis == "piecewise_linear":
        if tau_knots is None:
            raise ValueError("piecewise_linear model requires tau_knots")
        return _piecewise_linear_tau_features(float(tau), np.asarray(tau_knots, dtype=float))
    return tau_polynomial_features(float(tau), tau_degree, tau_min, tau_max)


def _piecewise_linear_tau_features(tau: float, knots: np.ndarray) -> np.ndarray:
    if knots.ndim != 1 or knots.size < 2:
        raise ValueError("piecewise-linear tau basis requires at least two knots")
    if tau <= knots[0]:
        features = np.zeros(knots.size, dtype=float)
        features[0] = 1.0
        return features
    if tau >= knots[-1]:
        features = np.zeros(knots.size, dtype=float)
        features[-1] = 1.0
        return features
    upper = int(np.searchsorted(knots, tau, side="right"))
    lower = upper - 1
    span = knots[upper] - knots[lower]
    weight_upper = (tau - knots[lower]) / span
    features = np.zeros(knots.size, dtype=float)
    features[lower] = 1.0 - weight_upper
    features[upper] = weight_upper
    return features


def evaluate_ancilla_vqc(
    model: AncillaVQCModel,
    features: np.ndarray,
    labels: np.ndarray,
) -> AncillaVQCEvaluation:
    labels = np.asarray(labels, dtype=bool)
    angles = model.angles(features)
    target_angles = np.pi * labels.astype(float)
    angle_errors = wrapped_phase_error(angles, target_angles)
    predicted = np.cos(angles) < 0.0
    leakage_probability = np.sin(angles) ** 2
    return AncillaVQCEvaluation(
        angle_mae=float(np.mean(angle_errors)),
        max_angle_error=float(np.max(angle_errors)),
        max_leakage_probability=float(np.max(leakage_probability)),
        mean_leakage_probability=float(np.mean(leakage_probability)),
        marked_accuracy=float(np.mean(predicted == labels)),
        correct_marked_set=bool(np.array_equal(predicted, labels)),
        false_positive_count=int(np.logical_and(predicted, ~labels).sum()),
        false_negative_count=int(np.logical_and(~predicted, labels).sum()),
    )


def _is_better_leakage_evaluation(
    candidate: AncillaVQCEvaluation,
    incumbent: AncillaVQCEvaluation,
) -> bool:
    if candidate.correct_marked_set and not incumbent.correct_marked_set:
        return True
    if incumbent.correct_marked_set and not candidate.correct_marked_set:
        return False
    return (
        candidate.max_leakage_probability,
        candidate.mean_leakage_probability,
    ) < (
        incumbent.max_leakage_probability,
        incumbent.mean_leakage_probability,
    )


def verify_ancilla_oracle(oracle: np.ndarray, atol: float = 1e-10) -> dict[str, bool]:
    identity = np.eye(oracle.shape[0])
    return {
        "unitary": bool(np.allclose(oracle.T.conj() @ oracle, identity, atol=atol)),
        "self_inverse": bool(np.allclose(oracle @ oracle, identity, atol=atol)),
    }


def ancilla_oracle_errors(oracle: np.ndarray) -> dict[str, float]:
    identity = np.eye(oracle.shape[0])
    return {
        "unitarity_error": float(np.max(np.abs(oracle.T.conj() @ oracle - identity))),
        "self_inverse_error": float(np.max(np.abs(oracle @ oracle - identity))),
    }


def ancilla_block_oracle_errors(model: AncillaVQCModel, features: np.ndarray) -> dict[str, float]:
    angles = model.angles(features)
    norm_errors = np.abs(np.cos(angles) ** 2 + np.sin(angles) ** 2 - 1.0)
    max_error = float(np.max(norm_errors))
    return {
        "unitarity_error": max_error,
        "self_inverse_error": max_error,
    }


def controlled_ancilla_block_oracle_errors(
    model: AncillaVQCModel,
    features: np.ndarray,
    feasible_mask: np.ndarray,
) -> dict[str, float]:
    feasible_mask = np.asarray(feasible_mask, dtype=bool)
    if features.shape[0] != feasible_mask.size:
        raise ValueError("features and feasible_mask must have the same number of states")
    angles = model.angles(features)[feasible_mask]
    norm_errors = np.abs(np.cos(angles) ** 2 + np.sin(angles) ** 2 - 1.0)
    max_error = float(np.max(norm_errors))
    return {
        "unitarity_error": max_error,
        "self_inverse_error": max_error,
    }


def grover_with_ancilla_oracle(
    oracle: np.ndarray,
    labels: np.ndarray,
    iterations: int | None = None,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=bool)
    dimension = labels.size
    target_count = int(labels.sum())
    if target_count == 0:
        raise ValueError("Grover search needs at least one target state")
    if oracle.shape != (2 * dimension, 2 * dimension):
        raise ValueError("oracle dimension must be 2 * number of x states")
    if iterations is None:
        iterations = max(1, int(np.floor(np.pi / 4.0 * np.sqrt(dimension / target_count))))

    state = np.zeros(2 * dimension, dtype=complex)
    state[0::2] = 1.0 / np.sqrt(dimension)
    diffuser_x = _diffuser(dimension)
    diffuser = np.kron(diffuser_x, np.eye(2))
    for _ in range(iterations):
        state = diffuser @ (oracle @ state)

    probabilities = np.abs(state.reshape(dimension, 2)) ** 2
    x_probabilities = probabilities.sum(axis=1)
    return {
        "iterations": iterations,
        "target_x_probability": float(x_probabilities[labels].sum()),
        "non_target_x_probability": float(x_probabilities[~labels].sum()),
        "zero_ancilla_probability": float(probabilities[:, 0].sum()),
        "one_ancilla_probability": float(probabilities[:, 1].sum()),
        "target_zero_ancilla_probability": float(probabilities[labels, 0].sum()),
        "state_probabilities": [float(v) for v in x_probabilities],
    }


def grover_with_controlled_ancilla_model(
    model: AncillaVQCModel,
    features: np.ndarray,
    feasible_mask: np.ndarray,
    labels: np.ndarray,
    iterations: int | None = None,
) -> dict[str, object]:
    """Run Grover with the value oracle controlled by a feasibility oracle."""

    labels = np.asarray(labels, dtype=bool)
    feasible_mask = np.asarray(feasible_mask, dtype=bool)
    dimension = labels.size
    target_count = int(labels.sum())
    if target_count == 0:
        raise ValueError("Grover search needs at least one target state")
    if features.shape[0] != dimension or feasible_mask.shape != labels.shape:
        raise ValueError("features, feasible_mask, and labels must have compatible shapes")
    if np.any(labels & ~feasible_mask):
        raise ValueError("target labels must be a subset of feasible states")
    if iterations is None:
        iterations = max(1, int(np.floor(np.pi / 4.0 * np.sqrt(dimension / target_count))))

    state = np.zeros((dimension, 2), dtype=complex)
    state[:, 0] = 1.0 / np.sqrt(dimension)
    angles = model.angles(features)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    for _ in range(iterations):
        state = apply_controlled_ancilla_oracle_state(
            state,
            cos_angles,
            sin_angles,
            feasible_mask,
        )
        state = apply_x_diffuser_state(state)

    probabilities = np.abs(state) ** 2
    x_probabilities = probabilities.sum(axis=1)
    feasible_probability = float(x_probabilities[feasible_mask].sum())
    return {
        "iterations": iterations,
        "target_x_probability": float(x_probabilities[labels].sum()),
        "non_target_x_probability": float(x_probabilities[~labels].sum()),
        "feasible_x_probability": feasible_probability,
        "infeasible_x_probability": float(1.0 - feasible_probability),
        "zero_ancilla_probability": float(probabilities[:, 0].sum()),
        "one_ancilla_probability": float(probabilities[:, 1].sum()),
        "target_zero_ancilla_probability": float(probabilities[labels, 0].sum()),
        "state_probabilities": [float(v) for v in x_probabilities],
    }


def grover_with_explicit_two_ancilla_model(
    model: AncillaVQCModel,
    features: np.ndarray,
    feasible_mask: np.ndarray,
    labels: np.ndarray,
    iterations: int | None = None,
) -> dict[str, object]:
    """Run Grover with explicit feasibility and value ancilla registers."""

    labels = np.asarray(labels, dtype=bool)
    feasible_mask = np.asarray(feasible_mask, dtype=bool)
    dimension = labels.size
    target_count = int(labels.sum())
    if target_count == 0:
        raise ValueError("Grover search needs at least one target state")
    if features.shape[0] != dimension or feasible_mask.shape != labels.shape:
        raise ValueError("features, feasible_mask, and labels must have compatible shapes")
    if np.any(labels & ~feasible_mask):
        raise ValueError("target labels must be a subset of feasible states")
    if iterations is None:
        iterations = max(1, int(np.floor(np.pi / 4.0 * np.sqrt(dimension / target_count))))

    state = np.zeros((dimension, 2, 2), dtype=complex)
    state[:, 0, 0] = 1.0 / np.sqrt(dimension)
    angles = model.angles(features)
    for _ in range(iterations):
        state = apply_explicit_two_ancilla_oracle_state(state, angles, feasible_mask)
        state = apply_x_diffuser_register_state(state)

    probabilities = np.abs(state) ** 2
    x_probabilities = probabilities.sum(axis=(1, 2))
    feasible_probability = float(x_probabilities[feasible_mask].sum())
    return {
        "iterations": iterations,
        "target_x_probability": float(x_probabilities[labels].sum()),
        "non_target_x_probability": float(x_probabilities[~labels].sum()),
        "feasible_x_probability": feasible_probability,
        "infeasible_x_probability": float(1.0 - feasible_probability),
        "zero_feasibility_probability": float(probabilities[:, 0, :].sum()),
        "one_feasibility_probability": float(probabilities[:, 1, :].sum()),
        "zero_value_ancilla_probability": float(probabilities[:, :, 0].sum()),
        "one_value_ancilla_probability": float(probabilities[:, :, 1].sum()),
        "clean_ancilla_probability": float(probabilities[:, 0, 0].sum()),
        "dirty_ancilla_probability": float(1.0 - probabilities[:, 0, 0].sum()),
        "target_clean_probability": float(probabilities[labels, 0, 0].sum()),
        "target_dirty_probability": float(
            probabilities[labels].sum() - probabilities[labels, 0, 0].sum()
        ),
        "state_probabilities": [float(v) for v in x_probabilities],
    }


def grover_with_ancilla_model(
    model: AncillaVQCModel,
    features: np.ndarray,
    labels: np.ndarray,
    iterations: int | None = None,
) -> dict[str, object]:
    """Run Grover without materializing the full 2N x 2N oracle matrix."""

    labels = np.asarray(labels, dtype=bool)
    dimension = labels.size
    target_count = int(labels.sum())
    if target_count == 0:
        raise ValueError("Grover search needs at least one target state")
    if features.shape[0] != dimension:
        raise ValueError("features and labels must have the same number of states")
    if iterations is None:
        iterations = max(1, int(np.floor(np.pi / 4.0 * np.sqrt(dimension / target_count))))

    state = np.zeros((dimension, 2), dtype=complex)
    state[:, 0] = 1.0 / np.sqrt(dimension)
    cos_angles = np.cos(model.angles(features))
    sin_angles = np.sin(model.angles(features))
    for _ in range(iterations):
        state = apply_ancilla_oracle_state(state, cos_angles, sin_angles)
        state = apply_x_diffuser_state(state)

    probabilities = np.abs(state) ** 2
    x_probabilities = probabilities.sum(axis=1)
    return {
        "iterations": iterations,
        "target_x_probability": float(x_probabilities[labels].sum()),
        "non_target_x_probability": float(x_probabilities[~labels].sum()),
        "zero_ancilla_probability": float(probabilities[:, 0].sum()),
        "one_ancilla_probability": float(probabilities[:, 1].sum()),
        "target_zero_ancilla_probability": float(probabilities[labels, 0].sum()),
        "state_probabilities": [float(v) for v in x_probabilities],
    }


def apply_ancilla_oracle_state(
    state: np.ndarray,
    cos_angles: np.ndarray,
    sin_angles: np.ndarray,
) -> np.ndarray:
    """Apply each real 2x2 block of U^dagger Z U to a state shaped (N, 2)."""

    state = np.asarray(state, dtype=complex)
    if state.ndim != 2 or state.shape[1] != 2:
        raise ValueError("state must have shape (states, 2)")
    if cos_angles.shape != (state.shape[0],) or sin_angles.shape != (state.shape[0],):
        raise ValueError("angle arrays must have one entry per state")
    output = np.empty_like(state)
    output[:, 0] = cos_angles * state[:, 0] - sin_angles * state[:, 1]
    output[:, 1] = -sin_angles * state[:, 0] - cos_angles * state[:, 1]
    return output


def apply_explicit_two_ancilla_oracle_state(
    state: np.ndarray,
    angles: np.ndarray,
    feasible_mask: np.ndarray,
) -> np.ndarray:
    """Apply A_f, U_theta, CCZ(f,a), U_theta^dagger, A_f^dagger."""

    state = apply_feasibility_compute_state(state, feasible_mask)
    state = apply_value_rotation_state(state, angles)
    state = apply_ccz_feasibility_value_state(state)
    state = apply_value_rotation_state(state, -np.asarray(angles, dtype=float))
    return apply_feasibility_compute_state(state, feasible_mask)


def apply_feasibility_compute_state(state: np.ndarray, feasible_mask: np.ndarray) -> np.ndarray:
    """Toggle the feasibility ancilla f by f(x), i.e. |f> -> |f xor f(x)>."""

    state = np.asarray(state, dtype=complex)
    feasible_mask = np.asarray(feasible_mask, dtype=bool)
    if state.ndim != 3 or state.shape[1:] != (2, 2):
        raise ValueError("state must have shape (states, 2, 2)")
    if feasible_mask.shape != (state.shape[0],):
        raise ValueError("feasible_mask must have one entry per state")
    output = state.copy()
    output[feasible_mask, 0, :] = state[feasible_mask, 1, :]
    output[feasible_mask, 1, :] = state[feasible_mask, 0, :]
    return output


def apply_value_rotation_state(state: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Apply an x-controlled R_y(theta_x) to the value ancilla a."""

    state = np.asarray(state, dtype=complex)
    angles = np.asarray(angles, dtype=float)
    if state.ndim != 3 or state.shape[1:] != (2, 2):
        raise ValueError("state must have shape (states, 2, 2)")
    if angles.shape != (state.shape[0],):
        raise ValueError("angles must have one entry per state")
    cos_half = np.cos(angles / 2.0)[:, None]
    sin_half = np.sin(angles / 2.0)[:, None]
    output = np.empty_like(state)
    output[:, :, 0] = cos_half * state[:, :, 0] - sin_half * state[:, :, 1]
    output[:, :, 1] = sin_half * state[:, :, 0] + cos_half * state[:, :, 1]
    return output


def apply_ccz_feasibility_value_state(state: np.ndarray) -> np.ndarray:
    """Flip phase only on |f=1,a=1>."""

    state = np.asarray(state, dtype=complex)
    if state.ndim != 3 or state.shape[1:] != (2, 2):
        raise ValueError("state must have shape (states, 2, 2)")
    output = state.copy()
    output[:, 1, 1] *= -1.0
    return output


def apply_controlled_ancilla_oracle_state(
    state: np.ndarray,
    cos_angles: np.ndarray,
    sin_angles: np.ndarray,
    feasible_mask: np.ndarray,
) -> np.ndarray:
    """Apply identity on infeasible x and U^dagger Z U blocks on feasible x."""

    state = np.asarray(state, dtype=complex)
    feasible_mask = np.asarray(feasible_mask, dtype=bool)
    if state.ndim != 2 or state.shape[1] != 2:
        raise ValueError("state must have shape (states, 2)")
    if (
        cos_angles.shape != (state.shape[0],)
        or sin_angles.shape != (state.shape[0],)
        or feasible_mask.shape != (state.shape[0],)
    ):
        raise ValueError("angle arrays and feasible_mask must have one entry per state")
    output = state.copy()
    feasible_state = state[feasible_mask]
    output[feasible_mask, 0] = (
        cos_angles[feasible_mask] * feasible_state[:, 0]
        - sin_angles[feasible_mask] * feasible_state[:, 1]
    )
    output[feasible_mask, 1] = (
        -sin_angles[feasible_mask] * feasible_state[:, 0]
        - cos_angles[feasible_mask] * feasible_state[:, 1]
    )
    return output


def apply_x_diffuser_state(state: np.ndarray) -> np.ndarray:
    """Apply the standard Grover diffuser on x, separately for each ancilla value."""

    state = np.asarray(state, dtype=complex)
    if state.ndim != 2 or state.shape[1] != 2:
        raise ValueError("state must have shape (states, 2)")
    means = np.mean(state, axis=0, keepdims=True)
    return 2.0 * means - state


def apply_x_diffuser_register_state(state: np.ndarray) -> np.ndarray:
    """Apply the standard Grover diffuser on x for every trailing register state."""

    state = np.asarray(state, dtype=complex)
    if state.ndim < 2:
        raise ValueError("state must include an x axis and at least one register axis")
    means = np.mean(state, axis=0, keepdims=True)
    return 2.0 * means - state


def _ry(angle: float) -> np.ndarray:
    half = angle / 2.0
    return np.array(
        [
            [np.cos(half), -np.sin(half)],
            [np.sin(half), np.cos(half)],
        ],
        dtype=float,
    )


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    rows = sum(block.shape[0] for block in blocks)
    cols = sum(block.shape[1] for block in blocks)
    matrix = np.zeros((rows, cols), dtype=complex)
    row = 0
    col = 0
    for block in blocks:
        r, c = block.shape
        matrix[row : row + r, col : col + c] = block
        row += r
        col += c
    return matrix


def _diffuser(dimension: int) -> np.ndarray:
    uniform = np.ones((dimension, 1), dtype=complex) / np.sqrt(dimension)
    return 2 * (uniform @ uniform.T.conj()) - np.eye(dimension)
