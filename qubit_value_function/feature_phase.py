from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phase_vqc import wrapped_phase_error


@dataclass(frozen=True)
class FeaturePhaseModel:
    names: list[str]
    coefficients: np.ndarray

    def phase(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=float) @ self.coefficients

    def diagonal(self, features: np.ndarray) -> np.ndarray:
        return np.exp(1j * self.phase(features))

    def oracle_matrix(self, features: np.ndarray) -> np.ndarray:
        return np.diag(self.diagonal(features))

    def marked_by_phase(self, features: np.ndarray) -> np.ndarray:
        return np.real(self.diagonal(features)) < 0.0


def fit_feature_phase_model(
    features: np.ndarray,
    labels: np.ndarray,
    names: list[str],
) -> FeaturePhaseModel:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    target = np.pi * labels.astype(float)
    coefficients, *_ = np.linalg.lstsq(features, target, rcond=None)
    return FeaturePhaseModel(names=list(names), coefficients=np.asarray(coefficients, dtype=float))


def evaluate_feature_phase_model(
    model: FeaturePhaseModel,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=bool)
    target_phase = np.pi * labels.astype(float)
    phase = model.phase(features)
    factors = np.exp(1j * phase)
    target_factors = np.exp(1j * target_phase)
    predicted = np.real(factors) < 0.0
    return {
        "marked_accuracy": float(np.mean(predicted == labels)),
        "correct_marked_set": bool(np.array_equal(predicted, labels)),
        "target_count": int(labels.sum()),
        "predicted_count": int(predicted.sum()),
        "false_positive_count": int(np.logical_and(predicted, ~labels).sum()),
        "false_negative_count": int(np.logical_and(~predicted, labels).sum()),
        "max_phase_error": float(np.max(wrapped_phase_error(phase, target_phase))),
        "max_phase_factor_error": float(np.max(np.abs(factors - target_factors))),
    }


def selected_feature_matrix(
    full_features: np.ndarray,
    names: list[str],
    selected: list[int],
) -> tuple[np.ndarray, list[str]]:
    return full_features[:, selected], [names[idx] for idx in selected]


def forward_select_phase_features(
    features: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    *,
    max_terms: int,
    mandatory: list[int] | None = None,
) -> list[dict[str, object]]:
    """Greedily select phase features by least-squares phase-factor error."""

    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    target_factors = np.exp(1j * np.pi * labels.astype(float))
    selected = list(mandatory or [])
    remaining = [idx for idx in range(features.shape[1]) if idx not in selected]
    history: list[dict[str, object]] = []

    for _ in range(max_terms - len(selected)):
        best_candidate = None
        best_score = float("inf")
        best_model = None
        best_eval = None
        for candidate in remaining:
            trial = selected + [candidate]
            trial_features, trial_names = selected_feature_matrix(features, names, trial)
            model = fit_feature_phase_model(trial_features, labels, trial_names)
            factors = model.diagonal(trial_features)
            score = float(np.mean(np.abs(factors - target_factors) ** 2))
            evaluation = evaluate_feature_phase_model(model, trial_features, labels)
            if score < best_score:
                best_candidate = candidate
                best_score = score
                best_model = model
                best_eval = evaluation
        if best_candidate is None or best_model is None or best_eval is None:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        history.append(
            {
                "term_count": len(selected),
                "selected_indices": list(selected),
                "selected_names": [names[idx] for idx in selected],
                "score": best_score,
                "evaluation": best_eval,
                "coefficients": [float(v) for v in best_model.coefficients],
            }
        )

    return history
