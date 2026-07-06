from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class PhaseVQCModel:
    """Threshold-conditioned diagonal phase VQC.

    Each x-monomial corresponds to a controlled phase term. For a classical
    threshold tau, the term angle is a polynomial in the normalized threshold.
    """

    coefficients: np.ndarray
    x_subsets: tuple[tuple[int, ...], ...]
    tau_degree: int
    tau_min: float
    tau_max: float

    def phase(self, bits: np.ndarray, tau: float) -> np.ndarray:
        x_features = x_monomial_features(bits, self.x_subsets)
        tau_features = tau_polynomial_features(tau, self.tau_degree, self.tau_min, self.tau_max)
        return x_features @ self.coefficients @ tau_features

    def diagonal(self, bits: np.ndarray, tau: float) -> np.ndarray:
        return np.exp(1j * self.phase(bits, tau))

    def oracle_matrix(self, bits: np.ndarray, tau: float) -> np.ndarray:
        return np.diag(self.diagonal(bits, tau))

    def marked_by_phase(self, bits: np.ndarray, tau: float) -> np.ndarray:
        return np.real(self.diagonal(bits, tau)) < 0.0

    def gate_terms(self) -> list[dict[str, object]]:
        terms = []
        for row, subset in enumerate(self.x_subsets):
            terms.append(
                {
                    "controls": subset,
                    "angle_tau_polynomial": [float(v) for v in self.coefficients[row]],
                }
            )
        return terms


@dataclass(frozen=True)
class PhaseVQCTrainingResult:
    model: PhaseVQCModel
    thresholds: np.ndarray
    labels: np.ndarray
    max_phase_error: float
    max_phase_factor_error: float
    correct_marked_sets: bool
    residual_norm: float


@dataclass(frozen=True)
class PhaseVQCEvaluation:
    thresholds: np.ndarray
    labels: np.ndarray
    predicted_labels: np.ndarray
    max_phase_error: float
    max_phase_factor_error: float
    marked_accuracy: float
    correct_marked_sets: bool
    threshold_summaries: list[dict[str, object]]


def train_threshold_phase_vqc(
    bits: np.ndarray,
    values: np.ndarray,
    thresholds: list[float] | np.ndarray,
    *,
    x_order: int | None = None,
    tau_degree: int | None = None,
) -> PhaseVQCTrainingResult:
    """Fit U_theta(tau)|x> = exp(i phi_theta(x,tau))|x>.

    The target phase is 0 for V_d(x) > tau and pi for V_d(x) <= tau. The fitted
    model is a diagonal unitary for every tau because its entries are complex
    phases with unit modulus.
    """

    bits = np.asarray(bits, dtype=int)
    values = np.asarray(values, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if bits.ndim != 2:
        raise ValueError("bits must have shape (states, qubits)")
    if values.shape != (bits.shape[0],):
        raise ValueError("values must have one entry per bitstring")
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("at least one threshold is required")

    if x_order is None:
        x_order = bits.shape[1]
    if tau_degree is None:
        tau_degree = max(0, thresholds.size - 1)

    x_subsets = all_x_subsets(bits.shape[1], x_order)
    tau_min = float(thresholds.min())
    tau_max = float(thresholds.max())

    labels = values[:, None] <= thresholds[None, :]
    target_phases = np.pi * labels.astype(float)
    x_features = x_monomial_features(bits, x_subsets)
    tau_features = np.vstack(
        [tau_polynomial_features(tau, tau_degree, tau_min, tau_max) for tau in thresholds]
    )

    design_rows = []
    target = []
    for state_idx in range(bits.shape[0]):
        for tau_idx in range(thresholds.size):
            design_rows.append(np.kron(tau_features[tau_idx], x_features[state_idx]))
            target.append(target_phases[state_idx, tau_idx])
    design = np.vstack(design_rows)
    target_vec = np.asarray(target, dtype=float)
    raw_coefficients, residuals, *_ = np.linalg.lstsq(design, target_vec, rcond=None)
    coefficients = raw_coefficients.reshape((tau_degree + 1, len(x_subsets))).T

    model = PhaseVQCModel(
        coefficients=coefficients,
        x_subsets=x_subsets,
        tau_degree=tau_degree,
        tau_min=tau_min,
        tau_max=tau_max,
    )
    predicted_phases = np.column_stack([model.phase(bits, tau) for tau in thresholds])
    phase_errors = wrapped_phase_error(predicted_phases, target_phases)
    phase_factor_errors = np.abs(np.exp(1j * predicted_phases) - np.exp(1j * target_phases))
    evaluation = evaluate_threshold_phase_vqc(model, bits, values, thresholds)

    return PhaseVQCTrainingResult(
        model=model,
        thresholds=thresholds,
        labels=labels,
        max_phase_error=float(np.max(phase_errors)),
        max_phase_factor_error=float(np.max(phase_factor_errors)),
        correct_marked_sets=evaluation.correct_marked_sets,
        residual_norm=float(np.sqrt(residuals.sum()) if residuals.size else np.linalg.norm(design @ raw_coefficients - target_vec)),
    )


def evaluate_threshold_phase_vqc(
    model: PhaseVQCModel,
    bits: np.ndarray,
    values: np.ndarray,
    thresholds: list[float] | np.ndarray,
) -> PhaseVQCEvaluation:
    bits = np.asarray(bits, dtype=int)
    values = np.asarray(values, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    labels = values[:, None] <= thresholds[None, :]
    target_phases = np.pi * labels.astype(float)
    predicted_phases = np.column_stack([model.phase(bits, tau) for tau in thresholds])
    phase_errors = wrapped_phase_error(predicted_phases, target_phases)
    phase_factor_errors = np.abs(np.exp(1j * predicted_phases) - np.exp(1j * target_phases))
    predicted_labels = np.column_stack([model.marked_by_phase(bits, tau) for tau in thresholds])
    threshold_summaries: list[dict[str, object]] = []
    for idx, tau in enumerate(thresholds):
        exact = labels[:, idx]
        predicted = predicted_labels[:, idx]
        threshold_summaries.append(
            {
                "threshold": float(tau),
                "target_count": int(exact.sum()),
                "predicted_count": int(predicted.sum()),
                "accuracy": float(np.mean(exact == predicted)),
                "correct_marked_set": bool(np.array_equal(exact, predicted)),
                "false_positive_count": int(np.logical_and(predicted, ~exact).sum()),
                "false_negative_count": int(np.logical_and(~predicted, exact).sum()),
                "max_phase_factor_error": float(np.max(phase_factor_errors[:, idx])),
            }
        )

    return PhaseVQCEvaluation(
        thresholds=thresholds,
        labels=labels,
        predicted_labels=predicted_labels,
        max_phase_error=float(np.max(phase_errors)),
        max_phase_factor_error=float(np.max(phase_factor_errors)),
        marked_accuracy=float(np.mean(labels == predicted_labels)),
        correct_marked_sets=bool(np.array_equal(labels, predicted_labels)),
        threshold_summaries=threshold_summaries,
    )


def all_x_subsets(num_bits: int, order: int) -> tuple[tuple[int, ...], ...]:
    subsets: list[tuple[int, ...]] = [()]
    for size in range(1, order + 1):
        subsets.extend(tuple(combo) for combo in combinations(range(num_bits), size))
    return tuple(subsets)


def x_monomial_features(bits: np.ndarray, subsets: tuple[tuple[int, ...], ...]) -> np.ndarray:
    bits = np.asarray(bits, dtype=int)
    if bits.ndim == 1:
        bits = bits.reshape(1, -1)
    features = np.ones((bits.shape[0], len(subsets)), dtype=float)
    for col, subset in enumerate(subsets):
        if subset:
            features[:, col] = np.prod(bits[:, subset], axis=1)
    return features


def tau_polynomial_features(tau: float, degree: int, tau_min: float, tau_max: float) -> np.ndarray:
    if tau_max == tau_min:
        normalized = 0.0
    else:
        normalized = 2.0 * (float(tau) - tau_min) / (tau_max - tau_min) - 1.0
    return np.array([normalized**power for power in range(degree + 1)], dtype=float)


def wrapped_phase_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = predicted - target
    return np.abs((diff + np.pi) % (2 * np.pi) - np.pi)
