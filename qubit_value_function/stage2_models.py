"""Uniform, deterministic model adapters for the Stage B protocol.

All ``fit`` methods consume only the caller-provided fit rows.  Validation
selection belongs to the benchmark orchestration layer; no model accepts test
rows or test labels.
"""

from __future__ import annotations

from time import perf_counter
from typing import Sequence

import numpy as np

from .load_conditioned_qnn import (
    ExpectationQNNConfig,
    ThresholdQNNConfig,
    expectation_feature_matrix,
    fit_expectation_qnn,
    fit_threshold_qnn,
)
from .stage2_baselines import fit_constant_regressor, fit_ridge_regressor
from .stage2_generalization import TargetNormalizer, select_threshold_cutoff_from_validation


def _fit_arrays(
    state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]], costs: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(state_bits, dtype=int)
    loads = np.asarray(normalized_loads, dtype=float)
    targets = np.asarray(costs, dtype=float)
    if states.ndim != 2 or states.shape[1] != 4 or not np.all(np.isin(states, [0, 1])):
        raise ValueError("state_bits must have shape (n, 4) and be binary")
    if loads.shape != (len(states), 2) or not np.all(np.isfinite(loads)):
        raise ValueError("normalized_loads must have shape (n, 2) and be finite")
    if targets.shape != (len(states),) or not len(targets) or not np.all(np.isfinite(targets)):
        raise ValueError("costs must be finite and aligned with samples")
    return states, loads, targets


class Stage2Model:
    """Common reporting surface required for all Stage B candidate models."""

    def __init__(self, *, seed: int) -> None:
        self.random_seed = int(seed)
        self.parameter_count = 0
        self.fit_status = "not_fitted"
        self.optimizer_message = "not_fitted"
        self.iteration_count = 0
        self.objective_initial = float("nan")
        self.objective_final = float("nan")
        self.converged = False
        self.runtime_seconds = 0.0

    def _record(
        self,
        *,
        parameter_count: int,
        status: str,
        iterations: int,
        objective_initial: float,
        objective_final: float,
        converged: bool,
        started: float,
        optimizer_message: str | None = None,
    ) -> None:
        values = (objective_initial, objective_final)
        if parameter_count < 0 or iterations < 0 or not all(np.isfinite(values)):
            raise ValueError("model fit report must be finite and non-negative")
        self.parameter_count = int(parameter_count)
        self.fit_status = str(status)
        self.optimizer_message = str(optimizer_message or status)
        self.iteration_count = int(iterations)
        self.objective_initial = float(objective_initial)
        self.objective_final = float(objective_final)
        self.converged = bool(converged)
        self.runtime_seconds = float(perf_counter() - started)


class ConstantBaselineModel(Stage2Model):
    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._model = None

    def fit(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]], costs: Sequence[float]) -> "ConstantBaselineModel":
        _states, _loads, targets = _fit_arrays(state_bits, normalized_loads, costs)
        started = perf_counter()
        self._model = fit_constant_regressor(targets)
        initial = float(np.mean((targets - 0.0) ** 2))
        final = float(np.mean((targets - self._model.value) ** 2))
        self._record(parameter_count=self._model.parameter_count, status="closed_form", iterations=0, objective_initial=initial, objective_final=final, converged=True, started=started)
        return self

    def predict(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model_not_fitted")
        return self._model.predict(state_bits, normalized_loads)


class LoadConditionedRidgeModel(Stage2Model):
    def __init__(self, *, degree: int, regularization: float, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.degree = int(degree)
        self.regularization = float(regularization)
        self._model = None

    def fit(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]], costs: Sequence[float]) -> "LoadConditionedRidgeModel":
        states, loads, targets = _fit_arrays(state_bits, normalized_loads, costs)
        started = perf_counter()
        self._model = fit_ridge_regressor(states, loads, targets, degree=self.degree, regularization=self.regularization)
        prediction = self._model.predict(states, loads)
        self._record(parameter_count=self._model.parameter_count, status="closed_form", iterations=0, objective_initial=float(np.mean((targets - targets.mean()) ** 2)), objective_final=float(np.mean((targets - prediction) ** 2)), converged=True, started=started)
        return self

    def predict(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model_not_fitted")
        return self._model.predict(state_bits, normalized_loads)


class LoadConditionedMLPModel(Stage2Model):
    """Small deterministic tanh MLP trained only on caller-supplied fit rows."""

    def __init__(self, *, hidden_units: int = 8, maxiter: int = 80, learning_rate: float = 0.03, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.hidden_units, self.maxiter, self.learning_rate = int(hidden_units), int(maxiter), float(learning_rate)
        if self.hidden_units <= 0 or self.maxiter <= 0 or self.learning_rate <= 0.0:
            raise ValueError("hidden_units, maxiter, and learning_rate must be positive")
        self._weights: tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None = None

    def fit(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]], costs: Sequence[float]) -> "LoadConditionedMLPModel":
        states, loads, targets = _fit_arrays(state_bits, normalized_loads, costs)
        started = perf_counter()
        features = np.column_stack([states, loads])
        center, scale = float(targets.mean()), float(max(targets.std(), 1.0))
        y = (targets - center) / scale
        rng = np.random.default_rng(self.random_seed)
        w1 = rng.normal(0.0, 0.15, (features.shape[1], self.hidden_units))
        b1 = np.zeros(self.hidden_units)
        w2 = rng.normal(0.0, 0.15, self.hidden_units)
        b2 = 0.0

        def objective() -> float:
            hidden = np.tanh(features @ w1 + b1)
            return float(np.mean((hidden @ w2 + b2 - y) ** 2))

        initial = objective()
        for _ in range(self.maxiter):
            hidden = np.tanh(features @ w1 + b1)
            residual = hidden @ w2 + b2 - y
            gradient_output = 2.0 * residual / len(features)
            grad_w2 = hidden.T @ gradient_output
            grad_b2 = float(gradient_output.sum())
            grad_hidden = np.outer(gradient_output, w2) * (1.0 - hidden * hidden)
            w1 -= self.learning_rate * (features.T @ grad_hidden)
            b1 -= self.learning_rate * grad_hidden.sum(axis=0)
            w2 -= self.learning_rate * grad_w2
            b2 -= self.learning_rate * grad_b2
        final = objective()
        self._weights = (w1, b1, w2, b2, center, scale)
        self._record(parameter_count=w1.size + b1.size + w2.size + 1, status="maxiter_reached", iterations=self.maxiter, objective_initial=initial, objective_final=final, converged=False, started=started)
        return self

    def predict(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]]) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("model_not_fitted")
        states, loads, _unused = _fit_arrays(state_bits, normalized_loads, np.ones(len(state_bits)))
        w1, b1, w2, b2, center, scale = self._weights
        return center + scale * (np.tanh(np.column_stack([states, loads]) @ w1 + b1) @ w2 + b2)


def _qnn_status(success: bool, iterations: int, maxiter: int) -> tuple[str, bool]:
    if success:
        return "converged", True
    if iterations >= maxiter:
        return "maxiter_reached", False
    return "optimizer_failed", False


class _ExpectationQNNModel(Stage2Model):
    readout = "all_nonempty_z_strings"

    def __init__(self, *, layers: int = 1, head_regularization: float = 0.5, theta_regularization: float = 1e-5, maxiter: int = 20, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.layers, self.head_regularization, self.theta_regularization, self.maxiter = int(layers), float(head_regularization), float(theta_regularization), int(maxiter)
        self._model = None

    def fit(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]], costs: Sequence[float]) -> "_ExpectationQNNModel":
        states, loads, targets = _fit_arrays(state_bits, normalized_loads, costs)
        started = perf_counter()
        result = fit_expectation_qnn(state_bits=states, normalized_loads=loads, costs=targets, config=ExpectationQNNConfig(layers=self.layers, head_regularization=self.head_regularization, theta_regularization=self.theta_regularization, maxiter=self.maxiter, seed=self.random_seed, readout=self.readout))
        self._model = result.model
        status, converged = _qnn_status(result.optimizer_success, result.iterations, self.maxiter)
        self._record(parameter_count=len(result.model.theta) + len(result.model.head), status=status, iterations=result.iterations, objective_initial=result.initial_loss, objective_final=result.final_loss, converged=converged, started=started, optimizer_message=result.optimizer_message)
        return self

    def predict(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model_not_fitted")
        return self._model.predict(state_bits, normalized_loads)


class SimplifiedExpectationQNNModel(_ExpectationQNNModel):
    """Trainable data-reuploading QNN with the four single-qubit Z readouts."""

    readout = "single_z"


class FullExpectationQNNModel(_ExpectationQNNModel):
    """Current 15 non-empty Z-string expectation QNN used by the first pilot."""


class RandomQuantumFeatureRidgeModel(Stage2Model):
    def __init__(self, *, layers: int = 1, circuit_seed: int = 0, regularization: float = 1e-3, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.layers, self.circuit_seed, self.regularization = int(layers), int(circuit_seed), float(regularization)
        self.quantum_parameters = tuple(float(value) for value in np.random.default_rng(self.circuit_seed).normal(0.0, 0.25, self.layers * 8))
        self._head: np.ndarray | None = None

    def fit(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]], costs: Sequence[float]) -> "RandomQuantumFeatureRidgeModel":
        states, loads, targets = _fit_arrays(state_bits, normalized_loads, costs)
        started = perf_counter()
        features = expectation_feature_matrix(states, loads, theta=self.quantum_parameters, layers=self.layers)
        design = np.column_stack([np.ones(len(features)), features])
        penalty = np.eye(design.shape[1]) * self.regularization
        penalty[0, 0] = 0.0
        self._head = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        prediction = design @ self._head
        self._record(parameter_count=len(self._head), status="closed_form_fixed_quantum_features", iterations=0, objective_initial=float(np.mean((targets - targets.mean()) ** 2)), objective_final=float(np.mean((targets - prediction) ** 2)), converged=True, started=started)
        return self

    def predict(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]]) -> np.ndarray:
        if self._head is None:
            raise RuntimeError("model_not_fitted")
        features = expectation_feature_matrix(state_bits, normalized_loads, theta=self.quantum_parameters, layers=self.layers)
        return np.column_stack([np.ones(len(features)), features]) @ self._head


class ThresholdConditionedQNNModel(Stage2Model):
    """Fit-only threshold classifier; decision cutoff is selected from validation only."""

    def __init__(self, *, layers: int = 1, theta_regularization: float = 1e-4, maxiter: int = 15, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.layers, self.theta_regularization, self.maxiter = int(layers), float(theta_regularization), int(maxiter)
        self.training_threshold = float("nan")
        self.decision_cutoff: float | None = None
        self._target_normalizer: TargetNormalizer | None = None
        self._model = None

    def fit(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]], costs: Sequence[float]) -> "ThresholdConditionedQNNModel":
        states, loads, targets = _fit_arrays(state_bits, normalized_loads, costs)
        started = perf_counter()
        self.training_threshold = float(np.median(targets))
        self._target_normalizer = TargetNormalizer(mean=float(targets.mean()), scale=float(max(targets.std(), 1.0)))
        normalized_threshold = self._target_normalizer.transform(self.training_threshold)
        result = fit_threshold_qnn(state_bits=states, normalized_loads=loads, normalized_thresholds=np.full(len(states), normalized_threshold), labels=targets <= self.training_threshold, config=ThresholdQNNConfig(layers=self.layers, theta_regularization=self.theta_regularization, maxiter=self.maxiter, seed=self.random_seed))
        self._model = result.model
        status, converged = _qnn_status(result.optimizer_success, result.iterations, self.maxiter)
        self._record(parameter_count=len(result.model.theta), status=status, iterations=result.iterations, objective_initial=result.initial_loss, objective_final=result.final_loss, converged=converged, started=started, optimizer_message=result.optimizer_message)
        return self

    def predict(self, state_bits: Sequence[Sequence[int]], normalized_loads: Sequence[Sequence[float]]) -> np.ndarray:
        if self._model is None or self._target_normalizer is None:
            raise RuntimeError("model_not_fitted")
        threshold = self._target_normalizer.transform(self.training_threshold)
        return self._model.predict_proba(state_bits, normalized_loads, np.full(len(state_bits), threshold))

    def select_decision_cutoff(self, validation_state_bits: Sequence[Sequence[int]], validation_loads: Sequence[Sequence[float]], validation_costs: Sequence[float], *, cutoffs: Sequence[float]) -> float:
        _states, _loads, costs = _fit_arrays(validation_state_bits, validation_loads, validation_costs)
        labels = costs <= self.training_threshold
        self.decision_cutoff = select_threshold_cutoff_from_validation(validation_labels=labels, validation_probabilities=self.predict(validation_state_bits, validation_loads), cutoffs=cutoffs)
        return self.decision_cutoff
