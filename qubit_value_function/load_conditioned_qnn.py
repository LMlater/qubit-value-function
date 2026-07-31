from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ExpectationQNNConfig:
    layers: int = 2
    head_regularization: float = 0.5
    theta_regularization: float = 1e-5
    maxiter: int = 20
    seed: int = 0
    readout: str = "all_nonempty_z_strings"

    def __post_init__(self) -> None:
        if int(self.layers) <= 0:
            raise ValueError("layers must be positive")
        if float(self.head_regularization) < 0.0:
            raise ValueError("head_regularization must be nonnegative")
        if float(self.theta_regularization) < 0.0:
            raise ValueError("theta_regularization must be nonnegative")
        if int(self.maxiter) <= 0:
            raise ValueError("maxiter must be positive")
        if self.readout not in {"all_nonempty_z_strings", "single_z"}:
            raise ValueError("readout must be all_nonempty_z_strings or single_z")


@dataclass(frozen=True)
class ThresholdQNNConfig:
    layers: int = 2
    theta_regularization: float = 1e-4
    maxiter: int = 15
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.layers) <= 0:
            raise ValueError("layers must be positive")
        if float(self.theta_regularization) < 0.0:
            raise ValueError("theta_regularization must be nonnegative")
        if int(self.maxiter) <= 0:
            raise ValueError("maxiter must be positive")


@dataclass(frozen=True)
class ExpectationQNNRegressor:
    """Exact-statevector data-reuploading QNN with a linear classical readout.

    The circuit contains non-commuting data rotations, trainable rotations, and
    a CNOT ring. It is a conventional expectation-value QNN diagnostic rather
    than the directly compilable diagonal phase surrogate used in Stage A.
    """

    num_state_qubits: int
    num_load_features: int
    layers: int
    theta: tuple[float, ...]
    head: tuple[float, ...]
    target_center: float
    target_scale: float
    readout: str = "all_nonempty_z_strings"

    def predict(
        self,
        state_bits: Sequence[Sequence[int]],
        normalized_loads: Sequence[Sequence[float]],
    ) -> np.ndarray:
        features = expectation_feature_matrix(
            state_bits,
            normalized_loads,
            theta=self.theta,
            layers=self.layers,
            readout=self.readout,
        )
        design = np.column_stack([np.ones(len(features), dtype=float), features])
        normalized = design @ np.asarray(self.head, dtype=float)
        return self.target_center + self.target_scale * normalized

    def as_dict(self) -> dict[str, object]:
        return {
            "model_type": "exact_statevector_expectation_qnn_regressor",
            "num_state_qubits": int(self.num_state_qubits),
            "num_load_features": int(self.num_load_features),
            "layers": int(self.layers),
            "theta": list(self.theta),
            "head": list(self.head),
            "target_center": float(self.target_center),
            "target_scale": float(self.target_scale),
            "readout": self.readout,
        }


@dataclass(frozen=True)
class ThresholdQNNClassifier:
    """Threshold-conditioned soft classifier using one output-qubit probability."""

    num_state_qubits: int
    num_load_features: int
    layers: int
    theta: tuple[float, ...]

    def predict_proba(
        self,
        state_bits: Sequence[Sequence[int]],
        normalized_loads: Sequence[Sequence[float]],
        normalized_thresholds: Sequence[float],
    ) -> np.ndarray:
        return np.array(
            [
                threshold_probability(bits, load, threshold, self.theta, self.layers)
                for bits, load, threshold in zip(
                    state_bits, normalized_loads, normalized_thresholds
                )
            ],
            dtype=float,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "model_type": "exact_statevector_threshold_qnn_classifier",
            "num_state_qubits": int(self.num_state_qubits),
            "num_load_features": int(self.num_load_features),
            "layers": int(self.layers),
            "theta": list(self.theta),
            "output": "probability_of_one_on_qubit_0",
        }


@dataclass(frozen=True)
class QNNFitResult:
    model: ExpectationQNNRegressor | ThresholdQNNClassifier
    initial_loss: float
    final_loss: float
    optimizer_success: bool
    optimizer_message: str
    iterations: int
    function_evaluations: int

    def as_dict(self) -> dict[str, object]:
        return {
            "initial_loss": float(self.initial_loss),
            "final_loss": float(self.final_loss),
            "optimizer_success": bool(self.optimizer_success),
            "optimizer_message": str(self.optimizer_message),
            "iterations": int(self.iterations),
            "function_evaluations": int(self.function_evaluations),
            "model": self.model.as_dict(),
        }


def fit_expectation_qnn(
    *,
    state_bits: Sequence[Sequence[int]],
    normalized_loads: Sequence[Sequence[float]],
    costs: Sequence[float],
    config: ExpectationQNNConfig = ExpectationQNNConfig(),
) -> QNNFitResult:
    bits, loads = _validate_inputs(state_bits, normalized_loads)
    targets = np.asarray(costs, dtype=float)
    if targets.shape != (len(bits),) or not np.all(np.isfinite(targets)):
        raise ValueError("costs must be one finite value per sample")

    center = float(targets.mean())
    scale = float(max(targets.std(), 1.0))
    normalized_targets = (targets - center) / scale
    num_parameters = int(config.layers) * bits.shape[1] * 2
    initial = np.random.default_rng(int(config.seed)).normal(0.0, 0.25, num_parameters)

    def solve_head(features: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(features), dtype=float), features])
        regularizer = np.eye(design.shape[1], dtype=float) * float(
            config.head_regularization
        )
        regularizer[0, 0] = 0.0
        return np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ normalized_targets,
        )

    def objective(theta: np.ndarray) -> float:
        features = expectation_feature_matrix(
            bits, loads, theta=theta, layers=int(config.layers), readout=config.readout
        )
        head = solve_head(features)
        design = np.column_stack([np.ones(len(features), dtype=float), features])
        residual = design @ head - normalized_targets
        return float(
            np.mean(residual * residual)
            + float(config.theta_regularization) * np.mean(theta * theta)
        )

    initial_loss = objective(initial)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-np.pi, np.pi)] * num_parameters,
        options={
            "maxiter": int(config.maxiter),
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxls": 20,
        },
    )
    final_theta = np.asarray(result.x, dtype=float)
    final_features = expectation_feature_matrix(
        bits, loads, theta=final_theta, layers=int(config.layers), readout=config.readout
    )
    final_head = solve_head(final_features)
    model = ExpectationQNNRegressor(
        num_state_qubits=int(bits.shape[1]),
        num_load_features=int(loads.shape[1]),
        layers=int(config.layers),
        theta=tuple(float(value) for value in final_theta),
        head=tuple(float(value) for value in final_head),
        target_center=center,
        target_scale=scale,
        readout=config.readout,
    )
    return QNNFitResult(
        model=model,
        initial_loss=float(initial_loss),
        final_loss=float(objective(final_theta)),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        function_evaluations=int(getattr(result, "nfev", 0)),
    )


def fit_threshold_qnn(
    *,
    state_bits: Sequence[Sequence[int]],
    normalized_loads: Sequence[Sequence[float]],
    normalized_thresholds: Sequence[float],
    labels: Sequence[int | bool],
    config: ThresholdQNNConfig = ThresholdQNNConfig(),
) -> QNNFitResult:
    bits, loads = _validate_inputs(state_bits, normalized_loads)
    thresholds = np.asarray(normalized_thresholds, dtype=float)
    targets = np.asarray(labels, dtype=float)
    if thresholds.shape != (len(bits),) or not np.all(np.isfinite(thresholds)):
        raise ValueError("normalized_thresholds must be finite and aligned")
    if targets.shape != (len(bits),) or not np.all(np.isin(targets, [0.0, 1.0])):
        raise ValueError("labels must contain one binary value per sample")

    num_parameters = int(config.layers) * bits.shape[1] * 2
    initial = np.random.default_rng(int(config.seed)).normal(0.0, 0.25, num_parameters)

    def objective(theta: np.ndarray) -> float:
        probabilities = np.array(
            [
                threshold_probability(row, load, threshold, theta, int(config.layers))
                for row, load, threshold in zip(bits, loads, thresholds)
            ],
            dtype=float,
        )
        data_loss = -np.mean(
            targets * np.log(probabilities)
            + (1.0 - targets) * np.log(1.0 - probabilities)
        )
        return float(
            data_loss
            + float(config.theta_regularization) * np.mean(theta * theta)
        )

    initial_loss = objective(initial)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-np.pi, np.pi)] * num_parameters,
        options={
            "maxiter": int(config.maxiter),
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxls": 20,
        },
    )
    final_theta = np.asarray(result.x, dtype=float)
    model = ThresholdQNNClassifier(
        num_state_qubits=int(bits.shape[1]),
        num_load_features=int(loads.shape[1]),
        layers=int(config.layers),
        theta=tuple(float(value) for value in final_theta),
    )
    return QNNFitResult(
        model=model,
        initial_loss=float(initial_loss),
        final_loss=float(objective(final_theta)),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        function_evaluations=int(getattr(result, "nfev", 0)),
    )


def expectation_feature_matrix(
    state_bits: Sequence[Sequence[int]],
    normalized_loads: Sequence[Sequence[float]],
    *,
    theta: Sequence[float],
    layers: int,
    readout: str = "all_nonempty_z_strings",
) -> np.ndarray:
    bits, loads = _validate_inputs(state_bits, normalized_loads)
    if readout not in {"all_nonempty_z_strings", "single_z"}:
        raise ValueError("readout must be all_nonempty_z_strings or single_z")
    features = np.vstack(
        [
            _expectation_features(row, load, theta, int(layers))
            for row, load in zip(bits, loads)
        ]
    )
    return features[:, : bits.shape[1]] if readout == "single_z" else features


def threshold_probability(
    state_bits: Sequence[int],
    normalized_load: Sequence[float],
    normalized_threshold: float,
    theta: Sequence[float],
    layers: int,
) -> float:
    bits = np.asarray(state_bits, dtype=float)
    loads = np.asarray(normalized_load, dtype=float)
    if bits.ndim != 1 or loads.ndim != 1 or len(bits) != 4 or len(loads) != 2:
        raise ValueError("Stage-B pilot expects four state bits and two load values")
    state = np.zeros(2 ** len(bits), dtype=complex)
    state[0] = 1.0
    parameters = np.asarray(theta, dtype=float).reshape(int(layers), len(bits), 2)
    load_values = np.tanh(loads)
    threshold_value = float(np.tanh(float(normalized_threshold)))
    for layer in range(int(layers)):
        for qubit in range(len(bits)):
            time_index = qubit % len(load_values)
            state_value = float(bits[qubit])
            load_value = float(load_values[time_index])
            threshold_sign = 1.0 if qubit < len(bits) // 2 else -1.0
            state = _apply_single(
                state,
                _ry(np.pi * (0.50 * state_value + 0.20 * load_value + 0.15 * threshold_value)),
                qubit,
            )
            state = _apply_single(
                state,
                _rz(
                    np.pi
                    * (
                        0.50 * state_value
                        - 0.15 * load_value
                        + threshold_sign * 0.20 * threshold_value
                    )
                ),
                qubit,
            )
            state = _apply_single(state, _ry(parameters[layer, qubit, 0]), qubit)
            state = _apply_single(state, _rz(parameters[layer, qubit, 1]), qubit)
        state = _apply_cnot_ring(state)
    probability = (1.0 - _z_expectation(state, 0)) / 2.0
    return float(np.clip(probability, 1e-8, 1.0 - 1e-8))


def _expectation_features(
    state_bits: Sequence[int],
    normalized_load: Sequence[float],
    theta: Sequence[float],
    layers: int,
) -> np.ndarray:
    bits = np.asarray(state_bits, dtype=float)
    loads = np.asarray(normalized_load, dtype=float)
    if bits.ndim != 1 or loads.ndim != 1 or len(bits) != 4 or len(loads) != 2:
        raise ValueError("Stage-B pilot expects four state bits and two load values")
    state = np.zeros(2 ** len(bits), dtype=complex)
    state[0] = 1.0
    parameters = np.asarray(theta, dtype=float).reshape(int(layers), len(bits), 2)
    load_values = np.tanh(loads)
    for layer in range(int(layers)):
        for qubit in range(len(bits)):
            time_index = qubit % len(load_values)
            state_value = float(bits[qubit])
            load_value = float(load_values[time_index])
            state = _apply_single(
                state,
                _ry(np.pi * (0.50 * state_value + 0.25 * load_value)),
                qubit,
            )
            state = _apply_single(
                state,
                _rz(np.pi * (0.50 * state_value - 0.25 * load_value)),
                qubit,
            )
            state = _apply_single(state, _ry(parameters[layer, qubit, 0]), qubit)
            state = _apply_single(state, _rz(parameters[layer, qubit, 1]), qubit)
        state = _apply_cnot_ring(state)
    return _all_nonempty_z_string_expectations(state)


def _validate_inputs(
    state_bits: Sequence[Sequence[int]],
    normalized_loads: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    bits = np.asarray(state_bits, dtype=int)
    loads = np.asarray(normalized_loads, dtype=float)
    if bits.ndim != 2 or bits.shape[1] != 4 or not np.all(np.isin(bits, [0, 1])):
        raise ValueError("state_bits must have shape (n, 4) and be binary")
    if loads.ndim != 2 or loads.shape != (len(bits), 2):
        raise ValueError("normalized_loads must have shape (n, 2)")
    if not np.all(np.isfinite(loads)):
        raise ValueError("normalized_loads must contain only finite values")
    return bits, loads


def _ry(angle: float) -> np.ndarray:
    cosine = np.cos(float(angle) / 2.0)
    sine = np.sin(float(angle) / 2.0)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=complex)


def _rz(angle: float) -> np.ndarray:
    value = float(angle) / 2.0
    return np.diag([np.exp(-1.0j * value), np.exp(1.0j * value)])


def _apply_single(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    num_qubits = int(round(np.log2(len(state))))
    tensor = state.reshape([2] * num_qubits)
    tensor = np.moveaxis(tensor, int(qubit), 0).reshape(2, -1)
    tensor = (gate @ tensor).reshape([2] + [2] * (num_qubits - 1))
    return np.moveaxis(tensor, 0, int(qubit)).reshape(-1)


def _apply_cnot_ring(state: np.ndarray) -> np.ndarray:
    output = state
    num_qubits = int(round(np.log2(len(state))))
    for control, target in tuple((q, (q + 1) % num_qubits) for q in range(num_qubits)):
        next_state = np.zeros_like(output)
        for index, amplitude in enumerate(output):
            control_bit = (index >> (num_qubits - 1 - control)) & 1
            mapped = index
            if control_bit:
                mapped ^= 1 << (num_qubits - 1 - target)
            next_state[mapped] += amplitude
        output = next_state
    return output


def _z_expectation(state: np.ndarray, qubit: int) -> float:
    num_qubits = int(round(np.log2(len(state))))
    probabilities = np.abs(state) ** 2
    return float(
        sum(
            (1.0 if ((index >> (num_qubits - 1 - qubit)) & 1) == 0 else -1.0)
            * probability
            for index, probability in enumerate(probabilities)
        )
    )


def _all_nonempty_z_string_expectations(state: np.ndarray) -> np.ndarray:
    num_qubits = int(round(np.log2(len(state))))
    probabilities = np.abs(state) ** 2
    output: list[float] = []
    for mask in range(1, 2**num_qubits):
        expectation = 0.0
        for index, probability in enumerate(probabilities):
            sign = 1.0
            for qubit in range(num_qubits):
                if (mask >> qubit) & 1:
                    bit = (index >> (num_qubits - 1 - qubit)) & 1
                    sign *= 1.0 if bit == 0 else -1.0
            expectation += sign * probability
        output.append(float(expectation))
    return np.asarray(output, dtype=float)
