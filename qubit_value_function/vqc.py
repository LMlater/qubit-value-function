from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VqcTrainingResult:
    predictions: np.ndarray
    mae: float
    rmse: float
    max_abs_error: float
    normalized_loss: float
    steps: int
    seed: int


def train_vqc_value_function(
    bits: np.ndarray,
    values: np.ndarray,
    *,
    steps: int = 800,
    seed: int = 7,
    learning_rate: float = 0.08,
) -> VqcTrainingResult:
    """Train a small VQC regressor for V_d(x).

    The circuit uses binary basis encoding, variational single-qubit rotations,
    a CNOT ring, probability readout, and a trainable linear head. The benchmark
    is intentionally tiny so the first-stage oracle can be exhaustively checked.
    """

    import pennylane as qml
    from pennylane import numpy as pnp

    rng = np.random.default_rng(seed)
    x_np = np.asarray(bits, dtype=float)
    y_np = np.asarray(values, dtype=float)
    y_min = float(y_np.min())
    y_range = float(y_np.max() - y_min)
    if y_range <= 0:
        raise ValueError("training values must not be constant")
    y_scaled = (y_np - y_min) / y_range

    num_qubits = x_np.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x_row: np.ndarray, weights: pnp.ndarray) -> pnp.ndarray:
        for wire in range(num_qubits):
            qml.RY(np.pi * x_row[wire], wires=wire)
        for wire in range(num_qubits):
            qml.RY(weights[wire, 0], wires=wire)
            qml.RZ(weights[wire, 1], wires=wire)
        for wire in range(num_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])
        if num_qubits > 1:
            qml.CNOT(wires=[num_qubits - 1, 0])
        for wire in range(num_qubits):
            qml.RY(weights[wire, 2], wires=wire)
        return qml.probs(wires=range(num_qubits))

    x = x_np
    y = pnp.array(y_scaled, requires_grad=False)
    initial_weights = np.zeros((num_qubits, 3), dtype=float)
    initial_features = np.vstack(
        [np.asarray(circuit(row, pnp.array(initial_weights)), dtype=float) for row in x]
    )
    design = np.column_stack([initial_features, np.ones(initial_features.shape[0])])
    head, *_ = np.linalg.lstsq(design, y_scaled, rcond=None)
    initial_params = np.concatenate(
        [
            initial_weights.reshape(-1),
            head[:-1],
            np.array([head[-1]], dtype=float),
        ]
    )
    params = pnp.array(initial_params, requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=learning_rate)

    def unpack(params_vector: pnp.ndarray) -> tuple[pnp.ndarray, pnp.ndarray, pnp.ndarray]:
        split_1 = num_qubits * 3
        split_2 = split_1 + 2**num_qubits
        q_weights = pnp.reshape(params_vector[:split_1], (num_qubits, 3))
        q_linear = params_vector[split_1:split_2]
        q_bias = params_vector[split_2]
        return q_weights, q_linear, q_bias

    def forward(params_vector: pnp.ndarray) -> pnp.ndarray:
        q_weights, q_linear, q_bias = unpack(params_vector)
        probs = [circuit(row, q_weights) for row in x]
        return pnp.stack(probs) @ q_linear + q_bias

    def loss_fn(params_vector: pnp.ndarray) -> pnp.ndarray:
        pred = forward(params_vector)
        return pnp.mean((pred - y) ** 2)

    best_predictions = np.asarray(forward(params), dtype=float)
    best_loss = float(loss_fn(params))
    for _ in range(steps):
        params = optimizer.step(loss_fn, params)
        value = float(loss_fn(params))
        if value < best_loss:
            best_loss = value
            best_predictions = np.asarray(forward(params), dtype=float)

    predictions = best_predictions * y_range + y_min
    errors = predictions - y_np
    return VqcTrainingResult(
        predictions=np.asarray(predictions, dtype=float),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        max_abs_error=float(np.max(np.abs(errors))),
        normalized_loss=float(best_loss),
        steps=steps,
        seed=seed,
    )
