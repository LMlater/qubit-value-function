from __future__ import annotations

import numpy as np


def phase_oracle_matrix(marked: np.ndarray) -> np.ndarray:
    marked = np.asarray(marked, dtype=bool)
    diagonal = np.ones(marked.size, dtype=float)
    diagonal[marked] = -1.0
    return np.diag(diagonal)


def verify_phase_oracle(oracle: np.ndarray, atol: float = 1e-10) -> dict[str, bool]:
    identity = np.eye(oracle.shape[0])
    return {
        "unitary": bool(np.allclose(oracle.T.conj() @ oracle, identity, atol=atol)),
        "self_inverse": bool(np.allclose(oracle @ oracle, identity, atol=atol)),
        "real_diagonal": bool(np.allclose(oracle, np.diag(np.diag(oracle)), atol=atol)),
    }


def phase_oracle_errors(oracle: np.ndarray) -> dict[str, float]:
    identity = np.eye(oracle.shape[0])
    return {
        "unitarity_error": float(np.max(np.abs(oracle.T.conj() @ oracle - identity))),
        "self_inverse_error": float(np.max(np.abs(oracle @ oracle - identity))),
        "diagonal_error": float(np.max(np.abs(oracle - np.diag(np.diag(oracle))))),
    }


def grover_search_probabilities(marked: np.ndarray, iterations: int | None = None) -> dict[str, object]:
    marked = np.asarray(marked, dtype=bool)
    dimension = marked.size
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("number of states must be a power of two")
    count = int(marked.sum())
    if count == 0:
        raise ValueError("Grover search needs at least one marked state")

    if iterations is None:
        iterations = max(1, int(np.floor(np.pi / 4.0 * np.sqrt(dimension / count))))

    state = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
    oracle = phase_oracle_matrix(marked)
    uniform = np.ones((dimension, 1), dtype=complex) / np.sqrt(dimension)
    diffuser = 2 * (uniform @ uniform.T.conj()) - np.eye(dimension)
    for _ in range(iterations):
        state = diffuser @ (oracle @ state)

    probabilities = np.abs(state) ** 2
    return {
        "iterations": iterations,
        "marked_probability": float(probabilities[marked].sum()),
        "unmarked_probability": float(probabilities[~marked].sum()),
        "probabilities": [float(v) for v in probabilities],
    }


def grover_with_oracle_matrix(
    oracle: np.ndarray,
    target_marked: np.ndarray,
    iterations: int | None = None,
) -> dict[str, object]:
    target_marked = np.asarray(target_marked, dtype=bool)
    dimension = target_marked.size
    if oracle.shape != (dimension, dimension):
        raise ValueError("oracle shape does not match the marked-state mask")
    count = int(target_marked.sum())
    if count == 0:
        raise ValueError("Grover search needs at least one target state")
    if iterations is None:
        iterations = max(1, int(np.floor(np.pi / 4.0 * np.sqrt(dimension / count))))

    state = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
    uniform = np.ones((dimension, 1), dtype=complex) / np.sqrt(dimension)
    diffuser = 2 * (uniform @ uniform.T.conj()) - np.eye(dimension)
    for _ in range(iterations):
        state = diffuser @ (oracle @ state)

    probabilities = np.abs(state) ** 2
    return {
        "iterations": iterations,
        "target_probability": float(probabilities[target_marked].sum()),
        "non_target_probability": float(probabilities[~target_marked].sum()),
        "probabilities": [float(v) for v in probabilities],
    }
