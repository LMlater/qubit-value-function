from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import stage2_case14_load_qnn_pilot_cli as stage2_cli
from qubit_value_function.experiment_utils import time_window_instance
from qubit_value_function.load_conditioned_qnn import (
    ExpectationQNNConfig,
    ThresholdQNNConfig,
    expectation_feature_matrix,
    fit_expectation_qnn,
    fit_threshold_qnn,
    threshold_probability,
)
from qubit_value_function.load_scenarios import fit_load_normalizer, scaled_load_instance
from qubit_value_function.uc_loader import load_uc_instance

ROOT = Path(__file__).resolve().parents[1]


def test_scaled_load_instance_preserves_source_and_other_data() -> None:
    source = load_uc_instance(ROOT / "data" / "case14.json.gz")
    window = time_window_instance(source, start=0, horizon=2)
    scaled = scaled_load_instance(window, 1.1)

    assert scaled.fixed_load == [1.1 * value for value in window.fixed_load]
    assert window.fixed_load != scaled.fixed_load
    assert scaled.generators == window.generators
    assert scaled.reserves == window.reserves


def test_load_normalizer_uses_training_vectors_only() -> None:
    normalizer = fit_load_normalizer([[10.0, 20.0], [20.0, 40.0]])
    first = normalizer.transform([10.0, 20.0])
    second = normalizer.transform([20.0, 40.0])

    assert np.allclose(first, [-1.0, -1.0])
    assert np.allclose(second, [1.0, 1.0])


def test_expectation_features_are_deterministic_and_bounded() -> None:
    bits = np.array([[0, 0, 0, 0], [1, 0, 1, 0]], dtype=int)
    loads = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=float)
    theta = np.linspace(-0.2, 0.2, 16)

    first = expectation_feature_matrix(bits, loads, theta=theta, layers=2)
    second = expectation_feature_matrix(bits, loads, theta=theta, layers=2)

    assert first.shape == (2, 15)
    assert np.allclose(first, second)
    assert np.max(np.abs(first)) <= 1.0 + 1e-12


def test_expectation_qnn_fit_does_not_increase_training_objective() -> None:
    bits = np.array(
        [[(index >> qubit) & 1 for qubit in range(4)] for index in range(8)],
        dtype=int,
    )
    loads = np.array([[(-1.0 + 2.0 * (index % 3) / 2.0)] * 2 for index in range(8)])
    costs = 100.0 + 10.0 * bits[:, 0] + 20.0 * bits[:, 1] + 5.0 * loads[:, 0]

    fit = fit_expectation_qnn(
        state_bits=bits,
        normalized_loads=loads,
        costs=costs,
        config=ExpectationQNNConfig(layers=1, maxiter=3, seed=3),
    )
    prediction = fit.model.predict(bits, loads)

    assert fit.final_loss <= fit.initial_loss + 1e-12
    assert prediction.shape == costs.shape
    assert np.all(np.isfinite(prediction))


def test_threshold_qnn_probabilities_and_fit_are_valid() -> None:
    bits = np.array(
        [[(index >> qubit) & 1 for qubit in range(4)] for index in range(8)],
        dtype=int,
    )
    loads = np.zeros((8, 2), dtype=float)
    thresholds = np.zeros(8, dtype=float)
    labels = (bits[:, 0] == 1).astype(int)

    initial_probability = threshold_probability(bits[0], loads[0], 0.0, np.zeros(8), 1)
    fit = fit_threshold_qnn(
        state_bits=bits,
        normalized_loads=loads,
        normalized_thresholds=thresholds,
        labels=labels,
        config=ThresholdQNNConfig(layers=1, maxiter=3, seed=2),
    )
    probabilities = fit.model.predict_proba(bits, loads, thresholds)

    assert 0.0 < initial_probability < 1.0
    assert fit.final_loss <= fit.initial_loss + 1e-12
    assert probabilities.shape == (8,)
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))


def test_training_quantile_thresholds_exclude_unseen_state_truth() -> None:
    rows = [
        {"state_index": 0, "true_cost": 10.0},
        {"state_index": 1, "true_cost": 20.0},
        {"state_index": 2, "true_cost": -1000.0},
        {"state_index": 3, "true_cost": 1000.0},
    ]

    assert hasattr(stage2_cli, "training_quantile_thresholds")
    thresholds = stage2_cli.training_quantile_thresholds(rows, frozenset({0, 1}))

    assert np.allclose(thresholds, np.quantile([10.0, 20.0], (0.2, 0.4, 0.6, 0.8)))


def test_confusion_metrics_reports_f1() -> None:
    metrics = stage2_cli.confusion_metrics([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2], 0.5)

    assert metrics["f1"] == 0.5
