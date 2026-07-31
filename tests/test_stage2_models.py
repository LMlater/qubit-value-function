from __future__ import annotations

import math

import numpy as np

from qubit_value_function.stage2_models import (
    ConstantBaselineModel,
    FullExpectationQNNModel,
    LoadConditionedMLPModel,
    LoadConditionedRidgeModel,
    RandomQuantumFeatureRidgeModel,
    SimplifiedExpectationQNNModel,
    ThresholdConditionedQNNModel,
)


def _training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(
        [
            [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
            [1, 1, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0],
            [0, 1, 1, 0], [1, 1, 1, 0],
        ],
        dtype=int,
    )
    loads = np.asarray(
        [[-1.0, -1.0], [-1.0, -0.2], [-0.4, 0.2], [-0.4, 1.0], [0.2, -1.0], [0.2, -0.2], [1.0, 0.2], [1.0, 1.0]],
        dtype=float,
    )
    costs = np.asarray([10.0, 12.0, 15.0, 17.0, 19.0, 22.0, 24.0, 27.0])
    return states, loads, costs


def _models():
    return (
        ConstantBaselineModel(seed=3),
        LoadConditionedRidgeModel(degree=1, regularization=1e-3, seed=3),
        LoadConditionedRidgeModel(degree=2, regularization=1e-3, seed=3),
        LoadConditionedMLPModel(hidden_units=4, maxiter=8, seed=3),
        SimplifiedExpectationQNNModel(layers=1, maxiter=1, seed=3),
        FullExpectationQNNModel(layers=1, maxiter=1, seed=3),
        RandomQuantumFeatureRidgeModel(layers=1, circuit_seed=11, seed=3),
        ThresholdConditionedQNNModel(layers=1, maxiter=1, seed=3),
    )


def test_every_stage2_model_exposes_a_finite_uniform_fit_report_and_prediction() -> None:
    states, loads, costs = _training_data()

    for model in _models():
        model.fit(states, loads, costs)
        prediction = model.predict(states, loads)

        assert isinstance(model.parameter_count, int)
        assert model.parameter_count >= 0
        assert isinstance(model.fit_status, str) and model.fit_status
        assert isinstance(model.optimizer_message, str) and model.optimizer_message
        assert isinstance(model.iteration_count, int) and model.iteration_count >= 0
        assert isinstance(model.converged, bool)
        assert model.random_seed == 3
        assert len(prediction) == len(costs)
        assert all(math.isfinite(float(value)) for value in prediction)
        for value in (model.objective_initial, model.objective_final, model.runtime_seconds):
            assert math.isfinite(float(value))
        assert model.runtime_seconds >= 0.0


def test_same_seed_is_reproducible_and_simplified_qnn_has_fewer_trainable_parameters() -> None:
    states, loads, costs = _training_data()
    first = SimplifiedExpectationQNNModel(layers=1, maxiter=1, seed=3).fit(states, loads, costs)
    second = SimplifiedExpectationQNNModel(layers=1, maxiter=1, seed=3).fit(states, loads, costs)
    full = FullExpectationQNNModel(layers=1, maxiter=1, seed=3).fit(states, loads, costs)

    assert np.array_equal(first.predict(states, loads), second.predict(states, loads))
    assert first.parameter_count < full.parameter_count
    assert first._model.as_dict()["readout"] == "single_z"
    assert full._model.as_dict()["readout"] == "all_nonempty_z_strings"


def test_random_quantum_feature_circuit_parameters_remain_fixed_during_ridge_fit() -> None:
    states, loads, costs = _training_data()
    model = RandomQuantumFeatureRidgeModel(layers=1, circuit_seed=11, seed=3)
    before = model.quantum_parameters
    model.fit(states, loads, costs)

    assert model.quantum_parameters == before
    assert model.circuit_seed == 11


def test_qnn_nonconvergence_is_preserved_honestly() -> None:
    states, loads, costs = _training_data()
    model = FullExpectationQNNModel(layers=1, maxiter=1, seed=3).fit(states, loads, costs)

    assert model.fit_status in {"converged", "maxiter_reached", "optimizer_failed"}
    if not model.converged:
        assert model.fit_status != "converged"
