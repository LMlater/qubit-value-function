from __future__ import annotations

from pathlib import Path

import numpy as np

from qubit_value_function.experiment_utils import (
    leading_time_window_instance,
    time_window_instance,
)
from qubit_value_function.sparse_phase_vqc import (
    SparsePhaseVQC,
    basis_phase_factor_from_statevector,
    build_local_phase_features,
    build_phase_circuit,
    fit_sparse_phase_vqc,
    hadamard_phase_expectation,
)
from qubit_value_function.uc_loader import load_uc_instance


ROOT = Path(__file__).resolve().parents[1]


def _project_bitstring(index: int, num_qubits: int) -> str:
    return "".join(str((index >> qubit) & 1) for qubit in range(num_qubits))


def test_time_window_instance_supports_shifted_independent_scenarios() -> None:
    source = load_uc_instance(ROOT / "data" / "case14.json.gz")
    shifted = time_window_instance(source, start=1, horizon=2)
    leading = leading_time_window_instance(source, horizon=2)

    assert shifted.time_horizon == 2
    assert shifted.fixed_load == source.fixed_load[1:3]
    assert shifted.power_balance_penalty == source.power_balance_penalty[1:3]
    assert leading.fixed_load == source.fixed_load[:2]
    assert shifted.generators[0].initial_status == source.generators[0].initial_status
    assert shifted.generators[0].initial_power == source.generators[0].initial_power
    assert len(shifted.reserves) == len(source.reserves)
    for shifted_reserve, source_reserve in zip(shifted.reserves, source.reserves):
        assert shifted_reserve.name == source_reserve.name
        assert shifted_reserve.amount == source_reserve.amount[1:3]
        assert shifted_reserve.penalty == source_reserve.penalty[1:3]


def test_sparse_feature_count_scales_with_local_graph_not_full_state_space() -> None:
    features_2x2 = build_local_phase_features(2, 2)
    features_3x2 = build_local_phase_features(3, 2)
    features_3x3 = build_local_phase_features(3, 3)

    assert len(features_2x2) == 8
    assert len(features_3x2) == 13
    assert len(features_3x3) == 21

    model = SparsePhaseVQC(
        num_generators=3,
        num_periods=3,
        features=features_3x3,
        phase_intercept=0.1,
        phase_weights=tuple(0.01 for _ in features_3x3),
        cost_center=100.0,
        cost_scale=20.0,
    )
    circuit = build_phase_circuit(model)
    assert circuit.num_qubits == 9
    assert len(circuit.data) == len(features_3x3)


def test_sparse_phase_training_reduces_exact_phase_loss_on_synthetic_costs() -> None:
    bitstrings = [_project_bitstring(index, 4) for index in range(16)]
    features = build_local_phase_features(2, 2, generator_edges=((0, 1),))
    true_model = SparsePhaseVQC(
        num_generators=2,
        num_periods=2,
        features=features,
        phase_intercept=0.16,
        phase_weights=(0.12, -0.09, 0.03, 0.06, 0.05, -0.04, 0.02, 0.015),
        cost_center=1000.0,
        cost_scale=500.0,
        phase_center=0.25,
        phase_scale=0.20,
        generator_edges=((0, 1),),
    )
    costs = [true_model.predict_cost(bits) for bits in bitstrings]

    fit = fit_sparse_phase_vqc(
        bitstrings=bitstrings,
        costs=costs,
        num_generators=2,
        num_periods=2,
        generator_edges=((0, 1),),
        seed=7,
        regularization=1e-6,
        maxiter=500,
    )
    predicted = fit.model.predict_costs(bitstrings)

    assert fit.final_loss < fit.initial_loss * 1e-3
    assert np.mean(np.abs(predicted - np.asarray(costs))) < 0.1


def test_basis_and_hadamard_phase_encoding_match_analytic_model() -> None:
    features = build_local_phase_features(2, 2, generator_edges=((0, 1),))
    model = SparsePhaseVQC(
        num_generators=2,
        num_periods=2,
        features=features,
        phase_intercept=0.13,
        phase_weights=(0.07, -0.03, 0.05, 0.02, 0.04, -0.06, 0.01, 0.025),
        cost_center=100.0,
        cost_scale=10.0,
        generator_edges=((0, 1),),
    )

    for index in range(16):
        bits = _project_bitstring(index, 4)
        expected = model.phase_factor(bits)
        basis = basis_phase_factor_from_statevector(model, bits)
        hadamard = hadamard_phase_expectation(model, bits)
        assert abs(expected - basis) < 1e-12
        assert abs(expected - hadamard) < 1e-12
