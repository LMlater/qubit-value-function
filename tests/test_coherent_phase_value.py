from __future__ import annotations

from pathlib import Path

import numpy as np

from qubit_value_function.commitment import is_logic_feasible
from qubit_value_function.coherent_phase_value import (
    QuantizedSparseValueModel,
    basis_value_code_probe,
    build_phase_to_value_circuit,
    build_sparse_vqc_threshold_phase_oracle,
    conservative_integer_bounds,
    phase_to_value_superposition_probe,
    quantize_sparse_phase_model,
    simulate_sparse_vqc_threshold_phase_oracle,
)
from qubit_value_function.ed import FixedCommitmentEvaluator
from qubit_value_function.experiment_utils import (
    embedded_selected_commitments,
    time_window_instance,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig
from qubit_value_function.gate_level_oracle import bitstring_from_index
from qubit_value_function.sparse_phase_vqc import (
    SparsePhaseVQC,
    build_local_phase_features,
    fit_sparse_phase_vqc,
)
from qubit_value_function.uc_loader import load_uc_instance


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_INDEX_ORDER = (0, 15, 3, 12, 5, 10, 6, 9, 1, 2, 4, 8, 7, 11, 13, 14)


def _integer_model() -> QuantizedSparseValueModel:
    features = build_local_phase_features(2, 2, generator_edges=((0, 1),))
    integer_weights = (1, -2, 3, 0, 1, -1, 2, -2)
    lower, upper = conservative_integer_bounds(2, integer_weights)
    shift = -lower
    shifted_upper = upper + shift
    return QuantizedSparseValueModel(
        num_generators=2,
        num_periods=2,
        features=features,
        fixed_point_config=FixedPointConfig(fractional_bits=0, unit=1.0),
        real_intercept=2.0,
        real_weights=tuple(float(value) for value in integer_weights),
        integer_intercept=2,
        integer_weights=integer_weights,
        coefficient_quantization_errors=(0.0,) * 9,
        lower_bound=lower,
        upper_bound=upper,
        value_shift=shift,
        shifted_upper_bound=shifted_upper,
        num_value_qubits=max(1, int(shifted_upper).bit_length()),
    )


def test_conservative_bounds_and_shift_handle_signed_weights_without_enumeration() -> None:
    model = _integer_model()
    assert model.lower_bound == -3
    assert model.upper_bound == 9
    assert model.value_shift == 3
    assert model.shifted_upper_bound == 12
    assert model.num_value_qubits == 4
    assert model.shifted_upper_bound < model.phase_modulus


def test_all_2x2_basis_states_encode_the_exact_shifted_integer_value() -> None:
    model = _integer_model()
    for index in range(16):
        bitstring = bitstring_from_index(index, 4)
        probe = basis_value_code_probe(model, bitstring)
        assert probe.expected_code == model.shifted_integer_value(bitstring)
        assert probe.most_likely_code == probe.expected_code
        assert probe.correct_code_probability > 1.0 - 1e-12


def test_superposition_pairing_and_inverse_uncompute_are_exact() -> None:
    probe = phase_to_value_superposition_probe(_integer_model())
    assert probe.pairing_probability > 1.0 - 1e-12
    assert probe.inverse_auxiliary_zero_probability > 1.0 - 1e-12


def test_threshold_phase_oracle_matches_signed_integer_model_and_uncomputes() -> None:
    model = _integer_model()
    threshold = 3
    probe = simulate_sparse_vqc_threshold_phase_oracle(
        model,
        encoded_real_threshold=threshold,
        strict=True,
    )
    expected = np.array(
        [
            model.integer_value(bitstring_from_index(index, 4)) < threshold
            for index in range(16)
        ],
        dtype=bool,
    )
    assert np.array_equal(probe.marked_mask, expected)
    assert probe.max_phase_error < 1e-12
    assert probe.auxiliary_zero_probability > 1.0 - 1e-12


def test_quantized_model_uses_phase_cost_mapping_and_fixed_point_scale() -> None:
    features = build_local_phase_features(2, 2, generator_edges=((0, 1),))
    phase_model = SparsePhaseVQC(
        num_generators=2,
        num_periods=2,
        features=features,
        phase_intercept=0.30,
        phase_weights=(0.02, -0.04, 0.01, 0.03, 0.05, -0.02, 0.01, 0.04),
        cost_center=20_000.0,
        cost_scale=5_000.0,
        phase_center=0.25,
        phase_scale=0.20,
        generator_edges=((0, 1),),
    )
    config = FixedPointConfig(fractional_bits=2, unit=1000.0)
    model = quantize_sparse_phase_model(phase_model, config)
    expected_intercept = 20_000.0 + 5_000.0 * (0.30 - 0.25) / 0.20
    assert np.isclose(model.real_intercept, expected_intercept)
    assert model.integer_intercept == config.encode(expected_intercept)
    assert model.integer_weights[0] == config.encode(500.0)
    assert model.integer_weights[1] == config.encode(-1000.0)


def test_window_zero_trained_phase_model_can_be_quantized_and_encoded() -> None:
    source = load_uc_instance(ROOT / "data" / "case14.json.gz")
    instance = time_window_instance(source, start=0, horizon=2)
    commitments = embedded_selected_commitments(
        np.ones((len(instance.generators), 2), dtype=int),
        (0, 5),
    )
    evaluator = FixedCommitmentEvaluator(instance)
    train_bitstrings: list[str] = []
    train_costs: list[float] = []
    for index in REPRESENTATIVE_INDEX_ORDER:
        commitment = commitments[index]
        if not is_logic_feasible(instance, commitment):
            continue
        result = evaluator.evaluate(commitment)
        if result.success and np.isfinite(result.total_cost):
            train_bitstrings.append(bitstring_from_index(index, 4))
            train_costs.append(float(result.total_cost))
        if len(train_costs) == 8:
            break

    fit = fit_sparse_phase_vqc(
        bitstrings=train_bitstrings,
        costs=train_costs,
        num_generators=2,
        num_periods=2,
        generator_edges=((0, 1),),
        seed=0,
        regularization=1e-4,
        maxiter=300,
    )
    model = quantize_sparse_phase_model(
        fit.model,
        FixedPointConfig(fractional_bits=2, unit=1000.0),
    )
    probe = basis_value_code_probe(model, train_bitstrings[0])
    assert probe.correct_code_probability > 1.0 - 1e-12
    assert model.lower_bound + model.value_shift == 0
    assert model.shifted_upper_bound < model.phase_modulus


def test_3x2_and_3x3_builders_scale_by_sparse_terms_without_state_simulation() -> None:
    for generators, periods in ((3, 2), (3, 3)):
        features = build_local_phase_features(generators, periods)
        integer_weights = tuple(1 for _ in features)
        lower, upper = conservative_integer_bounds(0, integer_weights)
        model = QuantizedSparseValueModel(
            num_generators=generators,
            num_periods=periods,
            features=features,
            fixed_point_config=FixedPointConfig(fractional_bits=0, unit=1.0),
            real_intercept=0.0,
            real_weights=tuple(float(value) for value in integer_weights),
            integer_intercept=0,
            integer_weights=integer_weights,
            coefficient_quantization_errors=(0.0,) * (1 + len(integer_weights)),
            lower_bound=lower,
            upper_bound=upper,
            value_shift=-lower,
            shifted_upper_bound=upper - lower,
            num_value_qubits=max(1, int(upper - lower).bit_length()),
        )
        compute = build_phase_to_value_circuit(model)
        oracle = build_sparse_vqc_threshold_phase_oracle(
            model,
            encoded_real_threshold=2,
            strict=True,
        )
        assert compute.num_qubits == model.num_x_qubits + model.num_value_qubits
        assert oracle.num_qubits >= compute.num_qubits + 1
        assert len(features) == generators * periods + generators * (periods - 1) + (generators - 1) * periods
