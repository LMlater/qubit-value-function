from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from experiments.stage1_case14_t2_fixed_point_affine_gas import (
    AerSimulator,
    build_argument_parser,
    estimate_statevector_memory_gb,
    execute_grover_circuit,
    predicted_cost_diagnostics,
    select_initial_index,
    select_measured_candidate,
)
from qubit_value_function.fixed_point_oracle import (
    FixedPointAffineSpec,
    FixedPointConfig,
    build_fixed_point_grover_circuit,
    fit_affine_cost_model,
    simulate_fixed_point_grover,
    simulate_fixed_point_phase_oracle,
)


def test_fixed_point_config_uses_common_cost_scale() -> None:
    config = FixedPointConfig(fractional_bits=2, unit=1000.0)
    assert config.scale == 4
    assert config.quantum == 250.0
    assert config.encode(20625.0) == 82
    assert config.decode(82) == 20500.0
    assert config.max_abs_rounding_error == 125.0


def test_fixed_point_config_has_chinese_validation_message() -> None:
    with pytest.raises(ValueError, match="fractional_bits 不能为负数"):
        FixedPointConfig(fractional_bits=-1)


def test_negative_coefficients_are_rewritten_for_weighted_adder() -> None:
    config = FixedPointConfig(fractional_bits=0, unit=1.0)
    spec = FixedPointAffineSpec.from_real_coefficients(
        config=config,
        intercept=10.0,
        coefficients=(3.0, -4.0, 2.0, -1.0),
    )
    assert spec.encoded_offset == 5
    assert spec.weights == (3, 4, 2, 1)
    assert spec.inverted_bit_indices == (1, 3)
    for index in range(16):
        assert spec.encoded_cost_for_index(index) == spec.direct_encoded_cost_for_index(index)


def test_affine_fit_uses_one_deterministic_method() -> None:
    bitstrings = ["0000", "1000", "0100", "0010", "0001", "1111"]
    true_intercept = 20.0
    true_coefficients = np.array([3.0, -4.0, 2.0, 1.0])
    costs = [
        true_intercept
        + float(np.dot(true_coefficients, np.array([int(bit) for bit in bitstring], dtype=float)))
        for bitstring in bitstrings
    ]
    intercept, coefficients = fit_affine_cost_model(bitstrings=bitstrings, costs=costs)
    assert np.isclose(intercept, true_intercept, atol=1e-6)
    assert np.allclose(coefficients, true_coefficients, atol=1e-6)


def test_default_initialization_and_simulation_options() -> None:
    args = build_argument_parser().parse_args([])
    assert args.initialization_policy == "first"
    assert args.seed == 0
    assert args.simulation_method == "mps"
    assert args.shots == 4096
    assert args.verify_phase_oracle is False
    assert args.max_statevector_memory_gb == 1.0


def test_first_initialization_does_not_choose_best_training_cost() -> None:
    train_indices = [8, 3, 5]
    observed = {8: 30.0, 3: 10.0, 5: 20.0}
    assert select_initial_index(train_indices, observed, policy="first") == 8
    assert select_initial_index(train_indices, observed, policy="best-training") == 3


def test_random_initialization_is_reproducible() -> None:
    train_indices = [8, 3, 5, 2]
    observed = {8: 30.0, 3: 10.0, 5: 20.0, 2: 40.0}
    first = select_initial_index(train_indices, observed, policy="random", seed=17)
    second = select_initial_index(train_indices, observed, policy="random", seed=17)
    assert first == second
    assert first in train_indices


def test_predicted_cost_diagnostics_has_complete_threshold_fields() -> None:
    config = FixedPointConfig(fractional_bits=0, unit=1.0)
    spec = FixedPointAffineSpec.from_real_coefficients(
        config=config,
        intercept=3.0,
        coefficients=(-1.0, -1.0, -1.0),
    )
    diagnostics = predicted_cost_diagnostics(spec, real_threshold=2.0)
    assert len(diagnostics["predicted_encoded_costs"]) == 8
    assert diagnostics["minimum_predicted_index"] == 7
    assert diagnostics["minimum_predicted_encoded_cost"] == 0
    assert diagnostics["encoded_threshold"] == 2
    assert diagnostics["minimum_predicted_minus_threshold"] == -2
    assert diagnostics["marked_count"] == len(diagnostics["marked_indices"])


def test_statevector_memory_estimate_and_guard() -> None:
    assert estimate_statevector_memory_gb(27) == 2.0
    circuit = QuantumCircuit(27)
    with pytest.raises(RuntimeError, match="请改用 --simulation-method mps"):
        execute_grover_circuit(
            circuit,
            num_x_qubits=1,
            simulation_method="statevector",
            shots=1,
            seed=0,
            max_statevector_memory_gb=1.0,
        )


def test_measured_candidate_comes_from_nonzero_allowed_distribution() -> None:
    probabilities = np.array([0.05, 0.70, 0.20, 0.05])
    candidate = select_measured_candidate(
        probabilities,
        observed_indices={1},
        allowed_indices={1, 2},
    )
    assert candidate == 2
    assert select_measured_candidate(
        probabilities,
        observed_indices={1, 2},
        allowed_indices={1, 2},
    ) is None


def test_fixed_point_phase_oracle_marks_cost_below_true_threshold_and_uncomputes() -> None:
    config = FixedPointConfig(fractional_bits=0, unit=1.0)
    spec = FixedPointAffineSpec.from_real_coefficients(
        config=config,
        intercept=10.0,
        coefficients=(-4.0, 2.0, 3.0),
    )
    probe = simulate_fixed_point_phase_oracle(spec, real_threshold=7.0, strict=True)
    expected_marked = np.array(
        [spec.encoded_cost_for_index(index) < config.encode(7.0) for index in range(8)],
        dtype=bool,
    )
    assert np.array_equal(probe.marked_mask, expected_marked)
    assert probe.auxiliary_zero_probability > 1.0 - 1e-12
    assert probe.max_phase_error < 1e-12


def test_fixed_point_grover_amplifies_unique_marked_state() -> None:
    config = FixedPointConfig(fractional_bits=0, unit=1.0)
    spec = FixedPointAffineSpec.from_real_coefficients(
        config=config,
        intercept=3.0,
        coefficients=(-1.0, -1.0, -1.0),
    )
    result = simulate_fixed_point_grover(spec, real_threshold=1.0, strict=True)
    assert int(result.marked_mask.sum()) == 1
    assert result.marked_probability > 0.9
    assert result.auxiliary_zero_probability > 1.0 - 1e-12


@pytest.mark.skipif(AerSimulator is None, reason="qiskit-aer 未安装")
def test_mps_executes_complete_grover_circuit_and_measures_marked_state() -> None:
    config = FixedPointConfig(fractional_bits=0, unit=1.0)
    spec = FixedPointAffineSpec.from_real_coefficients(
        config=config,
        intercept=2.0,
        coefficients=(-1.0, -1.0),
    )
    circuit = build_fixed_point_grover_circuit(
        spec,
        real_threshold=1.0,
        iterations=1,
        strict=True,
    )
    result = execute_grover_circuit(
        circuit,
        num_x_qubits=2,
        simulation_method="mps",
        shots=512,
        seed=23,
        max_statevector_memory_gb=1.0,
    )
    assert result.simulation_method == "mps"
    assert result.x_counts is not None
    assert result.x_counts.get("11", 0) > 450
    assert result.x_probabilities[3] > 0.88
    assert result.auxiliary_zero_probability > 0.99
