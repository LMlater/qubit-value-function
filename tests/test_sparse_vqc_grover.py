from __future__ import annotations

import inspect

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import experiments.stage1_case14_2x2_sparse_vqc_grover as grover_experiment
from qubit_value_function.coherent_phase_value import (
    QuantizedSparseValueModel,
    conservative_integer_bounds,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig
from qubit_value_function.sparse_phase_vqc import build_local_phase_features
from qubit_value_function.sparse_vqc_grover import (
    AerSimulator,
    append_search_register_diffuser,
    build_sparse_vqc_grover_circuit,
    execute_sparse_vqc_grover_mps,
    marked_semantics_diagnostics,
    ordinary_grover_validation_plan,
    select_measured_candidate,
    simulate_sparse_vqc_grover_statevector,
)


def _binary_value_model(num_generators: int = 2, num_periods: int = 2) -> QuantizedSparseValueModel:
    features = build_local_phase_features(num_generators, num_periods)
    num_x_qubits = num_generators * num_periods
    linear_weights = tuple(2**index for index in range(num_x_qubits))
    integer_weights = linear_weights + (0,) * (len(features) - num_x_qubits)
    lower, upper = conservative_integer_bounds(0, integer_weights)
    return QuantizedSparseValueModel(
        num_generators=num_generators,
        num_periods=num_periods,
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


def test_grover_builder_requires_explicit_iterations_and_does_not_enumerate_states() -> None:
    source = inspect.getsource(build_sparse_vqc_grover_circuit)
    assert "range(2" not in source
    model = _binary_value_model()
    circuit = build_sparse_vqc_grover_circuit(
        model,
        encoded_threshold=1,
        iterations=0,
    )
    assert circuit.num_qubits > model.num_x_qubits
    assert circuit.count_ops().get("h", 0) == model.num_x_qubits
    with pytest.raises(ValueError, match="iterations"):
        build_sparse_vqc_grover_circuit(model, encoded_threshold=1, iterations=-1)


def test_search_diffuser_does_not_touch_auxiliary_qubits() -> None:
    circuit = QuantumCircuit(6)
    circuit.x([4, 5])
    append_search_register_diffuser(circuit, circuit.qubits[:4])
    probabilities = Statevector.from_instruction(circuit).probabilities()
    assert np.isclose(
        sum(probability for index, probability in enumerate(probabilities) if (index >> 4) == 3),
        1.0,
    )


def test_unique_marked_state_is_amplified_by_exact_ordinary_grover() -> None:
    model = _binary_value_model()
    plan = ordinary_grover_validation_plan(model, encoded_threshold=1)
    assert plan.marked_indices == (0,)
    assert plan.iterations == 3
    probe = simulate_sparse_vqc_grover_statevector(
        model,
        encoded_threshold=1,
        iterations=plan.iterations,
    )
    assert probe.marked_probability > 0.9
    assert probe.auxiliary_zero_probability > 1.0 - 1e-12


def test_three_marked_states_are_amplified_by_exact_ordinary_grover() -> None:
    model = _binary_value_model()
    plan = ordinary_grover_validation_plan(model, encoded_threshold=3)
    assert plan.marked_indices == (0, 1, 2)
    assert plan.iterations == 1
    probe = simulate_sparse_vqc_grover_statevector(
        model,
        encoded_threshold=3,
        iterations=plan.iterations,
    )
    assert probe.marked_probability > 0.9
    assert probe.auxiliary_zero_probability > 1.0 - 1e-12


def test_no_marked_plan_stops_without_requesting_a_grover_iteration() -> None:
    plan = ordinary_grover_validation_plan(_binary_value_model(), encoded_threshold=0)
    assert plan.marked_indices == ()
    assert plan.marked_count == 0
    assert plan.iterations == 0
    assert plan.initial_marked_probability == 0.0


@pytest.mark.skipif(AerSimulator is None, reason="qiskit-aer is not installed")
def test_mps_measurements_match_exact_probability_and_preserve_auxiliary_zero() -> None:
    model = _binary_value_model()
    plan = ordinary_grover_validation_plan(model, encoded_threshold=3)
    circuit = build_sparse_vqc_grover_circuit(
        model,
        encoded_threshold=3,
        iterations=plan.iterations,
    )
    exact = simulate_sparse_vqc_grover_statevector(
        model,
        encoded_threshold=3,
        iterations=plan.iterations,
    )
    measured = execute_sparse_vqc_grover_mps(
        circuit,
        num_x_qubits=model.num_x_qubits,
        shots=1024,
        seed=17,
    )
    measured_marked = sum(measured.x_probabilities[index] for index in plan.marked_indices)
    assert abs(measured_marked - exact.marked_probability) < 0.08
    assert measured.auxiliary_zero_probability > 0.99
    assert sum(measured.x_counts.values()) == 1024
    assert sum(measured.raw_counts.values()) == 1024


def test_measured_candidate_uses_actual_counts_and_prefers_unobserved_allowed_state() -> None:
    candidate = select_measured_candidate(
        {"1000": 50, "0100": 60, "1100": 200, "0000": 714},
        num_x_qubits=4,
        allowed_indices=(1, 2),
        observed_indices=(2,),
    )
    assert candidate is not None
    assert candidate.index == 1
    assert candidate.count == 50
    assert candidate.was_observed is False
    assert np.isclose(candidate.probability, 50 / 1024)


def test_marked_semantics_disagreement_is_diagnostic_only() -> None:
    diagnostics = marked_semantics_diagnostics((1, 2, 3), (2, 3, 4))
    assert diagnostics["marked_set_intersection"] == [2, 3]
    assert diagnostics["integer_only_marked_indices"] == [1]
    assert diagnostics["direct_float_only_marked_indices"] == [4]
    assert diagnostics["classification_disagreement_count"] == 2


def test_candidate_ed_lp_verification_keeps_threshold_fixed() -> None:
    class Result:
        success = True
        total_cost = 90.0
        message = "ok"

    class Evaluator:
        def evaluate(self, commitment):
            assert commitment == "candidate"
            return Result()

    record = grover_experiment._verify_candidate_without_threshold_update(
        evaluator=Evaluator(),
        commitments=np.array(["candidate"], dtype=object),
        candidate_index=0,
        incumbent_true_cost=100.0,
        encoded_threshold=8,
    )
    assert record["would_improve_incumbent"] is True
    assert record["encoded_threshold_before"] == 8
    assert record["encoded_threshold_after"] == 8
    assert record["threshold_updated"] is False


def test_3x2_and_3x3_grover_builders_only_construct_sparse_resources() -> None:
    for generators, periods in ((3, 2), (3, 3)):
        model = _binary_value_model(generators, periods)
        circuit = build_sparse_vqc_grover_circuit(
            model,
            encoded_threshold=2,
            iterations=1,
        )
        assert circuit.num_qubits > model.num_x_qubits
        assert any("sparse_vqc_threshold" in name for name in circuit.count_ops())
