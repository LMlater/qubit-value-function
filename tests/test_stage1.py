from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qubit_value_function.commitment import all_commitments, commitment_to_bitstring
from qubit_value_function.ancilla_vqc import (
    apply_controlled_ancilla_oracle_state,
    apply_explicit_two_ancilla_oracle_state,
    evaluate_ancilla_vqc,
    evaluate_threshold_conditioned_ancilla_vqc,
    fit_ancilla_vqc,
    fit_leakage_reweighted_ancilla_vqc,
    fit_threshold_conditioned_ancilla_vqc,
    grover_with_ancilla_oracle,
    grover_with_ancilla_model,
    grover_with_controlled_ancilla_model,
    grover_with_explicit_two_ancilla_model,
    verify_ancilla_oracle,
)
from qubit_value_function.ed import FixedCommitmentEvaluator
from qubit_value_function.feature_phase import (
    evaluate_feature_phase_model,
    fit_feature_phase_model,
)
from qubit_value_function.grover_minimum import (
    grover_threshold_probabilities,
    run_grover_minimum_finding,
    summarize_minimum_finding_runs,
)
from qubit_value_function.gate_level_oracle import (
    GateLevelAffinePieceSpec,
    GateLevelAffineOracleSpec,
    GateLevelMaxAffineOracleSpec,
    bitstring_from_index,
    build_max_affine_phase_oracle_circuit,
    build_affine_phase_oracle_circuit,
    circuit_resource_summary,
    simulate_affine_grover,
    simulate_affine_phase_oracle,
    simulate_max_affine_grover,
    simulate_max_affine_phase_oracle,
)
from qubit_value_function.qft_weighted_sum_oracle import (
    build_qft_max_affine_phase_oracle_circuit,
    simulate_qft_max_affine_grover,
    simulate_qft_max_affine_phase_oracle,
)
from qubit_value_function.max_affine import (
    fit_max_affine_value_function,
    max_affine_gate_counts,
)
from qubit_value_function.oracle import (
    grover_search_probabilities,
    grover_with_oracle_matrix,
    phase_oracle_matrix,
    verify_phase_oracle,
)
from qubit_value_function.phase_vqc import train_threshold_phase_vqc
from qubit_value_function.structured_features import structured_commitment_features
from qubit_value_function.uc_loader import load_uc_instance
from qubit_value_function.value_surrogate import (
    evaluate_scalar_value_function,
    fit_scalar_value_function,
    quantize_values,
)
from experiments.stage1_case14_t2_ancilla_vqc import (
    commitment_row,
    evaluate_values,
    leading_time_window_instance,
)
from experiments.stage1_case14_t2_joint_oracle_training import joint_oracle_score
from experiments.stage1_case14_t2_value_register_comparator import (
    tie_tolerant_threshold_case_for_top_count,
)
from experiments.stage1_case14_t2_structured_value_surrogate import (
    calibrated_prediction_threshold,
)
from experiments.stage1_case14_t2_max_affine_value_surrogate import (
    boundary_candidate_order,
)
from experiments.stage1_case14_t2_gate_level_grover_oracle import (
    case14_t2_gate_level_proxy_spec,
    embedded_selected_commitments,
)
from experiments.stage1_case14_t2_gate_level_max_affine_oracle import (
    case14_t2_gate_level_max_affine_spec,
)
from experiments.stage1_case14_t2_learned_small_max_affine_gate_level_oracle import (
    calibrate_integer_threshold,
    learn_gap_weighted_max_affine_pieces,
)


def test_load_aelmp_simple_instance() -> None:
    instance = load_uc_instance("data/aelmp_simple.json.gz")
    assert instance.time_horizon == 1
    assert [gen.name for gen in instance.generators] == ["W", "X", "Y"]
    assert instance.fixed_load == [365.0]


def test_exact_value_function_has_single_best_commitment() -> None:
    instance = load_uc_instance("data/aelmp_simple.json.gz")
    evaluator = FixedCommitmentEvaluator(instance)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    rows = []
    for commitment in commitments:
        result = evaluator.evaluate(commitment)
        rows.append((commitment_to_bitstring(commitment), result.total_cost))

    rows.sort(key=lambda item: item[1])
    assert rows[0][0] == "101"
    assert rows[0][1] < rows[1][1]
    assert np.isfinite(rows[0][1])


def test_threshold_phase_oracle_is_reversible() -> None:
    marked = np.array([False, False, False, False, False, False, True, False])
    oracle = phase_oracle_matrix(marked)
    checks = verify_phase_oracle(oracle)
    assert checks == {"unitary": True, "self_inverse": True, "real_diagonal": True}


def test_grover_amplifies_marked_state_for_three_qubits() -> None:
    marked = np.array([False, False, False, False, False, False, True, False])
    result = grover_search_probabilities(marked)
    assert result["iterations"] == 2
    assert result["marked_probability"] > 0.9


def test_state_vector_threshold_grover_amplifies_marked_set() -> None:
    marked = np.array([False, False, False, False, False, False, True, False])
    probabilities = grover_threshold_probabilities(marked, iterations=2)

    assert probabilities[marked].sum() > 0.9
    assert np.isclose(probabilities.sum(), 1.0)


def test_gate_level_affine_phase_oracle_uncomputes_auxiliaries() -> None:
    spec = GateLevelAffineOracleSpec(
        weights=(1, 1, 1),
        threshold=0,
        inverted_bit_indices=(0, 1, 2),
    )

    probe = simulate_affine_phase_oracle(spec)

    assert [bitstring_from_index(index, 3) for index in np.flatnonzero(probe.marked_mask)] == ["111"]
    assert probe.aux_zero_probability > 1.0 - 1e-12
    assert probe.max_phase_error < 1e-12


def test_gate_level_affine_grover_amplifies_t2_toy_pattern() -> None:
    spec = GateLevelAffineOracleSpec(
        weights=(1, 1, 1, 1),
        threshold=0,
        inverted_bit_indices=(0, 1, 2, 3),
        bit_labels=("g1_t0", "g1_t1", "g2_t0", "g2_t1"),
    )

    result = simulate_affine_grover(spec)

    assert result.iterations == 3
    assert result.marked_probability > 0.9
    assert result.aux_zero_probability > 1.0 - 1e-12
    assert bitstring_from_index(int(np.argmax(result.x_probabilities)), 4) == "1111"


def test_gate_level_resource_summary_reports_extra_value_qubits() -> None:
    spec = GateLevelAffineOracleSpec(weights=(1, 2, 1), threshold=1)
    circuit = build_affine_phase_oracle_circuit(spec)
    resources = circuit_resource_summary(circuit, decompose_reps=0)

    assert resources["num_qubits"] > spec.num_x_qubits
    assert resources["operations"]["adder"] == 1
    assert resources["operations"]["adder_dg"] == 1


def test_gate_level_max_affine_oracle_uses_piece_intersection() -> None:
    spec = GateLevelMaxAffineOracleSpec(
        pieces=(
            GateLevelAffinePieceSpec(
                weights=(1, 1, 1, 0),
                inverted_bit_indices=(0, 1),
                name="L0",
            ),
            GateLevelAffinePieceSpec(
                weights=(0, 0, 1, 1),
                name="L1",
            ),
        ),
        threshold=0,
    )

    probe = simulate_max_affine_phase_oracle(spec)
    result = simulate_max_affine_grover(spec)
    marked_bitstrings = [
        bitstring_from_index(index, spec.num_x_qubits)
        for index in np.flatnonzero(spec.marked_mask())
    ]

    assert marked_bitstrings == ["1100"]
    assert probe.aux_zero_probability > 1.0 - 1e-12
    assert probe.max_phase_error < 1e-12
    assert result.marked_probability > 0.9
    assert bitstring_from_index(int(np.argmax(result.x_probabilities)), 4) == "1100"


def test_gate_level_max_affine_resources_have_two_piece_blocks() -> None:
    spec = GateLevelMaxAffineOracleSpec(
        pieces=(
            GateLevelAffinePieceSpec((1, 1, 1, 0), (0, 1)),
            GateLevelAffinePieceSpec((0, 0, 1, 1)),
        ),
        threshold=0,
    )
    circuit = build_max_affine_phase_oracle_circuit(spec)
    resources = circuit_resource_summary(circuit, decompose_reps=0)

    assert resources["operations"]["adder"] == 2
    assert resources["operations"]["cmp"] == 2
    assert resources["operations"]["adder_dg"] == 2
    assert resources["operations"]["cmp_dg"] == 2




def test_qft_weighted_sum_max_affine_matches_weighted_adder_marking() -> None:
    spec = GateLevelMaxAffineOracleSpec(
        pieces=(
            GateLevelAffinePieceSpec(
                weights=(7, 6, 0, 0),
                inverted_bit_indices=(0, 1),
                name="L0",
            ),
            GateLevelAffinePieceSpec(
                weights=(0, 0, 1, 1),
                name="L1",
            ),
        ),
        threshold=0,
    )

    weighted_probe = simulate_max_affine_phase_oracle(spec)
    qft_probe = simulate_qft_max_affine_phase_oracle(spec)
    qft_result = simulate_qft_max_affine_grover(spec)

    marked_bitstrings = [
        bitstring_from_index(index, spec.num_x_qubits)
        for index in np.flatnonzero(qft_probe.marked_mask)
    ]

    assert np.array_equal(qft_probe.marked_mask, weighted_probe.marked_mask)
    assert marked_bitstrings == ["1100"]
    assert qft_probe.aux_zero_probability > 1.0 - 1e-12
    assert qft_probe.max_phase_error < 1e-12
    assert qft_result.marked_probability > 0.9
    assert qft_result.aux_zero_probability > 1.0 - 1e-11


def test_qft_weighted_sum_uses_fewer_qubits_on_small_case14_piece_set() -> None:
    spec = GateLevelMaxAffineOracleSpec(
        pieces=(
            GateLevelAffinePieceSpec((7, 6, 0, 0), (0, 1)),
            GateLevelAffinePieceSpec((0, 0, 1, 1)),
        ),
        threshold=0,
    )
    weighted_resources = circuit_resource_summary(
        build_max_affine_phase_oracle_circuit(spec),
        decompose_reps=0,
    )
    qft_resources = circuit_resource_summary(
        build_qft_max_affine_phase_oracle_circuit(spec),
        decompose_reps=0,
    )

    assert qft_resources["num_qubits"] < weighted_resources["num_qubits"]
    assert any(name.startswith("qft_sum") for name in qft_resources["operations"])
    assert "adder" in weighted_resources["operations"]
def test_grover_minimum_finding_reaches_exact_minimum_on_toy_values() -> None:
    values = np.array([8.0, 7.0, 1.0, 5.0, 4.0, 3.0, 2.0, 6.0])
    feasible = np.ones(8, dtype=bool)
    run = run_grover_minimum_finding(
        values,
        values,
        feasible,
        initial_index=0,
        max_rounds=8,
        seed=0,
    )
    summary = summarize_minimum_finding_runs([run])

    assert run.success
    assert run.best_index == 2
    assert summary["success_count"] == 1


def test_threshold_conditioned_phase_vqc_oracle_matches_exact_marks() -> None:
    bits = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=int,
    )
    values = np.array(
        [365000.0, 253100.0, 226360.0, 114460.0, 118270.0, 40525.0, 49595.0, 61795.0]
    )
    thresholds = [45060.0, 60000.0, 90000.0, 150000.0, 250000.0]
    result = train_threshold_phase_vqc(bits, values, thresholds)
    assert result.correct_marked_sets
    assert result.max_phase_factor_error < 1e-8

    oracle = result.model.oracle_matrix(bits, 45060.0)
    checks = verify_phase_oracle(oracle, atol=1e-8)
    assert checks["unitary"]
    assert checks["self_inverse"]
    assert checks["real_diagonal"]

    marked = values <= 45060.0
    grover = grover_with_oracle_matrix(oracle, marked)
    assert grover["iterations"] == 2
    assert grover["target_probability"] > 0.9


def test_feature_phase_model_is_unitary_diagonal_oracle() -> None:
    features = np.eye(4)
    labels = np.array([False, True, False, True])
    model = fit_feature_phase_model(features, labels, ["s0", "s1", "s2", "s3"])
    evaluation = evaluate_feature_phase_model(model, features, labels)
    oracle = model.oracle_matrix(features)

    assert evaluation["correct_marked_set"]
    assert all(verify_phase_oracle(oracle).values())


def test_ancilla_vqc_oracle_is_reversible_when_angles_are_deterministic() -> None:
    features = np.eye(4)
    labels = np.array([False, True, False, True])
    model = fit_ancilla_vqc(features, labels, ["s0", "s1", "s2", "s3"])
    evaluation = evaluate_ancilla_vqc(model, features, labels)
    oracle = model.oracle_matrix(features)
    grover = grover_with_ancilla_oracle(oracle, labels)

    assert evaluation.correct_marked_set
    assert evaluation.max_leakage_probability < 1e-12
    assert all(verify_ancilla_oracle(oracle).values())
    assert grover["zero_ancilla_probability"] > 1.0 - 1e-12


def test_state_vector_ancilla_grover_matches_matrix_simulation() -> None:
    features = np.eye(4)
    labels = np.array([False, True, False, True])
    model = fit_ancilla_vqc(features, labels, ["s0", "s1", "s2", "s3"])
    matrix_grover = grover_with_ancilla_oracle(model.oracle_matrix(features), labels, iterations=1)
    state_grover = grover_with_ancilla_model(model, features, labels, iterations=1)

    assert np.isclose(state_grover["target_x_probability"], matrix_grover["target_x_probability"])
    assert np.isclose(state_grover["zero_ancilla_probability"], matrix_grover["zero_ancilla_probability"])


def test_controlled_ancilla_oracle_leaves_infeasible_blocks_unchanged() -> None:
    features = np.eye(4)
    feasible = np.array([False, True, False, True])
    labels = np.array([False, True, False, False])
    model = fit_ancilla_vqc(features[feasible], labels[feasible], ["s0", "s1", "s2", "s3"])
    angles = model.angles(features)
    state = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=complex,
    )
    output = apply_controlled_ancilla_oracle_state(
        state,
        np.cos(angles),
        np.sin(angles),
        feasible,
    )
    grover = grover_with_controlled_ancilla_model(model, features, feasible, labels, iterations=1)

    assert np.array_equal(output[~feasible], state[~feasible])
    assert np.isclose(np.linalg.norm(output), np.linalg.norm(state))
    assert grover["target_x_probability"] >= 0.0


def test_explicit_two_ancilla_oracle_matches_controlled_model() -> None:
    features = np.eye(4)
    feasible = np.array([False, True, False, True])
    labels = np.array([False, True, False, False])
    model = fit_ancilla_vqc(features[feasible], labels[feasible], ["s0", "s1", "s2", "s3"])
    explicit = grover_with_explicit_two_ancilla_model(
        model,
        features,
        feasible,
        labels,
        iterations=1,
    )
    equivalent = grover_with_controlled_ancilla_model(
        model,
        features,
        feasible,
        labels,
        iterations=1,
    )

    assert np.isclose(explicit["target_x_probability"], equivalent["target_x_probability"])
    assert np.isclose(explicit["one_value_ancilla_probability"], equivalent["one_ancilla_probability"])
    assert explicit["one_feasibility_probability"] < 1e-12


def test_explicit_two_ancilla_oracle_uncomputes_feasibility_bit() -> None:
    feasible = np.array([False, True, False, True])
    angles = np.array([0.1, np.pi, 0.4, 0.0])
    state = np.zeros((4, 2, 2), dtype=complex)
    state[:, 0, 0] = 0.5
    output = apply_explicit_two_ancilla_oracle_state(state, angles, feasible)

    assert np.allclose(output[:, 1, :], 0.0)
    assert np.isclose(np.linalg.norm(output), np.linalg.norm(state))


def test_leakage_reweighted_training_can_reduce_synthetic_max_leakage() -> None:
    features = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    labels = np.array([False, False, True, True])
    result = fit_leakage_reweighted_ancilla_vqc(
        features,
        labels,
        ["bias", "x"],
        alphas=(1.0, 5.0, 20.0),
        iterations=5,
    )

    assert result.final_evaluation.correct_marked_set
    assert (
        result.final_evaluation.max_leakage_probability
        < result.initial_evaluation.max_leakage_probability
    )
    assert result.selected_alpha is not None


def test_threshold_conditioned_ancilla_vqc_represents_threshold_family() -> None:
    features = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    thresholds = np.array([0.5, 1.5])
    labels = np.array(
        [
            [True, True],
            [False, True],
        ]
    )
    model = fit_threshold_conditioned_ancilla_vqc(
        features,
        labels,
        thresholds,
        ["bias", "x"],
    )
    evaluation = evaluate_threshold_conditioned_ancilla_vqc(
        model,
        features,
        labels,
        thresholds,
    )

    assert evaluation.correct_marked_sets
    assert evaluation.max_leakage_probability < 1e-12
    assert evaluate_ancilla_vqc(model.fixed_tau_model(0.5), features, labels[:, 0]).correct_marked_set


def test_piecewise_tau_basis_interpolates_between_trained_ancilla_oracles() -> None:
    features = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    thresholds = np.array([0.0, 1.0])
    labels = np.array(
        [
            [False, True],
            [False, True],
        ]
    )
    model = fit_threshold_conditioned_ancilla_vqc(
        features,
        labels,
        thresholds,
        ["bias", "x"],
        tau_basis="piecewise_linear",
    )

    assert np.allclose(model.fixed_tau_model(0.0).angles(features), 0.0)
    assert np.allclose(model.fixed_tau_model(1.0).angles(features), np.pi)
    assert np.allclose(model.fixed_tau_model(0.5).angles(features), np.pi / 2.0)


def test_scalar_value_surrogate_supports_monotone_threshold_comparison() -> None:
    features = np.eye(4)
    values = np.array([10.0, 20.0, 30.0, 40.0])
    model = fit_scalar_value_function(features, values, ["s0", "s1", "s2", "s3"])
    evaluation = evaluate_scalar_value_function(model, features, values)
    predictions = model.predict(features)

    assert evaluation.max_abs_error < 1e-12
    assert np.array_equal(predictions <= 25.0, np.array([True, True, False, False]))
    assert np.all((predictions <= 25.0) <= (predictions <= 35.0))


def test_fixed_point_quantization_is_monotone_for_value_register() -> None:
    values = np.array([0.0, 0.25, 0.5, 1.0])
    register = quantize_values(values, value_min=0.0, value_max=1.0, bits=3)

    assert np.array_equal(register, np.array([0, 2, 4, 7]))
    assert np.all(register[:-1] <= register[1:])


def test_joint_oracle_score_balances_target_probability_and_leakage() -> None:
    weights = {"max_leakage": 0.35, "mean_leakage": 0.05, "mark_error": 10.0}
    high_probability_score = joint_oracle_score(
        target_probability=0.78,
        max_leakage=0.11,
        mean_leakage=0.01,
        mark_error_count=0,
        weights=weights,
    )
    balanced_score = joint_oracle_score(
        target_probability=0.76,
        max_leakage=0.04,
        mean_leakage=0.01,
        mark_error_count=0,
        weights=weights,
    )
    wrong_mark_score = joint_oracle_score(
        target_probability=0.99,
        max_leakage=0.0,
        mean_leakage=0.0,
        mark_error_count=1,
        weights=weights,
    )

    assert balanced_score > high_probability_score
    assert wrong_mark_score < high_probability_score


def test_case14_two_period_value_function_uses_real_data() -> None:
    instance = leading_time_window_instance(load_uc_instance("data/case14.json.gz"), 2)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    values, logic_feasible = evaluate_values(instance, commitments)
    rows = sorted(
        (commitment_to_bitstring(commitment), value)
        for commitment, value in zip(commitments, values)
        if np.isfinite(value)
    )
    best_bitstring, best_cost = min(rows, key=lambda item: item[1])

    assert commitments.shape == (4096, 6, 2)
    assert int(logic_feasible.sum()) == 768
    assert len(rows) == 768
    assert best_bitstring == "110011111100"
    assert np.isclose(best_cost, 20578.2152604)


def test_case14_two_period_commitment_row_is_readable() -> None:
    instance = leading_time_window_instance(load_uc_instance("data/case14.json.gz"), 2)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    values, _ = evaluate_values(instance, commitments)
    best_idx = int(np.nanargmin(values))
    row = commitment_row(commitments, [gen.name for gen in instance.generators], values, best_idx)

    assert row["bitstring_generator_major"] == "110011111100"
    assert row["bitstring_time_major"] == "101110101110"
    assert row["schedule_table"] == [
        {"generator": "g1", "t0": 1, "t1": 1},
        {"generator": "g2", "t0": 0, "t1": 0},
        {"generator": "g3", "t0": 1, "t1": 1},
        {"generator": "g4", "t0": 1, "t1": 1},
        {"generator": "g5", "t0": 1, "t1": 1},
        {"generator": "g6", "t0": 0, "t1": 0},
    ]


def test_case14_t2_gate_level_proxy_marks_expected_subregister() -> None:
    instance = leading_time_window_instance(load_uc_instance("data/case14.json.gz"), 2)
    spec = case14_t2_gate_level_proxy_spec(instance, (0, 1, 2))

    marked_bitstrings = [
        bitstring_from_index(index, spec.num_x_qubits)
        for index in np.flatnonzero(spec.marked_mask())
    ]

    assert spec.bit_labels == ("g1_t0", "g1_t1", "g2_t0", "g2_t1", "g3_t0", "g3_t1")
    assert marked_bitstrings == ["110011"]


def test_case14_t2_gate_level_max_affine_marks_expected_subregister() -> None:
    instance = leading_time_window_instance(load_uc_instance("data/case14.json.gz"), 2)
    spec = case14_t2_gate_level_max_affine_spec(instance, (0, 1))

    marked_bitstrings = [
        bitstring_from_index(index, spec.num_x_qubits)
        for index in np.flatnonzero(spec.marked_mask())
    ]

    assert spec.bit_labels == ("g1_t0", "g1_t1", "g2_t0", "g2_t1")
    assert marked_bitstrings == ["1100"]




def test_gate_level_max_affine_piece_bias_shifts_threshold() -> None:
    spec = GateLevelMaxAffineOracleSpec(
        pieces=(
            GateLevelAffinePieceSpec(weights=(1,), bias=1),
        ),
        threshold=1,
    )

    probe = simulate_max_affine_phase_oracle(spec)
    marked_bitstrings = [
        bitstring_from_index(index, spec.num_x_qubits)
        for index in np.flatnonzero(spec.marked_mask())
    ]

    assert marked_bitstrings == ["0"]
    assert probe.aux_zero_probability > 1.0 - 1e-12
    assert probe.max_phase_error < 1e-12


def test_case14_t2_learned_small_max_affine_uses_true_cost_gaps() -> None:
    instance = leading_time_window_instance(load_uc_instance("data/case14.json.gz"), 2)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    values, _ = evaluate_values(instance, commitments)
    base_commitment = commitments[int(np.nanargmin(values))]
    embedded_commitments = embedded_selected_commitments(base_commitment, (0, 5))
    embedded_values, _ = evaluate_values(instance, embedded_commitments)
    embedded_best_index = int(np.nanargmin(embedded_values))

    learned = learn_gap_weighted_max_affine_pieces(
        instance=instance,
        selected_generator_indices=(0, 5),
        embedded_values=embedded_values,
        embedded_best_index=embedded_best_index,
        max_weight=7,
    )
    top1_spec = GateLevelMaxAffineOracleSpec(
        pieces=learned.pieces,
        threshold=0,
        bit_labels=learned.bit_labels,
    )
    top3_calibration = calibrate_integer_threshold(
        predicted_values=top1_spec.values_for_all_x(),
        true_values=embedded_values,
        target_count=3,
    )
    top3_spec = GateLevelMaxAffineOracleSpec(
        pieces=learned.pieces,
        threshold=int(top3_calibration["selected_threshold"]),
        bit_labels=learned.bit_labels,
    )

    top1_marked = [
        bitstring_from_index(index, top1_spec.num_x_qubits)
        for index in np.flatnonzero(top1_spec.marked_mask())
    ]
    top3_marked = [
        bitstring_from_index(index, top3_spec.num_x_qubits)
        for index in np.flatnonzero(top3_spec.marked_mask())
    ]

    assert learned.bit_labels == ("g1_t0", "g1_t1", "g6_t0", "g6_t1")
    assert learned.integer_weights == (7, 6, 1, 1)
    assert top1_marked == ["1100"]
    assert top3_calibration["selected_threshold"] == 1
    assert top3_calibration["selected_row"]["exact_marked_set"]
    assert top3_marked == ["1100", "1110", "1101"]
    assert commitment_to_bitstring(embedded_commitments[embedded_best_index]) == "110011111100"


def test_case14_t2_adaptive_max_affine_finds_embedded_true_optimum() -> None:
    from experiments.stage1_case14_t2_adaptive_max_affine_minimum_search import (
        run_adaptive_trial,
    )

    instance = leading_time_window_instance(load_uc_instance("data/case14.json.gz"), 2)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    values, _ = evaluate_values(instance, commitments)
    base_commitment = commitments[int(np.nanargmin(values))]
    embedded_commitments = embedded_selected_commitments(base_commitment, (0, 5))
    embedded_values, _ = evaluate_values(instance, embedded_commitments)
    embedded_best_index = int(np.nanargmin(embedded_values))
    learned = learn_gap_weighted_max_affine_pieces(
        instance=instance,
        selected_generator_indices=(0, 5),
        embedded_values=embedded_values,
        embedded_best_index=embedded_best_index,
        max_weight=7,
    )
    template_spec = GateLevelMaxAffineOracleSpec(
        pieces=learned.pieces,
        threshold=0,
        bit_labels=learned.bit_labels,
    )

    result = run_adaptive_trial(
        predicted_values=template_spec.values_for_all_x(),
        true_values=embedded_values,
        embedded_commitments=embedded_commitments,
        learned_pieces=learned.pieces,
        bit_labels=learned.bit_labels,
        initial_index=7,
        max_rounds=5,
        rng=np.random.default_rng(0),
    )

    assert result["success"]
    assert result["best_index"] == embedded_best_index
    assert result["best_true_value"] == embedded_values[embedded_best_index]
    assert result["accepted_update_count"] >= 1


def test_case14_t2_adaptive_success_uses_true_not_predicted_optimum() -> None:
    from experiments.stage1_case14_t2_adaptive_max_affine_minimum_search import (
        run_adaptive_trial,
    )

    pieces = (
        GateLevelAffinePieceSpec(
            weights=(1, 1),
            inverted_bit_indices=(1,),
            name="misordered_piece",
        ),
    )
    predicted_values = GateLevelMaxAffineOracleSpec(
        pieces=pieces,
        threshold=0,
        bit_labels=("x0", "x1"),
    ).values_for_all_x()
    true_values = np.array([0.0, 10.0, 1.0, 2.0])
    embedded_commitments = np.array(
        [
            [[0, 0]],
            [[1, 0]],
            [[0, 1]],
            [[1, 1]],
        ],
        dtype=int,
    )

    result = run_adaptive_trial(
        predicted_values=predicted_values,
        true_values=true_values,
        embedded_commitments=embedded_commitments,
        learned_pieces=pieces,
        bit_labels=("x0", "x1"),
        initial_index=3,
        max_rounds=4,
        rng=np.random.default_rng(0),
    )

    assert int(np.argmin(predicted_values)) != int(np.argmin(true_values))
    assert result["success"] == (result["best_index"] == int(np.argmin(true_values)))
    assert not result["success"]


def test_max_affine_adaptive_grover_sampling_returns_valid_probability_row() -> None:
    from experiments.stage1_case14_t2_max_affine_adaptive_grover_search import (
        sample_after_grover_iterations,
    )

    marked = np.array([False, True, False, False])
    result = sample_after_grover_iterations(
        marked,
        iterations=1,
        rng=np.random.default_rng(0),
    )

    assert 0 <= result["sampled_index"] < marked.size
    assert np.isclose(result["marked_probability"] + result["nonmarked_probability"], 1.0)
    assert result["target_probability"] == result["marked_probability"]


def test_max_affine_adaptive_bbht_finds_true_improvement_with_exact_marks() -> None:
    from experiments.stage1_case14_t2_max_affine_adaptive_grover_search import (
        bbht_search_current_threshold,
    )

    values = np.array([10.0, 1.0, 12.0, 13.0])
    marked = np.array([False, True, False, False])
    result = bbht_search_current_threshold(
        marked=marked,
        true_improving_labels=marked,
        values=values,
        incumbent_value=10.0,
        rng=np.random.default_rng(0),
        lambda_growth=8.0 / 7.0,
        max_trials=20,
        tie_tolerance=1e-9,
    )

    assert result["improved_index"] == 1
    assert any(trial["success"] for trial in result["trials"])


def test_max_affine_adaptive_outer_loop_stops_without_true_improvement() -> None:
    from experiments.stage1_case14_t2_max_affine_adaptive_grover_search import (
        run_adaptive_minimum_search,
    )

    values = np.array([1.0, 2.0, 3.0, np.inf])
    predictions = np.array([1.0, 2.0, 3.0, 4.0])
    value_domain = np.array([True, True, True, False])
    result = run_adaptive_minimum_search(
        values=values,
        predictions=predictions,
        value_domain=value_domain,
        initial_index=0,
        rng=np.random.default_rng(0),
        lambda_growth=8.0 / 7.0,
        max_rounds=5,
        max_bbht_trials_per_threshold=4,
        use_calibrated_threshold=True,
        stop_after_no_improvement=2,
        tie_tolerance=1e-9,
    )

    assert result["stop_reason"] == "no_true_improving_state"
    assert result["rounds"] == []
    assert result["final_incumbent_index"] == 0


def test_max_affine_adaptive_calibrated_oracle_reports_diagnostics() -> None:
    from experiments.stage1_case14_t2_max_affine_adaptive_grover_search import (
        max_affine_threshold_oracle_labels,
    )

    values = np.array([5.0, 1.0, 2.0, 10.0])
    predictions = np.array([5.0, 4.0, 1.0, 2.0])
    value_domain = np.ones(4, dtype=bool)
    result = max_affine_threshold_oracle_labels(
        predictions=predictions,
        value_domain=value_domain,
        tau_true=5.0,
        values=values,
        use_calibrated_threshold=True,
        tie_tolerance=1e-9,
    )

    diagnostics = result["oracle_diagnostics"]
    assert "false_positive_count" in diagnostics
    assert "false_negative_count" in diagnostics
    assert "calibrated_prediction_threshold" in diagnostics
    assert "calibration_margin" in diagnostics
    assert result["true_improving_labels"].tolist() == [False, True, True, False]


def test_max_affine_adaptive_fixed_point_oracle_uses_register_comparison() -> None:
    from experiments.stage1_case14_t2_max_affine_adaptive_grover_search import (
        max_affine_threshold_oracle_labels,
    )

    values = np.array([5.0, 1.0, 4.0, 10.0])
    predictions = np.array([0.27, 0.24, 0.26, 1.0])
    value_domain = np.ones(4, dtype=bool)
    result = max_affine_threshold_oracle_labels(
        predictions=predictions,
        value_domain=value_domain,
        tau_true=5.0,
        values=values,
        use_calibrated_threshold=True,
        tie_tolerance=1e-9,
        oracle_mode="fixed_point_register",
        register_bits=2,
    )

    diagnostics = result["oracle_diagnostics"]
    assert diagnostics["oracle_mode"] == "fixed_point_register"
    assert diagnostics["register_bits"] == 2
    assert diagnostics["tau_register"] == 0
    assert result["marked"].tolist() == [True, True, True, False]
    assert diagnostics["false_positive_count"] == 1


def test_max_affine_adaptive_summary_separates_oracle_modes_and_is_strict_json() -> None:
    from experiments.stage1_case14_t2_max_affine_adaptive_grover_search import (
        sanitize_for_strict_json,
        write_strict_json,
    )

    summary = {
        "floating_comparator_oracle": {"calibration_margin": np.float64(1.25)},
        "fixed_point_register_oracle": {"tau_register": np.int64(3)},
        "not_finite": float("inf"),
    }
    cleaned = sanitize_for_strict_json(summary)
    path = Path("results/test_strict_json_summary.json")

    try:
        write_strict_json(path, summary)
        loaded = json.loads(path.read_text(encoding="utf-8"))

        assert cleaned["not_finite"] is None
        assert set(loaded) >= {
            "floating_comparator_oracle",
            "fixed_point_register_oracle",
        }
        assert "Infinity" not in path.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)


def test_small_sample_gate_level_learner_uses_only_train_samples() -> None:
    from experiments.stage1_case14_t2_small_sample_gate_level_max_affine_gas import (
        learn_small_sample_integer_max_affine_pieces,
    )

    train_bitstrings = ["0000", "1100", "1111", "0101"]
    train_values = np.array([4.0, 1.0, 8.0, 3.0])
    learned = learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=train_bitstrings,
        train_values=train_values,
        num_bits=4,
        num_pieces=2,
        max_weight=7,
        seed=0,
    )

    assert learned.diagnostics["train_sample_count"] == 4
    assert learned.diagnostics["train_best_bitstring"] == "1100"
    assert learned.used_training_indices == [0, 1, 2, 3]


def test_small_sample_gate_level_learner_builds_max_affine_spec() -> None:
    from experiments.stage1_case14_t2_small_sample_gate_level_max_affine_gas import (
        learn_small_sample_integer_max_affine_pieces,
        max_affine_spec_from_learned,
    )

    learned = learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=["0000", "1100", "1111"],
        train_values=np.array([4.0, 1.0, 8.0]),
        num_bits=4,
        num_pieces=2,
        max_weight=7,
        seed=0,
    )
    spec = max_affine_spec_from_learned(learned, tau_int=0)

    assert isinstance(spec, GateLevelMaxAffineOracleSpec)
    assert spec.num_x_qubits == 4
    assert all(isinstance(weight, int) for piece in spec.pieces for weight in piece.weights)


def test_small_sample_gate_level_builds_grover_circuit_with_iterations() -> None:
    from experiments.stage1_case14_t2_small_sample_gate_level_max_affine_gas import (
        build_gate_level_grover_circuit_for_threshold,
        learn_small_sample_integer_max_affine_pieces,
    )

    learned = learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=["0000", "1100", "1111"],
        train_values=np.array([4.0, 1.0, 8.0]),
        num_bits=4,
        num_pieces=2,
        max_weight=7,
        seed=0,
    )
    circuit = build_gate_level_grover_circuit_for_threshold(learned, tau_int=0, iterations=2)

    assert circuit.num_qubits > 4
    assert circuit.count_ops().get("measure", 0) == 4


def test_small_sample_gate_level_qasm_execution_returns_mappable_counts() -> None:
    from experiments.stage1_case14_t2_small_sample_gate_level_max_affine_gas import (
        build_gate_level_grover_circuit_for_threshold,
        execute_gate_level_circuit,
        learn_small_sample_integer_max_affine_pieces,
        measured_bitstring_to_index,
    )

    learned = learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=["0000", "1100", "1111"],
        train_values=np.array([4.0, 1.0, 8.0]),
        num_bits=4,
        num_pieces=2,
        max_weight=7,
        seed=0,
    )
    circuit = build_gate_level_grover_circuit_for_threshold(learned, tau_int=0, iterations=1)
    execution = execute_gate_level_circuit(circuit, backend="qasm", shots=64, seed=0)

    assert sum(execution["counts"].values()) == 64
    first_bitstring = next(iter(execution["counts"]))
    assert 0 <= measured_bitstring_to_index(first_bitstring) < 16


def test_small_sample_gate_level_adaptive_updates_only_on_true_ed_improvement() -> None:
    from experiments.stage1_case14_t2_small_sample_gate_level_max_affine_gas import (
        CandidateEvaluationCache,
        adaptive_gate_level_search,
        learn_small_sample_integer_max_affine_pieces,
    )

    learned = learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=["0000", "1100", "1111"],
        train_values=np.array([4.0, 1.0, 8.0]),
        num_bits=4,
        num_pieces=2,
        max_weight=7,
        seed=0,
    )
    cache = CandidateEvaluationCache({0: 4.0, 12: 1.0, 15: 8.0})
    result = adaptive_gate_level_search(
        learned=learned,
        evaluator=cache.evaluate,
        initial_index=0,
        backend="qasm",
        shots=64,
        seed=3,
        lambda_growth=8.0 / 7.0,
        max_rounds=2,
        max_trials_per_threshold=3,
        max_candidates_per_shotbatch=3,
    )

    for row in result["rounds"]:
        for trial in row["trials"]:
            before = trial["incumbent_true_cost_before"]
            after = trial["incumbent_true_cost_after"]
            if trial["accepted_update"]:
                assert after < before


def test_small_sample_gate_level_hidden_reference_is_not_algorithmic_ed_calls() -> None:
    from experiments.stage1_case14_t2_small_sample_gate_level_max_affine_gas import run

    summary = run(
        instance_path=Path("data/case14.json.gz"),
        results_path=Path("results/test_small_sample_gate_level_gas.json"),
        backend="qasm",
        shots=64,
        seed=1,
        lambda_growth=8.0 / 7.0,
        max_rounds=1,
        max_trials_per_threshold=1,
        selected_generator_indices=(0, 5),
        train_sample_count=3,
        initial_index="random",
        num_pieces=2,
        max_weight=7,
        save_qasm=False,
        draw_circuit=False,
        max_candidates_per_shotbatch=2,
    )

    try:
        assert "hidden_reference_not_used_by_algorithm" in summary
        assert summary["ed_calls"]["hidden_reference"] > 0
        assert summary["ed_calls"]["total_algorithmic"] == (
            summary["ed_calls"]["training"]
            + summary["ed_calls"]["search_verification"]
        )
    finally:
        Path("results/test_small_sample_gate_level_gas.json").unlink(missing_ok=True)
def test_tie_tolerant_threshold_keeps_nearly_equal_costs_together() -> None:
    values = np.array([1.0, 2.0, 2.0 + 1e-12, 3.0])
    strict = tie_tolerant_threshold_case_for_top_count(values, 2, 0.0)
    tolerant = tie_tolerant_threshold_case_for_top_count(values, 2, 1e-9)

    assert strict["actual_target_count"] == 2
    assert tolerant["actual_target_count"] == 3


def test_structured_features_are_finite_and_expand_with_local_order() -> None:
    instance = leading_time_window_instance(load_uc_instance("data/case14.json.gz"), 2)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    order1 = structured_commitment_features(
        instance,
        commitments,
        same_time_interaction_order=1,
        include_dispatch_proxy=True,
    )
    order4 = structured_commitment_features(
        instance,
        commitments,
        same_time_interaction_order=4,
        include_dispatch_proxy=True,
    )

    assert order1.features.shape[0] == commitments.shape[0]
    assert order4.features.shape[0] == commitments.shape[0]
    assert order4.features.shape[1] > order1.features.shape[1]
    assert len(order4.names) == order4.features.shape[1]
    assert len(set(order4.names)) == len(order4.names)
    assert np.all(np.isfinite(order4.features))


def test_calibrated_prediction_threshold_detects_rank_separation() -> None:
    predictions = np.array([1.0, 2.0, 5.0, 6.0])
    labels = np.array([True, True, False, False])
    value_domain = np.ones(4, dtype=bool)

    threshold, margin = calibrated_prediction_threshold(predictions, labels, value_domain)

    assert np.isclose(threshold, 3.5)
    assert np.isclose(margin, 3.0)


def test_max_affine_model_represents_simple_convex_function() -> None:
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    features = np.column_stack([np.ones_like(x), x])
    values = np.maximum(0.0, x)

    model, diagnostics = fit_max_affine_value_function(
        features,
        values,
        ["1", "x"],
        piece_count=2,
        candidate_count=features.shape[0],
        initialization="floor",
    )

    assert diagnostics.actual_piece_count == 2
    assert np.allclose(model.predict(features), values)


def test_max_affine_gate_counts_track_piece_comparators() -> None:
    counts = max_affine_gate_counts(feature_count=5, piece_count=3)

    assert counts["affine_accumulators"] == 3
    assert counts["affine_weighted_additions"] == 15
    assert counts["max_comparators"] == 2
    assert counts["threshold_comparators"] == 1


def test_max_affine_weighted_least_squares_shifts_fit_toward_weighted_states() -> None:
    x = np.array([0.0, 1.0, 2.0])
    features = np.column_stack([np.ones_like(x), x])
    values = np.array([0.0, 0.0, 10.0])

    unweighted_model, _ = fit_max_affine_value_function(
        features,
        values,
        ["1", "x"],
        piece_count=1,
        initialization="least_squares",
    )
    weighted_model, _ = fit_max_affine_value_function(
        features,
        values,
        ["1", "x"],
        piece_count=1,
        initialization="least_squares",
        sample_weights=np.array([1.0, 1.0, 25.0]),
    )

    unweighted_error = abs(unweighted_model.predict(features)[-1] - values[-1])
    weighted_error = abs(weighted_model.predict(features)[-1] - values[-1])

    assert weighted_error < unweighted_error


def test_boundary_candidate_order_prioritizes_heavier_boundary_weights() -> None:
    order = boundary_candidate_order(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 8.0, 6.0, 1.0]),
    )

    assert list(order[:2]) == [1, 2]
    assert set(order.tolist()) == {0, 1, 2, 3}


