from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.stage1_case14_t2_gate_level_grover_oracle import (
    case14_t2_gate_level_proxy_spec,
)
from experiments.stage1_case14_t2_gate_level_max_affine_oracle import (
    case14_t2_gate_level_max_affine_spec,
)
from experiments.stage1_case14_t2_learned_small_max_affine_gate_level_oracle import (
    calibrate_integer_threshold,
    learn_gap_weighted_max_affine_pieces,
)
from experiments.stage1_case14_t2_small_sample_gate_level_max_affine_gas import (
    CandidateEvaluationCache,
    adaptive_gate_level_search,
    build_gate_level_grover_circuit_for_threshold,
    execute_gate_level_circuit,
    learn_small_sample_integer_max_affine_pieces,
    max_affine_spec_from_learned,
    measured_bitstring_to_index,
    run as run_small_sample_gas,
)
from experiments.stage1_case14_t2_small_sample_gate_level_gas_sweep import (
    build_grouped_summary,
    parse_configs,
    parse_train_sample_counts,
)
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring
from qubit_value_function.ed import FixedCommitmentEvaluator
from qubit_value_function.experiment_utils import (
    commitment_row,
    embedded_selected_commitments,
    evaluate_values,
    leading_time_window_instance,
    write_strict_json,
)
from qubit_value_function.gate_level_oracle import (
    GateLevelAffinePieceSpec,
    GateLevelAffineOracleSpec,
    GateLevelMaxAffineOracleSpec,
    bitstring_from_index,
    build_max_affine_grover_circuit,
    build_max_affine_phase_oracle_circuit,
    circuit_resource_summary,
    simulate_affine_phase_oracle,
    simulate_max_affine_grover,
    simulate_max_affine_phase_oracle,
)
from qubit_value_function.qft_weighted_sum_oracle import (
    build_qft_max_affine_phase_oracle_circuit,
    simulate_qft_max_affine_grover,
    simulate_qft_max_affine_phase_oracle,
)
from qubit_value_function.uc_loader import load_uc_instance


def test_load_aelmp_simple_instance_and_ed_best_commitment() -> None:
    instance = load_uc_instance("data/aelmp_simple.json.gz")
    evaluator = FixedCommitmentEvaluator(instance)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    rows = [
        (commitment_to_bitstring(commitment), evaluator.evaluate(commitment).total_cost)
        for commitment in commitments
    ]

    best_bitstring, best_cost = min(rows, key=lambda row: row[1])

    assert instance.time_horizon == 1
    assert [gen.name for gen in instance.generators] == ["W", "X", "Y"]
    assert best_bitstring == "101"
    assert np.isfinite(best_cost)


def test_experiment_utils_embed_and_commitment_row_for_case14_t2() -> None:
    instance = leading_time_window_instance(load_uc_instance("data/case14.json.gz"), 2)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    values, logic_feasible = evaluate_values(instance, commitments)
    best_idx = int(np.nanargmin(values))
    embedded = embedded_selected_commitments(commitments[best_idx], (0, 5))
    row = commitment_row(commitments, [gen.name for gen in instance.generators], values, best_idx)

    assert commitments.shape == (4096, 6, 2)
    assert int(logic_feasible.sum()) == 768
    assert embedded.shape == (16, 6, 2)
    assert row["bitstring_generator_major"] == "110011111100"
    assert np.isclose(row["total_cost"], 20578.2152604)


def test_strict_json_writer_removes_nonfinite_values() -> None:
    path = Path("results/test_strict_json_summary.json")
    try:
        write_strict_json(path, {"finite": np.float64(1.25), "bad": float("inf")})
        text = path.read_text(encoding="utf-8")
        loaded = json.loads(text)
        assert loaded == {"finite": 1.25, "bad": None}
        assert "Infinity" not in text
    finally:
        path.unlink(missing_ok=True)


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


def test_gate_level_max_affine_oracle_and_grover_mark_expected_state() -> None:
    spec = GateLevelMaxAffineOracleSpec(
        pieces=(
            GateLevelAffinePieceSpec((1, 1, 1, 0), (0, 1)),
            GateLevelAffinePieceSpec((0, 0, 1, 1)),
        ),
        threshold=0,
    )
    probe = simulate_max_affine_phase_oracle(spec)
    result = simulate_max_affine_grover(spec)
    resources = circuit_resource_summary(build_max_affine_phase_oracle_circuit(spec), decompose_reps=0)

    marked = [bitstring_from_index(index, spec.num_x_qubits) for index in np.flatnonzero(spec.marked_mask())]
    assert marked == ["1100"]
    assert probe.aux_zero_probability > 1.0 - 1e-12
    assert result.marked_probability > 0.9
    assert resources["operations"]["adder"] == 2
    assert resources["operations"]["cmp"] == 2


def test_qft_weighted_sum_max_affine_matches_weighted_adder_marking() -> None:
    spec = GateLevelMaxAffineOracleSpec(
        pieces=(
            GateLevelAffinePieceSpec((7, 6, 0, 0), (0, 1)),
            GateLevelAffinePieceSpec((0, 0, 1, 1)),
        ),
        threshold=0,
    )
    weighted_probe = simulate_max_affine_phase_oracle(spec)
    qft_probe = simulate_qft_max_affine_phase_oracle(spec)
    qft_result = simulate_qft_max_affine_grover(spec)
    qft_resources = circuit_resource_summary(
        build_qft_max_affine_phase_oracle_circuit(spec),
        decompose_reps=0,
    )

    assert np.array_equal(qft_probe.marked_mask, weighted_probe.marked_mask)
    assert qft_probe.aux_zero_probability > 1.0 - 1e-12
    assert qft_result.marked_probability > 0.9
    assert any(name.startswith("qft_sum") for name in qft_resources["operations"])


def test_case14_t2_gate_level_proxy_marks_expected_subregister() -> None:
    instance = leading_time_window_instance(load_uc_instance("data/case14.json.gz"), 2)
    spec = case14_t2_gate_level_proxy_spec(instance, (0, 1, 2))
    marked = [bitstring_from_index(index, spec.num_x_qubits) for index in np.flatnonzero(spec.marked_mask())]

    assert spec.bit_labels == ("g1_t0", "g1_t1", "g2_t0", "g2_t1", "g3_t0", "g3_t1")
    assert marked == ["110011"]


def test_case14_t2_gate_level_max_affine_marks_expected_subregister() -> None:
    instance = leading_time_window_instance(load_uc_instance("data/case14.json.gz"), 2)
    spec = case14_t2_gate_level_max_affine_spec(instance, (0, 1))
    marked = [bitstring_from_index(index, spec.num_x_qubits) for index in np.flatnonzero(spec.marked_mask())]

    assert spec.bit_labels == ("g1_t0", "g1_t1", "g2_t0", "g2_t1")
    assert marked == ["1100"]


def test_learned_small_max_affine_uses_true_cost_gaps_in_selected_subspace() -> None:
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
    top3_spec = GateLevelMaxAffineOracleSpec(
        pieces=learned.pieces,
        threshold=int(
            calibrate_integer_threshold(
                predicted_values=GateLevelMaxAffineOracleSpec(
                    pieces=learned.pieces,
                    threshold=0,
                    bit_labels=learned.bit_labels,
                ).values_for_all_x(),
                true_values=embedded_values,
                target_count=3,
            )["selected_threshold"]
        ),
        bit_labels=learned.bit_labels,
    )

    marked = [bitstring_from_index(index, top3_spec.num_x_qubits) for index in np.flatnonzero(top3_spec.marked_mask())]
    assert learned.integer_weights == (7, 6, 1, 1)
    assert marked == ["1100", "1110", "1101"]


def test_small_sample_gate_level_learner_uses_only_train_samples() -> None:
    learned = learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=["0000", "1100", "1111", "0101"],
        train_values=np.array([4.0, 1.0, 8.0, 3.0]),
        num_bits=4,
        num_pieces=2,
        max_weight=7,
        seed=0,
    )

    assert learned.diagnostics["train_sample_count"] == 4
    assert learned.diagnostics["train_best_bitstring"] == "1100"
    assert learned.used_training_indices == [0, 1, 2, 3]


def test_small_sample_gate_level_learner_builds_max_affine_spec() -> None:
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
            if trial["accepted_update"]:
                assert trial["incumbent_true_cost_after"] < trial["incumbent_true_cost_before"]


def test_small_sample_gate_level_hidden_reference_is_not_algorithmic_ed_calls() -> None:
    path = Path("results/test_small_sample_gate_level_gas.json")
    summary = run_small_sample_gas(
        instance_path=Path("data/case14.json.gz"),
        results_path=path,
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
            summary["ed_calls"]["training"] + summary["ed_calls"]["search_verification"]
        )
    finally:
        path.unlink(missing_ok=True)


def test_small_sample_gate_level_run_can_exclude_hidden_optimum_from_training() -> None:
    path = Path("results/test_small_sample_gate_level_gas_exclude_hidden.json")
    summary = run_small_sample_gas(
        instance_path=Path("data/case14.json.gz"),
        results_path=path,
        backend="qasm",
        shots=64,
        seed=1,
        lambda_growth=8.0 / 7.0,
        max_rounds=1,
        max_trials_per_threshold=1,
        selected_generator_indices=(0, 5),
        train_sample_count=4,
        initial_index="random",
        num_pieces=2,
        max_weight=7,
        save_qasm=False,
        draw_circuit=False,
        max_candidates_per_shotbatch=1,
        exclude_hidden_optimum_from_training=True,
    )

    try:
        assert summary["exclude_hidden_optimum_from_training"] is True
        assert summary["training_contains_hidden_optimum"] is False
        assert summary["hidden_best_index"] not in summary["train_indices"]
        assert summary["hidden_best_bitstring"] not in summary["train_bitstrings"]
        assert summary["algorithmic_ed_lp_calls"] == summary["ed_calls"]["total_algorithmic"]
        assert summary["hidden_reference_ed_lp_calls"] == summary["ed_calls"]["hidden_reference"]
        assert summary["found_hidden_exact_optimum"] == (
            summary["final_bitstring"] == summary["hidden_best_bitstring"]
        )
        allocation = summary["learned_max_affine_oracle"]["register_allocation"]
        assert len(allocation) == summary["learned_max_affine_oracle"]["training_diagnostics"]["num_pieces"]
        assert {
            "piece_index",
            "weights",
            "bias",
            "inverted_bit_indices",
            "max_weighted_sum",
            "value_register_bits",
            "compare_value",
            "flag_qubits",
            "carry_qubits",
            "control_qubits",
            "comparator_ancillas",
        }.issubset(allocation[0])
    finally:
        path.unlink(missing_ok=True)


def test_sweep_parsers_accept_config_and_train_sample_lists() -> None:
    assert parse_configs("0,5;0,1,5") == [(0, 5), (0, 1, 5)]
    assert parse_train_sample_counts("4,8,12") == [4, 8, 12]


def test_sweep_grouped_summary_computes_success_rates() -> None:
    runs = [
        {
            "selected_generators": [0, 5],
            "num_search_qubits": 4,
            "train_sample_count": 4,
            "exclude_hidden_optimum_from_training": True,
            "training_contains_hidden_optimum": False,
            "found_hidden_exact_optimum": True,
            "status": "ok",
            "algorithmic_ed_lp_calls": 6,
            "circuit_executions": 2,
            "total_shots": 200,
            "max_qubits": 18,
            "max_circuit_depth": 20,
            "max_transpiled_depth": 100,
        },
        {
            "selected_generators": [0, 5],
            "num_search_qubits": 4,
            "train_sample_count": 4,
            "exclude_hidden_optimum_from_training": True,
            "training_contains_hidden_optimum": True,
            "found_hidden_exact_optimum": False,
            "status": "ok",
            "algorithmic_ed_lp_calls": 8,
            "circuit_executions": 4,
            "total_shots": 400,
            "max_qubits": 20,
            "max_circuit_depth": 40,
            "max_transpiled_depth": 200,
        },
    ]

    grouped = build_grouped_summary(runs)

    assert len(grouped) == 1
    row = grouped[0]
    assert row["num_runs"] == 2
    assert row["num_success"] == 1
    assert row["success_rate"] == 0.5
    assert row["num_runs_hidden_not_in_training"] == 1
    assert row["success_rate_when_hidden_optimum_not_in_training"] == 1.0
    assert row["training_contains_hidden_optimum_rate"] == 0.5
    assert row["avg_algorithmic_ed_lp_calls"] == 7.0
