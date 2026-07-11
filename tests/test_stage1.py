from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import experiments.stage1_case14_t2_small_sample_gate_level_max_affine_gas as small_sample_gas_module

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
    pairwise_hinge_diagnostics,
    run as run_small_sample_gas,
    select_best_measured_candidate,
)
from experiments.stage1_case14_t2_small_sample_gate_level_gas_surrogate_sweep import (
    _markdown_summary as surrogate_markdown_summary,
    build_surrogate_grouped_summary,
)
from experiments.stage1_case14_t2_small_sample_gate_level_gas_diagnostics import (
    classify_run_failure,
    hidden_perfect_uniform_marked_run,
    _markdown_summary,
)
from experiments.stage1_case14_t2_small_sample_gate_level_gas_sweep import (
    build_grouped_summary,
    parse_configs,
    parse_train_sample_counts,
)
from qubit_value_function.diagnostics import (
    gap_metrics,
    hypergeometric_hit_probability,
    random_baseline_probabilities,
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


def test_accepted_refit_excludes_verified_but_rejected_candidates(monkeypatch) -> None:
    learned = learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=[bitstring_from_index(0, 2), bitstring_from_index(3, 2)],
        train_values=np.array([5.0, 9.0]),
        num_bits=2,
        num_pieces=1,
        max_weight=7,
        seed=0,
    )
    _patch_adaptive_circuit_execution(
        monkeypatch,
        counts_by_call=[
            {
                bitstring_from_index(1, 2): 10,
                bitstring_from_index(2, 2): 9,
            }
        ],
    )

    result = adaptive_gate_level_search(
        learned=learned,
        evaluator=CandidateEvaluationCache({0: 5.0, 1: 6.0, 2: 4.0, 3: 9.0}).evaluate,
        initial_index=0,
        backend="qasm",
        shots=8,
        seed=0,
        lambda_growth=8.0 / 7.0,
        max_rounds=1,
        max_trials_per_threshold=1,
        max_candidates_per_shotbatch=2,
        refit_policy="accepted",
        learner_name="mismatch",
        num_pieces=1,
        max_weight=7,
    )

    assert result["verified_candidates"] == 2
    assert result["refit_count"] == 1
    assert result["observed_sample_count"] == 3
    assert result["observed_indices"] == [0, 2, 3]
    assert result["refit_history"][-1]["observed_indices"] == [0, 2, 3]


def test_round_logging_uses_pre_refit_oracle_snapshot(monkeypatch) -> None:
    learned = learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=[bitstring_from_index(0, 2), bitstring_from_index(3, 2)],
        train_values=np.array([5.0, 9.0]),
        num_bits=2,
        num_pieces=1,
        max_weight=7,
        seed=0,
    )
    refit_piece = GateLevelAffinePieceSpec(
        weights=(7, 0),
        inverted_bit_indices=(),
        name="refit-piece",
        bias=3,
    )
    refit_learned = small_sample_gas_module.SmallSampleLearnedOracle(
        pieces=(refit_piece,),
        bit_labels=learned.bit_labels,
        train_bitstrings=learned.train_bitstrings,
        train_values=learned.train_values,
        used_training_indices=learned.used_training_indices,
        diagnostics={**learned.diagnostics, "learner_fallback": None},
    )
    monkeypatch.setattr(small_sample_gas_module, "_refit_from_observed", lambda *_args, **_kwargs: refit_learned)
    _patch_adaptive_circuit_execution(
        monkeypatch,
        counts_by_call=[
            {bitstring_from_index(2, 2): 10},
            {bitstring_from_index(0, 2): 10},
        ],
    )

    result = adaptive_gate_level_search(
        learned=learned,
        evaluator=CandidateEvaluationCache({0: 5.0, 2: 4.0, 3: 9.0}).evaluate,
        initial_index=0,
        backend="qasm",
        shots=8,
        seed=0,
        lambda_growth=8.0 / 7.0,
        max_rounds=2,
        max_trials_per_threshold=1,
        max_candidates_per_shotbatch=1,
        refit_policy="accepted",
        learner_name="mismatch",
        num_pieces=1,
        max_weight=7,
    )

    first_round, second_round = result["rounds"]
    assert first_round["refit_version_before"] == 0
    assert first_round["refit_version_after"] == 1
    assert first_round["oracle_pieces_before"][0]["name"] != "refit-piece"
    assert second_round["refit_version_before"] == 1
    assert second_round["oracle_pieces_before"][0]["name"] == "refit-piece"


def test_adaptive_search_stops_cleanly_when_calibration_marks_no_states(monkeypatch) -> None:
    learned = small_sample_gas_module.SmallSampleLearnedOracle(
        pieces=(GateLevelAffinePieceSpec(weights=(0, 0), inverted_bit_indices=(), name="empty", bias=1),),
        bit_labels=("x0", "x1"),
        train_bitstrings=(bitstring_from_index(0, 2), bitstring_from_index(3, 2)),
        train_values=(5.0, 9.0),
        used_training_indices=[0, 3],
        diagnostics={"learner": "rank_hinge", "learner_fallback": None},
    )
    monkeypatch.setattr(
        small_sample_gas_module,
        "calibrate_integer_threshold_from_samples_or_incumbent",
        lambda **_kwargs: {"tau_int": 0},
    )

    result = adaptive_gate_level_search(
        learned=learned,
        evaluator=CandidateEvaluationCache({0: 5.0, 3: 9.0}).evaluate,
        initial_index=0,
        backend="qasm",
        shots=8,
        seed=0,
        lambda_growth=8.0 / 7.0,
        max_rounds=1,
        max_trials_per_threshold=1,
        max_candidates_per_shotbatch=1,
    )

    assert result["stop_reason"] == "no_marked_state_at_threshold"
    assert result["rounds"][0]["trials"] == []
    assert result["total_quantum_circuit_executions"] == 0


def _patch_adaptive_circuit_execution(monkeypatch, *, counts_by_call: list[dict[str, int]]) -> None:
    class _Circuit:
        def remove_final_measurements(self, *, inplace: bool = False):
            return self

    calls = iter(counts_by_call)
    monkeypatch.setattr(
        small_sample_gas_module,
        "build_gate_level_grover_circuit_for_threshold",
        lambda *_args, **_kwargs: _Circuit(),
    )
    monkeypatch.setattr(small_sample_gas_module, "build_max_affine_phase_oracle_circuit", lambda *_args, **_kwargs: _Circuit())
    monkeypatch.setattr(
        small_sample_gas_module,
        "execute_gate_level_circuit",
        lambda *_args, **_kwargs: {
            "backend_name": "synthetic",
            "counts": next(calls),
            "transpiled_depth": 1,
            "transpiled_ops": {},
        },
    )
    monkeypatch.setattr(
        small_sample_gas_module,
        "circuit_resource_summary",
        lambda *_args, **_kwargs: {"num_qubits": 2, "depth": 1},
    )


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
        assert "register_allocation" not in summary["learned_max_affine_oracle"]
        allocation = summary["learned_max_affine_oracle"]["reference_register_allocation_at_tau0"]
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
        for round_row in summary["adaptive_search"]["rounds"]:
            assert "register_allocation" in round_row
            assert round_row["register_allocation"][0]["compare_value"] >= 0
    finally:
        path.unlink(missing_ok=True)


def test_small_sample_gate_level_run_can_exclude_hidden_optimum_from_random_initial() -> None:
    path = Path("results/test_small_sample_gate_level_gas_exclude_hidden_initial.json")
    summary = run_small_sample_gas(
        instance_path=Path("data/case14.json.gz"),
        results_path=path,
        backend="qasm",
        shots=64,
        seed=7,
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
        exclude_hidden_optimum_from_initial=True,
    )

    try:
        assert summary["exclude_hidden_optimum_from_initial"] is True
        assert summary["initial_index"] != summary["hidden_best_index"]
        assert summary["initial_bitstring"] != summary["hidden_best_bitstring"]
        assert summary["initial_matches_hidden_optimum"] is False
        assert summary["initial_true_cost"] is not None
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
            "initial_matches_hidden_optimum": False,
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
            "initial_matches_hidden_optimum": False,
            "found_hidden_exact_optimum": False,
            "status": "ok",
            "algorithmic_ed_lp_calls": 8,
            "circuit_executions": 4,
            "total_shots": 400,
            "max_qubits": 20,
            "max_circuit_depth": 40,
            "max_transpiled_depth": 200,
        },
        {
            "selected_generators": [0, 5],
            "num_search_qubits": 4,
            "train_sample_count": 4,
            "exclude_hidden_optimum_from_training": True,
            "training_contains_hidden_optimum": None,
            "initial_matches_hidden_optimum": None,
            "found_hidden_exact_optimum": None,
            "status": "error",
            "algorithmic_ed_lp_calls": None,
            "circuit_executions": None,
            "total_shots": None,
            "max_qubits": None,
            "max_circuit_depth": None,
            "max_transpiled_depth": None,
        },
    ]

    grouped = build_grouped_summary(runs)

    assert len(grouped) == 1
    row = grouped[0]
    assert row["num_runs"] == 3
    assert row["num_ok_runs"] == 2
    assert row["num_error_runs"] == 1
    assert row["num_success"] == 1
    assert row["success_rate"] == 0.5
    assert row["success_rate_over_ok_runs"] == 0.5
    assert row["num_runs_hidden_not_in_training"] == 1
    assert row["success_rate_when_hidden_optimum_not_in_training"] == 1.0
    assert row["num_runs_hidden_not_in_training_and_not_initial"] == 1
    assert row["success_rate_when_hidden_optimum_not_in_training_and_not_initial"] == 1.0
    assert row["training_contains_hidden_optimum_rate"] == 0.5
    assert row["avg_algorithmic_ed_lp_calls"] == 7.0


def test_gap_metrics_cover_exact_near_optimal_and_nonfinite_values() -> None:
    exact = gap_metrics(100.0, 100.0)
    near = gap_metrics(102.0, 100.0)
    bad = gap_metrics(float("inf"), 100.0)

    assert exact["absolute_gap_to_hidden_best"] == 0.0
    assert exact["success_within_1_percent"] is True
    assert near["relative_gap_to_hidden_best"] == 0.02
    assert near["success_within_1_percent"] is False
    assert near["success_within_3_percent"] is True
    assert bad["absolute_gap_to_hidden_best"] is None
    assert bad["success_within_5_percent"] is False


def test_random_baseline_without_replacement_probability() -> None:
    assert hypergeometric_hit_probability(dimension=16, num_good_states=1, draws=1) == 1 / 16
    assert hypergeometric_hit_probability(dimension=16, num_good_states=1, draws=16) == 1.0
    assert np.isclose(
        hypergeometric_hit_probability(dimension=16, num_good_states=2, draws=2),
        1.0 - (14 * 13) / (16 * 15),
    )
    baseline = random_baseline_probabilities(
        values=np.array([1.0, 1.01, 1.04, 2.0]),
        hidden_best_true_cost=1.0,
        draws=1,
    )
    assert baseline["random_exact_success_probability"] == 0.25
    assert baseline["random_success_within_1_percent_probability"] == 0.5
    assert baseline["random_success_within_5_percent_probability"] == 0.75


def test_measurement_policy_records_single_shot_and_shot_batch() -> None:
    path = Path("results/test_measurement_policy_single_shot.json")
    summary = run_small_sample_gas(
        instance_path=Path("data/case14.json.gz"),
        results_path=path,
        backend="qasm",
        shots=64,
        seed=2,
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
        max_candidates_per_shotbatch=3,
        exclude_hidden_optimum_from_training=True,
        exclude_hidden_optimum_from_initial=True,
        measurement_policy="single_shot",
    )

    try:
        assert summary["measurement_policy"] == "single_shot"
        assert summary["shots_per_circuit"] == 1
        assert summary["adaptive_search"]["total_shots"] == summary["adaptive_search"]["total_quantum_circuit_executions"]
        assert summary["verified_candidates"] >= summary["ed_calls"]["search_verification"]
    finally:
        path.unlink(missing_ok=True)


def test_candidate_budget_evaluates_unique_top_k_bitstrings() -> None:
    cache = CandidateEvaluationCache({0: 5.0, 3: 1.0, 7: 2.0})
    selected_index, selected_bitstring, selected_cost, evaluated = select_best_measured_candidate(
        [
            {"bitstring": "000", "count": 9},
            {"bitstring": "110", "count": 8},
            {"bitstring": "110", "count": 7},
            {"bitstring": "111", "count": 6},
        ],
        cache.evaluate,
        max_candidates=3,
    )

    assert selected_index == 3
    assert selected_bitstring == "110"
    assert selected_cost == 1.0
    assert [row["index"] for row in evaluated] == [0, 3, 7]


def test_refit_policy_and_pairwise_learner_are_reported_without_hidden_training() -> None:
    path = Path("results/test_refit_pairwise.json")
    summary = run_small_sample_gas(
        instance_path=Path("data/case14.json.gz"),
        results_path=path,
        backend="qasm",
        shots=64,
        seed=3,
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
        exclude_hidden_optimum_from_initial=True,
        learner="pairwise_ranking",
        refit_policy="accepted",
    )

    try:
        assert summary["learner"] == "pairwise_ranking"
        assert summary["refit_policy"] == "accepted"
        assert summary["learned_max_affine_oracle"]["training_diagnostics"]["learner_name"] == "pairwise_ranking"
        assert summary["adaptive_search"]["observed_sample_count"] >= summary["train_sample_count"]
        assert summary["hidden_best_index"] not in summary["adaptive_search"]["observed_indices"]
    finally:
        path.unlink(missing_ok=True)


def test_rank_hinge_learner_outputs_integer_max_affine_oracle_and_diagnostics() -> None:
    learned = learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=["0000", "1000", "1100", "1110", "1111"],
        train_values=np.array([9.0, 7.0, 3.0, 2.0, 8.0]),
        num_bits=4,
        num_pieces=2,
        max_weight=7,
        seed=0,
        learner="rank_hinge",
    )
    spec = max_affine_spec_from_learned(learned, tau_int=0)

    assert spec.num_x_qubits == 4
    assert learned.diagnostics["learner_name"] == "rank_hinge"
    assert "learner_fallback" in learned.diagnostics
    assert "pairwise_hinge_loss" in learned.diagnostics
    assert "num_pairwise_violations" in learned.diagnostics
    for piece in learned.pieces:
        assert isinstance(piece.bias, int)
        for weight in piece.weights:
            assert isinstance(weight, int)
            assert 0 <= weight <= 7


def test_pairwise_hinge_diagnostics_count_violations_and_loss() -> None:
    diagnostics = pairwise_hinge_diagnostics(
        true_values=np.array([1.0, 2.0, 3.0]),
        predicted_values=np.array([0.0, 0.0, 4.0]),
        margin=1,
    )

    assert diagnostics["num_pairs"] == 3
    assert diagnostics["num_pairwise_violations"] == 1
    assert diagnostics["pairwise_hinge_loss"] == 1.0
    assert diagnostics["train_pairwise_order_accuracy"] == 2 / 3


def test_rank_hinge_records_fallback_field_for_tiny_training_set() -> None:
    learned = learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=["0"],
        train_values=np.array([1.0]),
        num_bits=1,
        num_pieces=1,
        max_weight=3,
        seed=0,
        learner="rank_hinge",
    )

    assert learned.diagnostics["learner_fallback"] is not None


def test_surrogate_sweep_grouping_tracks_learner_refit_and_skips() -> None:
    runs = [
        {
            "learner": "rank_hinge",
            "refit_policy": "accepted",
            "num_search_qubits": 4,
            "status": "ok",
            "found_hidden_exact_optimum": True,
            "success_within_3_percent": True,
            "success_within_1_percent": True,
            "success_within_5_percent": True,
            "algorithmic_ed_lp_calls": 10,
            "total_shots": 2000,
            "max_qubits": 20,
            "max_transpiled_depth": 100,
            "pairwise_hinge_loss": 2.0,
            "train_pairwise_order_accuracy": 0.75,
            "refit_count": 1,
            "observed_sample_count": 6,
            "random_exact_success_probability": 0.25,
        },
        {
            "learner": "rank_hinge",
            "refit_policy": "accepted",
            "num_search_qubits": 4,
            "status": "skipped_resource_limit",
        },
        {
            "learner": "rank_hinge",
            "refit_policy": "accepted",
            "num_search_qubits": 4,
            "status": "skipped_invalid_config",
        },
    ]

    grouped = build_surrogate_grouped_summary(runs)
    row = grouped["by_learner"][0]

    assert row["learner"] == "rank_hinge"
    assert row["refit_policy"] is None
    assert row["num_ok_runs"] == 1
    assert row["num_skipped_resource_limit_runs"] == 1
    assert row["num_skipped_invalid_config_runs"] == 1
    assert row["exact_success_rate"] == 1.0
    assert row["within_3_percent_success_rate"] == 1.0
    assert row["avg_pairwise_order_accuracy"] == 0.75
    assert row["avg_algorithmic_ed_lp_calls"] == 10.0

    markdown = surrogate_markdown_summary(
        {
            "fixed_settings": {
                "backend": "qasm",
                "shots": 2000,
                "measurement_policy": "shot_batch",
                "max_candidates_per_shotbatch": 3,
                "oracle_mode": "learned",
                "exclude_hidden_optimum_from_training": True,
                "exclude_hidden_optimum_from_initial": True,
            },
            "runs": runs,
            "grouped_summary": grouped,
        }
    )

    assert "skipped invalid config runs" in markdown
    assert "skipped resource limit runs" in markdown
    assert "fallback_rate" in markdown
    assert "## Learner × Refit" in markdown
    assert "## Formal-Sweep Decision" in markdown


def test_hidden_perfect_uniform_marked_budget_does_not_imply_success() -> None:
    result = hidden_perfect_uniform_marked_run(
        selected_generator_indices=(0,),
        train_sample_count=1,
        seed=1,
        search_verification_budget=1,
        values=np.array([10.0, 8.0, 6.0, 1.0]),
        initial_index=0,
    )

    assert result["oracle_mode"] == "hidden_perfect_uniform_marked"
    assert result["diagnostic_only"] is True
    assert result["is_gate_level"] is False
    assert result["uses_hidden_reference_for_sampling"] is True
    assert result["training_contains_hidden_optimum"] is False
    assert result["initial_matches_hidden_optimum"] is False
    assert result["search_verification_calls"] == 1
    assert result["found_hidden_exact_optimum"] is False


def test_single_shot_repeated_respects_equal_shot_limits() -> None:
    path = Path("results/test_single_shot_repeated.json")
    summary = run_small_sample_gas(
        instance_path=Path("data/case14.json.gz"),
        results_path=path,
        backend="qasm",
        shots=64,
        seed=4,
        lambda_growth=8.0 / 7.0,
        max_rounds=4,
        max_trials_per_threshold=4,
        selected_generator_indices=(0, 5),
        train_sample_count=4,
        initial_index="random",
        num_pieces=2,
        max_weight=7,
        save_qasm=False,
        draw_circuit=False,
        max_candidates_per_shotbatch=1,
        exclude_hidden_optimum_from_training=True,
        exclude_hidden_optimum_from_initial=True,
        measurement_policy="single_shot_repeated",
        max_total_shots_per_run=2,
        max_total_circuit_executions_per_run=2,
    )

    try:
        assert summary["measurement_policy"] == "single_shot_repeated"
        assert summary["shots_per_circuit"] == 1
        assert summary["total_shots"] <= 2
        assert summary["circuit_executions"] <= 2
    finally:
        path.unlink(missing_ok=True)


def test_resource_limit_errors_are_classified_as_skipped() -> None:
    status, error_type = classify_run_failure(
        "'Number of qubits (30) in small_sample_gate_level_tau_0 is greater than maximum (29) in the coupling_map'"
    )

    assert status == "skipped_resource_limit"
    assert error_type == "resource_limit"


def test_diagnostic_grouped_summary_counts_resource_limit_skips() -> None:
    from experiments.stage1_case14_t2_small_sample_gate_level_gas_diagnostics import _group_diagnostics

    rows = [
        {
            "selected_generators": [0, 5],
            "num_search_qubits": 4,
            "train_sample_count": 4,
            "measurement_policy": "shot_batch",
            "max_candidates_per_shotbatch": 1,
            "refit_policy": "none",
            "learner": "mismatch",
            "oracle_mode": "learned",
            "status": "ok",
            "found_hidden_exact_optimum": True,
            "success_within_1_percent": True,
            "success_within_3_percent": True,
            "success_within_5_percent": True,
            "algorithmic_ed_lp_calls": 6,
        },
        {
            "selected_generators": [0, 5],
            "num_search_qubits": 4,
            "train_sample_count": 4,
            "measurement_policy": "shot_batch",
            "max_candidates_per_shotbatch": 1,
            "refit_policy": "none",
            "learner": "mismatch",
            "oracle_mode": "learned",
            "status": "skipped_resource_limit",
            "error_type": "resource_limit",
        },
        {
            "selected_generators": [0, 5],
            "num_search_qubits": 4,
            "train_sample_count": 4,
            "measurement_policy": "shot_batch",
            "max_candidates_per_shotbatch": 1,
            "refit_policy": "none",
            "learner": "mismatch",
            "oracle_mode": "learned",
            "status": "error",
        },
    ]

    summary = _group_diagnostics(rows)[0]

    assert summary["num_ok_runs"] == 1
    assert summary["num_error_runs"] == 1
    assert summary["num_skipped_resource_limit_runs"] == 1


def test_diagnostic_markdown_summary_includes_requested_comparison_sections() -> None:
    learned_row = {
        "selected_generators": [0, 5],
        "num_search_qubits": 4,
        "dimension": 16,
        "seed": 0,
        "train_sample_count": 4,
        "measurement_policy": "shot_batch",
        "shots": 2000,
        "shots_per_circuit": 2000,
        "max_candidates_per_shotbatch": 3,
        "refit_policy": "accepted",
        "learner": "pairwise_ranking",
        "oracle_mode": "learned",
        "diagnostic_only": False,
        "training_contains_hidden_optimum": False,
        "initial_matches_hidden_optimum": False,
        "found_hidden_exact_optimum": True,
        "success_within_1_percent": True,
        "success_within_3_percent": True,
        "success_within_5_percent": True,
        "random_exact_success_probability": 0.25,
        "random_success_within_1_percent_probability": 0.25,
        "random_success_within_3_percent_probability": 0.5,
        "random_success_within_5_percent_probability": 0.75,
        "algorithmic_ed_lp_calls": 10,
        "search_verification_calls": 6,
        "circuit_executions": 2,
        "total_shots": 4000,
        "verified_candidates": 6,
        "status": "ok",
    }
    perfect_row = {
        **learned_row,
        "measurement_policy": "classical_uniform_marked_diagnostic",
        "shots": 0,
        "shots_per_circuit": 0,
        "max_candidates_per_shotbatch": 0,
        "refit_policy": "none",
        "learner": "hidden_reference",
        "oracle_mode": "hidden_perfect_uniform_marked",
        "diagnostic_only": True,
        "uses_hidden_reference_for_sampling": True,
        "is_gate_level": False,
        "random_exact_success_probability": None,
        "random_success_within_1_percent_probability": None,
        "random_success_within_3_percent_probability": None,
        "random_success_within_5_percent_probability": None,
        "algorithmic_ed_lp_calls": 6,
        "search_verification_calls": 2,
        "circuit_executions": 0,
        "total_shots": 0,
        "verified_candidates": 2,
    }
    payload = {
        "runs": [learned_row, perfect_row],
        "grouped_summary": [
            {
                "selected_generators": [0, 5],
                "num_search_qubits": 4,
                "train_sample_count": 4,
                "measurement_policy": "shot_batch",
                "max_candidates_per_shotbatch": 3,
                "refit_policy": "accepted",
                "learner": "pairwise_ranking",
                "oracle_mode": "learned",
                "exact_success_rate": 1.0,
                "within_1_percent_success_rate": 1.0,
                "within_3_percent_success_rate": 1.0,
                "within_5_percent_success_rate": 1.0,
                "avg_random_exact_success_probability": 0.25,
                "avg_random_within_1_percent_probability": 0.25,
                "avg_random_within_3_percent_probability": 0.5,
                "avg_random_within_5_percent_probability": 0.75,
                "avg_algorithmic_ed_lp_calls": 10.0,
                "avg_search_verification_calls": 6.0,
                "avg_circuit_executions": 2.0,
                "avg_total_shots": 4000.0,
                "avg_verified_candidates": 6.0,
                "avg_max_transpiled_depth": 100.0,
            }
        ],
    }

    summary = _markdown_summary(payload)

    for heading in [
        "## Random Baseline Comparison",
        "## Equal-Shot Measurement Comparison",
        "## Candidate Budget Comparison",
        "## Refit Comparison",
        "## Learner Comparison",
        "## Perfect Oracle Diagnostic",
        "## Cautious Conclusions",
    ]:
        assert heading in summary
    assert "hidden_perfect_uniform_marked uses hidden reference for diagnostic sampling" in summary
    assert "hidden_oracle_trivial_upper_bound is a trivial upper bound" in summary
