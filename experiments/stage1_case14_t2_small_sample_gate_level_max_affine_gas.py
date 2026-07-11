from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit.quantum_info import Statevector

try:
    from qiskit_aer import AerSimulator
except Exception:  # pragma: no cover - exercised only when Aer is unavailable
    AerSimulator = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.experiment_utils import (  # noqa: E402
    embedded_selected_commitments,
    finite_or_none,
    leading_time_window_instance,
    parse_indices,
    sanitize_for_strict_json,
    write_strict_json,
)
from qubit_value_function.commitment import commitment_to_bitstring  # noqa: E402
from qubit_value_function.diagnostics import gap_metrics  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.gate_level_oracle import (  # noqa: E402
    GateLevelAffinePieceSpec,
    GateLevelMaxAffineOracleSpec,
    bitstring_from_index,
    build_max_affine_grover_circuit,
    build_max_affine_phase_oracle_circuit,
    circuit_resource_summary,
    max_affine_register_allocation,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


@dataclass(frozen=True)
class SmallSampleLearnedOracle:
    pieces: tuple[GateLevelAffinePieceSpec, ...]
    bit_labels: tuple[str, ...]
    train_bitstrings: tuple[str, ...]
    train_values: tuple[float, ...]
    used_training_indices: list[int]
    diagnostics: dict[str, object]


class CandidateEvaluationCache:
    def __init__(self, values_by_index: dict[int, float]) -> None:
        self.values_by_index = {int(key): float(value) for key, value in values_by_index.items()}
        self.call_count = 0

    def evaluate(self, index: int) -> float:
        self.call_count += 1
        return float(self.values_by_index.get(int(index), float("inf")))


class EmbeddedEDEvaluator:
    def __init__(self, instance, embedded_commitments: np.ndarray) -> None:
        self.evaluator = FixedCommitmentEvaluator(instance)
        self.embedded_commitments = np.asarray(embedded_commitments, dtype=int)
        self.cache: dict[int, float] = {}
        self.call_count = 0

    def evaluate(self, index: int) -> float:
        index = int(index)
        if index not in self.cache:
            result = self.evaluator.evaluate(self.embedded_commitments[index])
            self.cache[index] = float(result.total_cost) if result.success else float("inf")
            self.call_count += 1
        return self.cache[index]


def learn_small_sample_integer_max_affine_pieces(
    *,
    train_bitstrings: list[str] | tuple[str, ...],
    train_values: np.ndarray,
    num_bits: int,
    num_pieces: int,
    max_weight: int,
    seed: int,
    learner: str = "mismatch",
) -> SmallSampleLearnedOracle:
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")
    if num_pieces <= 0:
        raise ValueError("num_pieces must be positive")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    train_values = np.asarray(train_values, dtype=float)
    if len(train_bitstrings) != train_values.size:
        raise ValueError("train_bitstrings and train_values must have matching length")
    if any(len(bitstring) != num_bits for bitstring in train_bitstrings):
        raise ValueError("all train bitstrings must have num_bits characters")

    order = np.argsort(train_values)
    best_row = int(order[0])
    best_bitstring = str(train_bitstrings[best_row])
    best_bits = np.array([int(bit) for bit in best_bitstring], dtype=int)
    groups = _piece_bit_groups(num_bits, num_pieces)
    if learner == "mismatch":
        bit_weights = _rank_correlation_weights(
            train_bitstrings,
            train_values,
            best_bits,
            max_weight=max_weight,
        )
    elif learner == "pairwise_ranking":
        bit_weights = _pairwise_ranking_weights(
            train_bitstrings,
            train_values,
            best_bits,
            max_weight=max_weight,
        )
    elif learner == "rank_hinge":
        bit_weights = _pairwise_ranking_weights(
            train_bitstrings,
            train_values,
            best_bits,
            max_weight=max_weight,
        )
    else:
        raise ValueError("learner must be mismatch, pairwise_ranking, or rank_hinge")

    pieces = []
    for piece_index, group in enumerate(groups):
        weights = [0] * num_bits
        inverted = []
        for bit_index in group:
            weights[bit_index] = int(bit_weights[bit_index])
            if best_bits[bit_index] == 1:
                inverted.append(bit_index)
        pieces.append(
            GateLevelAffinePieceSpec(
                weights=tuple(weights),
                inverted_bit_indices=tuple(inverted),
                name=f"L{piece_index}_small_sample_{learner}",
                bias=0,
            )
        )

    learner_fallback: str | None = None
    if learner == "rank_hinge":
        initial_pieces = tuple(pieces)
        pieces, learner_fallback = _rank_hinge_local_search(
            initial_pieces,
            train_bitstrings=train_bitstrings,
            train_values=train_values,
            max_weight=max_weight,
            seed=seed,
            name_prefix="rank_hinge",
        )

    piece_tuple = tuple(pieces)
    probe = max_affine_spec_from_pieces(
        piece_tuple,
        tau_int=max(0, _predicted_value(piece_tuple, bitstring_to_index(best_bitstring))),
        bit_labels=tuple(f"x{idx}" for idx in range(num_bits)),
    )
    train_predictions = np.array(
        [_predicted_value(piece_tuple, bitstring_to_index(bitstring)) for bitstring in train_bitstrings],
        dtype=float,
    )
    ranking_diagnostics = _training_ranking_diagnostics(
        train_bitstrings=train_bitstrings,
        train_values=train_values,
        predicted_values=train_predictions,
        pieces=piece_tuple,
        best_row=best_row,
        max_weight=max_weight,
        learner=learner,
        learner_fallback=learner_fallback,
    )
    diagnostics = {
        "train_sample_count": int(train_values.size),
        "train_best_bitstring": best_bitstring,
        "train_best_true_cost": _finite_or_none(float(train_values[best_row])),
        "train_predicted_values": [int(value) for value in train_predictions],
        "num_pieces": int(num_pieces),
        "max_weight": int(max_weight),
        "learner_name": learner,
        "seed": int(seed),
        "probe_marked_count_at_best_tau": int(probe.marked_mask().sum()),
        **ranking_diagnostics,
    }
    return SmallSampleLearnedOracle(
        pieces=piece_tuple,
        bit_labels=tuple(f"x{idx}" for idx in range(num_bits)),
        train_bitstrings=tuple(str(bitstring) for bitstring in train_bitstrings),
        train_values=tuple(float(value) for value in train_values.tolist()),
        used_training_indices=list(range(len(train_bitstrings))),
        diagnostics=diagnostics,
    )


def max_affine_spec_from_learned(
    learned: SmallSampleLearnedOracle,
    *,
    tau_int: int,
) -> GateLevelMaxAffineOracleSpec:
    return max_affine_spec_from_pieces(
        learned.pieces,
        tau_int=int(max(tau_int, 0)),
        bit_labels=learned.bit_labels,
    )


def max_affine_spec_from_pieces(
    pieces: tuple[GateLevelAffinePieceSpec, ...],
    *,
    tau_int: int,
    bit_labels: tuple[str, ...],
) -> GateLevelMaxAffineOracleSpec:
    return GateLevelMaxAffineOracleSpec(
        pieces=pieces,
        threshold=int(max(tau_int, 0)),
        bit_labels=bit_labels,
        name=f"small_sample_gate_level_tau_{int(max(tau_int, 0))}",
    )


def calibrate_integer_threshold_from_samples_or_incumbent(
    *,
    pieces: tuple[GateLevelAffinePieceSpec, ...],
    train_bitstrings: tuple[str, ...],
    train_values: tuple[float, ...],
    incumbent_index: int,
    incumbent_true_value: float,
    tie_tolerance: float,
) -> dict[str, object]:
    train_values_array = np.asarray(train_values, dtype=float)
    predictions = np.array(
        [_predicted_value(pieces, bitstring_to_index(bitstring)) for bitstring in train_bitstrings],
        dtype=int,
    )
    improving = train_values_array < float(incumbent_true_value) - float(tie_tolerance)
    if np.any(improving):
        candidates = sorted({int(value) for value in predictions.tolist()})
        rows = []
        for tau in candidates:
            marked = predictions <= tau
            fp = int(np.logical_and(marked, ~improving).sum())
            fn = int(np.logical_and(~marked, improving).sum())
            rows.append(
                {
                    "tau_int": int(tau),
                    "train_marked_count": int(marked.sum()),
                    "train_false_positive_count": fp,
                    "train_false_negative_count": fn,
                    "error_count": fp + fn,
                }
            )
        best = min(rows, key=lambda row: (row["error_count"], row["train_false_negative_count"], row["tau_int"]))
        margin = _integer_margin(predictions, improving, int(best["tau_int"]))
        strategy = "train_improving_samples"
    else:
        incumbent_predicted = _predicted_value(pieces, int(incumbent_index))
        best = {
            "tau_int": int(max(incumbent_predicted - 1, 0)),
            "train_marked_count": int((predictions <= max(incumbent_predicted - 1, 0)).sum()),
            "train_false_positive_count": int((predictions <= max(incumbent_predicted - 1, 0)).sum()),
            "train_false_negative_count": 0,
            "error_count": int((predictions <= max(incumbent_predicted - 1, 0)).sum()),
        }
        margin = None
        strategy = "incumbent_predicted_value_minus_one"
    return {
        **best,
        "strategy": strategy,
        "margin": margin,
        "incumbent_true_value": _finite_or_none(float(incumbent_true_value)),
    }


def build_gate_level_grover_circuit_for_threshold(
    learned: SmallSampleLearnedOracle,
    *,
    tau_int: int,
    iterations: int,
):
    spec = max_affine_spec_from_learned(learned, tau_int=tau_int)
    circuit = build_max_affine_grover_circuit(spec, int(iterations))
    classical = ClassicalRegister(spec.num_x_qubits, "meas")
    circuit.add_register(classical)
    for bit_index in range(spec.num_x_qubits):
        circuit.measure(bit_index, classical[bit_index])
    return circuit


def execute_gate_level_circuit(circuit, *, backend: str, shots: int, seed: int) -> dict[str, object]:
    backend_name = backend
    if backend in {"qasm", "fake", "ibm"}:
        if backend == "ibm":
            backend_name = "qasm_fallback_no_ibm_runtime"
        elif backend == "fake":
            backend_name = "qasm_fallback_no_fake_backend"
        if AerSimulator is None:
            raise RuntimeError("qasm backend requires qiskit-aer")
        simulator = AerSimulator(seed_simulator=int(seed))
        transpiled = transpile(circuit, simulator, seed_transpiler=int(seed), optimization_level=0)
        result = simulator.run(transpiled, shots=int(shots)).result()
        counts = {str(key): int(value) for key, value in result.get_counts().items()}
        return {
            "backend_name": backend_name,
            "counts": counts,
            "transpiled_depth": int(transpiled.depth()),
            "transpiled_ops": {str(key): int(value) for key, value in transpiled.count_ops().items()},
        }
    if backend == "statevector":
        state_circuit = circuit.remove_final_measurements(inplace=False)
        state = Statevector.from_instruction(state_circuit)
        probabilities = state.probabilities()
        num_bits = len(circuit.cregs[0]) if circuit.cregs else 0
        x_mask = 2**num_bits - 1
        x_probabilities = np.zeros(2**num_bits, dtype=float)
        for basis_index, probability in enumerate(probabilities):
            x_probabilities[basis_index & x_mask] += probability
        rng = np.random.default_rng(seed)
        sampled = rng.choice(np.arange(2**num_bits), size=int(shots), p=x_probabilities / x_probabilities.sum())
        counts: dict[str, int] = {}
        for index in sampled:
            bitstring = bitstring_from_index(int(index), num_bits)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return {
            "backend_name": "statevector_debug_sampled_counts",
            "counts": counts,
            "transpiled_depth": int(circuit.depth()),
            "transpiled_ops": {str(key): int(value) for key, value in circuit.count_ops().items()},
        }
    raise ValueError("backend must be statevector, qasm, fake, or ibm")


def adaptive_gate_level_search(
    *,
    learned: SmallSampleLearnedOracle,
    evaluator: Callable[[int], float],
    initial_index: int,
    backend: str,
    shots: int,
    seed: int,
    lambda_growth: float,
    max_rounds: int,
    max_trials_per_threshold: int,
    max_candidates_per_shotbatch: int,
    measurement_policy: str = "shot_batch",
    max_total_shots_per_run: int | None = None,
    max_total_circuit_executions_per_run: int | None = None,
    refit_policy: str = "none",
    learner_name: str = "mismatch",
    num_pieces: int = 2,
    max_weight: int = 7,
    observed_indices: list[int] | None = None,
    tie_tolerance: float = 1e-9,
) -> dict[str, object]:
    if measurement_policy not in {"shot_batch", "single_shot", "single_shot_repeated"}:
        raise ValueError("measurement_policy must be shot_batch, single_shot, or single_shot_repeated")
    if refit_policy not in {"none", "accepted"}:
        raise ValueError("refit_policy must be none or accepted")
    rng = np.random.default_rng(seed)
    current_learned = learned
    dimension = 2 ** len(current_learned.bit_labels)
    incumbent = int(initial_index)
    incumbent_value = float(evaluator(incumbent))
    z = 1.0
    observed_by_index: dict[int, float] = {}
    for idx, bitstring in zip(range(len(learned.train_bitstrings)), learned.train_bitstrings):
        observed_by_index[bitstring_to_index(bitstring)] = float(learned.train_values[idx])
    for index in observed_indices or []:
        observed_by_index.setdefault(int(index), float(evaluator(int(index))))
    rounds: list[dict[str, object]] = []
    circuit_executions = 0
    total_shots = 0
    verified_candidates = 0
    verified_candidate_indices: set[int] = set()
    accepted_refit_indices: set[int] = set()
    refit_count = 0
    refit_history: list[dict[str, object]] = []
    stop_reason = "max_rounds"
    max_qubits = 0
    max_transpiled_depth = 0
    max_depth = 0
    aggregate_ops: dict[str, int] = {}

    for round_index in range(int(max_rounds)):
        round_learned = current_learned
        round_refit_version_before = refit_count
        before = incumbent
        before_value = incumbent_value
        calibration = calibrate_integer_threshold_from_samples_or_incumbent(
            pieces=round_learned.pieces,
            train_bitstrings=round_learned.train_bitstrings,
            train_values=round_learned.train_values,
            incumbent_index=incumbent,
            incumbent_true_value=incumbent_value,
            tie_tolerance=tie_tolerance,
        )
        tau_int = int(calibration["tau_int"])
        round_spec = max_affine_spec_from_learned(round_learned, tau_int=tau_int)
        round_register_allocation = max_affine_register_allocation(round_spec)
        if int(round_spec.marked_mask().sum()) == 0:
            rounds.append(
                {
                    "round": int(round_index),
                    "refit_version": int(round_refit_version_before),
                    "refit_version_before": int(round_refit_version_before),
                    "refit_version_after": int(refit_count),
                    "refit_version_next_round": int(refit_count),
                    "learner_before": str(round_learned.diagnostics["learner"]),
                    "learner_fallback_before": round_learned.diagnostics["learner_fallback"],
                    "oracle_pieces_before": _pieces_to_dict(round_learned.pieces),
                    "refit_triggered": False,
                    "incumbent_before": _incumbent_row(before, before_value, len(learned.bit_labels)),
                    "threshold_before": _finite_or_none(before_value),
                    "tau_int": tau_int,
                    "register_allocation": round_register_allocation,
                    "calibration": calibration,
                    "trials": [],
                    "threshold_after": _finite_or_none(incumbent_value),
                    "incumbent_after": _incumbent_row(incumbent, incumbent_value, len(learned.bit_labels)),
                }
            )
            stop_reason = "no_marked_state_at_threshold"
            break
        round_trials = []
        improved_this_round = False
        for trial_index in range(int(max_trials_per_threshold)):
            if max_total_circuit_executions_per_run is not None and circuit_executions >= int(max_total_circuit_executions_per_run):
                stop_reason = "max_total_circuit_executions_per_run"
                break
            k = int(rng.integers(0, max(1, int(np.ceil(z)))))
            circuit = build_gate_level_grover_circuit_for_threshold(round_learned, tau_int=tau_int, iterations=k)
            phase = build_max_affine_phase_oracle_circuit(round_spec)
            shots_per_circuit = 1 if measurement_policy in {"single_shot", "single_shot_repeated"} else int(shots)
            if max_total_shots_per_run is not None and total_shots + shots_per_circuit > int(max_total_shots_per_run):
                stop_reason = "max_total_shots_per_run"
                break
            execution = execute_gate_level_circuit(circuit, backend=backend, shots=shots_per_circuit, seed=int(rng.integers(0, 2**31 - 1)))
            circuit_executions += 1
            total_shots += int(shots_per_circuit)
            counts_top = _counts_top(execution["counts"])
            selected_index, selected_bitstring, selected_cost, evaluated_candidates = select_best_measured_candidate(
                counts_top,
                evaluator,
                max_candidates=1 if measurement_policy in {"single_shot", "single_shot_repeated"} else max_candidates_per_shotbatch,
            )
            verified_candidates += len(evaluated_candidates)
            verified_candidate_indices.update(int(candidate["index"]) for candidate in evaluated_candidates)
            improved = bool(selected_cost < incumbent_value - tie_tolerance)
            if improved:
                incumbent = int(selected_index)
                incumbent_value = float(selected_cost)
                z = 1.0
                improved_this_round = True
                if refit_policy == "accepted":
                    observed_by_index[int(selected_index)] = float(selected_cost)
                    accepted_refit_indices.add(int(selected_index))
                    current_learned = _refit_from_observed(
                        observed_by_index,
                        num_bits=len(round_learned.bit_labels),
                        num_pieces=num_pieces,
                        max_weight=max_weight,
                        seed=seed + refit_count + 1,
                        learner=learner_name,
                    )
                    refit_count += 1
                    refit_history.append(
                        {
                            "refit_count": int(refit_count),
                            "observed_sample_count": int(len(observed_by_index)),
                            "observed_indices": sorted(int(index) for index in observed_by_index),
                            "learner": learner_name,
                            "train_pairwise_order_accuracy": current_learned.diagnostics[
                                "train_pairwise_order_accuracy"
                            ],
                            "current_integer_pieces": _pieces_to_dict(current_learned.pieces),
                            "learner_diagnostics": current_learned.diagnostics,
                        }
                    )
            else:
                z = min(float(lambda_growth) * z, float(np.sqrt(dimension)))

            resources = {
                "phase_oracle": circuit_resource_summary(phase, decompose_reps=2),
                "grover_circuit": circuit_resource_summary(circuit.remove_final_measurements(inplace=False), decompose_reps=1),
                "transpiled_depth": int(execution["transpiled_depth"]),
                "transpiled_ops": execution["transpiled_ops"],
                "backend_name": execution["backend_name"],
            }
            max_qubits = max(max_qubits, int(resources["grover_circuit"]["num_qubits"]))
            max_depth = max(max_depth, int(resources["grover_circuit"]["depth"]))
            max_transpiled_depth = max(max_transpiled_depth, int(resources["transpiled_depth"]))
            for name, count in resources["transpiled_ops"].items():
                aggregate_ops[name] = aggregate_ops.get(name, 0) + int(count)

            round_trials.append(
                {
                    "trial": int(trial_index),
                    "z": float(z),
                    "k": int(k),
                    "counts_top": counts_top,
                    "measured_bitstring": selected_bitstring,
                    "raw_counts_top": counts_top,
                    "selected_index": int(selected_index),
                    "true_cost": _finite_or_none(selected_cost),
                    "evaluated_candidates": evaluated_candidates,
                    "incumbent_true_cost_before": _finite_or_none(before_value),
                    "incumbent_true_cost_after": _finite_or_none(incumbent_value),
                    "accepted_update": bool(improved),
                    "resources": resources,
                }
            )
            if improved:
                break
        rounds.append(
            {
                "round": int(round_index),
                "refit_version": int(round_refit_version_before),
                "refit_version_before": int(round_refit_version_before),
                "refit_version_after": int(refit_count),
                "refit_version_next_round": int(refit_count),
                "learner_before": str(round_learned.diagnostics["learner"]),
                "learner_fallback_before": round_learned.diagnostics["learner_fallback"],
                "oracle_pieces_before": _pieces_to_dict(round_learned.pieces),
                "refit_triggered": bool(refit_count > round_refit_version_before),
                "incumbent_before": _incumbent_row(before, before_value, len(learned.bit_labels)),
                "threshold_before": _finite_or_none(before_value),
                "tau_int": tau_int,
                "register_allocation": round_register_allocation,
                "calibration": calibration,
                "trials": round_trials,
                "threshold_after": _finite_or_none(incumbent_value),
                "incumbent_after": _incumbent_row(incumbent, incumbent_value, len(learned.bit_labels)),
            }
        )
        if stop_reason in {"max_total_shots_per_run", "max_total_circuit_executions_per_run"}:
            break
        if not improved_this_round and len(round_trials) >= int(max_trials_per_threshold):
            stop_reason = "trial_budget_exhausted_for_threshold"
            break
    return {
        "lambda_growth": float(lambda_growth),
        "rounds": rounds,
        "final_incumbent": _incumbent_row(incumbent, incumbent_value, len(learned.bit_labels)),
        "stop_reason": stop_reason,
        "total_quantum_circuit_executions": int(circuit_executions),
        "total_shots": int(total_shots),
        "verified_candidates": int(verified_candidates),
        "verified_candidate_indices": sorted(verified_candidate_indices),
        "accepted_refit_indices": sorted(accepted_refit_indices),
        "refit_policy": refit_policy,
        "refit_count": int(refit_count),
        "observed_sample_count": int(len(observed_by_index)),
        "observed_indices": sorted(int(index) for index in observed_by_index),
        "refit_history": refit_history,
        "max_qubits": int(max_qubits),
        "max_depth": int(max_depth),
        "max_transpiled_depth": int(max_transpiled_depth),
        "aggregate_transpiled_ops": aggregate_ops,
    }


def run(
    *,
    instance_path: Path,
    results_path: Path,
    backend: str,
    shots: int,
    seed: int,
    lambda_growth: float,
    max_rounds: int,
    max_trials_per_threshold: int,
    selected_generator_indices: tuple[int, ...],
    train_sample_count: int,
    initial_index: str,
    num_pieces: int,
    max_weight: int,
    save_qasm: bool,
    draw_circuit: bool,
    max_candidates_per_shotbatch: int,
    exclude_hidden_optimum_from_training: bool = False,
    exclude_hidden_optimum_from_initial: bool = False,
    measurement_policy: str = "shot_batch",
    max_total_shots_per_run: int | None = None,
    max_total_circuit_executions_per_run: int | None = None,
    learner: str = "mismatch",
    refit_policy: str = "none",
) -> dict[str, object]:
    source = load_uc_instance(instance_path)
    instance = leading_time_window_instance(source, 2)
    base_commitment = np.ones((len(instance.generators), instance.time_horizon), dtype=int)
    embedded_commitments = embedded_selected_commitments(base_commitment, selected_generator_indices)
    hidden_values, hidden_calls = _hidden_reference(instance, embedded_commitments)
    hidden_best = int(np.nanargmin(hidden_values))
    hidden_best_bitstring = bitstring_from_index(hidden_best, _num_search_bits(selected_generator_indices, 2))
    evaluator = EmbeddedEDEvaluator(instance, embedded_commitments)
    rng = np.random.default_rng(seed)
    dimension = int(embedded_commitments.shape[0])
    train_indices = _choose_train_indices(
        dimension,
        train_sample_count,
        rng,
        excluded_index=hidden_best if exclude_hidden_optimum_from_training else None,
    )
    train_values = np.array([evaluator.evaluate(index) for index in train_indices], dtype=float)
    train_bitstrings = [bitstring_from_index(index, _num_search_bits(selected_generator_indices, 2)) for index in train_indices]
    train_best_row = int(np.nanargmin(train_values))
    train_best_bitstring = train_bitstrings[train_best_row]
    train_best_true_cost = float(train_values[train_best_row])
    training_contains_hidden_optimum = bool(int(hidden_best) in {int(index) for index in train_indices})

    learned = learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=train_bitstrings,
        train_values=train_values,
        num_bits=_num_search_bits(selected_generator_indices, 2),
        num_pieces=num_pieces,
        max_weight=max_weight,
        seed=seed,
        learner=learner,
    )
    incumbent_index = _choose_initial_index(
        initial_index,
        dimension=dimension,
        rng=rng,
        train_indices=train_indices,
        train_values=train_values,
        hidden_best_index=hidden_best,
        exclude_hidden_optimum_from_initial=exclude_hidden_optimum_from_initial,
    )

    search_start_calls = evaluator.call_count
    adaptive = adaptive_gate_level_search(
        learned=learned,
        evaluator=evaluator.evaluate,
        initial_index=incumbent_index,
        backend=backend,
        shots=shots,
        seed=seed,
        lambda_growth=lambda_growth,
        max_rounds=max_rounds,
        max_trials_per_threshold=max_trials_per_threshold,
        max_candidates_per_shotbatch=max_candidates_per_shotbatch,
        measurement_policy=measurement_policy,
        max_total_shots_per_run=max_total_shots_per_run,
        max_total_circuit_executions_per_run=max_total_circuit_executions_per_run,
        refit_policy=refit_policy,
        learner_name=learner,
        num_pieces=num_pieces,
        max_weight=max_weight,
        observed_indices=[int(index) for index in train_indices],
    )
    search_calls = evaluator.call_count - search_start_calls
    final_index = int(adaptive["final_incumbent"]["index"])
    final_bitstring = bitstring_from_index(final_index, _num_search_bits(selected_generator_indices, 2))
    final_true_cost = adaptive["final_incumbent"]["true_cost"]
    hidden_best_true_cost = float(hidden_values[hidden_best])
    gap = gap_metrics(final_true_cost, hidden_best_true_cost)
    initial_true_cost = evaluator.cache.get(int(incumbent_index))
    if initial_true_cost is None:
        initial_true_cost = evaluator.evaluate(int(incumbent_index))

    qasm_paths = []
    if save_qasm:
        qasm_paths = _save_reference_qasm(learned, results_path)
    if draw_circuit:
        _ = build_gate_level_grover_circuit_for_threshold(learned, tau_int=0, iterations=1).draw(output="text")

    summary = {
        "method": "small-sample gate-level max-affine Grover adaptive search for UC",
        "not_full_enumeration_training": True,
        "uses_gate_level_circuits": True,
        "uses_statevector_amplitude_update": False,
        "notes": [
            "This is a small-sample gate-level max-affine GAS experiment.",
            "The hidden full subspace enumeration is used only for evaluation.",
            "Algorithmic ED/LP calls include only training samples and measured candidates.",
            "The experiment evaluates whether shot-based gate-level GAS can recover the hidden subspace optimum under limited ED/LP supervision.",
        ],
        "backend": backend,
        "shots": int(shots),
        "measurement_policy": measurement_policy,
        "shots_per_circuit": 1 if measurement_policy in {"single_shot", "single_shot_repeated"} else int(shots),
        "max_total_shots_per_run": None if max_total_shots_per_run is None else int(max_total_shots_per_run),
        "max_total_circuit_executions_per_run": None
        if max_total_circuit_executions_per_run is None
        else int(max_total_circuit_executions_per_run),
        "learner": learner,
        "refit_policy": refit_policy,
        "seed": int(seed),
        "selected_generators": [int(index) for index in selected_generator_indices],
        "num_search_qubits": _num_search_bits(selected_generator_indices, 2),
        "train_sample_count": int(len(train_indices)),
        "train_indices": [int(index) for index in train_indices],
        "train_bitstrings": train_bitstrings,
        "train_best_bitstring": train_best_bitstring,
        "train_best_true_cost": _finite_or_none(train_best_true_cost),
        "exclude_hidden_optimum_from_training": bool(exclude_hidden_optimum_from_training),
        "exclude_hidden_optimum_from_initial": bool(exclude_hidden_optimum_from_initial),
        "training_contains_hidden_optimum": training_contains_hidden_optimum,
        "hidden_best_index": int(hidden_best),
        "hidden_best_bitstring": hidden_best_bitstring,
        "hidden_best_true_cost": _finite_or_none(hidden_best_true_cost),
        "initial_index": int(incumbent_index),
        "initial_bitstring": bitstring_from_index(incumbent_index, _num_search_bits(selected_generator_indices, 2)),
        "initial_true_cost": _finite_or_none(float(initial_true_cost)),
        "initial_matches_hidden_optimum": bool(int(incumbent_index) == int(hidden_best)),
        "final_bitstring": final_bitstring,
        "final_true_cost": _finite_or_none(float(final_true_cost)) if final_true_cost is not None else None,
        "found_hidden_exact_optimum": bool(final_index == hidden_best),
        **gap,
        "algorithmic_ed_lp_calls": int(len(train_indices) + search_calls),
        "hidden_reference_ed_lp_calls": int(hidden_calls),
        "circuit_executions": int(adaptive["total_quantum_circuit_executions"]),
        "total_shots": int(adaptive["total_shots"]),
        "verified_candidates": int(adaptive["verified_candidates"]),
        "max_qubits": int(adaptive["max_qubits"]),
        "max_circuit_depth": int(adaptive["max_depth"]),
        "max_transpiled_depth": int(adaptive["max_transpiled_depth"]),
        "ed_calls": {
            "training": int(len(train_indices)),
            "search_verification": int(search_calls),
            "hidden_reference": int(hidden_calls),
            "total_algorithmic": int(len(train_indices) + search_calls),
        },
        "learned_max_affine_oracle": {
            "integer_pieces": _pieces_to_dict(learned.pieces),
            "training_diagnostics": learned.diagnostics,
            "reference_register_allocation_at_tau0": max_affine_register_allocation(
                max_affine_spec_from_learned(learned, tau_int=0)
            ),
        },
        "adaptive_search": adaptive,
        "hidden_reference_not_used_by_algorithm": {
            "exact_best_bitstring": hidden_best_bitstring,
            "exact_best_true_cost": _finite_or_none(float(hidden_values[hidden_best])),
            "whether_final_incumbent_matches_reference": bool(final_index == hidden_best),
        },
        "saved_qasm_files": qasm_paths,
    }
    write_strict_json(results_path, summary)
    return summary


def _choose_train_indices(
    dimension: int,
    sample_count: int,
    rng: np.random.Generator,
    *,
    excluded_index: int | None = None,
) -> np.ndarray:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    sample_count = max(int(sample_count), 1)
    candidates = np.arange(dimension, dtype=int)
    if excluded_index is not None:
        excluded_index = int(excluded_index)
        candidates = candidates[candidates != excluded_index]
    if sample_count > candidates.size:
        raise ValueError(
            f"train_sample_count={sample_count} exceeds available training candidates "
            f"({int(candidates.size)})"
        )
    return np.asarray(rng.choice(candidates, size=sample_count, replace=False), dtype=int)


def _choose_initial_index(
    initial_index: str,
    *,
    dimension: int,
    rng: np.random.Generator,
    train_indices: np.ndarray,
    train_values: np.ndarray,
    hidden_best_index: int,
    exclude_hidden_optimum_from_initial: bool,
) -> int:
    if initial_index == "best_of_train":
        return int(train_indices[int(np.nanargmin(train_values))])
    if initial_index == "random":
        candidates = np.arange(int(dimension), dtype=int)
        if exclude_hidden_optimum_from_initial:
            candidates = candidates[candidates != int(hidden_best_index)]
        if candidates.size <= 0:
            raise ValueError("no available random initial candidates after excluding hidden optimum")
        return int(rng.choice(candidates))
    if str(initial_index).isdigit():
        return int(initial_index)
    raise ValueError("initial-index must be random, best_of_train, or an integer")


def _rank_correlation_weights(
    train_bitstrings: list[str] | tuple[str, ...],
    train_values: np.ndarray,
    best_bits: np.ndarray,
    *,
    max_weight: int,
) -> list[int]:
    weights = []
    for bit_index in range(best_bits.size):
        mismatch = np.array(
            [int(bitstring[bit_index]) != int(best_bits[bit_index]) for bitstring in train_bitstrings],
            dtype=float,
        )
        if np.max(mismatch) == 0.0:
            weights.append(1)
            continue
        low = train_values[mismatch == 0.0]
        high = train_values[mismatch == 1.0]
        gap = float(np.nanmean(high) - np.nanmean(low)) if high.size and low.size else 0.0
        scaled = 1 if gap <= 0.0 else min(max_weight, max(1, int(np.ceil(gap / max(np.nanstd(train_values), 1.0)))))
        weights.append(int(scaled))
    return weights


def _pairwise_ranking_weights(
    train_bitstrings: list[str] | tuple[str, ...],
    train_values: np.ndarray,
    best_bits: np.ndarray,
    *,
    max_weight: int,
) -> list[int]:
    weights = []
    order = np.argsort(np.asarray(train_values, dtype=float))
    ranks = np.empty(len(order), dtype=float)
    for rank, row in enumerate(order):
        ranks[int(row)] = float(rank)
    for bit_index in range(best_bits.size):
        mismatch = np.array(
            [int(bitstring[bit_index]) != int(best_bits[bit_index]) for bitstring in train_bitstrings],
            dtype=float,
        )
        if np.max(mismatch) == 0.0:
            weights.append(1)
            continue
        low_rank = ranks[mismatch == 0.0]
        high_rank = ranks[mismatch == 1.0]
        gap = float(np.mean(high_rank) - np.mean(low_rank)) if high_rank.size and low_rank.size else 0.0
        weights.append(int(min(max_weight, max(1, round(abs(gap) + 1)))))
    return weights


def pairwise_hinge_diagnostics(
    *,
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    margin: int = 1,
) -> dict[str, object]:
    true_values = np.asarray(true_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)
    if true_values.size != predicted_values.size:
        raise ValueError("true_values and predicted_values must have matching length")

    total = 0
    violations = 0
    loss = 0.0
    for i in range(true_values.size):
        for j in range(i + 1, true_values.size):
            if true_values[i] == true_values[j]:
                continue
            if true_values[i] < true_values[j]:
                low, high = i, j
            else:
                low, high = j, i
            hinge = max(0.0, float(margin) + float(predicted_values[low]) - float(predicted_values[high]))
            total += 1
            loss += hinge
            violations += int(hinge > 0.0)
    return {
        "num_pairs": int(total),
        "num_pairwise_violations": int(violations),
        "pairwise_hinge_loss": float(loss),
        "train_pairwise_order_accuracy": 1.0 if total == 0 else float((total - violations) / total),
    }


def _training_ranking_diagnostics(
    *,
    train_bitstrings: list[str] | tuple[str, ...],
    train_values: np.ndarray,
    predicted_values: np.ndarray,
    pieces: tuple[GateLevelAffinePieceSpec, ...],
    best_row: int,
    max_weight: int,
    learner: str,
    learner_fallback: str | None,
) -> dict[str, object]:
    hinge = pairwise_hinge_diagnostics(true_values=train_values, predicted_values=predicted_values, margin=1)
    ordered_by_prediction = sorted(
        range(predicted_values.size),
        key=lambda idx: (float(predicted_values[idx]), float(train_values[idx]), str(train_bitstrings[idx])),
    )
    best_rank = int(ordered_by_prediction.index(int(best_row)) + 1)
    weights = [int(weight) for piece in pieces for weight in piece.weights]
    return {
        **hinge,
        "learner": str(learner),
        "learner_fallback": learner_fallback,
        "train_best_rank": best_rank,
        "train_best_bitstring": str(train_bitstrings[best_row]),
        "train_best_true_cost": _finite_or_none(float(train_values[best_row])),
        "train_predicted_values": [int(value) for value in predicted_values],
        "train_true_values": [_finite_or_none(float(value)) for value in train_values],
        "surrogate_piece_count": int(len(pieces)),
        "surrogate_max_weight": int(max(weights) if weights else 0),
        "surrogate_total_weight": int(sum(abs(weight) for weight in weights)),
        "surrogate_configured_max_weight": int(max_weight),
    }


def _rank_hinge_local_search(
    initial_pieces: tuple[GateLevelAffinePieceSpec, ...],
    *,
    train_bitstrings: list[str] | tuple[str, ...],
    train_values: np.ndarray,
    max_weight: int,
    seed: int,
    name_prefix: str,
) -> tuple[tuple[GateLevelAffinePieceSpec, ...], str | None]:
    if len(train_bitstrings) < 2:
        return initial_pieces, "too_few_training_samples"

    rng = np.random.default_rng(seed)
    pieces = tuple(initial_pieces)
    best_loss = _rank_hinge_objective(pieces, train_bitstrings=train_bitstrings, train_values=train_values)
    initial_loss = best_loss
    changed = False

    for _ in range(40):
        improved = False
        piece_order = list(range(len(pieces)))
        rng.shuffle(piece_order)
        for piece_index in piece_order:
            bit_order = list(range(pieces[piece_index].num_x_qubits))
            rng.shuffle(bit_order)
            for bit_index in bit_order:
                for delta in (-1, 1):
                    updated = _replace_piece_weight(
                        pieces,
                        piece_index=piece_index,
                        bit_index=bit_index,
                        value=min(max(int(pieces[piece_index].weights[bit_index]) + delta, 0), int(max_weight)),
                        name_prefix=name_prefix,
                    )
                    loss = _rank_hinge_objective(
                        updated,
                        train_bitstrings=train_bitstrings,
                        train_values=train_values,
                    )
                    if loss + 1e-12 < best_loss:
                        pieces = updated
                        best_loss = loss
                        improved = True
                        changed = True
            for delta in (-1, 1):
                updated = _replace_piece_bias(
                    pieces,
                    piece_index=piece_index,
                    value=min(max(int(pieces[piece_index].bias) + delta, -int(max_weight)), int(max_weight)),
                    name_prefix=name_prefix,
                )
                loss = _rank_hinge_objective(
                    updated,
                    train_bitstrings=train_bitstrings,
                    train_values=train_values,
                )
                if loss + 1e-12 < best_loss:
                    pieces = updated
                    best_loss = loss
                    improved = True
                    changed = True
        if not improved:
            break

    fallback = None if changed and best_loss + 1e-12 < initial_loss else "pairwise_ranking_no_improvement"
    if fallback is not None:
        return initial_pieces, fallback
    return pieces, None


def _rank_hinge_objective(
    pieces: tuple[GateLevelAffinePieceSpec, ...],
    *,
    train_bitstrings: list[str] | tuple[str, ...],
    train_values: np.ndarray,
) -> float:
    predicted = np.array(
        [_predicted_value(pieces, bitstring_to_index(bitstring)) for bitstring in train_bitstrings],
        dtype=float,
    )
    diagnostics = pairwise_hinge_diagnostics(true_values=train_values, predicted_values=predicted, margin=1)
    best_row = int(np.nanargmin(train_values))
    best_pred = float(predicted[best_row])
    misrank_best = float(sum(value < best_pred for idx, value in enumerate(predicted) if idx != best_row))
    total_weight = float(sum(abs(int(weight)) for piece in pieces for weight in piece.weights))
    total_bias = float(sum(abs(int(piece.bias)) for piece in pieces))
    return float(diagnostics["pairwise_hinge_loss"]) + 4.0 * misrank_best + 0.01 * total_weight + 0.01 * total_bias


def _replace_piece_weight(
    pieces: tuple[GateLevelAffinePieceSpec, ...],
    *,
    piece_index: int,
    bit_index: int,
    value: int,
    name_prefix: str,
) -> tuple[GateLevelAffinePieceSpec, ...]:
    out = list(pieces)
    piece = out[piece_index]
    weights = list(piece.weights)
    weights[bit_index] = int(value)
    out[piece_index] = GateLevelAffinePieceSpec(
        weights=tuple(weights),
        inverted_bit_indices=piece.inverted_bit_indices,
        name=f"L{piece_index}_small_sample_{name_prefix}",
        bias=piece.bias,
    )
    return tuple(out)


def _replace_piece_bias(
    pieces: tuple[GateLevelAffinePieceSpec, ...],
    *,
    piece_index: int,
    value: int,
    name_prefix: str,
) -> tuple[GateLevelAffinePieceSpec, ...]:
    out = list(pieces)
    piece = out[piece_index]
    out[piece_index] = GateLevelAffinePieceSpec(
        weights=piece.weights,
        inverted_bit_indices=piece.inverted_bit_indices,
        name=f"L{piece_index}_small_sample_{name_prefix}",
        bias=int(value),
    )
    return tuple(out)


def _refit_from_observed(
    observed_by_index: dict[int, float],
    *,
    num_bits: int,
    num_pieces: int,
    max_weight: int,
    seed: int,
    learner: str,
) -> SmallSampleLearnedOracle:
    ordered = sorted(observed_by_index)
    return learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=[bitstring_from_index(index, num_bits) for index in ordered],
        train_values=np.asarray([observed_by_index[index] for index in ordered], dtype=float),
        num_bits=num_bits,
        num_pieces=num_pieces,
        max_weight=max_weight,
        seed=seed,
        learner=learner,
    )


def _piece_bit_groups(num_bits: int, num_pieces: int) -> list[list[int]]:
    groups = [[] for _ in range(num_pieces)]
    for bit_index in range(num_bits):
        groups[bit_index % num_pieces].append(bit_index)
    return [group for group in groups if group]


def _predicted_value(pieces: tuple[GateLevelAffinePieceSpec, ...], index: int) -> int:
    return int(max(piece.values_for_all_x()[int(index)] for piece in pieces))


def bitstring_to_index(bitstring: str) -> int:
    value = 0
    for bit_index, bit in enumerate(str(bitstring)):
        value |= int(bit) << bit_index
    return int(value)


def measured_bitstring_to_index(bitstring: str) -> int:
    return bitstring_to_index(str(bitstring).replace(" ", ""))


def _counts_top(counts: dict[str, int], limit: int = 10) -> list[dict[str, object]]:
    return [
        {"bitstring": str(bitstring), "count": int(count)}
        for bitstring, count in sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    ]


def select_best_measured_candidate(
    counts_top: list[dict[str, object]],
    evaluator: Callable[[int], float],
    *,
    max_candidates: int,
) -> tuple[int, str, float, list[dict[str, object]]]:
    best: tuple[int, str, float] | None = None
    evaluated = []
    seen = set()
    for row in counts_top:
        bitstring = str(row["bitstring"])
        if bitstring in seen:
            continue
        seen.add(bitstring)
        index = measured_bitstring_to_index(bitstring)
        cost = float(evaluator(index))
        evaluated.append(
            {
                "index": int(index),
                "bitstring": bitstring,
                "true_cost": _finite_or_none(cost),
                "count": int(row.get("count", 0)),
            }
        )
        if best is None or cost < best[2]:
            best = (index, bitstring, cost)
        if len(evaluated) >= max(1, int(max_candidates)):
            break
    if best is None:
        raise ValueError("no measured candidates available")
    return best[0], best[1], best[2], evaluated


def _select_best_measured_candidate(
    counts_top: list[dict[str, object]],
    evaluator: Callable[[int], float],
    *,
    max_candidates: int,
) -> tuple[int, str, float]:
    selected_index, selected_bitstring, selected_cost, _ = select_best_measured_candidate(
        counts_top,
        evaluator,
        max_candidates=max_candidates,
    )
    return selected_index, selected_bitstring, selected_cost


def _hidden_reference(instance, embedded_commitments: np.ndarray) -> tuple[np.ndarray, int]:
    evaluator = FixedCommitmentEvaluator(instance)
    values = np.full(embedded_commitments.shape[0], np.inf, dtype=float)
    calls = 0
    for index, commitment in enumerate(embedded_commitments):
        result = evaluator.evaluate(commitment)
        calls += 1
        if result.success:
            values[index] = float(result.total_cost)
    return values, calls


def _incumbent_row(index: int, true_cost: float, num_bits: int) -> dict[str, object]:
    return {
        "index": int(index),
        "bitstring": bitstring_from_index(int(index), int(num_bits)),
        "true_cost": _finite_or_none(float(true_cost)),
    }


def _pieces_to_dict(pieces: tuple[GateLevelAffinePieceSpec, ...]) -> list[dict[str, object]]:
    return [
        {
            "name": piece.name,
            "weights": [int(weight) for weight in piece.weights],
            "bias": int(piece.bias),
            "inverted_bit_indices": [int(index) for index in piece.inverted_bit_indices],
        }
        for piece in pieces
    ]


def _pairwise_order_accuracy(values: np.ndarray, predictions: np.ndarray) -> float:
    total = 0
    correct = 0
    for i in range(values.size):
        for j in range(i + 1, values.size):
            if values[i] == values[j]:
                continue
            total += 1
            correct += int((values[i] < values[j]) == (predictions[i] <= predictions[j]))
    return 1.0 if total == 0 else float(correct / total)


def _integer_margin(predictions: np.ndarray, labels: np.ndarray, tau: int) -> int | None:
    if not np.any(labels) or not np.any(~labels):
        return None
    return int(np.min(predictions[~labels]) - np.max(predictions[labels]))


def _num_search_bits(selected_generator_indices: tuple[int, ...], horizon: int) -> int:
    return int(len(selected_generator_indices) * horizon)


def _save_reference_qasm(learned: SmallSampleLearnedOracle, results_path: Path) -> list[str]:
    circuit = build_gate_level_grover_circuit_for_threshold(learned, tau_int=0, iterations=1)
    qasm_path = results_path.with_suffix(".qasm")
    try:
        qasm_path.write_text(circuit.qasm(), encoding="utf-8")
    except Exception:
        qasm_path.write_text(str(circuit), encoding="utf-8")
    return [str(qasm_path)]


def _finite_or_none(value: float) -> float | None:
    return finite_or_none(value)


def parse_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_small_sample_gate_level_max_affine_gas.json"),
    )
    parser.add_argument("--backend", choices=("statevector", "qasm", "fake", "ibm"), default="qasm")
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lambda-growth", type=float, default=8.0 / 7.0)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--max-trials-per-threshold", type=int, default=20)
    parser.add_argument("--selected-generators", type=parse_indices, default=(0, 5))
    parser.add_argument("--train-sample-count", type=int, default=8)
    parser.add_argument("--initial-index", type=str, default="random")
    parser.add_argument("--num-pieces", type=int, default=2)
    parser.add_argument("--max-weight", type=int, default=7)
    parser.add_argument("--save-qasm", type=parse_bool, default=True)
    parser.add_argument("--draw-circuit", type=parse_bool, default=False)
    parser.add_argument("--max-candidates-per-shotbatch", type=int, default=3)
    parser.add_argument("--exclude-hidden-optimum-from-training", action="store_true")
    parser.add_argument("--exclude-hidden-optimum-from-initial", action="store_true")
    parser.add_argument("--measurement-policy", choices=("shot_batch", "single_shot", "single_shot_repeated"), default="shot_batch")
    parser.add_argument("--max-total-shots-per-run", type=int, default=None)
    parser.add_argument("--max-total-circuit-executions-per-run", type=int, default=None)
    parser.add_argument("--learner", choices=("mismatch", "pairwise_ranking", "rank_hinge"), default="mismatch")
    parser.add_argument("--refit-policy", choices=("none", "accepted"), default="none")
    args = parser.parse_args()

    summary = run(
        instance_path=args.instance,
        results_path=args.results,
        backend=args.backend,
        shots=args.shots,
        seed=args.seed,
        lambda_growth=args.lambda_growth,
        max_rounds=args.max_rounds,
        max_trials_per_threshold=args.max_trials_per_threshold,
        selected_generator_indices=args.selected_generators,
        train_sample_count=args.train_sample_count,
        initial_index=args.initial_index,
        num_pieces=args.num_pieces,
        max_weight=args.max_weight,
        save_qasm=args.save_qasm,
        draw_circuit=args.draw_circuit,
        max_candidates_per_shotbatch=args.max_candidates_per_shotbatch,
        exclude_hidden_optimum_from_training=args.exclude_hidden_optimum_from_training,
        exclude_hidden_optimum_from_initial=args.exclude_hidden_optimum_from_initial,
        measurement_policy=args.measurement_policy,
        max_total_shots_per_run=args.max_total_shots_per_run,
        max_total_circuit_executions_per_run=args.max_total_circuit_executions_per_run,
        learner=args.learner,
        refit_policy=args.refit_policy,
    )
    compact = {
        "selected_generators": summary["selected_generators"],
        "train_sample_count": summary["train_sample_count"],
        "seed": summary["seed"],
        "exclude_hidden_optimum_from_training": summary["exclude_hidden_optimum_from_training"],
        "exclude_hidden_optimum_from_initial": summary["exclude_hidden_optimum_from_initial"],
        "training_contains_hidden_optimum": summary["training_contains_hidden_optimum"],
        "initial_matches_hidden_optimum": summary["initial_matches_hidden_optimum"],
        "measurement_policy": summary["measurement_policy"],
        "learner": summary["learner"],
        "refit_policy": summary["refit_policy"],
        "final_incumbent": summary["adaptive_search"]["final_incumbent"],
        "success_within_3_percent": summary["success_within_3_percent"],
        "hidden_exact_optimum": summary["hidden_reference_not_used_by_algorithm"],
        "matched_hidden_optimum": summary["hidden_reference_not_used_by_algorithm"]["whether_final_incumbent_matches_reference"],
        "total_ed_calls": summary["ed_calls"]["total_algorithmic"],
        "total_quantum_circuit_executions": summary["adaptive_search"]["total_quantum_circuit_executions"],
        "total_shots": summary["adaptive_search"]["total_shots"],
        "max_qubits": summary["adaptive_search"]["max_qubits"],
        "max_transpiled_depth": summary["adaptive_search"]["max_transpiled_depth"],
        "stop_reason": summary["adaptive_search"]["stop_reason"],
    }
    print(json.dumps(sanitize_for_strict_json(compact), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
