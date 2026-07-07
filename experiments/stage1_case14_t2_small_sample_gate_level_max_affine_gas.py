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

from experiments.stage1_case14_t2_ancilla_vqc import (  # noqa: E402
    leading_time_window_instance,
)
from experiments.stage1_case14_t2_gate_level_grover_oracle import (  # noqa: E402
    embedded_selected_commitments,
)
from experiments.stage1_case14_t2_max_affine_adaptive_grover_search import (  # noqa: E402
    sanitize_for_strict_json,
    write_strict_json,
)
from qubit_value_function.commitment import commitment_to_bitstring  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.gate_level_oracle import (  # noqa: E402
    GateLevelAffinePieceSpec,
    GateLevelMaxAffineOracleSpec,
    bitstring_from_index,
    build_max_affine_grover_circuit,
    build_max_affine_phase_oracle_circuit,
    circuit_resource_summary,
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
    bit_weights = _rank_correlation_weights(
        train_bitstrings,
        train_values,
        best_bits,
        max_weight=max_weight,
    )

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
                name=f"L{piece_index}_small_sample_mismatch",
                bias=0,
            )
        )

    probe = max_affine_spec_from_pieces(
        tuple(pieces),
        tau_int=max(0, _predicted_value(tuple(pieces), bitstring_to_index(best_bitstring))),
        bit_labels=tuple(f"x{idx}" for idx in range(num_bits)),
    )
    train_predictions = np.array(
        [_predicted_value(tuple(pieces), bitstring_to_index(bitstring)) for bitstring in train_bitstrings],
        dtype=float,
    )
    diagnostics = {
        "train_sample_count": int(train_values.size),
        "train_best_bitstring": best_bitstring,
        "train_best_true_cost": _finite_or_none(float(train_values[best_row])),
        "train_pairwise_order_accuracy": _pairwise_order_accuracy(train_values, train_predictions),
        "train_predicted_values": [int(value) for value in train_predictions],
        "num_pieces": int(num_pieces),
        "max_weight": int(max_weight),
        "seed": int(seed),
        "probe_marked_count_at_best_tau": int(probe.marked_mask().sum()),
    }
    return SmallSampleLearnedOracle(
        pieces=tuple(pieces),
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
    tie_tolerance: float = 1e-9,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    dimension = 2 ** len(learned.bit_labels)
    incumbent = int(initial_index)
    incumbent_value = float(evaluator(incumbent))
    z = 1.0
    rounds: list[dict[str, object]] = []
    circuit_executions = 0
    total_shots = 0
    stop_reason = "max_rounds"
    max_qubits = 0
    max_transpiled_depth = 0
    max_depth = 0
    aggregate_ops: dict[str, int] = {}

    for round_index in range(int(max_rounds)):
        before = incumbent
        before_value = incumbent_value
        calibration = calibrate_integer_threshold_from_samples_or_incumbent(
            pieces=learned.pieces,
            train_bitstrings=learned.train_bitstrings,
            train_values=learned.train_values,
            incumbent_index=incumbent,
            incumbent_true_value=incumbent_value,
            tie_tolerance=tie_tolerance,
        )
        tau_int = int(calibration["tau_int"])
        round_trials = []
        improved_this_round = False
        for trial_index in range(int(max_trials_per_threshold)):
            k = int(rng.integers(0, max(1, int(np.ceil(z)))))
            circuit = build_gate_level_grover_circuit_for_threshold(learned, tau_int=tau_int, iterations=k)
            spec = max_affine_spec_from_learned(learned, tau_int=tau_int)
            phase = build_max_affine_phase_oracle_circuit(spec)
            execution = execute_gate_level_circuit(circuit, backend=backend, shots=shots, seed=int(rng.integers(0, 2**31 - 1)))
            circuit_executions += 1
            total_shots += int(shots)
            counts_top = _counts_top(execution["counts"])
            selected_index, selected_bitstring, selected_cost = _select_best_measured_candidate(
                counts_top,
                evaluator,
                max_candidates=max_candidates_per_shotbatch,
            )
            improved = bool(selected_cost < incumbent_value - tie_tolerance)
            if improved:
                incumbent = int(selected_index)
                incumbent_value = float(selected_cost)
                z = 1.0
                improved_this_round = True
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
                "incumbent_before": _incumbent_row(before, before_value, len(learned.bit_labels)),
                "threshold_before": _finite_or_none(before_value),
                "tau_int": tau_int,
                "calibration": calibration,
                "trials": round_trials,
                "threshold_after": _finite_or_none(incumbent_value),
                "incumbent_after": _incumbent_row(incumbent, incumbent_value, len(learned.bit_labels)),
            }
        )
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
) -> dict[str, object]:
    source = load_uc_instance(instance_path)
    instance = leading_time_window_instance(source, 2)
    base_commitment = np.ones((len(instance.generators), instance.time_horizon), dtype=int)
    embedded_commitments = embedded_selected_commitments(base_commitment, selected_generator_indices)
    evaluator = EmbeddedEDEvaluator(instance, embedded_commitments)
    rng = np.random.default_rng(seed)
    dimension = int(embedded_commitments.shape[0])
    train_indices = _choose_train_indices(dimension, train_sample_count, rng)
    train_values = np.array([evaluator.evaluate(index) for index in train_indices], dtype=float)
    train_bitstrings = [bitstring_from_index(index, _num_search_bits(selected_generator_indices, 2)) for index in train_indices]

    learned = learn_small_sample_integer_max_affine_pieces(
        train_bitstrings=train_bitstrings,
        train_values=train_values,
        num_bits=_num_search_bits(selected_generator_indices, 2),
        num_pieces=num_pieces,
        max_weight=max_weight,
        seed=seed,
    )
    if initial_index == "best_of_train":
        incumbent_index = int(train_indices[int(np.nanargmin(train_values))])
    elif initial_index == "random":
        incumbent_index = int(rng.integers(0, dimension))
    elif str(initial_index).isdigit():
        incumbent_index = int(initial_index)
    else:
        raise ValueError("initial-index must be random, best_of_train, or an integer")

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
    )
    search_calls = evaluator.call_count - search_start_calls
    hidden_values, hidden_calls = _hidden_reference(instance, embedded_commitments)
    hidden_best = int(np.nanargmin(hidden_values))
    final_index = int(adaptive["final_incumbent"]["index"])

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
        "backend": backend,
        "shots": int(shots),
        "selected_generators": [int(index) for index in selected_generator_indices],
        "num_search_qubits": _num_search_bits(selected_generator_indices, 2),
        "train_sample_count": int(len(train_indices)),
        "train_indices": [int(index) for index in train_indices],
        "train_bitstrings": train_bitstrings,
        "ed_calls": {
            "training": int(len(train_indices)),
            "search_verification": int(search_calls),
            "hidden_reference": int(hidden_calls),
            "total_algorithmic": int(len(train_indices) + search_calls),
        },
        "learned_max_affine_oracle": {
            "integer_pieces": _pieces_to_dict(learned.pieces),
            "training_diagnostics": learned.diagnostics,
        },
        "adaptive_search": adaptive,
        "hidden_reference_not_used_by_algorithm": {
            "exact_best_bitstring": bitstring_from_index(hidden_best, _num_search_bits(selected_generator_indices, 2)),
            "exact_best_true_cost": _finite_or_none(float(hidden_values[hidden_best])),
            "whether_final_incumbent_matches_reference": bool(final_index == hidden_best),
        },
        "saved_qasm_files": qasm_paths,
    }
    write_strict_json(results_path, summary)
    return summary


def _choose_train_indices(dimension: int, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    sample_count = min(max(int(sample_count), 1), int(dimension))
    return np.asarray(rng.choice(np.arange(dimension), size=sample_count, replace=False), dtype=int)


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


def _select_best_measured_candidate(
    counts_top: list[dict[str, object]],
    evaluator: Callable[[int], float],
    *,
    max_candidates: int,
) -> tuple[int, str, float]:
    best: tuple[int, str, float] | None = None
    for row in counts_top[: max(1, int(max_candidates))]:
        bitstring = str(row["bitstring"])
        index = measured_bitstring_to_index(bitstring)
        cost = float(evaluator(index))
        if best is None or cost < best[2]:
            best = (index, bitstring, cost)
    if best is None:
        raise ValueError("no measured candidates available")
    return best


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
    value = float(value)
    if np.isfinite(value):
        return value
    return None


def parse_indices(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


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
    )
    compact = {
        "train_sample_count": summary["train_sample_count"],
        "final_incumbent": summary["adaptive_search"]["final_incumbent"],
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
