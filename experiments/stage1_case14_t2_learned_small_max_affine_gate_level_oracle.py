from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_t2_ancilla_vqc import (  # noqa: E402
    commitment_row,
    evaluate_values,
    leading_time_window_instance,
)
from experiments.stage1_case14_t2_gate_level_grover_oracle import (  # noqa: E402
    embedded_selected_commitments,
)
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.gate_level_oracle import (  # noqa: E402
    GateLevelAffinePieceSpec,
    GateLevelMaxAffineOracleSpec,
    bitstring_from_index,
    build_max_affine_grover_circuit,
    build_max_affine_phase_oracle_circuit,
    circuit_resource_summary,
    simulate_max_affine_grover,
    simulate_max_affine_phase_oracle,
)
from qubit_value_function.uc_loader import UCInstance, load_uc_instance  # noqa: E402


def run(
    instance_path: Path,
    results_path: Path,
    horizon: int,
    selected_generator_indices: tuple[int, ...],
    target_counts: tuple[int, ...],
    max_weight: int,
) -> dict[str, object]:
    if horizon != 2:
        raise ValueError("this learned gate-level prototype is restricted to T=2")
    if len(selected_generator_indices) < 2:
        raise ValueError("at least two selected generators are needed for max-affine pieces")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")

    source_instance = load_uc_instance(instance_path)
    instance = leading_time_window_instance(source_instance, horizon)
    generator_names = [gen.name for gen in instance.generators]
    commitments = all_commitments(len(instance.generators), instance.time_horizon)

    value_start = time.perf_counter()
    values, logic_feasible = evaluate_values(instance, commitments)
    value_seconds = time.perf_counter() - value_start
    finite = np.isfinite(values)
    finite_sorted_indices = [int(index) for index in np.argsort(values) if finite[index]]
    full_optimum_index = finite_sorted_indices[0]
    full_optimum_commitment = commitments[full_optimum_index]

    embedded_commitments = embedded_selected_commitments(
        full_optimum_commitment,
        selected_generator_indices,
    )
    embedded_values, embedded_logic_feasible = evaluate_values(instance, embedded_commitments)
    embedded_finite = np.isfinite(embedded_values)
    embedded_best_index = int(np.nanargmin(embedded_values))

    learned = learn_gap_weighted_max_affine_pieces(
        instance=instance,
        selected_generator_indices=selected_generator_indices,
        embedded_values=embedded_values,
        embedded_best_index=embedded_best_index,
        max_weight=max_weight,
    )
    template_spec = GateLevelMaxAffineOracleSpec(
        pieces=learned.pieces,
        threshold=0,
        bit_labels=learned.bit_labels,
        name="case14_t2_learned_gap_weighted_max_affine_oracle",
    )
    predicted_values = template_spec.values_for_all_x()

    target_case_rows = []
    for target_count in target_counts:
        calibration = calibrate_integer_threshold(
            predicted_values=predicted_values,
            true_values=embedded_values,
            target_count=int(target_count),
        )
        spec = GateLevelMaxAffineOracleSpec(
            pieces=learned.pieces,
            threshold=int(calibration["selected_threshold"]),
            bit_labels=learned.bit_labels,
            name=f"case14_t2_learned_gap_weighted_top_{int(target_count)}_oracle",
        )
        target_case_rows.append(
            run_target_case(
                spec=spec,
                calibration=calibration,
                embedded_commitments=embedded_commitments,
                embedded_values=embedded_values,
            )
        )

    summary = {
        "instance": f"case14_T{horizon}",
        "source": str(instance_path),
        "method": "T=2 true-cost-learned small max-affine Qiskit gate-level oracle",
        "scope": {
            "full_case14_bits": int(commitments.reshape((commitments.shape[0], -1)).shape[1]),
            "gate_level_x_bits": int(template_spec.num_x_qubits),
            "piece_count": int(template_spec.piece_count),
            "note": (
                "This experiment learns a small integer max-affine oracle from true UC costs "
                "on an embedded T=2 subspace. It is a data-driven gate-level proof of concept, "
                "not the full 12-bit, 207-feature, 32-piece synthesis."
            ),
        },
        "fixed_load_mw": instance.fixed_load,
        "generators": generator_names,
        "selected_generators": [
            generator_names[index] for index in selected_generator_indices
        ],
        "bit_order": "selected generator-major order: g_i_t0,g_i_t1,...",
        "learning_rule": {
            "summary": (
                "Find the true best selected-subspace commitment, flip each selected bit once, "
                "use the true UC cost increase as a local sensitivity, quantize sensitivities "
                "to small nonnegative integers, and build generator-wise mismatch pieces."
            ),
            "max_integer_weight": int(max_weight),
            "encoded_object": "integer max-affine mismatch surrogate, not a true-cost lookup table",
        },
        "learned_model": learned_to_dict(learned, template_spec),
        "oracle_and_circuit_explanation": oracle_and_circuit_explanation(
            learned,
            template_spec,
        ),
        "full_t2_reference": {
            "logic_feasible_count": int(logic_feasible.sum()),
            "finite_value_count": int(finite.sum()),
            "value_evaluation_seconds": float(value_seconds),
            "optimum": commitment_row(
                commitments,
                generator_names,
                values,
                full_optimum_index,
            ),
        },
        "embedded_subspace_validation": {
            "base_commitment": commitment_to_bitstring(full_optimum_commitment),
            "base_note": "Non-selected generators are fixed to the exhaustive T=2 optimum for validation only.",
            "subspace_size": int(embedded_commitments.shape[0]),
            "logic_feasible_count": int(embedded_logic_feasible.sum()),
            "finite_value_count": int(embedded_finite.sum()),
            "best_selected_bitstring": bitstring_from_index(
                embedded_best_index,
                template_spec.num_x_qubits,
            ),
            "best_full_bitstring": commitment_to_bitstring(
                embedded_commitments[embedded_best_index]
            ),
            "best_true_cost": _finite_or_none(embedded_values[embedded_best_index]),
        },
        "oracle_decomposition": {
            "piece_registers": "Qiskit WeightedAdder computes each learned integer mismatch piece L_r(x)",
            "piece_comparators": "each piece is compared against the calibrated integer threshold tau",
            "max_threshold_logic": "max_r L_r(x) <= tau is implemented as AND_r[L_r(x) <= tau]",
            "phase_mark": "multi-controlled phase on all piece-comparison flags",
            "uncompute": "inverse comparators and adders restore every non-x register to zero",
            "diffuser": "standard Grover diffuser acts only on the x register",
        },
        "target_cases": target_case_rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class LearnedPieces:
    def __init__(
        self,
        *,
        pieces: tuple[GateLevelAffinePieceSpec, ...],
        bit_labels: tuple[str, ...],
        optimum_index: int,
        single_flip_gaps: list[float | None],
        integer_weights: tuple[int, ...],
        piece_groups: list[list[int]],
    ) -> None:
        self.pieces = pieces
        self.bit_labels = bit_labels
        self.optimum_index = int(optimum_index)
        self.single_flip_gaps = single_flip_gaps
        self.integer_weights = integer_weights
        self.piece_groups = piece_groups


def learn_gap_weighted_max_affine_pieces(
    *,
    instance: UCInstance,
    selected_generator_indices: tuple[int, ...],
    embedded_values: np.ndarray,
    embedded_best_index: int,
    max_weight: int,
) -> LearnedPieces:
    horizon = instance.time_horizon
    num_bits = len(selected_generator_indices) * horizon
    bit_labels = tuple(
        f"{instance.generators[generator_index].name}_t{time_index}"
        for generator_index in selected_generator_indices
        for time_index in range(horizon)
    )
    optimum_bits = tuple((int(embedded_best_index) >> bit_index) & 1 for bit_index in range(num_bits))
    best_value = float(embedded_values[int(embedded_best_index)])

    raw_gaps: list[float | None] = []
    positive_finite_gaps: list[float] = []
    for bit_index in range(num_bits):
        flipped_index = int(embedded_best_index) ^ (1 << bit_index)
        flipped_value = float(embedded_values[flipped_index])
        if np.isfinite(flipped_value):
            gap = max(flipped_value - best_value, 0.0)
            raw_gaps.append(float(gap))
            if gap > 0.0:
                positive_finite_gaps.append(float(gap))
        else:
            raw_gaps.append(None)

    scale_gap = max(positive_finite_gaps) if positive_finite_gaps else 1.0
    integer_weights = []
    for gap in raw_gaps:
        if gap is None:
            integer_weights.append(int(max_weight))
        elif gap <= 0.0:
            integer_weights.append(1)
        else:
            integer_weights.append(max(1, int(np.ceil(float(max_weight) * gap / scale_gap))))

    pieces = []
    piece_groups: list[list[int]] = []
    for local_generator_index, generator_index in enumerate(selected_generator_indices):
        group = [local_generator_index * horizon + time_index for time_index in range(horizon)]
        piece_groups.append(group)
        weights = [0] * num_bits
        inverted: list[int] = []
        for bit_index in group:
            weights[bit_index] = int(integer_weights[bit_index])
            if optimum_bits[bit_index] == 1:
                inverted.append(bit_index)
        pieces.append(
            GateLevelAffinePieceSpec(
                weights=tuple(weights),
                inverted_bit_indices=tuple(inverted),
                name=f"L{local_generator_index}_{instance.generators[generator_index].name}_learned_mismatch",
                bias=0,
            )
        )

    return LearnedPieces(
        pieces=tuple(pieces),
        bit_labels=bit_labels,
        optimum_index=int(embedded_best_index),
        single_flip_gaps=raw_gaps,
        integer_weights=tuple(int(weight) for weight in integer_weights),
        piece_groups=piece_groups,
    )


def calibrate_integer_threshold(
    *,
    predicted_values: np.ndarray,
    true_values: np.ndarray,
    target_count: int,
) -> dict[str, object]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    finite = np.isfinite(true_values)
    finite_order = [int(index) for index in np.argsort(true_values) if finite[index]]
    actual_target_count = min(int(target_count), len(finite_order))
    target_indices = set(finite_order[:actual_target_count])
    labels = np.array([index in target_indices for index in range(true_values.shape[0])], dtype=bool)

    candidates = sorted({int(value) for value in predicted_values.tolist()})
    rows = []
    for threshold in candidates:
        marked = predicted_values <= threshold
        false_positive = np.flatnonzero(marked & ~labels)
        false_negative = np.flatnonzero((~marked) & labels)
        rows.append(
            {
                "threshold": int(threshold),
                "marked_count": int(marked.sum()),
                "false_positive_count": int(false_positive.size),
                "false_negative_count": int(false_negative.size),
                "error_count": int(false_positive.size + false_negative.size),
                "exact_marked_set": bool(false_positive.size == 0 and false_negative.size == 0),
            }
        )
    best = min(
        rows,
        key=lambda row: (
            int(row["error_count"]),
            abs(int(row["marked_count"]) - actual_target_count),
            int(row["threshold"]),
        ),
    )
    return {
        "target_count_request": int(target_count),
        "actual_target_count": int(actual_target_count),
        "selected_threshold": int(best["threshold"]),
        "selected_row": best,
        "labels": labels.tolist(),
        "target_selected_bitstrings": [
            bitstring_from_index(index, int(np.log2(true_values.shape[0])))
            for index in finite_order[:actual_target_count]
        ],
        "candidate_rows": rows,
    }


def run_target_case(
    *,
    spec: GateLevelMaxAffineOracleSpec,
    calibration: dict[str, object],
    embedded_commitments: np.ndarray,
    embedded_values: np.ndarray,
) -> dict[str, object]:
    phase_start = time.perf_counter()
    phase_probe = simulate_max_affine_phase_oracle(spec)
    phase_seconds = time.perf_counter() - phase_start

    grover_start = time.perf_counter()
    grover_result = simulate_max_affine_grover(spec)
    grover_seconds = time.perf_counter() - grover_start

    phase_circuit = build_max_affine_phase_oracle_circuit(spec)
    grover_circuit = build_max_affine_grover_circuit(spec, grover_result.iterations)
    piece_values = spec.piece_values_for_all_x()
    max_values = spec.values_for_all_x()
    marked_indices = [int(index) for index in np.flatnonzero(spec.marked_mask())]
    labels = np.asarray(calibration["labels"], dtype=bool)
    false_positive = np.flatnonzero(spec.marked_mask() & ~labels)
    false_negative = np.flatnonzero((~spec.marked_mask()) & labels)
    argmax_index = int(np.argmax(grover_result.x_probabilities))

    return {
        "target_count_request": int(calibration["target_count_request"]),
        "actual_target_count": int(calibration["actual_target_count"]),
        "threshold_tau": int(spec.threshold),
        "exact_marked_set": bool(false_positive.size == 0 and false_negative.size == 0),
        "false_positive_count": int(false_positive.size),
        "false_negative_count": int(false_negative.size),
        "target_selected_bitstrings": calibration["target_selected_bitstrings"],
        "marked_selected_bitstrings": [
            bitstring_from_index(index, spec.num_x_qubits)
            for index in marked_indices
        ],
        "marked_full_bitstrings": [
            commitment_to_bitstring(embedded_commitments[index])
            for index in marked_indices
        ],
        "marked_true_costs": [
            _finite_or_none(embedded_values[index])
            for index in marked_indices
        ],
        "phase_oracle_check": {
            "marked_count": int(phase_probe.marked_mask.sum()),
            "aux_zero_probability": float(phase_probe.aux_zero_probability),
            "max_phase_error": float(phase_probe.max_phase_error),
            "simulation_seconds": float(phase_seconds),
        },
        "grover_result": {
            "iterations": int(grover_result.iterations),
            "marked_probability": float(grover_result.marked_probability),
            "unmarked_probability": float(grover_result.unmarked_probability),
            "aux_zero_probability": float(grover_result.aux_zero_probability),
            "argmax_selected_bitstring": bitstring_from_index(argmax_index, spec.num_x_qubits),
            "argmax_full_bitstring": commitment_to_bitstring(embedded_commitments[argmax_index]),
            "argmax_true_cost": _finite_or_none(embedded_values[argmax_index]),
            "simulation_seconds": float(grover_seconds),
            "top_probability_rows": top_probability_rows(
                probabilities=grover_result.x_probabilities,
                piece_values=piece_values,
                max_values=max_values,
                marked=spec.marked_mask(),
                embedded_commitments=embedded_commitments,
                embedded_values=embedded_values,
                num_bits=spec.num_x_qubits,
            ),
        },
        "resources": {
            "phase_oracle": circuit_resource_summary(phase_circuit, decompose_reps=3),
            "grover_circuit": circuit_resource_summary(grover_circuit, decompose_reps=2),
        },
        "threshold_calibration": {
            "selected_row": calibration["selected_row"],
            "candidate_rows": calibration["candidate_rows"],
        },
    }



def oracle_and_circuit_explanation(
    learned: LearnedPieces,
    spec: GateLevelMaxAffineOracleSpec,
) -> dict[str, object]:
    return {
        "oracle_definition": {
            "search_variable": "selected commitment bits x only; fixed load d is a condition, not a Grover variable",
            "integer_surrogate": "V_hat_int(x) = max_r L_r(x)",
            "threshold_oracle": "O_tau |x> = (-1)^[V_hat_int(x) <= tau] |x>",
            "top1_threshold": "tau = 0 marks the learned selected-subspace optimum",
            "low_cost_set_threshold": "tau = 1 marks the calibrated top-3 low-cost selected-subspace set in the default run",
        },
        "circuit_registers": {
            "x": list(spec.bit_labels),
            "piece_value_registers": [
                f"v{piece_index} stores L{piece_index}(x) before comparison"
                for piece_index in range(spec.piece_count)
            ],
            "piece_flags": [
                f"flag{piece_index} stores [L{piece_index}(x) <= tau]"
                for piece_index in range(spec.piece_count)
            ],
            "adder_ancillas": "Qiskit WeightedAdder carry/control qubits for each piece",
            "comparator_ancillas": "Qiskit IntegerComparator ancillas for each piece",
        },
        "circuit_sequence": [
            "Apply Hadamard gates on all x qubits to create the Grover uniform superposition.",
            "For each learned affine piece L_r, temporarily flip inverted x inputs so (1-x_i) terms become addable x_i terms.",
            "Use WeightedAdder to write the integer piece value L_r(x) into value register v_r.",
            "Use IntegerComparator to write flag_r = [L_r(x) <= tau].",
            "Apply a multi-controlled phase on all flag qubits, implementing max_r L_r(x) <= tau as AND_r flag_r.",
            "Run inverse comparators and inverse adders, restoring all value and ancilla registers to zero.",
            "Apply the standard Grover diffuser on x only.",
        ],
        "reversibility": {
            "compute_phase_uncompute": "The oracle is U_f^dagger Z_flags U_f, so it is unitary and leaves only a phase on x.",
            "auxiliary_check": "The experiment reports aux_zero_probability to verify that non-x registers return to |0>.",
        },
        "grover_loop": {
            "iteration_choice": "The simulator uses the standard marked-count-based Grover iteration estimate for the current threshold.",
            "readout": "Statevector probabilities are marginalized onto the x register; no measurement is used inside the oracle.",
            "true_cost_validation": "The most amplified selected bitstring is embedded back into the full T=2 commitment and checked with the exact UC evaluator.",
        },
        "learned_piece_summary": {
            "optimum_selected_bitstring": bitstring_from_index(learned.optimum_index, spec.num_x_qubits),
            "integer_weights": [int(weight) for weight in learned.integer_weights],
            "piece_count": int(spec.piece_count),
        },
    }

def learned_to_dict(
    learned: LearnedPieces,
    spec: GateLevelMaxAffineOracleSpec,
) -> dict[str, object]:
    return {
        "value_formula": "V_hat_int(x) = max_r L_r(x)",
        "bit_labels": list(learned.bit_labels),
        "optimum_selected_bitstring": bitstring_from_index(learned.optimum_index, spec.num_x_qubits),
        "single_flip_cost_gaps": [
            None if gap is None else float(gap)
            for gap in learned.single_flip_gaps
        ],
        "integer_weights": [int(weight) for weight in learned.integer_weights],
        "piece_groups": learned.piece_groups,
        "pieces": [
            {
                "name": piece.name,
                "bias": int(piece.bias),
                "weights": [int(weight) for weight in piece.weights],
                "inverted_bit_indices": [int(index) for index in piece.inverted_bit_indices],
                "terms": piece_terms(piece, spec.bit_labels),
            }
            for piece in spec.pieces
        ],
    }


def piece_terms(
    piece: GateLevelAffinePieceSpec,
    bit_labels: tuple[str, ...],
) -> list[str]:
    inverted = set(piece.inverted_bit_indices)
    terms = []
    if piece.bias != 0:
        terms.append(str(int(piece.bias)))
    for bit_index, (label, weight) in enumerate(zip(bit_labels, piece.weights)):
        if weight == 0:
            continue
        if bit_index in inverted:
            terms.append(f"{weight}*(1-{label})")
        else:
            terms.append(f"{weight}*{label}")
    return terms


def top_probability_rows(
    *,
    probabilities: np.ndarray,
    piece_values: np.ndarray,
    max_values: np.ndarray,
    marked: np.ndarray,
    embedded_commitments: np.ndarray,
    embedded_values: np.ndarray,
    num_bits: int,
    limit: int = 10,
) -> list[dict[str, object]]:
    ranks = finite_rank_map(embedded_values)
    rows = []
    for state_index in np.argsort(probabilities)[::-1][:limit]:
        state_index = int(state_index)
        rows.append(
            {
                "selected_bitstring": bitstring_from_index(state_index, num_bits),
                "probability": float(probabilities[state_index]),
                "piece_values": [int(value) for value in piece_values[state_index].tolist()],
                "max_affine_value": int(max_values[state_index]),
                "marked": bool(marked[state_index]),
                "embedded_full_bitstring": commitment_to_bitstring(
                    embedded_commitments[state_index]
                ),
                "embedded_true_cost": _finite_or_none(embedded_values[state_index]),
                "embedded_true_rank": ranks.get(state_index),
            }
        )
    return rows


def finite_rank_map(values: np.ndarray) -> dict[int, int]:
    rows = [int(index) for index in np.argsort(values) if np.isfinite(values[index])]
    return {index: rank + 1 for rank, index in enumerate(rows)}


def parse_indices(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_counts(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _finite_or_none(value: float) -> float | None:
    if np.isfinite(value):
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_learned_small_max_affine_gate_level_oracle.json"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--selected-generators", type=parse_indices, default=(0, 5))
    parser.add_argument("--target-counts", type=parse_counts, default=(1, 3))
    parser.add_argument("--max-weight", type=int, default=7)
    args = parser.parse_args()

    summary = run(
        args.instance,
        args.results,
        args.horizon,
        args.selected_generators,
        args.target_counts,
        args.max_weight,
    )
    compact = {
        "instance": summary["instance"],
        "method": summary["method"],
        "scope": summary["scope"],
        "selected_generators": summary["selected_generators"],
        "learned_model": summary["learned_model"],
        "embedded_subspace_validation": summary["embedded_subspace_validation"],
        "target_cases": [
            {
                "target_count_request": row["target_count_request"],
                "actual_target_count": row["actual_target_count"],
                "threshold_tau": row["threshold_tau"],
                "exact_marked_set": row["exact_marked_set"],
                "marked_selected_bitstrings": row["marked_selected_bitstrings"],
                "marked_true_costs": row["marked_true_costs"],
                "phase_oracle_check": row["phase_oracle_check"],
                "grover_result": {
                    key: value
                    for key, value in row["grover_result"].items()
                    if key != "top_probability_rows"
                },
                "resources": {
                    "phase_qubits": row["resources"]["phase_oracle"]["num_qubits"],
                    "grover_qubits": row["resources"]["grover_circuit"]["num_qubits"],
                    "grover_depth": row["resources"]["grover_circuit"]["depth"],
                },
            }
            for row in summary["target_cases"]
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

