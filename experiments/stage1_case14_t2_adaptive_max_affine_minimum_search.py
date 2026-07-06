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
from experiments.stage1_case14_t2_learned_small_max_affine_gate_level_oracle import (  # noqa: E402
    learn_gap_weighted_max_affine_pieces,
)
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.gate_level_oracle import (  # noqa: E402
    GateLevelAffinePieceSpec,
    GateLevelMaxAffineOracleSpec,
    bitstring_from_index,
    simulate_max_affine_grover,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def run_adaptive_trial(
    *,
    predicted_values: np.ndarray,
    true_values: np.ndarray,
    embedded_commitments: np.ndarray,
    learned_pieces: tuple[GateLevelAffinePieceSpec, ...],
    bit_labels: tuple[str, ...],
    initial_index: int,
    max_rounds: int,
    rng: np.random.Generator,
    strict_tolerance: float = 1e-9,
) -> dict[str, object]:
    """Run one adaptive Grover minimum-search trial on the embedded subspace.

    The threshold oracle is built from the integer max-affine surrogate
    ``predicted_values``. The incumbent is updated only when the sampled
    candidate improves the exact UC/ED value ``true_values``.
    """

    predicted_values = np.asarray(predicted_values, dtype=int)
    true_values = np.asarray(true_values, dtype=float)
    if predicted_values.shape != true_values.shape:
        raise ValueError("predicted_values and true_values must have the same shape")
    dimension = int(predicted_values.size)
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("search space size must be a positive power of two")
    if embedded_commitments.shape[0] != dimension:
        raise ValueError("embedded_commitments must contain one row per search state")
    if max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative")

    finite_true_indices = np.flatnonzero(np.isfinite(true_values))
    if finite_true_indices.size == 0:
        raise ValueError("at least one finite true value is required")
    incumbent = int(initial_index)
    if incumbent < 0 or incumbent >= dimension:
        raise ValueError("initial_index is outside the embedded search space")
    if not np.isfinite(true_values[incumbent]):
        raise ValueError("initial_index must have a finite true value")

    num_bits = int(np.log2(dimension))
    embedded_optimum_index = int(
        finite_true_indices[np.argmin(true_values[finite_true_indices])]
    )
    rounds: list[dict[str, object]] = []
    oracle_calls = 0
    stop_reason = "max_rounds"

    for round_index in range(max_rounds):
        if incumbent == embedded_optimum_index:
            stop_reason = "true_optimum_reached"
            break

        incumbent_predicted = int(predicted_values[incumbent])
        tau = incumbent_predicted - 1
        if tau < 0:
            stop_reason = "threshold_below_zero"
            break

        marked = predicted_values <= tau
        marked_count = int(marked.sum())
        if marked_count == 0:
            stop_reason = "no_smaller_surrogate_value"
            break

        before = incumbent
        spec = GateLevelMaxAffineOracleSpec(
            pieces=learned_pieces,
            threshold=int(tau),
            bit_labels=bit_labels,
            name=f"adaptive_round_{round_index}_tau_{int(tau)}",
        )
        grover_result = simulate_max_affine_grover(spec)
        probabilities = np.asarray(grover_result.x_probabilities, dtype=float)
        probabilities = probabilities / probabilities.sum()
        sampled = int(rng.choice(dimension, p=probabilities))
        oracle_calls += int(grover_result.iterations)
        accepted = bool(
            np.isfinite(true_values[sampled])
            and true_values[sampled] < true_values[before] - strict_tolerance
        )
        if accepted:
            incumbent = sampled

        rounds.append(
            {
                "round_index": int(round_index),
                "incumbent_index_before": int(before),
                "incumbent_selected_bitstring_before": bitstring_from_index(before, num_bits),
                "incumbent_full_bitstring_before": commitment_to_bitstring(
                    embedded_commitments[before]
                ),
                "incumbent_true_value_before": _finite_or_none(true_values[before]),
                "incumbent_predicted_value_before": int(predicted_values[before]),
                "threshold_tau": int(tau),
                "marked_count": int(marked_count),
                "grover_iterations": int(grover_result.iterations),
                "grover_marked_probability": float(grover_result.marked_probability),
                "grover_aux_zero_probability": float(grover_result.aux_zero_probability),
                "sampled_index": int(sampled),
                "sampled_selected_bitstring": bitstring_from_index(sampled, num_bits),
                "sampled_full_bitstring": commitment_to_bitstring(
                    embedded_commitments[sampled]
                ),
                "sampled_true_value": _finite_or_none(true_values[sampled]),
                "sampled_predicted_value": int(predicted_values[sampled]),
                "sampled_is_marked": bool(marked[sampled]),
                "accepted": bool(accepted),
                "incumbent_index_after": int(incumbent),
                "incumbent_true_value_after": _finite_or_none(true_values[incumbent]),
                "incumbent_predicted_value_after": int(predicted_values[incumbent]),
            }
        )

    best_true_value = float(true_values[incumbent])
    embedded_optimum_true_value = float(true_values[embedded_optimum_index])
    accepted_update_count = int(sum(bool(row["accepted"]) for row in rounds))
    rejection_count = int(len(rounds) - accepted_update_count)
    return {
        "initial_index": int(initial_index),
        "initial_selected_bitstring": bitstring_from_index(int(initial_index), num_bits),
        "initial_full_bitstring": commitment_to_bitstring(
            embedded_commitments[int(initial_index)]
        ),
        "best_index": int(incumbent),
        "best_selected_bitstring": bitstring_from_index(incumbent, num_bits),
        "best_full_bitstring": commitment_to_bitstring(embedded_commitments[incumbent]),
        "best_true_value": _finite_or_none(best_true_value),
        "best_predicted_value": int(predicted_values[incumbent]),
        "embedded_optimum_index": int(embedded_optimum_index),
        "embedded_optimum_selected_bitstring": bitstring_from_index(
            embedded_optimum_index,
            num_bits,
        ),
        "embedded_optimum_full_bitstring": commitment_to_bitstring(
            embedded_commitments[embedded_optimum_index]
        ),
        "embedded_optimum_true_value": _finite_or_none(embedded_optimum_true_value),
        "success": bool(incumbent == embedded_optimum_index),
        "true_optimality_gap": float(best_true_value - embedded_optimum_true_value),
        "round_count": int(len(rounds)),
        "oracle_calls": int(oracle_calls),
        "accepted_update_count": int(accepted_update_count),
        "rejection_count": int(rejection_count),
        "stop_reason": stop_reason,
        "rounds": rounds,
    }


def run(
    instance_path: Path,
    results_path: Path,
    horizon: int,
    selected_generator_indices: tuple[int, ...],
    max_weight: int,
    trial_count: int,
    max_rounds: int,
    seed: int,
) -> dict[str, object]:
    """Run the case14 T=2 embedded adaptive max-affine Grover experiment."""

    if horizon != 2:
        raise ValueError("this gate-level proof of concept is intentionally restricted to T=2")
    if len(selected_generator_indices) < 2:
        raise ValueError("at least two selected generators are required")
    if trial_count <= 0:
        raise ValueError("trial_count must be positive")

    source_instance = load_uc_instance(instance_path)
    instance = leading_time_window_instance(source_instance, horizon)
    generator_names = [gen.name for gen in instance.generators]
    commitments = all_commitments(len(instance.generators), instance.time_horizon)

    value_start = time.perf_counter()
    values, logic_feasible = evaluate_values(instance, commitments)
    value_seconds = time.perf_counter() - value_start
    finite = np.isfinite(values)
    finite_sorted_indices = [int(index) for index in np.argsort(values) if finite[index]]
    full_optimum_index = int(finite_sorted_indices[0])
    full_optimum_commitment = commitments[full_optimum_index]

    embedded_commitments = embedded_selected_commitments(
        full_optimum_commitment,
        selected_generator_indices,
    )
    embedded_values, embedded_logic_feasible = evaluate_values(instance, embedded_commitments)
    embedded_finite_indices = np.flatnonzero(np.isfinite(embedded_values))
    embedded_best_index = int(
        embedded_finite_indices[np.argmin(embedded_values[embedded_finite_indices])]
    )

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
        name="case14_t2_adaptive_max_affine_minimum_search_template",
    )
    predicted_values = template_spec.values_for_all_x()
    rng = np.random.default_rng(seed)
    trials = []
    for trial_index in range(trial_count):
        initial_index = int(rng.choice(embedded_finite_indices))
        trial_result = run_adaptive_trial(
            predicted_values=predicted_values,
            true_values=embedded_values,
            embedded_commitments=embedded_commitments,
            learned_pieces=learned.pieces,
            bit_labels=learned.bit_labels,
            initial_index=initial_index,
            max_rounds=max_rounds,
            rng=rng,
        )
        trial_result["trial_index"] = int(trial_index)
        trials.append(trial_result)

    summary = {
        "method": "adaptive Grover minimum finding with learned small integer max-affine gate-level oracle",
        "important_note": (
            "The Grover oracle uses the max-affine surrogate, but success is "
            "validated against the exact UC value function V_d(x)."
        ),
        "simulator_note": (
            "This is an ideal statevector / Qiskit simulator proof of concept, "
            "not a hardware experiment."
        ),
        "scope": {
            "instance": f"case14_T{horizon}",
            "source": str(instance_path),
            "full_case14_bits": int(commitments.reshape((commitments.shape[0], -1)).shape[1]),
            "gate_level_x_bits": int(template_spec.num_x_qubits),
            "selected_generator_indices": [int(index) for index in selected_generator_indices],
            "selected_generators": [
                generator_names[index] for index in selected_generator_indices
            ],
            "subspace_size": int(embedded_commitments.shape[0]),
            "note": (
                "Non-selected generators are fixed to the exhaustive T=2 optimum "
                "for this gate-level proof of concept; this is not a full 12-bit "
                "Grover search."
            ),
        },
        "settings": {
            "horizon": int(horizon),
            "max_weight": int(max_weight),
            "trial_count": int(trial_count),
            "max_rounds": int(max_rounds),
            "seed": int(seed),
        },
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
        "learned_model": learned_model_to_dict(learned, template_spec),
        "embedded_subspace_reference": {
            "base_commitment": commitment_to_bitstring(full_optimum_commitment),
            "base_note": (
                "Non-selected generators are fixed to the exhaustive T=2 optimum "
                "only to embed the 4-bit selected-generator proof-of-concept subspace."
            ),
            "logic_feasible_count": int(embedded_logic_feasible.sum()),
            "finite_value_count": int(np.isfinite(embedded_values).sum()),
            "optimum_index": int(embedded_best_index),
            "optimum_selected_bitstring": bitstring_from_index(
                embedded_best_index,
                template_spec.num_x_qubits,
            ),
            "optimum_full_bitstring": commitment_to_bitstring(
                embedded_commitments[embedded_best_index]
            ),
            "optimum_true_cost": _finite_or_none(embedded_values[embedded_best_index]),
            "all_selected_bitstrings_ranked_by_true_cost": ranked_embedded_rows(
                embedded_values,
                predicted_values,
                embedded_commitments,
                template_spec.num_x_qubits,
            ),
        },
        "adaptive_search_summary": summarize_trials(trials),
        "trials": trials,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_trials(trials: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate adaptive-search trial outcomes using true-value gaps."""

    if not trials:
        raise ValueError("at least one trial is required")
    gaps = np.array([float(trial["true_optimality_gap"]) for trial in trials], dtype=float)
    oracle_calls = np.array([int(trial["oracle_calls"]) for trial in trials], dtype=float)
    rounds = np.array([int(trial["round_count"]) for trial in trials], dtype=float)
    accepted = np.array(
        [int(trial["accepted_update_count"]) for trial in trials],
        dtype=float,
    )
    rejections = np.array([int(trial["rejection_count"]) for trial in trials], dtype=float)
    successes = np.array([bool(trial["success"]) for trial in trials], dtype=bool)
    return {
        "success_rate": float(np.mean(successes)),
        "success_count": int(successes.sum()),
        "trial_count": int(len(trials)),
        "mean_true_optimality_gap": float(np.mean(gaps)),
        "median_true_optimality_gap": float(np.median(gaps)),
        "max_true_optimality_gap": float(np.max(gaps)),
        "mean_oracle_calls": float(np.mean(oracle_calls)),
        "median_oracle_calls": float(np.median(oracle_calls)),
        "mean_rounds": float(np.mean(rounds)),
        "median_rounds": float(np.median(rounds)),
        "mean_accepted_update_count": float(np.mean(accepted)),
        "mean_rejection_count": float(np.mean(rejections)),
    }


def learned_model_to_dict(learned, spec: GateLevelMaxAffineOracleSpec) -> dict[str, object]:
    return {
        "value_formula": "V_hat_int(x) = max_r L_r(x)",
        "bit_labels": list(learned.bit_labels),
        "optimum_selected_bitstring": bitstring_from_index(
            int(learned.optimum_index),
            spec.num_x_qubits,
        ),
        "single_flip_cost_gaps": [
            None if gap is None else float(gap) for gap in learned.single_flip_gaps
        ],
        "integer_weights": [int(weight) for weight in learned.integer_weights],
        "piece_groups": [[int(index) for index in group] for group in learned.piece_groups],
        "pieces": [
            {
                "name": piece.name,
                "bias": int(piece.bias),
                "weights": [int(weight) for weight in piece.weights],
                "inverted_bit_indices": [
                    int(index) for index in piece.inverted_bit_indices
                ],
            }
            for piece in spec.pieces
        ],
        "predicted_values_by_selected_bitstring": [
            {
                "index": int(index),
                "selected_bitstring": bitstring_from_index(index, spec.num_x_qubits),
                "predicted_value": int(value),
            }
            for index, value in enumerate(spec.values_for_all_x())
        ],
    }


def ranked_embedded_rows(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    embedded_commitments: np.ndarray,
    num_bits: int,
) -> list[dict[str, object]]:
    rows = []
    for rank, index in enumerate(
        [int(idx) for idx in np.argsort(true_values) if np.isfinite(true_values[idx])],
        start=1,
    ):
        rows.append(
            {
                "rank": int(rank),
                "index": int(index),
                "selected_bitstring": bitstring_from_index(index, num_bits),
                "full_bitstring": commitment_to_bitstring(embedded_commitments[index]),
                "true_cost": _finite_or_none(true_values[index]),
                "predicted_value": int(predicted_values[index]),
            }
        )
    return rows


def parse_indices(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    if np.isfinite(value):
        return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_adaptive_max_affine_minimum_search.json"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--selected-generators", type=parse_indices, default=(0, 5))
    parser.add_argument("--max-weight", type=int, default=7)
    parser.add_argument("--trial-count", type=int, default=50)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = run(
        args.instance,
        args.results,
        args.horizon,
        args.selected_generators,
        args.max_weight,
        args.trial_count,
        args.max_rounds,
        args.seed,
    )
    compact = {
        "method": summary["method"],
        "important_note": summary["important_note"],
        "simulator_note": summary["simulator_note"],
        "scope": summary["scope"],
        "settings": summary["settings"],
        "embedded_subspace_reference": {
            key: value
            for key, value in summary["embedded_subspace_reference"].items()
            if key != "all_selected_bitstrings_ranked_by_true_cost"
        },
        "adaptive_search_summary": summary["adaptive_search_summary"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
