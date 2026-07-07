from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_t2_ancilla_vqc import (  # noqa: E402
    commitment_row,
    leading_time_window_instance,
)
from experiments.stage1_case14_t2_max_affine_value_surrogate import (  # noqa: E402
    build_boundary_training_profile,
    diagnostics_to_dict,
    fit_boundary_aware_max_affine_model,
    parse_optional_target_counts,
)
from experiments.stage1_case14_t2_structured_value_surrogate import (  # noqa: E402
    calibrated_prediction_threshold,
    load_or_evaluate_values,
)
from experiments.stage1_case14_t2_value_register_comparator import (  # noqa: E402
    comparator_evaluation,
    tie_tolerant_threshold_case_for_top_count,
    value_evaluation_to_dict,
)
from qubit_value_function.commitment import all_commitments  # noqa: E402
from qubit_value_function.max_affine import (  # noqa: E402
    evaluate_max_affine_value_function,
    max_affine_gate_counts,
)
from qubit_value_function.structured_features import structured_commitment_features  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def max_affine_threshold_oracle_labels(
    *,
    predictions: np.ndarray,
    value_domain: np.ndarray,
    tau_true: float,
    values: np.ndarray,
    use_calibrated_threshold: bool,
    tie_tolerance: float = 1e-6,
) -> dict[str, object]:
    """Build labels for the incumbent-dependent max-affine threshold oracle.

    ``true_improving_labels`` is always defined by exact ED values. The oracle
    labels are induced by the max-affine predictions, optionally after
    calibrating the prediction-space threshold against the current true
    improving set.
    """

    predictions = np.asarray(predictions, dtype=float)
    values = np.asarray(values, dtype=float)
    value_domain = np.asarray(value_domain, dtype=bool)
    if predictions.shape != values.shape or value_domain.shape != values.shape:
        raise ValueError("predictions, values, and value_domain must have the same shape")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be nonnegative")

    true_improving = (
        value_domain
        & np.isfinite(values)
        & (values < float(tau_true) - float(tie_tolerance))
    )
    if not np.any(true_improving):
        marked = np.zeros(values.shape, dtype=bool)
        evaluation = comparator_evaluation(marked, true_improving)
        return {
            "marked": marked,
            "true_improving_labels": true_improving,
            "oracle_diagnostics": {
                **evaluation,
                "marked_count": int(marked.sum()),
                "true_improving_count": 0,
                "search_exhausted": True,
                "use_calibrated_threshold": bool(use_calibrated_threshold),
                "calibrated_prediction_threshold": None,
                "calibration_margin": None,
                "tau_true": float(tau_true),
            },
        }

    calibrated_threshold_value: float | None = None
    calibration_margin: float | None = None
    if use_calibrated_threshold:
        calibrated_threshold_value, calibration_margin = calibrated_prediction_threshold(
            predictions,
            true_improving,
            value_domain,
        )
        prediction_threshold = float(calibrated_threshold_value)
    else:
        prediction_threshold = float(tau_true)

    marked = value_domain & np.isfinite(predictions) & (predictions <= prediction_threshold)
    evaluation = comparator_evaluation(marked, true_improving)
    return {
        "marked": marked,
        "true_improving_labels": true_improving,
        "oracle_diagnostics": {
            **evaluation,
            "marked_count": int(marked.sum()),
            "true_improving_count": int(true_improving.sum()),
            "search_exhausted": False,
            "use_calibrated_threshold": bool(use_calibrated_threshold),
            "calibrated_prediction_threshold": _finite_or_none(
                calibrated_threshold_value
            ),
            "calibration_margin": _finite_or_none(calibration_margin),
            "tau_true": float(tau_true),
            "prediction_threshold_used": float(prediction_threshold),
        },
    }


def sample_after_grover_iterations(
    marked: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
    target_labels: np.ndarray | None = None,
) -> dict[str, object]:
    """Sample one state after state-vector Grover iterations.

    This keeps only the state vector. It never constructs a ``2^n x 2^n``
    oracle matrix.
    """

    marked = np.asarray(marked, dtype=bool)
    if target_labels is None:
        target_labels = marked
    target_labels = np.asarray(target_labels, dtype=bool)
    if marked.shape != target_labels.shape:
        raise ValueError("marked and target_labels must have the same shape")
    dimension = marked.size
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("state dimension must be a positive power of two")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")

    state = np.ones(dimension, dtype=float) / np.sqrt(dimension)
    for _ in range(int(iterations)):
        state[marked] *= -1.0
        state = 2.0 * np.mean(state) - state
    probabilities = state * state
    probabilities = probabilities / probabilities.sum()
    sampled = int(rng.choice(dimension, p=probabilities))
    marked_probability = float(probabilities[marked].sum())
    target_probability = float(probabilities[target_labels].sum())
    return {
        "sampled_index": sampled,
        "target_probability": target_probability,
        "marked_probability": marked_probability,
        "nonmarked_probability": float(1.0 - marked_probability),
    }


def bbht_search_current_threshold(
    *,
    marked: np.ndarray,
    true_improving_labels: np.ndarray,
    values: np.ndarray,
    incumbent_value: float,
    rng: np.random.Generator,
    lambda_growth: float,
    max_trials: int,
    tie_tolerance: float = 1e-6,
) -> dict[str, object]:
    """Run BBHT-style unknown-target-count Grover search for one threshold."""

    marked = np.asarray(marked, dtype=bool)
    true_improving_labels = np.asarray(true_improving_labels, dtype=bool)
    values = np.asarray(values, dtype=float)
    if marked.shape != true_improving_labels.shape or marked.shape != values.shape:
        raise ValueError("marked, true_improving_labels, and values must match")
    if not (1.0 < float(lambda_growth) < 4.0 / 3.0):
        raise ValueError("lambda_growth must be in (1, 4/3)")
    if max_trials < 0:
        raise ValueError("max_trials must be nonnegative")

    dimension = marked.size
    p = 1.0
    p_cap = float(np.sqrt(dimension))
    trials: list[dict[str, object]] = []
    total_iterations = 0
    improved_index: int | None = None
    for trial_index in range(int(max_trials)):
        upper = max(1, int(np.ceil(p)))
        iterations_j = int(rng.integers(0, upper))
        sample = sample_after_grover_iterations(
            marked,
            iterations_j,
            rng,
            target_labels=true_improving_labels,
        )
        sampled = int(sample["sampled_index"])
        total_iterations += iterations_j
        sampled_value = values[sampled]
        success = bool(
            np.isfinite(sampled_value)
            and sampled_value < float(incumbent_value) - float(tie_tolerance)
        )
        trials.append(
            {
                "trial": int(trial_index),
                "p": float(p),
                "iterations_j": int(iterations_j),
                "sampled_index": sampled,
                "sampled_value": _finite_or_none(sampled_value),
                "oracle_marked": bool(marked[sampled]),
                "true_improving": bool(true_improving_labels[sampled]),
                "success": bool(success),
                "target_probability": float(sample["target_probability"]),
                "marked_probability": float(sample["marked_probability"]),
                "nonmarked_probability": float(sample["nonmarked_probability"]),
            }
        )
        if success:
            improved_index = sampled
            break
        p = min(float(lambda_growth) * p, p_cap)

    return {
        "improved_index": improved_index,
        "trials": trials,
        "trial_count": int(len(trials)),
        "oracle_diffuser_iterations": int(total_iterations),
    }


def run_adaptive_minimum_search(
    *,
    values: np.ndarray,
    predictions: np.ndarray,
    value_domain: np.ndarray,
    initial_index: int,
    rng: np.random.Generator,
    lambda_growth: float,
    max_rounds: int,
    max_bbht_trials_per_threshold: int,
    use_calibrated_threshold: bool,
    stop_after_no_improvement: int,
    tie_tolerance: float,
) -> dict[str, object]:
    """Run the outer incumbent-update loop around BBHT threshold search."""

    values = np.asarray(values, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    value_domain = np.asarray(value_domain, dtype=bool)
    if values.shape != predictions.shape or values.shape != value_domain.shape:
        raise ValueError("values, predictions, and value_domain must have matching shape")
    if not value_domain[int(initial_index)] or not np.isfinite(values[int(initial_index)]):
        raise ValueError("initial_index must be finite and inside value_domain")

    finite_domain_indices = np.flatnonzero(value_domain & np.isfinite(values))
    exact_optimum_index = int(
        finite_domain_indices[np.argmin(values[finite_domain_indices])]
    )
    incumbent = int(initial_index)
    no_improvement_count = 0
    rounds: list[dict[str, object]] = []
    stop_reason = "max_rounds"

    for round_idx in range(int(max_rounds)):
        incumbent_value = float(values[incumbent])
        oracle = max_affine_threshold_oracle_labels(
            predictions=predictions,
            value_domain=value_domain,
            tau_true=incumbent_value,
            values=values,
            use_calibrated_threshold=use_calibrated_threshold,
            tie_tolerance=tie_tolerance,
        )
        true_improving_labels = np.asarray(oracle["true_improving_labels"], dtype=bool)
        if not np.any(true_improving_labels):
            stop_reason = "no_true_improving_state"
            break

        bbht = bbht_search_current_threshold(
            marked=np.asarray(oracle["marked"], dtype=bool),
            true_improving_labels=true_improving_labels,
            values=values,
            incumbent_value=incumbent_value,
            rng=rng,
            lambda_growth=lambda_growth,
            max_trials=max_bbht_trials_per_threshold,
            tie_tolerance=tie_tolerance,
        )
        improved_index = bbht["improved_index"]
        improved = improved_index is not None
        before = incumbent
        if improved:
            incumbent = int(improved_index)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        rounds.append(
            {
                "round": int(round_idx),
                "incumbent_index_before": int(before),
                "incumbent_value_before": float(incumbent_value),
                "true_improving_count": int(true_improving_labels.sum()),
                "oracle_marked_count": int(np.asarray(oracle["marked"], dtype=bool).sum()),
                "oracle_diagnostics": oracle["oracle_diagnostics"],
                "bbht_trials": bbht["trials"],
                "bbht_trial_count": int(bbht["trial_count"]),
                "oracle_diffuser_iterations": int(bbht["oracle_diffuser_iterations"]),
                "improved": bool(improved),
                "new_incumbent_index": int(incumbent),
                "new_incumbent_value": float(values[incumbent]),
                "no_improvement_count": int(no_improvement_count),
            }
        )
        if no_improvement_count >= int(stop_after_no_improvement):
            stop_reason = "no_improvement_budget"
            break
        if incumbent == exact_optimum_index:
            stop_reason = "exact_optimum_reached"
            break

    return {
        "initial_index": int(initial_index),
        "final_incumbent_index": int(incumbent),
        "final_incumbent_value": float(values[incumbent]),
        "exact_optimum_index": int(exact_optimum_index),
        "exact_optimum_value": float(values[exact_optimum_index]),
        "success_found_exact_optimum": bool(incumbent == exact_optimum_index),
        "rounds": rounds,
        "stop_reason": stop_reason,
        "total_outer_rounds": int(len(rounds)),
        "total_bbht_trials": int(
            sum(int(row["bbht_trial_count"]) for row in rounds)
        ),
        "total_oracle_diffuser_iterations": int(
            sum(int(row["oracle_diffuser_iterations"]) for row in rounds)
        ),
        "incumbent_cost_trajectory": [
            float(values[int(initial_index)])
        ]
        + [float(row["new_incumbent_value"]) for row in rounds],
    }


def run(
    instance_path: Path,
    results_path: Path,
    *,
    horizon: int,
    register_bits: int,
    same_time_order: int,
    adjacent_time_order: int,
    piece_count: int,
    candidate_count: int,
    initialization: str,
    ridge: float,
    cut_tolerance: float,
    tie_tolerance: float,
    value_cache: Path | None,
    lambda_growth: float,
    max_rounds: int,
    max_bbht_trials_per_threshold: int,
    seed: int,
    initial_index: str,
    initial_random_samples: int,
    use_calibrated_threshold: bool,
    stop_after_no_improvement: int,
    boundary_target_counts: list[int],
    boundary_rank_window: int,
    boundary_weight: float,
    boundary_target_side_weight: float,
    boundary_nontarget_side_weight: float,
    boundary_rounds: int,
    boundary_misorder_boost: float,
) -> dict[str, object]:
    source_instance = load_uc_instance(instance_path)
    instance = leading_time_window_instance(source_instance, horizon)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    generator_names = [gen.name for gen in instance.generators]

    values, logic_feasible, evaluation_seconds, value_source = load_or_evaluate_values(
        instance,
        commitments,
        instance_path,
        horizon,
        value_cache,
    )
    finite = np.isfinite(values)
    value_domain = logic_feasible & finite
    feasible_indices = np.flatnonzero(value_domain)
    feasible_values = values[value_domain]
    finite_sorted_indices = [int(idx) for idx in np.argsort(values) if finite[idx]]
    if feasible_indices.size == 0:
        raise ValueError("at least one finite feasible commitment is required")

    threshold_cases = [
        tie_tolerant_threshold_case_for_top_count(values, count, tie_tolerance)
        for count in boundary_target_counts
    ]
    matrix = structured_commitment_features(
        instance,
        commitments,
        same_time_interaction_order=same_time_order,
        adjacent_time_interaction_order=adjacent_time_order,
        include_dispatch_proxy=True,
    )
    feasible_features = matrix.features[value_domain]
    sample_weights, anchor_weights, candidate_order, boundary_training = build_boundary_training_profile(
        feasible_values,
        threshold_cases,
        boundary_target_counts=boundary_target_counts,
        boundary_rank_window=boundary_rank_window,
        boundary_weight=boundary_weight,
        boundary_target_side_weight=boundary_target_side_weight,
        boundary_nontarget_side_weight=boundary_nontarget_side_weight,
    )
    model, diagnostics, boundary_fit = fit_boundary_aware_max_affine_model(
        feasible_features=feasible_features,
        feasible_values=feasible_values,
        names=matrix.names,
        threshold_cases=threshold_cases,
        value_domain=value_domain,
        piece_count=piece_count,
        candidate_count=candidate_count,
        initialization=initialization,
        cut_tolerance=cut_tolerance,
        ridge=ridge,
        sample_weights=sample_weights,
        anchor_weights=anchor_weights,
        candidate_order=candidate_order,
        boundary_training=boundary_training,
        boundary_rounds=boundary_rounds,
        boundary_misorder_boost=boundary_misorder_boost,
    )
    predictions = model.predict(matrix.features)
    value_evaluation = evaluate_max_affine_value_function(
        model,
        feasible_features,
        feasible_values,
    )
    rng = np.random.default_rng(seed)
    incumbent_index = select_initial_incumbent(
        values=values,
        value_domain=value_domain,
        mode=initial_index,
        initial_random_samples=initial_random_samples,
        rng=rng,
    )
    adaptive = run_adaptive_minimum_search(
        values=values,
        predictions=predictions,
        value_domain=value_domain,
        initial_index=incumbent_index,
        rng=rng,
        lambda_growth=lambda_growth,
        max_rounds=max_rounds,
        max_bbht_trials_per_threshold=max_bbht_trials_per_threshold,
        use_calibrated_threshold=use_calibrated_threshold,
        stop_after_no_improvement=stop_after_no_improvement,
        tie_tolerance=tie_tolerance,
    )

    final_index = int(adaptive["final_incumbent_index"])
    optimum_index = int(adaptive["exact_optimum_index"])
    summary = {
        "instance": f"case14_T{horizon}",
        "source": str(instance_path),
        "method": "adaptive Grover minimum search with max-affine value-register comparator oracle",
        "horizon": int(horizon),
        "generators": generator_names,
        "fixed_load_mw": instance.fixed_load,
        "num_bits": int(commitments.reshape((commitments.shape[0], -1)).shape[1]),
        "num_commitments": int(commitments.shape[0]),
        "logic_feasible_count": int(logic_feasible.sum()),
        "finite_value_count": int(finite.sum()),
        "value_domain_count": int(value_domain.sum()),
        "value_evaluation_seconds": float(evaluation_seconds),
        "value_source": value_source,
        "register_bits": int(register_bits),
        "lambda_growth": float(lambda_growth),
        "max_rounds": int(max_rounds),
        "max_bbht_trials_per_threshold": int(max_bbht_trials_per_threshold),
        "seed": int(seed),
        "initial_index_mode": initial_index,
        "initial_random_samples": int(initial_random_samples),
        "use_calibrated_threshold": bool(use_calibrated_threshold),
        "tie_tolerance": float(tie_tolerance),
        "feature_family": {
            "same_time_interaction_order": int(same_time_order),
            "adjacent_time_interaction_order": int(adjacent_time_order),
            "include_dispatch_proxy": True,
            "feature_count": len(matrix.names),
            "feature_names": matrix.names,
        },
        "max_affine_model": {
            "initialization": initialization,
            "piece_count": int(piece_count),
            "actual_piece_count": int(model.coefficients.shape[0]),
            "candidate_count": int(candidate_count),
            "gate_counts": max_affine_gate_counts(
                len(matrix.names),
                model.coefficients.shape[0],
            ),
            "fit_diagnostics": diagnostics_to_dict(diagnostics),
            "boundary_training": boundary_training,
            "boundary_fit": boundary_fit,
            "value_regression": value_evaluation_to_dict(value_evaluation),
        },
        "oracle_decomposition": {
            "feature_register": "compute structured f(x) reversibly",
            "affine_piece_registers": "compute L_r(x)=b_r+theta_r*f(x) for each piece",
            "max_register": "reversible comparator tree computes max_r L_r(x)",
            "threshold_update": "load incumbent-dependent threshold tau before each adaptive search round",
            "threshold_comparator": "mark if feasible(x) and max-affine value <= tau_register",
            "bbht_inner_loop": "randomize Grover iterations because target count is unknown",
            "classical_verification": "evaluate measured commitment using exact ED/cache before incumbent update",
            "uncompute": "reverse max, affine pieces, features, and feasibility after phase marking",
        },
        "initial_incumbent": commitment_row(
            commitments,
            generator_names,
            values,
            int(adaptive["initial_index"]),
        ),
        "final_incumbent": commitment_row(
            commitments,
            generator_names,
            values,
            final_index,
        ),
        "exact_optimum": commitment_row(
            commitments,
            generator_names,
            values,
            optimum_index,
        ),
        "runner_up": commitment_row(
            commitments,
            generator_names,
            values,
            finite_sorted_indices[1],
        ),
        "success_found_exact_optimum": bool(adaptive["success_found_exact_optimum"]),
        "total_outer_rounds": int(adaptive["total_outer_rounds"]),
        "total_bbht_trials": int(adaptive["total_bbht_trials"]),
        "total_oracle_diffuser_iterations": int(
            adaptive["total_oracle_diffuser_iterations"]
        ),
        "incumbent_cost_trajectory": adaptive["incumbent_cost_trajectory"],
        "rounds": adaptive["rounds"],
        "stop_reason": adaptive["stop_reason"],
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def select_initial_incumbent(
    *,
    values: np.ndarray,
    value_domain: np.ndarray,
    mode: str,
    initial_random_samples: int,
    rng: np.random.Generator,
) -> int:
    feasible_indices = np.flatnonzero(np.asarray(value_domain, dtype=bool))
    if feasible_indices.size == 0:
        raise ValueError("no feasible finite initial states are available")
    if mode == "random":
        return int(rng.choice(feasible_indices))
    if mode == "best_of_random":
        sample_count = min(max(int(initial_random_samples), 1), feasible_indices.size)
        sampled = rng.choice(feasible_indices, size=sample_count, replace=False)
        return int(sampled[np.argmin(values[sampled])])
    if mode.isdigit():
        candidate = int(mode)
        if candidate < 0 or candidate >= values.size or not value_domain[candidate]:
            raise ValueError("numeric initial-index must be inside value_domain")
        return candidate
    raise ValueError("initial_index must be 'random', 'best_of_random', or an integer")


def parse_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
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
        default=Path("results/stage1_case14_t2_max_affine_adaptive_grover_search.json"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--register-bits", type=int, default=20)
    parser.add_argument("--same-time-order", type=int, default=4)
    parser.add_argument("--adjacent-time-order", type=int, default=1)
    parser.add_argument("--piece-count", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=128)
    parser.add_argument("--initialization", type=str, default="least_squares")
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--cut-tolerance", type=float, default=1e-9)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--value-cache", type=Path, default=None)
    parser.add_argument("--lambda-growth", type=float, default=8.0 / 7.0)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--max-bbht-trials-per-threshold", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-index", type=str, default="random")
    parser.add_argument("--initial-random-samples", type=int, default=8)
    parser.add_argument("--use-calibrated-threshold", type=parse_bool, default=True)
    parser.add_argument("--stop-after-no-improvement", type=int, default=10)
    parser.add_argument("--boundary-target-counts", type=parse_optional_target_counts, default=[])
    parser.add_argument("--boundary-rank-window", type=int, default=0)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--boundary-target-side-weight", type=float, default=1.0)
    parser.add_argument("--boundary-nontarget-side-weight", type=float, default=1.0)
    parser.add_argument("--boundary-rounds", type=int, default=0)
    parser.add_argument("--boundary-misorder-boost", type=float, default=1.0)
    args = parser.parse_args()
    summary = run(
        args.instance,
        args.results,
        horizon=args.horizon,
        register_bits=args.register_bits,
        same_time_order=args.same_time_order,
        adjacent_time_order=args.adjacent_time_order,
        piece_count=args.piece_count,
        candidate_count=args.candidate_count,
        initialization=args.initialization,
        ridge=args.ridge,
        cut_tolerance=args.cut_tolerance,
        tie_tolerance=args.tie_tolerance,
        value_cache=args.value_cache,
        lambda_growth=args.lambda_growth,
        max_rounds=args.max_rounds,
        max_bbht_trials_per_threshold=args.max_bbht_trials_per_threshold,
        seed=args.seed,
        initial_index=args.initial_index,
        initial_random_samples=args.initial_random_samples,
        use_calibrated_threshold=args.use_calibrated_threshold,
        stop_after_no_improvement=args.stop_after_no_improvement,
        boundary_target_counts=args.boundary_target_counts,
        boundary_rank_window=args.boundary_rank_window,
        boundary_weight=args.boundary_weight,
        boundary_target_side_weight=args.boundary_target_side_weight,
        boundary_nontarget_side_weight=args.boundary_nontarget_side_weight,
        boundary_rounds=args.boundary_rounds,
        boundary_misorder_boost=args.boundary_misorder_boost,
    )
    compact = {
        "final_incumbent_cost": summary["final_incumbent"]["total_cost"],
        "exact_optimum_cost": summary["exact_optimum"]["total_cost"],
        "success_found_exact_optimum": summary["success_found_exact_optimum"],
        "total_outer_rounds": summary["total_outer_rounds"],
        "total_bbht_trials": summary["total_bbht_trials"],
        "total_oracle_diffuser_iterations": summary["total_oracle_diffuser_iterations"],
        "stop_reason": summary["stop_reason"],
        "incumbent_cost_trajectory": summary["incumbent_cost_trajectory"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
