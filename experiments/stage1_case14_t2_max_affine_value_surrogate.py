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
    parse_target_counts,
)
from experiments.stage1_case14_t2_structured_value_surrogate import (  # noqa: E402
    calibrated_prediction_threshold,
    compact_calibrated_threshold_result,
    load_or_evaluate_values,
    parse_register_bits,
    run_calibrated_quantized_register_family,
    run_calibrated_threshold_case,
)
from experiments.stage1_case14_t2_value_register_comparator import (  # noqa: E402
    compact_threshold_result,
    run_floating_threshold_case,
    run_quantized_register_family,
    tie_tolerant_threshold_case_for_top_count,
    value_evaluation_to_dict,
)
from qubit_value_function.commitment import all_commitments  # noqa: E402
from qubit_value_function.max_affine import (  # noqa: E402
    evaluate_max_affine_value_function,
    fit_max_affine_value_function,
    max_affine_gate_counts,
)
from qubit_value_function.structured_features import structured_commitment_features  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def run(
    instance_path: Path,
    results_path: Path,
    horizon: int,
    target_counts: list[int],
    register_bits: list[int],
    same_time_order: int,
    adjacent_time_order: int,
    piece_counts: list[int],
    candidate_count: int,
    initializations: list[str],
    ridge: float,
    cut_tolerance: float,
    tie_tolerance: float,
    value_cache: Path | None,
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
    feasible_values = values[value_domain]
    finite_sorted_indices = [idx for idx in np.argsort(values) if finite[idx]]
    threshold_cases = [
        tie_tolerant_threshold_case_for_top_count(values, count, tie_tolerance)
        for count in target_counts
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

    model_rows = []
    for initialization in initializations:
        for piece_count in piece_counts:
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
            feasible_predictions = predictions[value_domain]
            value_evaluation = evaluate_max_affine_value_function(
                model,
                feasible_features,
                feasible_values,
            )
            model_rows.append(
                {
                    "initialization": initialization,
                    "piece_count": int(piece_count),
                    "actual_piece_count": int(model.coefficients.shape[0]),
                    "candidate_count": int(candidate_count),
                    "feature_count": len(matrix.names),
                    "gate_counts": max_affine_gate_counts(len(matrix.names), model.coefficients.shape[0]),
                    "fit_diagnostics": diagnostics_to_dict(diagnostics),
                    "boundary_fit": boundary_fit,
                    "value_regression": value_evaluation_to_dict(value_evaluation),
                    "floating_comparator_results": [
                        run_floating_threshold_case(
                            predictions=predictions,
                            value_domain=value_domain,
                            threshold_case=case,
                        )
                        for case in threshold_cases
                    ],
                    "calibrated_comparator_results": [
                        run_calibrated_threshold_case(
                            predictions=predictions,
                            value_domain=value_domain,
                            threshold_case=case,
                        )
                        for case in threshold_cases
                    ],
                    "fixed_point_register_results": [
                        run_quantized_register_family(
                            predictions=predictions,
                            feasible_predictions=feasible_predictions,
                            value_domain=value_domain,
                            threshold_cases=threshold_cases,
                            bits=count,
                        )
                        for count in register_bits
                    ],
                    "calibrated_fixed_point_register_results": [
                        run_calibrated_quantized_register_family(
                            predictions=predictions,
                            feasible_predictions=feasible_predictions,
                            value_domain=value_domain,
                            threshold_cases=threshold_cases,
                            bits=count,
                        )
                        for count in register_bits
                    ],
                }
            )

    summary = {
        "instance": f"case14_T{horizon}",
        "source": str(instance_path),
        "method": "max-affine structured value-function surrogate with value-register comparator simulation",
        "horizon": horizon,
        "generators": generator_names,
        "fixed_load_mw": instance.fixed_load,
        "num_bits": int(commitments.reshape((commitments.shape[0], -1)).shape[1]),
        "num_commitments": int(commitments.shape[0]),
        "logic_feasible_count": int(logic_feasible.sum()),
        "finite_value_count": int(finite.sum()),
        "value_domain_count": int(value_domain.sum()),
        "value_evaluation_seconds": float(evaluation_seconds),
        "value_source": value_source,
        "target_counts": [int(count) for count in target_counts],
        "register_bits": [int(count) for count in register_bits],
        "tie_tolerance": float(tie_tolerance),
        "boundary_training": boundary_training,
        "feature_family": {
            "same_time_interaction_order": int(same_time_order),
            "adjacent_time_interaction_order": int(adjacent_time_order),
            "include_dispatch_proxy": True,
            "feature_count": len(matrix.names),
            "feature_names": matrix.names,
        },
        "oracle_decomposition": {
            "feature_register": "compute structured f(x) reversibly",
            "affine_piece_registers": "compute L_r(x)=b_r+theta_r*f(x) for each piece",
            "max_register": "reversible comparator tree computes max_r L_r(x)",
            "threshold_comparator": "mark if feasible(x) and max-affine value <= tau_register",
            "uncompute": "reverse max, affine pieces, features, and feasibility after phase marking",
        },
        "optimum": commitment_row(commitments, generator_names, values, finite_sorted_indices[0]),
        "runner_up": commitment_row(commitments, generator_names, values, finite_sorted_indices[1]),
        "model_rows": model_rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def diagnostics_to_dict(diagnostics) -> dict[str, object]:
    return {
        "requested_piece_count": diagnostics.requested_piece_count,
        "actual_piece_count": diagnostics.actual_piece_count,
        "selected_anchor_indices": diagnostics.selected_anchor_indices,
        "selected_anchor_values": diagnostics.selected_anchor_values,
        "selected_anchor_residuals": diagnostics.selected_anchor_residuals,
        "lower_bound_violations": diagnostics.lower_bound_violations,
        "max_lower_bound_violation": diagnostics.max_lower_bound_violation,
    }


def parse_piece_counts(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_initializations(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_optional_target_counts(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    return parse_target_counts(raw)


def build_boundary_training_profile(
    feasible_values: np.ndarray,
    threshold_cases: list[dict[str, object]],
    *,
    boundary_target_counts: list[int],
    boundary_rank_window: int,
    boundary_weight: float,
    boundary_target_side_weight: float,
    boundary_nontarget_side_weight: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, object]]:
    feasible_values = np.asarray(feasible_values, dtype=float)
    metadata = {
        "active": False,
        "focus_target_counts": [int(count) for count in boundary_target_counts],
        "rank_window": int(boundary_rank_window),
        "boundary_weight": float(boundary_weight),
        "target_side_weight": float(boundary_target_side_weight),
        "nontarget_side_weight": float(boundary_nontarget_side_weight),
        "weighted_state_count": 0,
        "max_sample_weight": 1.0,
    }
    if (
        not boundary_target_counts
        or boundary_rank_window <= 0
        or boundary_weight <= 0.0
        or boundary_target_side_weight <= 0.0
        or boundary_nontarget_side_weight <= 0.0
    ):
        return None, None, None, metadata

    focus_counts = {int(count) for count in boundary_target_counts}
    weights = np.ones(feasible_values.shape[0], dtype=float)
    feasible_order = np.argsort(feasible_values)
    active_counts: list[int] = []
    for case in threshold_cases:
        requested = int(case["target_count_request"])
        if requested not in focus_counts:
            continue
        actual = int(case["actual_target_count"])
        window = min(boundary_rank_window, actual, feasible_values.shape[0] - actual)
        if window <= 0:
            continue
        active_counts.append(requested)
        for distance in range(window):
            bonus = float(boundary_weight) / float(distance + 1)
            target_row = int(feasible_order[actual - 1 - distance])
            non_target_row = int(feasible_order[actual + distance])
            weights[target_row] += float(boundary_target_side_weight) * bonus
            weights[non_target_row] += float(boundary_nontarget_side_weight) * bonus

    if not active_counts:
        return None, None, None, metadata

    candidate_order = boundary_candidate_order(feasible_values, weights)
    metadata.update(
        {
            "active": True,
            "active_target_counts": sorted(set(active_counts)),
            "weighted_state_count": int(np.sum(weights > 1.0)),
            "max_sample_weight": float(np.max(weights)),
        }
    )
    return weights, weights.copy(), candidate_order, metadata


def boundary_candidate_order(feasible_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    feasible_values = np.asarray(feasible_values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weight_first = np.lexsort((feasible_values, -weights))
    value_first = np.argsort(feasible_values)
    combined: list[int] = []
    seen: set[int] = set()
    for idx in np.concatenate([weight_first, value_first]):
        idx = int(idx)
        if idx in seen:
            continue
        combined.append(idx)
        seen.add(idx)
    return np.asarray(combined, dtype=int)


def fit_boundary_aware_max_affine_model(
    *,
    feasible_features: np.ndarray,
    feasible_values: np.ndarray,
    names: list[str],
    threshold_cases: list[dict[str, object]],
    value_domain: np.ndarray,
    piece_count: int,
    candidate_count: int,
    initialization: str,
    cut_tolerance: float,
    ridge: float,
    sample_weights: np.ndarray | None,
    anchor_weights: np.ndarray | None,
    candidate_order: np.ndarray | None,
    boundary_training: dict[str, object],
    boundary_rounds: int,
    boundary_misorder_boost: float,
) -> tuple[object, object, dict[str, object]]:
    current_sample_weights = None if sample_weights is None else np.array(sample_weights, dtype=float, copy=True)
    current_anchor_weights = None if anchor_weights is None else np.array(anchor_weights, dtype=float, copy=True)
    current_candidate_order = candidate_order
    rounds = max(int(boundary_rounds), 0)
    history: list[dict[str, object]] = []
    best_model = None
    best_diagnostics = None
    best_score: tuple[int, float, int] | None = None

    for round_idx in range(rounds + 1):
        if current_anchor_weights is not None:
            current_candidate_order = boundary_candidate_order(feasible_values, current_anchor_weights)
        model, diagnostics = fit_max_affine_value_function(
            feasible_features,
            feasible_values,
            names,
            piece_count=piece_count,
            candidate_count=candidate_count,
            initialization=initialization,
            cut_tolerance=cut_tolerance,
            ridge=ridge,
            sample_weights=current_sample_weights,
            anchor_weights=current_anchor_weights,
            candidate_order=current_candidate_order,
        )
        predictions = model.predict(feasible_features)
        focus_summary = evaluate_boundary_focus(
            predictions=predictions,
            threshold_cases=threshold_cases,
            value_domain=value_domain,
            focus_target_counts=boundary_training.get("focus_target_counts", []),
        )
        focus_summary["round"] = round_idx
        history.append(focus_summary)
        score = (
            int(focus_summary["exact_focus_count"]),
            float(focus_summary["min_margin"]),
            -int(focus_summary["total_focus_errors"]),
        )
        if best_score is None or score > best_score:
            best_model = model
            best_diagnostics = diagnostics
            best_score = score
        if (
            not boundary_training.get("active", False)
            or initialization != "least_squares"
            or round_idx >= rounds
            or focus_summary["all_focus_exact"]
            or current_sample_weights is None
            or current_anchor_weights is None
            or boundary_misorder_boost <= 1.0
        ):
            break
        current_sample_weights, current_anchor_weights = boost_boundary_misordered_states(
            current_sample_weights,
            current_anchor_weights,
            predictions=predictions,
            threshold_cases=threshold_cases,
            value_domain=value_domain,
            focus_target_counts=boundary_training.get("focus_target_counts", []),
            boost=float(boundary_misorder_boost),
        )

    return best_model, best_diagnostics, {
        "active": bool(boundary_training.get("active", False)),
        "rounds_requested": rounds,
        "misorder_boost": float(boundary_misorder_boost),
        "history": history,
        "best_score": None if best_score is None else list(best_score),
    }


def evaluate_boundary_focus(
    *,
    predictions: np.ndarray,
    threshold_cases: list[dict[str, object]],
    value_domain: np.ndarray,
    focus_target_counts: list[int],
) -> dict[str, object]:
    if not focus_target_counts:
        return {
            "focus_results": [],
            "exact_focus_count": 0,
            "all_focus_exact": True,
            "min_margin": float("inf"),
            "total_focus_errors": 0,
        }
    focus_set = {int(count) for count in focus_target_counts}
    full_predictions = embed_feasible_predictions(predictions, value_domain)
    focus_results = []
    for case in threshold_cases:
        requested = int(case["target_count_request"])
        if requested not in focus_set:
            continue
        result = run_calibrated_threshold_case(
            predictions=full_predictions,
            value_domain=value_domain,
            threshold_case=case,
        )
        focus_results.append(
            {
                "target_count_request": requested,
                "actual_target_count": int(case["actual_target_count"]),
                "exact_marked_set": bool(result["comparator_evaluation"]["correct_marked_set"]),
                "calibration_margin": float(result["calibration_margin"]),
                "false_positive_count": int(result["comparator_evaluation"]["false_positive_count"]),
                "false_negative_count": int(result["comparator_evaluation"]["false_negative_count"]),
            }
        )
    if not focus_results:
        return {
            "focus_results": [],
            "exact_focus_count": 0,
            "all_focus_exact": True,
            "min_margin": float("inf"),
            "total_focus_errors": 0,
        }
    return {
        "focus_results": focus_results,
        "exact_focus_count": int(sum(item["exact_marked_set"] for item in focus_results)),
        "all_focus_exact": bool(all(item["exact_marked_set"] for item in focus_results)),
        "min_margin": float(min(item["calibration_margin"] for item in focus_results)),
        "total_focus_errors": int(
            sum(item["false_positive_count"] + item["false_negative_count"] for item in focus_results)
        ),
    }


def boost_boundary_misordered_states(
    sample_weights: np.ndarray,
    anchor_weights: np.ndarray,
    *,
    predictions: np.ndarray,
    threshold_cases: list[dict[str, object]],
    value_domain: np.ndarray,
    focus_target_counts: list[int],
    boost: float,
) -> tuple[np.ndarray, np.ndarray]:
    boosted_sample_weights = np.array(sample_weights, dtype=float, copy=True)
    boosted_anchor_weights = np.array(anchor_weights, dtype=float, copy=True)
    focus_set = {int(count) for count in focus_target_counts}
    full_predictions = embed_feasible_predictions(predictions, value_domain)
    for case in threshold_cases:
        requested = int(case["target_count_request"])
        if requested not in focus_set:
            continue
        labels = np.asarray(case["labels"], dtype=bool)[value_domain]
        calibrated_threshold, _ = calibrated_prediction_threshold(
            full_predictions,
            np.asarray(case["labels"], dtype=bool),
            value_domain,
        )
        false_negative = np.flatnonzero(labels & (predictions > calibrated_threshold))
        false_positive = np.flatnonzero((~labels) & (predictions <= calibrated_threshold))
        for idx in np.concatenate([false_negative, false_positive]):
            idx = int(idx)
            boosted_sample_weights[idx] *= boost
            boosted_anchor_weights[idx] *= boost
    return boosted_sample_weights, boosted_anchor_weights


def embed_feasible_predictions(predictions: np.ndarray, value_domain: np.ndarray) -> np.ndarray:
    full_predictions = np.zeros(value_domain.shape[0], dtype=float)
    full_predictions[value_domain] = predictions
    return full_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_max_affine_value_surrogate.json"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument(
        "--target-counts",
        type=parse_target_counts,
        default=parse_target_counts("1,4,8,16,32,48,64"),
    )
    parser.add_argument("--register-bits", type=parse_register_bits, default=parse_register_bits("16,20"))
    parser.add_argument("--same-time-order", type=int, default=4)
    parser.add_argument("--adjacent-time-order", type=int, default=1)
    parser.add_argument("--piece-counts", type=parse_piece_counts, default=parse_piece_counts("2,4,8,16,32,64"))
    parser.add_argument("--candidate-count", type=int, default=128)
    parser.add_argument("--initializations", type=parse_initializations, default=parse_initializations("floor,least_squares"))
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--cut-tolerance", type=float, default=1e-9)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--value-cache", type=Path, default=None)
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
        args.horizon,
        args.target_counts,
        args.register_bits,
        args.same_time_order,
        args.adjacent_time_order,
        args.piece_counts,
        args.candidate_count,
        args.initializations,
        args.ridge,
        args.cut_tolerance,
        args.tie_tolerance,
        args.value_cache,
        args.boundary_target_counts,
        args.boundary_rank_window,
        args.boundary_weight,
        args.boundary_target_side_weight,
        args.boundary_nontarget_side_weight,
        args.boundary_rounds,
        args.boundary_misorder_boost,
    )
    compact = {
        key: value
        for key, value in summary.items()
        if key not in {"model_rows"}
    }
    compact["model_rows"] = [
        {
            "initialization": row["initialization"],
            "piece_count": row["piece_count"],
            "actual_piece_count": row["actual_piece_count"],
            "feature_count": row["feature_count"],
            "gate_counts": row["gate_counts"],
            "value_regression": row["value_regression"],
            "boundary_fit": row["boundary_fit"],
            "fit_diagnostics": {
                key: value
                for key, value in row["fit_diagnostics"].items()
                if key not in {"selected_anchor_indices", "selected_anchor_values", "selected_anchor_residuals"}
            },
            "floating_comparator_results": [
                compact_threshold_result(item)
                for item in row["floating_comparator_results"]
            ],
            "calibrated_comparator_results": [
                compact_calibrated_threshold_result(item)
                for item in row["calibrated_comparator_results"]
            ],
            "calibrated_fixed_point_register_results": [
                {
                    "register_bits": family["register_bits"],
                    "results": [
                        compact_calibrated_threshold_result(item)
                        for item in family["results"]
                    ],
                }
                for family in row["calibrated_fixed_point_register_results"]
            ],
        }
        for row in summary["model_rows"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
