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
    parse_target_counts,
)
from experiments.stage1_case14_t2_value_register_comparator import (  # noqa: E402
    compact_threshold_result,
    comparator_evaluation,
    grover_with_predicted_phase_oracle,
    ideal_grover_summary,
    run_floating_threshold_case,
    run_quantized_register_family,
    tie_tolerant_threshold_case_for_top_count,
    value_evaluation_to_dict,
)
from qubit_value_function.commitment import all_commitments  # noqa: E402
from qubit_value_function.structured_features import structured_commitment_features  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402
from qubit_value_function.value_surrogate import (  # noqa: E402
    evaluate_scalar_value_function,
    fit_scalar_value_function,
    quantize_values,
)


def run(
    instance_path: Path,
    results_path: Path,
    horizon: int,
    target_counts: list[int],
    register_bits: list[int],
    ridge: float,
    tie_tolerance: float,
    max_same_time_order: int,
    max_adjacent_time_order: int,
    value_cache: Path | None,
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

    family_rows = []
    for family in _feature_families(max_same_time_order, max_adjacent_time_order):
        matrix = structured_commitment_features(
            instance,
            commitments,
            same_time_interaction_order=family["same_time_interaction_order"],
            adjacent_time_interaction_order=family["adjacent_time_interaction_order"],
            include_dispatch_proxy=family["include_dispatch_proxy"],
        )
        feasible_features = matrix.features[value_domain]
        model = fit_scalar_value_function(
            feasible_features,
            feasible_values,
            matrix.names,
            ridge=ridge,
        )
        predictions = model.predict(matrix.features)
        feasible_predictions = predictions[value_domain]
        value_evaluation = evaluate_scalar_value_function(
            model,
            feasible_features,
            feasible_values,
        )
        family_rows.append(
            {
                "family": family["name"],
                "same_time_interaction_order": family["same_time_interaction_order"],
                "adjacent_time_interaction_order": family["adjacent_time_interaction_order"],
                "include_dispatch_proxy": family["include_dispatch_proxy"],
                "feature_count": len(matrix.names),
                "feature_names": matrix.names,
                "training_domain_count": int(value_domain.sum()),
                "ridge": float(ridge),
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
        "method": "structured scalar value-function surrogate with value-register comparator simulation",
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
        "oracle_decomposition": {
            "feasibility_oracle": "exact Boolean feasibility register",
            "value_register": "compute structured V_theta(x,u) features and fixed-point value register",
            "threshold_comparator": "mark if feasible(x) and value_register <= tau_register",
            "uncompute": "reverse feature/value-register and feasibility computations after phase marking",
        },
        "feature_family_note": (
            "Families use commitment bits, physical aggregate features, local same-time "
            "unit interactions, temporal on-pairs, startup terms, and optional deterministic "
            "merit-order dispatch proxy terms. No value table is used as an input feature."
        ),
        "optimum": commitment_row(commitments, generator_names, values, finite_sorted_indices[0]),
        "runner_up": commitment_row(commitments, generator_names, values, finite_sorted_indices[1]),
        "family_rows": family_rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_or_evaluate_values(
    instance,
    commitments: np.ndarray,
    instance_path: Path,
    horizon: int,
    value_cache: Path | None,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    cache_path = value_cache
    if cache_path is None:
        cache_path = Path("results") / f"value_cache_{instance_path.stem}_h{horizon}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        values = cached["values"]
        logic_feasible = cached["logic_feasible"].astype(bool)
        if values.shape == (commitments.shape[0],) and logic_feasible.shape == (commitments.shape[0],):
            return values, logic_feasible, 0.0, str(cache_path)

    t0 = time.perf_counter()
    values, logic_feasible = evaluate_values(instance, commitments)
    evaluation_seconds = time.perf_counter() - t0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        values=values,
        logic_feasible=logic_feasible.astype(bool),
        horizon=np.array([horizon], dtype=int),
        num_commitments=np.array([commitments.shape[0]], dtype=int),
    )
    return values, logic_feasible, float(evaluation_seconds), str(cache_path)


def run_calibrated_threshold_case(
    *,
    predictions: np.ndarray,
    value_domain: np.ndarray,
    threshold_case: dict[str, object],
) -> dict[str, object]:
    labels = np.asarray(threshold_case["labels"], dtype=bool)
    calibrated_threshold, margin = calibrated_prediction_threshold(
        predictions,
        labels,
        value_domain,
    )
    predicted_marked = np.zeros(labels.shape, dtype=bool)
    predicted_marked[value_domain] = predictions[value_domain] <= calibrated_threshold
    exact_grover = ideal_grover_summary(labels)
    grover = grover_with_predicted_phase_oracle(
        predicted_marked,
        labels,
        iterations=int(exact_grover["iterations"]),
    )
    return {
        "target_count_request": int(threshold_case["target_count_request"]),
        "actual_target_count": int(threshold_case["actual_target_count"]),
        "threshold": float(threshold_case["threshold"]),
        "calibrated_prediction_threshold": float(calibrated_threshold),
        "calibration_margin": float(margin),
        "rank_separable": bool(margin >= 0.0),
        "exact_grover": exact_grover,
        "comparator_evaluation": comparator_evaluation(predicted_marked, labels),
        "grover": grover,
    }


def run_calibrated_quantized_register_family(
    *,
    predictions: np.ndarray,
    feasible_predictions: np.ndarray,
    value_domain: np.ndarray,
    threshold_cases: list[dict[str, object]],
    bits: int,
) -> dict[str, object]:
    value_min = float(np.min(feasible_predictions))
    value_max = float(np.max(feasible_predictions))
    prediction_register = quantize_values(
        predictions,
        value_min=value_min,
        value_max=value_max,
        bits=bits,
    )
    return {
        "register_bits": int(bits),
        "value_min": value_min,
        "value_max": value_max,
        "results": [
            run_calibrated_quantized_threshold_case(
                predictions=predictions,
                prediction_register=prediction_register,
                value_min=value_min,
                value_max=value_max,
                value_domain=value_domain,
                threshold_case=case,
                bits=bits,
            )
            for case in threshold_cases
        ],
    }


def run_calibrated_quantized_threshold_case(
    *,
    predictions: np.ndarray,
    prediction_register: np.ndarray,
    value_min: float,
    value_max: float,
    value_domain: np.ndarray,
    threshold_case: dict[str, object],
    bits: int,
) -> dict[str, object]:
    labels = np.asarray(threshold_case["labels"], dtype=bool)
    calibrated_threshold, margin = calibrated_prediction_threshold(
        predictions,
        labels,
        value_domain,
    )
    tau_register = int(
        quantize_values(
            np.array([calibrated_threshold]),
            value_min=value_min,
            value_max=value_max,
            bits=bits,
        )[0]
    )
    predicted_marked = np.zeros(labels.shape, dtype=bool)
    predicted_marked[value_domain] = prediction_register[value_domain] <= tau_register
    exact_grover = ideal_grover_summary(labels)
    grover = grover_with_predicted_phase_oracle(
        predicted_marked,
        labels,
        iterations=int(exact_grover["iterations"]),
    )
    return {
        "target_count_request": int(threshold_case["target_count_request"]),
        "actual_target_count": int(threshold_case["actual_target_count"]),
        "threshold": float(threshold_case["threshold"]),
        "calibrated_prediction_threshold": float(calibrated_threshold),
        "calibration_margin": float(margin),
        "rank_separable": bool(margin >= 0.0),
        "tau_register": tau_register,
        "comparator_evaluation": comparator_evaluation(predicted_marked, labels),
        "grover": grover,
    }


def calibrated_prediction_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    value_domain: np.ndarray,
) -> tuple[float, float]:
    target_predictions = predictions[value_domain & labels]
    non_target_predictions = predictions[value_domain & ~labels]
    if target_predictions.size == 0 or non_target_predictions.size == 0:
        raise ValueError("calibration requires both target and non-target feasible states")
    max_target = float(np.max(target_predictions))
    min_non_target = float(np.min(non_target_predictions))
    return (max_target + min_non_target) / 2.0, min_non_target - max_target


def _feature_families(
    max_same_time_order: int,
    max_adjacent_time_order: int,
) -> list[dict[str, object]]:
    if max_same_time_order < 1:
        raise ValueError("max_same_time_order must be at least 1")
    if max_adjacent_time_order < 1:
        raise ValueError("max_adjacent_time_order must be at least 1")
    families: list[dict[str, object]] = [
        {
            "name": "physical_order1",
            "same_time_interaction_order": 1,
            "adjacent_time_interaction_order": 1,
            "include_dispatch_proxy": False,
        },
        {
            "name": "physical_same_time_pairs",
            "same_time_interaction_order": 2,
            "adjacent_time_interaction_order": 1,
            "include_dispatch_proxy": False,
        },
    ]
    for order in range(1, max_same_time_order + 1):
        if order == 1:
            name = "merit_proxy_order1"
        elif order == 2:
            name = "merit_proxy_same_time_pairs"
        elif order == 3:
            name = "merit_proxy_same_time_triples"
        else:
            name = f"merit_proxy_same_time_order{order}"
        families.append(
            {
                "name": name,
                "same_time_interaction_order": order,
                "adjacent_time_interaction_order": 1,
                "include_dispatch_proxy": True,
            }
        )
    if max_adjacent_time_order >= 2:
        for adjacent_order in range(2, max_adjacent_time_order + 1):
            families.append(
                {
                    "name": (
                        f"merit_proxy_same_time_order{max_same_time_order}"
                        f"_adjacent_order{adjacent_order}"
                    ),
                    "same_time_interaction_order": max_same_time_order,
                    "adjacent_time_interaction_order": adjacent_order,
                    "include_dispatch_proxy": True,
                }
            )
    return families


def parse_register_bits(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_structured_value_surrogate.json"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument(
        "--target-counts",
        type=parse_target_counts,
        default=parse_target_counts("1,4,8,16,32,48,64"),
    )
    parser.add_argument("--register-bits", type=parse_register_bits, default=parse_register_bits("16"))
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-same-time-order", type=int, default=3)
    parser.add_argument("--max-adjacent-time-order", type=int, default=1)
    parser.add_argument("--value-cache", type=Path, default=None)
    parser.add_argument(
        "--include-triples",
        action="store_true",
        help="Deprecated alias that sets --max-same-time-order to at least 3.",
    )
    args = parser.parse_args()
    max_same_time_order = max(args.max_same_time_order, 3 if args.include_triples else 1)
    summary = run(
        args.instance,
        args.results,
        args.horizon,
        args.target_counts,
        args.register_bits,
        args.ridge,
        args.tie_tolerance,
        max_same_time_order,
        args.max_adjacent_time_order,
        args.value_cache,
    )
    compact = {
        key: value
        for key, value in summary.items()
        if key not in {"family_rows"}
    }
    compact["family_rows"] = [
        {
            "family": row["family"],
            "feature_count": row["feature_count"],
            "same_time_interaction_order": row["same_time_interaction_order"],
            "adjacent_time_interaction_order": row["adjacent_time_interaction_order"],
            "include_dispatch_proxy": row["include_dispatch_proxy"],
            "value_regression": row["value_regression"],
            "floating_comparator_results": [
                compact_threshold_result(item)
                for item in row["floating_comparator_results"]
            ],
            "calibrated_comparator_results": [
                compact_calibrated_threshold_result(item)
                for item in row["calibrated_comparator_results"]
            ],
            "fixed_point_register_results": [
                {
                    "register_bits": family["register_bits"],
                    "results": [
                        compact_threshold_result(item)
                        for item in family["results"]
                    ],
                }
                for family in row["fixed_point_register_results"]
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
        for row in summary["family_rows"]
    ]
    print(json.dumps(compact, indent=2))


def compact_calibrated_threshold_result(item: dict[str, object]) -> dict[str, object]:
    compact = compact_threshold_result(item)
    compact["rank_separable"] = item["rank_separable"]
    compact["calibration_margin"] = item["calibration_margin"]
    if "tau_register" in item:
        compact["tau_register"] = item["tau_register"]
    return compact


if __name__ == "__main__":
    main()
