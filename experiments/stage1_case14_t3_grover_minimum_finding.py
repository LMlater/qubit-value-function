from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_t2_ancilla_vqc import commitment_row, leading_time_window_instance  # noqa: E402
from experiments.stage1_case14_t2_max_affine_value_surrogate import (  # noqa: E402
    build_boundary_training_profile,
    fit_boundary_aware_max_affine_model,
)
from experiments.stage1_case14_t2_structured_value_surrogate import load_or_evaluate_values  # noqa: E402
from experiments.stage1_case14_t2_value_register_comparator import (  # noqa: E402
    tie_tolerant_threshold_case_for_top_count,
    value_evaluation_to_dict,
)
from qubit_value_function.commitment import all_commitments  # noqa: E402
from qubit_value_function.grover_minimum import (  # noqa: E402
    run_grover_minimum_finding,
    summarize_minimum_finding_runs,
)
from qubit_value_function.max_affine import evaluate_max_affine_value_function  # noqa: E402
from qubit_value_function.structured_features import structured_commitment_features  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402
from qubit_value_function.value_surrogate import quantize_values  # noqa: E402


def run(
    instance_path: Path,
    results_path: Path,
    *,
    horizon: int,
    trials: int,
    max_rounds: int,
    seed: int,
    register_bits: int,
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

    matrix = structured_commitment_features(
        instance,
        commitments,
        same_time_interaction_order=4,
        adjacent_time_interaction_order=2,
        include_dispatch_proxy=True,
    )
    feasible_features = matrix.features[value_domain]
    target_counts = [1, 4, 8, 16, 32, 64, 128]
    threshold_cases = [
        tie_tolerant_threshold_case_for_top_count(values, count, 1e-6)
        for count in target_counts
    ]
    sample_weights, anchor_weights, candidate_order, boundary_training = build_boundary_training_profile(
        feasible_values,
        threshold_cases,
        boundary_target_counts=[128],
        boundary_rank_window=32,
        boundary_weight=16.0,
        boundary_target_side_weight=1.0,
        boundary_nontarget_side_weight=3.0,
    )
    model, diagnostics, boundary_fit = fit_boundary_aware_max_affine_model(
        feasible_features=feasible_features,
        feasible_values=feasible_values,
        names=matrix.names,
        threshold_cases=threshold_cases,
        value_domain=value_domain,
        piece_count=32,
        candidate_count=128,
        initialization="least_squares",
        cut_tolerance=1e-9,
        ridge=0.0,
        sample_weights=sample_weights,
        anchor_weights=anchor_weights,
        candidate_order=candidate_order,
        boundary_training=boundary_training,
        boundary_rounds=4,
        boundary_misorder_boost=3.0,
    )
    predictions = model.predict(matrix.features)
    feasible_predictions = predictions[value_domain]
    value_evaluation = evaluate_max_affine_value_function(model, feasible_features, feasible_values)

    prediction_min = float(np.min(feasible_predictions))
    prediction_max = float(np.max(feasible_predictions))
    quantized_predictions = quantize_values(
        predictions,
        value_min=prediction_min,
        value_max=prediction_max,
        bits=register_bits,
    ).astype(float)

    exact_runs = [
        run_grover_minimum_finding(
            values,
            values,
            value_domain,
            max_rounds=max_rounds,
            seed=seed + trial,
        )
        for trial in range(trials)
    ]
    surrogate_runs = [
        run_grover_minimum_finding(
            values,
            predictions,
            value_domain,
            max_rounds=max_rounds,
            seed=seed + trial,
        )
        for trial in range(trials)
    ]
    quantized_runs = [
        run_grover_minimum_finding(
            values,
            quantized_predictions,
            value_domain,
            max_rounds=max_rounds,
            seed=seed + trial,
            strict_tolerance=0.0,
        )
        for trial in range(trials)
    ]

    optimum_index = int(finite_sorted_indices[0])
    summary = {
        "instance": f"case14_T{horizon}",
        "source": str(instance_path),
        "method": "state-vector Grover minimum finding simulation with fixed-load max-affine value-register oracle",
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
        "trials": int(trials),
        "max_rounds": int(max_rounds),
        "seed": int(seed),
        "oracle_rule": (
            "Each round marks feasible states with predicted value below the incumbent predicted value; "
            "a sampled state updates the incumbent only if its true UC/SCUC value is lower."
        ),
        "value_register": {
            "surrogate": "V_hat_theta(x)=max_r(b_r+theta_r^T f(x))",
            "feature_count": len(matrix.names),
            "piece_count": int(model.coefficients.shape[0]),
            "boundary_training": boundary_training,
            "boundary_fit": boundary_fit,
            "register_bits": int(register_bits),
            "quantized_value_min": prediction_min,
            "quantized_value_max": prediction_max,
        },
        "value_regression": value_evaluation_to_dict(value_evaluation),
        "fit_diagnostics": {
            "requested_piece_count": diagnostics.requested_piece_count,
            "actual_piece_count": diagnostics.actual_piece_count,
            "lower_bound_violations": diagnostics.lower_bound_violations,
            "max_lower_bound_violation": diagnostics.max_lower_bound_violation,
        },
        "optimum": commitment_row(commitments, generator_names, values, optimum_index),
        "runner_up": commitment_row(commitments, generator_names, values, finite_sorted_indices[1]),
        "exact_value_oracle": {
            "summary": summarize_minimum_finding_runs(exact_runs),
            "runs": [run_to_dict(item) for item in exact_runs],
        },
        "max_affine_surrogate_oracle": {
            "summary": summarize_minimum_finding_runs(surrogate_runs),
            "runs": [run_to_dict(item) for item in surrogate_runs],
        },
        "quantized_max_affine_oracle": {
            "summary": summarize_minimum_finding_runs(quantized_runs),
            "runs": [run_to_dict(item) for item in quantized_runs],
        },
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_to_dict(run) -> dict[str, object]:
    item = asdict(run)
    item["rounds"] = [asdict(round_item) for round_item in run.rounds]
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t3_grover_minimum_finding.json"),
    )
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--register-bits", type=int, default=20)
    parser.add_argument("--value-cache", type=Path, default=None)
    args = parser.parse_args()
    summary = run(
        args.instance,
        args.results,
        horizon=args.horizon,
        trials=args.trials,
        max_rounds=args.max_rounds,
        seed=args.seed,
        register_bits=args.register_bits,
        value_cache=args.value_cache,
    )
    compact = {
        key: value
        for key, value in summary.items()
        if key
        not in {
            "exact_value_oracle",
            "max_affine_surrogate_oracle",
            "quantized_max_affine_oracle",
        }
    }
    compact["exact_value_oracle"] = summary["exact_value_oracle"]["summary"]
    compact["max_affine_surrogate_oracle"] = summary["max_affine_surrogate_oracle"]["summary"]
    compact["quantized_max_affine_oracle"] = summary["quantized_max_affine_oracle"]["summary"]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
