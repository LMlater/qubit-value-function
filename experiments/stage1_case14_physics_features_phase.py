from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_single_period import single_period_instance, threshold_grid  # noqa: E402
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator, startup_cost  # noqa: E402
from qubit_value_function.feature_phase import evaluate_feature_phase_model, fit_feature_phase_model  # noqa: E402
from qubit_value_function.oracle import grover_with_oracle_matrix, phase_oracle_errors, verify_phase_oracle  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def physics_features(instance, commitments: np.ndarray) -> tuple[np.ndarray, list[str]]:
    gens = instance.generators
    pmax = np.array([gen.p_max for gen in gens], dtype=float)
    pmin = np.array([gen.p_min for gen in gens], dtype=float)
    startup = np.array([
        startup_cost(instance, _single_generator_commitment(len(gens), idx)) for idx in range(len(gens))
    ])
    full_load = instance.fixed_load[0]
    bits = commitments.reshape((commitments.shape[0], -1)).astype(float)
    cap = bits @ pmax
    min_gen = bits @ pmin
    start = bits @ startup
    pmax_cost = bits @ np.array([gen.cost_usd[-1] for gen in gens], dtype=float)
    avg_cost = bits @ np.array([gen.cost_usd[-1] / max(gen.p_max, 1.0) for gen in gens], dtype=float)
    reserve_cap = bits @ np.array(
        [gen.p_max if gen.reserve_eligibility else 0.0 for gen in gens],
        dtype=float,
    )
    count = bits.sum(axis=1)
    margin = cap - full_load
    reserve_margin = reserve_cap - sum((reserve.amount[0] for reserve in instance.reserves), 0.0)
    names = [
        "1",
        "unit_count",
        "pmax_capacity",
        "pmin_generation",
        "startup_cost",
        "pmax_cost",
        "avg_cost_sum",
        "reserve_capacity",
        "capacity_margin",
        "reserve_margin",
        "capacity_margin_sq",
        "reserve_margin_sq",
        "startup_x_margin",
        "cost_x_margin",
        "capacity_x_reserve",
    ]
    raw = np.column_stack(
        [
            np.ones(bits.shape[0]),
            count,
            cap,
            min_gen,
            start,
            pmax_cost,
            avg_cost,
            reserve_cap,
            margin,
            reserve_margin,
            margin**2,
            reserve_margin**2,
            start * margin,
            pmax_cost * margin,
            cap * reserve_cap,
        ]
    )
    features = raw.copy()
    for col in range(1, features.shape[1]):
        scale = np.max(np.abs(features[:, col]))
        if scale > 0:
            features[:, col] = features[:, col] / scale
    return features, names


def run(instance_path: Path, results_path: Path, period: int) -> dict[str, object]:
    instance = single_period_instance(load_uc_instance(instance_path), period)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bitstrings = [commitment_to_bitstring(commitment) for commitment in commitments]
    values = np.array([FixedCommitmentEvaluator(instance).evaluate(c).total_cost for c in commitments])
    _, _, tau = threshold_grid(values)
    labels = values <= tau
    features, names = physics_features(instance, commitments)
    model = fit_feature_phase_model(features, labels, names)
    evaluation = evaluate_feature_phase_model(model, features, labels)
    oracle = model.oracle_matrix(features)
    grover = grover_with_oracle_matrix(oracle, labels)
    rows = sorted(
        [
            {"bitstring": bitstring, "total_cost": float(value), "target": bool(label)}
            for bitstring, value, label in zip(bitstrings, values, labels)
        ],
        key=lambda item: item["total_cost"],
    )
    summary = {
        "instance": str(instance_path),
        "method": "case14 fixed-threshold physics-feature phase model",
        "period": period,
        "fixed_load_mw": instance.fixed_load,
        "generators": [gen.name for gen in instance.generators],
        "num_bits": int(commitments.reshape((commitments.shape[0], -1)).shape[1]),
        "num_commitments": int(commitments.shape[0]),
        "threshold": float(tau),
        "optimum": rows[0],
        "runner_up": rows[1],
        "feature_names": names,
        "coefficients": [float(v) for v in model.coefficients],
        "evaluation": evaluation,
        "oracle_checks": verify_phase_oracle(oracle, atol=1e-8),
        "oracle_errors": phase_oracle_errors(oracle),
        "grover": {
            key: value for key, value in grover.items() if key != "probabilities"
        },
        "rows": rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _single_generator_commitment(num_generators: int, generator_idx: int) -> np.ndarray:
    commitment = np.zeros((num_generators, 1), dtype=int)
    commitment[generator_idx, 0] = 1
    return commitment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--results", type=Path, default=Path("results/stage1_case14_physics_features_phase.json"))
    parser.add_argument("--period", type=int, default=0)
    args = parser.parse_args()
    summary = run(args.instance, args.results, args.period)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
