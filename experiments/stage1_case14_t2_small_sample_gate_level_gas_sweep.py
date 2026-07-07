from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_t2_small_sample_gate_level_max_affine_gas import run as run_small_sample_gas  # noqa: E402
from qubit_value_function.experiment_utils import sanitize_for_strict_json, write_strict_json  # noqa: E402


RUN_FIELDS = [
    "config",
    "selected_generators",
    "num_search_qubits",
    "dimension",
    "seed",
    "train_sample_count",
    "exclude_hidden_optimum_from_training",
    "exclude_hidden_optimum_from_initial",
    "training_contains_hidden_optimum",
    "initial_matches_hidden_optimum",
    "hidden_best_bitstring",
    "hidden_best_true_cost",
    "initial_bitstring",
    "initial_true_cost",
    "final_bitstring",
    "final_true_cost",
    "found_hidden_exact_optimum",
    "algorithmic_ed_lp_calls",
    "hidden_reference_ed_lp_calls",
    "circuit_executions",
    "total_shots",
    "max_qubits",
    "max_circuit_depth",
    "max_transpiled_depth",
    "status",
    "error",
]


def parse_configs(raw: str) -> list[tuple[int, ...]]:
    configs = []
    for chunk in raw.split(";"):
        stripped = chunk.strip()
        if not stripped:
            continue
        config = tuple(int(part.strip()) for part in stripped.split(",") if part.strip())
        if not config:
            raise argparse.ArgumentTypeError("configs cannot contain an empty generator list")
        configs.append(config)
    if not configs:
        raise argparse.ArgumentTypeError("at least one config is required")
    return configs


def parse_train_sample_counts(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one train sample count is required")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("train sample counts must be positive")
    return values


def build_grouped_summary(runs: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[tuple[int, ...], int, int, bool, bool], list[dict[str, object]]] = {}
    for row in runs:
        selected = tuple(int(index) for index in row["selected_generators"])
        key = (
            selected,
            int(row["num_search_qubits"]),
            int(row["train_sample_count"]),
            bool(row["exclude_hidden_optimum_from_training"]),
            bool(row.get("exclude_hidden_optimum_from_initial", False)),
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for key, rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])):
        selected, num_qubits, train_sample_count, exclude_hidden_training, exclude_hidden_initial = key
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        hidden_not_in_training = [
            row for row in ok_rows if row.get("training_contains_hidden_optimum") is False
        ]
        hidden_not_in_training_and_not_initial = [
            row
            for row in hidden_not_in_training
            if row.get("initial_matches_hidden_optimum") is False
        ]
        num_success = int(sum(bool(row.get("found_hidden_exact_optimum")) for row in ok_rows))
        summary.append(
            {
                "selected_generators": list(selected),
                "num_search_qubits": int(num_qubits),
                "train_sample_count": int(train_sample_count),
                "exclude_hidden_optimum_from_training": bool(exclude_hidden_training),
                "exclude_hidden_optimum_from_initial": bool(exclude_hidden_initial),
                "num_runs": int(len(rows)),
                "num_ok_runs": int(len(ok_rows)),
                "num_error_runs": int(len(rows) - len(ok_rows)),
                "num_success": num_success,
                "success_rate": _rate(num_success, len(ok_rows)),
                "success_rate_over_ok_runs": _rate(num_success, len(ok_rows)),
                "success_rate_over_attempted_runs": _rate(num_success, len(rows)),
                "num_runs_hidden_not_in_training": int(len(hidden_not_in_training)),
                "success_rate_when_hidden_optimum_not_in_training": _rate(
                    sum(bool(row.get("found_hidden_exact_optimum")) for row in hidden_not_in_training),
                    len(hidden_not_in_training),
                ),
                "num_runs_hidden_not_in_training_and_not_initial": int(
                    len(hidden_not_in_training_and_not_initial)
                ),
                "success_rate_when_hidden_optimum_not_in_training_and_not_initial": _rate(
                    sum(
                        bool(row.get("found_hidden_exact_optimum"))
                        for row in hidden_not_in_training_and_not_initial
                    ),
                    len(hidden_not_in_training_and_not_initial),
                ),
                "training_contains_hidden_optimum_rate": _rate(
                    sum(bool(row.get("training_contains_hidden_optimum")) for row in ok_rows),
                    len(ok_rows),
                ),
                "avg_algorithmic_ed_lp_calls": _average(ok_rows, "algorithmic_ed_lp_calls"),
                "avg_circuit_executions": _average(ok_rows, "circuit_executions"),
                "avg_total_shots": _average(ok_rows, "total_shots"),
                "avg_max_qubits": _average(ok_rows, "max_qubits"),
                "avg_max_circuit_depth": _average(ok_rows, "max_circuit_depth"),
                "avg_max_transpiled_depth": _average(ok_rows, "max_transpiled_depth"),
            }
        )
    return summary


def run_sweep(
    *,
    instance_path: Path,
    backend: str,
    shots: int,
    seed_start: int,
    seed_count: int,
    configs: list[tuple[int, ...]],
    train_sample_counts: list[int],
    lambda_growth: float,
    max_rounds: int,
    max_trials_per_threshold: int,
    max_candidates_per_shotbatch: int,
    exclude_hidden_optimum_from_training: bool,
    exclude_hidden_optimum_from_initial: bool,
    output_json: Path,
    output_csv: Path,
) -> dict[str, object]:
    runs = []
    output_json.parent.mkdir(parents=True, exist_ok=True)
    scratch_path = output_json.with_name(f"{output_json.stem}_latest_run.json")
    for config in configs:
        num_search_qubits = 2 * len(config)
        dimension = 2**num_search_qubits
        max_train_samples = dimension - 1 if exclude_hidden_optimum_from_training else dimension
        for train_sample_count in train_sample_counts:
            for seed in range(int(seed_start), int(seed_start) + int(seed_count)):
                base_row = _base_run_row(
                    config=config,
                    num_search_qubits=num_search_qubits,
                    dimension=dimension,
                    seed=seed,
                    train_sample_count=train_sample_count,
                    exclude_hidden_optimum_from_training=exclude_hidden_optimum_from_training,
                    exclude_hidden_optimum_from_initial=exclude_hidden_optimum_from_initial,
                )
                if train_sample_count > max_train_samples:
                    runs.append(
                        {
                            **base_row,
                            "status": "error",
                            "error": (
                                f"train_sample_count={train_sample_count} exceeds available "
                                f"training candidates ({max_train_samples})"
                            ),
                        }
                    )
                    continue
                try:
                    summary = run_small_sample_gas(
                        instance_path=instance_path,
                        results_path=scratch_path,
                        backend=backend,
                        shots=shots,
                        seed=seed,
                        lambda_growth=lambda_growth,
                        max_rounds=max_rounds,
                        max_trials_per_threshold=max_trials_per_threshold,
                        selected_generator_indices=config,
                        train_sample_count=train_sample_count,
                        initial_index="random",
                        num_pieces=2,
                        max_weight=7,
                        save_qasm=False,
                        draw_circuit=False,
                        max_candidates_per_shotbatch=max_candidates_per_shotbatch,
                        exclude_hidden_optimum_from_training=exclude_hidden_optimum_from_training,
                        exclude_hidden_optimum_from_initial=exclude_hidden_optimum_from_initial,
                    )
                    runs.append({**base_row, **_summary_to_run_row(summary), "status": "ok", "error": None})
                except Exception as exc:  # noqa: BLE001 - sweep should keep running after one failed config.
                    runs.append({**base_row, "status": "error", "error": str(exc)})
                finally:
                    scratch_path.unlink(missing_ok=True)

    grouped = build_grouped_summary(runs)
    payload = {
        "method": "small-sample gate-level max-affine GAS sweep",
        "notes": [
            "This is a small-sample gate-level max-affine GAS experiment.",
            "The hidden full subspace enumeration is used only for evaluation.",
            "Algorithmic ED/LP calls include only training samples and measured candidates.",
            "The experiment evaluates whether shot-based gate-level GAS can recover the hidden subspace optimum under limited ED/LP supervision.",
        ],
        "runs": runs,
        "grouped_summary": grouped,
    }
    write_strict_json(output_json, payload)
    _write_csv(output_csv, runs)
    return payload


def _base_run_row(
    *,
    config: tuple[int, ...],
    num_search_qubits: int,
    dimension: int,
    seed: int,
    train_sample_count: int,
    exclude_hidden_optimum_from_training: bool,
    exclude_hidden_optimum_from_initial: bool,
) -> dict[str, object]:
    return {
        "config": ",".join(str(index) for index in config),
        "selected_generators": [int(index) for index in config],
        "num_search_qubits": int(num_search_qubits),
        "dimension": int(dimension),
        "seed": int(seed),
        "train_sample_count": int(train_sample_count),
        "exclude_hidden_optimum_from_training": bool(exclude_hidden_optimum_from_training),
        "exclude_hidden_optimum_from_initial": bool(exclude_hidden_optimum_from_initial),
        "training_contains_hidden_optimum": None,
        "initial_matches_hidden_optimum": None,
        "hidden_best_bitstring": None,
        "hidden_best_true_cost": None,
        "initial_bitstring": None,
        "initial_true_cost": None,
        "final_bitstring": None,
        "final_true_cost": None,
        "found_hidden_exact_optimum": None,
        "algorithmic_ed_lp_calls": None,
        "hidden_reference_ed_lp_calls": None,
        "circuit_executions": None,
        "total_shots": None,
        "max_qubits": None,
        "max_circuit_depth": None,
        "max_transpiled_depth": None,
    }


def _summary_to_run_row(summary: dict[str, object]) -> dict[str, object]:
    return {
        "training_contains_hidden_optimum": bool(summary["training_contains_hidden_optimum"]),
        "initial_matches_hidden_optimum": bool(summary["initial_matches_hidden_optimum"]),
        "hidden_best_bitstring": summary["hidden_best_bitstring"],
        "hidden_best_true_cost": summary["hidden_best_true_cost"],
        "initial_bitstring": summary["initial_bitstring"],
        "initial_true_cost": summary["initial_true_cost"],
        "final_bitstring": summary["final_bitstring"],
        "final_true_cost": summary["final_true_cost"],
        "found_hidden_exact_optimum": bool(summary["found_hidden_exact_optimum"]),
        "algorithmic_ed_lp_calls": int(summary["algorithmic_ed_lp_calls"]),
        "hidden_reference_ed_lp_calls": int(summary["hidden_reference_ed_lp_calls"]),
        "circuit_executions": int(summary["circuit_executions"]),
        "total_shots": int(summary["total_shots"]),
        "max_qubits": int(summary["max_qubits"]),
        "max_circuit_depth": int(summary["max_circuit_depth"]),
        "max_transpiled_depth": int(summary["max_transpiled_depth"]),
    }


def _write_csv(path: Path, runs: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for row in runs:
            writer.writerow({field: _csv_value(row.get(field)) for field in RUN_FIELDS})


def _csv_value(value: object) -> object:
    value = sanitize_for_strict_json(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def _average(rows: list[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--backend", choices=("statevector", "qasm", "fake", "ibm"), default="qasm")
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--configs", type=parse_configs, required=True)
    parser.add_argument("--train-sample-counts", type=parse_train_sample_counts, required=True)
    parser.add_argument("--lambda-growth", type=float, default=8.0 / 7.0)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-trials-per-threshold", type=int, default=12)
    parser.add_argument("--max-candidates-per-shotbatch", type=int, default=1)
    parser.add_argument("--exclude-hidden-optimum-from-training", action="store_true")
    parser.add_argument("--exclude-hidden-optimum-from-initial", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/stage1_case14_t2_small_sample_gate_level_gas_sweep.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/stage1_case14_t2_small_sample_gate_level_gas_sweep.csv"),
    )
    args = parser.parse_args()

    payload = run_sweep(
        instance_path=args.instance,
        backend=args.backend,
        shots=args.shots,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        configs=args.configs,
        train_sample_counts=args.train_sample_counts,
        lambda_growth=args.lambda_growth,
        max_rounds=args.max_rounds,
        max_trials_per_threshold=args.max_trials_per_threshold,
        max_candidates_per_shotbatch=args.max_candidates_per_shotbatch,
        exclude_hidden_optimum_from_training=args.exclude_hidden_optimum_from_training,
        exclude_hidden_optimum_from_initial=args.exclude_hidden_optimum_from_initial,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    compact = {
        "num_runs": len(payload["runs"]),
        "num_errors": sum(1 for row in payload["runs"] if row["status"] != "ok"),
        "grouped_summary": payload["grouped_summary"],
    }
    print(json.dumps(sanitize_for_strict_json(compact), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
