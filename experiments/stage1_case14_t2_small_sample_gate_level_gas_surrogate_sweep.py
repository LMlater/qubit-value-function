from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from statistics import mean
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_t2_random_budget_baseline import hidden_reference_distribution  # noqa: E402
from experiments.stage1_case14_t2_small_sample_gate_level_gas_diagnostics import classify_run_failure  # noqa: E402
from experiments.stage1_case14_t2_small_sample_gate_level_gas_sweep import parse_configs, parse_train_sample_counts  # noqa: E402
from experiments.stage1_case14_t2_small_sample_gate_level_max_affine_gas import run as run_small_sample_gas  # noqa: E402
from qubit_value_function.diagnostics import random_baseline_probabilities  # noqa: E402
from qubit_value_function.experiment_utils import sanitize_for_strict_json, write_strict_json  # noqa: E402


SURROGATE_FIELDS = [
    "selected_generators",
    "num_search_qubits",
    "dimension",
    "seed",
    "train_sample_count",
    "backend",
    "shots",
    "measurement_policy",
    "max_candidates_per_shotbatch",
    "oracle_mode",
    "learner",
    "refit_policy",
    "training_contains_hidden_optimum",
    "initial_matches_hidden_optimum",
    "found_hidden_exact_optimum",
    "success_within_1_percent",
    "success_within_3_percent",
    "success_within_5_percent",
    "random_exact_success_probability",
    "random_success_within_3_percent_probability",
    "algorithmic_ed_lp_calls",
    "search_verification_calls",
    "hidden_reference_ed_lp_calls",
    "circuit_executions",
    "total_shots",
    "verified_candidates",
    "max_qubits",
    "max_transpiled_depth",
    "learner_fallback",
    "train_pairwise_order_accuracy",
    "train_best_rank",
    "train_best_bitstring",
    "train_best_true_cost",
    "num_pairs",
    "num_pairwise_violations",
    "pairwise_hinge_loss",
    "surrogate_piece_count",
    "surrogate_max_weight",
    "surrogate_total_weight",
    "refit_count",
    "observed_sample_count",
    "observed_indices",
    "status",
    "error_type",
    "error",
]


def parse_csv_strings(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def build_surrogate_grouped_summary(runs: Iterable[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    rows = list(runs)
    return {
        "by_learner": _group_rows(rows, key_fields=("learner",), display_fields=("learner",)),
        "by_refit_policy": _group_rows(rows, key_fields=("refit_policy",), display_fields=("refit_policy",)),
        "by_qubit_count": _group_rows(
            rows,
            key_fields=("num_search_qubits", "learner"),
            display_fields=("num_search_qubits", "learner"),
        ),
        "by_learner_refit": _group_rows(
            rows,
            key_fields=("learner", "refit_policy"),
            display_fields=("learner", "refit_policy"),
        ),
    }


def run_surrogate_sweep(
    *,
    instance_path: Path,
    backend: str,
    shots: int,
    seed_start: int,
    seed_count: int,
    configs: list[tuple[int, ...]],
    train_sample_counts: list[int],
    learners: list[str],
    refit_policies: list[str],
    max_candidates_per_shotbatch: int,
    max_rounds: int,
    max_trials_per_threshold: int,
    exclude_hidden_optimum_from_training: bool,
    exclude_hidden_optimum_from_initial: bool,
    output_json: Path,
    output_csv: Path,
    output_summary: Path,
) -> dict[str, object]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    distributions = {
        config: hidden_reference_distribution(
            instance_path=instance_path,
            selected_generator_indices=config,
        )
        for config in configs
    }
    scratch_path = output_json.with_name(f"{output_json.stem}_latest_run.json")

    for config in configs:
        distribution = distributions[config]
        for train_sample_count in train_sample_counts:
            for seed in range(int(seed_start), int(seed_start) + int(seed_count)):
                for learner in learners:
                    for refit_policy in refit_policies:
                        base = _base_run_row(
                            config=config,
                            distribution=distribution,
                            seed=seed,
                            train_sample_count=train_sample_count,
                            backend=backend,
                            shots=shots,
                            max_candidates_per_shotbatch=max_candidates_per_shotbatch,
                            learner=learner,
                            refit_policy=refit_policy,
                        )
                        skip_reason = _invalid_train_sample_reason(
                            num_search_qubits=int(distribution["num_search_qubits"]),
                            train_sample_count=train_sample_count,
                            dimension=int(distribution["dimension"]),
                            exclude_hidden_optimum_from_training=exclude_hidden_optimum_from_training,
                        )
                        if skip_reason is not None:
                            runs.append({**base, "status": "skipped_invalid_config", "error_type": "invalid_config", "error": skip_reason})
                            continue
                        try:
                            summary = run_small_sample_gas(
                                instance_path=instance_path,
                                results_path=scratch_path,
                                backend=backend,
                                shots=shots,
                                seed=seed,
                                lambda_growth=8.0 / 7.0,
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
                                measurement_policy="shot_batch",
                                learner=learner,
                                refit_policy=refit_policy,
                            )
                            search_calls = int(summary["algorithmic_ed_lp_calls"]) - int(summary["train_sample_count"])
                            baseline = random_baseline_probabilities(
                                values=distribution["values"],
                                hidden_best_true_cost=float(distribution["hidden_best_true_cost"]),
                                draws=search_calls,
                            )
                            runs.append(
                                {
                                    **base,
                                    **_summary_to_run_row(summary),
                                    **baseline,
                                    "search_verification_calls": int(search_calls),
                                    "status": "ok",
                                    "error_type": None,
                                    "error": None,
                                }
                            )
                        except Exception as exc:  # noqa: BLE001 - focused sweep should keep moving.
                            status, error_type = classify_run_failure(str(exc))
                            runs.append({**base, "status": status, "error_type": error_type, "error": str(exc)})
                        finally:
                            scratch_path.unlink(missing_ok=True)

    payload = {
        "method": "small-sample gate-level GAS surrogate-focused sweep",
        "fixed_settings": {
            "backend": backend,
            "shots": int(shots),
            "measurement_policy": "shot_batch",
            "max_candidates_per_shotbatch": int(max_candidates_per_shotbatch),
            "oracle_mode": "learned",
            "exclude_hidden_optimum_from_training": bool(exclude_hidden_optimum_from_training),
            "exclude_hidden_optimum_from_initial": bool(exclude_hidden_optimum_from_initial),
        },
        "runs": runs,
        "grouped_summary": build_surrogate_grouped_summary(runs),
    }
    write_strict_json(output_json, payload)
    _write_csv(output_csv, runs)
    output_summary.write_text(_markdown_summary(payload), encoding="utf-8")
    return payload


def _base_run_row(
    *,
    config: tuple[int, ...],
    distribution: dict[str, object],
    seed: int,
    train_sample_count: int,
    backend: str,
    shots: int,
    max_candidates_per_shotbatch: int,
    learner: str,
    refit_policy: str,
) -> dict[str, object]:
    return {
        "selected_generators": [int(index) for index in config],
        "num_search_qubits": int(distribution["num_search_qubits"]),
        "dimension": int(distribution["dimension"]),
        "seed": int(seed),
        "train_sample_count": int(train_sample_count),
        "backend": backend,
        "shots": int(shots),
        "measurement_policy": "shot_batch",
        "max_candidates_per_shotbatch": int(max_candidates_per_shotbatch),
        "oracle_mode": "learned",
        "learner": learner,
        "refit_policy": refit_policy,
        "training_contains_hidden_optimum": None,
        "initial_matches_hidden_optimum": None,
    }


def _summary_to_run_row(summary: dict[str, object]) -> dict[str, object]:
    diagnostics = summary["learned_max_affine_oracle"]["training_diagnostics"]
    adaptive = summary["adaptive_search"]
    return {
        "training_contains_hidden_optimum": bool(summary["training_contains_hidden_optimum"]),
        "initial_matches_hidden_optimum": bool(summary["initial_matches_hidden_optimum"]),
        "found_hidden_exact_optimum": bool(summary["found_hidden_exact_optimum"]),
        "success_within_1_percent": bool(summary["success_within_1_percent"]),
        "success_within_3_percent": bool(summary["success_within_3_percent"]),
        "success_within_5_percent": bool(summary["success_within_5_percent"]),
        "algorithmic_ed_lp_calls": int(summary["algorithmic_ed_lp_calls"]),
        "hidden_reference_ed_lp_calls": int(summary["hidden_reference_ed_lp_calls"]),
        "circuit_executions": int(summary["circuit_executions"]),
        "total_shots": int(summary["total_shots"]),
        "verified_candidates": int(summary["verified_candidates"]),
        "max_qubits": int(summary["max_qubits"]),
        "max_transpiled_depth": int(summary["max_transpiled_depth"]),
        "learner_fallback": diagnostics.get("learner_fallback"),
        "train_pairwise_order_accuracy": diagnostics.get("train_pairwise_order_accuracy"),
        "train_best_rank": diagnostics.get("train_best_rank"),
        "train_best_bitstring": diagnostics.get("train_best_bitstring"),
        "train_best_true_cost": diagnostics.get("train_best_true_cost"),
        "train_predicted_values": diagnostics.get("train_predicted_values"),
        "train_true_values": diagnostics.get("train_true_values"),
        "num_pairs": diagnostics.get("num_pairs"),
        "num_pairwise_violations": diagnostics.get("num_pairwise_violations"),
        "pairwise_hinge_loss": diagnostics.get("pairwise_hinge_loss"),
        "surrogate_piece_count": diagnostics.get("surrogate_piece_count"),
        "surrogate_max_weight": diagnostics.get("surrogate_max_weight"),
        "surrogate_total_weight": diagnostics.get("surrogate_total_weight"),
        "refit_count": int(adaptive["refit_count"]),
        "observed_sample_count": int(adaptive["observed_sample_count"]),
        "observed_indices": adaptive["observed_indices"],
    }


def _invalid_train_sample_reason(
    *,
    num_search_qubits: int,
    train_sample_count: int,
    dimension: int,
    exclude_hidden_optimum_from_training: bool,
) -> str | None:
    if num_search_qubits == 4 and int(train_sample_count) not in {4, 8}:
        return "4q configs use train_sample_count 4 or 8"
    if num_search_qubits == 6 and int(train_sample_count) not in {8, 16}:
        return "6q configs use train_sample_count 8 or 16"
    max_train_samples = int(dimension) - (1 if exclude_hidden_optimum_from_training else 0)
    if int(train_sample_count) > max_train_samples:
        return f"train_sample_count={train_sample_count} exceeds available training candidates ({max_train_samples})"
    return None


def _group_rows(
    rows: list[dict[str, object]],
    *,
    key_fields: tuple[str, ...],
    display_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        groups.setdefault(key, []).append(row)

    out = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        ok = [row for row in group if row.get("status") == "ok"]
        skipped_resource = [row for row in group if row.get("status") == "skipped_resource_limit"]
        skipped_invalid = [row for row in group if row.get("status") == "skipped_invalid_config"]
        row = {
            field: key[display_fields.index(field)] if field in key_fields and field in display_fields else None
            for field in display_fields
        }
        for field in ("learner", "refit_policy", "num_search_qubits"):
            if field not in row:
                row[field] = None
        row.update(
            {
                "num_runs": int(len(group)),
                "num_ok_runs": int(len(ok)),
                "num_skipped_resource_limit_runs": int(len(skipped_resource)),
                "num_skipped_invalid_config_runs": int(len(skipped_invalid)),
                "num_error_runs": int(len(group) - len(ok) - len(skipped_resource) - len(skipped_invalid)),
                "exact_success_rate": _rate(sum(bool(item.get("found_hidden_exact_optimum")) for item in ok), len(ok)),
                "within_1_percent_success_rate": _rate(sum(bool(item.get("success_within_1_percent")) for item in ok), len(ok)),
                "within_3_percent_success_rate": _rate(sum(bool(item.get("success_within_3_percent")) for item in ok), len(ok)),
                "within_5_percent_success_rate": _rate(sum(bool(item.get("success_within_5_percent")) for item in ok), len(ok)),
                "avg_pairwise_order_accuracy": _avg(ok, "train_pairwise_order_accuracy"),
                "avg_pairwise_hinge_loss": _avg(ok, "pairwise_hinge_loss"),
                "avg_algorithmic_ed_lp_calls": _avg(ok, "algorithmic_ed_lp_calls"),
                "avg_total_shots": _avg(ok, "total_shots"),
                "avg_max_transpiled_depth": _avg(ok, "max_transpiled_depth"),
                "avg_refit_count": _avg(ok, "refit_count"),
                "avg_observed_sample_count": _avg(ok, "observed_sample_count"),
                "avg_random_exact_success_probability": _avg(ok, "random_exact_success_probability"),
                "fallback_rate": _rate(sum(item.get("learner_fallback") is not None for item in ok), len(ok)),
            }
        )
        out.append(row)
    return out


def _markdown_summary(payload: dict[str, object]) -> str:
    runs = payload["runs"]
    settings = payload["fixed_settings"]
    ok = [row for row in runs if row.get("status") == "ok"]
    skipped = [row for row in runs if str(row.get("status", "")).startswith("skipped_")]
    skipped_invalid = [row for row in runs if row.get("status") == "skipped_invalid_config"]
    skipped_resource = [row for row in runs if row.get("status") == "skipped_resource_limit"]
    errors = [row for row in runs if row.get("status") not in {"ok", "skipped_resource_limit", "skipped_invalid_config"}]
    grouped = payload["grouped_summary"]
    exact = _rate(sum(bool(row.get("found_hidden_exact_optimum")) for row in ok), len(ok))
    within1 = _rate(sum(bool(row.get("success_within_1_percent")) for row in ok), len(ok))
    within3 = _rate(sum(bool(row.get("success_within_3_percent")) for row in ok), len(ok))
    within5 = _rate(sum(bool(row.get("success_within_5_percent")) for row in ok), len(ok))
    rank_hinge = _first_group(grouped["by_learner"], "learner", "rank_hinge")
    mismatch = _first_group(grouped["by_learner"], "learner", "mismatch")
    rank_hinge_exact = float((rank_hinge or {}).get("exact_success_rate") or 0.0)
    mismatch_exact = float((mismatch or {}).get("exact_success_rate") or 0.0)
    rank_hinge_gain = rank_hinge_exact - mismatch_exact
    rank_hinge_6q = _first_group(
        [row for row in grouped["by_qubit_count"] if row.get("num_search_qubits") == 6],
        "learner",
        "rank_hinge",
    )
    mismatch_6q = _first_group(
        [row for row in grouped["by_qubit_count"] if row.get("num_search_qubits") == 6],
        "learner",
        "mismatch",
    )
    rank_hinge_6q_gain = float((rank_hinge_6q or {}).get("exact_success_rate") or 0.0) - float(
        (mismatch_6q or {}).get("exact_success_rate") or 0.0
    )
    refit_gain = _best_refit_gain(grouped["by_learner_refit"])
    improved = rank_hinge_gain >= 0.10 or rank_hinge_6q_gain >= 0.10 or refit_gain >= 0.10
    conclusion = (
        "The result supports that learned surrogate quality is a key performance bottleneck."
        if improved
        else "The result suggests that simply improving pairwise ranking within the current integer max-affine surrogate class is insufficient; a richer surrogate class may be needed."
    )
    hidden_exclusion_ok = all(
        row.get("training_contains_hidden_optimum") is False and row.get("initial_matches_hidden_optimum") is False
        for row in ok
    )
    return f"""# GAS Surrogate-Focused Sweep Summary

## Purpose

Prior diagnostics suggest learned surrogate quality is the main bottleneck. This sweep fixes shot-batch sampling and top-k candidate budget, then compares surrogate learners and accepted refit.

## Fixed Settings

- backend: {settings["backend"]}
- shots: {settings["shots"]}
- measurement_policy: {settings["measurement_policy"]}
- max_candidates_per_shotbatch: {settings["max_candidates_per_shotbatch"]}
- oracle_mode: {settings["oracle_mode"]}
- exclude_hidden_optimum_from_training: {settings["exclude_hidden_optimum_from_training"]}
- exclude_hidden_optimum_from_initial: {settings["exclude_hidden_optimum_from_initial"]}

## Overall Results

- total runs: {len(runs)}
- ok runs: {len(ok)}
- skipped runs: {len(skipped)}
- skipped invalid config runs: {len(skipped_invalid)}
- skipped resource limit runs: {len(skipped_resource)}
- error runs: {len(errors)}
- exact success: {_pct(exact)}
- within 1 percent success: {_pct(within1)}
- within 3 percent success: {_pct(within3)}
- within 5 percent success: {_pct(within5)}
- avg algorithmic ED/LP calls: {_fmt(_avg(ok, "algorithmic_ed_lp_calls"))}
- avg total shots: {_fmt(_avg(ok, "total_shots"))}
- avg circuit executions: {_fmt(_avg(ok, "circuit_executions"))}
- avg max qubits: {_fmt(_avg(ok, "max_qubits"))}
- avg max transpiled depth: {_fmt(_avg(ok, "max_transpiled_depth"))}
- all learned ok runs exclude hidden optimum from training and initial: {hidden_exclusion_ok}

## Learner Comparison

| learner | num_ok_runs | exact_success | within_3_percent_success | avg_pairwise_order_accuracy | avg_pairwise_hinge_loss | avg_algorithmic_ed_lp_calls | avg_total_shots | avg_max_transpiled_depth | fallback_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{_learner_rows(grouped["by_learner"])}

## Refit Comparison

| refit_policy | num_ok_runs | exact_success | within_3_percent_success | avg_refit_count | avg_observed_sample_count | avg_algorithmic_ed_lp_calls |
|---|---:|---:|---:|---:|---:|---:|
{_refit_rows(grouped["by_refit_policy"])}

## Learner × Refit

| learner | refit_policy | num_ok_runs | exact_success | within_3_percent_success | avg_algorithmic_ed_lp_calls |
|---|---|---:|---:|---:|---:|
{_learner_refit_rows(grouped["by_learner_refit"])}

## 4q vs 6q

| qubits | learner | num_ok_runs | exact_success | within_3_percent_success | avg_pairwise_order_accuracy |
|---:|---|---:|---:|---:|---:|
{_qubit_rows(grouped["by_qubit_count"])}

## Same-Budget Random Baseline

Same-budget random baseline is retained as a diagnostic comparison only.

- observed exact success: {_pct(exact)}
- average random exact probability: {_fmt(_avg(ok, "random_exact_success_probability"))}

## Formal-Sweep Decision

- rank_hinge overall exact gain vs mismatch: {_pct(rank_hinge_gain)}
- rank_hinge 6q exact gain vs mismatch: {_pct(rank_hinge_6q_gain)}
- best accepted-refit exact gain vs none with the same learner: {_pct(refit_gain)}
- {"A 20-seed focused formal sweep is justified by the configured 10 percentage-point threshold." if improved else "The smoke results did not show a sufficiently clear improvement to justify a 20-seed focused formal sweep."}

## Cautious Conclusion

{conclusion}

This is selected-generator subspace optimum recovery, not full 12-bit case14 T=2 global optimization.
This does not establish quantum advantage.
"""


def _learner_rows(rows: list[dict[str, object]]) -> str:
    return "\n".join(
        "| "
        + " | ".join(
            [
                str(row["learner"]),
                str(row["num_ok_runs"]),
                _pct(row["exact_success_rate"]),
                _pct(row["within_3_percent_success_rate"]),
                _fmt(row["avg_pairwise_order_accuracy"]),
                _fmt(row["avg_pairwise_hinge_loss"]),
                _fmt(row["avg_algorithmic_ed_lp_calls"]),
                _fmt(row["avg_total_shots"]),
                _fmt(row["avg_max_transpiled_depth"]),
                _pct(row["fallback_rate"]),
            ]
        )
        + " |"
        for row in rows
    )


def _refit_rows(rows: list[dict[str, object]]) -> str:
    return "\n".join(
        "| "
        + " | ".join(
            [
                str(row["refit_policy"]),
                str(row["num_ok_runs"]),
                _pct(row["exact_success_rate"]),
                _pct(row["within_3_percent_success_rate"]),
                _fmt(row["avg_refit_count"]),
                _fmt(row["avg_observed_sample_count"]),
                _fmt(row["avg_algorithmic_ed_lp_calls"]),
            ]
        )
        + " |"
        for row in rows
    )


def _learner_refit_rows(rows: list[dict[str, object]]) -> str:
    return "\n".join(
        "| "
        + " | ".join(
            [
                str(row["learner"]),
                str(row["refit_policy"]),
                str(row["num_ok_runs"]),
                _pct(row["exact_success_rate"]),
                _pct(row["within_3_percent_success_rate"]),
                _fmt(row["avg_algorithmic_ed_lp_calls"]),
            ]
        )
        + " |"
        for row in rows
    )


def _qubit_rows(rows: list[dict[str, object]]) -> str:
    return "\n".join(
        "| "
        + " | ".join(
            [
                str(row["num_search_qubits"]),
                str(row["learner"]),
                str(row["num_ok_runs"]),
                _pct(row["exact_success_rate"]),
                _pct(row["within_3_percent_success_rate"]),
                _fmt(row["avg_pairwise_order_accuracy"]),
            ]
        )
        + " |"
        for row in rows
    )


def _first_group(rows: list[dict[str, object]], field: str, value: object) -> dict[str, object] | None:
    for row in rows:
        if row.get(field) == value:
            return row
    return None


def _best_refit_gain(rows: list[dict[str, object]]) -> float:
    exact_by_key = {
        (str(row.get("learner")), str(row.get("refit_policy"))): float(row.get("exact_success_rate") or 0.0)
        for row in rows
    }
    gains = [
        exact_by_key[(learner, "accepted")] - exact_by_key[(learner, "none")]
        for learner, policy in exact_by_key
        if policy == "accepted" and (learner, "none") in exact_by_key
    ]
    return max(gains, default=0.0)


def _write_csv(path: Path, runs: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SURROGATE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in runs:
            writer.writerow({field: _csv_value(row.get(field)) for field in SURROGATE_FIELDS})


def _csv_value(value: object) -> object:
    value = sanitize_for_strict_json(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def _avg(rows: list[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return None
    return float(mean(values))


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--backend", choices=("statevector", "qasm", "fake", "ibm"), default="qasm")
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--configs", type=parse_configs, required=True)
    parser.add_argument("--train-sample-counts", type=parse_train_sample_counts, required=True)
    parser.add_argument("--learners", type=parse_csv_strings, default=["mismatch", "pairwise_ranking", "rank_hinge"])
    parser.add_argument("--refit-policies", type=parse_csv_strings, default=["none", "accepted"])
    parser.add_argument("--max-candidates-per-shotbatch", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-trials-per-threshold", type=int, default=12)
    parser.add_argument("--exclude-hidden-optimum-from-training", action="store_true")
    parser.add_argument("--exclude-hidden-optimum-from-initial", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/stage1_case14_t2_small_sample_gate_level_gas_surrogate_smoke.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/stage1_case14_t2_small_sample_gate_level_gas_surrogate_smoke.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/stage1_case14_t2_small_sample_gate_level_gas_surrogate_smoke_summary.md"),
    )
    args = parser.parse_args()

    payload = run_surrogate_sweep(
        instance_path=args.instance,
        backend=args.backend,
        shots=args.shots,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        configs=args.configs,
        train_sample_counts=args.train_sample_counts,
        learners=args.learners,
        refit_policies=args.refit_policies,
        max_candidates_per_shotbatch=args.max_candidates_per_shotbatch,
        max_rounds=args.max_rounds,
        max_trials_per_threshold=args.max_trials_per_threshold,
        exclude_hidden_optimum_from_training=args.exclude_hidden_optimum_from_training,
        exclude_hidden_optimum_from_initial=args.exclude_hidden_optimum_from_initial,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_summary=args.output_summary,
    )
    compact = {
        "num_runs": len(payload["runs"]),
        "num_ok_runs": sum(1 for row in payload["runs"] if row.get("status") == "ok"),
        "num_skipped_runs": sum(1 for row in payload["runs"] if str(row.get("status", "")).startswith("skipped_")),
        "num_error_runs": sum(
            1
            for row in payload["runs"]
            if row.get("status") not in {"ok", "skipped_resource_limit", "skipped_invalid_config"}
        ),
        "grouped_summary": payload["grouped_summary"],
    }
    print(json.dumps(sanitize_for_strict_json(compact), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
