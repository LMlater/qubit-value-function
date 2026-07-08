from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_t2_random_budget_baseline import hidden_reference_distribution  # noqa: E402
from experiments.stage1_case14_t2_small_sample_gate_level_gas_sweep import parse_configs, parse_train_sample_counts  # noqa: E402
from experiments.stage1_case14_t2_small_sample_gate_level_max_affine_gas import run as run_small_sample_gas  # noqa: E402
from qubit_value_function.diagnostics import gap_metrics, random_baseline_probabilities  # noqa: E402
from qubit_value_function.experiment_utils import sanitize_for_strict_json, write_strict_json  # noqa: E402


DIAGNOSTIC_FIELDS = [
    "selected_generators",
    "num_search_qubits",
    "dimension",
    "seed",
    "train_sample_count",
    "measurement_policy",
    "shots",
    "shots_per_circuit",
    "max_candidates_per_shotbatch",
    "refit_policy",
    "learner",
    "oracle_mode",
    "diagnostic_only",
    "training_contains_hidden_optimum",
    "initial_matches_hidden_optimum",
    "found_hidden_exact_optimum",
    "success_within_1_percent",
    "success_within_3_percent",
    "success_within_5_percent",
    "absolute_gap_to_hidden_best",
    "relative_gap_to_hidden_best",
    "random_exact_success_probability",
    "random_success_within_1_percent_probability",
    "random_success_within_3_percent_probability",
    "random_success_within_5_percent_probability",
    "algorithmic_ed_lp_calls",
    "search_verification_calls",
    "hidden_reference_ed_lp_calls",
    "circuit_executions",
    "total_shots",
    "verified_candidates",
    "max_qubits",
    "max_transpiled_depth",
    "status",
    "error",
]


def hidden_perfect_diagnostic_run(
    *,
    selected_generator_indices: tuple[int, ...],
    train_sample_count: int,
    seed: int,
    search_verification_budget: int,
    instance_path: Path = Path("data/case14.json.gz"),
) -> dict[str, object]:
    distribution = hidden_reference_distribution(
        instance_path=instance_path,
        selected_generator_indices=selected_generator_indices,
    )
    hidden_index = int(distribution["hidden_best_index"])
    hidden_cost = float(distribution["hidden_best_true_cost"])
    rng = np.random.default_rng(seed)
    dimension = int(distribution["dimension"])
    train_pool = np.asarray([index for index in range(dimension) if index != hidden_index], dtype=int)
    train_indices = rng.choice(train_pool, size=min(int(train_sample_count), train_pool.size), replace=False)
    initial_pool = np.asarray([index for index in range(dimension) if index != hidden_index], dtype=int)
    initial_index = int(rng.choice(initial_pool))
    budget = max(0, int(search_verification_budget))
    found = bool(budget > 0)
    final_cost = hidden_cost if found else float(distribution["values"][initial_index])
    gaps = gap_metrics(final_cost, hidden_cost)
    return {
        "selected_generators": [int(index) for index in selected_generator_indices],
        "num_search_qubits": int(distribution["num_search_qubits"]),
        "dimension": dimension,
        "seed": int(seed),
        "train_sample_count": int(train_sample_count),
        "measurement_policy": "classical_hidden_perfect_diagnostic",
        "shots": 0,
        "shots_per_circuit": 0,
        "max_candidates_per_shotbatch": 0,
        "refit_policy": "none",
        "learner": "hidden_reference",
        "oracle_mode": "hidden_perfect_diagnostic",
        "diagnostic_only": True,
        "training_contains_hidden_optimum": bool(hidden_index in {int(index) for index in train_indices}),
        "initial_matches_hidden_optimum": bool(initial_index == hidden_index),
        "found_hidden_exact_optimum": found,
        **gaps,
        "algorithmic_ed_lp_calls": int(len(train_indices) + min(budget, dimension)),
        "search_verification_calls": int(min(budget, dimension)),
        "hidden_reference_ed_lp_calls": dimension,
        "circuit_executions": 0,
        "total_shots": 0,
        "verified_candidates": int(min(budget, dimension)),
        "max_qubits": 0,
        "max_transpiled_depth": 0,
        "status": "ok",
        "error": None,
    }


def run_diagnostics(
    *,
    instance_path: Path,
    backend: str,
    configs: list[tuple[int, ...]],
    seed_start: int,
    seed_count: int,
    train_sample_counts: list[int],
    measurement_policies: list[str],
    shots_values: list[int],
    max_candidates_values: list[int],
    refit_policies: list[str],
    learners: list[str],
    oracle_modes: list[str],
    max_rounds: int,
    max_trials_per_threshold: int,
    exclude_hidden_optimum_from_training: bool,
    exclude_hidden_optimum_from_initial: bool,
    output_json: Path,
    output_csv: Path,
    output_summary: Path,
) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    distributions = {
        config: hidden_reference_distribution(
            instance_path=instance_path,
            selected_generator_indices=config,
        )
        for config in configs
    }
    for config in configs:
        num_bits = len(config) * 2
        for train_sample_count in train_sample_counts:
            if train_sample_count >= 2**num_bits:
                continue
            if num_bits == 4 and train_sample_count not in {4, 8}:
                continue
            if num_bits == 6 and train_sample_count not in {8, 16}:
                continue
            for seed in range(seed_start, seed_start + seed_count):
                for oracle_mode in oracle_modes:
                    if oracle_mode == "hidden_perfect_diagnostic":
                        runs.append(
                            hidden_perfect_diagnostic_run(
                                selected_generator_indices=config,
                                train_sample_count=train_sample_count,
                                seed=seed,
                                search_verification_budget=max_trials_per_threshold,
                                instance_path=instance_path,
                            )
                        )
                        continue
                    for measurement_policy in measurement_policies:
                        candidate_values = [1] if measurement_policy == "single_shot" else max_candidates_values
                        shot_values = [1] if measurement_policy == "single_shot" else shots_values
                        for shots in shot_values:
                            for max_candidates in candidate_values:
                                for refit_policy in refit_policies:
                                    for learner in learners:
                                        runs.append(
                                            _run_learned_diagnostic(
                                                instance_path=instance_path,
                                                backend=backend,
                                                config=config,
                                                train_sample_count=train_sample_count,
                                                seed=seed,
                                                measurement_policy=measurement_policy,
                                                shots=shots,
                                                max_candidates=max_candidates,
                                                refit_policy=refit_policy,
                                                learner=learner,
                                                max_rounds=max_rounds,
                                                max_trials_per_threshold=max_trials_per_threshold,
                                                exclude_hidden_optimum_from_training=exclude_hidden_optimum_from_training,
                                                exclude_hidden_optimum_from_initial=exclude_hidden_optimum_from_initial,
                                                distribution=distributions[config],
                                            )
                                        )

    payload = {
        "method": "small-sample gate-level GAS diagnostic ablations",
        "notes": [
            "Diagnostic smoke experiment for explaining formal exclude-hidden exact success rates.",
            "Random baselines and hidden_perfect_diagnostic use hidden reference only for diagnostic comparison.",
            "hidden_perfect_diagnostic is not an algorithmic result.",
            "The experiment remains selected-generator subspace optimum recovery, not full 12-bit case14 optimization.",
        ],
        "runs": runs,
        "grouped_summary": _group_diagnostics(runs),
    }
    write_strict_json(output_json, payload)
    _write_csv(output_csv, runs)
    output_summary.write_text(_markdown_summary(payload), encoding="utf-8")
    return payload


def _run_learned_diagnostic(
    *,
    instance_path: Path,
    backend: str,
    config: tuple[int, ...],
    train_sample_count: int,
    seed: int,
    measurement_policy: str,
    shots: int,
    max_candidates: int,
    refit_policy: str,
    learner: str,
    max_rounds: int,
    max_trials_per_threshold: int,
    exclude_hidden_optimum_from_training: bool,
    exclude_hidden_optimum_from_initial: bool,
    distribution: dict[str, object],
) -> dict[str, object]:
    scratch = Path("results") / "stage1_case14_t2_small_sample_gate_level_gas_diagnostics_latest_run.json"
    try:
        summary = run_small_sample_gas(
            instance_path=instance_path,
            results_path=scratch,
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
            max_candidates_per_shotbatch=max_candidates,
            exclude_hidden_optimum_from_training=exclude_hidden_optimum_from_training,
            exclude_hidden_optimum_from_initial=exclude_hidden_optimum_from_initial,
            measurement_policy=measurement_policy,
            learner=learner,
            refit_policy=refit_policy,
        )
        search_calls = int(summary["algorithmic_ed_lp_calls"]) - int(summary["train_sample_count"])
        baseline = random_baseline_probabilities(
            values=distribution["values"],
            hidden_best_true_cost=float(distribution["hidden_best_true_cost"]),
            draws=search_calls,
        )
        return {
            "selected_generators": summary["selected_generators"],
            "num_search_qubits": summary["num_search_qubits"],
            "dimension": int(distribution["dimension"]),
            "seed": int(seed),
            "train_sample_count": int(train_sample_count),
            "measurement_policy": measurement_policy,
            "shots": int(shots),
            "shots_per_circuit": int(summary["shots_per_circuit"]),
            "max_candidates_per_shotbatch": int(max_candidates),
            "refit_policy": refit_policy,
            "learner": learner,
            "oracle_mode": "learned",
            "diagnostic_only": False,
            "training_contains_hidden_optimum": summary["training_contains_hidden_optimum"],
            "initial_matches_hidden_optimum": summary["initial_matches_hidden_optimum"],
            "found_hidden_exact_optimum": summary["found_hidden_exact_optimum"],
            "success_within_1_percent": summary["success_within_1_percent"],
            "success_within_3_percent": summary["success_within_3_percent"],
            "success_within_5_percent": summary["success_within_5_percent"],
            "absolute_gap_to_hidden_best": summary["absolute_gap_to_hidden_best"],
            "relative_gap_to_hidden_best": summary["relative_gap_to_hidden_best"],
            **baseline,
            "algorithmic_ed_lp_calls": summary["algorithmic_ed_lp_calls"],
            "search_verification_calls": search_calls,
            "hidden_reference_ed_lp_calls": summary["hidden_reference_ed_lp_calls"],
            "circuit_executions": summary["circuit_executions"],
            "total_shots": summary["total_shots"],
            "verified_candidates": summary["verified_candidates"],
            "max_qubits": summary["max_qubits"],
            "max_transpiled_depth": summary["max_transpiled_depth"],
            "status": "ok",
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic sweep should keep moving.
        return {
            "selected_generators": [int(index) for index in config],
            "num_search_qubits": len(config) * 2,
            "dimension": int(distribution["dimension"]),
            "seed": int(seed),
            "train_sample_count": int(train_sample_count),
            "measurement_policy": measurement_policy,
            "shots": int(shots),
            "shots_per_circuit": 1 if measurement_policy == "single_shot" else int(shots),
            "max_candidates_per_shotbatch": int(max_candidates),
            "refit_policy": refit_policy,
            "learner": learner,
            "oracle_mode": "learned",
            "diagnostic_only": False,
            "status": "error",
            "error": str(exc),
        }
    finally:
        scratch.unlink(missing_ok=True)


def _group_diagnostics(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in runs:
        key = (
            tuple(row["selected_generators"]),
            row["num_search_qubits"],
            row["train_sample_count"],
            row["measurement_policy"],
            row["max_candidates_per_shotbatch"],
            row["refit_policy"],
            row["learner"],
            row["oracle_mode"],
        )
        groups.setdefault(key, []).append(row)
    out = []
    for key, rows in sorted(groups.items(), key=lambda item: item[0]):
        ok = [row for row in rows if row.get("status") == "ok"]
        out.append(
            {
                "selected_generators": list(key[0]),
                "num_search_qubits": key[1],
                "train_sample_count": key[2],
                "measurement_policy": key[3],
                "max_candidates_per_shotbatch": key[4],
                "refit_policy": key[5],
                "learner": key[6],
                "oracle_mode": key[7],
                "num_runs": len(rows),
                "num_ok_runs": len(ok),
                "num_error_runs": len(rows) - len(ok),
                "exact_success_rate": _rate(sum(bool(row.get("found_hidden_exact_optimum")) for row in ok), len(ok)),
                "within_1_percent_success_rate": _rate(sum(bool(row.get("success_within_1_percent")) for row in ok), len(ok)),
                "within_3_percent_success_rate": _rate(sum(bool(row.get("success_within_3_percent")) for row in ok), len(ok)),
                "within_5_percent_success_rate": _rate(sum(bool(row.get("success_within_5_percent")) for row in ok), len(ok)),
                "avg_random_exact_success_probability": _avg(ok, "random_exact_success_probability"),
                "avg_random_within_1_percent_probability": _avg(ok, "random_success_within_1_percent_probability"),
                "avg_random_within_3_percent_probability": _avg(ok, "random_success_within_3_percent_probability"),
                "avg_random_within_5_percent_probability": _avg(ok, "random_success_within_5_percent_probability"),
                "avg_algorithmic_ed_lp_calls": _avg(ok, "algorithmic_ed_lp_calls"),
                "avg_search_verification_calls": _avg(ok, "search_verification_calls"),
                "avg_circuit_executions": _avg(ok, "circuit_executions"),
                "avg_total_shots": _avg(ok, "total_shots"),
                "avg_verified_candidates": _avg(ok, "verified_candidates"),
                "avg_max_transpiled_depth": _avg(ok, "max_transpiled_depth"),
            }
        )
    return out


def _markdown_summary(payload: dict[str, object]) -> str:
    runs = payload["runs"]
    ok = [row for row in runs if row.get("status") == "ok"]
    learned = [row for row in ok if row.get("oracle_mode") == "learned"]
    errors = [row for row in runs if row.get("status") != "ok"]
    exact = _rate(sum(bool(row.get("found_hidden_exact_optimum")) for row in learned), len(learned))
    within1 = _rate(sum(bool(row.get("success_within_1_percent")) for row in learned), len(learned))
    within3 = _rate(sum(bool(row.get("success_within_3_percent")) for row in learned), len(learned))
    within5 = _rate(sum(bool(row.get("success_within_5_percent")) for row in learned), len(learned))
    rows = _grouped_summary_rows(payload["grouped_summary"])
    random_rows = _random_baseline_rows(learned)
    measurement_rows = _aggregate_markdown_rows(
        learned,
        key_fields=("measurement_policy", "shots_per_circuit"),
        display_fields=("measurement_policy", "shots_per_circuit", "avg_total_shots", "avg_circuit_executions"),
    )
    candidate_rows = _aggregate_markdown_rows(
        learned,
        key_fields=("max_candidates_per_shotbatch",),
        display_fields=("max_candidates_per_shotbatch", "avg_verified_candidates"),
    )
    refit_rows = _aggregate_markdown_rows(
        learned,
        key_fields=("refit_policy",),
        display_fields=("refit_policy",),
    )
    learner_rows = _aggregate_markdown_rows(
        learned,
        key_fields=("learner",),
        display_fields=("learner",),
    )
    perfect_rows = _aggregate_markdown_rows(
        [row for row in ok if row.get("oracle_mode") == "hidden_perfect_diagnostic"],
        key_fields=("oracle_mode",),
        display_fields=("oracle_mode",),
    )
    error_note = ""
    if errors:
        error_note = "\n\nResource-limit/error runs retained in JSON/CSV: " + "; ".join(
            sorted({str(row.get("error")) for row in errors})
        )
    return f"""# GAS Diagnostic Ablations Smoke Summary

## Purpose

This smoke diagnostic investigates why the formal exclude-hidden sweep has modest exact optimum recovery. It compares exact success, near-optimal success, same-budget random baselines, measurement policies, candidate budgets, refit policies, learners, and a diagnostic-only hidden-perfect oracle mode.

## Formal Result Context

- formal total runs: 640
- exact success over ok runs: 35.5%
- 4q grouped exact-success range: 35%-60%
- 6q grouped exact-success range: 0%-25%

## Smoke Overall

- total runs: {len(runs)}
- ok runs: {len(ok)}
- error runs: {len(runs) - len(ok)}
- learned exact success: {_pct(exact)}
- learned within 1% success: {_pct(within1)}
- learned within 3% success: {_pct(within3)}
- learned within 5% success: {_pct(within5)}
{error_note}

## Grouped Diagnostic Table

| selected_generators | qubits | train_n | measurement | top_k | refit | learner | oracle_mode | exact | within_3pct | random_exact | avg_ed_lp |
|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|
{rows}

## Random Baseline Comparison

Same-budget random baselines are diagnostic comparisons using hidden reference distributions, not algorithmic training signals.

| selected_generators | qubits | train_n | avg search budget | observed exact | random exact | observed 1pct | random 1pct | observed 3pct | random 3pct | observed 5pct | random 5pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{random_rows}

## Measurement Policy Comparison

Shot-batch sampling repeats circuit sampling and chooses top measured candidates; single-shot records one bitstring per circuit execution.

| measurement | shots_per_circuit | avg total shots | avg circuit executions | runs | exact | within_3pct | avg ED/LP |
|---|---:|---:|---:|---:|---:|---:|---:|
{measurement_rows}

## Candidate Budget Comparison

Larger top-k budgets can improve recovery, but they spend more search verification ED/LP calls.

| top_k | avg verified candidates | runs | exact | within_3pct | avg ED/LP |
|---:|---:|---:|---:|---:|---:|
{candidate_rows}

## Refit Comparison

The accepted refit policy updates the surrogate only from training samples plus measured and verified candidates.

| refit_policy | runs | exact | within_3pct | avg ED/LP |
|---|---:|---:|---:|---:|
{refit_rows}

## Learner Comparison

The pairwise_ranking learner is an ablation against the existing mismatch learner; neither uses hidden enumeration for training.

| learner | runs | exact | within_3pct | avg ED/LP |
|---|---:|---:|---:|---:|
{learner_rows}

## Perfect Oracle Diagnostic

hidden_perfect_diagnostic is not an algorithmic result. It uses hidden reference only to separate GAS/sampling limitations from learned-surrogate limitations.

| oracle_mode | runs | exact | within_3pct | avg ED/LP |
|---|---:|---:|---:|---:|
{perfect_rows}

## Cautious Conclusions

- The current prototype should be read as selected-generator subspace optimum recovery, not full 12-bit case14 T=2 global optimization.
- Non-random enrichment in small 4q subspaces is visible when observed success exceeds same-budget random probabilities, but this does not establish quantum advantage.
- Shot-batch sampling can improve candidate stability, and total_shots must be reported as part of the resource budget.
- If hidden_perfect_diagnostic succeeds while learned runs fail, the likely bottleneck is the learned affine surrogate; if it also fails, the GAS schedule, sampling, or candidate selection is implicated.

## Notes

- Random baseline and hidden_perfect_diagnostic use hidden reference only for diagnostic comparison.
- hidden_perfect_diagnostic is not an algorithmic result.
- This is selected-generator subspace optimum recovery, not full 12-bit case14 T=2 global optimization.
- Shot-batch sampling uses repeated circuit sampling and must be reported with total_shots.
"""


def _grouped_summary_rows(grouped_summary: list[dict[str, object]]) -> str:
    rows = []
    for row in grouped_summary:
        rows.append(
            "| "
            + " | ".join(
                [
                    ",".join(str(x) for x in row["selected_generators"]),
                    str(row["num_search_qubits"]),
                    str(row["train_sample_count"]),
                    str(row["measurement_policy"]),
                    str(row["max_candidates_per_shotbatch"]),
                    str(row["refit_policy"]),
                    str(row["learner"]),
                    str(row["oracle_mode"]),
                    _pct(row["exact_success_rate"]),
                    _pct(row["within_3_percent_success_rate"]),
                    _fmt(row["avg_random_exact_success_probability"]),
                    _fmt(row["avg_algorithmic_ed_lp_calls"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows) if rows else "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"


def _random_baseline_rows(learned_rows: list[dict[str, object]]) -> str:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in learned_rows:
        key = (
            tuple(row["selected_generators"]),
            row["num_search_qubits"],
            row["train_sample_count"],
        )
        grouped.setdefault(key, []).append(row)
    rows = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        rows.append(
            "| "
            + " | ".join(
                [
                    ",".join(str(x) for x in key[0]),
                    str(key[1]),
                    str(key[2]),
                    _fmt(_avg(group_rows, "search_verification_calls")),
                    _pct(_rate(sum(bool(row.get("found_hidden_exact_optimum")) for row in group_rows), len(group_rows))),
                    _fmt(_avg(group_rows, "random_exact_success_probability")),
                    _pct(_rate(sum(bool(row.get("success_within_1_percent")) for row in group_rows), len(group_rows))),
                    _fmt(_avg(group_rows, "random_success_within_1_percent_probability")),
                    _pct(_rate(sum(bool(row.get("success_within_3_percent")) for row in group_rows), len(group_rows))),
                    _fmt(_avg(group_rows, "random_success_within_3_percent_probability")),
                    _pct(_rate(sum(bool(row.get("success_within_5_percent")) for row in group_rows), len(group_rows))),
                    _fmt(_avg(group_rows, "random_success_within_5_percent_probability")),
                ]
            )
            + " |"
        )
    return "\n".join(rows) if rows else "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"


def _aggregate_markdown_rows(
    rows: list[dict[str, object]],
    *,
    key_fields: tuple[str, ...],
    display_fields: tuple[str, ...],
) -> str:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(field) for field in key_fields), []).append(row)
    lines = []
    for _, group_rows in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        aggregate = _aggregate_run_group(group_rows)
        display = [_display_aggregate_field(group_rows[0], aggregate, field) for field in display_fields]
        lines.append(
            "| "
            + " | ".join(
                display
                + [
                    str(aggregate["num_runs"]),
                    _pct(aggregate["exact_success_rate"]),
                    _pct(aggregate["within_3_percent_success_rate"]),
                    _fmt(aggregate["avg_algorithmic_ed_lp_calls"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) if lines else "| n/a | n/a | n/a | n/a | n/a |"


def _aggregate_run_group(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "num_runs": len(rows),
        "exact_success_rate": _rate(sum(bool(row.get("found_hidden_exact_optimum")) for row in rows), len(rows)),
        "within_3_percent_success_rate": _rate(sum(bool(row.get("success_within_3_percent")) for row in rows), len(rows)),
        "avg_algorithmic_ed_lp_calls": _avg(rows, "algorithmic_ed_lp_calls"),
        "avg_total_shots": _avg(rows, "total_shots"),
        "avg_circuit_executions": _avg(rows, "circuit_executions"),
        "avg_verified_candidates": _avg(rows, "verified_candidates"),
    }


def _display_aggregate_field(row: dict[str, object], aggregate: dict[str, object], field: str) -> str:
    if field.startswith("avg_"):
        return _fmt(aggregate.get(field))
    value = row.get(field)
    if isinstance(value, float):
        return _fmt(value)
    return str(value)


def _write_csv(path: Path, runs: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        for row in runs:
            writer.writerow({field: _csv_value(row.get(field)) for field in DIAGNOSTIC_FIELDS})


def _csv_value(value: object) -> object:
    value = sanitize_for_strict_json(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def _avg(rows: list[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return None if not values else float(mean(values))


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator / denominator)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_csv_strings(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--backend", choices=("qasm", "statevector", "fake", "ibm"), default="qasm")
    parser.add_argument("--shots-values", type=_parse_csv_ints, default=[2000])
    parser.add_argument("--measurement-policies", type=_parse_csv_strings, default=["shot_batch"])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--configs", type=parse_configs, default=[(0, 5), (0, 1, 2)])
    parser.add_argument("--train-sample-counts", type=parse_train_sample_counts, default=[4, 8, 16])
    parser.add_argument("--max-candidates-per-shotbatch-values", type=_parse_csv_ints, default=[1])
    parser.add_argument("--refit-policies", type=_parse_csv_strings, default=["none"])
    parser.add_argument("--learners", type=_parse_csv_strings, default=["mismatch"])
    parser.add_argument("--oracle-modes", type=_parse_csv_strings, default=["learned"])
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-trials-per-threshold", type=int, default=4)
    parser.add_argument("--exclude-hidden-optimum-from-training", action="store_true")
    parser.add_argument("--exclude-hidden-optimum-from-initial", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("results/stage1_case14_t2_small_sample_gate_level_gas_diagnostics_smoke.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/stage1_case14_t2_small_sample_gate_level_gas_diagnostics_smoke.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/stage1_case14_t2_small_sample_gate_level_gas_diagnostics_smoke_summary.md"))
    args = parser.parse_args()

    payload = run_diagnostics(
        instance_path=args.instance,
        backend=args.backend,
        configs=args.configs,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        train_sample_counts=args.train_sample_counts,
        measurement_policies=args.measurement_policies,
        shots_values=args.shots_values,
        max_candidates_values=args.max_candidates_per_shotbatch_values,
        refit_policies=args.refit_policies,
        learners=args.learners,
        oracle_modes=args.oracle_modes,
        max_rounds=args.max_rounds,
        max_trials_per_threshold=args.max_trials_per_threshold,
        exclude_hidden_optimum_from_training=args.exclude_hidden_optimum_from_training,
        exclude_hidden_optimum_from_initial=args.exclude_hidden_optimum_from_initial,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_summary=args.output_summary,
    )
    print(json.dumps({"num_runs": len(payload["runs"]), "num_errors": sum(row.get("status") != "ok" for row in payload["runs"])}, indent=2))


if __name__ == "__main__":
    main()
