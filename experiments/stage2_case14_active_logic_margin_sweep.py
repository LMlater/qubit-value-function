from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_active_logic_joint_bbht import (  # noqa: E402
    INITIALIZATION_POLICIES,
    _collect_training_data,
    _make_exact_evaluator,
    _select_initial_index,
    _validation_landscape,
)
from qubit_value_function.coherent_phase_value import (  # noqa: E402
    quantize_sparse_phase_model,
)
from qubit_value_function.experiment_utils import (  # noqa: E402
    embedded_selected_commitments,
    time_window_instance,
    write_strict_json,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig  # noqa: E402
from qubit_value_function.gate_level_oracle import bitstring_from_index  # noqa: E402
from qubit_value_function.logic_feasibility_oracle import (  # noqa: E402
    LogicFeasibilitySpec,
    compile_logic_feasibility_spec,
)
from qubit_value_function.logic_subspace_scan import (  # noqa: E402
    DEFAULT_MAX_SCAN_QUBITS,
    LogicSubspaceScanRow,
    build_logic_safe_base_commitment,
    scan_logic_feasibility_subspaces,
    select_active_logic_subspace,
)
from qubit_value_function.sparse_phase_vqc import fit_sparse_phase_vqc  # noqa: E402
from qubit_value_function.sparse_vqc_bbht import (  # noqa: E402
    BBHTConfig,
    SparseVQCBBHTResult,
    run_sparse_vqc_bbht,
)
from qubit_value_function.sparse_vqc_grover import (  # noqa: E402
    ordinary_grover_validation_plan,
)
from qubit_value_function.uc_loader import UCInstance, load_uc_instance  # noqa: E402


DEFAULT_MARGINS = (0, 1, 2, 4, 8)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("至少需要一个整数")
    return values


def _validate_margins(margins: Sequence[int]) -> tuple[int, ...]:
    rows = tuple(int(value) for value in margins)
    if not rows:
        raise ValueError("surrogate margins 不能为空")
    if any(value < 0 for value in rows):
        raise ValueError("surrogate margins 必须为非负整数")
    if len(set(rows)) != len(rows):
        raise ValueError("surrogate margins 不能重复")
    return rows


def _window_execution_plan(
    *,
    source_horizon: int,
    horizon: int,
    requested_window_starts: Sequence[int],
) -> tuple[tuple[int, ...], tuple[dict[str, object], ...]]:
    executed: list[int] = []
    skipped: list[dict[str, object]] = []
    for raw_start in requested_window_starts:
        start = int(raw_start)
        if start < 0:
            skipped.append(
                {
                    "window_start": start,
                    "reason": "window_start must be nonnegative",
                }
            )
        elif start + int(horizon) > int(source_horizon):
            skipped.append(
                {
                    "window_start": start,
                    "reason": "window_start + horizon exceeds source horizon",
                    "window_end_exclusive": int(start + horizon),
                    "source_horizon": int(source_horizon),
                }
            )
        else:
            executed.append(start)
    return tuple(executed), tuple(skipped)


def _resolve_selected_subspace(
    source: UCInstance,
    *,
    scan_horizons: Sequence[int],
    scan_window_start: int,
    selected_generator_indices: tuple[int, int] | None,
    horizon: int | None,
    train_sample_count: int,
    max_scan_qubits: int,
) -> tuple[LogicSubspaceScanRow, tuple[LogicSubspaceScanRow, ...]]:
    explicit_selected = selected_generator_indices is not None
    explicit_horizon = horizon is not None
    if explicit_selected != explicit_horizon:
        raise ValueError(
            "selected_generator_indices 与 horizon 必须同时提供或同时缺省"
        )

    horizons = tuple(dict.fromkeys(int(value) for value in scan_horizons))
    if explicit_horizon and int(horizon) not in horizons:
        horizons = (*horizons, int(horizon))
    rows = scan_logic_feasibility_subspaces(
        source,
        horizons=horizons,
        selected_generator_count=2,
        window_start=int(scan_window_start),
        max_scan_qubits=int(max_scan_qubits),
    )

    if not explicit_selected:
        return (
            select_active_logic_subspace(
                rows,
                minimum_feasible_states=int(train_sample_count),
            ),
            rows,
        )

    selected = tuple(int(value) for value in selected_generator_indices or ())
    if len(selected) != 2:
        raise ValueError("本实验只支持两个 selected generators")
    match = next(
        (
            row
            for row in rows
            if row.selected_generator_indices == selected
            and int(row.horizon) == int(horizon)
        ),
        None,
    )
    if match is None:
        raise ValueError("显式 selected generators/horizon 不在受保护扫描范围内")
    if not match.has_active_filter or not match.classical_spec_agree:
        raise ValueError("显式 selected generators/horizon 没有有效 active logic filter")
    if match.logic_feasible_count < int(train_sample_count):
        raise ValueError("显式子空间的逻辑可行训练状态不足")
    return match, rows


def _bits(index: int, num_qubits: int) -> tuple[int, ...]:
    return tuple((int(index) >> bit) & 1 for bit in range(int(num_qubits)))


def _annotated_trace(
    result: SparseVQCBBHTResult,
    spec: LogicFeasibilitySpec,
    value_model,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in result.trial_trace:
        row = dict(source)
        index = int(row["measured_index"])
        bits = _bits(index, value_model.num_x_qubits)
        effective = int(row["effective_oracle_threshold_before"])
        row["classical_hard_logic_feasible"] = bool(spec.is_feasible(bits))
        row["classical_joint_feasible_and_better"] = bool(
            spec.is_feasible(bits)
            and value_model.integer_value(bits) < effective
        )
        rows.append(row)
    return rows


def _method_report(
    result: SparseVQCBBHTResult,
    *,
    spec: LogicFeasibilitySpec,
    value_model,
) -> dict[str, object]:
    trace = _annotated_trace(result, spec, value_model)
    hard_infeasible = sum(
        not bool(row["classical_hard_logic_feasible"]) for row in trace
    )
    hard_infeasible_better = sum(
        (not bool(row["classical_hard_logic_feasible"]))
        and bool(row["surrogate_better"])
        for row in trace
    )
    return {
        "result": result.as_dict(),
        "annotated_trial_trace": trace,
        "measurement_diagnostics": {
            "hard_logic_infeasible_measurements": int(hard_infeasible),
            "hard_logic_infeasible_and_surrogate_better_measurements": int(
                hard_infeasible_better
            ),
        },
        "resources": {
            "circuit_build_seconds_total": float(
                sum(float(row.get("circuit_build_seconds", 0.0)) for row in trace)
            ),
            "transpile_seconds_total": float(
                sum(float(row.get("transpile_seconds", 0.0)) for row in trace)
            ),
            "backend_run_seconds_total": float(
                sum(float(row.get("backend_run_seconds", 0.0)) for row in trace)
            ),
            "mps_elapsed_seconds_total": float(
                sum(float(row.get("elapsed_seconds", 0.0)) for row in trace)
            ),
            "total_trial_seconds": float(
                sum(
                    float(
                        row.get(
                            "total_trial_seconds",
                            row.get("elapsed_seconds", 0.0),
                        )
                    )
                    for row in trace
                )
            ),
            "maximum_trial_qubits": int(
                max((int(row["total_qubits"]) for row in trace), default=0)
            ),
            "maximum_trial_top_level_depth": int(
                max(
                    (
                        int(row["circuit_resources"]["depth"])
                        for row in trace
                        if row.get("circuit_resources")
                    ),
                    default=0,
                )
            ),
            "minimum_auxiliary_zero_probability": float(
                min(
                    (float(row["auxiliary_zero_probability"]) for row in trace),
                    default=1.0,
                )
            ),
        },
    }


def _threshold_diagnostics(
    *,
    value_model,
    spec: LogicFeasibilitySpec,
    result: SparseVQCBBHTResult,
    true_costs: Mapping[int, float],
    use_feasibility: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for history in result.threshold_history:
        encoded = int(history["encoded_threshold"])
        effective = int(history["effective_oracle_threshold"])
        true_threshold = float(history["true_threshold"])
        plan = ordinary_grover_validation_plan(
            value_model,
            encoded_threshold=effective,
            max_validation_qubits=12,
            feasibility_spec=spec if use_feasibility else None,
        )
        marked = set(plan.marked_indices)
        baseline = set(
            ordinary_grover_validation_plan(
                value_model,
                encoded_threshold=encoded,
                max_validation_qubits=12,
                feasibility_spec=spec if use_feasibility else None,
            ).marked_indices
        )
        true_improving = {
            int(index)
            for index, cost in true_costs.items()
            if float(cost) < true_threshold
        }
        intersection = marked & true_improving
        recall = (
            float(len(intersection) / len(true_improving))
            if true_improving
            else 1.0
        )
        precision = float(len(intersection) / len(marked)) if marked else 1.0
        candidate_inflation: float | None
        if baseline:
            candidate_inflation = float(len(marked) / len(baseline))
        elif not marked:
            candidate_inflation = 1.0
        else:
            candidate_inflation = None
        cost_only_set = set(
            ordinary_grover_validation_plan(
                value_model,
                encoded_threshold=effective,
                max_validation_qubits=12,
            ).marked_indices
        )
        joint_set = set(
            ordinary_grover_validation_plan(
                value_model,
                encoded_threshold=effective,
                max_validation_qubits=12,
                feasibility_spec=spec,
            ).marked_indices
        )
        filtering_ratio = (
            float(1.0 - len(joint_set) / len(cost_only_set))
            if cost_only_set
            else 0.0
        )
        rows.append(
            {
                "update_number": int(history["update_number"]),
                "true_threshold": true_threshold,
                "encoded_threshold": encoded,
                "surrogate_integer_margin": int(
                    history["surrogate_integer_margin"]
                ),
                "effective_oracle_threshold": effective,
                "marked_indices": sorted(marked),
                "marked_count": int(len(marked)),
                "margin_zero_marked_indices_at_same_incumbent": sorted(baseline),
                "candidate_inflation": candidate_inflation,
                "true_improving_logic_feasible_indices": sorted(true_improving),
                "surrogate_true_positive_indices": sorted(intersection),
                "surrogate_false_positive_indices": sorted(marked - true_improving),
                "surrogate_false_negative_indices": sorted(true_improving - marked),
                "surrogate_recall": recall,
                "surrogate_precision": precision,
                "cost_only_better_indices": sorted(cost_only_set),
                "joint_feasible_and_better_indices": sorted(joint_set),
                "hard_infeasible_better_indices_removed": sorted(
                    cost_only_set - joint_set
                ),
                "logic_filtering_ratio": filtering_ratio,
            }
        )
    return rows


def _prepare_window(
    *,
    source: UCInstance,
    window_start: int,
    horizon: int,
    selected_generator_indices: tuple[int, int],
    train_sample_count: int,
    fixed_point: FixedPointConfig,
    initialization_policy: str,
    seed: int,
    regularization: float,
    maxiter: int,
) -> dict[str, object]:
    instance = time_window_instance(
        source,
        start=int(window_start),
        horizon=int(horizon),
    )
    base = build_logic_safe_base_commitment(instance)
    commitments = embedded_selected_commitments(base, selected_generator_indices)
    num_x_qubits = len(selected_generator_indices) * int(horizon)
    spec = compile_logic_feasibility_spec(
        instance,
        selected_generator_indices=selected_generator_indices,
        base_commitment=base,
    )
    if spec.always_infeasible or not spec.patterns:
        raise RuntimeError("选定场景没有 active hard-logic filter")

    (
        train_indices,
        train_bitstrings,
        train_costs,
        training_cache,
        training_trace,
        training_lp_calls,
    ) = _collect_training_data(
        instance,
        commitments,
        train_sample_count=int(train_sample_count),
        num_x_qubits=num_x_qubits,
    )
    fit = fit_sparse_phase_vqc(
        bitstrings=train_bitstrings,
        costs=train_costs,
        num_generators=len(selected_generator_indices),
        num_periods=int(horizon),
        generator_edges=((0, 1),),
        seed=int(seed) + int(window_start),
        regularization=float(regularization),
        maxiter=int(maxiter),
    )
    phase_model = fit.model
    value_model = quantize_sparse_phase_model(phase_model, fixed_point)
    initial_index = _select_initial_index(
        train_indices,
        training_cache,
        policy=initialization_policy,
        seed=int(seed) + int(window_start),
    )
    return {
        "instance": instance,
        "base": base,
        "commitments": commitments,
        "num_x_qubits": num_x_qubits,
        "spec": spec,
        "training_indices": train_indices,
        "training_bitstrings": train_bitstrings,
        "training_costs": train_costs,
        "training_cache": training_cache,
        "training_trace": training_trace,
        "training_lp_calls": training_lp_calls,
        "fit": fit,
        "phase_model": phase_model,
        "value_model": value_model,
        "initial_index": initial_index,
    }


def _run_window_sweep(
    *,
    source: UCInstance,
    window_start: int,
    horizon: int,
    selected_generator_indices: tuple[int, int],
    margins: Sequence[int],
    train_sample_count: int,
    fixed_point: FixedPointConfig,
    initialization_policy: str,
    seed: int,
    regularization: float,
    maxiter: int,
    base_config: BBHTConfig,
) -> dict[str, object]:
    prepared = _prepare_window(
        source=source,
        window_start=window_start,
        horizon=horizon,
        selected_generator_indices=selected_generator_indices,
        train_sample_count=train_sample_count,
        fixed_point=fixed_point,
        initialization_policy=initialization_policy,
        seed=seed,
        regularization=regularization,
        maxiter=maxiter,
    )
    instance = prepared["instance"]
    commitments = prepared["commitments"]
    spec = prepared["spec"]
    value_model = prepared["value_model"]
    training_cache = prepared["training_cache"]
    initial_index = int(prepared["initial_index"])

    search_results: list[dict[str, object]] = []
    for margin in margins:
        config = BBHTConfig(
            lambda_factor=base_config.lambda_factor,
            max_trials=base_config.max_trials,
            max_oracle_calls=base_config.max_oracle_calls,
            max_new_ed_lp_calls=base_config.max_new_ed_lp_calls,
            max_threshold_updates=base_config.max_threshold_updates,
            max_consecutive_nonimproving_marked=(
                base_config.max_consecutive_nonimproving_marked
            ),
            max_same_encoded_threshold_updates=(
                base_config.max_same_encoded_threshold_updates
            ),
            max_auxiliary_syndrome_rejections=(
                base_config.max_auxiliary_syndrome_rejections
            ),
            surrogate_integer_margin=int(margin),
            shots_per_trial=1,
            minimum_auxiliary_zero_probability=(
                base_config.minimum_auxiliary_zero_probability
            ),
            seed=int(seed) + 10_000 * int(window_start),
        )
        cost_only = run_sparse_vqc_bbht(
            value_model,
            initial_incumbent_index=initial_index,
            initial_exact_cache=dict(training_cache),
            evaluate_candidate=_make_exact_evaluator(instance, commitments),
            config=config,
            feasibility_spec=None,
        )
        joint = run_sparse_vqc_bbht(
            value_model,
            initial_incumbent_index=initial_index,
            initial_exact_cache=dict(training_cache),
            evaluate_candidate=_make_exact_evaluator(instance, commitments),
            config=config,
            feasibility_spec=spec,
        )
        search_results.append(
            {
                "surrogate_integer_margin": int(margin),
                "cost_only_result": cost_only,
                "joint_result": joint,
            }
        )

    true_costs, landscape_rows, validation_lp_calls = _validation_landscape(
        instance,
        commitments,
        num_x_qubits=int(prepared["num_x_qubits"]),
    )
    if not true_costs:
        raise RuntimeError("validation-only landscape 没有有限逻辑可行状态")
    global_index = int(min(true_costs, key=lambda index: true_costs[index]))
    global_cost = float(true_costs[global_index])

    runs: list[dict[str, object]] = []
    for row in search_results:
        margin = int(row["surrogate_integer_margin"])
        cost_only = row["cost_only_result"]
        joint = row["joint_result"]
        cost_report = _method_report(cost_only, spec=spec, value_model=value_model)
        joint_report = _method_report(joint, spec=spec, value_model=value_model)
        cost_diagnostics = _threshold_diagnostics(
            value_model=value_model,
            spec=spec,
            result=cost_only,
            true_costs=true_costs,
            use_feasibility=False,
        )
        joint_diagnostics = _threshold_diagnostics(
            value_model=value_model,
            spec=spec,
            result=joint,
            true_costs=true_costs,
            use_feasibility=True,
        )
        cost_counters = cost_only.as_dict()["counters"]
        joint_counters = joint.as_dict()["counters"]
        runs.append(
            {
                "surrogate_integer_margin": margin,
                "margin_real_cost_units": float(margin * fixed_point.quantum),
                "cost_only": {
                    **cost_report,
                    "threshold_diagnostics": cost_diagnostics,
                    "final_is_true_global_optimum": bool(
                        cost_only.final_incumbent_index == global_index
                    ),
                    "final_true_optimality_gap": float(
                        float(cost_only.final_incumbent_true_cost) - global_cost
                    ),
                },
                "joint": {
                    **joint_report,
                    "threshold_diagnostics": joint_diagnostics,
                    "final_is_true_global_optimum": bool(
                        joint.final_incumbent_index == global_index
                    ),
                    "final_true_optimality_gap": float(
                        float(joint.final_incumbent_true_cost) - global_cost
                    ),
                },
                "comparison": {
                    "shared_training_indices": True,
                    "shared_training_labels": True,
                    "shared_phase_model": True,
                    "shared_integer_value_model": True,
                    "shared_initial_incumbent": True,
                    "shared_bbht_config_seed_and_margin": True,
                    "joint_minus_cost_only": {
                        "trials": int(
                            joint_counters["circuit_executions"]
                            - cost_counters["circuit_executions"]
                        ),
                        "oracle_calls": int(
                            joint_counters["total_oracle_calls"]
                            - cost_counters["total_oracle_calls"]
                        ),
                        "new_exact_evaluation_attempts": int(
                            joint_counters["new_exact_evaluation_attempts"]
                            - cost_counters["new_exact_evaluation_attempts"]
                        ),
                        "actual_ed_lp_solves": int(
                            joint_counters["actual_ed_lp_solves"]
                            - cost_counters["actual_ed_lp_solves"]
                        ),
                        "logic_precheck_rejections": int(
                            joint_counters["logic_precheck_rejections"]
                            - cost_counters["logic_precheck_rejections"]
                        ),
                        "circuit_build_seconds": float(
                            joint_report["resources"]["circuit_build_seconds_total"]
                            - cost_report["resources"]["circuit_build_seconds_total"]
                        ),
                        "transpile_seconds": float(
                            joint_report["resources"]["transpile_seconds_total"]
                            - cost_report["resources"]["transpile_seconds_total"]
                        ),
                        "backend_run_seconds": float(
                            joint_report["resources"]["backend_run_seconds_total"]
                            - cost_report["resources"]["backend_run_seconds_total"]
                        ),
                        "total_trial_seconds": float(
                            joint_report["resources"]["total_trial_seconds"]
                            - cost_report["resources"]["total_trial_seconds"]
                        ),
                    },
                },
            }
        )

    initial_encoded = int(
        value_model.fixed_point_config.encode(
            float(training_cache[initial_index].total_cost)
        )
    )
    initial_margin_metrics: list[dict[str, object]] = []
    for margin in margins:
        effective = initial_encoded + int(margin)
        cost_set = set(
            ordinary_grover_validation_plan(
                value_model,
                encoded_threshold=effective,
                max_validation_qubits=12,
            ).marked_indices
        )
        joint_set = set(
            ordinary_grover_validation_plan(
                value_model,
                encoded_threshold=effective,
                max_validation_qubits=12,
                feasibility_spec=spec,
            ).marked_indices
        )
        true_improving = {
            index
            for index, cost in true_costs.items()
            if cost < float(training_cache[initial_index].total_cost)
        }
        initial_margin_metrics.append(
            {
                "surrogate_integer_margin": int(margin),
                "effective_oracle_threshold": int(effective),
                "cost_only_marked_count": int(len(cost_set)),
                "joint_marked_count": int(len(joint_set)),
                "hard_infeasible_removed_count": int(len(cost_set - joint_set)),
                "joint_surrogate_recall": (
                    float(len(joint_set & true_improving) / len(true_improving))
                    if true_improving
                    else 1.0
                ),
                "joint_surrogate_false_negative_count": int(
                    len(true_improving - joint_set)
                ),
                "joint_surrogate_false_positive_count": int(
                    len(joint_set - true_improving)
                ),
            }
        )

    return {
        "window_start": int(window_start),
        "window_end_exclusive": int(window_start + horizon),
        "horizon": int(horizon),
        "selected_generator_indices": list(selected_generator_indices),
        "selected_generator_names": [
            source.generators[index].name for index in selected_generator_indices
        ],
        "base_commitment": prepared["base"].tolist(),
        "hard_logic_feasibility_spec": spec.as_dict(),
        "hard_logic_feasible_indices": [
            index
            for index in range(2 ** int(prepared["num_x_qubits"]))
            if spec.is_feasible(_bits(index, int(prepared["num_x_qubits"])))
        ],
        "hard_logic_infeasible_indices": [
            index
            for index in range(2 ** int(prepared["num_x_qubits"]))
            if not spec.is_feasible(_bits(index, int(prepared["num_x_qubits"])))
        ],
        "training": {
            "indices": prepared["training_indices"],
            "bitstrings": prepared["training_bitstrings"],
            "costs": prepared["training_costs"],
            "trace": prepared["training_trace"],
            "actual_ed_lp_solves": int(prepared["training_lp_calls"]),
        },
        "fit": prepared["fit"].as_dict(),
        "phase_model": prepared["phase_model"].as_dict(),
        "quantized_value_model": value_model.as_dict(),
        "initialization_policy": initialization_policy,
        "initial_incumbent_index": initial_index,
        "initial_margin_metrics": initial_margin_metrics,
        "runs": runs,
        "validation_only": {
            "evaluated_after_all_margin_searches": True,
            "actual_ed_lp_solves": int(validation_lp_calls),
            "rows": landscape_rows,
            "true_global_optimum_index": int(global_index),
            "true_global_optimum_bitstring": bitstring_from_index(
                global_index, int(prepared["num_x_qubits"])
            ),
            "true_global_optimum_cost": float(global_cost),
        },
    }


def run(
    *,
    instance_path: Path,
    results_path: Path,
    scan_horizons: tuple[int, ...] = (2, 3),
    scan_window_start: int = 0,
    selected_generator_indices: tuple[int, int] | None = None,
    horizon: int | None = None,
    requested_window_starts: tuple[int, ...] = (0, 1, 2),
    margins: tuple[int, ...] = DEFAULT_MARGINS,
    max_scan_qubits: int = DEFAULT_MAX_SCAN_QUBITS,
    train_sample_count: int = 8,
    fractional_bits: int = 2,
    cost_unit: float = 1000.0,
    initialization_policy: str = "worst-training",
    seed: int = 0,
    regularization: float = 1e-4,
    maxiter: int = 300,
    lambda_factor: float = 1.2,
    max_trials: int = 64,
    max_oracle_calls: int = 128,
    max_new_ed_lp_calls: int = 16,
    max_threshold_updates: int = 8,
    max_consecutive_nonimproving_marked: int = 16,
    max_same_encoded_threshold_updates: int = 3,
    max_auxiliary_syndrome_rejections: int = 8,
    minimum_auxiliary_zero_probability: float = 1.0 - 1e-12,
) -> dict[str, object]:
    source = load_uc_instance(instance_path)
    margins = _validate_margins(margins)
    selected_row, scan_rows = _resolve_selected_subspace(
        source,
        scan_horizons=scan_horizons,
        scan_window_start=scan_window_start,
        selected_generator_indices=selected_generator_indices,
        horizon=horizon,
        train_sample_count=train_sample_count,
        max_scan_qubits=max_scan_qubits,
    )
    selected = tuple(selected_row.selected_generator_indices)
    selected_horizon = int(selected_row.horizon)

    executed, skipped = _window_execution_plan(
        source_horizon=source.time_horizon,
        horizon=selected_horizon,
        requested_window_starts=requested_window_starts,
    )
    if not executed:
        raise ValueError("请求的 window starts 中没有可执行窗口")

    fixed_point = FixedPointConfig(
        fractional_bits=int(fractional_bits),
        unit=float(cost_unit),
        rounding="nearest",
    )
    base_config = BBHTConfig(
        lambda_factor=float(lambda_factor),
        max_trials=int(max_trials),
        max_oracle_calls=int(max_oracle_calls),
        max_new_ed_lp_calls=int(max_new_ed_lp_calls),
        max_threshold_updates=int(max_threshold_updates),
        max_consecutive_nonimproving_marked=int(
            max_consecutive_nonimproving_marked
        ),
        max_same_encoded_threshold_updates=int(
            max_same_encoded_threshold_updates
        ),
        max_auxiliary_syndrome_rejections=int(
            max_auxiliary_syndrome_rejections
        ),
        surrogate_integer_margin=0,
        shots_per_trial=1,
        minimum_auxiliary_zero_probability=float(
            minimum_auxiliary_zero_probability
        ),
        seed=int(seed),
    )
    scenarios = [
        _run_window_sweep(
            source=source,
            window_start=start,
            horizon=selected_horizon,
            selected_generator_indices=selected,
            margins=margins,
            train_sample_count=train_sample_count,
            fixed_point=fixed_point,
            initialization_policy=initialization_policy,
            seed=seed,
            regularization=regularization,
            maxiter=maxiter,
            base_config=base_config,
        )
        for start in executed
    ]
    payload = {
        "method": "stage2 active-logic sparse-VQC BBHT surrogate-margin sweep",
        "source_instance": str(instance_path),
        "selection": {
            "selected_subspace": selected_row.as_dict(),
            "uses_cost": False,
            "uses_ed_lp": False,
            "uses_vqc": False,
            "uses_bbht_outcome": False,
            "uses_hidden_optimum": False,
            "explicit_pair_and_horizon_contract": True,
        },
        "scan_summary": {
            "horizons": [int(value) for value in scan_horizons],
            "window_start": int(scan_window_start),
            "max_scan_qubits": int(max_scan_qubits),
            "num_scanned_subspaces": int(len(scan_rows)),
            "num_active_subspaces": int(
                sum(row.has_active_filter for row in scan_rows)
            ),
        },
        "selected_generator_indices": list(selected),
        "horizon": selected_horizon,
        "window_execution": {
            "requested_window_starts": [
                int(value) for value in requested_window_starts
            ],
            "executed_window_starts": list(executed),
            "skipped_windows": list(skipped),
        },
        "surrogate_integer_margins": list(margins),
        "surrogate_margin_real_cost_units": [
            float(value * fixed_point.quantum) for value in margins
        ],
        "train_sample_count": int(train_sample_count),
        "initialization_policy": initialization_policy,
        "fixed_point": {
            "fractional_bits": int(fixed_point.fractional_bits),
            "cost_unit": float(fixed_point.unit),
            "quantum": float(fixed_point.quantum),
        },
        "base_bbht_config": base_config.__dict__,
        "fair_comparison": {
            "same_training_data_across_methods_and_margins": True,
            "same_vqc_and_integer_model_across_methods_and_margins": True,
            "same_initial_incumbent_across_methods_and_margins": True,
            "same_bbht_seed_and_budgets": True,
            "only_margin_difference": (
                "effective oracle threshold = encoded true incumbent threshold + "
                "surrogate integer margin"
            ),
            "threshold_updates_require_strict_true_ed_lp_improvement": True,
        },
        "scenarios": scenarios,
        "scope_boundary": {
            "encoded_hard_constraints": [
                "must_run",
                "initial_remaining_minimum_uptime",
                "initial_remaining_minimum_downtime",
                "minimum_uptime_after_startup",
                "minimum_downtime_after_shutdown",
            ],
            "not_encoded": [
                "capacity adequacy",
                "reserve adequacy",
                "minimum-output adequacy",
                "continuous ramp feasibility",
                "network power flow",
                "N-1 security",
            ],
        },
    }
    write_strict_json(results_path, payload)
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="active logic 子空间中的 sparse-VQC surrogate margin sweep"
    )
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage2_case14_active_logic_margin_sweep.json"),
    )
    parser.add_argument("--scan-horizons", type=_parse_int_tuple, default=(2, 3))
    parser.add_argument("--scan-window-start", type=int, default=0)
    parser.add_argument("--selected-generators", type=_parse_int_tuple, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--window-starts", type=_parse_int_tuple, default=(0, 1, 2))
    parser.add_argument("--margins", type=_parse_int_tuple, default=DEFAULT_MARGINS)
    parser.add_argument("--max-scan-qubits", type=int, default=DEFAULT_MAX_SCAN_QUBITS)
    parser.add_argument("--train-sample-count", type=int, default=8)
    parser.add_argument("--fractional-bits", type=int, default=2)
    parser.add_argument("--cost-unit", type=float, default=1000.0)
    parser.add_argument(
        "--initialization-policy",
        choices=INITIALIZATION_POLICIES,
        default="worst-training",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--lambda-factor", type=float, default=1.2)
    parser.add_argument("--max-trials", type=int, default=64)
    parser.add_argument("--max-oracle-calls", type=int, default=128)
    parser.add_argument("--max-new-ed-lp-calls", type=int, default=16)
    parser.add_argument("--max-threshold-updates", type=int, default=8)
    parser.add_argument("--max-consecutive-nonimproving-marked", type=int, default=16)
    parser.add_argument("--max-same-encoded-threshold-updates", type=int, default=3)
    parser.add_argument("--max-auxiliary-syndrome-rejections", type=int, default=8)
    parser.add_argument(
        "--minimum-auxiliary-zero-probability",
        type=float,
        default=1.0 - 1e-12,
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    selected = (
        tuple(args.selected_generators)
        if args.selected_generators is not None
        else None
    )
    if selected is not None and len(selected) != 2:
        raise ValueError("本实验只支持两个 selected generators")
    payload = run(
        instance_path=args.instance,
        results_path=args.results,
        scan_horizons=tuple(args.scan_horizons),
        scan_window_start=args.scan_window_start,
        selected_generator_indices=selected,
        horizon=args.horizon,
        requested_window_starts=tuple(args.window_starts),
        margins=tuple(args.margins),
        max_scan_qubits=args.max_scan_qubits,
        train_sample_count=args.train_sample_count,
        fractional_bits=args.fractional_bits,
        cost_unit=args.cost_unit,
        initialization_policy=args.initialization_policy,
        seed=args.seed,
        regularization=args.regularization,
        maxiter=args.maxiter,
        lambda_factor=args.lambda_factor,
        max_trials=args.max_trials,
        max_oracle_calls=args.max_oracle_calls,
        max_new_ed_lp_calls=args.max_new_ed_lp_calls,
        max_threshold_updates=args.max_threshold_updates,
        max_consecutive_nonimproving_marked=(
            args.max_consecutive_nonimproving_marked
        ),
        max_same_encoded_threshold_updates=(
            args.max_same_encoded_threshold_updates
        ),
        max_auxiliary_syndrome_rejections=(
            args.max_auxiliary_syndrome_rejections
        ),
        minimum_auxiliary_zero_probability=(
            args.minimum_auxiliary_zero_probability
        ),
    )
    print(
        json.dumps(
            {
                "selected_generator_indices": payload[
                    "selected_generator_indices"
                ],
                "horizon": payload["horizon"],
                "window_execution": payload["window_execution"],
                "margins": payload["surrogate_integer_margins"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
