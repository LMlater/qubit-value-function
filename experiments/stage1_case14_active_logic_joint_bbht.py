from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.commitment import is_logic_feasible  # noqa: E402
from qubit_value_function.coherent_phase_value import quantize_sparse_phase_model  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
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
    LogicSubspaceScanRow,
    build_logic_safe_base_commitment,
    scan_logic_feasibility_subspaces,
    select_active_logic_subspace,
)
from qubit_value_function.sparse_phase_vqc import fit_sparse_phase_vqc  # noqa: E402
from qubit_value_function.sparse_vqc_bbht import (  # noqa: E402
    BBHTConfig,
    ExactCandidateEvaluation,
    SparseVQCBBHTResult,
    run_sparse_vqc_bbht,
)
from qubit_value_function.sparse_vqc_grover import (  # noqa: E402
    ordinary_grover_validation_plan,
)
from qubit_value_function.uc_loader import UCInstance, load_uc_instance  # noqa: E402


INITIALIZATION_POLICIES = ("first", "random", "best-training", "worst-training")


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("至少需要一个整数")
    return values


def _representative_index_order(num_bits: int) -> tuple[int, ...]:
    num_bits = int(num_bits)
    dimension = 2**num_bits
    full = dimension - 1
    bit_reversal = sorted(
        range(dimension),
        key=lambda index: int(f"{index:0{num_bits}b}"[::-1], 2),
    )
    rows: list[int] = []
    for index in (0, full, *bit_reversal):
        if int(index) not in rows:
            rows.append(int(index))
    return tuple(rows)


def _collect_training_data(
    instance: UCInstance,
    commitments: np.ndarray,
    *,
    train_sample_count: int,
    num_x_qubits: int,
) -> tuple[
    list[int],
    list[str],
    list[float],
    dict[int, ExactCandidateEvaluation],
    list[dict[str, object]],
    int,
]:
    evaluator = FixedCommitmentEvaluator(instance)
    indices: list[int] = []
    bitstrings: list[str] = []
    costs: list[float] = []
    cache: dict[int, ExactCandidateEvaluation] = {}
    trace: list[dict[str, object]] = []
    lp_calls = 0
    for index in _representative_index_order(num_x_qubits):
        commitment = commitments[int(index)]
        bitstring = bitstring_from_index(index, num_x_qubits)
        if not is_logic_feasible(instance, commitment):
            trace.append(
                {
                    "index": int(index),
                    "bitstring": bitstring,
                    "status": "skipped_logic_infeasible",
                }
            )
            continue
        result = evaluator.evaluate(commitment)
        lp_calls += 1
        finite = bool(result.success and np.isfinite(result.total_cost))
        trace.append(
            {
                "index": int(index),
                "bitstring": bitstring,
                "status": "finite" if finite else "ed_failed",
                "true_cost": float(result.total_cost) if finite else None,
                "dispatch_cost": float(result.dispatch_cost) if finite else None,
                "startup_cost": float(result.startup_cost) if finite else None,
                "balance_penalty": float(result.balance_penalty) if finite else None,
                "reserve_penalty": float(result.reserve_penalty) if finite else None,
            }
        )
        if not finite:
            continue
        record = ExactCandidateEvaluation(
            success=True,
            total_cost=float(result.total_cost),
            message=str(result.message),
            source="training_exact_cache",
            lp_solve_performed=True,
            logic_precheck_rejected=False,
        )
        cache[int(index)] = record
        indices.append(int(index))
        bitstrings.append(bitstring)
        costs.append(float(result.total_cost))
        if len(indices) >= int(train_sample_count):
            break
    if len(indices) < int(train_sample_count):
        raise RuntimeError(
            f"有限且逻辑可行的训练样本不足：{len(indices)} < {train_sample_count}"
        )
    return indices, bitstrings, costs, cache, trace, int(lp_calls)


def _select_initial_index(
    training_indices: Sequence[int],
    cache: Mapping[int, ExactCandidateEvaluation],
    *,
    policy: str,
    seed: int,
) -> int:
    if policy not in INITIALIZATION_POLICIES:
        raise ValueError(f"initialization_policy 必须是 {INITIALIZATION_POLICIES} 之一")
    indices = [int(index) for index in training_indices]
    if policy == "first":
        return indices[0]
    if policy == "random":
        return int(np.random.default_rng(int(seed)).choice(np.asarray(indices, dtype=int)))
    key: Callable[[int], float] = lambda index: float(cache[index].total_cost)
    return int(min(indices, key=key) if policy == "best-training" else max(indices, key=key))


def _make_exact_evaluator(
    instance: UCInstance,
    commitments: np.ndarray,
) -> Callable[[int], ExactCandidateEvaluation]:
    evaluator = FixedCommitmentEvaluator(instance)

    def evaluate(index: int) -> ExactCandidateEvaluation:
        commitment = commitments[int(index)]
        if not is_logic_feasible(instance, commitment):
            return ExactCandidateEvaluation(
                success=False,
                total_cost=None,
                message="classical logic precheck rejected candidate",
                source="logic_infeasible_precheck",
                lp_solve_performed=False,
                logic_precheck_rejected=True,
            )
        result = evaluator.evaluate(commitment)
        finite = bool(result.success and np.isfinite(result.total_cost))
        return ExactCandidateEvaluation(
            success=finite,
            total_cost=float(result.total_cost) if finite else None,
            message=str(result.message),
            source="new_ed_lp_call",
            lp_solve_performed=True,
            logic_precheck_rejected=False,
        )

    return evaluate


def _annotated_trace(
    result: SparseVQCBBHTResult,
    spec: LogicFeasibilitySpec,
    value_model,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in result.trial_trace:
        row = dict(source)
        index = int(row["measured_index"])
        bits = tuple((index >> bit) & 1 for bit in range(value_model.num_x_qubits))
        row["classical_hard_logic_feasible"] = bool(spec.is_feasible(bits))
        row["classical_joint_feasible_and_better"] = bool(
            spec.is_feasible(bits)
            and value_model.integer_value(bits) < int(row["encoded_threshold_before"])
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
    hard_infeasible_measurements = sum(
        not bool(row["classical_hard_logic_feasible"]) for row in trace
    )
    hard_infeasible_better_measurements = sum(
        (not bool(row["classical_hard_logic_feasible"]))
        and bool(row["surrogate_better"])
        for row in trace
    )
    return {
        "result": result.as_dict(),
        "annotated_trial_trace": trace,
        "measurement_diagnostics": {
            "hard_logic_infeasible_measurements": int(hard_infeasible_measurements),
            "hard_logic_infeasible_and_surrogate_better_measurements": int(
                hard_infeasible_better_measurements
            ),
        },
        "resources": {
            "mps_elapsed_seconds_total": float(
                sum(float(row["elapsed_seconds"]) for row in trace)
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


def _validation_landscape(
    instance: UCInstance,
    commitments: np.ndarray,
    *,
    num_x_qubits: int,
) -> tuple[dict[int, float], list[dict[str, object]], int]:
    evaluator = FixedCommitmentEvaluator(instance)
    costs: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    calls = 0
    for index, commitment in enumerate(commitments):
        feasible = bool(is_logic_feasible(instance, commitment))
        if not feasible:
            rows.append(
                {
                    "index": int(index),
                    "bitstring": bitstring_from_index(index, num_x_qubits),
                    "logic_feasible": False,
                    "status": "logic_infeasible",
                    "true_cost": None,
                }
            )
            continue
        result = evaluator.evaluate(commitment)
        calls += 1
        finite = bool(result.success and np.isfinite(result.total_cost))
        if finite:
            costs[int(index)] = float(result.total_cost)
        rows.append(
            {
                "index": int(index),
                "bitstring": bitstring_from_index(index, num_x_qubits),
                "logic_feasible": True,
                "status": "finite" if finite else "ed_failed",
                "true_cost": float(result.total_cost) if finite else None,
            }
        )
    return costs, rows, int(calls)


def _threshold_diagnostics(
    value_model,
    spec: LogicFeasibilitySpec,
    result: SparseVQCBBHTResult,
    true_costs: Mapping[int, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for history in result.threshold_history:
        encoded = int(history["encoded_threshold"])
        true_threshold = float(history["true_threshold"])
        cost_only = ordinary_grover_validation_plan(
            value_model,
            encoded_threshold=encoded,
            max_validation_qubits=12,
        )
        joint = ordinary_grover_validation_plan(
            value_model,
            encoded_threshold=encoded,
            max_validation_qubits=12,
            feasibility_spec=spec,
        )
        cost_set = set(cost_only.marked_indices)
        joint_set = set(joint.marked_indices)
        true_improving = {
            int(index)
            for index, cost in true_costs.items()
            if float(cost) < true_threshold
        }
        rows.append(
            {
                "update_number": int(history["update_number"]),
                "true_threshold": true_threshold,
                "encoded_threshold": encoded,
                "cost_only_better_indices": sorted(cost_set),
                "joint_feasible_and_better_indices": sorted(joint_set),
                "hard_infeasible_better_indices_removed": sorted(cost_set - joint_set),
                "true_improving_logic_feasible_indices": sorted(true_improving),
                "joint_surrogate_false_positive_indices": sorted(joint_set - true_improving),
                "joint_surrogate_false_negative_indices": sorted(true_improving - joint_set),
                "joint_is_strict_subset": bool(joint_set < cost_set),
            }
        )
    return rows


def _evaluate_scenario(
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
    config: BBHTConfig,
) -> dict[str, object]:
    instance = time_window_instance(source, start=int(window_start), horizon=int(horizon))
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
        train_sample_count=train_sample_count,
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

    run_config = BBHTConfig(
        lambda_factor=config.lambda_factor,
        max_trials=config.max_trials,
        max_oracle_calls=config.max_oracle_calls,
        max_new_ed_lp_calls=config.max_new_ed_lp_calls,
        max_threshold_updates=config.max_threshold_updates,
        max_consecutive_nonimproving_marked=(
            config.max_consecutive_nonimproving_marked
        ),
        max_same_encoded_threshold_updates=(
            config.max_same_encoded_threshold_updates
        ),
        max_auxiliary_syndrome_rejections=(
            config.max_auxiliary_syndrome_rejections
        ),
        shots_per_trial=1,
        minimum_auxiliary_zero_probability=(
            config.minimum_auxiliary_zero_probability
        ),
        seed=int(seed) + 10_000 * int(window_start),
    )
    cost_only = run_sparse_vqc_bbht(
        value_model,
        initial_incumbent_index=initial_index,
        initial_exact_cache=dict(training_cache),
        evaluate_candidate=_make_exact_evaluator(instance, commitments),
        config=run_config,
        feasibility_spec=None,
    )
    joint = run_sparse_vqc_bbht(
        value_model,
        initial_incumbent_index=initial_index,
        initial_exact_cache=dict(training_cache),
        evaluate_candidate=_make_exact_evaluator(instance, commitments),
        config=run_config,
        feasibility_spec=spec,
    )

    true_costs, landscape_rows, validation_lp_calls = _validation_landscape(
        instance,
        commitments,
        num_x_qubits=num_x_qubits,
    )
    if not true_costs:
        raise RuntimeError("validation-only landscape 没有有限逻辑可行状态")
    global_index = int(min(true_costs, key=lambda index: true_costs[index]))
    global_cost = float(true_costs[global_index])
    cost_report = _method_report(cost_only, spec=spec, value_model=value_model)
    joint_report = _method_report(joint, spec=spec, value_model=value_model)
    cost_thresholds = _threshold_diagnostics(value_model, spec, cost_only, true_costs)
    joint_thresholds = _threshold_diagnostics(value_model, spec, joint, true_costs)

    cost_counters = cost_only.as_dict()["counters"]
    joint_counters = joint.as_dict()["counters"]
    return {
        "window_start": int(window_start),
        "window_end_exclusive": int(window_start + horizon),
        "horizon": int(horizon),
        "selected_generator_indices": [
            int(index) for index in selected_generator_indices
        ],
        "selected_generator_names": [
            source.generators[index].name for index in selected_generator_indices
        ],
        "base_commitment": base.tolist(),
        "hard_logic_feasibility_spec": spec.as_dict(),
        "hard_logic_feasible_indices": [
            index
            for index in range(2**num_x_qubits)
            if spec.is_feasible(tuple((index >> bit) & 1 for bit in range(num_x_qubits)))
        ],
        "hard_logic_infeasible_indices": [
            index
            for index in range(2**num_x_qubits)
            if not spec.is_feasible(tuple((index >> bit) & 1 for bit in range(num_x_qubits)))
        ],
        "training": {
            "indices": train_indices,
            "bitstrings": train_bitstrings,
            "costs": train_costs,
            "trace": training_trace,
            "actual_ed_lp_solves": int(training_lp_calls),
        },
        "fit": fit.as_dict(),
        "phase_model": phase_model.as_dict(),
        "quantized_value_model": value_model.as_dict(),
        "initialization_policy": initialization_policy,
        "initial_incumbent_index": int(initial_index),
        "cost_only": {
            **cost_report,
            "threshold_diagnostics": cost_thresholds,
            "final_is_true_global_optimum": bool(
                cost_only.final_incumbent_index == global_index
            ),
            "final_true_optimality_gap": float(
                cost_only.final_incumbent_true_cost - global_cost
            ),
        },
        "joint": {
            **joint_report,
            "threshold_diagnostics": joint_thresholds,
            "final_is_true_global_optimum": bool(
                joint.final_incumbent_index == global_index
            ),
            "final_true_optimality_gap": float(
                joint.final_incumbent_true_cost - global_cost
            ),
        },
        "comparison": {
            "shared_training_indices": True,
            "shared_training_labels": True,
            "shared_phase_model": True,
            "shared_integer_value_model": True,
            "shared_initial_incumbent": True,
            "shared_bbht_config_and_seed": True,
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
                "threshold_updates": int(
                    joint_counters["threshold_updates"]
                    - cost_counters["threshold_updates"]
                ),
                "maximum_trial_qubits": int(
                    joint_report["resources"]["maximum_trial_qubits"]
                    - cost_report["resources"]["maximum_trial_qubits"]
                ),
                "maximum_trial_top_level_depth": int(
                    joint_report["resources"]["maximum_trial_top_level_depth"]
                    - cost_report["resources"]["maximum_trial_top_level_depth"]
                ),
                "mps_elapsed_seconds_total": float(
                    joint_report["resources"]["mps_elapsed_seconds_total"]
                    - cost_report["resources"]["mps_elapsed_seconds_total"]
                ),
            },
            "joint_marked_set_is_strict_subset_at_any_reported_threshold": bool(
                any(row["joint_is_strict_subset"] for row in cost_thresholds + joint_thresholds)
            ),
            "joint_logic_precheck_rejections_not_greater": bool(
                joint_counters["logic_precheck_rejections"]
                <= cost_counters["logic_precheck_rejections"]
            ),
        },
        "validation_only": {
            "evaluated_after_both_searches": True,
            "actual_ed_lp_solves": int(validation_lp_calls),
            "rows": landscape_rows,
            "true_global_optimum_index": int(global_index),
            "true_global_optimum_bitstring": bitstring_from_index(
                global_index, num_x_qubits
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
    window_starts: tuple[int, ...] = (0, 1, 2),
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
    scan_rows = scan_logic_feasibility_subspaces(
        source,
        horizons=scan_horizons,
        selected_generator_count=2,
        window_start=scan_window_start,
    )
    if selected_generator_indices is None or horizon is None:
        selected_row = select_active_logic_subspace(
            scan_rows,
            minimum_feasible_states=train_sample_count,
        )
        selected_generator_indices = tuple(selected_row.selected_generator_indices)  # type: ignore[assignment]
        horizon = int(selected_row.horizon)
    else:
        selected_row = next(
            (
                row
                for row in scan_rows
                if tuple(row.selected_generator_indices)
                == tuple(selected_generator_indices)
                and int(row.horizon) == int(horizon)
            ),
            None,
        )
        if selected_row is None:
            raise ValueError("显式 selected generators/horizon 不在扫描范围内")
        if not selected_row.has_active_filter:
            raise ValueError("显式 selected generators/horizon 没有 active logic filter")

    fixed_point = FixedPointConfig(
        fractional_bits=int(fractional_bits),
        unit=float(cost_unit),
        rounding="nearest",
    )
    config = BBHTConfig(
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
        shots_per_trial=1,
        minimum_auxiliary_zero_probability=float(
            minimum_auxiliary_zero_probability
        ),
        seed=int(seed),
    )
    scenarios = [
        _evaluate_scenario(
            source=source,
            window_start=int(window_start),
            horizon=int(horizon),
            selected_generator_indices=tuple(selected_generator_indices),
            train_sample_count=int(train_sample_count),
            fixed_point=fixed_point,
            initialization_policy=initialization_policy,
            seed=int(seed),
            regularization=float(regularization),
            maxiter=int(maxiter),
            config=config,
        )
        for window_start in window_starts
        if int(window_start) + int(horizon) <= source.time_horizon
    ]
    payload = {
        "method": "active hard-logic cost-only versus joint sparse-VQC BBHT comparison",
        "source_instance": str(instance_path),
        "selection": {
            "selected_subspace": selected_row.as_dict(),
            "uses_cost": False,
            "uses_ed_lp": False,
            "uses_vqc": False,
            "uses_bbht_outcome": False,
            "uses_hidden_optimum": False,
        },
        "scan_summary": {
            "horizons": [int(value) for value in scan_horizons],
            "window_start": int(scan_window_start),
            "num_scanned_subspaces": int(len(scan_rows)),
            "num_active_subspaces": int(
                sum(row.has_active_filter for row in scan_rows)
            ),
        },
        "selected_generator_indices": [
            int(index) for index in selected_generator_indices
        ],
        "horizon": int(horizon),
        "window_starts": [int(value) for value in window_starts],
        "train_sample_count": int(train_sample_count),
        "initialization_policy": initialization_policy,
        "fixed_point": {
            "fractional_bits": int(fixed_point.fractional_bits),
            "cost_unit": float(fixed_point.unit),
            "quantum": float(fixed_point.quantum),
        },
        "bbht_config": config.__dict__,
        "fair_comparison": {
            "same_training_data": True,
            "same_vqc_and_integer_model": True,
            "same_initial_incumbent": True,
            "same_bbht_config_and_seed": True,
            "only_quantum_oracle_difference": (
                "cost-only threshold versus hard-logic-feasible AND threshold"
            ),
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
        description="case14 active logic 子空间中 cost-only 与 joint BBHT 公平对照"
    )
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_active_logic_joint_bbht.json"),
    )
    parser.add_argument("--scan-horizons", type=_parse_int_tuple, default=(2, 3))
    parser.add_argument("--scan-window-start", type=int, default=0)
    parser.add_argument("--selected-generators", type=_parse_int_tuple, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--window-starts", type=_parse_int_tuple, default=(0, 1, 2))
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
        window_starts=tuple(args.window_starts),
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
                "selected_generator_indices": payload["selected_generator_indices"],
                "horizon": payload["horizon"],
                "scenarios": [
                    {
                        "window_start": row["window_start"],
                        "cost_only_final": row["cost_only"]["result"][
                            "final_incumbent_index"
                        ],
                        "joint_final": row["joint"]["result"][
                            "final_incumbent_index"
                        ],
                        "cost_only_logic_rejections": row["cost_only"]["result"][
                            "counters"
                        ]["logic_precheck_rejections"],
                        "joint_logic_rejections": row["joint"]["result"][
                            "counters"
                        ]["logic_precheck_rejections"],
                        "joint_strictly_filters": row["comparison"][
                            "joint_marked_set_is_strict_subset_at_any_reported_threshold"
                        ],
                    }
                    for row in payload["scenarios"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
