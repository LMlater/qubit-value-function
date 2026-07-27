from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

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
from qubit_value_function.sparse_phase_vqc import fit_sparse_phase_vqc  # noqa: E402
from qubit_value_function.sparse_vqc_bbht import (  # noqa: E402
    BBHTConfig,
    ExactCandidateEvaluation,
    run_sparse_vqc_bbht,
    select_initial_incumbent,
)
from qubit_value_function.sparse_vqc_grover import (  # noqa: E402
    direct_float_marked_indices_for_validation,
    marked_semantics_diagnostics,
    ordinary_grover_validation_plan,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


REPRESENTATIVE_INDEX_ORDER = (0, 15, 3, 12, 5, 10, 6, 9, 1, 2, 4, 8, 7, 11, 13, 14)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("至少需要一个整数")
    return values


def _collect_training_data(
    instance,
    commitments: np.ndarray,
    *,
    train_sample_count: int,
    num_x_qubits: int,
    training_indices: Sequence[int] | None = None,
) -> tuple[
    list[int],
    list[str],
    list[float],
    dict[int, ExactCandidateEvaluation],
    list[dict[str, object]],
    int,
]:
    evaluator = FixedCommitmentEvaluator(instance)
    train_indices: list[int] = []
    bitstrings: list[str] = []
    costs: list[float] = []
    cache: dict[int, ExactCandidateEvaluation] = {}
    trace: list[dict[str, object]] = []
    calls = 0
    index_order = (
        REPRESENTATIVE_INDEX_ORDER
        if training_indices is None
        else tuple(int(index) for index in training_indices)
    )
    for raw_index in index_order:
        index = int(raw_index)
        commitment = commitments[index]
        if not is_logic_feasible(instance, commitment):
            trace.append(
                {
                    "index": index,
                    "bitstring": bitstring_from_index(index, num_x_qubits),
                    "status": "skipped_logic_infeasible",
                }
            )
            continue
        result = evaluator.evaluate(commitment)
        calls += 1
        finite = bool(result.success and np.isfinite(result.total_cost))
        trace.append(
            {
                "index": index,
                "bitstring": bitstring_from_index(index, num_x_qubits),
                "status": "finite" if finite else "ed_failed",
                "true_cost": float(result.total_cost) if finite else None,
            }
        )
        if finite:
            cost = float(result.total_cost)
            train_indices.append(index)
            bitstrings.append(bitstring_from_index(index, num_x_qubits))
            costs.append(cost)
            cache[index] = ExactCandidateEvaluation(
                success=True,
                total_cost=cost,
                message=str(result.message),
                source="training_exact_cache",
            )
        if len(costs) >= int(train_sample_count):
            break
    if len(costs) < int(train_sample_count):
        raise RuntimeError("没有获得足够的有限训练 ED/LP 样本")
    return train_indices, bitstrings, costs, cache, trace, calls


def _validation_only_exact_landscape(
    instance,
    commitments: np.ndarray,
    *,
    algorithm_cache: dict[int, ExactCandidateEvaluation],
    num_x_qubits: int,
) -> tuple[dict[int, float], list[dict[str, object]], int]:
    evaluator = FixedCommitmentEvaluator(instance)
    true_costs: dict[int, float] = {
        int(index): float(record.total_cost)
        for index, record in algorithm_cache.items()
        if record.success and record.total_cost is not None
    }
    rows: list[dict[str, object]] = []
    calls = 0
    for index in range(commitments.shape[0]):
        if index in true_costs:
            rows.append(
                {
                    "index": int(index),
                    "bitstring": bitstring_from_index(index, num_x_qubits),
                    "status": "algorithm_exact_cache_reused_for_validation",
                    "true_cost": float(true_costs[index]),
                }
            )
            continue
        commitment = commitments[index]
        if not is_logic_feasible(instance, commitment):
            rows.append(
                {
                    "index": int(index),
                    "bitstring": bitstring_from_index(index, num_x_qubits),
                    "status": "logic_infeasible",
                    "true_cost": None,
                }
            )
            continue
        result = evaluator.evaluate(commitment)
        calls += 1
        finite = bool(result.success and np.isfinite(result.total_cost))
        if finite:
            true_costs[int(index)] = float(result.total_cost)
        rows.append(
            {
                "index": int(index),
                "bitstring": bitstring_from_index(index, num_x_qubits),
                "status": "finite" if finite else "ed_failed",
                "true_cost": float(result.total_cost) if finite else None,
            }
        )
    return true_costs, rows, calls


def _threshold_validation_rows(value_model, phase_model, threshold_history):
    rows: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for history_row in threshold_history:
        encoded_threshold = int(history_row["encoded_threshold"])
        update_number = int(history_row["update_number"])
        key = (update_number, encoded_threshold)
        if key in seen:
            continue
        seen.add(key)
        plan = ordinary_grover_validation_plan(
            value_model,
            encoded_threshold=encoded_threshold,
            max_validation_qubits=12,
        )
        direct = direct_float_marked_indices_for_validation(
            value_model,
            predict_cost=phase_model.predict_cost,
            encoded_threshold=encoded_threshold,
            max_validation_qubits=12,
        )
        rows.append(
            {
                "update_number": update_number,
                "true_threshold": float(history_row["true_threshold"]),
                "encoded_threshold": encoded_threshold,
                "validation_only_sparse_marked_indices": [
                    int(index) for index in plan.marked_indices
                ],
                "validation_only_sparse_marked_count": int(plan.marked_count),
                **marked_semantics_diagnostics(plan.marked_indices, direct),
            }
        )
    return rows


def _evaluate_window(
    *,
    source,
    window_start: int,
    selected_generator_indices: tuple[int, int],
    train_sample_count: int,
    fixed_point: FixedPointConfig,
    initialization_policy: str,
    seed: int,
    regularization: float,
    maxiter: int,
    bbht_config: BBHTConfig,
) -> dict[str, object]:
    horizon = 2
    num_x_qubits = len(selected_generator_indices) * horizon
    instance = time_window_instance(source, start=int(window_start), horizon=horizon)
    base_commitment = np.ones((len(instance.generators), horizon), dtype=int)
    commitments = embedded_selected_commitments(base_commitment, selected_generator_indices)
    (
        train_indices,
        train_bitstrings,
        train_costs,
        training_cache,
        training_trace,
        training_calls,
    ) = _collect_training_data(
        instance,
        commitments,
        train_sample_count=int(train_sample_count),
        num_x_qubits=num_x_qubits,
    )

    fit = fit_sparse_phase_vqc(
        bitstrings=train_bitstrings,
        costs=train_costs,
        num_generators=2,
        num_periods=2,
        generator_edges=((0, 1),),
        seed=int(seed) + int(window_start),
        regularization=float(regularization),
        maxiter=int(maxiter),
    )
    phase_model = fit.model
    value_model = quantize_sparse_phase_model(phase_model, fixed_point)
    initial_index = select_initial_incumbent(
        train_indices,
        training_cache,
        policy=initialization_policy,
        seed=int(seed) + int(window_start),
    )
    evaluator = FixedCommitmentEvaluator(instance)

    def evaluate_candidate(index: int) -> ExactCandidateEvaluation:
        commitment = commitments[int(index)]
        if not is_logic_feasible(instance, commitment):
            return ExactCandidateEvaluation(
                success=False,
                total_cost=None,
                message="logic infeasible before ED/LP",
                source="logic_infeasible_precheck",
            )
        result = evaluator.evaluate(commitment)
        finite = bool(result.success and np.isfinite(result.total_cost))
        return ExactCandidateEvaluation(
            success=finite,
            total_cost=float(result.total_cost) if finite else None,
            message=str(result.message),
            source="new_ed_lp_call",
        )

    run_config = BBHTConfig(
        lambda_factor=bbht_config.lambda_factor,
        max_trials=bbht_config.max_trials,
        max_oracle_calls=bbht_config.max_oracle_calls,
        max_new_ed_lp_calls=bbht_config.max_new_ed_lp_calls,
        max_threshold_updates=bbht_config.max_threshold_updates,
        max_consecutive_nonimproving_marked=(
            bbht_config.max_consecutive_nonimproving_marked
        ),
        max_same_encoded_threshold_updates=(
            bbht_config.max_same_encoded_threshold_updates
        ),
        shots_per_trial=1,
        seed=int(seed) + 10_000 * int(window_start),
    )
    bbht_result = run_sparse_vqc_bbht(
        value_model,
        initial_incumbent_index=initial_index,
        initial_exact_cache=training_cache,
        evaluate_candidate=evaluate_candidate,
        config=run_config,
        training_indices=train_indices,
    )

    true_costs, landscape_rows, validation_calls = _validation_only_exact_landscape(
        instance,
        commitments,
        algorithm_cache=bbht_result.exact_cache,
        num_x_qubits=num_x_qubits,
    )
    finite_indices = sorted(true_costs)
    true_global_index = int(min(finite_indices, key=lambda index: true_costs[index]))
    true_global_cost = float(true_costs[true_global_index])
    threshold_validation = _threshold_validation_rows(
        value_model,
        phase_model,
        bbht_result.threshold_history,
    )
    selected_names = [source.generators[index].name for index in selected_generator_indices]
    trial_elapsed = sum(float(row["elapsed_seconds"]) for row in bbht_result.trial_trace)
    maximum_trial_qubits = max(
        (int(row["total_qubits"]) for row in bbht_result.trial_trace),
        default=0,
    )
    maximum_trial_depth = max(
        (
            int(row["circuit_resources"]["depth"])
            for row in bbht_result.trial_trace
            if row.get("circuit_resources")
        ),
        default=0,
    )
    return {
        "window_start": int(window_start),
        "window_end_exclusive": int(window_start + horizon),
        "fixed_load": [float(value) for value in instance.fixed_load],
        "selected_generator_indices": [int(value) for value in selected_generator_indices],
        "selected_generator_names": selected_names,
        "bit_labels": [
            f"{selected_names[0]}_t0",
            f"{selected_names[0]}_t1",
            f"{selected_names[1]}_t0",
            f"{selected_names[1]}_t1",
        ],
        "train_indices": [int(index) for index in train_indices],
        "training_trace": training_trace,
        "training_ed_lp_calls": int(training_calls),
        "fit": fit.as_dict(),
        "phase_model": phase_model.as_dict(),
        "quantized_value_model": value_model.as_dict(),
        "initialization_policy": initialization_policy,
        "bbht": bbht_result.as_dict(),
        "algorithmic_ed_lp_calls": int(training_calls + bbht_result.new_ed_lp_calls),
        "validation_only": {
            "full_landscape_evaluated_after_search": True,
            "ed_lp_calls": int(validation_calls),
            "rows": landscape_rows,
            "true_global_optimum_index": int(true_global_index),
            "true_global_optimum_bitstring": bitstring_from_index(
                true_global_index, num_x_qubits
            ),
            "true_global_optimum_cost": float(true_global_cost),
            "final_incumbent_is_true_global_optimum": bool(
                bbht_result.final_incumbent_index == true_global_index
            ),
            "final_true_optimality_gap": float(
                bbht_result.final_incumbent_true_cost - true_global_cost
            ),
            "threshold_marked_set_diagnostics": threshold_validation,
        },
        "resources": {
            "trial_mps_elapsed_seconds_total": float(trial_elapsed),
            "maximum_trial_qubits": int(maximum_trial_qubits),
            "maximum_trial_top_level_depth": int(maximum_trial_depth),
        },
        "uses_marked_count_in_bbht": False,
        "uses_full_state_enumeration_in_bbht": False,
        "full_state_enumeration_used_after_search_for_validation_only": True,
        "threshold_updates_require_true_ed_lp_improvement": True,
        "uses_adaptive_threshold": True,
        "uses_bbht": True,
    }


def run(
    *,
    instance_path: Path,
    results_path: Path,
    selected_generator_indices: tuple[int, int] = (0, 5),
    window_starts: tuple[int, ...] = (0, 1, 2),
    train_sample_count: int = 8,
    fractional_bits: int = 2,
    cost_unit: float = 1000.0,
    initialization_policy: str = "first",
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
) -> dict[str, object]:
    if len(selected_generator_indices) != 2:
        raise ValueError("本实验只验证两台可变机组")
    if len(set(window_starts)) != len(window_starts):
        raise ValueError("window_starts 不能重复")
    if train_sample_count <= 0 or train_sample_count > 16:
        raise ValueError("train_sample_count 必须位于 1 到 16")

    source = load_uc_instance(instance_path)
    fixed_point = FixedPointConfig(
        fractional_bits=int(fractional_bits),
        unit=float(cost_unit),
        rounding="nearest",
    )
    base_bbht_config = BBHTConfig(
        lambda_factor=float(lambda_factor),
        max_trials=int(max_trials),
        max_oracle_calls=int(max_oracle_calls),
        max_new_ed_lp_calls=int(max_new_ed_lp_calls),
        max_threshold_updates=int(max_threshold_updates),
        max_consecutive_nonimproving_marked=int(
            max_consecutive_nonimproving_marked
        ),
        max_same_encoded_threshold_updates=int(max_same_encoded_threshold_updates),
        shots_per_trial=1,
        seed=int(seed),
    )
    scenarios = [
        _evaluate_window(
            source=source,
            window_start=int(window_start),
            selected_generator_indices=selected_generator_indices,
            train_sample_count=int(train_sample_count),
            fixed_point=fixed_point,
            initialization_policy=initialization_policy,
            seed=int(seed),
            regularization=float(regularization),
            maxiter=int(maxiter),
            bbht_config=base_bbht_config,
        )
        for window_start in window_starts
    ]
    summary = {
        "method": "sparse VQC adaptive BBHT with true ED/LP threshold updates",
        "research_scope": (
            "研究内容1的未知marked-count BBHT、自适应真实incumbent和threshold下降闭环"
        ),
        "source_instance": str(instance_path),
        "selected_generator_indices": [int(value) for value in selected_generator_indices],
        "selected_generator_names": [
            source.generators[index].name for index in selected_generator_indices
        ],
        "window_starts": [int(value) for value in window_starts],
        "train_sample_count": int(train_sample_count),
        "initialization_policy": initialization_policy,
        "fixed_point": {
            "fractional_bits": int(fixed_point.fractional_bits),
            "cost_unit": float(fixed_point.unit),
            "quantum": float(fixed_point.quantum),
        },
        "bbht_config": {
            "lambda_factor": float(base_bbht_config.lambda_factor),
            "max_trials": int(base_bbht_config.max_trials),
            "max_oracle_calls": int(base_bbht_config.max_oracle_calls),
            "max_new_ed_lp_calls": int(base_bbht_config.max_new_ed_lp_calls),
            "max_threshold_updates": int(base_bbht_config.max_threshold_updates),
            "max_consecutive_nonimproving_marked": int(
                base_bbht_config.max_consecutive_nonimproving_marked
            ),
            "max_same_encoded_threshold_updates": int(
                base_bbht_config.max_same_encoded_threshold_updates
            ),
            "shots_per_trial": 1,
            "seed": int(seed),
        },
        "uses_bbht": True,
        "uses_adaptive_threshold": True,
        "uses_marked_count_in_algorithm": False,
        "threshold_updates_require_true_ed_lp_improvement": True,
        "scenarios": scenarios,
        "notes": [
            "每个BBHT trial只执行1 shot，并从实际Aer MPS测量得到一个候选。",
            "VQC参数在搜索期间固定；真实incumbent改善后只更新IntegerComparator threshold。",
            "训练ED/LP与后续候选ED/LP共享exact cache，重复候选不重复求解。",
            "保守整数下界可以在不枚举状态时证明严格marked集合为空。",
            "完整16状态与真实全局最优仅在BBHT结束后用于2×2验证，不控制搜索。",
            "当前结果不用于声称量子优势。",
        ],
    }
    write_strict_json(results_path, summary)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="case14 三个2×2稀疏VQC自适应BBHT实验")
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_2x2_sparse_vqc_bbht.json"),
    )
    parser.add_argument("--selected-generators", type=_parse_int_tuple, default=(0, 5))
    parser.add_argument("--window-starts", type=_parse_int_tuple, default=(0, 1, 2))
    parser.add_argument("--train-sample-count", type=int, default=8)
    parser.add_argument("--fractional-bits", type=int, default=2)
    parser.add_argument("--cost-unit", type=float, default=1000.0)
    parser.add_argument(
        "--initialization-policy",
        choices=("first", "random", "best-training"),
        default="first",
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
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    summary = run(
        instance_path=args.instance,
        results_path=args.results,
        selected_generator_indices=tuple(args.selected_generators),
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
    )
    compact = {
        "method": summary["method"],
        "scenarios": [
            {
                "window_start": scenario["window_start"],
                "initial_incumbent": scenario["bbht"]["initial_incumbent_index"],
                "final_incumbent": scenario["bbht"]["final_incumbent_index"],
                "initial_true_cost": scenario["bbht"]["initial_incumbent_true_cost"],
                "final_true_cost": scenario["bbht"]["final_incumbent_true_cost"],
                "threshold_updates": scenario["bbht"]["counters"][
                    "threshold_updates"
                ],
                "trials": scenario["bbht"]["counters"]["circuit_executions"],
                "oracle_calls": scenario["bbht"]["counters"][
                    "total_oracle_calls"
                ],
                "stop_reason": scenario["bbht"]["stop_reason"],
                "true_global_optimum_index": scenario["validation_only"][
                    "true_global_optimum_index"
                ],
                "final_is_global_optimum": scenario["validation_only"][
                    "final_incumbent_is_true_global_optimum"
                ],
            }
            for scenario in summary["scenarios"]
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
