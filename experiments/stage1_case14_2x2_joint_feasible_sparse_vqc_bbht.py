from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_2x2_sparse_vqc_bbht import (  # noqa: E402
    _collect_training_data,
    _validation_only_exact_landscape,
)
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
    compile_logic_feasibility_spec,
)
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


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("至少需要一个整数")
    return values


def _threshold_validation_rows(
    value_model,
    phase_model,
    feasibility_spec,
    threshold_history,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for history_row in threshold_history:
        encoded_threshold = int(history_row["encoded_threshold"])
        cost_only = ordinary_grover_validation_plan(
            value_model,
            encoded_threshold=encoded_threshold,
            max_validation_qubits=12,
        )
        joint = ordinary_grover_validation_plan(
            value_model,
            encoded_threshold=encoded_threshold,
            max_validation_qubits=12,
            feasibility_spec=feasibility_spec,
        )
        direct_joint = direct_float_marked_indices_for_validation(
            value_model,
            predict_cost=phase_model.predict_cost,
            encoded_threshold=encoded_threshold,
            max_validation_qubits=12,
            feasibility_spec=feasibility_spec,
        )
        rows.append(
            {
                "update_number": int(history_row["update_number"]),
                "true_threshold": float(history_row["true_threshold"]),
                "encoded_threshold": encoded_threshold,
                "validation_only_cost_better_indices_before_feasibility": [
                    int(index) for index in cost_only.marked_indices
                ],
                "validation_only_joint_feasible_better_indices": [
                    int(index) for index in joint.marked_indices
                ],
                "validation_only_hard_infeasible_better_indices_removed": sorted(
                    set(cost_only.marked_indices) - set(joint.marked_indices)
                ),
                **marked_semantics_diagnostics(joint.marked_indices, direct_joint),
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
    feasibility_spec = compile_logic_feasibility_spec(
        instance,
        selected_generator_indices=selected_generator_indices,
        base_commitment=base_commitment,
    )

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
                message="classical defense rejected a hard-logic-infeasible candidate",
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
        minimum_auxiliary_zero_probability=(
            bbht_config.minimum_auxiliary_zero_probability
        ),
        seed=int(seed) + 10_000 * int(window_start),
    )
    bbht_result = run_sparse_vqc_bbht(
        value_model,
        initial_incumbent_index=initial_index,
        initial_exact_cache=training_cache,
        evaluate_candidate=evaluate_candidate,
        config=run_config,
        feasibility_spec=feasibility_spec,
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
    hard_feasible_indices = [
        int(index)
        for index in range(commitments.shape[0])
        if feasibility_spec.is_feasible(
            tuple((int(index) >> bit) & 1 for bit in range(num_x_qubits))
        )
    ]
    threshold_validation = _threshold_validation_rows(
        value_model,
        phase_model,
        feasibility_spec,
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
    minimum_auxiliary_zero = min(
        (
            float(row["auxiliary_zero_probability"])
            for row in bbht_result.trial_trace
        ),
        default=1.0,
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
        "hard_logic_feasibility_spec": feasibility_spec.as_dict(),
        "initialization_policy": initialization_policy,
        "bbht": bbht_result.as_dict(),
        "algorithmic_actual_ed_lp_solves": int(
            training_calls + bbht_result.actual_ed_lp_solves
        ),
        "algorithmic_new_exact_evaluation_attempts": int(
            bbht_result.new_exact_evaluation_attempts
        ),
        "validation_only": {
            "full_landscape_evaluated_after_search": True,
            "ed_lp_calls": int(validation_calls),
            "rows": landscape_rows,
            "hard_logic_feasible_indices": hard_feasible_indices,
            "hard_logic_infeasible_indices": sorted(
                set(range(commitments.shape[0])) - set(hard_feasible_indices)
            ),
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
            "minimum_auxiliary_zero_probability": float(minimum_auxiliary_zero),
        },
        "uses_hard_logic_feasibility_oracle": True,
        "uses_joint_feasible_and_better_phase_oracle": True,
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
    minimum_auxiliary_zero_probability: float = 1.0 - 1e-12,
) -> dict[str, object]:
    if len(selected_generator_indices) != 2:
        raise ValueError("本实验只验证两台可变机组")
    source = load_uc_instance(instance_path)
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
        max_same_encoded_threshold_updates=int(max_same_encoded_threshold_updates),
        shots_per_trial=1,
        minimum_auxiliary_zero_probability=float(
            minimum_auxiliary_zero_probability
        ),
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
            bbht_config=config,
        )
        for window_start in window_starts
    ]
    summary: dict[str, Any] = {
        "method": "joint hard-logic-feasible and sparse-VQC-better adaptive BBHT",
        "research_scope": (
            "研究内容1的硬UC逻辑可行性oracle、稀疏VQC成本阈值oracle与真实ED/LP阈值下降闭环"
        ),
        "source_instance": str(instance_path),
        "selected_generator_indices": [int(value) for value in selected_generator_indices],
        "window_starts": [int(value) for value in window_starts],
        "train_sample_count": int(train_sample_count),
        "initialization_policy": initialization_policy,
        "fixed_point": {
            "fractional_bits": int(fixed_point.fractional_bits),
            "cost_unit": float(fixed_point.unit),
            "quantum": float(fixed_point.quantum),
        },
        "bbht_config": config.__dict__,
        "uses_hard_logic_feasibility_oracle": True,
        "uses_joint_feasible_and_better_phase_oracle": True,
        "uses_marked_count_in_algorithm": False,
        "threshold_updates_require_true_ed_lp_improvement": True,
        "scenarios": scenarios,
        "notes": [
            "硬逻辑oracle精确编码must-run、初始剩余最小开停机时间和时域内最小开停机约束。",
            "连续ED/LP、精确连续爬坡和网络安全约束仍由测量后的经典求解器最终验证。",
            "每个BBHT trial只执行1 shot；辅助位非零的测量被拒绝，不进入候选验证。",
            "完整16状态、联合marked集合和真实全局最优仅在搜索结束后用于验证。",
        ],
    }
    write_strict_json(results_path, summary)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="case14 三个2×2联合可行性稀疏VQC BBHT实验")
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_2x2_joint_feasible_sparse_vqc_bbht.json"),
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
    parser.add_argument("--minimum-auxiliary-zero-probability", type=float, default=1.0 - 1e-12)
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
        minimum_auxiliary_zero_probability=(
            args.minimum_auxiliary_zero_probability
        ),
    )
    compact = {
        "method": summary["method"],
        "scenarios": [
            {
                "window_start": scenario["window_start"],
                "initial_incumbent": scenario["bbht"]["initial_incumbent_index"],
                "final_incumbent": scenario["bbht"]["final_incumbent_index"],
                "threshold_updates": scenario["bbht"]["counters"]["threshold_updates"],
                "trials": scenario["bbht"]["counters"]["circuit_executions"],
                "actual_ed_lp_solves": scenario["bbht"]["counters"]["actual_ed_lp_solves"],
                "auxiliary_rejections": scenario["bbht"]["counters"]["auxiliary_syndrome_rejections"],
                "stop_reason": scenario["bbht"]["stop_reason"],
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
