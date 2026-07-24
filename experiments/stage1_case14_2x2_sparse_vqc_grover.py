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

from qubit_value_function.commitment import is_logic_feasible  # noqa: E402
from qubit_value_function.coherent_phase_value import quantize_sparse_phase_model  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.experiment_utils import (  # noqa: E402
    embedded_selected_commitments,
    time_window_instance,
    write_strict_json,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig  # noqa: E402
from qubit_value_function.gate_level_oracle import (  # noqa: E402
    bitstring_from_index,
    circuit_resource_summary,
)
from qubit_value_function.sparse_phase_vqc import fit_sparse_phase_vqc  # noqa: E402
from qubit_value_function.sparse_vqc_grover import (  # noqa: E402
    build_sparse_vqc_grover_circuit,
    direct_float_marked_indices_for_validation,
    execute_sparse_vqc_grover_mps,
    marked_semantics_diagnostics,
    ordinary_grover_validation_plan,
    select_measured_candidate,
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
) -> tuple[list[int], list[str], list[float], list[dict[str, object]], int]:
    evaluator = FixedCommitmentEvaluator(instance)
    train_indices: list[int] = []
    bitstrings: list[str] = []
    costs: list[float] = []
    trace: list[dict[str, object]] = []
    calls = 0
    for raw_index in REPRESENTATIVE_INDEX_ORDER:
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
            train_indices.append(index)
            bitstrings.append(bitstring_from_index(index, num_x_qubits))
            costs.append(float(result.total_cost))
        if len(costs) >= int(train_sample_count):
            break
    if len(costs) < int(train_sample_count):
        raise RuntimeError("没有获得足够的有限训练 ED/LP 样本")
    return train_indices, bitstrings, costs, trace, calls


def _verify_candidate_without_threshold_update(
    *,
    evaluator: FixedCommitmentEvaluator,
    commitments: np.ndarray,
    candidate_index: int,
    incumbent_true_cost: float,
    encoded_threshold: int,
) -> dict[str, object]:
    """Run exact ED/LP once and explicitly keep the ordinary-Grover threshold fixed."""

    result = evaluator.evaluate(commitments[int(candidate_index)])
    finite = bool(result.success and np.isfinite(result.total_cost))
    true_cost = float(result.total_cost) if finite else None
    would_improve = bool(finite and true_cost < float(incumbent_true_cost))
    return {
        "candidate_ed_lp_success": finite,
        "candidate_true_ed_lp_cost": true_cost,
        "candidate_ed_lp_message": str(result.message),
        "incumbent_true_cost": float(incumbent_true_cost),
        "would_improve_incumbent": would_improve,
        "encoded_threshold_before": int(encoded_threshold),
        "encoded_threshold_after": int(encoded_threshold),
        "threshold_updated": False,
        "candidate_ed_lp_calls": 1,
    }


def _evaluate_window(
    *,
    source,
    window_start: int,
    selected_generator_indices: tuple[int, int],
    train_sample_count: int,
    fixed_point: FixedPointConfig,
    shots: int,
    seed: int,
    regularization: float,
    maxiter: int,
) -> dict[str, object]:
    horizon = 2
    num_x_qubits = len(selected_generator_indices) * horizon
    instance = time_window_instance(source, start=int(window_start), horizon=horizon)
    base_commitment = np.ones((len(instance.generators), horizon), dtype=int)
    commitments = embedded_selected_commitments(base_commitment, selected_generator_indices)
    train_indices, train_bitstrings, train_costs, training_trace, training_calls = (
        _collect_training_data(
            instance,
            commitments,
            train_sample_count=int(train_sample_count),
            num_x_qubits=num_x_qubits,
        )
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
    evaluator = FixedCommitmentEvaluator(instance)

    training_cost_by_index = {
        int(index): float(cost) for index, cost in zip(train_indices, train_costs)
    }
    first_incumbent_index = int(train_indices[0])
    best_incumbent_index = int(
        min(train_indices, key=lambda index: training_cost_by_index[int(index)])
    )
    threshold_specs = (
        (
            "first_training_incumbent",
            first_incumbent_index,
            training_cost_by_index[first_incumbent_index],
        ),
        (
            "best_training_incumbent",
            best_incumbent_index,
            training_cost_by_index[best_incumbent_index],
        ),
    )

    threshold_rows: list[dict[str, object]] = []
    for threshold_number, (label, incumbent_index, incumbent_cost) in enumerate(threshold_specs):
        encoded_threshold = int(fixed_point.encode(incumbent_cost))
        plan = ordinary_grover_validation_plan(
            value_model,
            encoded_threshold=encoded_threshold,
        )
        direct_marked = direct_float_marked_indices_for_validation(
            value_model,
            predict_cost=phase_model.predict_cost,
            encoded_threshold=encoded_threshold,
        )
        semantics = marked_semantics_diagnostics(plan.marked_indices, direct_marked)
        row: dict[str, Any] = {
            "label": label,
            "status": "ready" if plan.marked_count > 0 else "no_marked_state",
            "incumbent_index": int(incumbent_index),
            "incumbent_bitstring": bitstring_from_index(incumbent_index, num_x_qubits),
            "real_incumbent_threshold": float(incumbent_cost),
            "encoded_threshold": int(encoded_threshold),
            **semantics,
            "marked_count": int(plan.marked_count),
            "marked_count_known_for_validation_only": True,
            "grover_iterations": int(plan.iterations),
            "initial_marked_probability": float(plan.initial_marked_probability),
            "measured_marked_probability": None,
            "amplification_factor": None,
            "raw_counts": None,
            "x_counts": None,
            "auxiliary_zero_probability": None,
            "candidate": None,
            "circuit_resources": None,
            "elapsed_seconds": None,
            "estimated_statevector_memory_gb": None,
            "threshold_updated": False,
        }
        if plan.marked_count == 0:
            threshold_rows.append(row)
            continue

        circuit = build_sparse_vqc_grover_circuit(
            value_model,
            encoded_threshold=encoded_threshold,
            iterations=plan.iterations,
        )
        execution = execute_sparse_vqc_grover_mps(
            circuit,
            num_x_qubits=value_model.num_x_qubits,
            shots=int(shots),
            seed=int(seed) + 100 * int(window_start) + int(threshold_number),
        )
        measured_marked_probability = float(
            sum(execution.x_probabilities[index] for index in plan.marked_indices)
        )
        candidate = select_measured_candidate(
            execution.x_counts,
            num_x_qubits=value_model.num_x_qubits,
            allowed_indices=plan.marked_indices,
            observed_indices=train_indices,
        )
        candidate_row = None
        if candidate is not None:
            verification = _verify_candidate_without_threshold_update(
                evaluator=evaluator,
                commitments=commitments,
                candidate_index=candidate.index,
                incumbent_true_cost=incumbent_cost,
                encoded_threshold=encoded_threshold,
            )
            candidate_row = {
                "candidate_index": int(candidate.index),
                "candidate_bitstring": candidate.bitstring,
                "candidate_count": int(candidate.count),
                "candidate_probability": float(candidate.probability),
                "candidate_selection_source": "measured_gate_level_counts",
                "candidate_was_training_state": bool(candidate.was_observed),
                **verification,
            }

        row.update(
            {
                "status": "completed",
                "measured_marked_probability": measured_marked_probability,
                "amplification_factor": float(
                    measured_marked_probability / plan.initial_marked_probability
                ),
                "raw_counts": execution.raw_counts,
                "x_counts": execution.x_counts,
                "auxiliary_zero_probability": float(
                    execution.auxiliary_zero_probability
                ),
                "candidate": candidate_row,
                "circuit_resources": circuit_resource_summary(circuit, decompose_reps=1),
                "elapsed_seconds": float(execution.elapsed_seconds),
                "estimated_statevector_memory_gb": float(
                    execution.estimated_statevector_memory_gb
                ),
            }
        )
        threshold_rows.append(row)

    selected_names = [source.generators[index].name for index in selected_generator_indices]
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
        "quantized_value_model": value_model.as_dict(),
        "threshold_experiments": threshold_rows,
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
    shots: int = 4096,
    seed: int = 0,
    regularization: float = 1e-4,
    maxiter: int = 300,
) -> dict[str, object]:
    if len(selected_generator_indices) != 2:
        raise ValueError("本实验只验证两台可变机组")
    if len(set(window_starts)) != len(window_starts):
        raise ValueError("window_starts 不能重复")
    if train_sample_count <= 0 or train_sample_count > 16:
        raise ValueError("train_sample_count 必须位于 1 到 16")
    if int(shots) <= 0:
        raise ValueError("shots 必须为正数")

    source = load_uc_instance(instance_path)
    fixed_point = FixedPointConfig(
        fractional_bits=int(fractional_bits),
        unit=float(cost_unit),
        rounding="nearest",
    )
    scenarios = [
        _evaluate_window(
            source=source,
            window_start=int(window_start),
            selected_generator_indices=selected_generator_indices,
            train_sample_count=int(train_sample_count),
            fixed_point=fixed_point,
            shots=int(shots),
            seed=int(seed),
            regularization=float(regularization),
            maxiter=int(maxiter),
        )
        for window_start in window_starts
    ]
    summary = {
        "method": "sparse VQC coherent-value ordinary Grover with Aer MPS measurements",
        "research_scope": (
            "研究内容1的固定threshold普通Grover振幅放大验证；"
            "尚未实现BBHT或多轮自适应threshold更新"
        ),
        "source_instance": str(instance_path),
        "selected_generator_indices": [int(value) for value in selected_generator_indices],
        "selected_generator_names": [
            source.generators[index].name for index in selected_generator_indices
        ],
        "window_starts": [int(value) for value in window_starts],
        "train_sample_count": int(train_sample_count),
        "fixed_point": {
            "fractional_bits": int(fixed_point.fractional_bits),
            "scale": int(fixed_point.scale),
            "cost_unit": float(fixed_point.unit),
            "quantum": float(fixed_point.quantum),
        },
        "shots": int(shots),
        "seed": int(seed),
        "regularization": float(regularization),
        "maxiter": int(maxiter),
        "simulation_method": "matrix_product_state",
        "uses_sparse_integer_oracle_semantics": True,
        "uses_direct_float_marked_set_for_diagnostics_only": True,
        "uses_full_state_lookup": False,
        "marked_count_known_for_validation_only": True,
        "uses_ordinary_grover": True,
        "uses_adaptive_grover": False,
        "updates_threshold": False,
        "scenarios": scenarios,
        "notes": [
            "普通Grover的迭代次数在2×2验收中使用已知marked count选择，核心builder不枚举状态。",
            "量子oracle以逐项量化后的sparse integer model为正式成本语义。",
            "direct-rounded float marked set只用于量化诊断，不修正电路或候选。",
            "候选只能来自实际Aer MPS测量counts，并由真实ED/LP校验。",
            "本阶段threshold保持固定；即使候选真实成本更低，也只记录would_improve_incumbent。",
            "下一阶段才实现BBHT、自适应threshold更新、预算和停止条件。",
        ],
    }
    write_strict_json(results_path, summary)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="case14三个2×2稀疏VQC普通Grover验证")
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_2x2_sparse_vqc_grover.json"),
    )
    parser.add_argument("--selected-generators", type=_parse_int_tuple, default=(0, 5))
    parser.add_argument("--window-starts", type=_parse_int_tuple, default=(0, 1, 2))
    parser.add_argument("--train-sample-count", type=int, default=8)
    parser.add_argument("--fractional-bits", type=int, default=2)
    parser.add_argument("--cost-unit", type=float, default=1000.0)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--maxiter", type=int, default=300)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if len(args.selected_generators) != 2:
        raise ValueError("--selected-generators 必须恰好包含两个索引")
    summary = run(
        instance_path=args.instance,
        results_path=args.results,
        selected_generator_indices=tuple(args.selected_generators),
        window_starts=tuple(args.window_starts),
        train_sample_count=args.train_sample_count,
        fractional_bits=args.fractional_bits,
        cost_unit=args.cost_unit,
        shots=args.shots,
        seed=args.seed,
        regularization=args.regularization,
        maxiter=args.maxiter,
    )
    compact = {
        "method": summary["method"],
        "scenarios": [
            {
                "window_start": scenario["window_start"],
                "thresholds": [
                    {
                        "label": row["label"],
                        "status": row["status"],
                        "marked_count": row["marked_count"],
                        "iterations": row["grover_iterations"],
                        "initial_marked_probability": row[
                            "initial_marked_probability"
                        ],
                        "measured_marked_probability": row[
                            "measured_marked_probability"
                        ],
                        "auxiliary_zero_probability": row[
                            "auxiliary_zero_probability"
                        ],
                        "candidate": row["candidate"],
                    }
                    for row in scenario["threshold_experiments"]
                ],
            }
            for scenario in summary["scenarios"]
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
