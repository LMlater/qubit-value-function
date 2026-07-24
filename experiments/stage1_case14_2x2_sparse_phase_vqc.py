from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.commitment import is_logic_feasible  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.experiment_utils import (  # noqa: E402
    embedded_selected_commitments,
    time_window_instance,
    write_strict_json,
)
from qubit_value_function.gate_level_oracle import (  # noqa: E402
    bitstring_from_index,
    circuit_resource_summary,
)
from qubit_value_function.sparse_phase_vqc import (  # noqa: E402
    basis_phase_factor_from_statevector,
    build_hadamard_readout_circuit,
    build_phase_circuit,
    fit_sparse_phase_vqc,
    hadamard_phase_expectation,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


REPRESENTATIVE_INDEX_ORDER = (0, 15, 3, 12, 5, 10, 6, 9, 1, 2, 4, 8, 7, 11, 13, 14)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("至少需要一个整数")
    return values


def _pairwise_ranking_accuracy(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float | None:
    correct = 0
    total = 0
    for first in range(len(true_values)):
        for second in range(first + 1, len(true_values)):
            true_difference = float(true_values[first] - true_values[second])
            if np.isclose(true_difference, 0.0):
                continue
            predicted_difference = float(predicted_values[first] - predicted_values[second])
            total += 1
            if true_difference * predicted_difference > 0.0:
                correct += 1
    if total == 0:
        return None
    return float(correct / total)


def _regression_metrics(
    indices: Iterable[int],
    *,
    true_costs: dict[int, float],
    predicted_costs: dict[int, float],
    constant_baseline: float,
) -> dict[str, object]:
    selected = [int(index) for index in indices if int(index) in true_costs]
    if not selected:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "constant_baseline_mae": None,
            "pairwise_ranking_accuracy": None,
            "true_top1_index": None,
            "predicted_top1_index": None,
            "top1_match": None,
        }

    true_values = np.array([true_costs[index] for index in selected], dtype=float)
    predicted_values = np.array([predicted_costs[index] for index in selected], dtype=float)
    errors = predicted_values - true_values
    true_top1 = selected[int(np.argmin(true_values))]
    predicted_top1 = selected[int(np.argmin(predicted_values))]
    return {
        "count": int(len(selected)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "constant_baseline_mae": float(
            np.mean(np.abs(true_values - float(constant_baseline)))
        ),
        "pairwise_ranking_accuracy": _pairwise_ranking_accuracy(
            true_values,
            predicted_values,
        ),
        "true_top1_index": int(true_top1),
        "predicted_top1_index": int(predicted_top1),
        "top1_match": bool(true_top1 == predicted_top1),
    }


def _evaluate_window(
    *,
    source,
    window_start: int,
    horizon: int,
    selected_generator_indices: tuple[int, int],
    train_sample_count: int,
    seed: int,
    regularization: float,
    maxiter: int,
) -> dict[str, object]:
    instance = time_window_instance(source, start=window_start, horizon=horizon)
    base_commitment = np.ones((len(instance.generators), horizon), dtype=int)
    commitments = embedded_selected_commitments(base_commitment, selected_generator_indices)
    evaluator = FixedCommitmentEvaluator(instance)

    training_trace: list[dict[str, object]] = []
    training_costs: dict[int, float] = {}
    evaluated_indices: set[int] = set()
    training_ed_lp_calls = 0

    for index in REPRESENTATIVE_INDEX_ORDER:
        index = int(index)
        commitment = commitments[index]
        if not is_logic_feasible(instance, commitment):
            training_trace.append(
                {
                    "index": index,
                    "bitstring": bitstring_from_index(index, 4),
                    "status": "skipped_logic_infeasible",
                }
            )
            continue

        result = evaluator.evaluate(commitment)
        evaluated_indices.add(index)
        training_ed_lp_calls += 1
        finite = bool(result.success and np.isfinite(result.total_cost))
        training_trace.append(
            {
                "index": index,
                "bitstring": bitstring_from_index(index, 4),
                "status": "finite" if finite else "ed_failed",
                "true_cost": float(result.total_cost) if finite else None,
                "dispatch_cost": float(result.dispatch_cost) if finite else None,
                "startup_cost": float(result.startup_cost) if finite else None,
                "balance_penalty": float(result.balance_penalty) if finite else None,
                "reserve_penalty": float(result.reserve_penalty) if finite else None,
            }
        )
        if finite:
            training_costs[index] = float(result.total_cost)
        if len(training_costs) >= int(train_sample_count):
            break

    if len(training_costs) < int(train_sample_count):
        raise RuntimeError(
            f"window_start={window_start} 没有足够的有限训练样本："
            f"{len(training_costs)} < {train_sample_count}"
        )

    train_indices = list(training_costs)
    train_bitstrings = [bitstring_from_index(index, 4) for index in train_indices]
    train_values = [training_costs[index] for index in train_indices]

    # Hidden ED/LP labels are evaluated only after model fitting.
    fit_result = fit_sparse_phase_vqc(
        bitstrings=train_bitstrings,
        costs=train_values,
        num_generators=2,
        num_periods=horizon,
        generator_edges=((0, 1),),
        seed=seed + window_start,
        regularization=regularization,
        maxiter=maxiter,
    )
    model = fit_result.model

    hidden_trace: list[dict[str, object]] = []
    hidden_costs: dict[int, float] = {}
    hidden_ed_lp_calls = 0
    for index in range(commitments.shape[0]):
        if index in training_costs:
            continue
        commitment = commitments[index]
        if not is_logic_feasible(instance, commitment):
            hidden_trace.append(
                {
                    "index": int(index),
                    "bitstring": bitstring_from_index(index, 4),
                    "status": "logic_infeasible",
                }
            )
            continue
        if index in evaluated_indices:
            hidden_trace.append(
                {
                    "index": int(index),
                    "bitstring": bitstring_from_index(index, 4),
                    "status": "previous_ed_failure",
                }
            )
            continue

        result = evaluator.evaluate(commitment)
        hidden_ed_lp_calls += 1
        finite = bool(result.success and np.isfinite(result.total_cost))
        hidden_trace.append(
            {
                "index": int(index),
                "bitstring": bitstring_from_index(index, 4),
                "status": "finite" if finite else "ed_failed",
                "true_cost": float(result.total_cost) if finite else None,
                "dispatch_cost": float(result.dispatch_cost) if finite else None,
                "startup_cost": float(result.startup_cost) if finite else None,
                "balance_penalty": float(result.balance_penalty) if finite else None,
                "reserve_penalty": float(result.reserve_penalty) if finite else None,
            }
        )
        if finite:
            hidden_costs[int(index)] = float(result.total_cost)

    true_costs = {**training_costs, **hidden_costs}
    predicted_costs = {
        index: float(model.predict_cost(bitstring_from_index(index, 4)))
        for index in range(commitments.shape[0])
    }
    constant_baseline = float(np.mean(np.array(train_values, dtype=float)))

    phase_rows: list[dict[str, object]] = []
    max_basis_error = 0.0
    max_hadamard_error = 0.0
    for index in range(commitments.shape[0]):
        bitstring = bitstring_from_index(index, 4)
        analytic = model.phase_factor(bitstring)
        basis_value = basis_phase_factor_from_statevector(model, bitstring)
        hadamard_value = hadamard_phase_expectation(model, bitstring)
        basis_error = float(abs(analytic - basis_value))
        hadamard_error = float(abs(analytic - hadamard_value))
        max_basis_error = max(max_basis_error, basis_error)
        max_hadamard_error = max(max_hadamard_error, hadamard_error)
        phase_rows.append(
            {
                "index": int(index),
                "bitstring": bitstring,
                "unwrapped_phase_cycles": float(model.unwrapped_phase(bitstring)),
                "wrapped_phase_cycles": float(model.wrapped_phase(bitstring)),
                "analytic_phase": {
                    "real": float(np.real(analytic)),
                    "imag": float(np.imag(analytic)),
                },
                "basis_statevector_phase": {
                    "real": float(np.real(basis_value)),
                    "imag": float(np.imag(basis_value)),
                },
                "hadamard_expectation": {
                    "real": float(np.real(hadamard_value)),
                    "imag": float(np.imag(hadamard_value)),
                },
                "basis_phase_error": basis_error,
                "hadamard_phase_error": hadamard_error,
            }
        )

    prediction_rows = []
    for index in range(commitments.shape[0]):
        split = (
            "training"
            if index in training_costs
            else (
                "hidden_evaluation"
                if index in hidden_costs
                else "unavailable_or_infeasible"
            )
        )
        prediction_rows.append(
            {
                "index": int(index),
                "bitstring": bitstring_from_index(index, 4),
                "split": split,
                "predicted_cost": float(predicted_costs[index]),
                "true_cost": float(true_costs[index]) if index in true_costs else None,
            }
        )

    hidden_indices = sorted(hidden_costs)
    finite_indices = sorted(true_costs)
    phase_circuit = build_phase_circuit(model, "0000")
    hadamard_circuit = build_hadamard_readout_circuit(model, "0000")
    selected_names = [source.generators[index].name for index in selected_generator_indices]
    return {
        "window_start": int(window_start),
        "window_end_exclusive": int(window_start + horizon),
        "horizon": int(horizon),
        "fixed_load": [float(value) for value in instance.fixed_load],
        "selected_generator_indices": [int(value) for value in selected_generator_indices],
        "selected_generator_names": selected_names,
        "bit_labels": [
            f"{selected_names[0]}_t0",
            f"{selected_names[0]}_t1",
            f"{selected_names[1]}_t0",
            f"{selected_names[1]}_t1",
        ],
        "training_trace": training_trace,
        "hidden_evaluation_trace": hidden_trace,
        "train_indices": [int(index) for index in train_indices],
        "hidden_indices": [int(index) for index in hidden_indices],
        "fit": fit_result.as_dict(),
        "model": model.as_dict(),
        "metrics": {
            "training": _regression_metrics(
                train_indices,
                true_costs=true_costs,
                predicted_costs=predicted_costs,
                constant_baseline=constant_baseline,
            ),
            "hidden": _regression_metrics(
                hidden_indices,
                true_costs=true_costs,
                predicted_costs=predicted_costs,
                constant_baseline=constant_baseline,
            ),
            "all_finite_evaluation_only": _regression_metrics(
                finite_indices,
                true_costs=true_costs,
                predicted_costs=predicted_costs,
                constant_baseline=constant_baseline,
            ),
        },
        "phase_encoding_validation": {
            "max_basis_statevector_phase_error": float(max_basis_error),
            "max_hadamard_phase_error": float(max_hadamard_error),
            "rows": phase_rows,
        },
        "prediction_table": prediction_rows,
        "resources": {
            "phase_circuit": circuit_resource_summary(phase_circuit, decompose_reps=1),
            "hadamard_readout_circuit": circuit_resource_summary(
                hadamard_circuit,
                decompose_reps=1,
            ),
        },
        "training_ed_lp_calls": int(training_ed_lp_calls),
        "hidden_evaluation_ed_lp_calls": int(hidden_ed_lp_calls),
        "hidden_labels_used_for_training": False,
        "full_state_lookup_used": False,
        "full_enumeration_used_for_evaluation_only": True,
    }


def run(
    *,
    instance_path: Path,
    results_path: Path,
    selected_generator_indices: tuple[int, int] = (0, 5),
    window_starts: tuple[int, ...] = (0, 1, 2),
    horizon: int = 2,
    train_sample_count: int = 8,
    seed: int = 0,
    regularization: float = 1e-4,
    maxiter: int = 300,
) -> dict[str, object]:
    if len(selected_generator_indices) != 2:
        raise ValueError("本实验只验证 2 台可变机组")
    if int(horizon) != 2:
        raise ValueError("本实验当前只验证 2 个时间步")
    if train_sample_count <= 0 or train_sample_count > 16:
        raise ValueError("train_sample_count 必须位于 1 到 16")
    if len(set(window_starts)) != len(window_starts):
        raise ValueError("window_starts 不能重复")

    source = load_uc_instance(instance_path)
    scenarios = [
        _evaluate_window(
            source=source,
            window_start=int(window_start),
            horizon=int(horizon),
            selected_generator_indices=selected_generator_indices,
            train_sample_count=int(train_sample_count),
            seed=int(seed),
            regularization=float(regularization),
            maxiter=int(maxiter),
        )
        for window_start in window_starts
    ]
    summary = {
        "method": "scalable sparse diagonal phase VQC training prototype",
        "research_scope": (
            "研究内容1的可扩展相位值函数训练与相位编码验证；"
            "尚未接入相干值寄存器、Grover 或自适应 Grover"
        ),
        "source_instance": str(instance_path),
        "num_generators_in_source": int(len(source.generators)),
        "source_time_horizon": int(source.time_horizon),
        "selected_generator_indices": [int(value) for value in selected_generator_indices],
        "selected_generator_names": [
            source.generators[index].name for index in selected_generator_indices
        ],
        "window_starts": [int(value) for value in window_starts],
        "horizon": int(horizon),
        "train_sample_count": int(train_sample_count),
        "seed": int(seed),
        "regularization": float(regularization),
        "maxiter": int(maxiter),
        "uses_sparse_local_phase_terms": True,
        "uses_full_state_lookup": False,
        "uses_grover": False,
        "uses_adaptive_grover": False,
        "scenarios": scenarios,
        "notes": [
            (
                "每个 shifted window 保留源实例的机组 initial status 和 initial power，"
                "因此是独立负荷/备用场景，不是前一窗口的时序延续。"
            ),
            "隐藏 ED/LP 标签只在模型训练完成后用于评估。",
            (
                "训练目标使用无噪声 Hadamard-test 期望的解析等价损失；"
                "最终参数再通过真实 Statevector 相位和 Hadamard readout 验证。"
            ),
            (
                "模型只包含单变量、同机组相邻时段和稀疏机组边的局部相位项，"
                "参数量不依赖 2^(G*T) 全状态表。"
            ),
            "Grover 和自适应 Grover 将在相干 phase-to-value 编码验证后接入。",
        ],
    }
    write_strict_json(results_path, summary)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="case14 三个 2×2 稀疏相位 VQC 训练示例")
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_2x2_sparse_phase_vqc.json"),
    )
    parser.add_argument("--selected-generators", type=_parse_int_tuple, default=(0, 5))
    parser.add_argument("--window-starts", type=_parse_int_tuple, default=(0, 1, 2))
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--train-sample-count", type=int, default=8)
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
        horizon=args.horizon,
        train_sample_count=args.train_sample_count,
        seed=args.seed,
        regularization=args.regularization,
        maxiter=args.maxiter,
    )
    compact = {
        "method": summary["method"],
        "selected_generator_names": summary["selected_generator_names"],
        "scenario_metrics": [
            {
                "window_start": scenario["window_start"],
                "training": scenario["metrics"]["training"],
                "hidden": scenario["metrics"]["hidden"],
                "phase_encoding_validation": {
                    "max_basis_statevector_phase_error": scenario[
                        "phase_encoding_validation"
                    ]["max_basis_statevector_phase_error"],
                    "max_hadamard_phase_error": scenario[
                        "phase_encoding_validation"
                    ]["max_hadamard_phase_error"],
                },
            }
            for scenario in summary["scenarios"]
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
