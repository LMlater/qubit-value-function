from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.commitment import is_logic_feasible  # noqa: E402
from qubit_value_function.coherent_phase_value import (  # noqa: E402
    basis_value_code_probe,
    build_integer_value_phase_circuit,
    build_phase_to_value_circuit,
    build_sparse_vqc_threshold_phase_oracle,
    estimate_statevector_memory_gb,
    phase_to_value_superposition_probe,
    quantize_sparse_phase_model,
    simulate_sparse_vqc_threshold_phase_oracle,
)
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
                    "bitstring": bitstring_from_index(index, 4),
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
                "bitstring": bitstring_from_index(index, 4),
                "status": "finite" if finite else "ed_failed",
                "true_cost": float(result.total_cost) if finite else None,
            }
        )
        if finite:
            train_indices.append(index)
            bitstrings.append(bitstring_from_index(index, 4))
            costs.append(float(result.total_cost))
        if len(costs) >= int(train_sample_count):
            break
    if len(costs) < int(train_sample_count):
        raise RuntimeError("没有获得足够的有限训练 ED/LP 样本")
    return train_indices, bitstrings, costs, trace, calls


def _evaluate_all_states_after_training(
    instance,
    commitments: np.ndarray,
    *,
    known_costs: dict[int, float],
) -> tuple[dict[int, float], list[dict[str, object]], int]:
    evaluator = FixedCommitmentEvaluator(instance)
    true_costs = dict(known_costs)
    trace: list[dict[str, object]] = []
    calls = 0
    for index in range(commitments.shape[0]):
        if index in true_costs:
            trace.append(
                {
                    "index": int(index),
                    "bitstring": bitstring_from_index(index, 4),
                    "status": "training_value_reused",
                    "true_cost": float(true_costs[index]),
                }
            )
            continue
        commitment = commitments[index]
        if not is_logic_feasible(instance, commitment):
            trace.append(
                {
                    "index": int(index),
                    "bitstring": bitstring_from_index(index, 4),
                    "status": "logic_infeasible",
                }
            )
            continue
        result = evaluator.evaluate(commitment)
        calls += 1
        finite = bool(result.success and np.isfinite(result.total_cost))
        trace.append(
            {
                "index": int(index),
                "bitstring": bitstring_from_index(index, 4),
                "status": "finite" if finite else "ed_failed",
                "true_cost": float(result.total_cost) if finite else None,
            }
        )
        if finite:
            true_costs[int(index)] = float(result.total_cost)
    return true_costs, trace, calls


def _evaluate_window(
    *,
    source,
    window_start: int,
    selected_generator_indices: tuple[int, int],
    train_sample_count: int,
    seed: int,
    regularization: float,
    maxiter: int,
    fixed_point: FixedPointConfig,
) -> dict[str, object]:
    horizon = 2
    instance = time_window_instance(source, start=window_start, horizon=horizon)
    base_commitment = np.ones((len(instance.generators), horizon), dtype=int)
    commitments = embedded_selected_commitments(base_commitment, selected_generator_indices)

    train_indices, train_bitstrings, train_costs, training_trace, training_calls = (
        _collect_training_data(
            instance,
            commitments,
            train_sample_count=train_sample_count,
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

    known_training_costs = {
        int(index): float(cost) for index, cost in zip(train_indices, train_costs)
    }
    true_costs, evaluation_trace, evaluation_calls = _evaluate_all_states_after_training(
        instance,
        commitments,
        known_costs=known_training_costs,
    )

    basis_rows: list[dict[str, object]] = []
    minimum_code_probability = 1.0
    maximum_code_difference = 0
    maximum_prediction_quantization_error = 0.0
    for index in range(commitments.shape[0]):
        bitstring = bitstring_from_index(index, 4)
        probe = basis_value_code_probe(value_model, bitstring)
        direct_code = int(fixed_point.encode(phase_model.predict_cost(bitstring)))
        sparse_code = int(value_model.integer_value(bitstring))
        code_difference = int(sparse_code - direct_code)
        quantization_error = float(
            fixed_point.decode(sparse_code) - phase_model.predict_cost(bitstring)
        )
        minimum_code_probability = min(
            minimum_code_probability,
            float(probe.correct_code_probability),
        )
        maximum_code_difference = max(maximum_code_difference, abs(code_difference))
        maximum_prediction_quantization_error = max(
            maximum_prediction_quantization_error,
            abs(quantization_error),
        )
        basis_rows.append(
            {
                "index": int(index),
                "bitstring": bitstring,
                "predicted_cost": float(phase_model.predict_cost(bitstring)),
                "direct_encoded_prediction": direct_code,
                "sparse_integer_code": sparse_code,
                "shifted_value_code": int(value_model.shifted_integer_value(bitstring)),
                "code_difference_from_direct_rounding": code_difference,
                "decoded_sparse_minus_predicted_cost": quantization_error,
                "most_likely_value_code": int(probe.most_likely_code),
                "correct_value_code_probability": float(probe.correct_code_probability),
                "true_cost_evaluation_only": (
                    float(true_costs[index]) if index in true_costs else None
                ),
            }
        )

    superposition = phase_to_value_superposition_probe(value_model)
    threshold_specs = [
        ("first_training_incumbent", float(train_costs[0])),
        ("best_training_incumbent", float(min(train_costs))),
    ]
    threshold_rows: list[dict[str, object]] = []
    maximum_phase_error = 0.0
    minimum_oracle_auxiliary_zero_probability = 1.0
    maximum_oracle_qubits = 0
    for label, real_threshold in threshold_specs:
        encoded_threshold = int(fixed_point.encode(real_threshold))
        oracle_probe = simulate_sparse_vqc_threshold_phase_oracle(
            value_model,
            encoded_real_threshold=encoded_threshold,
            strict=True,
        )
        oracle = build_sparse_vqc_threshold_phase_oracle(
            value_model,
            encoded_real_threshold=encoded_threshold,
            strict=True,
        )
        marked_indices = [int(index) for index in np.flatnonzero(oracle_probe.marked_mask)]
        maximum_phase_error = max(maximum_phase_error, float(oracle_probe.max_phase_error))
        minimum_oracle_auxiliary_zero_probability = min(
            minimum_oracle_auxiliary_zero_probability,
            float(oracle_probe.auxiliary_zero_probability),
        )
        maximum_oracle_qubits = max(maximum_oracle_qubits, int(oracle.num_qubits))
        threshold_rows.append(
            {
                "label": label,
                "real_threshold": real_threshold,
                "encoded_threshold": encoded_threshold,
                "shifted_compare_value": int(
                    value_model.shifted_compare_value(encoded_threshold, strict=True)
                ),
                "marked_indices": marked_indices,
                "marked_bitstrings": [bitstring_from_index(index, 4) for index in marked_indices],
                "phase_max_error": float(oracle_probe.max_phase_error),
                "auxiliary_zero_probability": float(
                    oracle_probe.auxiliary_zero_probability
                ),
                "resources": circuit_resource_summary(oracle, decompose_reps=1),
                "estimated_statevector_memory_gb": estimate_statevector_memory_gb(
                    oracle.num_qubits
                ),
            }
        )

    integer_phase = build_integer_value_phase_circuit(value_model)
    phase_to_value = build_phase_to_value_circuit(value_model)
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
        "evaluation_trace_after_training": evaluation_trace,
        "training_ed_lp_calls": int(training_calls),
        "evaluation_only_ed_lp_calls": int(evaluation_calls),
        "fit": fit.as_dict(),
        "phase_model": phase_model.as_dict(),
        "quantized_value_model": value_model.as_dict(),
        "quantization_diagnostics": {
            "minimum_exact_value_code_probability": float(minimum_code_probability),
            "maximum_code_difference_from_direct_rounding": int(maximum_code_difference),
            "maximum_abs_decoded_sparse_minus_predicted_cost": float(
                maximum_prediction_quantization_error
            ),
            "rows": basis_rows,
        },
        "coherent_phase_to_value_validation": {
            "superposition_pairing_probability": float(
                superposition.pairing_probability
            ),
            "inverse_auxiliary_zero_probability": float(
                superposition.inverse_auxiliary_zero_probability
            ),
        },
        "threshold_oracle_validation": {
            "thresholds": threshold_rows,
            "maximum_phase_error": float(maximum_phase_error),
            "minimum_auxiliary_zero_probability": float(
                minimum_oracle_auxiliary_zero_probability
            ),
        },
        "resources": {
            "integer_value_phase": circuit_resource_summary(
                integer_phase,
                decompose_reps=1,
            ),
            "phase_to_value": circuit_resource_summary(
                phase_to_value,
                decompose_reps=1,
            ),
            "maximum_threshold_oracle_qubits": int(maximum_oracle_qubits),
            "phase_to_value_estimated_statevector_memory_gb": (
                estimate_statevector_memory_gb(phase_to_value.num_qubits)
            ),
        },
        "uses_full_state_lookup": False,
        "full_state_enumeration_used_for_validation_only": True,
        "uses_grover": False,
        "uses_adaptive_grover": False,
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
            seed=int(seed),
            regularization=float(regularization),
            maxiter=int(maxiter),
            fixed_point=fixed_point,
        )
        for window_start in window_starts
    ]
    summary = {
        "method": "scalable sparse VQC coherent phase-to-fixed-point-value prototype",
        "research_scope": (
            "研究内容1的相干 phase-to-value 和 threshold comparator 验证；"
            "尚未接入 Grover 或自适应 Grover"
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
        "seed": int(seed),
        "regularization": float(regularization),
        "maxiter": int(maxiter),
        "uses_qft_for_coherent_phase_readout": True,
        "uses_qft_for_weighted_adder_arithmetic": False,
        "uses_full_state_lookup": False,
        "uses_grover": False,
        "uses_adaptive_grover": False,
        "scenarios": scenarios,
        "notes": [
            "QFTGate inverse 仅用于将对角相位本征值相干读取到固定点值寄存器。",
            "整数范围由稀疏系数的保守上下界计算，不通过枚举 2^(G*T) 状态获得。",
            "16状态遍历仅用于当前2×2验收，不参与量子线路构造或VQC训练。",
            "threshold 来自真实训练 ED/LP incumbent；本阶段只验证 comparator，不执行搜索。",
            "下一阶段将在该 compute-comparator-uncompute oracle 上接入普通 Grover 和 BBHT 自适应 Grover。",
        ],
    }
    write_strict_json(results_path, summary)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="case14 三个2×2相干 phase-to-value 验证")
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_2x2_coherent_phase_value.json"),
    )
    parser.add_argument("--selected-generators", type=_parse_int_tuple, default=(0, 5))
    parser.add_argument("--window-starts", type=_parse_int_tuple, default=(0, 1, 2))
    parser.add_argument("--train-sample-count", type=int, default=8)
    parser.add_argument("--fractional-bits", type=int, default=2)
    parser.add_argument("--cost-unit", type=float, default=1000.0)
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
        seed=args.seed,
        regularization=args.regularization,
        maxiter=args.maxiter,
    )
    compact = {
        "method": summary["method"],
        "scenarios": [
            {
                "window_start": scenario["window_start"],
                "num_value_qubits": scenario["quantized_value_model"][
                    "num_value_qubits"
                ],
                "minimum_exact_value_code_probability": scenario[
                    "quantization_diagnostics"
                ]["minimum_exact_value_code_probability"],
                "superposition_pairing_probability": scenario[
                    "coherent_phase_to_value_validation"
                ]["superposition_pairing_probability"],
                "inverse_auxiliary_zero_probability": scenario[
                    "coherent_phase_to_value_validation"
                ]["inverse_auxiliary_zero_probability"],
                "threshold_phase_max_error": scenario[
                    "threshold_oracle_validation"
                ]["maximum_phase_error"],
            }
            for scenario in summary["scenarios"]
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
