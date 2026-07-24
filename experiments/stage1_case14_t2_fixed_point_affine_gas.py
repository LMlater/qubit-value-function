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
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.experiment_utils import (  # noqa: E402
    embedded_selected_commitments,
    leading_time_window_instance,
    write_strict_json,
)
from qubit_value_function.fixed_point_oracle import (  # noqa: E402
    FixedPointAffineSpec,
    FixedPointConfig,
    build_fixed_point_grover_circuit,
    build_fixed_point_phase_oracle_circuit,
    fit_affine_cost_model,
    simulate_fixed_point_grover,
    simulate_fixed_point_phase_oracle,
)
from qubit_value_function.gate_level_oracle import (  # noqa: E402
    bitstring_from_index,
    circuit_resource_summary,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


REPRESENTATIVE_INDEX_ORDER = (0, 15, 3, 12, 5, 10, 6, 9, 1, 2, 4, 8, 7, 11, 13, 14)
INITIALIZATION_POLICIES = ("first", "random", "best-training")


def select_initial_index(
    train_indices: list[int] | tuple[int, ...],
    observed: dict[int, float],
    *,
    policy: str = "first",
    seed: int = 0,
) -> int:
    """按指定策略选择 GAS 初始 incumbent，不额外调用 ED/LP。"""

    indices = [int(index) for index in train_indices]
    if not indices:
        raise ValueError("train_indices 不能为空")
    if any(index not in observed for index in indices):
        raise ValueError("train_indices 中的每个索引都必须存在于 observed")
    if policy not in INITIALIZATION_POLICIES:
        raise ValueError(f"initialization_policy 必须是 {INITIALIZATION_POLICIES} 之一")
    if policy == "first":
        return int(indices[0])
    if policy == "random":
        rng = np.random.default_rng(int(seed))
        return int(rng.choice(np.asarray(indices, dtype=int)))
    return int(min(indices, key=lambda index: float(observed[index])))


def predicted_cost_diagnostics(
    spec: FixedPointAffineSpec,
    *,
    real_threshold: float,
) -> dict[str, object]:
    """汇总当前固定点 surrogate 与真实 incumbent threshold 的关系。"""

    encoded_values = np.array(
        [spec.encoded_cost_for_index(index) for index in range(2**spec.num_x_qubits)],
        dtype=int,
    )
    encoded_threshold = int(spec.config.encode(real_threshold))
    minimum_index = int(np.argmin(encoded_values))
    minimum_value = int(encoded_values[minimum_index])
    marked = encoded_values < encoded_threshold
    return {
        "predicted_encoded_costs": [int(value) for value in encoded_values],
        "minimum_predicted_index": minimum_index,
        "minimum_predicted_bitstring": bitstring_from_index(minimum_index, spec.num_x_qubits),
        "minimum_predicted_encoded_cost": minimum_value,
        "encoded_threshold": encoded_threshold,
        "minimum_predicted_minus_threshold": int(minimum_value - encoded_threshold),
        "marked_indices": [int(index) for index in np.flatnonzero(marked)],
        "marked_count": int(marked.sum()),
    }


def run(
    *,
    instance_path: Path,
    results_path: Path,
    selected_generator_indices: tuple[int, int] = (0, 5),
    train_sample_count: int = 6,
    max_rounds: int = 3,
    fractional_bits: int = 2,
    cost_unit: float = 1000.0,
    initialization_policy: str = "first",
    seed: int = 0,
) -> dict[str, object]:
    if len(selected_generator_indices) != 2:
        raise ValueError("本原型只验证 2 台机组")
    if train_sample_count < 5:
        raise ValueError("4-bit 仿射模型至少需要 5 个有限训练样本")
    if max_rounds <= 0:
        raise ValueError("max_rounds 必须为正数")
    if initialization_policy not in INITIALIZATION_POLICIES:
        raise ValueError(f"initialization_policy 必须是 {INITIALIZATION_POLICIES} 之一")

    source = load_uc_instance(instance_path)
    instance = leading_time_window_instance(source, 2)
    base_commitment = np.ones((len(instance.generators), 2), dtype=int)
    commitments = embedded_selected_commitments(base_commitment, selected_generator_indices)
    evaluator = FixedCommitmentEvaluator(instance)

    observed: dict[int, float] = {}
    training_trace: list[dict[str, object]] = []
    ed_lp_calls = 0
    for index in REPRESENTATIVE_INDEX_ORDER:
        commitment = commitments[int(index)]
        if not is_logic_feasible(instance, commitment):
            training_trace.append(
                {
                    "index": int(index),
                    "bitstring": bitstring_from_index(index, 4),
                    "status": "skipped_logic_infeasible",
                }
            )
            continue
        result = evaluator.evaluate(commitment)
        ed_lp_calls += 1
        training_trace.append(
            {
                "index": int(index),
                "bitstring": bitstring_from_index(index, 4),
                "status": "finite" if result.success and np.isfinite(result.total_cost) else "ed_failed",
                "true_cost": float(result.total_cost) if result.success and np.isfinite(result.total_cost) else None,
            }
        )
        if result.success and np.isfinite(result.total_cost):
            observed[int(index)] = float(result.total_cost)
        if len(observed) >= int(train_sample_count):
            break

    if len(observed) < int(train_sample_count):
        raise RuntimeError("代表性样本中没有足够的有限 ED/LP 结果")

    train_indices = list(observed)
    train_bitstrings = [bitstring_from_index(index, 4) for index in train_indices]
    train_costs = np.array([observed[index] for index in train_indices], dtype=float)
    intercept, coefficients = fit_affine_cost_model(
        bitstrings=train_bitstrings,
        costs=train_costs,
    )
    fixed_point = FixedPointConfig(
        fractional_bits=fractional_bits,
        unit=cost_unit,
        rounding="nearest",
    )
    spec = FixedPointAffineSpec.from_real_coefficients(
        config=fixed_point,
        intercept=intercept,
        coefficients=coefficients,
        bit_labels=("g1_t0", "g1_t1", "g2_t0", "g2_t1"),
        name="case14_t2_fixed_point_affine_cost",
    )

    initial_index = select_initial_index(
        train_indices,
        observed,
        policy=initialization_policy,
        seed=seed,
    )
    incumbent_index = int(initial_index)
    incumbent_cost = float(observed[initial_index])
    initial_diagnostics = predicted_cost_diagnostics(spec, real_threshold=incumbent_cost)
    rounds: list[dict[str, object]] = []

    for round_index in range(int(max_rounds)):
        threshold_before = float(incumbent_cost)
        diagnostics = predicted_cost_diagnostics(spec, real_threshold=threshold_before)
        probe = simulate_fixed_point_phase_oracle(
            spec,
            real_threshold=threshold_before,
            strict=True,
        )
        marked_indices = [int(index) for index in np.flatnonzero(probe.marked_mask)]
        if marked_indices != diagnostics["marked_indices"]:
            raise RuntimeError("经典固定点诊断与门级 phase oracle 的 marked states 不一致")

        round_record: dict[str, object] = {
            "round": int(round_index),
            "threshold_before": threshold_before,
            "encoded_threshold_before": int(diagnostics["encoded_threshold"]),
            "minimum_predicted_index": int(diagnostics["minimum_predicted_index"]),
            "minimum_predicted_encoded_cost": int(diagnostics["minimum_predicted_encoded_cost"]),
            "minimum_predicted_minus_threshold": int(diagnostics["minimum_predicted_minus_threshold"]),
            "marked_indices": marked_indices,
            "marked_count": int(diagnostics["marked_count"]),
            "grover_iterations": 0,
            "marked_probability": None,
            "phase_max_error": float(probe.max_phase_error),
            "auxiliary_zero_probability": float(probe.auxiliary_zero_probability),
            "candidate_index": None,
            "candidate_bitstring": None,
            "candidate_probability": None,
            "candidate_status": None,
            "candidate_true_cost": None,
            "accepted_update": False,
            "threshold_after": threshold_before,
            "encoded_threshold_after": int(fixed_point.encode(threshold_before)),
        }

        if not marked_indices:
            round_record["stop_reason"] = "no_marked_state_at_fixed_point_threshold"
            rounds.append(round_record)
            break

        grover = simulate_fixed_point_grover(
            spec,
            real_threshold=threshold_before,
            strict=True,
        )
        phase_circuit = build_fixed_point_phase_oracle_circuit(
            spec,
            real_threshold=threshold_before,
            strict=True,
        )
        grover_circuit = build_fixed_point_grover_circuit(
            spec,
            real_threshold=threshold_before,
            iterations=grover.iterations,
            strict=True,
        )
        round_record.update(
            {
                "grover_iterations": int(grover.iterations),
                "marked_probability": float(grover.marked_probability),
                "auxiliary_zero_probability": float(grover.auxiliary_zero_probability),
                "resources": {
                    "phase_oracle": circuit_resource_summary(phase_circuit, decompose_reps=1),
                    "grover_circuit": circuit_resource_summary(grover_circuit, decompose_reps=1),
                },
            }
        )

        unobserved_marked = [index for index in marked_indices if index not in observed]
        if not unobserved_marked:
            round_record["stop_reason"] = "all_marked_states_already_verified"
            rounds.append(round_record)
            break

        candidate_index = max(
            unobserved_marked,
            key=lambda index: float(grover.x_probabilities[int(index)]),
        )
        candidate_commitment = commitments[candidate_index]
        candidate_cost: float | None = None
        accepted = False
        status = "logic_infeasible"
        if is_logic_feasible(instance, candidate_commitment):
            result = evaluator.evaluate(candidate_commitment)
            ed_lp_calls += 1
            if result.success and np.isfinite(result.total_cost):
                candidate_cost = float(result.total_cost)
                observed[candidate_index] = candidate_cost
                status = "finite"
                if candidate_cost < incumbent_cost:
                    incumbent_index = candidate_index
                    incumbent_cost = candidate_cost
                    accepted = True
            else:
                status = "ed_failed"

        round_record.update(
            {
                "candidate_index": int(candidate_index),
                "candidate_bitstring": bitstring_from_index(candidate_index, 4),
                "candidate_probability": float(grover.x_probabilities[candidate_index]),
                "candidate_status": status,
                "candidate_true_cost": candidate_cost,
                "accepted_update": bool(accepted),
                "incumbent_index_after": int(incumbent_index),
                "incumbent_true_cost_after": float(incumbent_cost),
                "threshold_after": float(incumbent_cost),
                "encoded_threshold_after": int(fixed_point.encode(incumbent_cost)),
            }
        )
        rounds.append(round_record)
        if not accepted:
            break

    summary = {
        "method": "2-generator 2-period fixed-point affine gate-level GAS prototype",
        "research_scope": "研究内容1的固定点门级算术基线，不是 VQC 值函数",
        "uses_qft": False,
        "uses_weighted_adder": True,
        "uses_integer_comparator": True,
        "uses_statevector_for_gate_level_execution": True,
        "uses_hidden_full_enumeration_for_training": False,
        "selected_generators": [int(index) for index in selected_generator_indices],
        "num_search_qubits": 4,
        "training_trace": training_trace,
        "train_indices": train_indices,
        "train_bitstrings": train_bitstrings,
        "train_true_costs": train_costs.tolist(),
        "affine_model": {
            "intercept": float(intercept),
            "coefficients": coefficients.tolist(),
        },
        "fixed_point": {
            "fractional_bits": int(fixed_point.fractional_bits),
            "scale": int(fixed_point.scale),
            "cost_unit": float(fixed_point.unit),
            "quantum": float(fixed_point.quantum),
            "max_abs_rounding_error": float(fixed_point.max_abs_rounding_error),
            "encoded_offset": int(spec.encoded_offset),
            "weights": [int(weight) for weight in spec.weights],
            "inverted_bit_indices": [int(index) for index in spec.inverted_bit_indices],
            "max_weighted_sum": int(spec.max_weighted_sum),
        },
        "initialization_policy": initialization_policy,
        "seed": int(seed),
        "initial_incumbent": {
            "index": int(initial_index),
            "bitstring": bitstring_from_index(initial_index, 4),
            "true_cost": float(observed[initial_index]),
        },
        "initial_search_diagnostics": initial_diagnostics,
        "rounds": rounds,
        "final_incumbent": {
            "index": int(incumbent_index),
            "bitstring": bitstring_from_index(incumbent_index, 4),
            "true_cost": float(incumbent_cost),
        },
        "algorithmic_ed_lp_calls": int(ed_lp_calls),
        "verified_finite_indices": sorted(int(index) for index in observed),
        "notes": [
            "训练样本按代表性索引顺序逐个调用 ED/LP，达到指定数量后立即停止。",
            (
                "默认使用第一个有限训练样本作为初始 incumbent，"
                "不再预先选择训练集最低成本。"
            ),
            "所有成本、仿射系数和 GAS threshold 使用同一个固定点配置。",
            "statevector 只执行完整门级电路并读取概率，不直接改写量子振幅。",
            "当前模型是单一经典 ridge 仿射值函数基线，尚未实现 VQC。",
        ],
    }
    write_strict_json(results_path, summary)
    return summary


def _parse_selected_generators(raw: str) -> tuple[int, int]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 2:
        raise argparse.ArgumentTypeError("selected-generators 必须恰好包含 2 个索引")
    return values


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2台机组×2时间步固定点门级 GAS 原型")
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_fixed_point_affine_gas.json"),
    )
    parser.add_argument("--selected-generators", type=_parse_selected_generators, default=(0, 5))
    parser.add_argument("--train-sample-count", type=int, default=6)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--fractional-bits", type=int, default=2)
    parser.add_argument("--cost-unit", type=float, default=1000.0)
    parser.add_argument(
        "--initialization-policy",
        choices=INITIALIZATION_POLICIES,
        default="first",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    summary = run(
        instance_path=args.instance,
        results_path=args.results,
        selected_generator_indices=args.selected_generators,
        train_sample_count=args.train_sample_count,
        max_rounds=args.max_rounds,
        fractional_bits=args.fractional_bits,
        cost_unit=args.cost_unit,
        initialization_policy=args.initialization_policy,
        seed=args.seed,
    )
    print(json.dumps(summary["final_incumbent"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
