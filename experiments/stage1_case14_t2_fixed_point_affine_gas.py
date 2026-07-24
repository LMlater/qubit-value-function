from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

try:
    from qiskit_aer import AerSimulator
except Exception:  # pragma: no cover - 仅在缺少 Aer 时触发
    AerSimulator = None

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
    fit_affine_cost_model,
    optimal_grover_iterations,
    simulate_fixed_point_phase_oracle,
    x_marginal_probabilities,
)
from qubit_value_function.gate_level_oracle import (  # noqa: E402
    bitstring_from_index,
    circuit_resource_summary,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


REPRESENTATIVE_INDEX_ORDER = (0, 15, 3, 12, 5, 10, 6, 9, 1, 2, 4, 8, 7, 11, 13, 14)
INITIALIZATION_POLICIES = ("first", "random", "best-training")
SIMULATION_METHODS = ("mps", "statevector")


@dataclass(frozen=True)
class GateExecutionResult:
    """一次完整门级 Grover 电路的执行结果。"""

    simulation_method: str
    x_probabilities: np.ndarray
    x_counts: dict[str, int] | None
    auxiliary_zero_probability: float
    total_qubits: int
    estimated_statevector_memory_gb: float


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


def estimate_statevector_memory_gb(num_qubits: int) -> float:
    """估算 complex128 全状态向量的最低存储空间。"""

    if int(num_qubits) < 0:
        raise ValueError("num_qubits 不能为负数")
    return float((2**int(num_qubits)) * 16 / (1024**3))


def execute_grover_circuit(
    circuit: QuantumCircuit,
    *,
    num_x_qubits: int,
    simulation_method: str,
    shots: int,
    seed: int,
    max_statevector_memory_gb: float,
) -> GateExecutionResult:
    """执行完整门级 Grover 电路，不使用经典 marked mask 改写振幅。"""

    if simulation_method not in SIMULATION_METHODS:
        raise ValueError(f"simulation_method 必须是 {SIMULATION_METHODS} 之一")
    if int(num_x_qubits) <= 0 or int(num_x_qubits) > circuit.num_qubits:
        raise ValueError("num_x_qubits 与量子电路不兼容")
    if int(shots) <= 0:
        raise ValueError("shots 必须为正数")
    if not np.isfinite(max_statevector_memory_gb) or max_statevector_memory_gb <= 0.0:
        raise ValueError("max_statevector_memory_gb 必须为有限正数")

    total_qubits = int(circuit.num_qubits)
    memory_gb = estimate_statevector_memory_gb(total_qubits)

    if simulation_method == "statevector":
        if memory_gb > float(max_statevector_memory_gb):
            raise RuntimeError(
                "预计 Statevector 至少需要 "
                f"{memory_gb:.3f} GB，超过限制 {max_statevector_memory_gb:.3f} GB；"
                "请改用 --simulation-method mps，或显式提高内存限制"
            )
        statevector = Statevector.from_instruction(circuit)
        x_probabilities, auxiliary_zero_probability = x_marginal_probabilities(
            statevector.probabilities(),
            int(num_x_qubits),
        )
        return GateExecutionResult(
            simulation_method="statevector",
            x_probabilities=x_probabilities,
            x_counts=None,
            auxiliary_zero_probability=float(auxiliary_zero_probability),
            total_qubits=total_qubits,
            estimated_statevector_memory_gb=memory_gb,
        )

    if AerSimulator is None:
        raise RuntimeError("MPS 模拟需要安装 qiskit-aer")

    measured_circuit = circuit.copy()
    measurement_register = ClassicalRegister(total_qubits, "measure")
    measured_circuit.add_register(measurement_register)
    measured_circuit.measure(measured_circuit.qubits, measurement_register)

    backend = AerSimulator(method="matrix_product_state")
    compiled = transpile(
        measured_circuit,
        backend,
        optimization_level=1,
        seed_transpiler=int(seed),
    )
    result = backend.run(
        compiled,
        shots=int(shots),
        seed_simulator=int(seed),
    ).result()
    raw_counts = result.get_counts(compiled)

    x_dimension = 2**int(num_x_qubits)
    x_mask = x_dimension - 1
    x_index_counts = np.zeros(x_dimension, dtype=int)
    auxiliary_zero_count = 0
    for raw_bitstring, count in raw_counts.items():
        compact = str(raw_bitstring).replace(" ", "")
        full_index = int(compact, 2)
        x_index = full_index & x_mask
        x_index_counts[x_index] += int(count)
        if full_index >> int(num_x_qubits) == 0:
            auxiliary_zero_count += int(count)

    actual_shots = int(x_index_counts.sum())
    if actual_shots <= 0:
        raise RuntimeError("MPS 模拟没有返回有效测量结果")
    x_probabilities = x_index_counts.astype(float) / float(actual_shots)
    x_counts = {
        bitstring_from_index(index, int(num_x_qubits)): int(count)
        for index, count in enumerate(x_index_counts)
        if int(count) > 0
    }
    return GateExecutionResult(
        simulation_method="mps",
        x_probabilities=x_probabilities,
        x_counts=x_counts,
        auxiliary_zero_probability=float(auxiliary_zero_count / actual_shots),
        total_qubits=total_qubits,
        estimated_statevector_memory_gb=memory_gb,
    )


def select_measured_candidate(
    x_probabilities: np.ndarray,
    *,
    observed_indices: set[int] | list[int] | tuple[int, ...],
    allowed_indices: set[int] | list[int] | tuple[int, ...] | None = None,
) -> int | None:
    """从实际电路输出分布中选择最高概率的未验证允许状态。"""

    probabilities = np.asarray(x_probabilities, dtype=float)
    observed = {int(index) for index in observed_indices}
    allowed = (
        set(range(probabilities.size))
        if allowed_indices is None
        else {int(index) for index in allowed_indices}
    )
    candidates = [
        int(index)
        for index, probability in enumerate(probabilities)
        if int(index) in allowed and int(index) not in observed and float(probability) > 0.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda index: (float(probabilities[index]), -int(index)))


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
    simulation_method: str = "mps",
    shots: int = 4096,
    verify_phase_oracle: bool = False,
    max_statevector_memory_gb: float = 1.0,
) -> dict[str, object]:
    if len(selected_generator_indices) != 2:
        raise ValueError("本原型只验证 2 台机组")
    if train_sample_count < 5:
        raise ValueError("4-bit 仿射模型至少需要 5 个有限训练样本")
    if max_rounds <= 0:
        raise ValueError("max_rounds 必须为正数")
    if initialization_policy not in INITIALIZATION_POLICIES:
        raise ValueError(f"initialization_policy 必须是 {INITIALIZATION_POLICIES} 之一")
    if simulation_method not in SIMULATION_METHODS:
        raise ValueError(f"simulation_method 必须是 {SIMULATION_METHODS} 之一")
    if shots <= 0:
        raise ValueError("shots 必须为正数")

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
        marked_indices = [int(index) for index in diagnostics["marked_indices"]]

        round_record: dict[str, object] = {
            "round": int(round_index),
            "threshold_before": threshold_before,
            "encoded_threshold_before": int(diagnostics["encoded_threshold"]),
            "minimum_predicted_index": int(diagnostics["minimum_predicted_index"]),
            "minimum_predicted_encoded_cost": int(diagnostics["minimum_predicted_encoded_cost"]),
            "minimum_predicted_minus_threshold": int(diagnostics["minimum_predicted_minus_threshold"]),
            "marked_indices": marked_indices,
            "marked_count": int(diagnostics["marked_count"]),
            "simulation_method": simulation_method,
            "shots": int(shots) if simulation_method == "mps" else None,
            "phase_oracle_exact_verification": False,
            "phase_max_error": None,
            "phase_auxiliary_zero_probability": None,
            "auxiliary_zero_probability": None,
            "grover_iterations": 0,
            "marked_probability": None,
            "measured_marked_probability": None,
            "x_counts": None,
            "candidate_selection_source": None,
            "candidate_index": None,
            "candidate_bitstring": None,
            "candidate_probability": None,
            "candidate_predicted_marked": None,
            "candidate_status": None,
            "candidate_true_cost": None,
            "accepted_update": False,
            "threshold_after": threshold_before,
            "encoded_threshold_after": int(fixed_point.encode(threshold_before)),
        }

        # 无 marked state 时直接停止，避免运行指数内存的 phase-oracle Statevector。
        if not marked_indices:
            round_record["stop_reason"] = "no_marked_state_at_fixed_point_threshold"
            rounds.append(round_record)
            break

        if verify_phase_oracle:
            probe = simulate_fixed_point_phase_oracle(
                spec,
                real_threshold=threshold_before,
                strict=True,
            )
            probe_marked = [int(index) for index in np.flatnonzero(probe.marked_mask)]
            if probe_marked != marked_indices:
                raise RuntimeError("经典固定点诊断与门级 phase oracle 的 marked states 不一致")
            round_record.update(
                {
                    "phase_oracle_exact_verification": True,
                    "phase_max_error": float(probe.max_phase_error),
                    "phase_auxiliary_zero_probability": float(probe.auxiliary_zero_probability),
                }
            )

        iterations = optimal_grover_iterations(2**spec.num_x_qubits, len(marked_indices))
        grover_circuit = build_fixed_point_grover_circuit(
            spec,
            real_threshold=threshold_before,
            iterations=iterations,
            strict=True,
        )
        resources = circuit_resource_summary(grover_circuit, decompose_reps=1)
        execution = execute_grover_circuit(
            grover_circuit,
            num_x_qubits=spec.num_x_qubits,
            simulation_method=simulation_method,
            shots=shots,
            seed=seed + round_index,
            max_statevector_memory_gb=max_statevector_memory_gb,
        )
        marked_probability = float(execution.x_probabilities[marked_indices].sum())
        round_record.update(
            {
                "grover_iterations": int(iterations),
                "marked_probability": marked_probability,
                "measured_marked_probability": marked_probability if simulation_method == "mps" else None,
                "x_counts": execution.x_counts,
                "auxiliary_zero_probability": float(execution.auxiliary_zero_probability),
                "total_qubits": int(execution.total_qubits),
                "estimated_statevector_memory_gb": float(execution.estimated_statevector_memory_gb),
                "resources": {"grover_circuit": resources},
            }
        )

        candidate_index = select_measured_candidate(
            execution.x_probabilities,
            observed_indices=set(observed),
            allowed_indices=set(marked_indices),
        )
        if candidate_index is None:
            round_record["stop_reason"] = "no_unverified_marked_state_in_quantum_output"
            rounds.append(round_record)
            break

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
                "candidate_selection_source": (
                    "measured_gate_level_counts"
                    if simulation_method == "mps"
                    else "exact_gate_level_statevector"
                ),
                "candidate_index": int(candidate_index),
                "candidate_bitstring": bitstring_from_index(candidate_index, 4),
                "candidate_probability": float(execution.x_probabilities[candidate_index]),
                "candidate_predicted_marked": bool(candidate_index in marked_indices),
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
        "uses_statevector_for_gate_level_execution": simulation_method == "statevector",
        "uses_aer_mps_for_gate_level_execution": simulation_method == "mps",
        "uses_hidden_full_enumeration_for_training": False,
        "simulation_method": simulation_method,
        "shots": int(shots) if simulation_method == "mps" else None,
        "verify_phase_oracle": bool(verify_phase_oracle),
        "max_statevector_memory_gb": float(max_statevector_memory_gb),
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
            "默认使用第一个有限训练样本作为初始 incumbent，不再预先选择训练集最低成本。",
            "默认使用 Aer MPS + shots 执行完整门级 Grover 电路，候选来自实际测量分布。",
            "无 marked state 时直接停止，不再启动昂贵的 phase-oracle Statevector。",
            "精确 phase 和 uncompute 验证由专项测试覆盖，也可用 --verify-phase-oracle 显式开启。",
            "所有成本、仿射系数和 GAS threshold 使用同一个固定点配置。",
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
    parser.add_argument(
        "--simulation-method",
        choices=SIMULATION_METHODS,
        default="mps",
    )
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument(
        "--verify-phase-oracle",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-statevector-memory-gb", type=float, default=1.0)
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
        simulation_method=args.simulation_method,
        shots=args.shots,
        verify_phase_oracle=args.verify_phase_oracle,
        max_statevector_memory_gb=args.max_statevector_memory_gb,
    )
    print(json.dumps(summary["final_incumbent"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
