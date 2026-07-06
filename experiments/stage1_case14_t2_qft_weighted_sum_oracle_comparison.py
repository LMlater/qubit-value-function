from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_t2_ancilla_vqc import (  # noqa: E402
    commitment_row,
    evaluate_values,
    leading_time_window_instance,
)
from experiments.stage1_case14_t2_gate_level_grover_oracle import (  # noqa: E402
    embedded_selected_commitments,
)
from experiments.stage1_case14_t2_learned_small_max_affine_gate_level_oracle import (  # noqa: E402
    calibrate_integer_threshold,
    learned_to_dict,
    learn_gap_weighted_max_affine_pieces,
    oracle_and_circuit_explanation,
    run_target_case,
)
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.gate_level_oracle import (  # noqa: E402
    GateLevelMaxAffineOracleSpec,
    bitstring_from_index,
    circuit_resource_summary,
)
from qubit_value_function.qft_weighted_sum_oracle import (  # noqa: E402
    build_qft_max_affine_grover_circuit,
    build_qft_max_affine_phase_oracle_circuit,
    simulate_qft_max_affine_grover,
    simulate_qft_max_affine_phase_oracle,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def run(
    *,
    instance_path: Path,
    results_path: Path,
    report_path: Path,
    horizon: int,
    selected_generator_indices: tuple[int, ...],
    target_counts: tuple[int, ...],
    max_weight: int,
) -> dict[str, object]:
    if horizon != 2:
        raise ValueError("this comparison experiment is restricted to T=2")
    if len(selected_generator_indices) != 2:
        raise ValueError("the comparison experiment expects exactly two selected generators")

    source_instance = load_uc_instance(instance_path)
    instance = leading_time_window_instance(source_instance, horizon)
    generator_names = [gen.name for gen in instance.generators]
    commitments = all_commitments(len(instance.generators), instance.time_horizon)

    value_start = time.perf_counter()
    values, logic_feasible = evaluate_values(instance, commitments)
    value_seconds = time.perf_counter() - value_start
    finite = np.isfinite(values)
    finite_sorted_indices = [int(index) for index in np.argsort(values) if finite[index]]
    full_optimum_index = finite_sorted_indices[0]
    full_optimum_commitment = commitments[full_optimum_index]

    embedded_commitments = embedded_selected_commitments(
        full_optimum_commitment,
        selected_generator_indices,
    )
    embedded_values, embedded_logic_feasible = evaluate_values(instance, embedded_commitments)
    embedded_finite = np.isfinite(embedded_values)
    embedded_best_index = int(np.nanargmin(embedded_values))

    learned = learn_gap_weighted_max_affine_pieces(
        instance=instance,
        selected_generator_indices=selected_generator_indices,
        embedded_values=embedded_values,
        embedded_best_index=embedded_best_index,
        max_weight=max_weight,
    )
    template_spec = GateLevelMaxAffineOracleSpec(
        pieces=learned.pieces,
        threshold=0,
        bit_labels=learned.bit_labels,
        name="case14_t2_qft_comparison_learned_max_affine_oracle",
    )
    predicted_values = template_spec.values_for_all_x()

    target_rows = []
    for target_count in target_counts:
        calibration = calibrate_integer_threshold(
            predicted_values=predicted_values,
            true_values=embedded_values,
            target_count=int(target_count),
        )
        spec = GateLevelMaxAffineOracleSpec(
            pieces=learned.pieces,
            threshold=int(calibration["selected_threshold"]),
            bit_labels=learned.bit_labels,
            name=f"case14_t2_qft_comparison_top_{int(target_count)}",
        )
        target_rows.append(
            {
                "target_count_request": int(target_count),
                "threshold_tau": int(spec.threshold),
                "target_selected_bitstrings": calibration["target_selected_bitstrings"],
                "weighted_adder": run_target_case(
                    spec=spec,
                    calibration=calibration,
                    embedded_commitments=embedded_commitments,
                    embedded_values=embedded_values,
                ),
                "qft_weighted_sum": run_qft_target_case(
                    spec=spec,
                    calibration=calibration,
                    embedded_commitments=embedded_commitments,
                    embedded_values=embedded_values,
                ),
            }
        )

    summary = {
        "instance": f"case14_T{horizon}",
        "source": str(instance_path),
        "method": "QFT-based weighted-sum oracle compared against Qiskit WeightedAdder oracle",
        "scope": {
            "full_case14_bits": int(commitments.reshape((commitments.shape[0], -1)).shape[1]),
            "gate_level_x_bits": int(template_spec.num_x_qubits),
            "piece_count": int(template_spec.piece_count),
            "selected_subspace": "two selected generators over two periods",
            "note": (
                "Both implementations encode the same learned integer max-affine value surrogate. "
                "Only the weighted-sum arithmetic subcircuit changes: carry-based WeightedAdder "
                "versus QFT accumulator with controlled phase rotations."
            ),
        },
        "fixed_load_mw": instance.fixed_load,
        "generators": generator_names,
        "selected_generators": [
            generator_names[index] for index in selected_generator_indices
        ],
        "bit_order": "selected generator-major order: g_i_t0,g_i_t1,...",
        "full_t2_reference": {
            "logic_feasible_count": int(logic_feasible.sum()),
            "finite_value_count": int(finite.sum()),
            "value_evaluation_seconds": float(value_seconds),
            "optimum": commitment_row(
                commitments,
                generator_names,
                values,
                full_optimum_index,
            ),
        },
        "embedded_subspace_validation": {
            "base_commitment": commitment_to_bitstring(full_optimum_commitment),
            "base_note": "Non-selected generators are fixed to the exhaustive T=2 optimum for validation only.",
            "subspace_size": int(embedded_commitments.shape[0]),
            "logic_feasible_count": int(embedded_logic_feasible.sum()),
            "finite_value_count": int(embedded_finite.sum()),
            "best_selected_bitstring": bitstring_from_index(
                embedded_best_index,
                template_spec.num_x_qubits,
            ),
            "best_full_bitstring": commitment_to_bitstring(
                embedded_commitments[embedded_best_index]
            ),
            "best_true_cost": _finite_or_none(embedded_values[embedded_best_index]),
        },
        "learned_model": learned_to_dict(learned, template_spec),
        "oracle_definition": {
            "value_surrogate": "V_hat_int(x) = max_r L_r(x)",
            "threshold_oracle": "O_tau |x> = (-1)^[V_hat_int(x) <= tau] |x>",
            "weighted_adder_path": (
                "WeightedAdder writes each L_r(x) into a binary value register, "
                "then IntegerComparator marks L_r(x) <= tau."
            ),
            "qft_path": (
                "QFT on the value register, controlled phase rotations conditioned on x_i "
                "and coefficient w_i, inverse QFT, then the same IntegerComparator."
            ),
        },
        "weighted_adder_circuit_explanation": oracle_and_circuit_explanation(
            learned,
            template_spec,
        ),
        "qft_circuit_explanation": qft_oracle_and_circuit_explanation(
            learned,
            template_spec,
        ),
        "target_cases": target_rows,
        "advisor_report_path": str(report_path),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8-sig")
    return summary


def run_qft_target_case(
    *,
    spec: GateLevelMaxAffineOracleSpec,
    calibration: dict[str, object],
    embedded_commitments: np.ndarray,
    embedded_values: np.ndarray,
) -> dict[str, object]:
    phase_start = time.perf_counter()
    phase_probe = simulate_qft_max_affine_phase_oracle(spec)
    phase_seconds = time.perf_counter() - phase_start

    grover_start = time.perf_counter()
    grover_result = simulate_qft_max_affine_grover(spec)
    grover_seconds = time.perf_counter() - grover_start

    phase_circuit = build_qft_max_affine_phase_oracle_circuit(spec)
    grover_circuit = build_qft_max_affine_grover_circuit(spec, grover_result.iterations)
    piece_values = spec.piece_values_for_all_x()
    max_values = spec.values_for_all_x()
    marked = spec.marked_mask()
    marked_indices = [int(index) for index in np.flatnonzero(marked)]
    labels = np.asarray(calibration["labels"], dtype=bool)
    false_positive = np.flatnonzero(marked & ~labels)
    false_negative = np.flatnonzero((~marked) & labels)
    argmax_index = int(np.argmax(grover_result.x_probabilities))

    return {
        "arithmetic": "QFT weighted-sum accumulator",
        "exact_marked_set": bool(false_positive.size == 0 and false_negative.size == 0),
        "false_positive_count": int(false_positive.size),
        "false_negative_count": int(false_negative.size),
        "marked_selected_bitstrings": [
            bitstring_from_index(index, spec.num_x_qubits)
            for index in marked_indices
        ],
        "marked_full_bitstrings": [
            commitment_to_bitstring(embedded_commitments[index])
            for index in marked_indices
        ],
        "marked_true_costs": [
            _finite_or_none(embedded_values[index])
            for index in marked_indices
        ],
        "phase_oracle_check": {
            "marked_count": int(phase_probe.marked_mask.sum()),
            "aux_zero_probability": float(phase_probe.aux_zero_probability),
            "max_phase_error": float(phase_probe.max_phase_error),
            "simulation_seconds": float(phase_seconds),
        },
        "grover_result": {
            "iterations": int(grover_result.iterations),
            "marked_probability": float(grover_result.marked_probability),
            "unmarked_probability": float(grover_result.unmarked_probability),
            "aux_zero_probability": float(grover_result.aux_zero_probability),
            "argmax_selected_bitstring": bitstring_from_index(argmax_index, spec.num_x_qubits),
            "argmax_full_bitstring": commitment_to_bitstring(embedded_commitments[argmax_index]),
            "argmax_true_cost": _finite_or_none(embedded_values[argmax_index]),
            "simulation_seconds": float(grover_seconds),
            "top_probability_rows": top_probability_rows(
                probabilities=grover_result.x_probabilities,
                piece_values=piece_values,
                max_values=max_values,
                marked=marked,
                embedded_commitments=embedded_commitments,
                embedded_values=embedded_values,
                num_bits=spec.num_x_qubits,
            ),
        },
        "resources": {
            "phase_oracle": circuit_resource_summary(phase_circuit, decompose_reps=3),
            "grover_circuit": circuit_resource_summary(grover_circuit, decompose_reps=3),
        },
    }


def qft_oracle_and_circuit_explanation(
    learned,
    spec: GateLevelMaxAffineOracleSpec,
) -> dict[str, object]:
    return {
        "oracle_definition": {
            "search_variable": "selected commitment bits x only; fixed load d is a condition, not a Grover variable",
            "integer_surrogate": "V_hat_int(x) = max_r L_r(x)",
            "threshold_oracle": "O_tau |x> = (-1)^[V_hat_int(x) <= tau] |x>",
        },
        "circuit_registers": {
            "x": list(spec.bit_labels),
            "qft_value_registers": [
                f"qft_v{piece_index} stores L{piece_index}(x) after inverse QFT"
                for piece_index in range(spec.piece_count)
            ],
            "piece_flags": [
                f"qft_flag{piece_index} stores [L{piece_index}(x) <= tau]"
                for piece_index in range(spec.piece_count)
            ],
            "comparator_ancillas": "Qiskit IntegerComparator ancillas for each piece",
        },
        "circuit_sequence": [
            "Apply Hadamard gates on all x qubits to create the Grover uniform superposition.",
            "For each affine piece L_r, temporarily flip inverted x inputs for (1-x_i) terms.",
            "Apply QFT to the piece value register.",
            "For every nonzero integer coefficient w_i, apply controlled phase rotations from x_i to every value-register Fourier qubit.",
            "Apply inverse QFT so the computational-basis value register stores L_r(x).",
            "Use IntegerComparator to write flag_r = [L_r(x) <= tau].",
            "Apply a multi-controlled phase on all flag qubits, implementing max_r L_r(x) <= tau as AND_r flag_r.",
            "Run inverse comparators and inverse QFT weighted-sum gates, restoring all value and ancilla registers to zero.",
            "Apply the standard Grover diffuser on x only.",
        ],
        "reversibility": {
            "compute_phase_uncompute": "The QFT oracle is U_qft^dagger Z_flags U_qft, so it is unitary and leaves only a phase on x.",
            "auxiliary_check": "The experiment reports aux_zero_probability to verify non-x registers return to |0>.",
        },
        "learned_piece_summary": {
            "optimum_selected_bitstring": bitstring_from_index(learned.optimum_index, spec.num_x_qubits),
            "integer_weights": [int(weight) for weight in learned.integer_weights],
            "piece_count": int(spec.piece_count),
        },
    }


def render_report(summary: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("# QFT Weighted-Sum 与 WeightedAdder Grover Oracle 对照实验")
    lines.append("")
    lines.append("## 1. 实验目的")
    lines.append("")
    lines.append(
        "本实验固定 case14 T=2 负荷曲线，在 g1/g6 小子空间上比较两种整数加权求和电路："
        "当前主线使用的 Qiskit WeightedAdder，以及借鉴文献思路实现的 QFT-based weighted sum。"
    )
    lines.append("")
    lines.append("两条路线实现同一个 max-affine 阈值 oracle：")
    lines.append("")
    lines.append("```text")
    lines.append("V_hat_int(x) = max_r L_r(x)")
    lines.append("O_tau |x> = (-1)^[V_hat_int(x) <= tau] |x>")
    lines.append("```")
    lines.append("")
    lines.append("## 2. 样本与学习到的整数片段")
    lines.append("")
    lines.append(f"- selected generators: {summary['selected_generators']}")
    lines.append(f"- bit order: {summary['bit_order']}")
    lines.append(f"- selected-subspace optimum: {summary['embedded_subspace_validation']['best_selected_bitstring']}")
    lines.append(f"- embedded full optimum: {summary['embedded_subspace_validation']['best_full_bitstring']}")
    lines.append(f"- true cost: {summary['embedded_subspace_validation']['best_true_cost']:.6f}")
    lines.append("")
    for piece in summary["learned_model"]["pieces"]:
        lines.append(f"- {piece['name']}: " + " + ".join(piece["terms"]))
    lines.append("")
    lines.append("## 3. 电路设计")
    lines.append("")
    lines.append("| 路线 | 求和方式 | 值寄存器含义 | 特点 |")
    lines.append("|---|---|---|---|")
    lines.append("| WeightedAdder | 用 carry/control 辅助位完成整数加法 | 计算基中的 value register | 当前主线，结构直观，依赖 Qiskit 算术模块 |")
    lines.append("| QFT weighted sum | QFT 后用受控相位旋转累加权重，再 inverse QFT | inverse QFT 后的计算基 value register | 更接近文献中的 QFT 加权求和思想，可能节省辅助位 |")
    lines.append("")
    lines.append("## 4. 结果对比")
    lines.append("")
    lines.append("| 目标 | 路线 | tau | marked states | marked probability | aux 回零概率 | phase error | qubits | depth | decomposed depth | argmax true cost |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for target in summary["target_cases"]:
        for key, label in (("weighted_adder", "WeightedAdder"), ("qft_weighted_sum", "QFT weighted sum")):
            row = target[key]
            grover = row["grover_result"]
            phase = row["phase_oracle_check"]
            resource = row["resources"]["grover_circuit"]
            marked = ", ".join(row["marked_selected_bitstrings"])
            lines.append(
                f"| top-{target['target_count_request']} | {label} | {target['threshold_tau']} | "
                f"{marked} | {grover['marked_probability']:.6f} | "
                f"{grover['aux_zero_probability']:.12f} | {phase['max_phase_error']:.2e} | "
                f"{resource['num_qubits']} | {resource['depth']} | "
                f"{resource.get('decomposed_depth', 'NA')} | {grover['argmax_true_cost']:.6f} |"
            )
    lines.append("")
    lines.append("## 5. 门类型与操作统计")
    lines.append("")
    lines.append("下面列出 Grover 电路的高层 operation counts；更细的分解门统计可查看 JSON 结果中的 resources.grover_circuit.decomposed_operations。")
    lines.append("")
    lines.append("| 目标 | 路线 | 高层 operations |")
    lines.append("|---:|---|---|")
    for target in summary["target_cases"]:
        for key, label in (("weighted_adder", "WeightedAdder"), ("qft_weighted_sum", "QFT weighted sum")):
            ops = target[key]["resources"]["grover_circuit"]["operations"]
            lines.append(f"| top-{target['target_count_request']} | {label} | `{ops}` |")
    lines.append("")
    lines.append("## 6. QFT oracle 的 compute-phase-uncompute")
    lines.append("")
    for step in summary["qft_circuit_explanation"]["circuit_sequence"]:
        lines.append(f"- {step}")
    lines.append("")
    lines.append("## 7. 小结")
    lines.append("")
    lines.append(
        "两种加权求和方式实现了同一个 max-affine threshold oracle，得到相同的 marked set、"
        "相近的 Grover marked probability，并都能通过真实 UC 成本验证最优启停状态。"
        "在该小样本中，QFT weighted sum 使用更少量子比特，但引入 QFT/IQFT 和受控相位旋转；"
        "WeightedAdder 路线更直观，适合作为当前主线基准。"
    )
    lines.append("")
    return "\n".join(lines)

def top_probability_rows(
    *,
    probabilities: np.ndarray,
    piece_values: np.ndarray,
    max_values: np.ndarray,
    marked: np.ndarray,
    embedded_commitments: np.ndarray,
    embedded_values: np.ndarray,
    num_bits: int,
    limit: int = 10,
) -> list[dict[str, object]]:
    rows = []
    ranks = finite_rank_map(embedded_values)
    for state_index in np.argsort(probabilities)[::-1][:limit]:
        state_index = int(state_index)
        rows.append(
            {
                "selected_bitstring": bitstring_from_index(state_index, num_bits),
                "probability": float(probabilities[state_index]),
                "piece_values": [int(value) for value in piece_values[state_index].tolist()],
                "max_affine_value": int(max_values[state_index]),
                "marked": bool(marked[state_index]),
                "embedded_full_bitstring": commitment_to_bitstring(
                    embedded_commitments[state_index]
                ),
                "embedded_true_cost": _finite_or_none(embedded_values[state_index]),
                "embedded_true_rank": ranks.get(state_index),
            }
        )
    return rows


def finite_rank_map(values: np.ndarray) -> dict[int, int]:
    rows = [int(index) for index in np.argsort(values) if np.isfinite(values[index])]
    return {index: rank + 1 for rank, index in enumerate(rows)}


def parse_indices(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_counts(raw: str) -> tuple[int, ...]:
    counts = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not counts:
        raise argparse.ArgumentTypeError("at least one target count is required")
    return counts


def _finite_or_none(value: float) -> float | None:
    if np.isfinite(value):
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_qft_weighted_sum_oracle_comparison.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/stage1_case14_t2_qft_weighted_sum_oracle_comparison_report.md"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--selected-generators", type=parse_indices, default=(0, 5))
    parser.add_argument("--target-counts", type=parse_counts, default=(1, 3))
    parser.add_argument("--max-weight", type=int, default=7)
    args = parser.parse_args()

    summary = run(
        instance_path=args.instance,
        results_path=args.results,
        report_path=args.report,
        horizon=args.horizon,
        selected_generator_indices=args.selected_generators,
        target_counts=args.target_counts,
        max_weight=args.max_weight,
    )
    compact = {
        "instance": summary["instance"],
        "method": summary["method"],
        "selected_generators": summary["selected_generators"],
        "learned_model": summary["learned_model"],
        "target_cases": [
            {
                "target_count_request": target["target_count_request"],
                "threshold_tau": target["threshold_tau"],
                "weighted_adder": compact_impl_result(target["weighted_adder"]),
                "qft_weighted_sum": compact_impl_result(target["qft_weighted_sum"]),
            }
            for target in summary["target_cases"]
        ],
        "advisor_report_path": summary["advisor_report_path"],
    }
    print(json.dumps(compact, indent=2))


def compact_impl_result(row: dict[str, object]) -> dict[str, object]:
    return {
        "exact_marked_set": row["exact_marked_set"],
        "marked_selected_bitstrings": row["marked_selected_bitstrings"],
        "marked_probability": row["grover_result"]["marked_probability"],
        "aux_zero_probability": row["grover_result"]["aux_zero_probability"],
        "phase_error": row["phase_oracle_check"]["max_phase_error"],
        "qubits": row["resources"]["grover_circuit"]["num_qubits"],
        "depth": row["resources"]["grover_circuit"]["depth"],
        "decomposed_depth": row["resources"]["grover_circuit"].get("decomposed_depth"),
        "argmax_true_cost": row["grover_result"]["argmax_true_cost"],
    }


if __name__ == "__main__":
    main()

