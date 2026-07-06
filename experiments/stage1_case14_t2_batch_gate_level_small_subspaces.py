from __future__ import annotations

import argparse
from itertools import combinations
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
    learn_gap_weighted_max_affine_pieces,
    learned_to_dict,
    oracle_and_circuit_explanation,
    run_target_case,
)
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.gate_level_oracle import (  # noqa: E402
    GateLevelMaxAffineOracleSpec,
    bitstring_from_index,
    build_max_affine_grover_circuit,
    build_max_affine_phase_oracle_circuit,
    circuit_resource_summary,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def run(
    *,
    instance_path: Path,
    results_path: Path,
    report_path: Path,
    horizon: int,
    pair_specs: tuple[tuple[int, int], ...] | None,
    screen_target_counts: tuple[int, ...],
    gate_target_counts: tuple[int, ...],
    max_weight: int,
    max_gate_level_cases: int,
) -> dict[str, object]:
    if horizon != 2:
        raise ValueError("this batch gate-level experiment is restricted to T=2")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    if max_gate_level_cases < 0:
        raise ValueError("max_gate_level_cases must be nonnegative")

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

    if pair_specs is None:
        pair_specs = tuple(combinations(range(len(instance.generators)), 2))
    pair_specs = tuple(validate_pair(pair, len(instance.generators)) for pair in pair_specs)

    screened_cases = []
    for pair in pair_specs:
        screened_cases.append(
            screen_pair_case(
                instance=instance,
                generator_names=generator_names,
                full_optimum_commitment=full_optimum_commitment,
                selected_generator_indices=pair,
                target_counts=screen_target_counts,
                max_weight=max_weight,
            )
        )

    selected_for_gate = select_gate_level_cases(screened_cases, max_gate_level_cases, gate_target_counts)
    gate_level_case_rows = []
    for screened in selected_for_gate:
        gate_level_case_rows.append(
            run_gate_level_case(
                screened=screened,
                target_counts=gate_target_counts,
            )
        )

    summary = {
        "instance": f"case14_T{horizon}",
        "source": str(instance_path),
        "method": "batch T=2 true-cost-learned small max-affine gate-level Grover oracle validation",
        "scope": {
            "full_case14_bits": int(commitments.reshape((commitments.shape[0], -1)).shape[1]),
            "screened_subspace_count": int(len(screened_cases)),
            "gate_level_subspace_count": int(len(gate_level_case_rows)),
            "subspace_type": "2 selected generators over 2 periods, giving 4 Grover x qubits per gate-level case",
            "note": (
                "The batch screens multiple small T=2 subspaces and runs full Qiskit statevector "
                "gate-level Grover simulations only for selected high-quality small subspaces."
            ),
        },
        "fixed_load_mw": instance.fixed_load,
        "generators": generator_names,
        "bit_order": "selected generator-major order: g_i_t0,g_i_t1,...",
        "learning_rule": {
            "summary": (
                "For each selected subspace, find the true best embedded commitment, flip each selected "
                "bit once, use exact UC cost increases as local sensitivities, quantize to nonnegative "
                "integer weights, and build generator-wise max-affine mismatch pieces."
            ),
            "max_integer_weight": int(max_weight),
            "encoded_object": "small integer max-affine value surrogate; not a true-cost lookup table",
        },
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
        "screening_summary": summarize_screening(screened_cases, screen_target_counts),
        "screened_cases": [compact_screened_case(case) for case in screened_cases],
        "gate_level_cases": gate_level_case_rows,
        "advisor_report_path": str(report_path),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8-sig")
    return summary


def screen_pair_case(
    *,
    instance,
    generator_names: list[str],
    full_optimum_commitment: np.ndarray,
    selected_generator_indices: tuple[int, int],
    target_counts: tuple[int, ...],
    max_weight: int,
) -> dict[str, object]:
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
        name="case14_t2_batch_learned_gap_weighted_max_affine_oracle",
    )
    predicted_values = template_spec.values_for_all_x()
    calibrations = [
        calibrate_integer_threshold(
            predicted_values=predicted_values,
            true_values=embedded_values,
            target_count=int(target_count),
        )
        for target_count in target_counts
    ]
    top1_resource_estimate = estimate_top1_resources(template_spec)
    pair_names = tuple(generator_names[index] for index in selected_generator_indices)
    finite_order = [int(index) for index in np.argsort(embedded_values) if np.isfinite(embedded_values[index])]
    return {
        "selected_generator_indices": tuple(int(index) for index in selected_generator_indices),
        "selected_generators": pair_names,
        "embedded_commitments": embedded_commitments,
        "embedded_values": embedded_values,
        "embedded_logic_feasible": embedded_logic_feasible,
        "embedded_finite": embedded_finite,
        "embedded_best_index": int(embedded_best_index),
        "embedded_best_selected_bitstring": bitstring_from_index(embedded_best_index, template_spec.num_x_qubits),
        "embedded_best_full_bitstring": commitment_to_bitstring(embedded_commitments[embedded_best_index]),
        "embedded_best_true_cost": _finite_or_none(embedded_values[embedded_best_index]),
        "embedded_true_order_preview": [
            {
                "rank": int(rank + 1),
                "selected_bitstring": bitstring_from_index(index, template_spec.num_x_qubits),
                "full_bitstring": commitment_to_bitstring(embedded_commitments[index]),
                "true_cost": _finite_or_none(embedded_values[index]),
            }
            for rank, index in enumerate(finite_order[: min(5, len(finite_order))])
        ],
        "subspace_size": int(embedded_commitments.shape[0]),
        "logic_feasible_count": int(embedded_logic_feasible.sum()),
        "finite_value_count": int(embedded_finite.sum()),
        "learned": learned,
        "template_spec": template_spec,
        "top1_resource_estimate": top1_resource_estimate,
        "learned_model": learned_to_dict(learned, template_spec),
        "oracle_and_circuit_explanation": oracle_and_circuit_explanation(learned, template_spec),
        "target_calibrations": [compact_calibration(calibration) for calibration in calibrations],
        "all_requested_targets_exact": bool(
            all(calibration["selected_row"]["exact_marked_set"] for calibration in calibrations)
        ),
        "exact_target_count": int(
            sum(bool(calibration["selected_row"]["exact_marked_set"]) for calibration in calibrations)
        ),
    }


def run_gate_level_case(
    *,
    screened: dict[str, object],
    target_counts: tuple[int, ...],
) -> dict[str, object]:
    learned = screened["learned"]
    embedded_commitments = screened["embedded_commitments"]
    embedded_values = screened["embedded_values"]
    template_spec = screened["template_spec"]
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
            name=(
                "case14_t2_batch_learned_gap_weighted_"
                f"{screened['selected_generators'][0]}_{screened['selected_generators'][1]}_"
                f"top_{int(target_count)}_oracle"
            ),
        )
        target_rows.append(
            run_target_case(
                spec=spec,
                calibration=calibration,
                embedded_commitments=embedded_commitments,
                embedded_values=embedded_values,
            )
        )
    return {
        "selected_generator_indices": list(screened["selected_generator_indices"]),
        "selected_generators": list(screened["selected_generators"]),
        "gate_level_x_bits": int(template_spec.num_x_qubits),
        "piece_count": int(template_spec.piece_count),
        "subspace_size": int(screened["subspace_size"]),
        "logic_feasible_count": int(screened["logic_feasible_count"]),
        "finite_value_count": int(screened["finite_value_count"]),
        "embedded_best_selected_bitstring": screened["embedded_best_selected_bitstring"],
        "embedded_best_full_bitstring": screened["embedded_best_full_bitstring"],
        "embedded_best_true_cost": screened["embedded_best_true_cost"],
        "learned_model": screened["learned_model"],
        "oracle_and_circuit_explanation": screened["oracle_and_circuit_explanation"],
        "embedded_true_order_preview": screened["embedded_true_order_preview"],
        "target_cases": target_rows,
    }


def select_gate_level_cases(
    screened_cases: list[dict[str, object]],
    max_gate_level_cases: int,
    gate_target_counts: tuple[int, ...],
) -> list[dict[str, object]]:
    ordered = sorted(
        screened_cases,
        key=lambda item: (
            target_error_count(item, gate_target_counts),
            int(item["top1_resource_estimate"]["grover_circuit"]["num_qubits"]),
            not bool(item["all_requested_targets_exact"]),
            -int(item["exact_target_count"]),
            tuple(item["selected_generator_indices"]),
        ),
    )
    return ordered[:max_gate_level_cases]



def estimate_top1_resources(spec: GateLevelMaxAffineOracleSpec) -> dict[str, object]:
    top1_spec = GateLevelMaxAffineOracleSpec(
        pieces=spec.pieces,
        threshold=0,
        bit_labels=spec.bit_labels,
        name=f"{spec.name}_top1_resource_estimate",
    )
    return {
        "phase_oracle": circuit_resource_summary(
            build_max_affine_phase_oracle_circuit(top1_spec),
            decompose_reps=0,
        ),
        "grover_circuit": circuit_resource_summary(
            build_max_affine_grover_circuit(top1_spec),
            decompose_reps=0,
        ),
    }


def target_error_count(case: dict[str, object], gate_target_counts: tuple[int, ...]) -> int:
    requested = {int(count) for count in gate_target_counts}
    total = 0
    for calibration in case["target_calibrations"]:
        if int(calibration["target_count_request"]) not in requested:
            continue
        total += int(calibration["selected_row"]["error_count"])
    return total
def summarize_screening(
    screened_cases: list[dict[str, object]],
    target_counts: tuple[int, ...],
) -> dict[str, object]:
    exact_by_target = {}
    for target_count in target_counts:
        exact_by_target[str(int(target_count))] = int(
            sum(
                any(
                    calibration["target_count_request"] == int(target_count)
                    and calibration["selected_row"]["exact_marked_set"]
                    for calibration in case["target_calibrations"]
                )
                for case in screened_cases
            )
        )
    return {
        "screened_subspace_count": int(len(screened_cases)),
        "target_counts": [int(count) for count in target_counts],
        "exact_by_target_count": exact_by_target,
        "all_requested_targets_exact_count": int(
            sum(bool(case["all_requested_targets_exact"]) for case in screened_cases)
        ),
        "selected_gate_level_rule": (
            "prefer subspaces with exact requested gate targets and lower estimated qubit count; "
            "then use screening exactness and generator-index order for deterministic selection"
        ),
    }


def compact_screened_case(case: dict[str, object]) -> dict[str, object]:
    return {
        "selected_generator_indices": list(case["selected_generator_indices"]),
        "selected_generators": list(case["selected_generators"]),
        "gate_level_x_bits": int(case["template_spec"].num_x_qubits),
        "piece_count": int(case["template_spec"].piece_count),
        "top1_resource_estimate": case["top1_resource_estimate"],
        "subspace_size": int(case["subspace_size"]),
        "logic_feasible_count": int(case["logic_feasible_count"]),
        "finite_value_count": int(case["finite_value_count"]),
        "embedded_best_selected_bitstring": case["embedded_best_selected_bitstring"],
        "embedded_best_full_bitstring": case["embedded_best_full_bitstring"],
        "embedded_best_true_cost": case["embedded_best_true_cost"],
        "learned_model": case["learned_model"],
        "oracle_and_circuit_explanation": case["oracle_and_circuit_explanation"],
        "target_calibrations": case["target_calibrations"],
        "all_requested_targets_exact": bool(case["all_requested_targets_exact"]),
        "exact_target_count": int(case["exact_target_count"]),
        "embedded_true_order_preview": case["embedded_true_order_preview"],
    }


def compact_calibration(calibration: dict[str, object]) -> dict[str, object]:
    return {
        "target_count_request": int(calibration["target_count_request"]),
        "actual_target_count": int(calibration["actual_target_count"]),
        "selected_threshold": int(calibration["selected_threshold"]),
        "selected_row": calibration["selected_row"],
        "target_selected_bitstrings": calibration["target_selected_bitstrings"],
    }


def render_report(summary: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("# T=2 固定负荷 case14 小样本门级 Grover Oracle 批量验证报告")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append(
        "在固定负荷曲线 d 下，批量验证多个 T=2 小子空间中的 Grover value-function oracle。"
        "搜索变量是选定机组的启停承诺 x，负荷 d 和未选机组承诺只作为条件量/嵌入验证量。"
    )
    lines.append("")
    lines.append("核心 oracle 形式：")
    lines.append("")
    lines.append("```text")
    lines.append("V_hat_int(x) = max_r L_r(x)")
    lines.append("O_tau |x> = (-1)^[V_hat_int(x) <= tau] |x>")
    lines.append("```")
    lines.append("")
    lines.append("## 2. 数据与全局参照")
    lines.append("")
    ref = summary["full_t2_reference"]
    optimum = ref["optimum"]
    lines.append(f"- 算例：{summary['instance']}，来源：`{summary['source']}`")
    lines.append(f"- 固定负荷 MW：{summary['fixed_load_mw']}")
    lines.append(f"- 完整 T=2 commitment bits：{summary['scope']['full_case14_bits']}")
    lines.append(f"- 完整枚举有限可行状态数：{ref['finite_value_count']}")
    lines.append(
        "- 完整 T=2 穷举最优："
        f"{optimum['bitstring_generator_major']}，真实成本 {optimum['total_cost']:.6f}"
    )
    lines.append("")
    lines.append("## 3. 批量筛选结果")
    lines.append("")
    screening = summary["screening_summary"]
    lines.append(f"- 筛选子空间数：{screening['screened_subspace_count']}")
    lines.append(f"- 请求目标集合大小：{screening['target_counts']}")
    lines.append(f"- 各目标集合精确分离数量：{screening['exact_by_target_count']}")
    lines.append(f"- 同时精确分离全部请求目标的子空间数：{screening['all_requested_targets_exact_count']}")
    lines.append("")
    lines.append("| 子空间 | 最优 selected bitstring | 整数权重 | top-1 | top-3 |")
    lines.append("|---|---:|---:|---:|---:|")
    for case in summary["screened_cases"]:
        target_map = {
            item["target_count_request"]: item["selected_row"]["exact_marked_set"]
            for item in case["target_calibrations"]
        }
        top1 = "精确" if target_map.get(1, False) else "不精确"
        top3 = "精确" if target_map.get(3, False) else "不精确"
        weights = case["learned_model"]["integer_weights"]
        lines.append(
            "| "
            + "/".join(case["selected_generators"])
            + f" | {case['embedded_best_selected_bitstring']} | {weights} | {top1} | {top3} |"
        )
    lines.append("")
    lines.append("## 4. 门级 Grover 仿真结果")
    lines.append("")
    lines.append("下面这些子空间完成了 Qiskit statevector 门级 oracle + Grover diffuser 仿真。")
    lines.append("")
    lines.append("| 子空间 | 目标 | tau | 标记 selected states | Grover 后标记概率 | aux 回零概率 | qubits | depth | argmax 真实成本 |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for case in summary["gate_level_cases"]:
        subspace = "/".join(case["selected_generators"])
        for target in case["target_cases"]:
            grover = target["grover_result"]
            phase = target["phase_oracle_check"]
            resources = target["resources"]["grover_circuit"]
            marked = ", ".join(target["marked_selected_bitstrings"])
            lines.append(
                f"| {subspace} | top-{target['target_count_request']} | {target['threshold_tau']} | "
                f"{marked} | {grover['marked_probability']:.6f} | "
                f"{grover['aux_zero_probability']:.12f} | {resources['num_qubits']} | "
                f"{resources['depth']} | {grover['argmax_true_cost']:.6f} |"
            )
            _ = phase
    lines.append("")
    lines.append("## 5. 每个门级实验的电路与 oracle 说明")
    for index, case in enumerate(summary["gate_level_cases"], start=1):
        explanation = case["oracle_and_circuit_explanation"]
        lines.append("")
        lines.append(f"### 5.{index} 子空间 {'/'.join(case['selected_generators'])}")
        lines.append("")
        lines.append(f"- 搜索寄存器 x：{explanation['circuit_registers']['x']}")
        lines.append(f"- 学得整数权重：{case['learned_model']['integer_weights']}")
        lines.append(f"- 值函数代理：{explanation['oracle_definition']['integer_surrogate']}")
        lines.append(f"- 阈值 oracle：{explanation['oracle_definition']['threshold_oracle']}")
        lines.append("- 仿射片段：")
        for piece in case["learned_model"]["pieces"]:
            lines.append(f"  - {piece['name']}: " + " + ".join(piece["terms"]))
        lines.append("- 电路寄存器：")
        lines.append(f"  - x：{explanation['circuit_registers']['x']}")
        lines.append(f"  - value registers：{explanation['circuit_registers']['piece_value_registers']}")
        lines.append(f"  - flag qubits：{explanation['circuit_registers']['piece_flags']}")
        lines.append("- compute-phase-uncompute 顺序：")
        for step in explanation["circuit_sequence"]:
            lines.append(f"  - {step}")
        lines.append(f"- 可逆性检查：{explanation['reversibility']['compute_phase_uncompute']}")
        lines.append(f"- 真实成本验证：{explanation['grover_loop']['true_cost_validation']}")
    lines.append("")
    lines.append("## 6. 当前结论")
    lines.append("")
    lines.append(
        "当前批量结果说明，在 T=2 固定负荷 case14 的多个小子空间中，"
        "可以从真实 UC 成本局部敏感性构造非查表的整数 max-affine oracle，"
        "并在门级 Qiskit 电路中完成 compute -> compare -> phase -> uncompute -> diffuse。"
    )
    lines.append(
        "该结果仍是小样本 proof-of-concept，不声称已经完成完整 12-bit、207-feature、32-piece T=2 门级综合；"
        "它的作用是证明研究内容1的 oracle 架构、可逆性、Grover 放大和真实 UC 成本验证链条是闭合的。"
    )
    lines.append("")
    return "\n".join(lines)


def validate_pair(pair: tuple[int, int], generator_count: int) -> tuple[int, int]:
    if len(pair) != 2:
        raise ValueError("each gate-level batch subspace must contain exactly two generators")
    first, second = int(pair[0]), int(pair[1])
    if first == second:
        raise ValueError("selected generator pair must not repeat an index")
    if first < 0 or second < 0 or first >= generator_count or second >= generator_count:
        raise ValueError("selected generator index is out of range")
    return tuple(sorted((first, second)))


def parse_pair_list(raw: str) -> tuple[tuple[int, int], ...] | None:
    raw = raw.strip()
    if not raw or raw.lower() == "all":
        return None
    pairs = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [int(part.strip()) for part in item.split(",") if part.strip()]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("pairs must have the form '0,5;0,3'")
        pairs.append((parts[0], parts[1]))
    if not pairs:
        return None
    return tuple(pairs)


def parse_counts(raw: str) -> tuple[int, ...]:
    counts = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not counts:
        raise argparse.ArgumentTypeError("at least one target count is required")
    if any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("target counts must be positive")
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
        default=Path("results/stage1_case14_t2_batch_gate_level_small_subspaces.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/stage1_case14_t2_batch_gate_level_small_subspaces_report.md"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--pairs", type=parse_pair_list, default=None)
    parser.add_argument("--screen-target-counts", type=parse_counts, default=(1, 3))
    parser.add_argument("--gate-target-counts", type=parse_counts, default=(1,))
    parser.add_argument("--max-weight", type=int, default=7)
    parser.add_argument("--max-gate-level-cases", type=int, default=3)
    args = parser.parse_args()

    summary = run(
        instance_path=args.instance,
        results_path=args.results,
        report_path=args.report,
        horizon=args.horizon,
        pair_specs=args.pairs,
        screen_target_counts=args.screen_target_counts,
        gate_target_counts=args.gate_target_counts,
        max_weight=args.max_weight,
        max_gate_level_cases=args.max_gate_level_cases,
    )
    compact = {
        "instance": summary["instance"],
        "method": summary["method"],
        "scope": summary["scope"],
        "screening_summary": summary["screening_summary"],
        "gate_level_cases": [
            {
                "selected_generators": case["selected_generators"],
                "learned_integer_weights": case["learned_model"]["integer_weights"],
                "target_cases": [
                    {
                        "target_count_request": target["target_count_request"],
                        "threshold_tau": target["threshold_tau"],
                        "exact_marked_set": target["exact_marked_set"],
                        "marked_selected_bitstrings": target["marked_selected_bitstrings"],
                        "marked_probability": target["grover_result"]["marked_probability"],
                        "aux_zero_probability": target["grover_result"]["aux_zero_probability"],
                        "qubits": target["resources"]["grover_circuit"]["num_qubits"],
                        "depth": target["resources"]["grover_circuit"]["depth"],
                    }
                    for target in case["target_cases"]
                ],
            }
            for case in summary["gate_level_cases"]
        ],
        "advisor_report_path": summary["advisor_report_path"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()




