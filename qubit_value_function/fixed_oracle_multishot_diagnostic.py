"""Fixed-oracle multi-shot Grover diagnostic, isolated from closed-loop search.

Only persisted formal training labels are used to deterministically reconstruct a
small sparse VQC.  This module never imports an ED/LP evaluator or a closed-loop
runner.  Global truth is read solely from validation after reconstruction for
metadata, never to choose labels, thresholds, or marked states.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .coherent_phase_value import basis_value_code_probe, phase_to_value_superposition_probe, quantize_sparse_phase_model
from .fixed_point_oracle import FixedPointConfig
from .gate_level_oracle import bitstring_from_index, circuit_resource_summary
from .logic_feasibility_oracle import compile_logic_feasibility_spec
from .sparse_phase_vqc import fit_sparse_phase_vqc
from .sparse_vqc_grover import (
    build_sparse_vqc_grover_circuit,
    execute_sparse_vqc_grover_mps,
    ordinary_grover_validation_plan,
)


DIMENSION = 16
SHOTS = 1024


class FixedOracleDiagnosticError(RuntimeError):
    pass


def ideal_grover_probability(marked_count: int, iterations: int, *, dimension: int = DIMENSION) -> float:
    marked_count, iterations, dimension = int(marked_count), int(iterations), int(dimension)
    if not 0 <= marked_count <= dimension or iterations < 0:
        raise ValueError("M/k/N 无效")
    return float(math.sin((2 * iterations + 1) * math.asin(math.sqrt(marked_count / dimension))) ** 2)


def strict_joint_marked_indices(values: Sequence[int], feasible: Sequence[bool], threshold: int) -> tuple[int, ...]:
    if len(values) != len(feasible): raise ValueError("values/feasible 长度不一致")
    return tuple(index for index, (value, ok) in enumerate(zip(values, feasible)) if int(value) < int(threshold) and bool(ok))


def wilson_interval(hits: int, shots: int, z: float = 1.959963984540054) -> tuple[float, float]:
    hits, shots = int(hits), int(shots)
    if shots <= 0 or not 0 <= hits <= shots: raise ValueError("Wilson 参数无效")
    p = hits / shots; denominator = 1 + z * z / shots
    centre = (p + z * z / (2 * shots)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * shots)) / shots) / denominator
    return float(centre - radius), float(centre + radius)


def validate_training_truth_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    protocol = metadata.get("training_selection_protocol", "random_truth_blind")
    if protocol not in {"random_truth_blind", "controlled_optimum_holdout"}:
        raise FixedOracleDiagnosticError("未知 training_selection_protocol")
    online = bool(metadata.get("global_truth_used_online", False))
    holdout = bool(metadata.get("controlled_optimum_holdout", False))
    if online: raise FixedOracleDiagnosticError("global truth 不得用于在线训练/阈值/oracle")
    if protocol == "controlled_optimum_holdout" and not holdout:
        raise FixedOracleDiagnosticError("controlled_optimum_holdout 必须显式标记")
    if protocol == "random_truth_blind" and holdout:
        raise FixedOracleDiagnosticError("random_truth_blind 不得伪装 controlled holdout")
    return {"training_selection_protocol": protocol, "controlled_optimum_holdout": holdout,
            "global_truth_used_online": False, "global_truth_used_for_posthoc_validation": True}


def _load(path: Path) -> dict[str, Any]:
    try: data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error: raise FixedOracleDiagnosticError(f"无法读取 {path}: {error}") from error
    if not isinstance(data, dict): raise FixedOracleDiagnosticError(f"JSON 根对象无效: {path}")
    return data


def _bits(index: int, width: int = 4) -> tuple[int, ...]: return tuple((int(index) >> bit) & 1 for bit in range(width))


def _find_run(completed: Path, scenario_id: str) -> dict[str, Any]:
    candidates = []
    for path in sorted(completed.glob("*joint_bbht_run0.json")):
        item = _load(path)
        if item.get("scenario", {}).get("scenario_id") == scenario_id: candidates.append(item)
    if len(candidates) != 1: raise FixedOracleDiagnosticError(f"{scenario_id} 的 joint_bbht run0 不唯一/缺失")
    return candidates[0]


def _reconstruct_model(run: Mapping[str, Any], *, source, instance_path: str | Path) -> tuple[Any, Any, dict[str, Any]]:
    """Only uses persisted training index/cost cache and deterministic fit settings."""
    scenario, result, spec = run["scenario"], run["result"], run["run_spec"]
    if not isinstance(scenario, Mapping) or not isinstance(result, Mapping) or not isinstance(spec, Mapping):
        raise FixedOracleDiagnosticError("formal run schema 无效")
    training = [int(index) for index in scenario["training_indices"]]
    exact_cache = result.get("exact_cache", {})
    if not isinstance(exact_cache, Mapping): raise FixedOracleDiagnosticError("formal run 缺 exact_cache")
    labels: list[float] = []
    for index in training:
        record = exact_cache.get(str(index))
        if not isinstance(record, Mapping) or not record.get("success") or record.get("total_cost") is None:
            raise FixedOracleDiagnosticError(f"训练标签缺失: {scenario['scenario_id']} index={index}")
        labels.append(float(record["total_cost"]))
    # Original Stage-A builder uses seed + window, fixed 1e-4 regularization and 300 iterations.
    window, seed = int(scenario["window_start"]), int(scenario["training_seed"])
    fit = fit_sparse_phase_vqc(bitstrings=[_bits(index) for index in training], costs=labels, num_generators=2,
                               num_periods=2, generator_edges=((0, 1),), seed=seed + window, regularization=1e-4, maxiter=300)
    fixed = spec["fixed_point_config"]
    model = quantize_sparse_phase_model(fit.model, FixedPointConfig(fractional_bits=int(fixed["fractional_bits"]), unit=float(fixed["cost_unit"]), rounding=str(fixed["rounding"])))
    from .experiment_utils import embedded_selected_commitments, time_window_instance
    instance = time_window_instance(source, start=window, horizon=2)
    base = np.ones((len(instance.generators), 2), dtype=int)
    pair = tuple(int(value) for value in scenario["generator_pair"])
    # embedded commitments is intentionally not evaluated; it only establishes the fixed hard-logic compilation context.
    _ = embedded_selected_commitments(base, pair)
    feasibility = compile_logic_feasibility_spec(instance, selected_generator_indices=pair, base_commitment=base)
    history = result.get("threshold_history", [])
    if not isinstance(history, list) or not history: raise FixedOracleDiagnosticError("formal run 缺初始 threshold_history")
    threshold = int(history[0]["encoded_threshold"])
    values = [int(model.integer_value(_bits(index))) for index in range(DIMENSION)]
    feasible = [bool(feasibility.is_feasible(_bits(index))) for index in range(DIMENSION)]
    cost_marked = tuple(index for index, value in enumerate(values) if value < threshold)
    joint_marked = strict_joint_marked_indices(values, feasible, threshold)
    artifact = {"scenario_id": scenario["scenario_id"], "generator_pair": list(pair), "window_start": window, "training_seed": seed,
                "training_indices": training, "training_labels": labels, "training_truth": validate_training_truth_metadata({"training_selection_protocol": "random_truth_blind", "global_truth_used_online": False}),
                "quantized_model": model.as_dict(), "integer_values": values, "hard_logic_feasible": feasible,
                "encoded_threshold": threshold, "cost_marked_indices": list(cost_marked), "joint_marked_indices": list(joint_marked)}
    return model, feasibility, artifact


def _verify_reconstruction(artifact: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    expected_cost = list(validation.get("initial_cost_marked_indices", [])); expected_joint = list(validation.get("initial_joint_marked_indices", []))
    if artifact["cost_marked_indices"] != expected_cost or artifact["joint_marked_indices"] != expected_joint:
        raise FixedOracleDiagnosticError(f"model_reconstruction_mismatch: {artifact['scenario_id']}")


def _gate_counts(circuit) -> dict[str, int]: return {str(key): int(value) for key, value in circuit.count_ops().items()}


def _aer_exact_x_probabilities(circuit, *, num_x_qubits: int) -> tuple[np.ndarray, float]:
    """No-shot C++ statevector execution of the supplied full gate-level circuit."""
    exact = circuit.copy(); exact.save_statevector()
    backend = AerSimulator(method="statevector")
    compiled = transpile(exact, backend, optimization_level=1, seed_transpiler=0)
    vector = np.asarray(backend.run(compiled).result().data(compiled)["statevector"], dtype=complex)
    probabilities = np.abs(vector) ** 2; dimension = 2 ** int(num_x_qubits)
    x = np.zeros(dimension, dtype=float)
    for index, probability in enumerate(probabilities): x[index & (dimension - 1)] += float(probability)
    return x, float(np.sum(probabilities[:dimension]))


def _phase_oracle_probe(model, feasibility, threshold: int) -> dict[str, object]:
    from .logic_feasibility_oracle import build_joint_feasible_better_phase_oracle
    oracle = build_joint_feasible_better_phase_oracle(model, encoded_threshold=int(threshold), feasibility_spec=feasibility)
    circuit = QuantumCircuit(oracle.num_qubits); circuit.h(list(circuit.qubits[:model.num_x_qubits])); circuit.compose(oracle, inplace=True)
    exact = circuit.copy(); exact.save_statevector(); backend = AerSimulator(method="statevector")
    compiled = transpile(exact, backend, optimization_level=1, seed_transpiler=0)
    vector = np.asarray(backend.run(compiled).result().data(compiled)["statevector"], dtype=complex)
    amplitude = 1 / math.sqrt(DIMENSION); expected = []
    for index in range(DIMENSION):
        marked = model.is_marked(_bits(index), int(threshold), strict=True) and feasibility.is_feasible(_bits(index))
        expected.append(-1 if marked else 1)
    reference = np.asarray(expected, dtype=complex) * amplitude
    global_phase = vector[0] / reference[0]
    relative_error = float(np.max(np.abs(vector[:DIMENSION] - global_phase * reference)))
    all_auxiliary_zero = float(np.sum(np.abs(vector[:DIMENSION]) ** 2))
    return {"relative_phase_max_error": relative_error, "global_phase_real": float(np.real(global_phase)),
            "global_phase_imag": float(np.imag(global_phase)), "auxiliary_zero_probability": all_auxiliary_zero,
            "phase_truth_table": [{"index": index, "expected_sign": expected[index],
                                   "actual_relative_phase_real": float(np.real(vector[index] / (amplitude * global_phase))),
                                   "actual_relative_phase_imag": float(np.imag(vector[index] / (amplitude * global_phase)))} for index in range(DIMENSION)]}


def _exact_custom_oracle_semantics(model, feasibility, artifact: Mapping[str, Any]) -> dict[str, object]:
    """Primary correctness evidence: exact statevector truth tables for custom circuits."""
    threshold = int(artifact["encoded_threshold"])
    value_rows = []
    for index in range(DIMENSION):
        probe = basis_value_code_probe(model, _bits(index))
        value_rows.append({"index": index, "bitstring": bitstring_from_index(index, model.num_x_qubits),
                           "classical_integer_value": int(artifact["integer_values"][index]), "expected_value_code": int(probe.expected_code),
                           "measured_value_code": int(probe.most_likely_code), "correct_code_probability": float(probe.correct_code_probability),
                           "passed": int(probe.expected_code) == int(probe.most_likely_code) and abs(float(probe.correct_code_probability)-1.0) <= 1e-12})
    if not all(row["passed"] for row in value_rows): raise FixedOracleDiagnosticError(f"值寄存器整数编码不一致: {artifact['scenario_id']}")
    compute_probe = phase_to_value_superposition_probe(model)
    if compute_probe.pairing_probability < 1-1e-10 or compute_probe.inverse_auxiliary_zero_probability < 1-1e-10:
        raise FixedOracleDiagnosticError(f"phase-to-value compute/uncompute 失败: {artifact['scenario_id']}")
    comparator_checks=[]
    for candidate in (threshold-1, threshold, threshold+1):
        probe=_phase_oracle_probe(model,feasibility,candidate)
        if float(probe["relative_phase_max_error"]) > 1e-8 or float(probe["auxiliary_zero_probability"]) < 1-1e-10:
            raise FixedOracleDiagnosticError(f"joint phase oracle 真值表失败: {artifact['scenario_id']} threshold={candidate}")
        comparator_checks.append({"encoded_threshold":candidate, "strict_less_than":True, **probe})
    return {"value_register_truth_table":value_rows, "compute_uncompute_pairing_probability":float(compute_probe.pairing_probability),
            "compute_uncompute_auxiliary_zero_probability":float(compute_probe.inverse_auxiliary_zero_probability),
            "threshold_comparator_checks":comparator_checks}


def _run_fixed_circuit(model, feasibility, artifact: Mapping[str, Any], k: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    threshold, marked = int(artifact["encoded_threshold"]), tuple(int(value) for value in artifact["joint_marked_indices"])
    circuit = build_sparse_vqc_grover_circuit(model, encoded_threshold=threshold, iterations=int(k), feasibility_spec=feasibility)
    logical = circuit_resource_summary(circuit, decompose_reps=1)
    backend = AerSimulator(method="matrix_product_state")
    transpiled = transpile(circuit, backend, optimization_level=1, seed_transpiler=int(seed))
    started = perf_counter(); mps = execute_sparse_vqc_grover_mps(circuit, num_x_qubits=model.num_x_qubits, shots=SHOTS, seed=int(seed)); wrapper_elapsed = perf_counter() - started
    exact_x, exact_auxiliary_zero = _aer_exact_x_probabilities(circuit, num_x_qubits=model.num_x_qubits)
    marked_shots = sum(int(mps.x_counts.get(bitstring_from_index(index, model.num_x_qubits), 0)) for index in marked)
    observed = marked_shots / SHOTS; exact_probability = float(sum(exact_x[index] for index in marked)); ideal = ideal_grover_probability(len(marked), int(k))
    low, high = wilson_interval(marked_shots, SHOTS)
    if abs(exact_probability - ideal) > 1e-8: raise FixedOracleDiagnosticError(f"exact circuit 与理想 Grover 概率不一致: {artifact['scenario_id']} k={k}")
    if exact_auxiliary_zero < 1 - 1e-10 or float(mps.auxiliary_zero_probability) < 1 - 1e-10:
        raise FixedOracleDiagnosticError(f"辅助寄存器未回零: {artifact['scenario_id']} k={k}")
    state_rows = []
    for index in range(DIMENSION):
        bitstring = bitstring_from_index(index, model.num_x_qubits); count = int(mps.x_counts.get(bitstring, 0))
        state_rows.append({"scenario_id": artifact["scenario_id"], "k": int(k), "index": index, "bitstring": bitstring, "shot_count": count,
                           "observed_probability": count / SHOTS, "exact_probability": float(exact_x[index]), "joint_marked": index in marked,
                           "quantized_integer_value": int(artifact["integer_values"][index]), "hard_logic_feasible": bool(artifact["hard_logic_feasible"][index])})
    row = {"scenario_id": artifact["scenario_id"], "generator_pair": artifact["generator_pair"], "window_start": artifact["window_start"], "training_seed": artifact["training_seed"],
           "encoded_threshold": threshold, "num_search_qubits": model.num_x_qubits, "N": DIMENSION, "M": len(marked), "joint_marked_indices": list(marked),
           "joint_marked_bitstrings": [bitstring_from_index(index, model.num_x_qubits) for index in marked], "k": int(k), "shots": SHOTS, "simulator_seed": int(seed),
           "num_qubits": circuit.num_qubits, "logical_or_pretranspile_depth": logical["depth"], "transpiled_depth": transpiled.depth(), "gate_counts": _gate_counts(circuit),
           "oracle_calls": int(k), "diffuser_calls": int(k), "mps_elapsed_seconds": float(mps.elapsed_seconds), "wrapper_elapsed_seconds": float(wrapper_elapsed),
           "marked_shot_count": marked_shots, "unmarked_shot_count": SHOTS-marked_shots, "observed_marked_probability": observed,
           "uniform_reference_probability": len(marked)/DIMENSION, "ideal_grover_probability": ideal, "exact_circuit_marked_probability": exact_probability,
           "observed_minus_uniform": observed-len(marked)/DIMENSION, "observed_minus_ideal": observed-ideal, "exact_minus_ideal": exact_probability-ideal,
           "wilson_ci_low": low, "wilson_ci_high": high, "auxiliary_zero_probability": float(mps.auxiliary_zero_probability),
           "auxiliary_rejection_count": int(round(SHOTS*(1-float(mps.auxiliary_zero_probability))),), "predicate_consistency_passed": True}
    return row, state_rows


def _write_json(path: Path, payload: Any) -> None: path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields=list(rows[0]) if rows else []
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value,ensure_ascii=False,allow_nan=False) if isinstance(value,(list,dict)) else value for key,value in row.items()})


def _write_plots(output_dir: Path, runs: Sequence[Mapping[str, Any]], states: Sequence[Mapping[str, Any]]) -> None:
    if not runs: return
    scenario=str(runs[0]["scenario_id"]); ordered=sorted(runs,key=lambda row:int(row["k"]))
    def svg(points: Sequence[tuple[float,float,str,str]], title: str, path: Path) -> None:
        width,height,left,bottom=720,420,70,50; plot_w,plot_h=610,310
        lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',f'<text x="20" y="25">{title}</text>',f'<line x1="{left}" y1="{height-bottom}" x2="{left+plot_w}" y2="{height-bottom}" stroke="black"/>',f'<line x1="{left}" y1="{height-bottom}" x2="{left}" y2="{height-bottom-plot_h}" stroke="black"/>']
        for x,y,color,label in points:
            px=left+x*plot_w/3; py=height-bottom-y*plot_h; lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{color}"/><text x="{px+5:.2f}" y="{py-5:.2f}" font-size="10">{label}</text>')
        lines.append('</svg>');path.write_text(''.join(lines),encoding="utf-8")
    points=[]
    for row in ordered:
        k=float(row["k"]);points.extend([(k,float(row["observed_marked_probability"]),"#1f77b4","observed"),(k,float(row["exact_circuit_marked_probability"]),"#d62728","exact"),(k,float(row["ideal_grover_probability"]),"#2ca02c","ideal"),(k,float(row["uniform_reference_probability"]),"#666666","uniform")])
    svg(points,f"{scenario}: joint-marked probability vs k",output_dir/"plots"/f"marked_probability_vs_k_{scenario}.svg")
    for row in ordered:
        subset=sorted((item for item in states if item["scenario_id"]==scenario and item["k"]==row["k"]),key=lambda item:int(item["index"]))
        points=[(float(item["index"])*3/(DIMENSION-1),float(item["observed_probability"]),"#ff7f0e" if item["joint_marked"] else "#1f77b4",str(item["index"])) for item in subset]
        svg(points,f"{scenario}, k={row['k']}, shots={row['shots']}, threshold={row['encoded_threshold']}",output_dir/"plots"/f"state_distribution_{scenario}_k{row['k']}.svg")


def run_fixed_oracle_multishot(formal_dir: Path, *, instance_path: Path, output_dir: Path, simulator_seed: int = 20260727) -> dict[str, Any]:
    """Run only new fixed-oracle diagnostics; never alter formal artifacts."""
    formal_dir, output_dir = Path(formal_dir), Path(output_dir)
    if output_dir.exists(): raise FixedOracleDiagnosticError(f"输出目录已存在，拒绝覆盖: {output_dir}")
    completed, validations = formal_dir/"runs"/"completed", formal_dir/"validation"/"scenarios"
    landscapes={path.stem:_load(path) for path in sorted(validations.glob("*.json"))}
    required=["case14-g0g5-w2-s0","case14-g0g5-w1-s0","case14-g0g5-w1-s1","case14-g0g5-w1-s2"]
    by_m: dict[int,str]={}
    for sid, land in sorted(landscapes.items()):
        by_m.setdefault(int(land.get("initial_joint_marked_count", -1)),sid)
    selected=list(required)
    unavailable=[m for m in (1,2,4,9) if m not in by_m]
    from .uc_loader import load_uc_instance
    source=load_uc_instance(instance_path); runs=[]; state_rows=[]; model_artifacts=[]
    output_dir.mkdir(parents=True); (output_dir/"scenario_models").mkdir(); (output_dir/"summaries").mkdir(); (output_dir/"plots").mkdir()
    primary_grover_scenario=required[0]
    for ordinal,sid in enumerate(selected):
        run=_find_run(completed,sid); model,feasibility,artifact=_reconstruct_model(run,source=source,instance_path=instance_path); _verify_reconstruction(artifact,landscapes[sid])
        validation=landscapes[sid]; optima=set(validation.get("true_global_optimum_indices",[])); artifact["natural_global_optimum_in_training"]=bool(optima & set(artifact["training_indices"]))
        semantics=_exact_custom_oracle_semantics(model,feasibility,artifact)
        artifact["exact_custom_circuit_semantics"]=semantics
        _write_json(output_dir/"scenario_models"/f"{sid}.json",artifact); model_artifacts.append(artifact)
        if sid == primary_grover_scenario:
            for k in range(4):
                row, states=_run_fixed_circuit(model,feasibility,artifact,k,int(simulator_seed)+ordinal*10+k)
                row["phase_max_error"]=float(semantics["threshold_comparator_checks"][1]["relative_phase_max_error"]); runs.append(row);state_rows.extend(states)
    summary={"scope":"primary evidence is exact statevector custom-circuit semantics on all 16 states; one M=3 fixed-oracle 1024-shot Aer MPS Grover integration diagnostic only; no ED/LP, closed-loop batch, threshold update, or BBHT window update", "selected_scenarios":selected, "primary_grover_scenario":primary_grover_scenario,
             "unavailable_in_existing_initial_threshold_cases":unavailable,"runs":len(runs),"shots_per_circuit":SHOTS,"training_truth_protocol":"random_truth_blind","global_truth_used_online":False,"global_truth_used_for_posthoc_validation":True}
    _write_json(output_dir/"manifest.json",summary); _write_json(output_dir/"fixed_oracle_runs.json",runs); _write_csv(output_dir/"fixed_oracle_runs.csv",runs); _write_csv(output_dir/"state_probabilities.csv",state_rows)
    _write_csv(output_dir/"summaries"/"by_scenario.csv",runs); _write_csv(output_dir/"summaries"/"by_m_k.csv",runs); _write_json(output_dir/"summaries"/"summary.json",summary)
    _write_plots(output_dir,runs,state_rows)
    semantic_counts = {
        "scenarios": len(model_artifacts),
        "all_16_value_register_checks": sum(
            len(item["exact_custom_circuit_semantics"]["value_register_truth_table"])
            for item in model_artifacts
        ),
        "threshold_triplet_phase_checks": sum(
            len(item["exact_custom_circuit_semantics"]["threshold_comparator_checks"])
            for item in model_artifacts
        ),
    }
    grover_lines = "\n".join(
        f"- k={row['k']}: exact={row['exact_circuit_marked_probability']:.12g}, "
        f"ideal={row['ideal_grover_probability']:.12g}, "
        f"MPS-1024={row['observed_marked_probability']:.12g}"
        for row in runs
    )
    (output_dir/"report.md").write_text(
        "# Fixed-oracle diagnostic report\n\n"
        "## Primary exact circuit-semantic evidence\n\n"
        f"- Scenarios: {semantic_counts['scenarios']} (all have natural M=3).\n"
        f"- Value-register checks: {semantic_counts['all_16_value_register_checks']} / "
        f"{semantic_counts['all_16_value_register_checks']} passed.\n"
        f"- Threshold-triplet joint-phase checks: {semantic_counts['threshold_triplet_phase_checks']} / "
        f"{semantic_counts['threshold_triplet_phase_checks']} passed.\n"
        "- Each check compares the Aer exact-statevector circuit output with the "
        "theoretical diagonal phase oracle up to global phase only.\n"
        "- Compute--phase-mark--uncompute returns all non-search registers to zero.\n\n"
        "## Representative fixed-oracle Grover integration (M=3)\n\n"
        f"Scenario: `{primary_grover_scenario}`; shots: 1024.\n\n{grover_lines}\n\n"
        "The exact curve, rather than shot frequency, is the correctness evidence. "
        "The MPS counts are an independent gate-level integration check only; this "
        "report makes no end-to-end quantum-speedup claim.\n",
        encoding="utf-8",
    )
    return summary
