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

from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.experiment_utils import (  # noqa: E402
    commitment_row,
    embedded_selected_commitments,
    evaluate_values,
    finite_or_none,
    leading_time_window_instance,
    parse_indices,
)
from qubit_value_function.gate_level_oracle import (  # noqa: E402
    GateLevelAffineOracleSpec,
    bitstring_from_index,
    build_affine_grover_circuit,
    build_affine_phase_oracle_circuit,
    circuit_resource_summary,
    simulate_affine_grover,
    simulate_affine_phase_oracle,
)
from qubit_value_function.uc_loader import UCInstance, load_uc_instance  # noqa: E402


def run(
    instance_path: Path,
    results_path: Path,
    horizon: int,
    selected_generator_indices: tuple[int, ...],
) -> dict[str, object]:
    if horizon != 2:
        raise ValueError("this gate-level prototype is intentionally restricted to T=2")

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

    spec = case14_t2_gate_level_proxy_spec(instance, selected_generator_indices)
    embedded_commitments = embedded_selected_commitments(
        full_optimum_commitment,
        selected_generator_indices,
    )
    embedded_values, embedded_logic_feasible = evaluate_values(instance, embedded_commitments)
    embedded_finite = np.isfinite(embedded_values)
    embedded_best_index = int(np.nanargmin(embedded_values))

    phase_start = time.perf_counter()
    phase_probe = simulate_affine_phase_oracle(spec)
    phase_seconds = time.perf_counter() - phase_start

    grover_start = time.perf_counter()
    grover_result = simulate_affine_grover(spec)
    grover_seconds = time.perf_counter() - grover_start

    phase_circuit = build_affine_phase_oracle_circuit(spec)
    grover_circuit = build_affine_grover_circuit(spec, grover_result.iterations)
    proxy_values = spec.values_for_all_x()
    marked_indices = [int(index) for index in np.flatnonzero(spec.marked_mask())]

    summary = {
        "instance": f"case14_T{horizon}",
        "source": str(instance_path),
        "method": (
            "T=2 selected-subregister Qiskit gate-level value-register Grover oracle"
        ),
        "scope": {
            "full_case14_bits": int(commitments.reshape((commitments.shape[0], -1)).shape[1]),
            "gate_level_x_bits": int(spec.num_x_qubits),
            "note": (
                "This is a gate-level prototype on a T=2 selected generator subregister; "
                "the full 12-bit, 207-feature, 32-piece T=2 surrogate is not yet gate-synthesized."
            ),
        },
        "fixed_load_mw": instance.fixed_load,
        "generators": generator_names,
        "selected_generators": [
            generator_names[index] for index in selected_generator_indices
        ],
        "bit_order": "selected generator-major order: g_i_t0,g_i_t1,...",
        "proxy_oracle": spec_to_dict(spec),
        "oracle_decomposition": {
            "value_register": "Qiskit WeightedAdder computes an integer affine proxy value",
            "threshold_comparator": "Qiskit IntegerComparator marks value <= tau",
            "phase_mark": "Z gate on the comparator flag qubit",
            "uncompute": "inverse comparator and inverse adder restore all non-x registers",
            "diffuser": "standard Grover diffuser acts only on the x register",
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
        "embedded_subspace_validation": {
            "base_commitment": commitment_to_bitstring(full_optimum_commitment),
            "base_note": "Non-selected generators are fixed to the exhaustive T=2 optimum for validation only.",
            "subspace_size": int(embedded_commitments.shape[0]),
            "logic_feasible_count": int(embedded_logic_feasible.sum()),
            "finite_value_count": int(embedded_finite.sum()),
            "best_selected_bitstring": bitstring_from_index(
                embedded_best_index,
                spec.num_x_qubits,
            ),
            "best_full_bitstring": commitment_to_bitstring(
                embedded_commitments[embedded_best_index]
            ),
            "best_true_cost": _finite_or_none(embedded_values[embedded_best_index]),
            "proxy_marked_selected_bitstrings": [
                bitstring_from_index(index, spec.num_x_qubits)
                for index in marked_indices
            ],
            "proxy_marked_full_bitstrings": [
                commitment_to_bitstring(embedded_commitments[index])
                for index in marked_indices
            ],
        },
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
            "argmax_selected_bitstring": bitstring_from_index(
                int(np.argmax(grover_result.x_probabilities)),
                spec.num_x_qubits,
            ),
            "simulation_seconds": float(grover_seconds),
            "top_probability_rows": top_probability_rows(
                grover_result.x_probabilities,
                proxy_values,
                spec.marked_mask(),
                embedded_commitments,
                embedded_values,
                spec.num_x_qubits,
            ),
        },
        "resources": {
            "phase_oracle": circuit_resource_summary(phase_circuit, decompose_reps=3),
            "grover_circuit": circuit_resource_summary(grover_circuit, decompose_reps=2),
        },
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def case14_t2_gate_level_proxy_spec(
    instance: UCInstance,
    selected_generator_indices: tuple[int, ...],
) -> GateLevelAffineOracleSpec:
    if not selected_generator_indices:
        raise ValueError("at least one selected generator is required")
    slopes = np.array(
        [
            _marginal_cost_slope(instance.generators[index])
            for index in selected_generator_indices
        ],
        dtype=float,
    )
    median_slope = float(np.median(slopes))
    weights: list[int] = []
    inverted: list[int] = []
    labels: list[str] = []
    for local_index, generator_index in enumerate(selected_generator_indices):
        generator = instance.generators[generator_index]
        desired_on = slopes[local_index] <= median_slope
        term_weight = 2 if desired_on else 1
        for time_index in range(instance.time_horizon):
            bit_index = len(weights)
            weights.append(term_weight)
            labels.append(f"{generator.name}_t{time_index}")
            if desired_on:
                inverted.append(bit_index)
    return GateLevelAffineOracleSpec(
        weights=tuple(weights),
        threshold=0,
        inverted_bit_indices=tuple(inverted),
        bit_labels=tuple(labels),
        name="case14_t2_affine_value_oracle",
    )


def top_probability_rows(
    probabilities: np.ndarray,
    proxy_values: np.ndarray,
    marked: np.ndarray,
    embedded_commitments: np.ndarray,
    embedded_values: np.ndarray,
    num_bits: int,
    limit: int = 10,
) -> list[dict[str, object]]:
    rows = []
    for state_index in np.argsort(probabilities)[::-1][:limit]:
        rows.append(
            {
                "selected_bitstring": bitstring_from_index(int(state_index), num_bits),
                "probability": float(probabilities[int(state_index)]),
                "proxy_value": int(proxy_values[int(state_index)]),
                "marked": bool(marked[int(state_index)]),
                "embedded_full_bitstring": commitment_to_bitstring(
                    embedded_commitments[int(state_index)]
                ),
                "embedded_true_cost": _finite_or_none(
                    embedded_values[int(state_index)]
                ),
            }
        )
    return rows


def spec_to_dict(spec: GateLevelAffineOracleSpec) -> dict[str, object]:
    terms = []
    inverted = set(spec.inverted_bit_indices)
    for bit_index, (label, weight) in enumerate(zip(spec.bit_labels, spec.weights)):
        terms.append(
            {
                "bit": label,
                "weight": int(weight),
                "term": f"{weight}*(1-{label})" if bit_index in inverted else f"{weight}*{label}",
            }
        )
    return {
        "value_formula": "sum integer weighted on/off penalties",
        "threshold_tau": int(spec.threshold),
        "weights": [int(weight) for weight in spec.weights],
        "inverted_bit_indices": [int(index) for index in spec.inverted_bit_indices],
        "bit_labels": list(spec.bit_labels),
        "terms": terms,
    }


def _marginal_cost_slope(generator) -> float:
    mw_span = generator.cost_mw[-1] - generator.cost_mw[0]
    if abs(mw_span) < 1e-12:
        return 0.0
    return float((generator.cost_usd[-1] - generator.cost_usd[0]) / mw_span)


def _finite_or_none(value: float) -> float | None:
    return finite_or_none(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_t2_gate_level_grover_oracle.json"),
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--selected-generators", type=parse_indices, default=(0, 1, 2))
    args = parser.parse_args()

    summary = run(
        args.instance,
        args.results,
        args.horizon,
        args.selected_generators,
    )
    compact = {
        "instance": summary["instance"],
        "method": summary["method"],
        "scope": summary["scope"],
        "selected_generators": summary["selected_generators"],
        "phase_oracle_check": summary["phase_oracle_check"],
        "grover_result": {
            key: value
            for key, value in summary["grover_result"].items()
            if key != "top_probability_rows"
        },
        "embedded_subspace_validation": summary["embedded_subspace_validation"],
        "resources": {
            "phase_qubits": summary["resources"]["phase_oracle"]["num_qubits"],
            "grover_qubits": summary["resources"]["grover_circuit"]["num_qubits"],
            "grover_depth": summary["resources"]["grover_circuit"]["depth"],
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
