from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_bundle_phase import exact_pattern_features, signed_bundle_column  # noqa: E402
from experiments.stage1_case14_hamming_phase import threshold_for_top_count  # noqa: E402
from experiments.stage1_case14_single_period import single_period_instance  # noqa: E402
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.feature_phase import evaluate_feature_phase_model, fit_feature_phase_model  # noqa: E402
from qubit_value_function.oracle import grover_search_probabilities, grover_with_oracle_matrix, phase_oracle_errors, verify_phase_oracle  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


DEFAULT_TARGET_COUNTS = [1, 2, 5, 10, 20, 32]


def run(
    instance_path: Path,
    results_path: Path,
    period: int,
    target_counts: list[int],
) -> dict[str, object]:
    instance = single_period_instance(load_uc_instance(instance_path), period)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bits = commitments.reshape((commitments.shape[0], -1))
    bitstrings = [commitment_to_bitstring(commitment) for commitment in commitments]
    values = np.array([FixedCommitmentEvaluator(instance).evaluate(c).total_cost for c in commitments])
    generator_names = [gen.name for gen in instance.generators]

    experiments = []
    for target_count in target_counts:
        tau = threshold_for_top_count(values, target_count)
        labels = values <= tau
        exact_grover = grover_search_probabilities(labels)
        key_stats, key_order = key_generator_stats(bits, labels, generator_names)

        prefix_rows = []
        for prefix_size in range(1, bits.shape[1] + 1):
            selected_keys = key_order[:prefix_size]
            features, names, templates = partial_prefix_pattern_features(
                bits,
                labels,
                selected_keys,
                generator_names,
            )
            prefix_rows.append(
                evaluate_feature_set(
                    features,
                    names,
                    labels,
                    exact_grover,
                    {
                        "prefix_size": prefix_size,
                        "selected_generators": [generator_names[idx] for idx in selected_keys],
                        "template_count": len(templates),
                        "templates": templates,
                    },
                )
            )

        frequency_rows = []
        for threshold in (1.0, 0.8, 0.6):
            features, names, templates = frequency_template_features(
                bits,
                labels,
                key_order,
                generator_names,
                threshold,
            )
            if features.shape[1] == 1:
                continue
            frequency_rows.append(
                evaluate_feature_set(
                    features,
                    names,
                    labels,
                    exact_grover,
                    {
                        "frequency_threshold": threshold,
                        "template_count": len(templates),
                        "templates": templates,
                    },
                )
            )

        full_features, full_names, full_templates = exact_pattern_features(
            bits,
            np.where(labels)[0],
            generator_names,
        )
        full_row = evaluate_feature_set(
            full_features,
            full_names,
            labels,
            exact_grover,
            {
                "feature_count": len(full_names),
                "templates": full_templates,
            },
        )

        experiments.append(
            {
                "target_count": int(labels.sum()),
                "requested_target_count": target_count,
                "threshold": tau,
                "exact_grover": {
                    key: value for key, value in exact_grover.items() if key != "probabilities"
                },
                "key_generator_stats": key_stats,
                "best_prefix_result": best_result(prefix_rows, exact_grover),
                "prefix_rows": prefix_rows,
                "frequency_rows": frequency_rows,
                "full_signed_pattern_result": full_row,
            }
        )

    rows = sorted(
        [
            {"bitstring": bitstring, "total_cost": float(value)}
            for bitstring, value in zip(bitstrings, values)
        ],
        key=lambda item: item["total_cost"],
    )
    summary = {
        "instance": str(instance_path),
        "method": "case14 partial signed-pattern phase templates",
        "period": period,
        "fixed_load_mw": instance.fixed_load,
        "generators": generator_names,
        "num_bits": int(bits.shape[1]),
        "num_commitments": int(bits.shape[0]),
        "target_counts": target_counts,
        "optimum": rows[0],
        "runner_up": rows[1],
        "experiments": experiments,
        "rows": rows,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def key_generator_stats(
    bits: np.ndarray,
    labels: np.ndarray,
    generator_names: list[str],
) -> tuple[list[dict[str, object]], list[int]]:
    target_bits = bits[labels]
    non_target_bits = bits[~labels]
    target_freq = target_bits.mean(axis=0) if target_bits.size else np.zeros(bits.shape[1])
    non_target_freq = non_target_bits.mean(axis=0) if non_target_bits.size else np.zeros(bits.shape[1])
    delta = target_freq - non_target_freq
    order = list(np.argsort(-np.abs(delta)))
    stats = [
        {
            "index": int(idx),
            "generator": generator_names[idx],
            "target_on_frequency": float(target_freq[idx]),
            "non_target_on_frequency": float(non_target_freq[idx]),
            "delta": float(delta[idx]),
            "preferred_state": int(target_freq[idx] >= 0.5),
        }
        for idx in order
    ]
    return stats, order


def partial_prefix_pattern_features(
    bits: np.ndarray,
    labels: np.ndarray,
    selected_keys: list[int],
    generator_names: list[str],
) -> tuple[np.ndarray, list[str], list[dict[str, object]]]:
    target_patterns = []
    seen = set()
    for row in bits[labels][:, selected_keys]:
        key = tuple(int(v) for v in row)
        if key not in seen:
            seen.add(key)
            target_patterns.append(key)

    columns = [np.ones(bits.shape[0])]
    names = ["1"]
    templates = []
    for pattern in target_patterns:
        on = tuple(idx for idx, value in zip(selected_keys, pattern) if value == 1)
        off = tuple(idx for idx, value in zip(selected_keys, pattern) if value == 0)
        columns.append(signed_bundle_column(bits, on, off))
        names.append(template_name(on, off, generator_names))
        templates.append(
            {
                "on": [generator_names[idx] for idx in on],
                "off": [generator_names[idx] for idx in off],
                "control_count": len(on) + len(off),
            }
        )
    return np.column_stack(columns), names, templates


def frequency_template_features(
    bits: np.ndarray,
    labels: np.ndarray,
    key_order: list[int],
    generator_names: list[str],
    threshold: float,
) -> tuple[np.ndarray, list[str], list[dict[str, object]]]:
    target_freq = bits[labels].mean(axis=0)
    high_on = tuple(idx for idx in key_order if target_freq[idx] >= threshold)
    low_off = tuple(idx for idx in key_order if target_freq[idx] <= 1.0 - threshold)
    signed_literals = [(idx, 1) for idx in high_on] + [(idx, 0) for idx in low_off]

    columns = [np.ones(bits.shape[0])]
    names = ["1"]
    templates = []
    if signed_literals:
        on = tuple(idx for idx, value in signed_literals if value == 1)
        off = tuple(idx for idx, value in signed_literals if value == 0)
        add_template(columns, names, templates, bits, on, off, generator_names)
    for idx, value in signed_literals:
        on = (idx,) if value == 1 else ()
        off = (idx,) if value == 0 else ()
        add_template(columns, names, templates, bits, on, off, generator_names)
    for left_pos in range(len(signed_literals)):
        for right_pos in range(left_pos + 1, len(signed_literals)):
            pair = [signed_literals[left_pos], signed_literals[right_pos]]
            on = tuple(idx for idx, value in pair if value == 1)
            off = tuple(idx for idx, value in pair if value == 0)
            add_template(columns, names, templates, bits, on, off, generator_names)
    return np.column_stack(columns), names, templates


def add_template(
    columns: list[np.ndarray],
    names: list[str],
    templates: list[dict[str, object]],
    bits: np.ndarray,
    on: tuple[int, ...],
    off: tuple[int, ...],
    generator_names: list[str],
) -> None:
    name = template_name(on, off, generator_names)
    if name in names:
        return
    columns.append(signed_bundle_column(bits, on, off))
    names.append(name)
    templates.append(
        {
            "on": [generator_names[idx] for idx in on],
            "off": [generator_names[idx] for idx in off],
            "control_count": len(on) + len(off),
        }
    )


def evaluate_feature_set(
    features: np.ndarray,
    names: list[str],
    labels: np.ndarray,
    exact_grover: dict[str, object],
    extra: dict[str, object],
) -> dict[str, object]:
    model = fit_feature_phase_model(features, labels, names)
    evaluation = evaluate_feature_phase_model(model, features, labels)
    oracle = model.oracle_matrix(features)
    grover = grover_with_oracle_matrix(oracle, labels)
    row = {
        **extra,
        "feature_count": len(names),
        "evaluation": evaluation,
        "oracle_checks": verify_phase_oracle(oracle, atol=1e-8),
        "oracle_errors": phase_oracle_errors(oracle),
        "target_probability": grover["target_probability"],
        "exact_target_probability": exact_grover["marked_probability"],
        "iterations": grover["iterations"],
    }
    return row


def best_result(rows: list[dict[str, object]], exact_grover: dict[str, object]) -> dict[str, object]:
    exact_probability = float(exact_grover["marked_probability"])
    successful = [
        row
        for row in rows
        if row["evaluation"]["correct_marked_set"]
        and row["oracle_errors"]["self_inverse_error"] < 1e-8
        and row["target_probability"] >= exact_probability - 1e-8
    ]
    if successful:
        row = min(successful, key=lambda item: item.get("prefix_size", item["feature_count"]))
    else:
        row = max(rows, key=lambda item: item["target_probability"])
    return compact_result(row)


def compact_result(row: dict[str, object]) -> dict[str, object]:
    return {
        "prefix_size": row.get("prefix_size"),
        "feature_count": row["feature_count"],
        "correct_marked_set": row["evaluation"]["correct_marked_set"],
        "max_phase_factor_error": row["evaluation"]["max_phase_factor_error"],
        "self_inverse_error": row["oracle_errors"]["self_inverse_error"],
        "target_probability": row["target_probability"],
        "exact_target_probability": row["exact_target_probability"],
        "selected_generators": row.get("selected_generators"),
        "template_count": row.get("template_count"),
    }


def template_name(on: tuple[int, ...], off: tuple[int, ...], generator_names: list[str]) -> str:
    parts = []
    if on:
        parts.append("ON[" + ",".join(generator_names[idx] for idx in on) + "]")
    if off:
        parts.append("OFF[" + ",".join(generator_names[idx] for idx in off) + "]")
    return ";".join(parts) if parts else "1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--results", type=Path, default=Path("results/stage1_case14_partial_signed_phase.json"))
    parser.add_argument("--period", type=int, default=0)
    parser.add_argument("--target-counts", type=int, nargs="*", default=DEFAULT_TARGET_COUNTS)
    args = parser.parse_args()
    summary = run(args.instance, args.results, args.period, args.target_counts)
    compact = {
        key: value
        for key, value in summary.items()
        if key not in {"rows", "experiments"}
    }
    compact["experiments"] = [
        {
            "target_count": exp["target_count"],
            "best_prefix_result": exp["best_prefix_result"],
            "frequency_results": [compact_result(row) for row in exp["frequency_rows"]],
            "full_signed_pattern": compact_result(exp["full_signed_pattern_result"]),
            "key_generators": exp["key_generator_stats"][:4],
        }
        for exp in summary["experiments"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
