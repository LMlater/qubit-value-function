"""Small, isolated comparison of first-training and best-training incumbents."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_2x2_joint_feasible_sparse_vqc_bbht import (  # noqa: E402
    build_case14_closed_loop_scenario,
)
from qubit_value_function.closed_loop_batch import atomic_write_json  # noqa: E402
from qubit_value_function.closed_loop_metadata import bitstring_from_index  # noqa: E402
from qubit_value_function.closed_loop_scenario import (  # noqa: E402
    ClosedLoopScenario,
    run_closed_loop_method,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig  # noqa: E402
from qubit_value_function.sparse_vqc_bbht import (  # noqa: E402
    BBHTConfig,
    InitialIncumbentSelection,
    select_training_initial_incumbent,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


SMOKE_SCENARIOS = (
    ("first_training_not_best", (0, 5), 2, 0),
    ("first_training_best", (1, 3), 1, 0),
    ("marked_empty", (1, 3), 0, 0),
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Small Stage A first-training versus best-training smoke"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/stage1_best_training_smoke"),
    )
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--train-sample-count", type=int, default=8)
    parser.add_argument("--fractional-bits", type=int, default=2)
    parser.add_argument("--cost-unit", type=float, default=1000.0)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--max-oracle-calls", type=int, default=4)
    parser.add_argument("--max-new-ed-lp-calls", type=int, default=2)
    parser.add_argument("--max-threshold-updates", type=int, default=1)
    return parser


def _selection_metadata(
    scenario: ClosedLoopScenario,
    selection: InitialIncumbentSelection,
) -> dict[str, object]:
    metadata = dict(scenario.reproducibility_metadata)
    metadata.update(
        {
            "initial_incumbent_policy": selection.policy,
            "initial_incumbent_index": int(selection.index),
            "initial_incumbent_bitstring": bitstring_from_index(
                selection.index, scenario.value_model.num_x_qubits
            ),
            "initial_incumbent_true_cost": float(selection.true_cost),
            "best_training_candidate_indices": [
                int(index) for index in selection.best_training_candidate_indices
            ],
            "best_training_tie_count": int(selection.best_training_tie_count),
            "best_training_tie_break_rule": selection.best_training_tie_break_rule,
            "training_selection_protocol": "random_truth_blind",
            "global_truth_used_online": False,
            "global_truth_used_for_posthoc_validation": True,
        }
    )
    return metadata


def _with_best_training_incumbent(scenario: ClosedLoopScenario) -> ClosedLoopScenario:
    selection = select_training_initial_incumbent(
        scenario.training_indices,
        scenario.initial_exact_cache,
        policy="best_training",
    )
    return replace(
        scenario,
        initial_incumbent_index=int(selection.index),
        reproducibility_metadata=_selection_metadata(scenario, selection),
    )


def _initial_marked_summary(scenario: ClosedLoopScenario) -> dict[str, object]:
    record = scenario.initial_exact_cache[scenario.initial_incumbent_index]
    true_cost = float(record.total_cost)
    encoded_threshold = int(scenario.value_model.fixed_point_config.encode(true_cost))
    cost_marked: list[int] = []
    joint_marked: list[int] = []
    for index in range(2 ** int(scenario.value_model.num_x_qubits)):
        bits = tuple(
            (int(index) >> offset) & 1
            for offset in range(int(scenario.value_model.num_x_qubits))
        )
        if int(scenario.value_model.integer_value(bits)) < encoded_threshold:
            cost_marked.append(int(index))
            if scenario.hard_logic_is_feasible(bits):
                joint_marked.append(int(index))
    return {
        "initial_incumbent_index": int(scenario.initial_incumbent_index),
        "initial_incumbent_bitstring": bitstring_from_index(
            scenario.initial_incumbent_index, scenario.value_model.num_x_qubits
        ),
        "initial_incumbent_true_cost": true_cost,
        "initial_encoded_threshold": encoded_threshold,
        "initial_cost_marked_indices": cost_marked,
        "initial_cost_marked_count": len(cost_marked),
        "initial_joint_marked_indices": joint_marked,
        "initial_joint_marked_count": len(joint_marked),
    }


def _run_summary(envelope: Mapping[str, object]) -> dict[str, object]:
    result = dict(envelope["result_schema"])
    trace = list(result.get("trial_trace", []))
    counters = dict(result.get("counters", {}))
    strict_improvements = [
        row for row in trace if bool(row.get("true_strict_improvement"))
    ]
    if any(
        bool(row.get("candidate_in_training_set")) for row in strict_improvements
    ):
        raise AssertionError("a training-cache candidate was recorded as strictly improving")
    if any(
        not bool(row.get("improvement_is_nontraining")) for row in strict_improvements
    ):
        raise AssertionError("a strict improvement was not marked as nontraining")
    dynamic_complete = all(
        {
            "encoded_threshold_before_trial",
            "cost_marked_indices_before",
            "cost_marked_count_before",
            "joint_marked_indices_before",
            "joint_marked_count_before",
            "measured_joint_marked",
            "sampled_grover_iterations",
            "cache_source",
            "new_ed_lp_solve",
            "true_strict_improvement",
            "threshold_updated",
        }.issubset(row)
        for row in trace
    )
    return {
        "total_trials": len(trace),
        "sampled_grover_iterations": [
            int(row["sampled_grover_iterations"]) for row in trace
        ],
        "oracle_calls": int(counters.get("total_oracle_calls", 0)),
        "mps_circuit_executions": int(counters.get("circuit_executions", 0)),
        "cache_hits": int(counters.get("cached_exact_lookups", 0)),
        "new_search_ed_lp": int(counters.get("actual_ed_lp_solves", 0)),
        "training_cache_improvement_count": 0,
        "nontraining_improvement_count": len(strict_improvements),
        "threshold_updates": int(counters.get("threshold_updates", 0)),
        "final_incumbent_index": int(result["final_incumbent_index"]),
        "final_true_cost": float(result["final_incumbent_true_cost"]),
        "stop_reason": str(result["stop_reason"]),
        "last_improvement_tail_trials": (
            len(trace) - max(
                (position for position, row in enumerate(trace, start=1)
                 if bool(row.get("true_strict_improvement"))),
                default=0,
            )
        ),
        "dynamic_metadata_available_for_all_trials": dynamic_complete,
        "trial_trace": trace,
    }


def _assert_shared_frozen_artifacts(
    first: ClosedLoopScenario,
    best: ClosedLoopScenario,
) -> None:
    if first.training_indices != best.training_indices:
        raise AssertionError("initial incumbent policy changed training indices")
    if first.training_labels != best.training_labels:
        raise AssertionError("initial incumbent policy changed training labels")
    if dict(first.initial_exact_cache) != dict(best.initial_exact_cache):
        raise AssertionError("initial incumbent policy changed the training cache")
    if first.value_model.as_dict() != best.value_model.as_dict():
        raise AssertionError("initial incumbent policy changed the quantized VQC model")
    if first.bbht_config != best.bbht_config:
        raise AssertionError("initial incumbent policy changed BBHT configuration")
    if (
        first.reproducibility_metadata["training_actual_ed_lp_solves"]
        != best.reproducibility_metadata["training_actual_ed_lp_solves"]
    ):
        raise AssertionError("initial incumbent policy changed training ED/LP calls")


def main() -> int:
    args = build_argument_parser().parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing smoke output: {output_dir}")
    output_dir.mkdir(parents=True)
    source = load_uc_instance(args.instance)
    fixed_point = FixedPointConfig(
        fractional_bits=args.fractional_bits,
        unit=args.cost_unit,
        rounding="nearest",
    )
    config = BBHTConfig(
        max_trials=args.max_trials,
        max_oracle_calls=args.max_oracle_calls,
        max_new_ed_lp_calls=args.max_new_ed_lp_calls,
        max_threshold_updates=args.max_threshold_updates,
        seed=0,
    )
    rows: list[dict[str, object]] = []
    for label, generator_pair, window_start, training_seed in SMOKE_SCENARIOS:
        first = build_case14_closed_loop_scenario(
            source=source,
            window_start=window_start,
            selected_generator_indices=generator_pair,
            train_sample_count=args.train_sample_count,
            fixed_point=fixed_point,
            initialization_policy="first_training",
            seed=training_seed,
            regularization=1e-4,
            maxiter=args.maxiter,
            bbht_config=config,
        )
        best = _with_best_training_incumbent(first)
        _assert_shared_frozen_artifacts(first, best)
        first_initial = _initial_marked_summary(first)
        best_initial = _initial_marked_summary(best)
        if label == "first_training_not_best" and (
            first.initial_incumbent_index == best.initial_incumbent_index
        ):
            raise AssertionError("selected nonbest smoke scenario has no initialization difference")
        if label == "first_training_best":
            if first.initial_incumbent_index != best.initial_incumbent_index:
                raise AssertionError("selected best smoke scenario changed its initial incumbent")
        if label == "marked_empty":
            if int(best_initial["initial_joint_marked_count"]) != 0:
                raise AssertionError("selected marked-empty smoke scenario is not marked-empty")
        policy_rows: dict[str, object] = {}
        for policy, scenario in (("first_training", first), ("best_training", best)):
            envelope = run_closed_loop_method(
                scenario,
                "joint_bbht",
                run_seed=args.run_seed,
                persist_dynamic_oracle_metadata=True,
            )
            policy_rows[policy] = {
                "scenario_metadata": scenario.metadata(),
                "initial": _initial_marked_summary(scenario),
                "search": _run_summary(envelope),
            }
        if label == "marked_empty":
            first_search = policy_rows["first_training"]["search"]
            best_search = policy_rows["best_training"]["search"]
            if first_search["stop_reason"] != best_search["stop_reason"]:
                raise AssertionError("marked-empty stop semantics changed between policies")
        rows.append(
            {
                "label": label,
                "scenario_id": first.scenario_id,
                "generator_pair": list(generator_pair),
                "window_start": int(window_start),
                "training_seed": int(training_seed),
                "run_seed": int(args.run_seed),
                "training_indices": [int(index) for index in first.training_indices],
                "training_labels": [
                    {"index": int(index), "true_cost": float(cost)}
                    for index, cost in first.training_labels
                ],
                "training_actual_ed_lp_solves": int(
                    first.reproducibility_metadata["training_actual_ed_lp_solves"]
                ),
                "policies": policy_rows,
            }
        )
    report = {
        "scope": "small independent smoke; no pilot, formal, validation landscape, or output overwrite",
        "training_selection_protocol": "random_truth_blind",
        "global_truth_used_online": False,
        "global_truth_used_for_posthoc_validation": False,
        "config": {
            **{key: value for key, value in vars(args).items() if key not in {"output_dir", "instance"}},
            "output_dir": str(output_dir),
            "instance": str(args.instance),
            "persist_dynamic_oracle_metadata": True,
        },
        "scenarios": rows,
    }
    atomic_write_json(output_dir / "best_training_smoke_report.json", report)
    print(json.dumps({"output_dir": str(output_dir), "scenarios": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
