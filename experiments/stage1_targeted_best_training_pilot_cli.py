"""Run the fixed, snapshot-restored best-training candidate case-study pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.candidate_acceptance_loop import (  # noqa: E402
    ClosedLoopBudgets,
    ExactCandidateEvaluation,
)
from qubit_value_function.candidate_discovery_training_splits import training_indices_hash  # noqa: E402
from qubit_value_function.closed_loop_batch import (  # noqa: E402
    ClosedLoopBatchExecutor,
    RunSpec,
    atomic_write_json,
)
from qubit_value_function.closed_loop_metadata import canonical_sha256  # noqa: E402
from qubit_value_function.closed_loop_scenario import (  # noqa: E402
    ClosedLoopScenario,
    run_closed_loop_method,
)
from qubit_value_function.coherent_phase_value import QuantizedSparseValueModel  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator  # noqa: E402
from qubit_value_function.experiment_utils import (  # noqa: E402
    embedded_selected_commitments,
    time_window_instance,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig  # noqa: E402
from qubit_value_function.logic_feasibility_oracle import (  # noqa: E402
    BooleanLiteral,
    ForbiddenBooleanPattern,
    LogicFeasibilitySpec,
)
from qubit_value_function.sparse_phase_vqc import PhaseFeature  # noqa: E402
from qubit_value_function.sparse_vqc_bbht import BBHTConfig  # noqa: E402
from qubit_value_function.targeted_pilot_diagnostics import initial_marked_counts  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


PRIMARY_METHOD = "joint_bbht"
FIXED_RUN_SEEDS = tuple(range(20))


class PilotPreflightError(RuntimeError):
    """The persisted artifacts cannot safely support this targeted pilot."""


def ensure_new_output_dir(path: Path) -> None:
    if path.exists():
        raise PilotPreflightError(f"refusing_existing_output_directory:{path}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot-restored targeted best-training candidate case-study pilot; "
            "not formal and not quantum-advantage evidence."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.set_defaults(run_seeds=None, seed_range=None)
    return parser


def _snapshot_threshold(snapshot: Mapping[str, object]) -> int:
    labels = list(snapshot.get("training_labels", []))
    if not labels:
        raise PilotPreflightError("snapshot_missing_training_labels")
    costs = [float(dict(item)["true_cost"]) for item in labels]
    threshold = int(np.rint(min(costs) * int(snapshot["encoded_cost_scale"]) / float(snapshot["cost_unit"])))
    if "best_training_encoded_threshold" in snapshot and int(snapshot["best_training_encoded_threshold"]) != threshold:
        raise PilotPreflightError("snapshot_best_training_encoded_threshold_mismatch")
    return threshold


def _load_snapshot(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise PilotPreflightError(f"snapshot_unavailable:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "state_proxy_table" not in payload:
        raise PilotPreflightError(f"snapshot_unsupported_schema:{path}")
    return payload


def preflight_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    """Read only saved training facts, surrogate values, and hard-logic table."""

    scenarios = list(manifest.get("scenarios", []))
    if not scenarios:
        raise PilotPreflightError("manifest_has_no_scenarios")
    rows: list[dict[str, object]] = []
    for entry in scenarios:
        item = dict(entry)
        role = str(item["role"])
        if role not in {
            "grover_calibration_training_marked",
            "nontraining_improvement_candidate",
            "joint_marked_empty_control",
        }:
            raise PilotPreflightError(f"unsupported_scenario_role:{role}")
        snapshot = _load_snapshot(Path(str(item["snapshot"])))
        if bool(manifest.get("fixed_candidate_manifest")) and str(snapshot.get("scenario_id")) != str(item["scenario_id"]):
            raise PilotPreflightError("snapshot_scenario_id_mismatch")
        threshold = _snapshot_threshold(snapshot)
        counts = initial_marked_counts(snapshot, encoded_threshold=threshold)
        labels = {int(row["state_index"]): float(row["true_cost"]) for row in snapshot["training_labels"]}
        incumbent_index = int(snapshot.get("initial_incumbent_index", min(labels, key=labels.get)))
        incumbent_cost = float(snapshot.get("initial_incumbent_true_cost", labels[incumbent_index]))
        training_indices = [int(index) for index in snapshot["training_indices"]]
        for key, actual in (
            ("training_data_seed", int(snapshot.get("training_data_seed", snapshot.get("training_seed", 0)))),
            ("model_seed", int(snapshot.get("model_seed", snapshot.get("training_seed", 0)))),
            ("training_indices_hash", str(snapshot.get("training_indices_hash", training_indices_hash(training_indices)))),
        ):
            if key in item and item[key] != actual:
                raise PilotPreflightError(f"snapshot_{key}_mismatch")
        expected = dict(item.get("expected_initial", {}))
        actual_expected = {
            "tau_int": threshold,
            "joint_marked_count": int(counts["initial_joint_marked_count"]),
            "training_joint_marked_count": int(counts["initial_training_joint_marked_count"]),
            "nontraining_joint_marked_indices": [
                int(index) for index in counts["initial_joint_marked_indices"]
                if int(index) not in set(training_indices)
            ],
        }
        if any(key in expected and expected[key] != actual_expected[key] for key in actual_expected):
            raise PilotPreflightError("snapshot_expected_initial_marked_set_mismatch")
        if "proxy_integer_value" in expected or "margin" in expected:
            candidate_index = int(actual_expected["nontraining_joint_marked_indices"][0])
            candidate_row = next(
                dict(row) for row in snapshot["state_proxy_table"]
                if int(row["state_index"]) == candidate_index
            )
            if (
                ("proxy_integer_value" in expected and int(expected["proxy_integer_value"]) != int(candidate_row["integer_vqc_value"]))
                or ("margin" in expected and int(expected["margin"]) != threshold - int(candidate_row["integer_vqc_value"]))
            ):
                raise PilotPreflightError("snapshot_expected_candidate_margin_mismatch")
        if list(snapshot.get("training_indices", [])) != list(item.get("training_indices", snapshot["training_indices"])):
            raise PilotPreflightError("snapshot_training_indices_mismatch")
        rows.append(
            {
                "scenario_id": str(item["scenario_id"]),
                "role": role,
                "snapshot": str(item["snapshot"]),
                "snapshot_sha256": canonical_sha256(snapshot),
                "training_data_seed": int(snapshot.get("training_data_seed", snapshot.get("training_seed", 0))),
                "model_seed": int(snapshot.get("model_seed", snapshot.get("training_seed", 0))),
                "training_indices": training_indices,
                "training_indices_hash": str(snapshot.get("training_indices_hash", training_indices_hash(training_indices))),
                "initial_incumbent_index": incumbent_index,
                "initial_incumbent_true_cost": incumbent_cost,
                "best_training_encoded_threshold": threshold,
                **counts,
                "has_initial_nontraining_joint_marked": bool(
                    counts["initial_nontraining_joint_marked_count"]
                ),
            }
        )
    if not any(
        row["role"] == "nontraining_improvement_candidate"
        and bool(row["has_initial_nontraining_joint_marked"])
        for row in rows
    ):
        raise PilotPreflightError("no_nontraining_joint_marked_candidate")
    return {
        "selection_status": "eligible",
        "targeted_case_study": True,
        "formal": False,
        "quantum_advantage_evidence": False,
        "global_optimum_used_in_preflight": False,
        "scenarios": rows,
    }


def _fixed_bbht_config(manifest: Mapping[str, object]) -> dict[str, object]:
    config = dict(manifest.get("bbht_config", {}))
    expected = {
        "lambda_factor": 1.2,
        "max_trials": 64,
        "max_oracle_calls": 128,
        "max_new_ed_lp_calls": 16,
        "max_threshold_updates": 8,
        "max_consecutive_nonimproving_marked": 16,
        "max_same_encoded_threshold_updates": 3,
        "max_auxiliary_syndrome_rejections": 8,
    }
    if config != expected:
        raise PilotPreflightError("targeted_pilot_bbht_config_mismatch")
    return config


def build_targeted_run_specs(
    manifest: Mapping[str, object], *, expected_head: str
) -> tuple[RunSpec, ...]:
    """Create exactly four frozen scenario groups times seeds 0..19."""

    if tuple(manifest.get("methods", ())) != (PRIMARY_METHOD,):
        raise PilotPreflightError("targeted_pilot_requires_joint_bbht_only")
    if tuple(int(seed) for seed in manifest.get("run_seeds", ())) != FIXED_RUN_SEEDS:
        raise PilotPreflightError("targeted_pilot_requires_run_seeds_0_through_19")
    if str(manifest.get("initial_incumbent_policy")) != "best_training":
        raise PilotPreflightError("targeted_pilot_requires_best_training_initialization")
    budget = _fixed_bbht_config(manifest)
    scenarios = list(manifest.get("scenarios", []))
    if len(scenarios) != 4:
        raise PilotPreflightError("targeted_pilot_requires_exactly_four_scenarios")
    specs: list[RunSpec] = []
    for item in scenarios:
        if str(item.get("role")) != "nontraining_improvement_candidate":
            raise PilotPreflightError("targeted_pilot_requires_candidate_role")
        pair = tuple(int(value) for value in item["generator_pair"])
        for run_seed in FIXED_RUN_SEEDS:
            specs.append(
                RunSpec(
                    batch_id=str(manifest["experiment_name"]),
                    generator_pair=(pair[0], pair[1]),
                    window_start=int(item["window_start"]),
                    training_seed=int(item["training_data_seed"]),
                    method=PRIMARY_METHOD,
                    run_seed=int(run_seed),
                    preset="targeted_candidate_case_study",
                    budget_config=budget,
                    fixed_point_config={"fractional_bits": 2, "cost_unit": 1000.0, "rounding": "nearest"},
                    initialization_policy="best_training",
                    expected_code_sha=str(expected_head),
                )
            )
    if len(specs) != 80 or len({spec.run_id for spec in specs}) != 80:
        raise PilotPreflightError("targeted_pilot_run_plan_is_not_80_unique_runs")
    return tuple(specs)


def augment_targeted_result_schema(
    result: Mapping[str, object],
    *,
    training_indices: Sequence[int],
    initial_nontraining_joint_marked_indices: Sequence[int],
) -> dict[str, object]:
    """Add case-study classifications without affecting any online decision."""

    training = {int(index) for index in training_indices}
    candidates = {int(index) for index in initial_nontraining_joint_marked_indices}
    first_outside_evaluation: int | None = None
    first_outside_improvement: int | None = None
    trace: list[dict[str, object]] = []
    for raw_row in list(result.get("trial_trace", [])):
        row = dict(raw_row)
        index = int(row["measured_index"])
        in_training = index in training
        row["sampled_in_training_set"] = in_training
        row["sampled_initial_nontraining_joint_marked_candidate"] = index in candidates
        if not in_training and bool(row.get("new_ed_lp_solve")) and first_outside_evaluation is None:
            first_outside_evaluation = int(row["trial_number"])
        if not in_training and bool(row.get("true_strict_improvement")) and first_outside_improvement is None:
            first_outside_improvement = int(row["trial_number"])
        trace.append(row)
    return {
        **dict(result),
        "trial_trace": trace,
        "targeted_pilot_events": {
            "first_outside_training_evaluation_trial": first_outside_evaluation,
            "first_outside_training_strict_improvement_trial": first_outside_improvement,
        },
        "global_optimum_posthoc_status": "not_assessed",
    }


def _restore_value_model(snapshot: Mapping[str, object]) -> QuantizedSparseValueModel:
    fixed = FixedPointConfig(
        fractional_bits=int(snapshot["fractional_bits"]),
        unit=float(snapshot["cost_unit"]),
        rounding=str(snapshot["quantization_mode"]),
    )
    features = tuple(
        PhaseFeature(
            kind=str(row["kind"]), qubits=tuple(int(value) for value in row["qubits"]), label=str(row["label"])
        )
        for row in snapshot["feature_definition"]
    )
    real_intercept = float(snapshot["real_intercept"])
    real_weights = tuple(float(value) for value in snapshot["real_weights"])
    integer_intercept = int(snapshot["integer_intercept"])
    integer_weights = tuple(int(value) for value in snapshot["integer_weights"])
    errors = (
        fixed.decode(integer_intercept) - real_intercept,
        *(fixed.decode(value) - real for value, real in zip(integer_weights, real_weights)),
    )
    return QuantizedSparseValueModel(
        num_generators=2,
        num_periods=2,
        features=features,
        fixed_point_config=fixed,
        real_intercept=real_intercept,
        real_weights=real_weights,
        integer_intercept=integer_intercept,
        integer_weights=integer_weights,
        coefficient_quantization_errors=errors,
        lower_bound=int(snapshot["integer_lower_bound"]),
        upper_bound=int(snapshot["integer_upper_bound"]),
        value_shift=-int(snapshot["integer_lower_bound"]),
        shifted_upper_bound=int(snapshot["integer_upper_bound"]) - int(snapshot["integer_lower_bound"]),
        num_value_qubits=(int(snapshot["integer_upper_bound"]) - int(snapshot["integer_lower_bound"])).bit_length(),
    )


def _restore_feasibility_spec(
    snapshot: Mapping[str, object], *, generator_pair: tuple[int, int]
) -> LogicFeasibilitySpec:
    rows = sorted((dict(row) for row in snapshot["state_proxy_table"]), key=lambda row: int(row["state_index"]))
    if [int(row["state_index"]) for row in rows] != list(range(16)):
        raise PilotPreflightError("snapshot_state_proxy_table_is_not_complete")
    patterns = tuple(
        ForbiddenBooleanPattern(
            literals=tuple(BooleanLiteral(qubit=qubit, value=(index >> qubit) & 1) for qubit in range(4)),
            label=f"persisted_hard_logic_state_{index}",
        )
        for index, row in enumerate(rows)
        if not bool(row["hard_logic_feasible"])
    )
    return LogicFeasibilitySpec(
        num_generators=max(generator_pair) + 1,
        num_periods=2,
        selected_generator_indices=generator_pair,
        patterns=patterns,
    )


def restore_snapshot_scenario(
    *,
    snapshot: Mapping[str, object],
    snapshot_path: Path,
    source,
    bbht_config: Mapping[str, object],
) -> ClosedLoopScenario:
    """Restore one frozen scenario without VQC fitting or training ED/LP calls."""

    pair = tuple(int(value) for value in snapshot["generator_pair"])
    model = _restore_value_model(snapshot)
    if model.num_x_qubits != 4:
        raise PilotPreflightError("snapshot_requires_four_search_qubits")
    feasibility_spec = _restore_feasibility_spec(snapshot, generator_pair=(pair[0], pair[1]))
    rows = {int(row["state_index"]): dict(row) for row in snapshot["state_proxy_table"]}
    for index in range(16):
        bits = tuple((index >> qubit) & 1 for qubit in range(4))
        if int(model.integer_value(bits)) != int(rows[index]["integer_vqc_value"]):
            raise PilotPreflightError("snapshot_quantized_model_mismatch")
        if bool(feasibility_spec.is_feasible(bits)) != bool(rows[index]["hard_logic_feasible"]):
            raise PilotPreflightError("snapshot_hard_logic_table_mismatch")
    training_indices = tuple(int(index) for index in snapshot["training_indices"])
    if training_indices_hash(training_indices) != str(snapshot["training_indices_hash"]):
        raise PilotPreflightError("snapshot_training_indices_hash_mismatch")
    labels = tuple((int(row["state_index"]), float(row["true_cost"])) for row in snapshot["training_labels"])
    if tuple(index for index, _ in labels) != training_indices:
        raise PilotPreflightError("snapshot_training_labels_do_not_match_indices")
    cache = {
        index: ExactCandidateEvaluation(
            success=True, total_cost=cost, message="restored persisted training label",
            source="training_snapshot_cache", lp_solve_performed=False,
        )
        for index, cost in labels
    }
    if sorted(cache) != sorted(int(index) for index in snapshot["initial_cache_indices"]):
        raise PilotPreflightError("snapshot_initial_cache_is_not_training_cache")
    incumbent = int(snapshot["initial_incumbent_index"])
    if incumbent not in cache or float(cache[incumbent].total_cost) != float(snapshot["initial_incumbent_true_cost"]):
        raise PilotPreflightError("snapshot_initial_incumbent_mismatch")
    threshold = model.fixed_point_config.encode(float(cache[incumbent].total_cost))
    if threshold != _snapshot_threshold(snapshot):
        raise PilotPreflightError("snapshot_best_training_threshold_mismatch")
    instance = time_window_instance(source, start=int(snapshot["window_start"]), horizon=2)
    base_commitment = np.ones((len(instance.generators), 2), dtype=int)
    commitments = embedded_selected_commitments(base_commitment, (pair[0], pair[1]))
    evaluator = FixedCommitmentEvaluator(instance)

    def evaluate_candidate(index: int) -> ExactCandidateEvaluation:
        bits = tuple((int(index) >> qubit) & 1 for qubit in range(4))
        if not feasibility_spec.is_feasible(bits):
            return ExactCandidateEvaluation(
                success=False, total_cost=None, message="persisted hard-logic table rejected candidate",
                source="logic_infeasible_precheck", lp_solve_performed=False, logic_precheck_rejected=True,
            )
        result = evaluator.evaluate(commitments[int(index)])
        success = bool(result.success and np.isfinite(result.total_cost))
        return ExactCandidateEvaluation(
            success=success, total_cost=float(result.total_cost) if success else None,
            message=str(result.message), source="new_ed_lp_call", lp_solve_performed=True,
        )

    counts = initial_marked_counts(snapshot, encoded_threshold=threshold)
    initial_nontraining_joint_marked_indices = [
        int(index) for index in counts["initial_joint_marked_indices"]
        if int(index) not in set(training_indices)
    ]
    config = BBHTConfig(**dict(bbht_config), seed=0)
    return ClosedLoopScenario(
        scenario_id=str(snapshot["scenario_id"]), generator_pair=(pair[0], pair[1]),
        window_start=int(snapshot["window_start"]), horizon=2,
        training_seed=int(snapshot["training_data_seed"]), training_indices=training_indices,
        training_labels=labels, value_model=model, initial_incumbent_index=incumbent,
        initial_exact_cache=cache, evaluate_candidate=evaluate_candidate,
        hard_logic_is_feasible=feasibility_spec.is_feasible, commitments=commitments,
        budgets=ClosedLoopBudgets(
            max_proposals=config.max_trials, max_new_exact_evaluations=config.max_new_ed_lp_calls,
            max_actual_ed_lp_solves=None, max_threshold_updates=config.max_threshold_updates,
            max_consecutive_nonimproving_marked=config.max_consecutive_nonimproving_marked,
            max_same_encoded_threshold_updates=config.max_same_encoded_threshold_updates,
            max_auxiliary_syndrome_rejections=config.max_auxiliary_syndrome_rejections,
        ),
        bbht_config=config,
        reproducibility_metadata={
            "initialization_policy": "best_training", "initial_incumbent_policy": "best_training",
            "initial_incumbent_true_cost": float(cache[incumbent].total_cost),
            "training_data_seed": int(snapshot["training_data_seed"]), "model_seed": int(snapshot["model_seed"]),
            "training_indices_hash": str(snapshot["training_indices_hash"]),
            "training_index_policy": str(snapshot["training_index_policy"]),
            "snapshot_path": str(snapshot_path), "snapshot_sha256": canonical_sha256(snapshot),
            "initial_nontraining_joint_marked_indices": initial_nontraining_joint_marked_indices,
            "best_training_encoded_threshold": threshold,
            "global_truth_used_online": False, "global_truth_used_for_posthoc_validation": False,
        },
        feasibility_spec=feasibility_spec,
    )


def _write_execution_summary(output_dir: Path, *, planned_runs: int) -> dict[str, object]:
    completed_paths = sorted((output_dir / "runs" / "completed").glob("*.json"))
    failures = list((output_dir / "runs" / "failed").glob("*.json"))
    trial_rows: list[Mapping[str, object]] = []
    per_scenario: dict[str, dict[str, int]] = {}
    search_edlp: list[int] = []
    strict_runs = 0
    stop_reason_counts: dict[str, int] = {}
    no_improvement_stops = 0
    for path in completed_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenario_id = str(dict(payload["scenario"])["scenario_id"])
        result = dict(payload["result"])
        trace = [dict(row) for row in result.get("trial_trace", [])]
        trial_rows.extend(trace)
        summary = per_scenario.setdefault(scenario_id, {"runs": 0, "strict_improvement_runs": 0})
        summary["runs"] += 1
        strict = any(bool(row.get("true_strict_improvement")) and not bool(row.get("sampled_in_training_set")) for row in trace)
        summary["strict_improvement_runs"] += int(strict)
        strict_runs += int(strict)
        stop_reason = str(result.get("stop_reason", "unknown"))
        stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1
        no_improvement_stops += int(not strict)
        search_edlp.append(int(dict(result.get("counters", {})).get("actual_ed_lp_solves", 0)))
    candidate_hits = sum(bool(row.get("sampled_initial_nontraining_joint_marked_candidate")) for row in trial_rows)
    outside_edlp = sum(not bool(row.get("sampled_in_training_set")) and bool(row.get("new_ed_lp_solve")) for row in trial_rows)
    training_marked = sum(bool(row.get("candidate_in_training_set")) and bool(row.get("measured_joint_marked")) for row in trial_rows)
    nontraining_marked = sum(not bool(row.get("candidate_in_training_set")) and bool(row.get("measured_joint_marked")) for row in trial_rows)
    ordered = sorted(search_edlp)
    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * fraction
        lower, upper = int(position), int(np.ceil(position))
        return float(ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))
    summary = {
        "scope": "targeted case-study pilot; not a population success-rate estimate",
        "planned_runs": int(planned_runs), "completed_runs": len(completed_paths), "failed_runs": len(failures),
        "per_scenario": per_scenario,
        "outside_training_candidate_sampled_count": candidate_hits,
        "outside_training_candidate_sampled_rate_per_trial": candidate_hits / len(trial_rows) if trial_rows else None,
        "outside_training_edlp_evaluation_count": outside_edlp,
        "outside_training_edlp_evaluation_rate_per_trial": outside_edlp / len(trial_rows) if trial_rows else None,
        "outside_training_strict_improvement_run_count": strict_runs,
        "outside_training_strict_improvement_run_rate": strict_runs / len(completed_paths) if completed_paths else None,
        "global_optimum_posthoc_status": "not_assessed",
        "training_state_marked_hits": training_marked, "nontraining_marked_hits": nontraining_marked,
        "stop_reason_counts": stop_reason_counts,
        "runs_stopped_without_outside_training_strict_improvement": no_improvement_stops,
        "search_stage_new_edlp": {
            "total": sum(search_edlp), "mean_per_run": sum(search_edlp) / len(search_edlp) if search_edlp else None,
            "median_per_run": percentile(0.5), "q25_per_run": percentile(0.25), "q75_per_run": percentile(0.75),
            "standard_deviation_per_run": float(np.std(search_edlp, ddof=1)) if len(search_edlp) >= 2 else None,
        },
        "error_count": len(failures),
    }
    atomic_write_json(output_dir / "summaries" / "targeted_pilot_summary.json", summary)
    return summary


def main() -> int:
    args = build_argument_parser().parse_args()
    if not args.preflight and not args.run:
        raise PilotPreflightError("choose --preflight or --run")
    manifest = json.loads(args.config.read_text(encoding="utf-8"))
    report = preflight_manifest(manifest)
    head = str(manifest.get("expected_code_sha", "")) or "snapshot-restored"
    specs = build_targeted_run_specs(manifest, expected_head=head)
    plan = {**report, "planned_runs": len(specs), "run_seeds": list(FIXED_RUN_SEEDS), "method": PRIMARY_METHOD}
    if args.preflight:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    ensure_new_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True)
    atomic_write_json(args.output_dir / "pilot_execution_manifest.json", {**plan, "source_manifest": str(args.config)})
    source = load_uc_instance(args.instance)
    entries = {str(item["scenario_id"]): dict(item) for item in manifest["scenarios"]}
    bbht_config = _fixed_bbht_config(manifest)

    def scenario_builder(spec: RunSpec) -> ClosedLoopScenario:
        entry = entries[spec.scenario_id]
        return restore_snapshot_scenario(
            snapshot=_load_snapshot(Path(str(entry["snapshot"]))), snapshot_path=Path(str(entry["snapshot"])),
            source=source, bbht_config=bbht_config,
        )

    def method_runner(scenario: ClosedLoopScenario, method: str, run_seed: int) -> dict[str, object]:
        envelope = run_closed_loop_method(scenario, method, run_seed=run_seed, persist_dynamic_oracle_metadata=True)
        snapshot = _load_snapshot(Path(str(scenario.reproducibility_metadata["snapshot_path"])))
        envelope["result_schema"] = augment_targeted_result_schema(
            dict(envelope["result_schema"]), training_indices=scenario.training_indices,
            initial_nontraining_joint_marked_indices=scenario.reproducibility_metadata[
                "initial_nontraining_joint_marked_indices"
            ],
        )
        envelope["quantized_model_snapshot"] = snapshot
        return envelope

    executor = ClosedLoopBatchExecutor(
        output_dir=args.output_dir,
        code={"execution": "snapshot_restored_targeted_candidate_pilot", "expected_head": head},
        scenario_builder=scenario_builder,
        method_runner=method_runner,
        workers=1,
    )
    counts = executor.execute(specs, continue_on_error=True)
    summary = _write_execution_summary(args.output_dir, planned_runs=len(specs))
    print(json.dumps({**plan, "counts": counts, "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
