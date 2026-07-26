from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from qubit_value_function.closed_loop_batch import ClosedLoopBatchExecutor, build_run_specs
from qubit_value_function.closed_loop_result_summary import SummaryValidationError, summarize_batch


def _specs():
    return build_run_specs(
        batch_id="summary", preset="custom", generator_pairs=((0, 1),), windows=(0,), training_seeds=(0,),
        methods=("full_space_random", "direct_logic_feasible_random"), run_seeds=(0,),
        budget_config={}, fixed_point_config={}, initialization_policy="first", expected_code_sha="head",
    )


def _runner(_scenario, method, _run_seed):
    return {
        "method": method,
        "method_role": "diagnostic_only" if method.startswith("direct") else "random_baseline",
        "diagnostic_only": method.startswith("direct"),
        "scenario": {"scenario_id": "case14-g0g1-w0-s0"},
        "result_schema": {
            "initial_incumbent_true_cost": 12.0,
            "final_incumbent_true_cost": 10.0,
            "stop_reason": "done",
            "counters": {"proposals_used": 2, "threshold_updates": 1, "actual_ed_lp_solves": 1},
        },
    }


def test_summary_reads_only_json_and_separates_diagnostic_rows(tmp_path: Path) -> None:
    invoked = 0

    def builder(_spec):
        nonlocal invoked
        invoked += 1
        return object()

    output = tmp_path / "output with spaces"
    executor = ClosedLoopBatchExecutor(output_dir=output, code={"head": "head"}, scenario_builder=builder, method_runner=_runner)
    executor.execute(_specs())
    before = invoked
    summary = summarize_batch(output)
    assert invoked == before
    assert summary["completed_runs"] == 2
    assert summary["diagnostic_only_runs"] == 1
    assert summary["global_optimum"].startswith("unavailable")
    with (output / "summaries" / "run_index.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_summary_rejects_completed_fingerprint_not_in_manifest(tmp_path: Path) -> None:
    output = tmp_path / "out"
    executor = ClosedLoopBatchExecutor(output_dir=output, code={"head": "head"}, scenario_builder=lambda _: object(), method_runner=_runner)
    executor.execute(_specs())
    completed = next((output / "runs" / "completed").glob("*.json"))
    text = completed.read_text(encoding="utf-8").replace('"fingerprint": "', '"fingerprint": "bad', 1)
    completed.write_text(text, encoding="utf-8")
    with pytest.raises(SummaryValidationError):
        summarize_batch(output)


def test_summary_treats_matching_completed_failed_overlap_as_recovered(tmp_path: Path) -> None:
    output = tmp_path / "out"
    executor = ClosedLoopBatchExecutor(output_dir=output, code={"head": "head"}, scenario_builder=lambda _: object(), method_runner=_runner)
    executor.execute(_specs())
    completed = next((output / "runs" / "completed").glob("*.json"))
    payload = json.loads(completed.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    payload["error"] = {"type": "Interrupted", "message": "after completed", "traceback": ""}
    failed = output / "runs" / "failed" / completed.name
    failed.parent.mkdir(parents=True, exist_ok=True)
    failed.write_text(json.dumps(payload), encoding="utf-8")
    summary = summarize_batch(output)
    assert summary["failed_runs"] == 0
    assert summary["recovered_failed_runs"] == 1


def test_summary_rejects_fingerprint_mismatched_completed_failed_overlap(tmp_path: Path) -> None:
    output = tmp_path / "out"
    executor = ClosedLoopBatchExecutor(output_dir=output, code={"head": "head"}, scenario_builder=lambda _: object(), method_runner=_runner)
    executor.execute(_specs())
    completed = next((output / "runs" / "completed").glob("*.json"))
    payload = json.loads(completed.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    payload["fingerprint"] = "incompatible"
    failed = output / "runs" / "failed" / completed.name
    failed.parent.mkdir(parents=True, exist_ok=True)
    failed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SummaryValidationError):
        summarize_batch(output)
