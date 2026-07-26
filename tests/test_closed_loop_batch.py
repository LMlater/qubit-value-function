from __future__ import annotations

import json
from pathlib import Path

import pytest

from qubit_value_function.closed_loop_batch import (
    BatchConfigurationError,
    ClosedLoopBatchExecutor,
    ResumeConflictError,
    atomic_write_json,
    build_run_specs,
    derive_seed,
    preset_selection,
)


def _specs(*, methods=("full_space_random",), run_seeds=(0,), batch_id="test"):
    return build_run_specs(
        batch_id=batch_id,
        preset="custom",
        generator_pairs=((0, 1),),
        windows=(0,),
        training_seeds=(0,),
        methods=methods,
        run_seeds=run_seeds,
        budget_config={"max_trials": 2},
        fixed_point_config={"fractional_bits": 2, "cost_unit": 1000.0},
        initialization_policy="first",
        expected_code_sha="abc123",
    )


def _runner(_scenario, method, run_seed):
    return {
        "method": method,
        "method_role": "random_baseline",
        "diagnostic_only": False,
        "scenario": {"scenario_id": "fake", "training_indices": [1]},
        "result": object(),
        "result_schema": {
            "method": method,
            "initial_incumbent_true_cost": 10.0,
            "final_incumbent_true_cost": 9.0,
            "stop_reason": "done",
            "counters": {"proposals_used": 1, "threshold_updates": 1},
        },
    }


def _executor(output_dir: Path, builds: list[object] | None = None, runner=_runner):
    builds = builds if builds is not None else []

    def build(spec):
        builds.append(spec.scenario_group_key)
        return {"group": spec.scenario_group_key}

    return ClosedLoopBatchExecutor(
        output_dir=output_dir,
        code={"branch": "test", "head": "abc123"},
        scenario_builder=build,
        method_runner=runner,
    )


def test_presets_have_required_run_and_mps_counts() -> None:
    for preset, expected in (("smoke", 48), ("pilot", 144), ("formal", 1080)):
        selected = preset_selection(preset)
        specs = build_run_specs(
            batch_id=preset,
            preset=preset,
            methods=(
                "joint_bbht", "cost_only_bbht", "full_space_random", "logic_rejection_random",
                "direct_logic_feasible_random", "classical_joint_marked_random",
            ),
            budget_config={}, fixed_point_config={}, initialization_policy="first", expected_code_sha="x",
            **selected,
        )
        assert len(specs) == expected
        assert sum(spec.method in {"joint_bbht", "cost_only_bbht"} for spec in specs) == expected // 3
    assert sum(spec.method in {"joint_bbht", "cost_only_bbht"} for spec in build_run_specs(
        batch_id="formal", preset="formal", methods=("joint_bbht", "cost_only_bbht", "full_space_random", "logic_rejection_random", "direct_logic_feasible_random", "classical_joint_marked_random"), budget_config={}, fixed_point_config={}, initialization_policy="first", expected_code_sha="x", **preset_selection("formal")
    )) == 360


def test_run_identity_fingerprint_and_derived_seeds_are_stable() -> None:
    spec = _specs()[0]
    assert spec.run_id == "case14_g0-g1_w0_train0_full_space_random_run0"
    assert spec.fingerprint == _specs()[0].fingerprint
    assert derive_seed(1, "s", "m", "p") == derive_seed(1, "s", "m", "p")
    changed = build_run_specs(
        batch_id="test", preset="custom", generator_pairs=((0, 1),), windows=(1,), training_seeds=(0,),
        methods=("full_space_random",), run_seeds=(0,), budget_config={"max_trials": 2},
        fixed_point_config={"fractional_bits": 2, "cost_unit": 1000.0}, initialization_policy="first", expected_code_sha="abc123",
    )[0]
    assert changed.fingerprint != spec.fingerprint


def test_atomic_write_and_completed_resume_keep_only_serializable_schema(tmp_path: Path) -> None:
    output = tmp_path / "directory with spaces"
    executor = _executor(output)
    specs = _specs()
    assert executor.execute(specs) == {"completed": 1, "skipped": 0, "failed": 0, "pending": 0}
    completed = output / "runs" / "completed" / f"{specs[0].run_id}.json"
    payload = json.loads(completed.read_text(encoding="utf-8"))
    assert payload["result"]["method"] == "full_space_random"
    assert "result_schema" not in payload and not list(completed.parent.glob("*.tmp"))
    assert executor.execute(specs, resume=True)["skipped"] == 1


def test_resume_rejects_corrupt_or_fingerprint_mismatched_completed(tmp_path: Path) -> None:
    output = tmp_path / "out"
    executor = _executor(output)
    specs = _specs()
    executor.execute(specs)
    path = output / "runs" / "completed" / f"{specs[0].run_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fingerprint"] = "wrong"
    atomic_write_json(path, payload)
    with pytest.raises(ResumeConflictError):
        executor.execute(specs, resume=True)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ResumeConflictError):
        executor.execute(specs, resume=True)


def test_failed_run_retries_and_other_runs_continue(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(_scenario, method, run_seed):
        calls.append(method)
        if method == "full_space_random":
            raise RuntimeError("expected fake failure")
        return _runner(_scenario, method, run_seed)

    specs = _specs(methods=("full_space_random", "logic_rejection_random"))
    executor = _executor(tmp_path / "out", runner=runner)
    counts = executor.execute(specs, continue_on_error=True)
    assert counts["failed"] == 1 and counts["completed"] == 1
    assert calls == ["full_space_random", "logic_rejection_random"]
    assert executor.execute(specs, resume=True, skip_failed=True)["skipped"] == 2


def test_scenario_is_built_once_per_group_and_workers_above_one_rejected(tmp_path: Path) -> None:
    builds: list[object] = []
    specs = _specs(methods=("full_space_random", "logic_rejection_random"), run_seeds=(0, 1))
    _executor(tmp_path / "out", builds).execute(specs)
    assert len(builds) == 1
    with pytest.raises(BatchConfigurationError, match="workers 1"):
        ClosedLoopBatchExecutor(output_dir=tmp_path, code={}, scenario_builder=lambda spec: spec, method_runner=_runner, workers=2)


def test_interrupt_preserves_completed_records_and_marks_manifest(tmp_path: Path) -> None:
    calls = 0

    def runner(_scenario, method, run_seed):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return _runner(_scenario, method, run_seed)

    executor = _executor(tmp_path / "out", runner=runner)
    specs = _specs(methods=("full_space_random", "logic_rejection_random"))
    with pytest.raises(KeyboardInterrupt):
        executor.execute(specs)
    assert len(list((tmp_path / "out" / "runs" / "completed").glob("*.json"))) == 1
    manifest = json.loads((tmp_path / "out" / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["interrupted"] is True
