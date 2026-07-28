"""可恢复的阶段 A 闭环基线批量执行基础设施。

本模块刻意不持久化 ``ClosedLoopScenario``：它包含 callable、冻结映射和量子
模型。恢复时按场景组确定性重建，方法运行仍委托给既有的
``run_closed_loop_method``。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4


BATCH_SCHEMA_VERSION = "stage-a-closed-loop-batch-v1"
METHODS = (
    "joint_bbht",
    "cost_only_bbht",
    "full_space_random",
    "logic_rejection_random",
    "direct_logic_feasible_random",
    "classical_joint_marked_random",
)
MPS_METHODS = frozenset(("joint_bbht", "cost_only_bbht"))
DEFAULT_GENERATOR_PAIRS = ((0, 1), (0, 5), (1, 3), (1, 5))


class BatchConfigurationError(ValueError):
    """批量计划或参数在执行前无效。"""


class ResumeConflictError(RuntimeError):
    """已有同名结果但其语义身份无法安全复用。"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def derive_seed(master_seed: int, scenario_id: str, method: str, purpose: str) -> int:
    """跨解释器稳定的 32 位派生种子，绝不依赖 Python ``hash``。"""

    digest = hashlib.sha256(
        f"{int(master_seed)}|{scenario_id}|{method}|{purpose}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


@dataclass(frozen=True)
class RunSpec:
    batch_id: str
    generator_pair: tuple[int, int]
    window_start: int
    training_seed: int
    method: str
    run_seed: int
    preset: str
    budget_config: Mapping[str, object]
    fixed_point_config: Mapping[str, object]
    initialization_policy: str
    expected_code_sha: str
    schema_version: str = BATCH_SCHEMA_VERSION

    @property
    def run_id(self) -> str:
        left, right = self.generator_pair
        return (
            f"case14_g{int(left)}-g{int(right)}_w{int(self.window_start)}"
            f"_train{int(self.training_seed)}_{self.method}_run{int(self.run_seed)}"
        )

    @property
    def scenario_id(self) -> str:
        left, right = self.generator_pair
        return f"case14-g{int(left)}g{int(right)}-w{int(self.window_start)}-s{int(self.training_seed)}"

    @property
    def method_seed(self) -> int:
        return derive_seed(self.run_seed, self.scenario_id, self.method, "bbht")

    def semantic_config(self) -> dict[str, object]:
        payload = asdict(self)
        payload["generator_pair"] = list(self.generator_pair)
        payload["budget_config"] = dict(self.budget_config)
        payload["fixed_point_config"] = dict(self.fixed_point_config)
        payload["run_id"] = self.run_id
        payload["scenario_id"] = self.scenario_id
        payload["method_seed"] = self.method_seed
        return payload

    @property
    def fingerprint(self) -> str:
        return _sha256(self.semantic_config())

    def as_dict(self) -> dict[str, object]:
        payload = self.semantic_config()
        payload["fingerprint"] = self.fingerprint
        return payload

    @property
    def scenario_group_key(self) -> tuple[tuple[int, int], int, int]:
        return (self.generator_pair, int(self.window_start), int(self.training_seed))

    def seed_metadata(self) -> dict[str, object]:
        return {
            "training_seed": int(self.training_seed),
            "master_run_seed": int(self.run_seed),
            "scenario_id": self.scenario_id,
            "method_seed": self.method_seed,
            "method_seed_derivation": {
                "algorithm": "sha256",
                "inputs": [int(self.run_seed), self.scenario_id, self.method, "bbht"],
            },
            "per_trial_execution_seeds_location": "result.trial_trace[*].execution_seed",
        }


def validate_selection(
    methods: Sequence[str],
    generator_pairs: Sequence[tuple[int, int]],
    windows: Sequence[int],
    training_seeds: Sequence[int],
    run_seeds: Sequence[int],
) -> None:
    groups = {
        "methods": tuple(methods),
        "generator_pairs": tuple(generator_pairs),
        "windows": tuple(windows),
        "training_seeds": tuple(training_seeds),
        "run_seeds": tuple(run_seeds),
    }
    for name, values in groups.items():
        if not values:
            raise BatchConfigurationError(f"{name} 不能为空")
        if len(set(values)) != len(values):
            raise BatchConfigurationError(f"{name} 不允许重复项")
    invalid = sorted(set(methods) - set(METHODS))
    if invalid:
        raise BatchConfigurationError(f"不支持的方法: {', '.join(invalid)}")
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in generator_pairs):
        raise BatchConfigurationError("generator_pairs 必须是两个不同机组索引组成的 pair")


def preset_selection(preset: str) -> dict[str, tuple[object, ...]]:
    if preset == "smoke":
        manifest = {
            "generator_pairs": DEFAULT_GENERATOR_PAIRS,
            "windows": (0,),
            "training_seeds": (0,),
            "run_seeds": (0, 1),
        }
    if preset == "pilot":
        return {
            "generator_pairs": DEFAULT_GENERATOR_PAIRS,
            "windows": (0, 1, 2),
            "training_seeds": (0,),
            "run_seeds": (0, 1),
        }
    if preset == "formal":
        return {
            "generator_pairs": DEFAULT_GENERATOR_PAIRS,
            "windows": (0, 1, 2),
            "training_seeds": (0, 1, 2),
            "run_seeds": (0, 1, 2, 3, 4),
        }
    if preset == "custom":
        return {}
    raise BatchConfigurationError(f"未知 preset: {preset}")


def build_run_specs(
    *,
    batch_id: str,
    preset: str,
    generator_pairs: Sequence[tuple[int, int]],
    windows: Sequence[int],
    training_seeds: Sequence[int],
    methods: Sequence[str] = METHODS,
    run_seeds: Sequence[int],
    budget_config: Mapping[str, object],
    fixed_point_config: Mapping[str, object],
    initialization_policy: str,
    expected_code_sha: str,
) -> tuple[RunSpec, ...]:
    validate_selection(methods, generator_pairs, windows, training_seeds, run_seeds)
    specs = tuple(
        RunSpec(
            batch_id=str(batch_id),
            generator_pair=(int(pair[0]), int(pair[1])),
            window_start=int(window),
            training_seed=int(training_seed),
            method=str(method),
            run_seed=int(run_seed),
            preset=str(preset),
            budget_config=dict(budget_config),
            fixed_point_config=dict(fixed_point_config),
            initialization_policy=str(initialization_policy),
            expected_code_sha=str(expected_code_sha),
        )
        for pair in generator_pairs
        for window in windows
        for training_seed in training_seeds
        for method in methods
        for run_seed in run_seeds
    )
    if len({spec.run_id for spec in specs}) != len(specs):
        raise BatchConfigurationError("计划中出现重复 run_id")
    return specs


def batch_config_fingerprint(specs: Sequence[RunSpec]) -> str:
    return _sha256([spec.as_dict() for spec in specs])


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """同目录临时文件 + flush/fsync + os.replace，避免半写 completed 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeConflictError(f"无法安全读取已有结果 {path}: {error}") from error
    if not isinstance(data, dict):
        raise ResumeConflictError(f"已有结果不是 JSON object: {path}")
    return data


def _validate_completed_payload(payload: Mapping[str, object], spec: RunSpec) -> None:
    required = {"schema_version", "status", "run_id", "fingerprint", "code", "run_spec", "seeds", "scenario", "result", "timing"}
    absent = sorted(required - set(payload))
    if absent:
        raise ResumeConflictError(f"completed JSON 缺少关键字段: {', '.join(absent)}")
    if payload["status"] != "completed":
        raise ResumeConflictError("已有 completed 文件的 status 非 completed")
    if payload["schema_version"] != BATCH_SCHEMA_VERSION:
        raise ResumeConflictError("已有 completed 文件 schema_version 不兼容")
    if payload["run_id"] != spec.run_id or payload["fingerprint"] != spec.fingerprint:
        raise ResumeConflictError("同名 run 的身份或 fingerprint 不一致；请使用新的 output-dir/batch-id")
    code = payload["code"]
    if not isinstance(code, Mapping) or code.get("head") != spec.expected_code_sha:
        raise ResumeConflictError("已有 completed 文件 code SHA 不一致；请使用新的 output-dir/batch-id")
    if not isinstance(payload["result"], Mapping):
        raise ResumeConflictError("已有 completed 文件 result 不是可汇总 JSON object")
    seeds = payload["seeds"]
    if not isinstance(seeds, Mapping) or seeds.get("method_seed") != spec.method_seed:
        raise ResumeConflictError("已有 completed 文件 method_seed 不匹配")
    scenario = payload["scenario"]
    if not isinstance(scenario, Mapping) or scenario.get("scenario_id") != spec.scenario_id:
        raise ResumeConflictError("已有 completed 文件 scenario_id 不匹配")


def _validate_failed_payload(payload: Mapping[str, object], spec: RunSpec) -> None:
    if payload.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise ResumeConflictError("已有 failed 文件 schema_version 不兼容")
    if payload.get("run_id") != spec.run_id or payload.get("fingerprint") != spec.fingerprint:
        raise ResumeConflictError("同名 failed run 的 fingerprint 不一致；请使用新的 output-dir/batch-id")


ScenarioBuilder = Callable[[RunSpec], object]
# 第三个参数是实际传给 run_closed_loop_method / numpy RNG 的 method_seed。
MethodRunner = Callable[[object, str, int], Mapping[str, object]]
ProgressReporter = Callable[[Mapping[str, object]], None]


class ClosedLoopBatchExecutor:
    """按场景组构建一次，然后为每个 method/run seed 运行独立闭环。"""

    def __init__(
        self,
        *,
        output_dir: Path,
        code: Mapping[str, object],
        scenario_builder: ScenarioBuilder,
        method_runner: MethodRunner,
        workers: int = 1,
    ) -> None:
        if int(workers) != 1:
            raise BatchConfigurationError(
                "当前 Windows 批量实现只支持 --workers 1；不能跨进程传递 ClosedLoopScenario"
            )
        self.output_dir = Path(output_dir)
        self.code = dict(code)
        self.scenario_builder = scenario_builder
        self.method_runner = method_runner
        self.workers = 1

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "batch_manifest.json"

    def _completed_path(self, spec: RunSpec) -> Path:
        return self.output_dir / "runs" / "completed" / f"{spec.run_id}.json"

    def _failed_path(self, spec: RunSpec) -> Path:
        return self.output_dir / "runs" / "failed" / f"{spec.run_id}.json"

    def _persist_quantized_model_snapshot(
        self,
        snapshot: Mapping[str, object] | None,
        *,
        scenario_id: str,
    ) -> dict[str, object] | None:
        """Write one immutable snapshot per scenario, never per method/run."""

        if snapshot is None:
            return None
        if str(snapshot.get("scenario_id")) != str(scenario_id):
            raise RuntimeError("quantized model snapshot scenario_id does not match completed run")
        relative = Path("scenario_snapshots") / f"{scenario_id}.json"
        path = self.output_dir / relative
        snapshot_dict = dict(snapshot)
        if path.exists():
            existing = _read_json(path)
            if _sha256(existing) != _sha256(snapshot_dict):
                raise ResumeConflictError("same scenario produced a different quantized model snapshot")
        else:
            atomic_write_json(path, snapshot_dict)
        return {
            "version": snapshot_dict.get("quantized_model_snapshot_version"),
            "path": relative.as_posix(),
            "sha256": _sha256(snapshot_dict),
        }

    def _existing_action(self, spec: RunSpec, *, resume: bool, skip_failed: bool) -> str:
        completed = self._completed_path(spec)
        failed = self._failed_path(spec)
        if completed.exists():
            payload = _read_json(completed)
            _validate_completed_payload(payload, spec)
            if failed.exists():
                _validate_failed_payload(_read_json(failed), spec)
            if not resume:
                raise ResumeConflictError(f"结果已存在: {completed}；请使用 --resume 或新的 output-dir")
            return "skip_completed"
        if failed.exists():
            payload = _read_json(failed)
            _validate_failed_payload(payload, spec)
            if not resume:
                raise ResumeConflictError(f"失败记录已存在: {failed}；请使用 --resume 或新的 output-dir")
            return "skip_failed" if skip_failed else "retry_failed"
        return "run"

    def _manifest(
        self,
        specs: Sequence[RunSpec],
        counts: Mapping[str, int],
        *,
        interrupted: bool,
        started_at: str,
    ) -> dict[str, object]:
        groups = sorted({
            (spec.generator_pair, spec.window_start, spec.training_seed) for spec in specs
        })
        manifest = {
            "schema_version": BATCH_SCHEMA_VERSION,
            "batch_id": specs[0].batch_id if specs else "",
            "preset": specs[0].preset if specs else "",
            "created_at": started_at,
            "updated_at": utc_now(),
            "planned_runs": len(specs),
            "planned_run_ids": [spec.run_id for spec in specs],
            "planned_fingerprints": {spec.run_id: spec.fingerprint for spec in specs},
            "methods": sorted({spec.method for spec in specs}),
            "scenarios": [
                {"generator_pair": list(pair), "window_start": window, "training_seed": seed}
                for pair, window, seed in groups
            ],
            "run_seeds": sorted({spec.run_seed for spec in specs}),
            "expected_head": specs[0].expected_code_sha if specs else "",
            "budget_config": dict(specs[0].budget_config) if specs else {},
            "fixed_point_config": dict(specs[0].fixed_point_config) if specs else {},
            "initialization_policy": specs[0].initialization_policy if specs else "",
            "config_fingerprint": batch_config_fingerprint(specs),
            "counts": dict(counts),
            "interrupted": bool(interrupted),
            "environment": dict(self.code),
            "workers": self.workers,
        }
        for key in (
            "actual_execution_head",
            "manifest_declared_code_sha",
            "externally_expected_head",
            "head_match",
            "working_tree_tracked_clean",
        ):
            if key in self.code:
                manifest[key] = self.code[key]
        return manifest

    def _write_manifest(
        self, specs: Sequence[RunSpec], counts: Mapping[str, int], *, interrupted: bool, started_at: str
    ) -> None:
        atomic_write_json(
            self.manifest_path,
            self._manifest(specs, counts, interrupted=interrupted, started_at=started_at),
        )

    def execute(
        self,
        specs: Sequence[RunSpec],
        *,
        resume: bool = False,
        continue_on_error: bool = True,
        skip_failed: bool = False,
        progress: ProgressReporter | None = None,
    ) -> dict[str, int]:
        if not specs:
            raise BatchConfigurationError("没有可执行的 run")
        if len({spec.run_id for spec in specs}) != len(specs):
            raise BatchConfigurationError("计划中出现重复 run_id")
        if self.manifest_path.exists():
            existing_manifest = _read_json(self.manifest_path)
            if (
                existing_manifest.get("config_fingerprint") != batch_config_fingerprint(specs)
                or existing_manifest.get("batch_id") != specs[0].batch_id
            ):
                raise ResumeConflictError(
                    "已有 manifest 与当前计划不一致；请使用新的 output-dir 或 batch-id"
                )
        actions = {spec.run_id: self._existing_action(spec, resume=resume, skip_failed=skip_failed) for spec in specs}
        counts = {"completed": 0, "skipped": 0, "failed": 0, "pending": len(specs)}
        started_at = utc_now()
        self._write_manifest(specs, counts, interrupted=False, started_at=started_at)
        grouped: dict[tuple[tuple[int, int], int, int], list[RunSpec]] = {}
        for spec in specs:
            grouped.setdefault(spec.scenario_group_key, []).append(spec)
        try:
            for group_specs in grouped.values():
                pending = [spec for spec in group_specs if actions[spec.run_id] in {"run", "retry_failed"}]
                for spec in group_specs:
                    if actions[spec.run_id].startswith("skip"):
                        counts["skipped"] += 1
                        counts["pending"] -= 1
                if not pending:
                    self._write_manifest(specs, counts, interrupted=False, started_at=started_at)
                    continue
                scenario = self.scenario_builder(pending[0])
                for spec in pending:
                    started = utc_now()
                    timer = datetime.now(timezone.utc)
                    try:
                        scenario_id = getattr(scenario, "scenario_id", spec.scenario_id)
                        if scenario_id != spec.scenario_id:
                            raise RuntimeError("场景构建器返回的 scenario_id 与 RunSpec 不一致")
                        envelope = self.method_runner(scenario, spec.method, spec.method_seed)
                        result = envelope.get("result_schema")
                        if not isinstance(result, Mapping):
                            raise TypeError("方法执行器未返回 JSON result_schema")
                        scenario_payload = dict(envelope.get("scenario", {}))
                        snapshot = envelope.get("quantized_model_snapshot")
                        if snapshot is not None and not isinstance(snapshot, Mapping):
                            raise TypeError("quantized_model_snapshot must be a JSON mapping")
                        snapshot_reference = self._persist_quantized_model_snapshot(
                            snapshot,
                            scenario_id=str(scenario_payload.get("scenario_id", "")),
                        )
                        if snapshot_reference is not None:
                            scenario_payload["quantized_model_snapshot"] = snapshot_reference
                        payload: dict[str, object] = {
                            "schema_version": BATCH_SCHEMA_VERSION,
                            "status": "completed",
                            "run_id": spec.run_id,
                            "fingerprint": spec.fingerprint,
                            "code": dict(self.code),
                            "batch": {"batch_id": spec.batch_id, "preset": spec.preset},
                            "run_spec": spec.as_dict(),
                            "seeds": spec.seed_metadata(),
                            "method": str(envelope.get("method", spec.method)),
                            "method_role": str(envelope.get("method_role", "unknown")),
                            "diagnostic_only": bool(envelope.get("diagnostic_only", False)),
                            "scenario": scenario_payload,
                            "result": dict(result),
                            "timing": {
                                "started_at": started,
                                "ended_at": utc_now(),
                                "elapsed_seconds": (datetime.now(timezone.utc) - timer).total_seconds(),
                            },
                        }
                        if "result_schema_version" in envelope:
                            payload["result_schema_version"] = str(envelope["result_schema_version"])
                        if payload["scenario"].get("scenario_id") != spec.scenario_id:
                            raise RuntimeError("方法执行器返回的 scenario_id 与 RunSpec 不一致")
                        json.dumps(payload, ensure_ascii=False)
                        atomic_write_json(self._completed_path(spec), payload)
                        failed_path = self._failed_path(spec)
                        if failed_path.exists():
                            _validate_failed_payload(_read_json(failed_path), spec)
                            try:
                                failed_path.unlink()
                            except OSError:
                                # completed 已是权威结果；下一次 resume/summary 会将旧 failed 视为 recovered。
                                pass
                        counts["completed"] += 1
                    except KeyboardInterrupt:
                        raise
                    except Exception as error:  # 单个 run 的失败必须可恢复地落盘。
                        payload = {
                            "schema_version": BATCH_SCHEMA_VERSION,
                            "status": "failed",
                            "run_id": spec.run_id,
                            "fingerprint": spec.fingerprint,
                            "code": dict(self.code),
                            "batch": {"batch_id": spec.batch_id, "preset": spec.preset},
                            "run_spec": spec.as_dict(),
                            "seeds": spec.seed_metadata(),
                            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
                            "timing": {
                                "started_at": started,
                                "ended_at": utc_now(),
                                "elapsed_seconds": (datetime.now(timezone.utc) - timer).total_seconds(),
                            },
                        }
                        atomic_write_json(self._failed_path(spec), payload)
                        counts["failed"] += 1
                        if not continue_on_error:
                            raise
                    finally:
                        counts["pending"] -= 1
                        self._write_manifest(specs, counts, interrupted=False, started_at=started_at)
                        if progress is not None:
                            progress({"run_id": spec.run_id, "method": spec.method, "total": len(specs), **counts})
        except KeyboardInterrupt:
            self._write_manifest(specs, counts, interrupted=True, started_at=started_at)
            raise
        return dict(counts)
