"""Windows CMD 入口：阶段 A 六方法闭环批量、恢复与只读汇总。"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_2x2_joint_feasible_sparse_vqc_bbht import (  # noqa: E402
    build_case14_closed_loop_scenario,
)
from qubit_value_function.closed_loop_batch import (  # noqa: E402
    METHODS,
    MPS_METHODS,
    BatchConfigurationError,
    ClosedLoopBatchExecutor,
    build_run_specs,
    preset_selection,
)
from qubit_value_function.closed_loop_result_summary import summarize_batch  # noqa: E402
from qubit_value_function.closed_loop_scenario import run_closed_loop_method  # noqa: E402
from qubit_value_function.fixed_point_oracle import FixedPointConfig  # noqa: E402
from qubit_value_function.sparse_vqc_bbht import BBHTConfig  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def _csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("参数不能为空")
    return values


def _pairs(raw: str) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for item in raw.split(";"):
        values = _csv_ints(item)
        if len(values) != 2:
            raise argparse.ArgumentTypeError("每个 generator pair 必须形如 0,1")
        pairs.append((values[0], values[1]))
    if not pairs:
        raise argparse.ArgumentTypeError("至少需要一个 generator pair")
    return tuple(pairs)


def _methods(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("methods 不能为空")
    return values


def _git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return completed.stdout.strip()


def _environment(expected_head: str, workers: int) -> dict[str, object]:
    status_lines = _git("status", "--porcelain").splitlines()
    try:
        import qiskit
        qiskit_version = getattr(qiskit, "__version__", "unknown")
    except ImportError:
        qiskit_version = "unavailable"
    try:
        import qiskit_aer
        aer_version = getattr(qiskit_aer, "__version__", "unknown")
    except ImportError:
        aer_version = "unavailable"
    return {
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "expected_head": expected_head,
        "tracked_dirty": any(line[:2] != "??" for line in status_lines),
        "untracked_nonresults": [
            line[3:].replace("\\", "/")
            for line in status_lines
            if line[:2] == "??" and not line[3:].replace("\\", "/").startswith("results/")
        ],
        "python": sys.version,
        "qiskit": qiskit_version,
        "qiskit_aer": aer_version,
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "workers": int(workers),
    }


def _require_clean() -> None:
    status = _git("status", "--porcelain")
    violations = []
    for line in status.splitlines():
        path = line[3:].replace("\\", "/")
        if line[:2] != "??":
            violations.append(line)
        elif not path.startswith("results/"):
            violations.append(line)
    if violations:
        raise RuntimeError("--require-clean 拒绝已跟踪改动或非 results 未跟踪文件: " + "; ".join(violations))


def default_output_dir(preset: str) -> Path:
    directories = {
        "smoke": Path("results/stage1_closed_loop_baseline_smoke"),
        "pilot": Path("results/stage1_closed_loop_baseline_pilot"),
        "formal": Path("results/stage1_closed_loop_baseline_formal"),
    }
    if preset not in directories:
        raise BatchConfigurationError("custom preset 必须显式提供 --output-dir")
    return directories[preset]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 A 可恢复闭环基线批量 CLI（Windows workers=1）")
    parser.add_argument("--preset", choices=("smoke", "pilot", "formal", "custom"), default="smoke")
    parser.add_argument("--methods", type=_methods, default=METHODS)
    parser.add_argument("--generator-pairs", type=_pairs)
    parser.add_argument("--windows", type=_csv_ints)
    parser.add_argument("--training-seeds", type=_csv_ints)
    parser.add_argument("--run-seeds", type=_csv_ints)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--expected-head")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--fail-fast", dest="continue_on_error", action="store_false")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--train-sample-count", type=int, default=8)
    parser.add_argument("--fractional-bits", type=int, default=2)
    parser.add_argument("--cost-unit", type=float, default=1000.0)
    parser.add_argument("--initialization-policy", choices=("first", "random", "best-training"), default="first")
    parser.add_argument(
        "--initial-incumbent-policy",
        choices=("first_training", "best_training"),
        default="first_training",
    )
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--lambda-factor", type=float, default=1.2)
    parser.add_argument("--max-trials", type=int, default=64)
    parser.add_argument("--max-oracle-calls", type=int, default=128)
    parser.add_argument("--max-new-ed-lp-calls", type=int, default=16)
    parser.add_argument("--max-threshold-updates", type=int, default=8)
    parser.add_argument("--persist-dynamic-oracle-metadata", action="store_true")
    return parser


def effective_initial_incumbent_policy(args: argparse.Namespace) -> str:
    """Resolve the new policy while retaining historical CLI spellings."""

    legacy_policy = str(args.initialization_policy)
    if legacy_policy == "best-training":
        return "best_training"
    if legacy_policy == "random":
        return "random"
    return str(args.initial_incumbent_policy)


def _selection(args: argparse.Namespace) -> dict[str, Sequence[object]]:
    selected = dict(preset_selection(args.preset))
    if args.preset == "custom":
        required = (args.generator_pairs, args.windows, args.training_seeds, args.run_seeds)
        if any(value is None for value in required):
            raise BatchConfigurationError("custom preset 必须提供 pair/window/training-seeds/run-seeds")
    for key, value in (("generator_pairs", args.generator_pairs), ("windows", args.windows), ("training_seeds", args.training_seeds), ("run_seeds", args.run_seeds)):
        if value is not None:
            selected[key] = value
    return selected


def main() -> int:
    args = build_argument_parser().parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = default_output_dir(args.preset)
    if args.summarize_only:
        print(json.dumps(summarize_batch(output_dir), ensure_ascii=False, indent=2))
        return 0
    if args.workers != 1:
        raise BatchConfigurationError("当前版本只支持 --workers 1，不传递 ClosedLoopScenario 给 ProcessPool")
    if args.require_clean:
        _require_clean()
    head = _git("rev-parse", "HEAD")
    expected_head = args.expected_head or head
    if head != expected_head:
        raise RuntimeError(f"当前 HEAD {head} 与 --expected-head {expected_head} 不一致")
    selected = _selection(args)
    initial_incumbent_policy = effective_initial_incumbent_policy(args)
    budget = {
        "max_trials": args.max_trials, "max_oracle_calls": args.max_oracle_calls,
        "max_new_ed_lp_calls": args.max_new_ed_lp_calls, "max_threshold_updates": args.max_threshold_updates,
        "lambda_factor": args.lambda_factor,
    }
    fixed_point = {"fractional_bits": args.fractional_bits, "cost_unit": args.cost_unit, "rounding": "nearest"}
    batch_id = args.batch_id or f"case14-{args.preset}-{head[:12]}"
    specs = build_run_specs(
        batch_id=batch_id, preset=args.preset, methods=args.methods, budget_config=budget,
        fixed_point_config=fixed_point, initialization_policy=initial_incumbent_policy,
        expected_code_sha=head, **selected,
    )
    plan = {
        "batch_id": batch_id, "output_dir": str(output_dir), "runs": len(specs),
        "mps_runs": sum(spec.method in MPS_METHODS for spec in specs), "methods": list(args.methods),
        "budget": budget, "fixed_point": fixed_point, "initialization_policy": initial_incumbent_policy,
        "persist_dynamic_oracle_metadata": bool(args.persist_dynamic_oracle_metadata),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    source = load_uc_instance(args.instance)
    fixed_config = FixedPointConfig(fractional_bits=args.fractional_bits, unit=args.cost_unit, rounding="nearest")
    template = BBHTConfig(
        lambda_factor=args.lambda_factor, max_trials=args.max_trials, max_oracle_calls=args.max_oracle_calls,
        max_new_ed_lp_calls=args.max_new_ed_lp_calls, max_threshold_updates=args.max_threshold_updates, seed=0,
    )

    def scenario_builder(spec):
        return build_case14_closed_loop_scenario(
            source=source, window_start=spec.window_start, selected_generator_indices=spec.generator_pair,
            train_sample_count=args.train_sample_count, fixed_point=fixed_config,
            initialization_policy=initial_incumbent_policy, seed=spec.training_seed,
            regularization=args.regularization, maxiter=args.maxiter, bbht_config=template,
        )

    def method_runner(scenario, method: str, run_seed: int):
        return run_closed_loop_method(
            scenario,
            method,
            run_seed=run_seed,
            persist_dynamic_oracle_metadata=bool(args.persist_dynamic_oracle_metadata),
        )

    def progress(row):
        done = row["completed"] + row["skipped"] + row["failed"]
        print(f"[{done}/{row['total']}] run_id={row['run_id']} method={row['method']} completed={row['completed']} skipped={row['skipped']} failed={row['failed']} remaining={row['pending']}", flush=True)

    executor = ClosedLoopBatchExecutor(
        output_dir=output_dir, code=_environment(expected_head, args.workers),
        scenario_builder=scenario_builder, method_runner=method_runner, workers=args.workers,
    )
    print(json.dumps({"batch_id": batch_id, "budget": budget, "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)
    try:
        counts = executor.execute(specs, resume=args.resume, skip_failed=args.skip_failed, continue_on_error=args.continue_on_error, progress=progress)
    except KeyboardInterrupt:
        print("收到 Ctrl+C；已完成 JSON 保留，manifest 已标记 interrupted，可用 --resume 继续。", file=sys.stderr)
        return 130
    print(json.dumps({**plan, "counts": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
