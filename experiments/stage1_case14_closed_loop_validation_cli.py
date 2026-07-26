"""对既有阶段 A 批量结果执行严格搜索后 validation-only 后处理。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_2x2_joint_feasible_sparse_vqc_bbht import build_case14_closed_loop_scenario  # noqa: E402
from qubit_value_function.closed_loop_validation import summarize_validated, validate_source_batch  # noqa: E402
from qubit_value_function.fixed_point_oracle import FixedPointConfig  # noqa: E402
from qubit_value_function.sparse_vqc_bbht import BBHTConfig  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def _require_clean() -> None:
    violations = [line for line in _git("status", "--porcelain").splitlines() if line[:2] != "??" or not line[3:].replace("\\", "/").startswith("results/")]
    if violations:
        raise RuntimeError("--require-clean 拒绝已跟踪改动或非 results 未跟踪文件: " + "; ".join(violations))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 A smoke 只读 validation-only 后处理")
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--expected-search-head")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-validated-only", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument("--train-sample-count", type=int, default=8)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--abs-tol", type=float, default=1e-6)
    parser.add_argument("--rel-tol", type=float, default=1e-9)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.require_clean:
        _require_clean()
    if args.summarize_validated_only:
        print(json.dumps(summarize_validated(args.source_output_dir), ensure_ascii=False, indent=2))
        return 0
    manifest = json.loads((args.source_output_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    budget = manifest["budget_config"]
    fixed = manifest["fixed_point_config"]
    source = load_uc_instance(args.instance)
    fixed_point = FixedPointConfig(fractional_bits=int(fixed["fractional_bits"]), unit=float(fixed["cost_unit"]), rounding=str(fixed["rounding"]))
    template = BBHTConfig(
        lambda_factor=float(budget["lambda_factor"]), max_trials=int(budget["max_trials"]),
        max_oracle_calls=int(budget["max_oracle_calls"]), max_new_ed_lp_calls=int(budget["max_new_ed_lp_calls"]),
        max_threshold_updates=int(budget["max_threshold_updates"]), seed=0,
    )
    def scenario_builder(spec):
        return build_case14_closed_loop_scenario(
            source=source, window_start=int(spec["window_start"]), selected_generator_indices=tuple(spec["generator_pair"]),
            train_sample_count=args.train_sample_count, fixed_point=fixed_point,
            initialization_policy=str(spec["initialization_policy"]), seed=int(spec["training_seed"]),
            regularization=args.regularization, maxiter=args.maxiter, bbht_config=template,
        )
    result = validate_source_batch(
        args.source_output_dir, scenario_builder=scenario_builder, validation_code_sha=_git("rev-parse", "HEAD"),
        expected_search_head=args.expected_search_head, resume=args.resume, abs_tol=args.abs_tol, rel_tol=args.rel_tol,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
