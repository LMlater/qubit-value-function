from __future__ import annotations

from experiments.stage1_dynamic_oracle_metadata_smoke_cli import build_argument_parser


def test_dynamic_metadata_smoke_cli_uses_new_output_directory_and_small_budgets() -> None:
    args = build_argument_parser().parse_args([])
    assert str(args.output_dir).replace("\\", "/") == "results/stage1_dynamic_oracle_metadata_smoke"
    assert args.max_trials == 3
    assert args.max_oracle_calls == 4
    assert args.max_new_ed_lp_calls == 2
    assert args.max_threshold_updates == 1
    assert args.run_seed == 0
