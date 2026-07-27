from __future__ import annotations

from experiments.stage1_best_training_smoke_cli import (
    SMOKE_SCENARIOS,
    build_argument_parser,
)


def test_best_training_smoke_cli_defaults_to_a_new_output_directory_and_small_budget() -> None:
    args = build_argument_parser().parse_args([])
    assert str(args.output_dir).replace("\\", "/") == "results/stage1_best_training_smoke"
    assert args.run_seed == 0
    assert args.max_trials == 3
    assert args.max_oracle_calls == 4
    assert args.max_new_ed_lp_calls == 2
    assert args.max_threshold_updates == 1


def test_best_training_smoke_uses_distinct_nonbest_best_and_marked_empty_scenarios() -> None:
    assert SMOKE_SCENARIOS == (
        ("first_training_not_best", (0, 5), 2, 0),
        ("first_training_best", (1, 3), 1, 0),
        ("marked_empty", (1, 3), 0, 0),
    )
