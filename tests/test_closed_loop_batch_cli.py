from __future__ import annotations

from experiments.stage1_case14_closed_loop_baseline_cli import (
    build_argument_parser,
    default_output_dir,
)


def test_preset_default_output_directories_are_isolated_and_dry_run_plan_has_budget() -> None:
    parser = build_argument_parser()
    smoke = parser.parse_args(["--preset", "smoke", "--dry-run"])
    pilot = parser.parse_args(["--preset", "pilot", "--dry-run"])
    formal = parser.parse_args(["--preset", "formal", "--dry-run"])
    assert len({default_output_dir(smoke.preset), default_output_dir(pilot.preset), default_output_dir(formal.preset)}) == 3
    assert smoke.max_trials == 64
    assert smoke.max_oracle_calls == 128
    assert smoke.max_new_ed_lp_calls == 16
    assert smoke.max_threshold_updates == 8
