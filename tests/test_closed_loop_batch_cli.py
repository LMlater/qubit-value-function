from __future__ import annotations

from experiments.stage1_case14_closed_loop_baseline_cli import (
    effective_initial_incumbent_policy,
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


def test_dynamic_oracle_metadata_recording_flag_is_opt_in() -> None:
    parser = build_argument_parser()
    assert parser.parse_args(["--preset", "custom", "--generator-pairs", "0,5", "--windows", "2", "--training-seeds", "0", "--run-seeds", "0"]).persist_dynamic_oracle_metadata is False
    assert parser.parse_args(["--preset", "custom", "--generator-pairs", "0,5", "--windows", "2", "--training-seeds", "0", "--run-seeds", "0", "--persist-dynamic-oracle-metadata"]).persist_dynamic_oracle_metadata is True


def test_initial_incumbent_policy_cli_defaults_to_first_training_and_rejects_unknown_values() -> None:
    parser = build_argument_parser()
    assert parser.parse_args([]).initial_incumbent_policy == "first_training"
    assert parser.parse_args(["--initial-incumbent-policy", "best_training"]).initial_incumbent_policy == "best_training"
    try:
        parser.parse_args(["--initial-incumbent-policy", "random"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("CLI must reject an unsupported initial incumbent policy")


def test_legacy_initialization_policy_remains_usable_when_new_policy_is_default() -> None:
    parser = build_argument_parser()
    assert effective_initial_incumbent_policy(parser.parse_args([])) == "first_training"
    assert effective_initial_incumbent_policy(
        parser.parse_args(["--initialization-policy", "best-training"])
    ) == "best_training"
    assert effective_initial_incumbent_policy(
        parser.parse_args(["--initialization-policy", "random"])
    ) == "random"
    assert effective_initial_incumbent_policy(
        parser.parse_args(["--initial-incumbent-policy", "best_training"])
    ) == "best_training"
