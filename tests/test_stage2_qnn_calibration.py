from __future__ import annotations

import inspect
import json

from qubit_value_function.stage2_qnn_calibration import (
    ITERATION_BUDGETS,
    MODEL_SEEDS,
    calibration_grid,
    select_frozen_config,
    strict_json_dumps,
)


def _regression_record(
    *,
    budget: int,
    seed: int,
    mae: float,
    regret: float,
    converged: bool = True,
    fit_status: str = "converged",
    has_nonfinite: bool = False,
) -> dict[str, object]:
    return {
        "model": "simplified_expectation_qnn",
        "maxiter": budget,
        "seed": seed,
        "validation_mae": mae,
        "validation_regret": regret,
        "validation_f1": None,
        "validation_recall": None,
        "validation_precision": None,
        "converged": converged,
        "fit_status": fit_status,
        "has_nonfinite": has_nonfinite,
    }


def _threshold_record(
    *,
    budget: int,
    seed: int,
    f1: float,
    recall: float,
    precision: float,
    converged: bool = True,
    fit_status: str = "converged",
    has_nonfinite: bool = False,
) -> dict[str, object]:
    return {
        "model": "threshold_conditioned_qnn",
        "maxiter": budget,
        "seed": seed,
        "validation_mae": None,
        "validation_regret": None,
        "validation_f1": f1,
        "validation_recall": recall,
        "validation_precision": precision,
        "converged": converged,
        "fit_status": fit_status,
        "has_nonfinite": has_nonfinite,
    }


def test_calibration_grid_has_fixed_budget_seed_order_and_thirty_six_runs() -> None:
    grid = calibration_grid()

    assert ITERATION_BUDGETS == (10, 20, 40, 80)
    assert MODEL_SEEDS == (11, 23, 47)
    assert len(grid) == 36
    assert grid[:4] == (
        ("simplified_expectation_qnn", 10, 11),
        ("simplified_expectation_qnn", 10, 23),
        ("simplified_expectation_qnn", 10, 47),
        ("simplified_expectation_qnn", 20, 11),
    )


def test_regression_selector_rejects_better_nonconverged_candidate_and_selects_budget_twenty() -> None:
    records = [
        _regression_record(budget=10, seed=seed, mae=1.0, regret=0.0, converged=False, fit_status="maxiter")
        for seed in MODEL_SEEDS
    ] + [
        _regression_record(budget=20, seed=seed, mae=10.0, regret=0.5) for seed in MODEL_SEEDS
    ] + [
        _regression_record(budget=40, seed=seed, mae=11.0, regret=0.0) for seed in MODEL_SEEDS
    ]

    selected = select_frozen_config(records, model="simplified_expectation_qnn")

    assert selected["selection_status"] == "convergence_qualified"
    assert selected["metric_only_selection"]["maxiter"] == 10
    assert selected["convergence_qualified_selection"]["maxiter"] == 20
    assert selected["maxiter"] == 20
    assert "test" not in inspect.signature(select_frozen_config).parameters
    assert select_frozen_config(records, model="simplified_expectation_qnn") == selected


def test_threshold_selector_selects_budget_forty_after_convergence_qualification() -> None:
    records = [
        _threshold_record(budget=10, seed=seed, f1=1.0, recall=1.0, precision=1.0, converged=False, fit_status="maxiter")
        for seed in MODEL_SEEDS
    ] + [
        _threshold_record(budget=20, seed=seed, f1=0.9, recall=0.9, precision=0.9, converged=False, fit_status="maxiter")
        for seed in MODEL_SEEDS
    ] + [
        _threshold_record(budget=40, seed=seed, f1=0.75, recall=1.0, precision=0.6)
        for seed in MODEL_SEEDS
    ]

    selected = select_frozen_config(records, model="threshold_conditioned_qnn")
    assert selected["metric_only_selection"]["maxiter"] == 10
    assert selected["convergence_qualified_selection"]["maxiter"] == 40


def test_full_readout_selector_selects_budget_twenty_and_keeps_deterministic_tie_break() -> None:
    records = [
        _regression_record(budget=10, seed=seed, mae=1.0, regret=0.0, converged=False, fit_status="maxiter")
        for seed in MODEL_SEEDS
    ] + [
        {
            **_regression_record(budget=20, seed=seed, mae=5.0, regret=2.0),
            "model": "full_expectation_qnn",
        }
        for seed in MODEL_SEEDS
    ] + [
        {
            **_regression_record(budget=40, seed=seed, mae=5.0, regret=2.0),
            "model": "full_expectation_qnn",
        }
        for seed in MODEL_SEEDS
    ]

    selected = select_frozen_config(records, model="full_expectation_qnn")

    assert selected["maxiter"] == 20
    assert selected["convergence_qualified_selection"]["candidate_order"] == 1


def test_optimizer_failure_nonfinite_and_missing_eligibility_return_explicit_block() -> None:
    records = [
        _regression_record(budget=10, seed=11, mae=1.0, regret=0.0, fit_status="optimizer_failed"),
        _regression_record(budget=10, seed=23, mae=1.0, regret=0.0),
        _regression_record(budget=10, seed=47, mae=1.0, regret=0.0),
        _regression_record(budget=20, seed=11, mae=float("inf"), regret=0.0, has_nonfinite=True),
        _regression_record(budget=20, seed=23, mae=1.0, regret=0.0),
        _regression_record(budget=20, seed=47, mae=1.0, regret=0.0),
    ]

    selected = select_frozen_config(records, model="simplified_expectation_qnn")

    assert selected["selection_status"] == "not_convergence_qualified"
    assert selected["convergence_qualified_selection"] is None
    assert selected["highest_budget_diagnostic"]["maxiter"] == 20
    assert "optimizer_failed" in selected["per_budget"][0]["ineligibility_reasons"]
    assert "nonfinite_output" in selected["per_budget"][1]["ineligibility_reasons"]


def test_strict_json_encodes_not_applicable_precision_as_null() -> None:
    payload = strict_json_dumps({"precision": None, "nested": {"value": 1.0}})

    assert json.loads(payload) == {"nested": {"value": 1.0}, "precision": None}
    assert "NaN" not in payload and "Infinity" not in payload
