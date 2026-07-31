"""Pure fit/validation selection primitives for Stage B QNN calibration."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Mapping, Sequence


ITERATION_BUDGETS = (10, 20, 40, 80)
MODEL_SEEDS = (11, 23, 47)
QNN_MODEL_ORDER = (
    "simplified_expectation_qnn",
    "full_expectation_qnn",
    "threshold_conditioned_qnn",
)


def calibration_grid() -> tuple[tuple[str, int, int], ...]:
    """Return the frozen 3 x 4 x 3 fit/validation-only optimization grid."""

    return tuple(
        (model, budget, seed)
        for model in QNN_MODEL_ORDER
        for budget in ITERATION_BUDGETS
        for seed in MODEL_SEEDS
    )


def strict_json_dumps(payload: object) -> str:
    """Serialize calibration metadata without emitting non-standard JSON values."""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _finite_mean(records: Sequence[Mapping[str, object]], field: str) -> float:
    values = [float(record[field]) for record in records]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field} must be finite for every calibration record")
    return float(sum(values) / len(values))


def _as_bool(value: object, *, default: bool = False) -> bool:
    """Read bool values consistently from JSON records and CSV re-summaries."""

    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _by_budget(records: Sequence[Mapping[str, object]], *, model: str) -> dict[int, list[Mapping[str, object]]]:
    selected = [record for record in records if str(record["model"]) == model]
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for record in selected:
        grouped[int(record["maxiter"])].append(record)
    if not grouped:
        raise ValueError(f"no calibration records for {model}")
    for budget, group in grouped.items():
        seeds = tuple(sorted(int(record["seed"]) for record in group))
        if seeds != MODEL_SEEDS:
            raise ValueError(f"budget {budget} for {model} must contain exactly MODEL_SEEDS")
    return grouped


def _metric_fields(model: str) -> tuple[str, ...]:
    return (
        ("validation_f1", "validation_recall", "validation_precision")
        if model == "threshold_conditioned_qnn"
        else ("validation_mae", "validation_regret")
    )


def _selection_key(summary: Mapping[str, object], *, model: str) -> tuple[float | int, ...]:
    if model == "threshold_conditioned_qnn":
        return (
            -float(summary["macro_validation_f1"]),
            -float(summary["macro_validation_recall"]),
            -float(summary["macro_validation_precision"]),
            int(summary["maxiter"]),
            int(summary["candidate_order"]),
        )
    return (
        float(summary["macro_validation_mae"]),
        float(summary["macro_validation_regret"]),
        int(summary["maxiter"]),
        int(summary["candidate_order"]),
    )


def _budget_summary(group: Sequence[Mapping[str, object]], *, budget: int, order: int, model: str) -> dict[str, object]:
    """Produce finite diagnostics and an explicit convergence eligibility verdict."""

    reasons: list[str] = []
    if any(not _as_bool(record.get("completed"), default=True) for record in group):
        reasons.append("incomplete_seed")
    if any(str(record.get("fit_status", "")) == "optimizer_failed" for record in group):
        reasons.append("optimizer_failed")
    if any(not _as_bool(record.get("converged"), default=False) for record in group):
        reasons.append("not_all_converged")
    if any(_as_bool(record.get("has_nonfinite"), default=False) for record in group):
        reasons.append("nonfinite_output")

    summary: dict[str, object] = {
        "maxiter": int(budget),
        "candidate_order": order,
        "seed_count": len(group),
        "converged_seed_count": sum(_as_bool(record.get("converged"), default=False) for record in group),
        "ineligibility_reasons": reasons,
    }
    try:
        if model == "threshold_conditioned_qnn":
            summary.update(
                macro_validation_f1=_finite_mean(group, "validation_f1"),
                macro_validation_recall=_finite_mean(group, "validation_recall"),
                macro_validation_precision=_finite_mean(group, "validation_precision"),
            )
        else:
            summary.update(
                macro_validation_mae=_finite_mean(group, "validation_mae"),
                macro_validation_regret=_finite_mean(group, "validation_regret"),
            )
    except (KeyError, TypeError, ValueError):
        reasons.append("nonfinite_or_uncomputable_metrics")
        for field in _metric_fields(model):
            summary[f"macro_{field}"] = None
    summary["eligible"] = not reasons
    return summary


def _select_summary(summaries: Sequence[Mapping[str, object]], *, model: str) -> dict[str, object] | None:
    computable = [summary for summary in summaries if all(summary.get(f"macro_{field}") is not None for field in _metric_fields(model))]
    return None if not computable else dict(min(computable, key=lambda item: _selection_key(item, model=model)))


def select_frozen_config(records: Sequence[Mapping[str, object]], *, model: str) -> dict[str, object]:
    """Select only convergence-qualified QNN budgets from validation records.

    Metric-only rankings are retained as historical diagnostics.  They never
    override eligibility: all three fixed seeds must complete with finite,
    computable outputs, no optimizer failure, and convergence for every seed.
    Test rows are deliberately not accepted by this interface.
    """

    grouped = _by_budget(records, model=model)
    summaries = [
        _budget_summary(group, budget=budget, order=order, model=model)
        for order, budget in enumerate(ITERATION_BUDGETS)
        if (group := grouped.get(budget)) is not None
    ]
    metric_only = _select_summary(summaries, model=model)
    eligible = [summary for summary in summaries if bool(summary["eligible"])]
    qualified = _select_summary(eligible, model=model)
    highest = max(summaries, key=lambda summary: int(summary["maxiter"]))
    result: dict[str, object] = {
        "model": model,
        "metric_only_selection": metric_only,
        "convergence_qualified_selection": qualified,
        "selection_status": "convergence_qualified" if qualified is not None else "not_convergence_qualified",
        "per_budget": summaries,
        "highest_budget_diagnostic": dict(highest),
    }
    if qualified is not None:
        result.update(qualified)
    return result
