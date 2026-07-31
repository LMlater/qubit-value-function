"""Pure integrity primitives for the Stage B supplemental ED/LP audit."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence


FORMAL_KEY_FIELDS = ("generator_pair", "window_start", "load_multiplier", "state_index")
ABSOLUTE_TOLERANCE = 1e-6
RELATIVE_TOLERANCE = 1e-9


def strict_json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def formal_key(row: Mapping[str, object]) -> tuple[str, int, float, int]:
    return (str(row["generator_pair"]), int(row["window_start"]), float(row["load_multiplier"]), int(row["state_index"]))


def validate_formal_truth_rows(rows: Sequence[Mapping[str, object]], *, expected_count: int = 1008) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"formal_truth_row_count:{len(rows)}")
    keys = [formal_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate_formal_key")
    if expected_count == 1008:
        pairs = {key[0] for key in keys}; windows = {key[1] for key in keys}; loads = {key[2] for key in keys}; states = {key[3] for key in keys}
        if pairs != {"[0, 1]", "[0, 5]", "[1, 5]"} or windows != {0, 1, 2} or loads != {0.8, 0.85, 0.925, 1.0, 1.075, 1.15, 1.2} or states != set(range(16)):
            raise ValueError("formal_truth_protocol_scope_mismatch")


def validate_audit_paths(formal_benchmark_dir: Path, output_dir: Path) -> None:
    formal = formal_benchmark_dir.resolve(); output = output_dir.resolve()
    if output == formal or formal in output.parents:
        raise ValueError("formal_benchmark_read_only")


def component_cost_sum(*, dispatch_cost: float, startup_cost: float, balance_penalty: float, reserve_penalty: float) -> float:
    values = (dispatch_cost, startup_cost, balance_penalty, reserve_penalty)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("component_costs_must_be_finite")
    return float(sum(float(value) for value in values))


def within_cost_tolerance(*, formal_true_cost: float, recomputed_total_cost: float) -> bool:
    return abs(float(recomputed_total_cost) - float(formal_true_cost)) <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(float(formal_true_cost))


def cost_consistency_record(formal_row: Mapping[str, object], component_row: Mapping[str, object]) -> dict[str, object]:
    formal = float(formal_row["true_cost"]); recomputed = component_row.get("recomputed_total_cost")
    if recomputed is None:
        return {**{field: formal_row[field] for field in FORMAL_KEY_FIELDS}, "formal_true_cost": formal, "recomputed_total_cost": None, "absolute_difference": None, "relative_difference": None, "within_tolerance": False}
    difference = abs(float(recomputed) - formal)
    return {**{field: formal_row[field] for field in FORMAL_KEY_FIELDS}, "formal_true_cost": formal, "recomputed_total_cost": float(recomputed), "absolute_difference": difference, "relative_difference": difference / max(abs(formal), 1.0), "within_tolerance": within_cost_tolerance(formal_true_cost=formal, recomputed_total_cost=float(recomputed))}


def require_hard_logic_agreement(rows: Sequence[Mapping[str, object]]) -> None:
    mismatches = [row for row in rows if bool(row["hard_logic_feasible"]) != bool(row["independent_hard_logic_feasible"])]
    if mismatches:
        raise RuntimeError(f"hard_logic_disagreement:{len(mismatches)}")


def join_by_formal_key(formal_rows: Sequence[Mapping[str, object]], supplemental_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    supplemental = {formal_key(row): row for row in supplemental_rows}
    if len(supplemental) != len(supplemental_rows):
        raise ValueError("duplicate_supplemental_key")
    missing = [formal_key(row) for row in formal_rows if formal_key(row) not in supplemental]
    if missing:
        raise ValueError(f"missing_supplemental_keys:{len(missing)}")
    return [{**dict(row), **{key: value for key, value in supplemental[formal_key(row)].items() if key not in FORMAL_KEY_FIELDS}} for row in formal_rows]
