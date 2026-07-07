from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .commitment import commitment_to_bitstring, is_logic_feasible
from .ed import FixedCommitmentEvaluator
from .uc_loader import Reserve, UCInstance


def leading_time_window_instance(instance: UCInstance, horizon: int) -> UCInstance:
    if horizon <= 0 or horizon > instance.time_horizon:
        raise ValueError("horizon must be within the source instance time horizon")
    reserves = [
        Reserve(
            name=reserve.name,
            amount=reserve.amount[:horizon],
            penalty=reserve.penalty[:horizon],
        )
        for reserve in instance.reserves
    ]
    return replace(
        instance,
        time_horizon=horizon,
        fixed_load=instance.fixed_load[:horizon],
        reserves=reserves,
        power_balance_penalty=instance.power_balance_penalty[:horizon],
    )


def evaluate_values(instance: UCInstance, commitments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    evaluator = FixedCommitmentEvaluator(instance)
    values = np.full(commitments.shape[0], np.inf, dtype=float)
    logic_feasible = np.zeros(commitments.shape[0], dtype=bool)
    for idx, commitment in enumerate(commitments):
        if not is_logic_feasible(instance, commitment):
            continue
        logic_feasible[idx] = True
        result = evaluator.evaluate(commitment)
        if result.success:
            values[idx] = result.total_cost
    return values, logic_feasible


def embedded_selected_commitments(
    base_commitment: np.ndarray,
    selected_generator_indices: tuple[int, ...],
) -> np.ndarray:
    base = np.asarray(base_commitment, dtype=int)
    horizon = base.shape[1]
    num_selected_bits = len(selected_generator_indices) * horizon
    rows = []
    for state_index in range(2**num_selected_bits):
        commitment = base.copy()
        for local_generator_index, generator_index in enumerate(selected_generator_indices):
            for time_index in range(horizon):
                bit_index = local_generator_index * horizon + time_index
                commitment[generator_index, time_index] = (state_index >> bit_index) & 1
        rows.append(commitment)
    return np.array(rows, dtype=int)


def commitment_row(
    commitments: np.ndarray,
    generator_names: list[str],
    values: np.ndarray,
    idx: int,
) -> dict[str, object]:
    commitment = commitments[int(idx)]
    return {
        "bitstring_generator_major": commitment_to_bitstring(commitment),
        "bitstring_time_major": "".join(str(int(v)) for v in commitment.T.reshape(-1)),
        "schedule_table": commitment_schedule_table(commitment, generator_names),
        "time_slices": commitment_time_slices(commitment, generator_names),
        "total_cost": float(values[int(idx)]),
    }


def commitment_schedule_table(
    commitment: np.ndarray,
    generator_names: list[str],
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for g_idx, name in enumerate(generator_names):
        row: dict[str, int | str] = {"generator": name}
        for t in range(commitment.shape[1]):
            row[f"t{t}"] = int(commitment[g_idx, t])
        rows.append(row)
    return rows


def commitment_time_slices(
    commitment: np.ndarray,
    generator_names: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for t in range(commitment.shape[1]):
        status_by_generator = {
            name: int(commitment[g_idx, t])
            for g_idx, name in enumerate(generator_names)
        }
        rows.append(
            {
                "time": f"t{t}",
                "status_by_generator": status_by_generator,
                "online_generators": [
                    name
                    for name, status in status_by_generator.items()
                    if status == 1
                ],
            }
        )
    return rows


def parse_indices(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if np.isfinite(value):
        return value
    return None


def sanitize_for_strict_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_for_strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_strict_json(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return finite_or_none(float(value))
    if isinstance(value, float):
        return finite_or_none(value)
    return value


def write_strict_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_for_strict_json(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
