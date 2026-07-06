from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .ed import startup_cost
from .uc_loader import UCInstance


@dataclass(frozen=True)
class StructuredFeatureMatrix:
    features: np.ndarray
    names: list[str]


def structured_commitment_features(
    instance: UCInstance,
    commitments: np.ndarray,
    *,
    same_time_interaction_order: int = 1,
    adjacent_time_interaction_order: int = 1,
    include_dispatch_proxy: bool = False,
) -> StructuredFeatureMatrix:
    """Build reversible-style structured features from commitment bits.

    The features are functions of the commitment register and instance
    constants only. They are not a value-function table: aggregate capacity,
    reserve, startup, local bundle, and optional merit-order proxy terms can be
    computed by reversible arithmetic/control logic before a value-register
    comparator is applied.
    """

    u = np.asarray(commitments, dtype=int)
    if u.ndim != 3:
        raise ValueError("commitments must have shape (states, generators, time)")
    num_states, num_generators, horizon = u.shape
    if num_generators != len(instance.generators) or horizon != instance.time_horizon:
        raise ValueError("commitment shape does not match instance dimensions")
    if same_time_interaction_order < 1:
        raise ValueError("same_time_interaction_order must be at least 1")
    if adjacent_time_interaction_order < 1:
        raise ValueError("adjacent_time_interaction_order must be at least 1")

    gens = instance.generators
    pmin = np.array([gen.p_min for gen in gens], dtype=float)
    pmax = np.array([gen.p_max for gen in gens], dtype=float)
    pmin_cost = np.array([gen.cost_usd[0] for gen in gens], dtype=float)
    pmax_cost = np.array([gen.cost_usd[-1] for gen in gens], dtype=float)
    average_cost = pmax_cost / np.maximum(pmax, 1.0)
    endpoint_slope = (pmax_cost - pmin_cost) / np.maximum(pmax - pmin, 1.0)

    columns: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=float)
        if arr.shape != (num_states,):
            raise ValueError(f"{name} has shape {arr.shape}, expected {(num_states,)}")
        columns.append(arr)
        names.append(name)

    add("1", np.ones(num_states))

    for g_idx, gen in enumerate(gens):
        for t in range(horizon):
            add(f"u_{gen.name}_t{t}", u[:, g_idx, t])

    for t, load in enumerate(instance.fixed_load):
        online = u[:, :, t].astype(float)
        pmax_sum = online @ pmax
        pmin_sum = online @ pmin
        reserve_requirement = sum(reserve.amount[t] for reserve in instance.reserves)
        reserve_capacity = np.zeros(num_states)
        for reserve in instance.reserves:
            eligible_pmax = np.array(
                [
                    gen.p_max if reserve.name in gen.reserve_eligibility else 0.0
                    for gen in gens
                ],
                dtype=float,
            )
            reserve_capacity += online @ eligible_pmax
            add(f"reserve_{reserve.name}_capacity_t{t}", online @ eligible_pmax)
            add(f"reserve_{reserve.name}_margin_t{t}", online @ eligible_pmax - reserve.amount[t])

        add(f"online_count_t{t}", online.sum(axis=1))
        add(f"pmax_capacity_t{t}", pmax_sum)
        add(f"pmin_generation_t{t}", pmin_sum)
        add(f"capacity_margin_t{t}", pmax_sum - load)
        add(f"load_minus_pmin_t{t}", load - pmin_sum)
        add(f"reserve_capacity_t{t}", reserve_capacity)
        add(f"reserve_margin_t{t}", reserve_capacity - reserve_requirement)
        add(f"pmin_cost_sum_t{t}", online @ pmin_cost)
        add(f"pmax_cost_sum_t{t}", online @ pmax_cost)
        add(f"endpoint_slope_sum_t{t}", online @ endpoint_slope)
        add(f"average_cost_sum_t{t}", online @ average_cost)
        add(f"pmax_weighted_slope_t{t}", online @ (pmax * endpoint_slope))

        for prefix, gen_indices in enumerate(_cheap_prefix_indices(gens), start=1):
            mask = np.zeros(num_generators)
            mask[list(gen_indices)] = pmax[list(gen_indices)]
            prefix_capacity = online @ mask
            add(f"cheap_prefix_{prefix}_capacity_t{t}", prefix_capacity)
            add(f"cheap_prefix_{prefix}_margin_t{t}", prefix_capacity - load)

    startup_total = np.array([startup_cost(instance, commitment) for commitment in u], dtype=float)
    add("startup_cost_exact", startup_total)
    startup_count = np.zeros(num_states)
    shutdown_count = np.zeros(num_states)
    for g_idx, gen in enumerate(gens):
        previous = np.full(num_states, 1 if gen.initial_status > 0 else 0, dtype=int)
        for t in range(horizon):
            current = u[:, g_idx, t]
            starts = (1 - previous) * current
            shuts = previous * (1 - current)
            add(f"startup_{gen.name}_t{t}", starts)
            add(f"shutdown_{gen.name}_t{t}", shuts)
            startup_count += starts
            shutdown_count += shuts
            previous = current
    add("startup_count", startup_count)
    add("shutdown_count", shutdown_count)
    add("transition_count", startup_count + shutdown_count)

    if same_time_interaction_order >= 2:
        for t in range(horizon):
            for order in range(2, same_time_interaction_order + 1):
                for combo in combinations(range(num_generators), order):
                    values = np.prod(u[:, combo, t], axis=1)
                    gen_names = "_".join(gens[g_idx].name for g_idx in combo)
                    add(f"same_time_o{order}_{gen_names}_t{t}", values)
    for g_idx, gen in enumerate(gens):
        for t in range(1, horizon):
            add(f"temporal_on_pair_{gen.name}_t{t-1}_t{t}", u[:, g_idx, t - 1] * u[:, g_idx, t])

    if adjacent_time_interaction_order >= 2:
        for t in range(1, horizon):
            specs = [(g_idx, t - 1) for g_idx in range(num_generators)]
            specs += [(g_idx, t) for g_idx in range(num_generators)]
            for order in range(2, adjacent_time_interaction_order + 1):
                for combo in combinations(specs, order):
                    times = {time_idx for _, time_idx in combo}
                    if len(times) < 2:
                        continue
                    values = np.ones(num_states, dtype=int)
                    labels = []
                    for g_idx, time_idx in combo:
                        values *= u[:, g_idx, time_idx]
                        labels.append(f"{gens[g_idx].name}_t{time_idx}")
                    add(f"adjacent_o{order}_{'_'.join(labels)}", values)

    if include_dispatch_proxy:
        proxy = merit_order_dispatch_proxy(instance, u)
        for key, values in proxy.items():
            if values.ndim == 1:
                add(key, values)
            else:
                for t in range(values.shape[1]):
                    add(f"{key}_t{t}", values[:, t])

    return StructuredFeatureMatrix(
        features=_scale_columns(np.column_stack(columns)),
        names=names,
    )


def merit_order_dispatch_proxy(instance: UCInstance, commitments: np.ndarray) -> dict[str, np.ndarray]:
    """Approximate ED by deterministic merit-order dispatch, without LP solves."""

    u = np.asarray(commitments, dtype=int)
    num_states, num_generators, horizon = u.shape
    production = np.zeros((num_states, horizon), dtype=float)
    balance = np.zeros((num_states, horizon), dtype=float)
    reserve = np.zeros((num_states, horizon), dtype=float)
    ramp_violation = np.zeros(num_states, dtype=float)

    for state_idx, commitment in enumerate(u):
        outputs = np.zeros((num_generators, horizon), dtype=float)
        for t, load in enumerate(instance.fixed_load):
            cost, balance_penalty, reserve_penalty, dispatch = _dispatch_one_period(
                instance,
                commitment[:, t],
                t,
                load,
            )
            production[state_idx, t] = cost
            balance[state_idx, t] = balance_penalty
            reserve[state_idx, t] = reserve_penalty
            outputs[:, t] = dispatch
        ramp_violation[state_idx] = _ramp_violation_mw(instance, outputs)

    startup = np.array([startup_cost(instance, commitment) for commitment in u], dtype=float)
    production_total = production.sum(axis=1)
    balance_total = balance.sum(axis=1)
    reserve_total = reserve.sum(axis=1)
    return {
        "merit_production_proxy": production,
        "merit_balance_proxy": balance,
        "merit_reserve_proxy": reserve,
        "merit_production_proxy_total": production_total,
        "merit_balance_proxy_total": balance_total,
        "merit_reserve_proxy_total": reserve_total,
        "merit_dispatch_proxy_total": production_total + balance_total + reserve_total,
        "merit_dispatch_plus_startup_proxy": production_total + balance_total + reserve_total + startup,
        "merit_ramp_violation_mw": ramp_violation,
    }


def _dispatch_one_period(
    instance: UCInstance,
    commitment_t: np.ndarray,
    t: int,
    load: float,
) -> tuple[float, float, float, np.ndarray]:
    gens = instance.generators
    dispatch = np.zeros(len(gens), dtype=float)
    production_cost = 0.0
    for g_idx, gen in enumerate(gens):
        if commitment_t[g_idx]:
            dispatch[g_idx] = gen.p_min
            production_cost += gen.cost_usd[0]

    remaining = float(load - dispatch.sum())
    balance_penalty = 0.0
    if remaining < 0.0:
        balance_penalty += -remaining * instance.power_balance_penalty[t]
        remaining = 0.0

    segments: list[tuple[float, int, float]] = []
    for g_idx, gen in enumerate(gens):
        if not commitment_t[g_idx]:
            continue
        for point in range(1, len(gen.cost_mw)):
            width = float(gen.cost_mw[point] - gen.cost_mw[point - 1])
            if width <= 0.0:
                continue
            slope = float(gen.cost_usd[point] - gen.cost_usd[point - 1]) / width
            segments.append((slope, g_idx, width))
    for slope, g_idx, width in sorted(segments, key=lambda item: item[0]):
        if remaining <= 0.0:
            break
        amount = min(width, remaining)
        dispatch[g_idx] += amount
        production_cost += slope * amount
        remaining -= amount
    if remaining > 0.0:
        balance_penalty += remaining * instance.power_balance_penalty[t]

    reserve_penalty = 0.0
    for reserve in instance.reserves:
        eligible = np.array(
            [
                gen.p_max if reserve.name in gen.reserve_eligibility else 0.0
                for gen in gens
            ],
            dtype=float,
        )
        eligible_output = sum(
            dispatch[g_idx]
            for g_idx, gen in enumerate(gens)
            if reserve.name in gen.reserve_eligibility
        )
        eligible_capacity = float(commitment_t @ eligible)
        headroom = eligible_capacity - eligible_output
        shortfall = max(0.0, float(reserve.amount[t]) - headroom)
        reserve_penalty += shortfall * reserve.penalty[t]

    return production_cost, balance_penalty, reserve_penalty, dispatch


def _ramp_violation_mw(instance: UCInstance, outputs: np.ndarray) -> float:
    total = 0.0
    for g_idx, gen in enumerate(instance.generators):
        previous = gen.initial_power
        for t in range(instance.time_horizon):
            current = outputs[g_idx, t]
            if gen.ramp_up is not None:
                total += max(0.0, current - previous - gen.ramp_up)
            if gen.ramp_down is not None:
                total += max(0.0, previous - current - gen.ramp_down)
            previous = current
    return float(total)


def _cheap_prefix_indices(gens) -> list[tuple[int, ...]]:
    slopes = []
    for idx, gen in enumerate(gens):
        slope = (gen.cost_usd[-1] - gen.cost_usd[0]) / max(gen.p_max - gen.p_min, 1.0)
        slopes.append((float(slope), idx))
    ordered = [idx for _, idx in sorted(slopes)]
    return [tuple(ordered[:count]) for count in range(1, len(ordered))]


def _scale_columns(raw: np.ndarray) -> np.ndarray:
    features = np.asarray(raw, dtype=float).copy()
    for col in range(1, features.shape[1]):
        scale = float(np.max(np.abs(features[:, col])))
        if scale > 0.0:
            features[:, col] /= scale
    return features
