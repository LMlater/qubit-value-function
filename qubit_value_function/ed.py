from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .uc_loader import UCInstance


@dataclass(frozen=True)
class ValueResult:
    total_cost: float
    dispatch_cost: float
    startup_cost: float
    balance_penalty: float
    reserve_penalty: float
    success: bool
    message: str


class FixedCommitmentEvaluator:
    """Evaluate V_d(x) by solving the continuous ED LP for a fixed commitment."""

    def __init__(self, instance: UCInstance):
        self.instance = instance

    def evaluate(self, commitment: np.ndarray) -> ValueResult:
        u = np.asarray(commitment, dtype=int)
        if u.shape != (len(self.instance.generators), self.instance.time_horizon):
            raise ValueError("commitment shape does not match the instance")

        problem = _LpBuilder(self.instance, u)
        result = linprog(
            c=problem.objective,
            A_ub=problem.a_ub if problem.a_ub else None,
            b_ub=problem.b_ub if problem.b_ub else None,
            A_eq=problem.a_eq,
            b_eq=problem.b_eq,
            bounds=problem.bounds,
            method="highs",
        )
        startup = startup_cost(self.instance, u)
        if not result.success:
            return ValueResult(
                total_cost=float("inf"),
                dispatch_cost=float("inf"),
                startup_cost=startup,
                balance_penalty=float("inf"),
                reserve_penalty=float("inf"),
                success=False,
                message=result.message,
            )

        x = result.x
        dispatch_cost = float(problem.production_cost @ x)
        balance_penalty = float(problem.balance_penalty_cost @ x)
        reserve_penalty = float(problem.reserve_penalty_cost @ x)
        total = dispatch_cost + balance_penalty + reserve_penalty + startup
        return ValueResult(
            total_cost=float(total),
            dispatch_cost=dispatch_cost,
            startup_cost=startup,
            balance_penalty=balance_penalty,
            reserve_penalty=reserve_penalty,
            success=True,
            message=result.message,
        )


class _LpBuilder:
    def __init__(self, instance: UCInstance, commitment: np.ndarray):
        self.instance = instance
        self.commitment = commitment
        self.index: dict[tuple, int] = {}
        self.bounds: list[tuple[float, float | None]] = []
        self._build_indices()
        self.objective = np.zeros(len(self.bounds))
        self.production_cost = np.zeros(len(self.bounds))
        self.balance_penalty_cost = np.zeros(len(self.bounds))
        self.reserve_penalty_cost = np.zeros(len(self.bounds))
        self.a_eq: list[list[float]] = []
        self.b_eq: list[float] = []
        self.a_ub: list[list[float]] = []
        self.b_ub: list[float] = []
        self._build_objective()
        self._build_lambda_equalities()
        self._build_balance_equalities()
        self._build_reserve_constraints()
        self._build_ramp_constraints()

    def _add_var(self, key: tuple, lb: float = 0.0, ub: float | None = None) -> None:
        self.index[key] = len(self.bounds)
        self.bounds.append((lb, ub))

    def _build_indices(self) -> None:
        for g_idx, gen in enumerate(self.instance.generators):
            for t in range(self.instance.time_horizon):
                for point in range(len(gen.cost_mw)):
                    self._add_var(("lambda", g_idx, t, point), 0.0, 1.0)
        for t in range(self.instance.time_horizon):
            self._add_var(("shortage", t), 0.0, None)
            self._add_var(("surplus", t), 0.0, None)
            for reserve in self.instance.reserves:
                self._add_var(("reserve_shortfall", reserve.name, t), 0.0, None)

    def _build_objective(self) -> None:
        for g_idx, gen in enumerate(self.instance.generators):
            for t in range(self.instance.time_horizon):
                for point, cost in enumerate(gen.cost_usd):
                    idx = self.index[("lambda", g_idx, t, point)]
                    self.objective[idx] = cost
                    self.production_cost[idx] = cost
        for t, penalty in enumerate(self.instance.power_balance_penalty):
            for name in ("shortage", "surplus"):
                idx = self.index[(name, t)]
                self.objective[idx] = penalty
                self.balance_penalty_cost[idx] = penalty
            for reserve in self.instance.reserves:
                idx = self.index[("reserve_shortfall", reserve.name, t)]
                r_penalty = reserve.penalty[t]
                self.objective[idx] = r_penalty
                self.reserve_penalty_cost[idx] = r_penalty

    def _empty_row(self) -> list[float]:
        return [0.0] * len(self.bounds)

    def _build_lambda_equalities(self) -> None:
        for g_idx, gen in enumerate(self.instance.generators):
            for t in range(self.instance.time_horizon):
                row = self._empty_row()
                for point in range(len(gen.cost_mw)):
                    row[self.index[("lambda", g_idx, t, point)]] = 1.0
                self.a_eq.append(row)
                self.b_eq.append(float(self.commitment[g_idx, t]))

    def _build_balance_equalities(self) -> None:
        for t, load in enumerate(self.instance.fixed_load):
            row = self._empty_row()
            for g_idx, gen in enumerate(self.instance.generators):
                for point, mw in enumerate(gen.cost_mw):
                    row[self.index[("lambda", g_idx, t, point)]] = mw
            row[self.index[("shortage", t)]] = 1.0
            row[self.index[("surplus", t)]] = -1.0
            self.a_eq.append(row)
            self.b_eq.append(float(load))

    def _build_reserve_constraints(self) -> None:
        for reserve in self.instance.reserves:
            for t, amount in enumerate(reserve.amount):
                row = self._empty_row()
                available_capacity = 0.0
                for g_idx, gen in enumerate(self.instance.generators):
                    if reserve.name in gen.reserve_eligibility:
                        available_capacity += gen.p_max * self.commitment[g_idx, t]
                        for point, mw in enumerate(gen.cost_mw):
                            row[self.index[("lambda", g_idx, t, point)]] = mw
                row[self.index[("reserve_shortfall", reserve.name, t)]] = -1.0
                self.a_ub.append(row)
                self.b_ub.append(available_capacity - amount)

    def _build_ramp_constraints(self) -> None:
        for g_idx, gen in enumerate(self.instance.generators):
            for t in range(self.instance.time_horizon):
                if gen.ramp_up is not None:
                    row = self._empty_row()
                    self._add_power(row, g_idx, t, 1.0)
                    if t == 0:
                        rhs = gen.ramp_up + gen.initial_power
                    else:
                        self._add_power(row, g_idx, t - 1, -1.0)
                        rhs = gen.ramp_up
                    self.a_ub.append(row)
                    self.b_ub.append(rhs)
                if gen.ramp_down is not None:
                    row = self._empty_row()
                    self._add_power(row, g_idx, t, -1.0)
                    if t == 0:
                        rhs = gen.ramp_down - gen.initial_power
                    else:
                        self._add_power(row, g_idx, t - 1, 1.0)
                        rhs = gen.ramp_down
                    self.a_ub.append(row)
                    self.b_ub.append(rhs)

    def _add_power(self, row: list[float], g_idx: int, t: int, coefficient: float) -> None:
        gen = self.instance.generators[g_idx]
        for point, mw in enumerate(gen.cost_mw):
            row[self.index[("lambda", g_idx, t, point)]] += coefficient * mw


def startup_cost(instance: UCInstance, commitment: np.ndarray) -> float:
    total = 0.0
    for g_idx, gen in enumerate(instance.generators):
        status = commitment[g_idx]
        previous_on = gen.initial_status > 0
        prior_off_duration = max(-gen.initial_status, 0)
        for t, current in enumerate(status):
            current_on = bool(current)
            if current_on and not previous_on:
                downtime = prior_off_duration
                for back in range(t - 1, -1, -1):
                    if status[back] == 0:
                        downtime += 1
                    else:
                        break
                total += _startup_cost_for_downtime(gen.startup_delays, gen.startup_costs, downtime)
            previous_on = current_on
    return float(total)


def _startup_cost_for_downtime(delays: list[int], costs: list[float], downtime: int) -> float:
    selected = costs[0]
    for delay, cost in sorted(zip(delays, costs), key=lambda item: item[0]):
        if downtime >= delay:
            selected = cost
    return float(selected)
