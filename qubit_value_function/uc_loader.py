from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Generator:
    name: str
    bus: str
    cost_mw: list[float]
    cost_usd: list[float]
    startup_delays: list[int]
    startup_costs: list[float]
    initial_status: int
    initial_power: float
    min_uptime: int
    min_downtime: int
    ramp_up: float | None
    ramp_down: float | None
    must_run: bool
    reserve_eligibility: tuple[str, ...]

    @property
    def p_min(self) -> float:
        return float(self.cost_mw[0])

    @property
    def p_max(self) -> float:
        return float(self.cost_mw[-1])


@dataclass(frozen=True)
class Reserve:
    name: str
    amount: list[float]
    penalty: list[float]


@dataclass(frozen=True)
class UCInstance:
    path: Path
    version: str
    time_horizon: int
    generators: list[Generator]
    fixed_load: list[float]
    reserves: list[Reserve]
    power_balance_penalty: list[float]


def _read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _time_series(value: Any, horizon: int, default: float | None = None) -> list[float]:
    if value is None:
        if default is None:
            raise ValueError("missing required time-series value")
        value = default
    if isinstance(value, list):
        if len(value) == horizon and all(not isinstance(item, list) for item in value):
            return [float(item) for item in value]
        if len(value) == 1:
            return [float(value[0])] * horizon
        raise ValueError(f"unsupported nested or mismatched time series: {value!r}")
    return [float(value)] * horizon


def load_uc_instance(path: str | Path) -> UCInstance:
    path = Path(path)
    raw = _read_json(path)
    params = raw.get("Parameters", {})
    horizon = int(params.get("Time horizon (h)", params.get("Time (h)")))
    version = str(params.get("Version", "unknown"))
    balance_penalty = _time_series(
        params.get("Power balance penalty ($/MW)", 1000.0),
        horizon,
    )

    fixed_load = [0.0] * horizon
    for bus_data in raw.get("Buses", {}).values():
        for t, value in enumerate(_time_series(bus_data.get("Load (MW)", 0.0), horizon)):
            fixed_load[t] += value

    generators: list[Generator] = []
    for name, gen_data in raw.get("Generators", {}).items():
        cost_mw = gen_data["Production cost curve (MW)"]
        cost_usd = gen_data["Production cost curve ($)"]
        if any(isinstance(item, list) for item in cost_mw + cost_usd):
            raise ValueError(
                f"{name}: time-dependent or nested cost curves are not supported in stage 1"
            )
        generators.append(
            Generator(
                name=name,
                bus=gen_data["Bus"],
                cost_mw=[float(v) for v in cost_mw],
                cost_usd=[float(v) for v in cost_usd],
                startup_delays=[int(v) for v in gen_data.get("Startup delays (h)", [1])],
                startup_costs=[float(v) for v in gen_data.get("Startup costs ($)", [0.0])],
                initial_status=int(gen_data.get("Initial status (h)", 0)),
                initial_power=float(gen_data.get("Initial power (MW)", 0.0)),
                min_uptime=int(gen_data.get("Minimum uptime (h)", 1)),
                min_downtime=int(gen_data.get("Minimum downtime (h)", 1)),
                ramp_up=_optional_float(gen_data.get("Ramp up limit (MW)")),
                ramp_down=_optional_float(gen_data.get("Ramp down limit (MW)")),
                must_run=bool(gen_data.get("Must run?", False)),
                reserve_eligibility=tuple(gen_data.get("Reserve eligibility", ())),
            )
        )

    reserves: list[Reserve] = []
    for name, reserve_data in raw.get("Reserves", {}).items():
        reserves.append(
            Reserve(
                name=name,
                amount=_time_series(
                    reserve_data.get("Amount (MW)", reserve_data.get("Spinning (MW)", 0.0)),
                    horizon,
                ),
                penalty=_time_series(
                    reserve_data.get("Shortfall penalty ($/MW)", 1000.0),
                    horizon,
                ),
            )
        )

    return UCInstance(
        path=path,
        version=version,
        time_horizon=horizon,
        generators=generators,
        fixed_load=fixed_load,
        reserves=reserves,
        power_balance_penalty=balance_penalty,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
