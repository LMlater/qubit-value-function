from __future__ import annotations

from pathlib import Path

import numpy as np

from qubit_value_function.commitment import is_logic_feasible
from qubit_value_function.logic_feasibility_oracle import compile_logic_feasibility_spec
from qubit_value_function.uc_loader import Generator, UCInstance


def _generator(*, initial_status: int, min_uptime: int, min_downtime: int) -> Generator:
    return Generator(
        name="g0",
        bus="b0",
        cost_mw=[10.0, 100.0],
        cost_usd=[100.0, 1000.0],
        startup_delays=[1],
        startup_costs=[10.0],
        initial_status=initial_status,
        initial_power=0.0,
        min_uptime=min_uptime,
        min_downtime=min_downtime,
        ramp_up=None,
        ramp_down=None,
        must_run=False,
        reserve_eligibility=(),
    )


def _instance(generator: Generator) -> UCInstance:
    return UCInstance(
        path=Path("synthetic.json"),
        version="test",
        time_horizon=3,
        generators=[generator],
        fixed_load=[0.0, 0.0, 0.0],
        reserves=[],
        power_balance_penalty=[1000.0, 1000.0, 1000.0],
    )


def _assert_exact_match(instance: UCInstance) -> None:
    spec = compile_logic_feasibility_spec(
        instance,
        selected_generator_indices=(0,),
        base_commitment=np.zeros((1, 3), dtype=int),
    )
    for index in range(8):
        bits = tuple((index >> bit) & 1 for bit in range(3))
        commitment = np.asarray(bits, dtype=int).reshape((1, 3))
        assert spec.is_feasible(bits) == is_logic_feasible(instance, commitment)


def test_initial_remaining_minimum_downtime_compiles_exactly() -> None:
    _assert_exact_match(
        _instance(_generator(initial_status=-1, min_uptime=2, min_downtime=3))
    )


def test_initial_remaining_minimum_uptime_compiles_exactly() -> None:
    _assert_exact_match(
        _instance(_generator(initial_status=1, min_uptime=3, min_downtime=2))
    )
