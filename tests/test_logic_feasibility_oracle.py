from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from qubit_value_function.commitment import is_logic_feasible
from qubit_value_function.logic_feasibility_oracle import (
    compile_logic_feasibility_spec,
    simulate_logic_feasibility_phase_oracle,
)
from qubit_value_function.uc_loader import Generator, UCInstance


def _generator(*, must_run: bool = False) -> Generator:
    return Generator(
        name="g0",
        bus="b0",
        cost_mw=[10.0, 100.0],
        cost_usd=[100.0, 1000.0],
        startup_delays=[1],
        startup_costs=[10.0],
        initial_status=0,
        initial_power=0.0,
        min_uptime=2,
        min_downtime=2,
        ramp_up=None,
        ramp_down=None,
        must_run=must_run,
        reserve_eligibility=(),
    )


def _instance(generators: list[Generator], horizon: int) -> UCInstance:
    return UCInstance(
        path=Path("synthetic.json"),
        version="test",
        time_horizon=horizon,
        generators=generators,
        fixed_load=[0.0] * horizon,
        reserves=[],
        power_balance_penalty=[1000.0] * horizon,
    )


def _bits(index: int, width: int) -> tuple[int, ...]:
    return tuple((int(index) >> bit) & 1 for bit in range(width))


def test_compiled_logic_spec_matches_classical_logic_for_every_state() -> None:
    horizon = 3
    instance = _instance([_generator()], horizon)
    spec = compile_logic_feasibility_spec(
        instance,
        selected_generator_indices=(0,),
        base_commitment=np.zeros((1, horizon), dtype=int),
    )
    assert spec.always_infeasible is False
    assert spec.num_violation_qubits > 0
    for index in range(2**horizon):
        bits = _bits(index, horizon)
        commitment = np.asarray(bits, dtype=int).reshape((1, horizon))
        assert spec.is_feasible(bits) == is_logic_feasible(instance, commitment)


def test_fixed_unselected_violation_sets_always_infeasible() -> None:
    instance = _instance([_generator(must_run=True), _generator()], 2)
    base = np.ones((2, 2), dtype=int)
    base[0, 0] = 0
    spec = compile_logic_feasibility_spec(
        instance,
        selected_generator_indices=(1,),
        base_commitment=base,
    )
    assert spec.always_infeasible is True
    assert all(not spec.is_feasible(_bits(index, 2)) for index in range(4))
    assert "range(2**" not in inspect.getsource(compile_logic_feasibility_spec)


def test_logic_phase_oracle_marks_exact_feasible_states_and_uncomputes() -> None:
    instance = _instance([_generator()], 3)
    spec = compile_logic_feasibility_spec(
        instance,
        selected_generator_indices=(0,),
        base_commitment=np.zeros((1, 3), dtype=int),
    )
    probe = simulate_logic_feasibility_phase_oracle(spec)
    assert probe.max_phase_error < 1e-10
    assert probe.auxiliary_zero_probability > 1.0 - 1e-12
    expected = np.array(
        [
            is_logic_feasible(
                instance,
                np.asarray(_bits(index, 3), dtype=int).reshape((1, 3)),
            )
            for index in range(8)
        ],
        dtype=bool,
    )
    assert np.array_equal(probe.feasible_mask, expected)
