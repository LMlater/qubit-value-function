from __future__ import annotations

import numpy as np

from qubit_value_function.logic_feasibility_oracle import (
    LogicFeasibilitySpec,
    simulate_logic_feasibility_phase_oracle,
)


def test_empty_constraint_set_marks_every_state_as_feasible() -> None:
    spec = LogicFeasibilitySpec(
        num_generators=1,
        num_periods=2,
        selected_generator_indices=(0,),
        patterns=(),
        always_infeasible=False,
    )
    probe = simulate_logic_feasibility_phase_oracle(spec)
    assert np.all(probe.feasible_mask)
    assert np.allclose(probe.phase_signs, -1.0, atol=1e-10)
    assert probe.auxiliary_zero_probability > 1.0 - 1e-12


def test_constant_compilation_failure_marks_no_state_as_feasible() -> None:
    spec = LogicFeasibilitySpec(
        num_generators=1,
        num_periods=2,
        selected_generator_indices=(0,),
        patterns=(),
        always_infeasible=True,
    )
    probe = simulate_logic_feasibility_phase_oracle(spec)
    assert not np.any(probe.feasible_mask)
    assert np.allclose(probe.phase_signs, 1.0, atol=1e-10)
    assert probe.auxiliary_zero_probability > 1.0 - 1e-12
