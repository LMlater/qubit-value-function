from __future__ import annotations

import numpy as np

from qubit_value_function.coherent_phase_value import (
    QuantizedSparseValueModel,
    conservative_integer_bounds,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig
from qubit_value_function.logic_feasibility_oracle import (
    BooleanLiteral,
    ForbiddenBooleanPattern,
    LogicFeasibilitySpec,
    simulate_joint_feasible_better_phase_oracle,
)
from qubit_value_function.sparse_phase_vqc import build_local_phase_features
from qubit_value_function.sparse_vqc_grover import (
    ordinary_grover_validation_plan,
    simulate_sparse_vqc_grover_statevector,
)


def _model() -> QuantizedSparseValueModel:
    features = build_local_phase_features(1, 2, generator_edges=())
    integer_weights = (1, 2) + (0,) * (len(features) - 2)
    lower, upper = conservative_integer_bounds(0, integer_weights)
    return QuantizedSparseValueModel(
        num_generators=1,
        num_periods=2,
        features=features,
        fixed_point_config=FixedPointConfig(fractional_bits=0, unit=1.0),
        real_intercept=0.0,
        real_weights=tuple(float(value) for value in integer_weights),
        integer_intercept=0,
        integer_weights=integer_weights,
        coefficient_quantization_errors=(0.0,) * (1 + len(integer_weights)),
        lower_bound=lower,
        upper_bound=upper,
        value_shift=-lower,
        shifted_upper_bound=upper - lower,
        num_value_qubits=max(1, int(upper - lower).bit_length()),
    )


def _spec() -> LogicFeasibilitySpec:
    return LogicFeasibilitySpec(
        num_generators=1,
        num_periods=2,
        selected_generator_indices=(0,),
        patterns=(
            ForbiddenBooleanPattern(
                literals=(BooleanLiteral(0, 0),),
                label="x0_must_be_one",
            ),
        ),
    )


def test_joint_phase_oracle_marks_only_feasible_and_better_states() -> None:
    probe = simulate_joint_feasible_better_phase_oracle(
        _model(),
        encoded_threshold=2,
        feasibility_spec=_spec(),
    )
    assert probe.max_phase_error < 1e-10
    assert probe.auxiliary_zero_probability > 1.0 - 1e-12
    assert np.flatnonzero(probe.better_mask).tolist() == [0, 1]
    assert np.flatnonzero(probe.feasible_mask).tolist() == [1, 3]
    assert np.flatnonzero(probe.marked_mask).tolist() == [1]


def test_joint_grover_amplifies_unique_feasible_better_state() -> None:
    model = _model()
    spec = _spec()
    plan = ordinary_grover_validation_plan(
        model,
        encoded_threshold=2,
        feasibility_spec=spec,
    )
    assert plan.marked_indices == (1,)
    assert plan.iterations == 1
    probe = simulate_sparse_vqc_grover_statevector(
        model,
        encoded_threshold=2,
        iterations=1,
        feasibility_spec=spec,
    )
    assert probe.marked_indices == (1,)
    assert probe.marked_probability > 1.0 - 1e-10
    assert probe.auxiliary_zero_probability > 1.0 - 1e-12
