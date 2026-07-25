from __future__ import annotations

import pytest

from qubit_value_function.coherent_phase_value import (
    QuantizedSparseValueModel,
    conservative_integer_bounds,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig
from qubit_value_function.logic_feasibility_oracle import (
    BooleanLiteral,
    ForbiddenBooleanPattern,
    LogicFeasibilitySpec,
)
from qubit_value_function.sparse_phase_vqc import build_local_phase_features
from qubit_value_function.sparse_vqc_bbht import execute_bbht_trial_mps
from qubit_value_function.sparse_vqc_grover import AerSimulator


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


@pytest.mark.skipif(AerSimulator is None, reason="qiskit-aer is not installed")
def test_joint_one_shot_mps_returns_unique_feasible_better_state() -> None:
    spec = LogicFeasibilitySpec(
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
    execution = execute_bbht_trial_mps(
        _model(),
        encoded_threshold=2,
        iterations=1,
        shots=1,
        seed=17,
        feasibility_spec=spec,
    )
    assert execution.measured_index == 1
    assert execution.measured_bitstring == "10"
    assert sum(execution.raw_counts.values()) == 1
    assert sum(execution.x_counts.values()) == 1
    assert execution.auxiliary_zero_probability > 1.0 - 1e-12
