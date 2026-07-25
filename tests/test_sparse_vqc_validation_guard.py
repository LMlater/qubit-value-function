from __future__ import annotations

import pytest

from qubit_value_function.coherent_phase_value import (
    QuantizedSparseValueModel,
    conservative_integer_bounds,
)
from qubit_value_function.fixed_point_oracle import FixedPointConfig
from qubit_value_function.sparse_phase_vqc import build_local_phase_features
from qubit_value_function.sparse_vqc_grover import (
    direct_float_marked_indices_for_validation,
    ordinary_grover_validation_plan,
)


def _large_validation_model() -> QuantizedSparseValueModel:
    num_generators = 13
    num_periods = 1
    features = build_local_phase_features(num_generators, num_periods)
    integer_weights = tuple(1 for _ in features)
    lower, upper = conservative_integer_bounds(0, integer_weights)
    return QuantizedSparseValueModel(
        num_generators=num_generators,
        num_periods=num_periods,
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


def test_small_instance_validation_enumeration_rejects_more_than_twelve_qubits() -> None:
    model = _large_validation_model()
    with pytest.raises(ValueError, match="validation enumeration"):
        ordinary_grover_validation_plan(
            model,
            encoded_threshold=2,
            max_validation_qubits=12,
        )
    with pytest.raises(ValueError, match="validation enumeration"):
        direct_float_marked_indices_for_validation(
            model,
            predict_cost=lambda bitstring: 0.0,
            encoded_threshold=2,
            max_validation_qubits=12,
        )
