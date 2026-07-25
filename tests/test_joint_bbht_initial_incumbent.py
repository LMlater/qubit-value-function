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
from qubit_value_function.sparse_vqc_bbht import (
    BBHTConfig,
    ExactCandidateEvaluation,
    run_sparse_vqc_bbht,
)


def _model() -> QuantizedSparseValueModel:
    features = build_local_phase_features(1, 2, generator_edges=())
    weights = (1, 2) + (0,) * (len(features) - 2)
    lower, upper = conservative_integer_bounds(0, weights)
    return QuantizedSparseValueModel(
        num_generators=1,
        num_periods=2,
        features=features,
        fixed_point_config=FixedPointConfig(fractional_bits=0, unit=1.0),
        real_intercept=0.0,
        real_weights=tuple(float(value) for value in weights),
        integer_intercept=0,
        integer_weights=weights,
        coefficient_quantization_errors=(0.0,) * (1 + len(weights)),
        lower_bound=lower,
        upper_bound=upper,
        value_shift=-lower,
        shifted_upper_bound=upper - lower,
        num_value_qubits=max(1, int(upper - lower).bit_length()),
    )


def test_joint_bbht_rejects_hard_infeasible_initial_incumbent() -> None:
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
    cache = {
        0: ExactCandidateEvaluation(
            success=True,
            total_cost=0.0,
            message="cached",
            source="training_exact_cache",
        )
    }
    with pytest.raises(ValueError, match="hard logic-feasibility"):
        run_sparse_vqc_bbht(
            _model(),
            initial_incumbent_index=0,
            initial_exact_cache=cache,
            evaluate_candidate=lambda index: pytest.fail("must stop before search"),
            config=BBHTConfig(),
            feasibility_spec=spec,
            trial_executor=lambda *args: pytest.fail("must stop before circuit execution"),
        )
