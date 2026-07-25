from __future__ import annotations

from experiments.stage1_case14_active_logic_joint_bbht import (
    _representative_index_order,
    _select_initial_index,
)
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
from qubit_value_function.sparse_vqc_bbht import ExactCandidateEvaluation
from qubit_value_function.sparse_vqc_grover import ordinary_grover_validation_plan


def _model() -> QuantizedSparseValueModel:
    features = build_local_phase_features(2, 2)
    weights = (1, 2, 4, 8) + (0,) * (len(features) - 4)
    lower, upper = conservative_integer_bounds(0, weights)
    return QuantizedSparseValueModel(
        num_generators=2,
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


def _record(cost: float) -> ExactCandidateEvaluation:
    return ExactCandidateEvaluation(
        success=True,
        total_cost=float(cost),
        message="ok",
        source="training_exact_cache",
    )


def test_representative_order_and_worst_training_initialization() -> None:
    order = _representative_index_order(4)
    assert len(order) == 16
    assert set(order) == set(range(16))
    assert order[:2] == (0, 15)
    cache = {0: _record(10.0), 3: _record(30.0), 5: _record(20.0)}
    assert _select_initial_index((0, 3, 5), cache, policy="worst-training", seed=0) == 3
    assert _select_initial_index((0, 3, 5), cache, policy="best-training", seed=0) == 0


def test_joint_marked_set_is_strict_subset_when_infeasible_better_state_exists() -> None:
    model = _model()
    spec = LogicFeasibilitySpec(
        num_generators=2,
        num_periods=2,
        selected_generator_indices=(0, 1),
        patterns=(
            ForbiddenBooleanPattern(
                literals=(BooleanLiteral(qubit=0, value=0),),
                label="must_run_g0_t0",
            ),
        ),
    )
    cost_only = ordinary_grover_validation_plan(model, encoded_threshold=4)
    joint = ordinary_grover_validation_plan(
        model,
        encoded_threshold=4,
        feasibility_spec=spec,
    )
    assert set(joint.marked_indices) < set(cost_only.marked_indices)
    assert 0 in cost_only.marked_indices
    assert 0 not in joint.marked_indices
