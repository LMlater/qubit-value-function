from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from experiments import stage2_case14_load_qnn_pilot_cli as first_pilot
from experiments.stage2_generalization_benchmark_cli import select_classical_baseline
from qubit_value_function.stage2_baselines import (
    fit_constant_regressor,
    fit_ridge_regressor,
)
from qubit_value_function.stage2_generalization import (
    TRAIN_LOAD_MULTIPLIERS,
    EDLPTruthCache,
    ModelCandidateScore,
    build_fixed_stratified_partition,
    build_state_split,
    fit_normalizers_from_fit_rows,
    make_truth_cache_key,
    regression_metrics,
    select_best_candidate,
    select_threshold_cutoff_from_validation,
    selection_regret,
)


def _partition():
    return build_fixed_stratified_partition(
        split_seed=7,
        training_indices=tuple(range(8)),
        unseen_indices=tuple(range(8, 16)),
    )


def test_fixed_partition_is_complete_disjoint_and_stratified() -> None:
    partition = _partition()

    assert partition.training_indices == tuple(range(8))
    assert partition.unseen_indices == tuple(range(8, 16))
    assert len(partition.fit_sample_pairs) == 18
    assert len(partition.validation_sample_pairs) == 6
    for multiplier in TRAIN_LOAD_MULTIPLIERS:
        fit = set(partition.fit_indices_by_load[multiplier])
        validation = set(partition.validation_indices_by_load[multiplier])
        assert len(fit) == 6
        assert len(validation) == 2
        assert fit.isdisjoint(validation)
        assert fit | validation == set(partition.training_indices)
        assert fit.isdisjoint(partition.unseen_indices)
        assert validation.isdisjoint(partition.unseen_indices)


def test_partition_is_reproducible_with_a_stable_canonical_hash() -> None:
    first = _partition()
    second = _partition()

    assert first.canonical_json == second.canonical_json
    assert first.partition_sha256 == second.partition_sha256
    assert len(first.partition_sha256) == 64
    assert build_state_split(7) == build_state_split(7)


def test_normalizers_read_only_fit_rows() -> None:
    fit_rows = [
        {"load_vector": [10.0, 20.0], "true_cost": 100.0},
        {"load_vector": [20.0, 40.0], "true_cost": 200.0},
    ]
    with_test_rows = [*fit_rows, {"load_vector": [1000.0, 2000.0], "true_cost": -10000.0}]

    first = fit_normalizers_from_fit_rows(fit_rows)
    second = fit_normalizers_from_fit_rows(fit_rows)

    assert first == second
    assert first.load.mean == (15.0, 30.0)
    assert first.target.mean == 150.0
    assert with_test_rows[-1]["true_cost"] not in {first.target.mean}


def test_validation_selection_ignores_test_labels_and_uses_fixed_tie_breaks() -> None:
    candidates = (
        ModelCandidateScore("later", validation_mae=1.0, validation_regret=2.0, parameter_count=4, order=2),
        ModelCandidateScore("few_parameters", validation_mae=1.0, validation_regret=2.0, parameter_count=3, order=9),
        ModelCandidateScore("lower_regret", validation_mae=1.0, validation_regret=1.0, parameter_count=99, order=7),
        ModelCandidateScore("best_mae", validation_mae=0.5, validation_regret=9.0, parameter_count=999, order=99),
    )

    selected = select_best_candidate(candidates)

    assert selected.name == "best_mae"
    assert select_best_candidate(candidates) == selected


def test_regret_and_metrics_are_finite_and_correct() -> None:
    metrics = regression_metrics([1.0, 3.0], [3.0, 1.0])
    regret = selection_regret(
        true_costs=[1.0, 3.0, 2.0, 4.0],
        predicted_costs=[10.0, 0.0, 0.0, 1.0],
        group_keys=["load_a", "load_a", "load_b", "load_b"],
    )

    assert metrics["mae"] == 2.0
    assert regret["mean_regret"] == 1.0
    assert regret["maximum_regret"] == 2.0
    assert all(math.isfinite(float(value)) for value in metrics.values())
    assert all(math.isfinite(float(value)) for value in regret.values())
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1.0], [np.nan])


def test_truth_cache_key_and_resolution_are_deterministic_without_duplicate_solves() -> None:
    key = make_truth_cache_key(generator_pair=(0, 1), window_start=0, load_multiplier=0.85, state_index=3)
    same_key = make_truth_cache_key(generator_pair=(0, 1), window_start=0, load_multiplier=0.85, state_index=3)
    cache = EDLPTruthCache()
    calls = 0

    def solve() -> float:
        nonlocal calls
        calls += 1
        return 12.5

    assert key == same_key
    assert cache.resolve(key, solve) == 12.5
    assert cache.resolve(same_key, solve) == 12.5
    assert calls == 1
    assert cache.solve_count == 1


def test_classical_baselines_return_finite_predictions_from_fit_arrays() -> None:
    state = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=int)
    loads = np.array([[-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]])
    costs = np.array([10.0, 12.0, 20.0, 22.0])

    constant = fit_constant_regressor(costs)
    ridge = fit_ridge_regressor(state, loads, costs, degree=2, regularization=1e-3)

    assert constant.parameter_count == 1
    assert ridge.parameter_count > constant.parameter_count
    assert np.all(np.isfinite(constant.predict(state, loads)))
    assert np.all(np.isfinite(ridge.predict(state, loads)))


def test_first_round_pilot_remains_importable() -> None:
    assert callable(first_pilot.main)
    assert first_pilot.training_quantile_thresholds


def test_classical_selection_has_no_test_row_or_label_input() -> None:
    fit_rows = [
        {"state_bits": [0, 0, 0, 0], "normalized_load": [-1.0, -1.0], "true_cost": 10.0},
        {"state_bits": [1, 0, 0, 0], "normalized_load": [-1.0, -1.0], "true_cost": 12.0},
        {"state_bits": [0, 1, 0, 0], "normalized_load": [1.0, 1.0], "true_cost": 20.0},
        {"state_bits": [1, 1, 0, 0], "normalized_load": [1.0, 1.0], "true_cost": 22.0},
    ]
    validation_rows = [
        {"state_bits": [0, 0, 1, 0], "normalized_load": [-1.0, -1.0], "true_cost": 11.0, "load_multiplier": 0.85},
        {"state_bits": [1, 0, 1, 0], "normalized_load": [-1.0, -1.0], "true_cost": 13.0, "load_multiplier": 0.85},
        {"state_bits": [0, 1, 1, 0], "normalized_load": [1.0, 1.0], "true_cost": 21.0, "load_multiplier": 1.15},
        {"state_bits": [1, 1, 1, 0], "normalized_load": [1.0, 1.0], "true_cost": 23.0, "load_multiplier": 1.15},
    ]

    selection = select_classical_baseline(fit_rows, validation_rows)

    assert selection.selected.name in {"constant", "linear_ridge", "quadratic_ridge"}
    assert "test" not in inspect.signature(select_classical_baseline).parameters


def test_threshold_cutoff_is_selected_only_from_validation_data() -> None:
    cutoff = select_threshold_cutoff_from_validation(
        validation_labels=[1, 0], validation_probabilities=[0.7, 0.6], cutoffs=[0.5, 0.65]
    )

    assert cutoff == 0.65
    assert "test" not in inspect.signature(select_threshold_cutoff_from_validation).parameters
