from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import stage1_case14_2x2_sparse_vqc_bbht as sparse_vqc_bbht

from qubit_value_function.candidate_discovery_training_splits import (
    FROZEN_SEED0_SPLIT_POLICY,
    SEEDED_WITHOUT_REPLACEMENT_POLICY,
    resolve_candidate_discovery_training_indices,
    training_indices_hash,
)


def test_seeded_split_is_deterministic_and_distinct_from_frozen_seed0() -> None:
    base_scenario_id = "case14-g0g1-w0"
    eligible_indices = tuple(range(16))
    frozen_seed0 = (0, 1, 2, 3, 9, 10, 12, 15)

    seed0 = resolve_candidate_discovery_training_indices(
        base_scenario_id=base_scenario_id,
        training_data_seed=0,
        training_index_policy=FROZEN_SEED0_SPLIT_POLICY,
        eligible_indices=eligible_indices,
        frozen_seed0_training_indices=frozen_seed0,
    )
    seed1 = resolve_candidate_discovery_training_indices(
        base_scenario_id=base_scenario_id,
        training_data_seed=1,
        training_index_policy=SEEDED_WITHOUT_REPLACEMENT_POLICY,
        eligible_indices=eligible_indices,
        frozen_seed0_training_indices=frozen_seed0,
    )
    seed2 = resolve_candidate_discovery_training_indices(
        base_scenario_id=base_scenario_id,
        training_data_seed=2,
        training_index_policy=SEEDED_WITHOUT_REPLACEMENT_POLICY,
        eligible_indices=eligible_indices,
        frozen_seed0_training_indices=frozen_seed0,
    )

    assert seed0 == frozen_seed0
    assert seed1 == resolve_candidate_discovery_training_indices(
        base_scenario_id=base_scenario_id,
        training_data_seed=1,
        training_index_policy=SEEDED_WITHOUT_REPLACEMENT_POLICY,
        eligible_indices=eligible_indices,
        frozen_seed0_training_indices=frozen_seed0,
    )
    assert len({seed0, seed1, seed2}) == 3


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_candidate_splits_have_eight_unique_in_range_indices(seed: int) -> None:
    indices = resolve_candidate_discovery_training_indices(
        base_scenario_id="case14-g1g5-w2",
        training_data_seed=seed,
        training_index_policy=(
            FROZEN_SEED0_SPLIT_POLICY if seed == 0 else SEEDED_WITHOUT_REPLACEMENT_POLICY
        ),
        eligible_indices=tuple(range(16)),
        frozen_seed0_training_indices=(0, 2, 3, 4, 6, 10, 12, 15),
    )

    assert len(indices) == 8
    assert len(set(indices)) == 8
    assert all(0 <= index <= 15 for index in indices)


def test_seeded_split_depends_only_on_declared_nontruth_inputs() -> None:
    kwargs = {
        "base_scenario_id": "case14-g0g5-w1",
        "training_data_seed": 1,
        "training_index_policy": SEEDED_WITHOUT_REPLACEMENT_POLICY,
        "eligible_indices": tuple(range(16)),
        "frozen_seed0_training_indices": (0, 3, 5, 6, 9, 10, 12, 15),
    }

    result = resolve_candidate_discovery_training_indices(**kwargs)

    assert result
    assert "cost" not in resolve_candidate_discovery_training_indices.__code__.co_names
    assert "label" not in resolve_candidate_discovery_training_indices.__code__.co_names
    assert "optimum" not in resolve_candidate_discovery_training_indices.__code__.co_names


def test_training_indices_hash_is_stable_for_the_actual_order() -> None:
    indices = (0, 15, 3, 12, 10, 9, 1, 2)

    assert training_indices_hash(indices) == hashlib.sha256(
        b'{"training_indices":[0,15,3,12,10,9,1,2]}'
    ).hexdigest()


def test_explicit_split_reaches_training_collection_without_changing_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEvaluator:
        def __init__(self, _instance: object) -> None:
            pass

        def evaluate(self, _commitment: object) -> SimpleNamespace:
            return SimpleNamespace(success=True, total_cost=1.0, message="fake")

    monkeypatch.setattr(sparse_vqc_bbht, "FixedCommitmentEvaluator", FakeEvaluator)
    monkeypatch.setattr(sparse_vqc_bbht, "is_logic_feasible", lambda *_args: True)
    commitments = np.zeros((16, 1, 1), dtype=int)
    requested = (8, 7, 6, 5, 4, 3, 2, 1)

    explicit = sparse_vqc_bbht._collect_training_data(
        object(), commitments, train_sample_count=8, num_x_qubits=4,
        training_indices=requested,
    )
    legacy_default = sparse_vqc_bbht._collect_training_data(
        object(), commitments, train_sample_count=8, num_x_qubits=4,
    )

    assert tuple(explicit[0]) == requested
    assert tuple(legacy_default[0]) == (0, 15, 3, 12, 5, 10, 6, 9)
