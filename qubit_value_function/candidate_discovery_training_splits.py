"""Truth-blind training split policies for best-training candidate discovery."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Sequence


FROZEN_SEED0_SPLIT_POLICY = "frozen_seed0_split_a_v1"
SEEDED_WITHOUT_REPLACEMENT_POLICY = "seeded_without_replacement_v1"
_SEARCH_SPACE_SIZE = 16


def training_indices_hash(training_indices: Sequence[int]) -> str:
    """Return the stable identifier for the exact persisted training order."""

    payload = json.dumps(
        {"training_indices": [int(index) for index in training_indices]},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_indices(indices: Sequence[int], *, sample_count: int) -> tuple[int, ...]:
    resolved = tuple(int(index) for index in indices)
    if len(resolved) != int(sample_count):
        raise ValueError("candidate discovery training split must have the requested sample count")
    if len(set(resolved)) != len(resolved):
        raise ValueError("candidate discovery training split must have unique indices")
    if any(index < 0 or index >= _SEARCH_SPACE_SIZE for index in resolved):
        raise ValueError("candidate discovery training split index is outside 0..15")
    return resolved


def _seeded_sample(
    *,
    base_scenario_id: str,
    training_data_seed: int,
    eligible_indices: tuple[int, ...],
    sample_count: int,
    nonce: int,
) -> tuple[int, ...]:
    seed_material = json.dumps(
        {
            "base_scenario_id": str(base_scenario_id),
            "nonce": int(nonce),
            "policy": SEEDED_WITHOUT_REPLACEMENT_POLICY,
            "training_data_seed": int(training_data_seed),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    entropy = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest(), "big")
    return tuple(random.Random(entropy).sample(eligible_indices, int(sample_count)))


def resolve_candidate_discovery_training_indices(
    *,
    base_scenario_id: str,
    training_data_seed: int,
    training_index_policy: str,
    eligible_indices: Sequence[int],
    frozen_seed0_training_indices: Sequence[int],
    sample_count: int = 8,
) -> tuple[int, ...]:
    """Resolve a reproducible split using IDs and hard-logic eligibility only.

    This helper intentionally accepts no labels, ED/LP values, or global-optimum
    facts.  Seed 0 is the explicitly frozen legacy split A.  Positive seeds use
    a stable hash-backed sample without replacement and are collision-resolved
    against preceding registered splits for the same base scenario.
    """

    data_seed = int(training_data_seed)
    frozen = _validate_indices(
        frozen_seed0_training_indices, sample_count=int(sample_count)
    )
    eligible = tuple(sorted({int(index) for index in eligible_indices}))
    if len(eligible) < int(sample_count):
        raise ValueError("candidate discovery needs at least eight hard-logic-eligible states")
    if any(index < 0 or index >= _SEARCH_SPACE_SIZE for index in eligible):
        raise ValueError("candidate discovery eligible index is outside 0..15")
    if not set(frozen).issubset(eligible):
        raise ValueError("frozen seed-0 split contains a non-eligible state")

    if data_seed == 0:
        if str(training_index_policy) != FROZEN_SEED0_SPLIT_POLICY:
            raise ValueError("seed 0 requires the frozen legacy split policy")
        return frozen
    if data_seed < 0:
        raise ValueError("training data seed must be nonnegative")
    if str(training_index_policy) != SEEDED_WITHOUT_REPLACEMENT_POLICY:
        raise ValueError("positive training data seeds require seeded sampling")

    occupied = {frozen}
    selected: tuple[int, ...] | None = None
    for prior_seed in range(1, data_seed + 1):
        nonce = 0
        while True:
            proposal = _seeded_sample(
                base_scenario_id=base_scenario_id,
                training_data_seed=prior_seed,
                eligible_indices=eligible,
                sample_count=int(sample_count),
                nonce=nonce,
            )
            if proposal not in occupied:
                selected = proposal
                occupied.add(proposal)
                break
            nonce += 1
    if selected is None:
        raise AssertionError("positive training data seed did not resolve a split")
    return selected
