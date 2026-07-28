"""Theory and reproducible classical sampling for an oracle-query diagnostic."""

from __future__ import annotations

import random
from typing import Sequence


def uniform_hit_probability(marked_count: int, dimension: int, budget: int, *, replacement: bool) -> float:
    """Probability of finding a Boolean-oracle target without revealing its index."""
    marked_count, dimension, budget = int(marked_count), int(dimension), int(budget)
    if not 0 <= marked_count <= dimension or budget < 0:
        raise ValueError("invalid query parameters")
    if replacement:
        return 1.0 - (1.0 - marked_count / dimension) ** budget
    return min(budget, dimension) / dimension if marked_count == 1 else 1.0 - _no_hit_without_replacement(marked_count, dimension, budget)


def _no_hit_without_replacement(marked_count: int, dimension: int, budget: int) -> float:
    value = 1.0
    for position in range(min(budget, dimension)):
        value *= (dimension - marked_count - position) / (dimension - position)
    return max(value, 0.0)


def sample_first_hit(marked_indices: Sequence[int], *, dimension: int, replacement: bool, seed: int) -> int:
    """Query a hidden Boolean predicate; sampling never draws from the marked set directly."""
    rng = random.Random(int(seed)); marked = {int(item) for item in marked_indices}
    population = list(range(int(dimension)))
    if replacement:
        for query in range(1, 100000):
            if rng.choice(population) in marked:
                return query
    rng.shuffle(population)
    for query, candidate in enumerate(population, start=1):
        if candidate in marked:
            return query
    raise RuntimeError("no marked state")
