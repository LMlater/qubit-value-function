"""Leakage-free protocol primitives for the Stage B generalization benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Hashable, Mapping, Sequence

import numpy as np

from .load_scenarios import LoadNormalizer, fit_load_normalizer


TRAIN_LOAD_MULTIPLIERS = (0.85, 1.0, 1.15)
INTERPOLATION_LOAD_MULTIPLIERS = (0.925, 1.075)
EXTRAPOLATION_LOAD_MULTIPLIERS = (0.80, 1.20)
PARTITION_ALGORITHM_VERSION = "stage2-fit-validation-v1"


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _finite(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional sequence")
    return array


def build_state_split(split_seed: int, *, state_count: int = 16, training_count: int = 8) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Select the original 8/8 state split independently of input ordering."""

    if state_count != 16 or training_count != 8:
        raise ValueError("stage2 protocol requires exactly sixteen states and eight training states")
    selected = np.random.default_rng(int(split_seed)).choice(state_count, size=training_count, replace=False)
    training = tuple(sorted(int(value) for value in selected))
    unseen = tuple(index for index in range(state_count) if index not in set(training))
    return training, unseen


@dataclass(frozen=True)
class FixedStratifiedPartition:
    split_seed: int
    training_indices: tuple[int, ...]
    unseen_indices: tuple[int, ...]
    fit_indices_by_load: Mapping[float, tuple[int, ...]]
    validation_indices_by_load: Mapping[float, tuple[int, ...]]
    algorithm_version: str
    canonical_json: str
    partition_sha256: str

    @property
    def fit_sample_pairs(self) -> tuple[tuple[float, int], ...]:
        return tuple((load, index) for load in TRAIN_LOAD_MULTIPLIERS for index in self.fit_indices_by_load[load])

    @property
    def validation_sample_pairs(self) -> tuple[tuple[float, int], ...]:
        return tuple((load, index) for load in TRAIN_LOAD_MULTIPLIERS for index in self.validation_indices_by_load[load])

    def as_dict(self) -> dict[str, object]:
        return json.loads(self.canonical_json)


def build_fixed_stratified_partition(
    *, split_seed: int, training_indices: Sequence[int], unseen_indices: Sequence[int]
) -> FixedStratifiedPartition:
    """Partition each training load into deterministic 6-state fit and 2-state validation sets."""

    training = tuple(sorted(int(value) for value in training_indices))
    unseen = tuple(sorted(int(value) for value in unseen_indices))
    if len(training) != 8 or len(set(training)) != 8:
        raise ValueError("training_indices must contain eight unique states")
    if len(unseen) != 8 or len(set(unseen)) != 8:
        raise ValueError("unseen_indices must contain eight unique states")
    if set(training) & set(unseen) or set(training) | set(unseen) != set(range(16)):
        raise ValueError("training_indices and unseen_indices must be a disjoint sixteen-state partition")
    fit_by_load: dict[float, tuple[int, ...]] = {}
    validation_by_load: dict[float, tuple[int, ...]] = {}
    for load in TRAIN_LOAD_MULTIPLIERS:
        selection_input = _canonical_json(
            {
                "algorithm_version": PARTITION_ALGORITHM_VERSION,
                "split_seed": int(split_seed),
                "training_indices": list(training),
                "load_multiplier": float(load),
            }
        )
        seed = int.from_bytes(hashlib.sha256(selection_input.encode("utf-8")).digest()[:8], "big")
        ordered = tuple(int(value) for value in np.random.default_rng(seed).permutation(training))
        validation_by_load[load] = tuple(sorted(ordered[:2]))
        fit_by_load[load] = tuple(sorted(ordered[2:]))
    payload = {
        "algorithm_version": PARTITION_ALGORITHM_VERSION,
        "split_seed": int(split_seed),
        "training_indices": list(training),
        "unseen_indices": list(unseen),
        "fit_indices_by_load": [
            {"load_multiplier": load, "indices": list(fit_by_load[load])} for load in TRAIN_LOAD_MULTIPLIERS
        ],
        "validation_indices_by_load": [
            {"load_multiplier": load, "indices": list(validation_by_load[load])} for load in TRAIN_LOAD_MULTIPLIERS
        ],
    }
    canonical = _canonical_json(payload)
    return FixedStratifiedPartition(
        split_seed=int(split_seed), training_indices=training, unseen_indices=unseen,
        fit_indices_by_load=fit_by_load, validation_indices_by_load=validation_by_load,
        algorithm_version=PARTITION_ALGORITHM_VERSION, canonical_json=canonical,
        partition_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class TargetNormalizer:
    mean: float
    scale: float

    def transform(self, value: float) -> float:
        return (float(value) - self.mean) / self.scale

    def inverse_transform(self, value: float) -> float:
        return self.mean + self.scale * float(value)


@dataclass(frozen=True)
class FitNormalizers:
    load: LoadNormalizer
    target: TargetNormalizer


def fit_normalizers_from_fit_rows(fit_rows: Sequence[Mapping[str, object]]) -> FitNormalizers:
    """Fit both normalizers from the caller-provided fit rows only."""

    if not fit_rows:
        raise ValueError("fit_rows must not be empty")
    load_vectors = [list(row["load_vector"]) for row in fit_rows]
    targets = _finite([float(row["true_cost"]) for row in fit_rows], name="fit targets")
    scale = float(max(targets.std(), 1.0))
    return FitNormalizers(
        load=fit_load_normalizer(load_vectors),
        target=TargetNormalizer(mean=float(targets.mean()), scale=scale),
    )


def regression_metrics(true_costs: Sequence[float], predicted_costs: Sequence[float]) -> dict[str, float]:
    truth = _finite(true_costs, name="true_costs")
    predicted = _finite(predicted_costs, name="predicted_costs")
    if truth.shape != predicted.shape:
        raise ValueError("true_costs and predicted_costs must have the same shape")
    error = predicted - truth
    return {
        "count": float(len(truth)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "mean_signed_error": float(np.mean(error)),
    }


def selection_regret(
    *, true_costs: Sequence[float], predicted_costs: Sequence[float], group_keys: Sequence[Hashable]
) -> dict[str, float]:
    truth = _finite(true_costs, name="true_costs")
    predicted = _finite(predicted_costs, name="predicted_costs")
    if truth.shape != predicted.shape or len(group_keys) != len(truth):
        raise ValueError("costs and group_keys must be aligned")
    groups: dict[Hashable, list[int]] = {}
    for position, key in enumerate(group_keys):
        groups.setdefault(key, []).append(position)
    regrets = []
    for positions in groups.values():
        selected_position = min(positions, key=lambda position: (float(predicted[position]), position))
        regrets.append(float(truth[selected_position] - np.min(truth[positions])))
    values = _finite(regrets, name="regrets")
    return {
        "group_count": float(len(values)),
        "mean_regret": float(values.mean()),
        "maximum_regret": float(values.max()),
    }


@dataclass(frozen=True)
class ModelCandidateScore:
    name: str
    validation_mae: float
    validation_regret: float
    parameter_count: int
    order: int


def select_best_candidate(candidates: Sequence[ModelCandidateScore]) -> ModelCandidateScore:
    if not candidates:
        raise ValueError("at least one candidate score is required")
    for candidate in candidates:
        if not np.isfinite(candidate.validation_mae) or not np.isfinite(candidate.validation_regret):
            raise ValueError("candidate validation metrics must be finite")
        if candidate.parameter_count < 0:
            raise ValueError("candidate parameter_count must be non-negative")
    return min(
        candidates,
        key=lambda item: (item.validation_mae, item.validation_regret, item.parameter_count, item.order),
    )


def select_threshold_cutoff_from_validation(
    *, validation_labels: Sequence[int | bool], validation_probabilities: Sequence[float], cutoffs: Sequence[float]
) -> float:
    """Choose a QNN decision cutoff from validation rows only.

    The fixed order is validation F1, validation accuracy, then supplied cutoff
    order. This function intentionally has no test-row argument.
    """

    labels = np.asarray(validation_labels, dtype=bool)
    probabilities = _finite(validation_probabilities, name="validation_probabilities")
    if labels.shape != probabilities.shape:
        raise ValueError("validation labels and probabilities must be aligned")
    candidates = _finite(cutoffs, name="cutoffs")
    if np.any((candidates < 0.0) | (candidates > 1.0)):
        raise ValueError("cutoffs must be within [0, 1]")
    ranked: list[tuple[float, float, int, float]] = []
    for order, cutoff in enumerate(candidates):
        predicted = probabilities >= cutoff
        tp = int(np.sum(predicted & labels))
        fp = int(np.sum(predicted & ~labels))
        fn = int(np.sum(~predicted & labels))
        tn = int(np.sum(~predicted & ~labels))
        f1 = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        accuracy = (tp + tn) / len(labels)
        ranked.append((-f1, -accuracy, order, float(cutoff)))
    return min(ranked)[3]


def make_truth_cache_key(
    *, generator_pair: Sequence[int], window_start: int, load_multiplier: float, state_index: int
) -> str:
    return _canonical_json(
        {
            "generator_pair": [int(value) for value in generator_pair],
            "load_multiplier": float(load_multiplier),
            "state_index": int(state_index),
            "window_start": int(window_start),
        }
    )


class EDLPTruthCache:
    """In-memory cache whose solve count changes only for a new canonical key."""

    def __init__(self) -> None:
        self._values: dict[str, float] = {}
        self.solve_count = 0

    def resolve(self, key: str, solver: Callable[[], float]) -> float:
        if key not in self._values:
            value = float(solver())
            if not np.isfinite(value):
                raise ValueError("ED/LP solver returned a non-finite cost")
            self._values[key] = value
            self.solve_count += 1
        return self._values[key]
