from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .uc_loader import UCInstance


@dataclass(frozen=True)
class LoadNormalizer:
    """Affine normalizer fitted only from training load vectors."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def transform(self, values: Sequence[float]) -> np.ndarray:
        row = np.asarray(values, dtype=float)
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        if row.shape != mean.shape:
            raise ValueError("load vector shape does not match the fitted normalizer")
        return (row - mean) / scale

    def as_dict(self) -> dict[str, object]:
        return {"mean": list(self.mean), "scale": list(self.scale)}


def scaled_load_instance(instance: UCInstance, multiplier: float) -> UCInstance:
    """Return an independent scenario with fixed load scaled uniformly.

    Reserve requirements and all generator data are intentionally unchanged in
    the first Stage-B pilot so the experiment isolates the effect of load.
    """

    multiplier = float(multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("load multiplier must be a finite positive value")
    return replace(
        instance,
        fixed_load=[multiplier * float(value) for value in instance.fixed_load],
    )


def fit_load_normalizer(load_vectors: Sequence[Sequence[float]]) -> LoadNormalizer:
    array = np.asarray(load_vectors, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("load_vectors must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("load_vectors must contain only finite values")
    mean = array.mean(axis=0)
    scale = array.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return LoadNormalizer(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
    )
