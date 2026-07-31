"""Fixed, fit-only controlled alternatives for the Stage B diagnostic protocol."""

from __future__ import annotations

from time import perf_counter
from typing import Sequence

import numpy as np

from .stage2_baselines import design_matrix
from .stage2_models import LoadConditionedMLPModel, LoadConditionedRidgeModel, SimplifiedExpectationQNNModel, Stage2Model


class QuadraticResidualModel:
    """Quadratic Ridge plus a correction trained only on fit residuals."""

    def __init__(self, *, correction: str, seed: int) -> None:
        self.base = LoadConditionedRidgeModel(degree=2, regularization=1e-3, seed=0)
        if correction == "linear": self.correction: Stage2Model = LoadConditionedRidgeModel(degree=1, regularization=1e-3, seed=seed)
        elif correction == "mlp": self.correction = LoadConditionedMLPModel(hidden_units=8, maxiter=80, learning_rate=0.03, seed=seed)
        elif correction == "simplified_qnn": self.correction = SimplifiedExpectationQNNModel(layers=1, head_regularization=0.5, theta_regularization=1e-5, maxiter=20, seed=seed)
        else: raise ValueError(f"unknown_residual_correction:{correction}")
        self.correction_name, self.random_seed = correction, int(seed)
        self.fit_residual_mae = float("nan"); self.runtime_seconds = 0.0

    def fit(self, states: Sequence[Sequence[int]], loads: Sequence[Sequence[float]], costs: Sequence[float]) -> "QuadraticResidualModel":
        started=perf_counter(); targets=np.asarray(costs,dtype=float)
        self.base.fit(states,loads,targets); residual=targets-self.base.predict(states,loads)
        self.correction.fit(states,loads,residual)
        self.fit_residual_mae=float(np.mean(np.abs(residual-self.correction.predict(states,loads))))
        self.runtime_seconds=float(perf_counter()-started)
        return self

    def predict(self, states: Sequence[Sequence[int]], loads: Sequence[Sequence[float]]) -> np.ndarray:
        return self.base.predict(states,loads)+self.correction.predict(states,loads)


class PairwiseLinearRanker:
    """Fixed logistic pairwise ranking loss; pairs are only within a load group."""

    def __init__(self, *, seed: int = 0, iterations: int = 200, learning_rate: float = 0.05) -> None:
        self.seed, self.iterations, self.learning_rate = int(seed), int(iterations), float(learning_rate)
        self.weights: np.ndarray | None = None; self.runtime_seconds=0.0

    def fit(self, states: Sequence[Sequence[int]], loads: Sequence[Sequence[float]], costs: Sequence[float], load_groups: Sequence[float]) -> "PairwiseLinearRanker":
        started=perf_counter(); features=design_matrix(states,loads,degree=1); costs=np.asarray(costs,dtype=float); groups=np.asarray(load_groups,dtype=float)
        differences=[]; labels=[]
        for load in sorted(set(groups.tolist())):
            indexes=np.where(groups==load)[0]
            for left in indexes:
                for right in indexes:
                    if left==right or costs[left]==costs[right]: continue
                    differences.append(features[left]-features[right]); labels.append(1.0 if costs[left]<costs[right] else 0.0)
        if not differences: raise ValueError("pairwise_ranking_requires_distinct_fit_costs")
        x=np.asarray(differences); y=np.asarray(labels); w=np.zeros(x.shape[1])
        for _ in range(self.iterations):
            p=1.0/(1.0+np.exp(-np.clip(x@w,-30,30))); w-=self.learning_rate*(x.T@(p-y)/len(y))
        self.weights=w; self.runtime_seconds=float(perf_counter()-started); return self

    def score(self, states: Sequence[Sequence[int]], loads: Sequence[Sequence[float]]) -> np.ndarray:
        if self.weights is None: raise RuntimeError("model_not_fitted")
        return design_matrix(states,loads,degree=1)@self.weights
