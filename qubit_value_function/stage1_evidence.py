"""Small, pure calculations shared by the Stage-A evidence audits.

These functions deliberately contain no solver, model fitting, or search code:
they make the post-run classification rules explicit and testable.
"""

from __future__ import annotations

import math
from typing import Sequence


def index_bitstring(index: int, width: int = 4) -> str:
    """Return the persisted little-endian search-bit order used by snapshots."""
    return "".join(str((int(index) >> bit) & 1) for bit in range(int(width)))


def confusion_metrics(marked: Sequence[bool], truth: Sequence[bool]) -> dict[str, float | int | None]:
    """Return oracle classification metrics without inventing zero-denominator values."""
    if len(marked) != len(truth):
        raise ValueError("marked/truth lengths differ")
    tp = sum(bool(m) and bool(t) for m, t in zip(marked, truth))
    fp = sum(bool(m) and not bool(t) for m, t in zip(marked, truth))
    fn = sum(not bool(m) and bool(t) for m, t in zip(marked, truth))
    tn = sum(not bool(m) and not bool(t) for m, t in zip(marked, truth))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    false_negative_rate = fn / (fn + tp) if fn + tp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "specificity": specificity, "false_negative_rate": false_negative_rate, "f1": f1, "states": len(truth)}


def classify_unreachable(*, true_improvement: Sequence[bool], cost_marked: Sequence[bool], joint_marked: Sequence[bool], logic_feasible: Sequence[bool]) -> list[str]:
    """Give every non-joint-reachable scenario its data-derived explanation labels."""
    if not (len(true_improvement) == len(cost_marked) == len(joint_marked) == len(logic_feasible)):
        raise ValueError("classification lengths differ")
    labels: list[str] = []
    improving = [i for i, item in enumerate(true_improvement) if item]
    cost_true = [i for i in improving if cost_marked[i]]
    cost_false = [i for i in improving if not cost_marked[i]]
    logic_excluded = [i for i in cost_true if not logic_feasible[i]]
    if not improving:
        labels.append("A_no_nontraining_true_strict_improvement")
    if improving and not cost_true:
        labels.append("B_true_improvement_all_cost_false_negative")
    if any(cost_marked) and not cost_true:
        labels.append("C_cost_marked_but_no_true_improvement")
    if logic_excluded:
        labels.append("D_true_improvement_cost_marked_but_hard_logic_excluded")
    if any(joint_marked[i] and true_improvement[i] for i in range(len(true_improvement))):
        labels.append("E_joint_marks_nontraining_true_improvement")
    if any(joint_marked[i] and not true_improvement[i] for i in range(len(true_improvement))):
        labels.append("F_joint_false_positive")
    return labels


def subspace_optimum(costs: Sequence[float | None], *, discovered_index: int | None) -> dict[str, object]:
    """Audit only the supplied 16-state subspace, treating unavailable values as non-optimal."""
    finite = [(index, float(value)) for index, value in enumerate(costs) if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"optimum_indices": [], "optimum_cost": None, "discovered_index": discovered_index,
                "discovered_is_optimum": None, "absolute_gap": None, "relative_gap": None}
    optimum_cost = min(value for _, value in finite)
    optimum_indices = [index for index, value in finite if math.isclose(value, optimum_cost, abs_tol=1e-9, rel_tol=1e-12)]
    discovered_cost = costs[discovered_index] if discovered_index is not None else None
    gap = float(discovered_cost) - optimum_cost if discovered_cost is not None else None
    return {"optimum_indices": optimum_indices, "optimum_cost": optimum_cost, "discovered_index": discovered_index,
            "discovered_is_optimum": discovered_index in optimum_indices if discovered_index is not None else None,
            "absolute_gap": gap, "relative_gap": gap / abs(optimum_cost) if gap is not None and optimum_cost else None}


def grover_probability(marked_count: int, iterations: int, dimension: int = 16) -> float:
    if not 0 <= int(marked_count) <= int(dimension) or int(iterations) < 0:
        raise ValueError("invalid Grover parameters")
    theta = math.asin(math.sqrt(int(marked_count) / int(dimension)))
    return math.sin((2 * int(iterations) + 1) * theta) ** 2
