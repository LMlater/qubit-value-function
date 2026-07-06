from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1_case14_single_period import single_period_instance  # noqa: E402
from experiments.stage1_case14_threshold_oracle_library import run as run_library  # noqa: E402
from qubit_value_function.commitment import all_commitments, commitment_to_bitstring  # noqa: E402
from qubit_value_function.ed import FixedCommitmentEvaluator, startup_cost  # noqa: E402
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def run(
    instance_path: Path,
    library_path: Path,
    results_path: Path,
    period: int,
) -> dict[str, object]:
    if not library_path.exists():
        run_library(
            instance_path=instance_path,
            results_path=library_path,
            period=period,
            max_terms=32,
            target_counts=[1, 2, 5, 10, 20, 32],
        )
    library = json.loads(library_path.read_text(encoding="utf-8"))
    instance = single_period_instance(load_uc_instance(instance_path), period)
    commitments = all_commitments(len(instance.generators), instance.time_horizon)
    bits = commitments.reshape((commitments.shape[0], -1))
    bitstrings = [commitment_to_bitstring(commitment) for commitment in commitments]
    values = np.array([FixedCommitmentEvaluator(instance).evaluate(c).total_cost for c in commitments])
    optimum_idx = int(np.argmin(values))
    optimum_bits = bits[optimum_idx]
    generators = [gen.name for gen in instance.generators]

    analyses = []
    for item in library["library"]:
        success = item.get("first_success")
        if not success:
            analyses.append(
                {
                    "target_count": item["target_count"],
                    "threshold": item["threshold"],
                    "success": False,
                    "reason": "No successful sparse oracle found within the term budget.",
                }
            )
            continue
        labels = values <= float(item["threshold"])
        term_rows = []
        for name in success["selected_names"]:
            subset = parse_monomial_name(name)
            term_rows.append(
                analyze_subset(
                    subset,
                    instance,
                    bits,
                    bitstrings,
                    values,
                    labels,
                    optimum_bits,
                )
            )
        analyses.append(
            {
                "target_count": item["target_count"],
                "threshold": item["threshold"],
                "success": True,
                "term_count": success["term_count"],
                "max_selected_order": success["max_selected_order"],
                "term_rows": term_rows,
            }
        )

    summary = {
        "instance": str(instance_path),
        "method": "physical interpretation of selected sparse phase monomials",
        "period": period,
        "fixed_load_mw": instance.fixed_load,
        "reserve_requirement_mw": float(sum(reserve.amount[0] for reserve in instance.reserves)),
        "generators": generators,
        "optimum_bitstring": bitstrings[optimum_idx],
        "optimum_cost": float(values[optimum_idx]),
        "analyses": analyses,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_monomial_name(name: str) -> tuple[int, ...]:
    if name == "1":
        return ()
    return tuple(int(part[1:]) for part in name.split("*"))


def analyze_subset(
    subset: tuple[int, ...],
    instance,
    bits: np.ndarray,
    bitstrings: list[str],
    values: np.ndarray,
    labels: np.ndarray,
    optimum_bits: np.ndarray,
) -> dict[str, object]:
    if subset:
        support = np.all(bits[:, subset] == 1, axis=1)
    else:
        support = np.ones(bits.shape[0], dtype=bool)
    support_values = values[support]
    support_indices = np.where(support)[0]
    target_support = np.logical_and(support, labels)
    pmax = sum(instance.generators[idx].p_max for idx in subset)
    pmin = sum(instance.generators[idx].p_min for idx in subset)
    startup = startup_cost(instance, _commitment_for_subset(len(instance.generators), subset))
    pmax_cost = sum(instance.generators[idx].cost_usd[-1] for idx in subset)
    reserve_capacity = sum(
        instance.generators[idx].p_max
        for idx in subset
        if instance.generators[idx].reserve_eligibility
    )
    load = instance.fixed_load[0]
    reserve_requirement = sum(reserve.amount[0] for reserve in instance.reserves)
    best_idx = int(support_indices[np.argmin(support_values)]) if support_indices.size else None
    return {
        "monomial": _subset_name(subset),
        "order": len(subset),
        "generators": [instance.generators[idx].name for idx in subset],
        "all_in_optimum_on_set": bool(all(optimum_bits[idx] == 1 for idx in subset)),
        "support_count": int(support.sum()),
        "target_support_count": int(target_support.sum()),
        "target_precision": float(target_support.sum() / support.sum()) if support.sum() else 0.0,
        "target_recall": float(target_support.sum() / labels.sum()) if labels.sum() else 0.0,
        "pmax_mw": float(pmax),
        "pmin_mw": float(pmin),
        "startup_cost": float(startup),
        "pmax_cost": float(pmax_cost),
        "reserve_capacity_mw": float(reserve_capacity),
        "capacity_margin_mw": float(pmax - load),
        "reserve_margin_mw": float(reserve_capacity - reserve_requirement),
        "best_support_bitstring": bitstrings[best_idx] if best_idx is not None else None,
        "best_support_cost": float(values[best_idx]) if best_idx is not None else None,
        "mean_support_cost": float(np.mean(support_values)) if support_values.size else None,
    }


def _commitment_for_subset(num_generators: int, subset: tuple[int, ...]) -> np.ndarray:
    commitment = np.zeros((num_generators, 1), dtype=int)
    for idx in subset:
        commitment[idx, 0] = 1
    return commitment


def _subset_name(subset: tuple[int, ...]) -> str:
    if not subset:
        return "1"
    return "*".join(f"x{i}" for i in subset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("results/stage1_case14_threshold_oracle_library.json"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/stage1_case14_sparse_term_analysis.json"),
    )
    parser.add_argument("--period", type=int, default=0)
    args = parser.parse_args()
    summary = run(args.instance, args.library, args.results, args.period)
    compact = {
        key: value
        for key, value in summary.items()
        if key != "analyses"
    }
    compact["analyses"] = [
        {
            "target_count": item["target_count"],
            "success": item["success"],
            "term_count": item.get("term_count"),
            "terms": [
                {
                    "monomial": row["monomial"],
                    "generators": row["generators"],
                    "support_count": row["support_count"],
                    "target_precision": row["target_precision"],
                    "capacity_margin_mw": row["capacity_margin_mw"],
                }
                for row in item.get("term_rows", [])
            ],
        }
        for item in summary["analyses"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
