from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.experiment_utils import write_strict_json  # noqa: E402
from qubit_value_function.logic_subspace_scan import (  # noqa: E402
    DEFAULT_MAX_SCAN_QUBITS,
    scan_logic_feasibility_subspaces,
    select_active_logic_subspace,
)
from qubit_value_function.uc_loader import load_uc_instance  # noqa: E402


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("至少需要一个整数")
    return values


def run(
    *,
    instance_path: Path,
    results_path: Path,
    horizons: tuple[int, ...] = (2, 3),
    selected_generator_count: int = 2,
    window_start: int = 0,
    minimum_feasible_states: int = 8,
    max_scan_qubits: int = DEFAULT_MAX_SCAN_QUBITS,
) -> dict[str, object]:
    source = load_uc_instance(instance_path)
    rows = scan_logic_feasibility_subspaces(
        source,
        horizons=horizons,
        selected_generator_count=selected_generator_count,
        window_start=window_start,
        max_scan_qubits=max_scan_qubits,
    )
    selected = select_active_logic_subspace(
        rows,
        minimum_feasible_states=minimum_feasible_states,
    )
    active_rows = [row for row in rows if row.has_active_filter]
    payload = {
        "method": "case14 logic-only selected-subspace scan",
        "source_instance": str(instance_path),
        "window_start": int(window_start),
        "horizons": [int(value) for value in horizons],
        "selected_generator_count": int(selected_generator_count),
        "minimum_feasible_states": int(minimum_feasible_states),
        "max_scan_qubits": int(max_scan_qubits),
        "selection_policy": {
            "uses_cost": False,
            "uses_ed_lp": False,
            "uses_vqc": False,
            "uses_bbht": False,
            "uses_hidden_optimum": False,
            "requirements": [
                "always_infeasible is false",
                "retained forbidden patterns > 0",
                "both logic-feasible and logic-infeasible states exist",
                "minimum feasible training-state count is met",
                "selected_generator_count * horizon <= max_scan_qubits",
            ],
            "ranking": [
                "prefer feasible ratio in [0.25, 0.75]",
                "maximize excluded logic-infeasible states",
                "minimize retained forbidden patterns",
                "prefer smaller horizon",
                "generator-index lexicographic tie break",
            ],
        },
        "num_scanned_subspaces": int(len(rows)),
        "num_active_subspaces": int(len(active_rows)),
        "selected_subspace": selected.as_dict(),
        "rows": [row.as_dict() for row in rows],
    }
    write_strict_json(results_path, payload)
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="扫描 case14 中会实际激活硬启停逻辑 oracle 的两机组子空间"
    )
    parser.add_argument("--instance", type=Path, default=Path("data/case14.json.gz"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/scan_case14_logic_feasibility_subspaces.json"),
    )
    parser.add_argument("--horizons", type=_parse_int_tuple, default=(2, 3))
    parser.add_argument("--selected-generator-count", type=int, default=2)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--minimum-feasible-states", type=int, default=8)
    parser.add_argument(
        "--max-scan-qubits",
        type=int,
        default=DEFAULT_MAX_SCAN_QUBITS,
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    payload = run(
        instance_path=args.instance,
        results_path=args.results,
        horizons=tuple(args.horizons),
        selected_generator_count=args.selected_generator_count,
        window_start=args.window_start,
        minimum_feasible_states=args.minimum_feasible_states,
        max_scan_qubits=args.max_scan_qubits,
    )
    selected = payload["selected_subspace"]
    print(
        json.dumps(
            {
                "num_scanned_subspaces": payload["num_scanned_subspaces"],
                "num_active_subspaces": payload["num_active_subspaces"],
                "max_scan_qubits": payload["max_scan_qubits"],
                "selected_generator_indices": selected["selected_generator_indices"],
                "selected_generator_names": selected["selected_generator_names"],
                "horizon": selected["horizon"],
                "num_forbidden_patterns": selected["num_forbidden_patterns"],
                "logic_feasible_count": selected["logic_feasible_count"],
                "logic_infeasible_count": selected["logic_infeasible_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
