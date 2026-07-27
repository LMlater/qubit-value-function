"""Print a read-only integrity audit of the persisted 80-run targeted pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.selected_split_best_training_audit import (  # noqa: E402
    SelectedSplitAuditError,
    audit_targeted_best_training_candidate_pilot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only targeted-pilot consistency audit")
    parser.add_argument("--pilot-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit_targeted_best_training_candidate_pilot(args.pilot_dir)
    except SelectedSplitAuditError as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
