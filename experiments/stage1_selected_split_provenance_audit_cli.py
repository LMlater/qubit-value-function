"""Generate a provenance correction sidecar for immutable selected-split results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.selected_split_provenance_audit import (  # noqa: E402
    ProvenanceAuditError,
    audit_selected_split_provenance,
)


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only selected-split provenance audit")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--actual-execution-head", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        correction = audit_selected_split_provenance(
            result_dir=args.result_dir,
            actual_execution_head=args.actual_execution_head,
            output_dir=args.output_dir,
            audit_code_head=_git_head(),
        )
    except ProvenanceAuditError as error:
        parser.error(str(error))
    print(json.dumps(correction, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
