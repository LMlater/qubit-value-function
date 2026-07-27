from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from qubit_value_function.closed_loop_oracle_grover_audit import OracleGroverAuditError, audit_oracle_grover  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="只读 formal joint-oracle/Grover trace 审计")
    parser.add_argument("--formal-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try: summary = audit_oracle_grover(args.formal_dir, output_dir=args.output_dir)
    except OracleGroverAuditError as error: parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
