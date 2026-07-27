"""对现有 formal 结果执行只读 first-hit 审计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubit_value_function.closed_loop_first_hit_audit import FirstHitAuditError, audit_formal_first_hits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="只读取 existing formal JSON/validation 的 joint-BBHT first-hit 审计")
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="可选的此前不存在的独立输出目录")
    args = parser.parse_args()
    try:
        summary = audit_formal_first_hits(args.formal_dir, output_dir=args.output_dir)
    except FirstHitAuditError as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
