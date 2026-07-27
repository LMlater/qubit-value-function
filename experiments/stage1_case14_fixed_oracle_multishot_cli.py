from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from qubit_value_function.fixed_oracle_multishot_diagnostic import FixedOracleDiagnosticError, run_fixed_oracle_multishot  # noqa: E402
def main() -> int:
    parser=argparse.ArgumentParser(description="独立固定-oracle 1024-shot MPS Grover 诊断")
    parser.add_argument("--formal-dir",type=Path,required=True);parser.add_argument("--instance",type=Path,default=Path("data/case14.json.gz"));parser.add_argument("--output-dir",type=Path,default=Path("results/stage1_fixed_oracle_multishot"));parser.add_argument("--seed",type=int,default=20260727)
    args=parser.parse_args()
    try: summary=run_fixed_oracle_multishot(args.formal_dir,instance_path=args.instance,output_dir=args.output_dir,simulator_seed=args.seed)
    except FixedOracleDiagnosticError as error: parser.error(str(error))
    print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False));return 0
if __name__=="__main__": raise SystemExit(main())
