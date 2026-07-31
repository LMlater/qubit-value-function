"""Read-only Stage B attribution join for formal predictions and ED/LP components."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from qubit_value_function.stage2_edlp_component_audit import strict_json_dumps
from qubit_value_function.stage2_error_attribution import PENALTY_DOMINANT_RATIO, error_components, join_formal_and_components, reserve_penalty_group_label


def _read(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf8") as handle: return [dict(row) for row in csv.DictReader(handle)]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="",encoding="utf8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="raise"); writer.writeheader(); writer.writerows(rows)


def _hash(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-benchmark-dir",type=Path,required=True); parser.add_argument("--component-audit-dir",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists(): raise RuntimeError(f"refusing_existing_output_directory:{args.output_dir}")
    truth_path=args.formal_benchmark_dir/'truth_table.csv'; prediction_path=args.formal_benchmark_dir/'predictions.csv'; component_path=args.component_audit_dir/'component_truth_table.csv'
    truth=_read(truth_path); components=_read(component_path); predictions=_read(prediction_path)
    joined_truth=join_formal_and_components(truth,components)
    component_by_key={(str(r['generator_pair']),int(r['window_start']),float(r['load_multiplier']),int(r['state_index'])):r for r in joined_truth}
    joined=[]
    for row in predictions:
        key=(str(list(ast.literal_eval(str(row['generator_pair'])))),int(row['window_start']),float(row['load_multiplier']),int(row['state_index']))
        component=component_by_key.get(key)
        if component is None: raise RuntimeError(f"prediction_component_join_missing:{key}")
        penalty_ratio=(float(component['balance_penalty'])+float(component['reserve_penalty']))/float(component['true_cost'])
        joined.append({**row,'hard_logic_feasible':component['hard_logic_feasible'],'balance_penalty':component['balance_penalty'],'reserve_penalty':component['reserve_penalty'],'penalty_ratio':penalty_ratio,'penalty_dominant':penalty_ratio>=PENALTY_DOMINANT_RATIO})
    groups: dict[tuple[str,str,str,str],list[dict[str,object]]]={}
    for row in joined:
        tags=[('all','all'),('hard_logic',str(row['hard_logic_feasible']).lower()),('balance_penalty','positive' if float(row['balance_penalty'])>0 else 'zero'),('penalty_dominant',str(bool(row['penalty_dominant'])).lower())]
        for dimension,value in tags: groups.setdefault((str(row['model']),str(row['slice']),dimension,value),[]).append(row)
    summary=[]
    for (model,slice_name,dimension,value),rows in sorted(groups.items()):
        if model=='threshold_conditioned_qnn': continue
        truth_values=[float(r['true_cost']) for r in rows]; predicted=[float(r['prediction']) for r in rows]
        summary.append({'model':model,'slice':slice_name,'dimension':dimension,'group':value,'sample_count':len(rows),**error_components(truth_values,predicted)})
    args.output_dir.mkdir(parents=True); _write(args.output_dir/'joined_predictions_with_components.csv',joined); _write(args.output_dir/'error_attribution_summary.csv',summary)
    manifest={'formal_benchmark_truth_sha256':_hash(truth_path),'formal_predictions_sha256':_hash(prediction_path),'component_truth_sha256':_hash(component_path),'formal_rows':len(truth),'component_rows':len(components),'prediction_rows':len(predictions),'joined_prediction_rows':len(joined),'penalty_dominant_ratio':PENALTY_DOMINANT_RATIO,'formal_and_component_evidence_separate':True,'reserve_penalty_group':reserve_penalty_group_label([float(r['reserve_penalty']) for r in components])}
    (args.output_dir/'manifest.json').write_text(strict_json_dumps(manifest),encoding='utf8')
    (args.output_dir/'report.md').write_text('# Stage B error attribution\n\nFormal benchmark predictions are joined to supplemental ED/LP components by the four formal key fields; neither input directory is modified. Full samples remain primary; penalty-dominant rows are sensitivity groups only.\n',encoding='utf8')
    print(strict_json_dumps(manifest)); return 0

if __name__=='__main__': raise SystemExit(main())
