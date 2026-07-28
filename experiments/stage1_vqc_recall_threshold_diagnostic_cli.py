"""Posthoc fixed-point margin recall diagnostic over truth-audit rows."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from qubit_value_function.stage1_evidence import confusion_metrics
def main():
 p=argparse.ArgumentParser();p.add_argument('--truth-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise RuntimeError('refusing_to_overwrite')
 rows=list(csv.DictReader((a.truth_dir/'scenario_state_truth_table.csv').open(encoding='utf8')));rows=[r for r in rows if r['in_nontraining_set']=='True'];out=[]
 for delta in range(-4,5):
  for oracle in ('cost','joint'):
   marked=[int(r['predicted_fixed_point_cost'])<int(r['encoded_threshold'])+delta and (oracle=='cost' or r['hard_logic_feasible']=='True') for r in rows];m=confusion_metrics(marked,[r['true_improvement']=='True' for r in rows]);out.append({'delta_code':delta,'oracle':oracle,**m,'scenario_coverage':len({r['scenario_id'] for r,x in zip(rows,marked) if x and r['true_improvement']=='True'})/12,'mean_marked_per_scenario':sum(marked)/12,'fixed_point_margin_cost_units':delta*250.0})
 a.output_dir.mkdir(parents=True)
 with (a.output_dir/'margin_metrics.csv').open('w',encoding='utf8',newline='') as h:w=csv.DictWriter(h,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 s={'scope':'posthoc diagnostic using selected-split nontraining truth; not independent generalization evidence','delta_codes':list(range(-4,5)),'completion_status':'completed'}
 for n in ('summary.json','manifest.json'): (a.output_dir/n).write_text(json.dumps(s,indent=2)+'\n')
 (a.output_dir/'report.md').write_text('# VQC margin recall diagnostic\n\nThis is explicitly a posthoc diagnostic, not a new unbiased generalization result.\n');print(json.dumps(s))
if __name__=='__main__':main()
