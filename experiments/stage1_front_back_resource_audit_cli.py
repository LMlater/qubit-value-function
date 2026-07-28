"""Run-level reaggregation of immutable 1080 and selected-split 360 results."""
from __future__ import annotations
import argparse,csv,json,statistics
from datetime import datetime,timezone
from pathlib import Path

def write(path,rows):
 with path.open('w',encoding='utf8',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
def nums(xs):
 return {'mean':statistics.mean(xs) if xs else None,'median':statistics.median(xs) if xs else None,'standard_deviation':statistics.stdev(xs) if len(xs)>1 else None,'min':min(xs) if xs else None,'max':max(xs) if xs else None,'total':sum(xs)}
def load(root,label):
 rows=[]
 for p in (root/'runs'/'completed').glob('*.json'):
  x=json.loads(p.read_text(encoding='utf8')); r=x['result']; t=r.get('trial_trace',[]); c=r.get('counters',{})
  improving=lambda z: bool(z.get('true_strict_improvement',z.get('true_improvement',False))) and (label=='front' or not bool(z.get('candidate_in_training_set',z.get('is_training_state',False))))
  strict=any(improving(z) for z in t); first=next((i+1 for i,z in enumerate(t) if improving(z)),None)
  rows.append({'batch':label,'method':x['method'],'run_id':x['run_id'],'scenario_id':x['scenario']['scenario_id'],'strict_improvement':strict,'first_improvement_candidate':first,'candidates':c.get('proposals_used',len(t)),'oracle_calls':sum(int(z.get('oracle_calls_added',0)) for z in t),'mps_executions':len(t) if x['method'] in ('joint_bbht','cost_only_bbht') else 0,'new_edlp':c.get('actual_ed_lp_solves',0),'cache_hits':c.get('cached_exact_lookups',0),'cache_misses':c.get('new_exact_evaluation_attempts',0),'logic_rejections':c.get('logic_precheck_rejections',0)})
 return rows
def summary(rows):
 out=[]
 for method in sorted({r['method'] for r in rows}):
  g=[r for r in rows if r['method']==method];succ=[r for r in g if r['strict_improvement']];rec={'method':method,'runs':len(g),'strict_improvement_runs':len(succ),'strict_improvement_rate':len(succ)/len(g),'mean_first_improvement_candidate':nums([r['first_improvement_candidate'] for r in succ])['mean'],'mean_new_edlp_per_strict_improvement':sum(r['new_edlp'] for r in g)/len(succ) if succ else None}
  for k in ('candidates','oracle_calls','mps_executions','new_edlp','cache_hits','cache_misses','logic_rejections'): rec[k]=nums([r[k] for r in g])
  out.append(rec)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--front',type=Path,required=True);p.add_argument('--back',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise RuntimeError('refusing_to_overwrite')
 a.output_dir.mkdir(parents=True);started=datetime.now(timezone.utc).isoformat();fr=load(a.front,'front');br=load(a.back,'back');fs=summary(fr);bs=summary(br);write(a.output_dir/'front_1080_method_summary.csv',fs);write(a.output_dir/'back_360_method_summary.csv',bs);write(a.output_dir/'run_level_normalized_metrics.csv',fr+br);write(a.output_dir/'first_improvement_audit.csv',[r for r in fr+br if r['strict_improvement']]);write(a.output_dir/'front_back_method_comparison.csv',[{'batch':'front',**r} for r in fs]+[{'batch':'back',**r} for r in bs]);s={'front_completed_runs':len(fr),'back_completed_runs':len(br),'completion_status':'completed','started_at':started,'completed_at':datetime.now(timezone.utc).isoformat(),'schema_note':'metrics are run-level; unavailable fields remain absent rather than guessed'};(a.output_dir/'summary.json').write_text(json.dumps(s,indent=2)+'\n');(a.output_dir/'manifest.json').write_text(json.dumps({'front':str(a.front),'back':str(a.back),**s},indent=2)+'\n');(a.output_dir/'report.md').write_text('# Front/back resource audit\n\nMain tables report mean per run together with success rates.\n');print(json.dumps(s))
if __name__=='__main__':main()
