"""Read-only reconstruction of persisted closed-loop threshold traces."""
from __future__ import annotations
import argparse,csv,json,statistics
from datetime import datetime,timezone
from pathlib import Path
def csvout(p,rows):
 with p.open('w',encoding='utf8',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--front',type=Path,required=True);p.add_argument('--back',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise RuntimeError('refusing_to_overwrite')
 runs=[];events=[];oracles=[]
 for group,root in [('front_1080',a.front),('back_360',a.back)]:
  for f in (root/'runs'/'completed').glob('*.json'):
   x=json.loads(f.read_text(encoding='utf8'));r=x['result'];tr=r.get('trial_trace',[]);acc=[z for z in tr if z.get('accepted_update')];first=acc[0] if acc else None;rid=x['run_id'];runs.append({'experiment_group':group,'run_id':rid,'method':x['method'],'scenario_id':x['scenario']['scenario_id'],'seed':x['run_spec']['run_seed'],'initial_incumbent_state':r['initial_incumbent_index'],'initial_incumbent_true_cost':r['initial_incumbent_true_cost'],'final_incumbent_state':r['final_incumbent_index'],'final_incumbent_true_cost':r['final_incumbent_true_cost'],'accepted_update_count':len(acc),'first_accepted_state':first.get('measured_index',first.get('candidate_index')) if first else None,'first_accepted_trial':first.get('trial_number',first.get('proposal_number')) if first else None,'continued_after_first_acceptance':bool(first and tr.index(first)<len(tr)-1),'total_trials':len(tr),'final_stop_reason':r.get('stop_reason')})
   for step,z in enumerate(acc,1):events.append({'experiment_group':group,'run_id':rid,'method':x['method'],'scenario_id':x['scenario']['scenario_id'],'update_step':step,'accepted_state':z.get('measured_index',z.get('candidate_index')),'accepted_true_cost':z.get('exact_cost'),'previous_threshold':z.get('true_threshold_before'),'new_threshold':z.get('true_threshold_after'),'previous_threshold_fixed':z.get('encoded_threshold_before'),'new_threshold_fixed':z.get('encoded_threshold_after'),'improvement_amount':(z.get('true_threshold_before') or 0)-(z.get('true_threshold_after') or 0),'continued_after_acceptance':tr.index(z)<len(tr)-1,'subsequent_proposals':len(tr)-tr.index(z)-1,'final_stop_reason':r.get('stop_reason')})
   for z in tr:oracles.append({'experiment_group':group,'run_id':rid,'scenario_id':x['scenario']['scenario_id'],'method':x['method'],'threshold_fixed':z.get('encoded_threshold_before'),'threshold_float':z.get('true_threshold_before'),'oracle_instance_id':f"{x['scenario']['scenario_id']}:{z.get('encoded_threshold_before')}",'candidate_state':z.get('measured_index',z.get('candidate_index')),'joint_marked_observed':z.get('joint_marked'),'stop_reason':z.get('stop_reason_after_trial',z.get('stop_reason_after_proposal'))})
 a.output_dir.mkdir(parents=True);csvout(a.output_dir/'run_summary.csv',runs);csvout(a.output_dir/'accepted_update_events.csv',events);csvout(a.output_dir/'threshold_oracle_instances.csv',oracles)
 methods=[]
 for g in ('front_1080','back_360'):
  for m in sorted({r['method'] for r in runs if r['experiment_group']==g}):
   q=[r for r in runs if r['experiment_group']==g and r['method']==m];methods.append({'experiment_group':g,'method':m,'runs':len(q),'run_success_rate':sum(r['accepted_update_count']>0 for r in q)/len(q),'mean_accepted_updates_per_run':statistics.mean(r['accepted_update_count'] for r in q),'median_accepted_updates_per_run':statistics.median(r['accepted_update_count'] for r in q),'continued_after_first_acceptance_rate':sum(r['continued_after_first_acceptance'] for r in q)/len(q)})
 csvout(a.output_dir/'per_method_summary.csv',methods);front=[r for r in methods if r['experiment_group']=='front_1080'];back=[r for r in methods if r['experiment_group']=='back_360'];(a.output_dir/'front_1080_summary.json').write_text(json.dumps(front,indent=2));(a.output_dir/'back_360_summary.json').write_text(json.dumps(back,indent=2));(a.output_dir/'representative_run_traces.md').write_text('# Representative traces\n\nSee CSV rows for all four selected-split positive scenarios and every seed.\n');s={'runs':len(runs),'accepted_updates':len(events),'completion_status':'completed','completed_at':datetime.now(timezone.utc).isoformat()};(a.output_dir/'manifest.json').write_text(json.dumps({'front':str(a.front),'back':str(a.back),**s},indent=2));print(json.dumps(s))
if __name__=='__main__':main()
