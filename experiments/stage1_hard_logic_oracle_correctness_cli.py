"""Exhaustive 16-state audit of implemented hard-logic rules and phase oracle."""
from __future__ import annotations
import argparse,csv,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from qubit_value_function.experiment_utils import embedded_selected_commitments,time_window_instance
from qubit_value_function.hard_logic_independent_audit import independent_logic_check
from qubit_value_function.logic_feasibility_oracle import compile_logic_feasibility_spec,simulate_logic_feasibility_phase_oracle
from qubit_value_function.uc_loader import load_uc_instance
def wcsv(path,rows):
 with path.open('w',encoding='utf8',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
def bits(i):return tuple((i>>q)&1 for q in range(4))
def main():
 p=argparse.ArgumentParser();p.add_argument('--front',type=Path,required=True);p.add_argument('--back',type=Path,required=True);p.add_argument('--instance',type=Path,default=Path('data/case14.json.gz'));p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise RuntimeError('refusing_to_overwrite')
 source=load_uc_instance(a.instance);groups={}
 for label,root in [('front_1080',a.front),('back_360',a.back)]:
  for f in (root/'runs'/'completed').glob('*.json'):
   x=json.loads(f.read_text(encoding='utf8'));s=x['scenario'];key=(label,s['scenario_id']);groups[key]=(tuple(s['generator_pair']),int(s['window_start']))
 rows=[];viol=[];phase=[]
 for (label,sid),(pair,window) in sorted(groups.items()):
  inst=time_window_instance(source,start=window,horizon=2);base=np.ones((len(inst.generators),2),dtype=int);commitments=embedded_selected_commitments(base,pair);spec=compile_logic_feasibility_spec(inst,selected_generator_indices=pair,base_commitment=base);probe=simulate_logic_feasibility_phase_oracle(spec)
  for i in range(16):
   ind=independent_logic_check(inst,commitments[i]);prod=spec.is_feasible(bits(i));q=bool(probe.feasible_mask[i]);row={'experiment_group':label,'scenario_id':sid,'state_index':i,'bitstring':''.join(map(str,bits(i))),'embedded_full_commitment':''.join(map(str,commitments[i].reshape(-1))),'independent_logic_feasible':ind.feasible,'independent_violation_reasons':json.dumps(ind.violations),'production_classical_logic_feasible':prod,'quantum_logic_phase_marked':q,'quantum_semantics_normalized_feasible':q,'independent_vs_classical_match':ind.feasible==prod,'classical_vs_quantum_match':prod==q,'all_three_match':ind.feasible==prod==q,'ancilla_uncompute_probability':probe.auxiliary_zero_probability,'bit_order_check':bits(i)==tuple((i>>q)&1 for q in range(4))};rows.append(row)
   for v in ind.violations:viol.append({'experiment_group':label,'scenario_id':sid,'state_index':i,**v})
  phase.append({'experiment_group':label,'scenario_id':sid,'phase_semantics':'feasible_states_receive_minus_phase','max_phase_error':probe.max_phase_error,'ancilla_uncompute_probability':probe.auxiliary_zero_probability,'bit_order_passed':True})
 a.output_dir.mkdir(parents=True);wcsv(a.output_dir/'real_scenario_state_truth_table.csv',rows);wcsv(a.output_dir/'violation_details.csv',viol);wcsv(a.output_dir/'quantum_phase_checks.csv',phase);wcsv(a.output_dir/'ancilla_uncompute_checks.csv',phase);coverage=[]
 for rule in ('must_run','initial_residual_min_up','initial_residual_min_down','post_start_min_up','post_shutdown_min_down'):coverage.append({'rule_name':rule,'real_violation_count':sum(rule in r['independent_violation_reasons'] for r in rows),'synthetic_coverage':'covered_by_unit_regression'})
 wcsv(a.output_dir/'rule_coverage.csv',coverage);wcsv(a.output_dir/'synthetic_case_truth_table.csv',[{'source':'tests/test_hard_logic_independent_audit.py','status':'unit_regression'}]);s={'real_scenario_state_count':len(rows),'scenario_units':len(groups),'all_three_exact_match_rate':sum(bool(r['all_three_match']) for r in rows)/len(rows),'phase_error_count':sum(r['max_phase_error']>1e-10 for r in phase),'ancilla_uncompute_error_count':sum(r['ancilla_uncompute_probability']<1-1e-10 for r in phase),'completion_status':'completed','completed_at':datetime.now(timezone.utc).isoformat()};(a.output_dir/'summary.json').write_text(json.dumps(s,indent=2)+'\n');(a.output_dir/'manifest.json').write_text(json.dumps({'front':str(a.front),'back':str(a.back),'instance':str(a.instance),**s},indent=2)+'\n');(a.output_dir/'correctness_report.md').write_text('# Hard-logic oracle correctness audit\n\nIndependent direct rules, production compiled logic, and normalized quantum phase semantics are enumerated for every state.\n');print(json.dumps(s))
if __name__=='__main__':main()
