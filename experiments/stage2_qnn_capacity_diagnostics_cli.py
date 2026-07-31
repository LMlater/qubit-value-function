"""Isolated synthetic-fit and 18-fit-sample QNN capacity diagnostics."""
from __future__ import annotations
import argparse, ast, csv
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from qubit_value_function.stage2_controlled_alternatives import QuadraticResidualModel
from qubit_value_function.stage2_edlp_component_audit import strict_json_dumps
from qubit_value_function.stage2_generalization import fit_normalizers_from_fit_rows
from qubit_value_function.stage2_models import FullExpectationQNNModel,SimplifiedExpectationQNNModel

def _read(p:Path):
 with p.open('r',newline='',encoding='utf8') as h:return [dict(r) for r in csv.DictReader(h)]
def _write(p:Path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf8') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def _bits():return np.asarray([[(i>>b)&1 for b in range(4)] for i in range(16)],int)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--formal-benchmark-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise RuntimeError(f'refusing_existing_output_directory:{a.output_dir}')
 rows=[];bits=_bits();loads=np.zeros((16,2));targets={'linear':bits@np.asarray([1.,2.,3.,4.]),'quadratic_pseudoboolean':bits@np.asarray([1.,2.,3.,4.])+2*bits[:,0]*bits[:,1]-3*bits[:,2]*bits[:,3],'nonlinear':np.sin(np.pi*(bits[:,0]+bits[:,1]/2))+bits[:,2]*bits[:,3]*3}
 for target_name,target in targets.items():
  for name,cls in [('simplified',SimplifiedExpectationQNNModel),('full',FullExpectationQNNModel)]:
   model=cls(layers=1,head_regularization=.5,theta_regularization=1e-5,maxiter=200,seed=11).fit(bits,loads,target);pred=model.predict(bits,loads);rows.append({'diagnostic':'synthetic','target':target_name,'model':name,'seed':11,'initial_mae':float(np.mean(np.abs(target-target.mean()))),'final_fit_mae':float(np.mean(np.abs(target-pred))),'objective_initial':model.objective_initial,'objective_final':model.objective_final,'converged':model.converged,'fit_status':model.fit_status,'runtime_seconds':model.runtime_seconds})
 truth=_read(a.formal_benchmark_dir/'truth_table.csv');splits=_read(a.formal_benchmark_dir/'state_splits.csv');split=next(r for r in splits if r['generator_pair']=='(0, 1)' and int(r['window_start'])==0 and int(r['split_seed'])==1);fit_by={float(x['load_multiplier']):set(x['indices']) for x in ast.literal_eval(split['fit_indices_by_load'])};unit=[]
 for r in truth:
  if str(ast.literal_eval(r['generator_pair']))!='(0, 1)' or int(r['window_start'])!=0:continue
  row={'load_multiplier':float(r['load_multiplier']),'state_index':int(r['state_index']),'state_bits':ast.literal_eval(r['state_bits']),'load_vector':ast.literal_eval(r['load_vector']),'true_cost':float(r['true_cost'])};unit.append(row)
 fit=[r for r in unit if r['load_multiplier'] in fit_by and r['state_index'] in fit_by[r['load_multiplier']]];norm=fit_normalizers_from_fit_rows(fit)
 for r in unit:r['normalized_load']=norm.load.transform(r['load_vector']).tolist()
 fs=np.asarray([r['state_bits'] for r in fit],int);fl=np.asarray([r['normalized_load'] for r in fit]);fy=np.asarray([r['true_cost'] for r in fit])
 for seed in (11,23,47):
  direct=SimplifiedExpectationQNNModel(layers=1,head_regularization=.5,theta_regularization=1e-5,maxiter=200,seed=seed).fit(fs,fl,fy);residual=QuadraticResidualModel(correction='simplified_qnn',seed=seed).fit(fs,fl,fy)
  for name,model in [('direct_simplified_qnn',direct),('qnn_residual',residual)]:
   report=model if name=='direct_simplified_qnn' else model.correction
   pred=model.predict(fs,fl);rows.append({'diagnostic':'fit_memory','target':'case14_pair01_window0_split1_fit18','model':name,'seed':seed,'initial_mae':None,'final_fit_mae':float(np.mean(np.abs(fy-pred))),'objective_initial':report.objective_initial,'objective_final':report.objective_final,'converged':report.converged,'fit_status':report.fit_status,'runtime_seconds':report.runtime_seconds})
 a.output_dir.mkdir(parents=True);_write(a.output_dir/'qnn_capacity_diagnostics.csv',rows);payload={'version':'stage2-qnn-capacity-diagnostics-v1','uses_formal_test_metrics':False,'synthetic_state_count':16,'memory_unit':'pair(0,1),window0,split1,fit18','maxiter':200,'rows':len(rows)};(a.output_dir/'protocol.json').write_text(strict_json_dumps(payload),encoding='utf8');(a.output_dir/'report.md').write_text('# QNN capacity diagnostics\n\nThis is not Case14 generalization evidence.\n',encoding='utf8');print(strict_json_dumps(payload))
if __name__=='__main__':main()
