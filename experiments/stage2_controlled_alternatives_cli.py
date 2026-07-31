"""Run fixed fit-only Stage B residual and margin alternatives from formal truth."""

from __future__ import annotations

import argparse, ast, csv
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from qubit_value_function.stage2_controlled_alternatives import PairwiseLinearRanker, QuadraticResidualModel
from qubit_value_function.stage2_edlp_component_audit import strict_json_dumps
from qubit_value_function.stage2_generalization import fit_normalizers_from_fit_rows, regression_metrics, selection_regret
from qubit_value_function.stage2_models import LoadConditionedMLPModel, LoadConditionedRidgeModel

SEEDS=(11,23,47); TRAIN=(0.85,1.0,1.15); INTERP=(0.925,1.075); EXTRA=(0.8,1.2)

def _read(path:Path)->list[dict[str,object]]:
    with path.open('r',newline='',encoding='utf8') as h:return [dict(r) for r in csv.DictReader(h)]
def _write(path:Path,rows:list[dict[str,object]])->None:
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf8') as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows)
def _pair(v:object)->tuple[int,int]:return tuple(int(x) for x in ast.literal_eval(str(v))) # type: ignore[return-value]
def _array(rows:list[dict[str,object]]):return np.asarray([r['state_bits'] for r in rows],int),np.asarray([r['normalized_load'] for r in rows],float),np.asarray([r['true_cost'] for r in rows],float)
def _slice(row:dict[str,object],training:set[int],fit:set[tuple[float,int]],validation:set[tuple[float,int]])->str:
    load=float(row['load_multiplier']);state=int(row['state_index'])
    if load in TRAIN:return 'train' if (load,state) in fit or (load,state) in validation else 'seen_load_unseen_state'
    if load in INTERP:return 'interpolation_unseen_state' if state not in training else 'interpolation_all_state'
    return 'extrapolation_unseen_state' if state not in training else 'extrapolation_all_state'
def _class(labels:np.ndarray,pred:np.ndarray)->dict[str,object]:
    tp=int(np.sum(labels&pred));fp=int(np.sum(~labels&pred));fn=int(np.sum(labels&~pred));tn=int(np.sum(~labels&~pred));precision=None if tp+fp==0 else tp/(tp+fp);recall=None if tp+fn==0 else tp/(tp+fn);f1=0.0 if precision is None or recall is None or precision+recall==0 else 2*precision*recall/(precision+recall)
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,'f1':f1,'accuracy':(tp+tn)/len(labels),'candidate_count':int(np.sum(pred))}

def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--formal-benchmark-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists():raise RuntimeError(f'refusing_existing_output_directory:{a.output_dir}')
    truth=_read(a.formal_benchmark_dir/'truth_table.csv');splits=_read(a.formal_benchmark_dir/'state_splits.csv')
    split_map={(str(r['generator_pair']),int(r['window_start']),int(r['split_seed'])):r for r in splits}; outputs=[];fits=[];reg=[];classification=[];ranking=[];started=datetime.now(timezone.utc)
    for key,split in split_map.items():
        pair,window,split_seed=key; unit=[{'generator_pair':str(r['generator_pair']),'window_start':int(r['window_start']),'load_multiplier':float(r['load_multiplier']),'state_index':int(r['state_index']),'state_bits':ast.literal_eval(str(r['state_bits'])),'load_vector':ast.literal_eval(str(r['load_vector'])),'true_cost':float(r['true_cost'])} for r in truth if str(_pair(r['generator_pair']))==pair and int(r['window_start'])==window]
        training=set(ast.literal_eval(str(split['training_indices']))); fit_by={float(x['load_multiplier']):set(x['indices']) for x in ast.literal_eval(str(split['fit_indices_by_load']))}; val_by={float(x['load_multiplier']):set(x['indices']) for x in ast.literal_eval(str(split['validation_indices_by_load']))}; fit_pairs={(l,s) for l,ss in fit_by.items() for s in ss};val_pairs={(l,s) for l,ss in val_by.items() for s in ss}
        fit_rows=[r for r in unit if (r['load_multiplier'],r['state_index']) in fit_pairs];val_rows=[r for r in unit if (r['load_multiplier'],r['state_index']) in val_pairs];norm=fit_normalizers_from_fit_rows(fit_rows)
        for r in unit:r['normalized_load']=norm.load.transform(r['load_vector']).tolist();r['slice']=_slice(r,training,fit_pairs,val_pairs)
        fs,fl,fy=_array(fit_rows);vs,vl,vy=_array(val_rows)
        train_threshold={load:min(r['true_cost'] for r in unit if r['load_multiplier']==load and r['state_index'] in training) for load in sorted({r['load_multiplier'] for r in unit})}
        models=[]
        models += [('quadratic_ridge',0,LoadConditionedRidgeModel(degree=2,regularization=1e-3,seed=0),'cost')]
        models += [('quadratic_linear_residual',0,QuadraticResidualModel(correction='linear',seed=0),'cost')]
        for seed in SEEDS:
            models += [('quadratic_mlp_residual',seed,QuadraticResidualModel(correction='mlp',seed=seed),'cost'),('quadratic_simplified_qnn_residual',seed,QuadraticResidualModel(correction='simplified_qnn',seed=seed),'cost')]
            models += [('ridge_margin',seed,LoadConditionedRidgeModel(degree=1,regularization=1e-3,seed=seed),'margin'),('mlp_margin',seed,LoadConditionedMLPModel(hidden_units=8,maxiter=80,learning_rate=.03,seed=seed),'margin')]
        models += [('pairwise_linear_ranker',0,PairwiseLinearRanker(seed=0),'rank')]
        margins=np.asarray([train_threshold[r['load_multiplier']]-r['true_cost'] for r in fit_rows])
        for name,seed,model,kind in models:
            if kind=='rank':model.fit(fs,fl,fy,[r['load_multiplier'] for r in fit_rows]);pred=model.score(np.asarray([r['state_bits'] for r in unit]),np.asarray([r['normalized_load'] for r in unit]));runtime=model.runtime_seconds;fit_mae=None;val_mae=None;status='fixed_pairwise'
            else:
                target=fy if kind=='cost' else margins
                model.fit(fs,fl,target)
                pred=model.predict(np.asarray([r['state_bits'] for r in unit]),np.asarray([r['normalized_load'] for r in unit]))
                runtime=getattr(model,'runtime_seconds',getattr(model,'correction',model).runtime_seconds)
                fit_mae=float(np.mean(np.abs(model.predict(fs,fl)-target)))
                val_target=vy if kind=='cost' else np.asarray([train_threshold[r['load_multiplier']]-r['true_cost'] for r in val_rows])
                val_mae=float(np.mean(np.abs(model.predict(vs,vl)-val_target)))
                status=getattr(model,'fit_status',getattr(model,'correction',model).fit_status)
            fits.append({'generator_pair':pair,'window_start':window,'split_seed':split_seed,'model':name,'seed':seed,'kind':kind,'fit_target_mae':fit_mae,'validation_target_mae':val_mae,'fit_status':status,'runtime_seconds':runtime,'converged':getattr(model,'converged',getattr(model,'correction',model).converged if kind!='rank' else True)})
            for r,value in zip(unit,pred):outputs.append({'generator_pair':pair,'window_start':window,'split_seed':split_seed,'model':name,'seed':seed,'kind':kind,'load_multiplier':r['load_multiplier'],'state_index':r['state_index'],'slice':r['slice'],'true_cost':r['true_cost'],'prediction':float(value),'training_threshold':train_threshold[r['load_multiplier']]})
    for name in sorted({str(r['model']) for r in outputs}):
      for slice_name in sorted({str(r['slice']) for r in outputs}):
       for seed in sorted({int(r['seed']) for r in outputs if r['model']==name}):
        rows=[r for r in outputs if r['model']==name and r['slice']==slice_name and r['seed']==seed]
        if not rows:continue
        if rows[0]['kind']=='cost':
          metrics=regression_metrics([r['true_cost'] for r in rows],[r['prediction'] for r in rows]); regret=selection_regret(true_costs=[r['true_cost'] for r in rows],predicted_costs=[r['prediction'] for r in rows],group_keys=[r['load_multiplier'] for r in rows]);reg.append({'model':name,'seed':seed,'slice':slice_name,**metrics,**regret})
        labels=np.asarray([r['true_cost']<r['training_threshold'] for r in rows]);candidate=np.asarray([r['prediction']<r['training_threshold'] if r['kind']=='cost' else r['prediction']>0 for r in rows]);classification.append({'model':name,'seed':seed,'slice':slice_name,**_class(labels,candidate)})
    a.output_dir.mkdir(parents=True);_write(a.output_dir/'alternative_predictions.csv',outputs);_write(a.output_dir/'alternative_fit_summary.csv',fits);_write(a.output_dir/'alternative_regression_metrics.csv',reg);_write(a.output_dir/'alternative_classification_metrics.csv',classification)
    protocol={'version':'stage2-controlled-alternatives-v1','truth_source':'formal benchmark truth_table only','edlp_calls':0,'residual_labels':'fit-only true_cost - fit-only quadratic prediction','qnn_residual':{'layers':1,'maxiter':20,'seeds':list(SEEDS)},'margin_tau':'training_indices minimum true cost at same load','no_test_selection':True}
    manifest={'started_at_utc':started.isoformat(),'finished_at_utc':datetime.now(timezone.utc).isoformat(),'unit_count':27,'fit_count':len(fits),'prediction_count':len(outputs),'protocol':protocol}
    (a.output_dir/'protocol.json').write_text(strict_json_dumps(protocol),encoding='utf8');(a.output_dir/'manifest.json').write_text(strict_json_dumps(manifest),encoding='utf8');(a.output_dir/'report.md').write_text('# Stage B controlled alternatives\n\nAll models are fixed before testing and reuse formal truth only.\n',encoding='utf8');print(strict_json_dumps(manifest));return 0
if __name__=='__main__':raise SystemExit(main())
