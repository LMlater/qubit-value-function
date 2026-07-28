"""Formal N=16, M=1 fixed-oracle, gate-level Aer-MPS Grover audit."""
from __future__ import annotations
import argparse, csv, json, math, statistics, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from qiskit import transpile
from qiskit_aer import AerSimulator
from experiments.stage1_targeted_best_training_pilot_cli import _load_snapshot, _restore_feasibility_spec, _restore_value_model
from qubit_value_function.fixed_oracle_multishot_diagnostic import _aer_exact_x_probabilities, _phase_oracle_probe, strict_joint_marked_indices, wilson_interval
from qubit_value_function.gate_level_oracle import circuit_resource_summary
from qubit_value_function.sparse_vqc_grover import build_sparse_vqc_grover_circuit, execute_sparse_vqc_grover_mps
from qubit_value_function.stage1_evidence import grover_probability, index_bitstring

def dump_json(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def dump_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def mean_ci(values):
    mean=statistics.mean(values); sd=statistics.stdev(values) if len(values)>1 else 0.0; half=1.959963984540054*sd/math.sqrt(len(values)) if len(values)>1 else 0.0
    return {'mean':mean,'standard_deviation':sd,'ci95_low':mean-half,'ci95_high':mean+half}

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--shots',type=int,default=4096);p.add_argument('--seeds',default='0,1,2,3,4');a=p.parse_args()
    if a.output_dir.exists():raise RuntimeError('refusing_to_overwrite')
    seeds=[int(x) for x in a.seeds.split(',')];
    if a.shots<4096 or len(seeds)<5:raise ValueError('requires_at_least_4096_shots_and_5_seeds')
    started=datetime.now(timezone.utc).isoformat(); snap=_load_snapshot(a.snapshot); pair=tuple(int(x) for x in snap['generator_pair']);model=_restore_value_model(snap); feasibility=_restore_feasibility_spec(snap,generator_pair=pair); threshold=int(snap['best_training_encoded_threshold']); values=[int(r['integer_vqc_value']) for r in snap['state_proxy_table']]; feasible=[bool(r['hard_logic_feasible']) for r in snap['state_proxy_table']];marked=strict_joint_marked_indices(values,feasible,threshold)
    if len(marked)!=1:raise RuntimeError(f'M_not_one:{marked}')
    a.output_dir.mkdir(parents=True); truth=[{'state_index':i,'bitstring':index_bitstring(i),'integer_value':values[i],'hard_logic_feasible':feasible[i],'cost_marked':values[i]<threshold,'joint_marked':i in marked} for i in range(16)]
    phase=_phase_oracle_probe(model,feasibility,threshold)
    if phase['relative_phase_max_error']>1e-8 or phase['auxiliary_zero_probability']<1-1e-10:raise RuntimeError('phase_oracle_validation_failed')
    exact_rows=[]; multishot=[]; resources=[]
    for k in range(4):
        circuit=build_sparse_vqc_grover_circuit(model,encoded_threshold=threshold,iterations=k,feasibility_spec=feasibility); exact,_=_aer_exact_x_probabilities(circuit,num_x_qubits=4); exact_probability=float(exact[marked[0]]); theory=grover_probability(1,k,16)
        if abs(exact_probability-theory)>1e-8:raise RuntimeError(f'exact_probability_mismatch:k={k}')
        exact_rows.append({'k':k,'N':16,'M':1,'marked_index':marked[0],'theoretical_probability':theory,'exact_circuit_probability':exact_probability,'absolute_error':abs(exact_probability-theory)})
        logical=circuit_resource_summary(circuit,decompose_reps=1);compiled=transpile(circuit,AerSimulator(method='matrix_product_state'),optimization_level=1,seed_transpiler=0);resources.append({'k':k,'num_qubits':circuit.num_qubits,'depth':circuit.depth(),'size':circuit.size(),'one_qubit_gate_count':sum(v for g,v in circuit.count_ops().items() if g in {'x','h','s','sdg','z','rz','ry','rx'}),'two_qubit_gate_count':sum(v for g,v in circuit.count_ops().items() if g in {'cx','cz','swap','cp','crz','cry','rzz'}),'logical_depth':logical['depth'],'transpiled_depth':compiled.depth(),'basis':'Aer default','optimization_level':1,'comparator_value_logic_ancilla_count':circuit.num_qubits-4})
        for seed in seeds:
            result=execute_sparse_vqc_grover_mps(circuit,num_x_qubits=4,shots=a.shots,seed=seed); hits=int(result.x_counts.get(index_bitstring(marked[0]),0));low,high=wilson_interval(hits,a.shots);multishot.append({'k':k,'simulator_seed':seed,'shots':a.shots,'marked_index':marked[0],'marked_shots':hits,'observed_probability':hits/a.shots,'theoretical_probability':theory,'exact_circuit_probability':exact_probability,'absolute_error':abs(hits/a.shots-theory),'wilson_ci_low':low,'wilson_ci_high':high,'auxiliary_zero_probability':result.auxiliary_zero_probability,'mps_elapsed_seconds':result.elapsed_seconds})
    aggregates=[]
    for k in range(4):
        group=[r for r in multishot if r['k']==k];stats=mean_ci([r['observed_probability'] for r in group]);aggregates.append({'k':k,'N':16,'M':1,'shots_per_seed':a.shots,'seed_count':len(seeds),'theoretical_probability':grover_probability(1,k,16),'exact_circuit_probability':exact_rows[k]['exact_circuit_probability'],'mean_observed_probability':stats['mean'],'standard_deviation':stats['standard_deviation'],'ci95_low':stats['ci95_low'],'ci95_high':stats['ci95_high'],'absolute_error_mean_vs_theory':abs(stats['mean']-grover_probability(1,k,16))})
    dump_csv(a.output_dir/'truth_table.csv',truth);dump_csv(a.output_dir/'exact_probabilities.csv',exact_rows);dump_csv(a.output_dir/'multishot_runs.csv',multishot);dump_csv(a.output_dir/'aggregate.csv',aggregates);dump_csv(a.output_dir/'circuit_resources.csv',resources)
    summary={'scope':'fixed one-marked-state gate-level Grover circuit executed by Aer MPS; not quantum hardware speedup','scenario_id':snap['scenario_id'],'N':16,'M':1,'threshold':threshold,'marked_index':marked[0],'phase_oracle_validation':phase,'completion_status':'completed','started_at':started,'completed_at':datetime.now(timezone.utc).isoformat()};dump_json(a.output_dir/'summary.json',summary);dump_json(a.output_dir/'manifest.json',{'snapshot':str(a.snapshot),'parameters':{'shots':a.shots,'seeds':seeds,'iterations':[0,1,2,3]},**summary});(a.output_dir/'report.md').write_text('# N=16, M=1 fixed-oracle Grover audit\n\nThe Boolean table marks exactly one state. Exact statevector probabilities and five-seed Aer MPS multishot estimates are tabulated; k=3 remains near the M=1 optimum.\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
