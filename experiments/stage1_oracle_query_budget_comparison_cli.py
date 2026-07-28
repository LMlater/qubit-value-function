"""M=1 oracle-query model diagnostic; no closed-loop search is run."""
from __future__ import annotations
import argparse, csv, json, statistics, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from qubit_value_function.stage1_evidence import grover_probability
from qubit_value_function.stage1_oracle_query_diagnostic import sample_first_hit, uniform_hit_probability

def write_csv(path, rows):
    with path.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--trials',type=int,default=20000);a=p.parse_args()
    if a.output_dir.exists(): raise RuntimeError('refusing_to_overwrite')
    a.output_dir.mkdir(parents=True); started=datetime.now(timezone.utc).isoformat(); rows=[]
    for budget in range(1,17):
        rows.append({'N':16,'M':1,'oracle_query_budget':budget,'grover_theory_probability':grover_probability(1,budget,16),
                     'classical_uniform_with_replacement_probability':uniform_hit_probability(1,16,budget,replacement=True),'classical_uniform_without_replacement_probability':uniform_hit_probability(1,16,budget,replacement=False)})
    hits=[]
    for replacement in (True,False):
        values=[sample_first_hit([3],dimension=16,replacement=replacement,seed=100000+i) for i in range(a.trials)]
        hits.append({'sampling':'with_replacement' if replacement else 'without_replacement','trials':a.trials,'mean_first_hit_queries':statistics.mean(values),'median_first_hit_queries':statistics.median(values),'p95_first_hit_queries':sorted(values)[int(.95*(len(values)-1))]})
    write_csv(a.output_dir/'oracle_query_budget_comparison.csv',rows);write_csv(a.output_dir/'oracle_query_first_hit_distribution.csv',hits)
    summary={'scope':'oracle-query-model diagnostic; Grover values are theoretical and MPS is a classical gate-level simulator, not wall-clock quantum advantage','N':16,'M':1,'marked_index_hidden_from_classical_sampler':True,'completion_status':'completed','started_at':started,'completed_at':datetime.now(timezone.utc).isoformat()}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');(a.output_dir/'manifest.json').write_text(json.dumps({'parameters':vars(a),'started_at':started,'completion_status':'completed'},default=str,indent=2)+'\n',encoding='utf-8');(a.output_dir/'report.md').write_text('# Oracle-query comparison\n\nClassical sampling queries the Boolean predicate after each uniform draw and never receives the target index.\n',encoding='utf-8')
    print(json.dumps(summary));return 0
if __name__=='__main__': raise SystemExit(main())
