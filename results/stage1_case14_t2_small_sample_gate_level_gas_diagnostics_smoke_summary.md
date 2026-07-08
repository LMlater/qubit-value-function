# GAS Diagnostic Ablations Smoke Summary

## Purpose

This smoke diagnostic investigates why the formal exclude-hidden sweep has modest exact optimum recovery. It compares exact success, near-optimal success, same-budget random baselines, measurement policies, candidate budgets, refit policies, learners, and a diagnostic-only hidden-perfect oracle mode.

## Formal Result Context

- formal total runs: 640
- exact success over ok runs: 35.5%
- 4q grouped exact-success range: 35%-60%
- 6q grouped exact-success range: 0%-25%

## Smoke Overall

- total runs: 102
- ok runs: 96
- error runs: 6
- learned exact success: 46.7%
- learned within 1% success: 46.7%
- learned within 3% success: 46.7%
- learned within 5% success: 46.7%


Resource-limit/error runs retained in JSON/CSV: 'Number of qubits (30) in small_sample_gate_level_tau_0 is greater than maximum (29) in the coupling_map'; 'Number of qubits (30) in small_sample_gate_level_tau_4 is greater than maximum (29) in the coupling_map'; 'Number of qubits (30) in small_sample_gate_level_tau_5 is greater than maximum (29) in the coupling_map'; 'Number of qubits (30) in small_sample_gate_level_tau_9 is greater than maximum (29) in the coupling_map'

## Grouped Diagnostic Table

| selected_generators | qubits | train_n | measurement | top_k | refit | learner | oracle_mode | exact | within_3pct | random_exact | avg_ed_lp |
|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|
| 0,1,2 | 6 | 8 | classical_hidden_perfect_diagnostic | 0 | none | hidden_reference | hidden_perfect_diagnostic | 100.0% | 100.0% | n/a | 10.000 |
| 0,1,2 | 6 | 8 | shot_batch | 1 | accepted | mismatch | learned | 0.0% | 0.0% | 0.062 | 12.000 |
| 0,1,2 | 6 | 8 | shot_batch | 1 | accepted | pairwise_ranking | learned | 0.0% | 0.0% | 0.062 | 12.000 |
| 0,1,2 | 6 | 8 | shot_batch | 1 | none | mismatch | learned | 0.0% | 0.0% | 0.062 | 12.000 |
| 0,1,2 | 6 | 8 | shot_batch | 1 | none | pairwise_ranking | learned | 0.0% | 0.0% | 0.062 | 12.000 |
| 0,1,2 | 6 | 8 | shot_batch | 3 | accepted | mismatch | learned | 0.0% | 0.0% | 0.125 | 16.000 |
| 0,1,2 | 6 | 8 | shot_batch | 3 | accepted | pairwise_ranking | learned | n/a | n/a | n/a | n/a |
| 0,1,2 | 6 | 8 | shot_batch | 3 | none | mismatch | learned | 0.0% | 0.0% | 0.125 | 16.000 |
| 0,1,2 | 6 | 8 | shot_batch | 3 | none | pairwise_ranking | learned | 0.0% | 0.0% | 0.125 | 16.000 |
| 0,1,2 | 6 | 8 | shot_batch | 5 | accepted | mismatch | learned | 50.0% | 50.0% | 0.188 | 20.000 |
| 0,1,2 | 6 | 8 | shot_batch | 5 | accepted | pairwise_ranking | learned | n/a | n/a | n/a | n/a |
| 0,1,2 | 6 | 8 | shot_batch | 5 | none | mismatch | learned | 50.0% | 50.0% | 0.188 | 20.000 |
| 0,1,2 | 6 | 8 | shot_batch | 5 | none | pairwise_ranking | learned | 0.0% | 0.0% | 0.188 | 20.000 |
| 0,1,2 | 6 | 8 | single_shot | 1 | accepted | mismatch | learned | 0.0% | 0.0% | 0.062 | 12.000 |
| 0,1,2 | 6 | 8 | single_shot | 1 | accepted | pairwise_ranking | learned | 0.0% | 0.0% | 0.062 | 12.000 |
| 0,1,2 | 6 | 8 | single_shot | 1 | none | mismatch | learned | 0.0% | 0.0% | 0.062 | 12.000 |
| 0,1,2 | 6 | 8 | single_shot | 1 | none | pairwise_ranking | learned | 0.0% | 0.0% | 0.062 | 12.000 |
| 0,5 | 4 | 4 | classical_hidden_perfect_diagnostic | 0 | none | hidden_reference | hidden_perfect_diagnostic | 100.0% | 100.0% | n/a | 6.000 |
| 0,5 | 4 | 4 | shot_batch | 1 | accepted | mismatch | learned | 50.0% | 50.0% | 0.156 | 6.500 |
| 0,5 | 4 | 4 | shot_batch | 1 | accepted | pairwise_ranking | learned | 50.0% | 50.0% | 0.156 | 6.500 |
| 0,5 | 4 | 4 | shot_batch | 1 | none | mismatch | learned | 50.0% | 50.0% | 0.156 | 6.500 |
| 0,5 | 4 | 4 | shot_batch | 1 | none | pairwise_ranking | learned | 50.0% | 50.0% | 0.156 | 6.500 |
| 0,5 | 4 | 4 | shot_batch | 3 | accepted | mismatch | learned | 100.0% | 100.0% | 0.375 | 10.000 |
| 0,5 | 4 | 4 | shot_batch | 3 | accepted | pairwise_ranking | learned | 100.0% | 100.0% | 0.344 | 9.500 |
| 0,5 | 4 | 4 | shot_batch | 3 | none | mismatch | learned | 100.0% | 100.0% | 0.375 | 10.000 |
| 0,5 | 4 | 4 | shot_batch | 3 | none | pairwise_ranking | learned | 100.0% | 100.0% | 0.406 | 10.500 |
| 0,5 | 4 | 4 | shot_batch | 5 | accepted | mismatch | learned | 100.0% | 100.0% | 0.438 | 11.000 |
| 0,5 | 4 | 4 | shot_batch | 5 | accepted | pairwise_ranking | learned | 100.0% | 100.0% | 0.500 | 12.000 |
| 0,5 | 4 | 4 | shot_batch | 5 | none | mismatch | learned | 100.0% | 100.0% | 0.500 | 12.000 |
| 0,5 | 4 | 4 | shot_batch | 5 | none | pairwise_ranking | learned | 100.0% | 100.0% | 0.500 | 12.000 |
| 0,5 | 4 | 4 | single_shot | 1 | accepted | mismatch | learned | 0.0% | 0.0% | 0.156 | 6.500 |
| 0,5 | 4 | 4 | single_shot | 1 | accepted | pairwise_ranking | learned | 0.0% | 0.0% | 0.188 | 7.000 |
| 0,5 | 4 | 4 | single_shot | 1 | none | mismatch | learned | 0.0% | 0.0% | 0.188 | 7.000 |
| 0,5 | 4 | 4 | single_shot | 1 | none | pairwise_ranking | learned | 0.0% | 0.0% | 0.188 | 7.000 |
| 0,5 | 4 | 8 | classical_hidden_perfect_diagnostic | 0 | none | hidden_reference | hidden_perfect_diagnostic | 100.0% | 100.0% | n/a | 10.000 |
| 0,5 | 4 | 8 | shot_batch | 1 | accepted | mismatch | learned | 50.0% | 50.0% | 0.125 | 10.000 |
| 0,5 | 4 | 8 | shot_batch | 1 | accepted | pairwise_ranking | learned | 50.0% | 50.0% | 0.125 | 10.000 |
| 0,5 | 4 | 8 | shot_batch | 1 | none | mismatch | learned | 50.0% | 50.0% | 0.125 | 10.000 |
| 0,5 | 4 | 8 | shot_batch | 1 | none | pairwise_ranking | learned | 50.0% | 50.0% | 0.094 | 9.500 |
| 0,5 | 4 | 8 | shot_batch | 3 | accepted | mismatch | learned | 100.0% | 100.0% | 0.312 | 13.000 |
| 0,5 | 4 | 8 | shot_batch | 3 | accepted | pairwise_ranking | learned | 100.0% | 100.0% | 0.250 | 12.000 |
| 0,5 | 4 | 8 | shot_batch | 3 | none | mismatch | learned | 100.0% | 100.0% | 0.281 | 12.500 |
| 0,5 | 4 | 8 | shot_batch | 3 | none | pairwise_ranking | learned | 100.0% | 100.0% | 0.250 | 12.000 |
| 0,5 | 4 | 8 | shot_batch | 5 | accepted | mismatch | learned | 100.0% | 100.0% | 0.375 | 14.000 |
| 0,5 | 4 | 8 | shot_batch | 5 | accepted | pairwise_ranking | learned | 100.0% | 100.0% | 0.375 | 14.000 |
| 0,5 | 4 | 8 | shot_batch | 5 | none | mismatch | learned | 100.0% | 100.0% | 0.375 | 14.000 |
| 0,5 | 4 | 8 | shot_batch | 5 | none | pairwise_ranking | learned | 100.0% | 100.0% | 0.312 | 13.000 |
| 0,5 | 4 | 8 | single_shot | 1 | accepted | mismatch | learned | 0.0% | 0.0% | 0.156 | 10.500 |
| 0,5 | 4 | 8 | single_shot | 1 | accepted | pairwise_ranking | learned | 0.0% | 0.0% | 0.125 | 10.000 |
| 0,5 | 4 | 8 | single_shot | 1 | none | mismatch | learned | 0.0% | 0.0% | 0.156 | 10.500 |
| 0,5 | 4 | 8 | single_shot | 1 | none | pairwise_ranking | learned | 0.0% | 0.0% | 0.125 | 10.000 |

## Random Baseline Comparison

Same-budget random baselines are diagnostic comparisons using hidden reference distributions, not algorithmic training signals.

| selected_generators | qubits | train_n | avg search budget | observed exact | random exact | observed 1pct | random 1pct | observed 3pct | random 3pct | observed 5pct | random 5pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,1,2 | 6 | 8 | 6.769 | 7.7% | 0.106 | 7.7% | 0.106 | 7.7% | 0.106 | 7.7% | 0.106 |
| 0,5 | 4 | 4 | 4.781 | 62.5% | 0.299 | 62.5% | 0.299 | 62.5% | 0.299 | 62.5% | 0.299 |
| 0,5 | 4 | 8 | 3.562 | 62.5% | 0.223 | 62.5% | 0.223 | 62.5% | 0.223 | 62.5% | 0.223 |

## Measurement Policy Comparison

Shot-batch sampling repeats circuit sampling and chooses top measured candidates; single-shot records one bitstring per circuit execution.

| measurement | shots_per_circuit | avg total shots | avg circuit executions | runs | exact | within_3pct | avg ED/LP |
|---|---:|---:|---:|---:|---:|---:|---:|
| shot_batch | 2000 | 5223.881 | 2.612 | 67 | 62.7% | 62.7% | 12.149 |
| single_shot | 1 | 3.000 | 3.000 | 23 | 0.0% | 0.0% | 9.609 |

## Candidate Budget Comparison

Larger top-k budgets can improve recovery, but they spend more search verification ED/LP calls.

| top_k | avg verified candidates | runs | exact | within_3pct | avg ED/LP |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.913 | 46 | 17.4% | 17.4% | 9.478 |
| 3 | 7.500 | 22 | 72.7% | 72.7% | 12.500 |
| 5 | 12.500 | 22 | 81.8% | 81.8% | 14.727 |

## Refit Comparison

The accepted refit policy updates the surrogate only from training samples plus measured and verified candidates.

| refit_policy | runs | exact | within_3pct | avg ED/LP |
|---|---:|---:|---:|---:|
| accepted | 42 | 50.0% | 50.0% | 11.167 |
| none | 48 | 43.8% | 43.8% | 11.792 |

## Learner Comparison

The pairwise_ranking learner is an ablation against the existing mismatch learner; neither uses hidden enumeration for training.

| learner | runs | exact | within_3pct | avg ED/LP |
|---|---:|---:|---:|---:|
| mismatch | 48 | 45.8% | 45.8% | 11.833 |
| pairwise_ranking | 42 | 47.6% | 47.6% | 11.119 |

## Perfect Oracle Diagnostic

hidden_perfect_diagnostic is not an algorithmic result. It uses hidden reference only to separate GAS/sampling limitations from learned-surrogate limitations.

| oracle_mode | runs | exact | within_3pct | avg ED/LP |
|---|---:|---:|---:|---:|
| hidden_perfect_diagnostic | 6 | 100.0% | 100.0% | 8.667 |

## Cautious Conclusions

- The current prototype should be read as selected-generator subspace optimum recovery, not full 12-bit case14 T=2 global optimization.
- Non-random enrichment in small 4q subspaces is visible when observed success exceeds same-budget random probabilities, but this does not establish quantum advantage.
- Shot-batch sampling can improve candidate stability, and total_shots must be reported as part of the resource budget.
- If hidden_perfect_diagnostic succeeds while learned runs fail, the likely bottleneck is the learned affine surrogate; if it also fails, the GAS schedule, sampling, or candidate selection is implicated.

## Notes

- Random baseline and hidden_perfect_diagnostic use hidden reference only for diagnostic comparison.
- hidden_perfect_diagnostic is not an algorithmic result.
- This is selected-generator subspace optimum recovery, not full 12-bit case14 T=2 global optimization.
- Shot-batch sampling uses repeated circuit sampling and must be reported with total_shots.
