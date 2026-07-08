# GAS Equal-Shot Diagnostic Smoke Summary

## Purpose

This follow-up smoke diagnostic fixes two issues from the previous diagnostic pass:

- previous hidden_perfect_diagnostic was a trivial upper bound
- previous single_shot vs shot_batch comparison had unequal total_shots

It compares learned oracle runs against hidden_perfect_uniform_marked and attempts a shot_batch vs single_shot_repeated comparison under explicit total-shot limits.

## Formal Result Context

- formal total runs: 640
- exact success over ok runs: 35.5%
- 4q grouped exact-success range: 35%-60%
- 6q grouped exact-success range: 0%-25%

## Smoke Overall

- total runs: 54
- ok runs: 54
- skipped resource-limit runs: 0
- error runs: 0
- learned exact success: 11.1%
- learned within 1% success: 11.1%
- learned within 3% success: 11.1%
- learned within 5% success: 11.1%


## Grouped Diagnostic Table

| selected_generators | qubits | train_n | measurement | top_k | refit | learner | oracle_mode | ok | skipped_resource_limit | error | exact | within_3pct | random_exact | avg_ed_lp |
|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0,1,2 | 6 | 8 | classical_uniform_marked_diagnostic | 1 | none | hidden_reference | hidden_perfect_uniform_marked | 3 | 0 | 0 | 100.0% | 100.0% | n/a | 12.000 |
| 0,1,2 | 6 | 8 | classical_uniform_marked_diagnostic | 3 | none | hidden_reference | hidden_perfect_uniform_marked | 3 | 0 | 0 | 100.0% | 100.0% | n/a | 14.333 |
| 0,1,2 | 6 | 8 | shot_batch | 1 | none | mismatch | learned | 3 | 0 | 0 | 0.0% | 0.0% | 0.026 | 9.667 |
| 0,1,2 | 6 | 8 | shot_batch | 3 | none | mismatch | learned | 3 | 0 | 0 | 0.0% | 0.0% | 0.057 | 11.667 |
| 0,1,2 | 6 | 8 | single_shot_repeated | 1 | none | mismatch | learned | 3 | 0 | 0 | 0.0% | 0.0% | 0.177 | 19.333 |
| 0,1,2 | 6 | 8 | single_shot_repeated | 3 | none | mismatch | learned | 3 | 0 | 0 | 0.0% | 0.0% | 0.177 | 19.333 |
| 0,5 | 4 | 4 | classical_uniform_marked_diagnostic | 1 | none | hidden_reference | hidden_perfect_uniform_marked | 3 | 0 | 0 | 100.0% | 100.0% | n/a | 6.000 |
| 0,5 | 4 | 4 | classical_uniform_marked_diagnostic | 3 | none | hidden_reference | hidden_perfect_uniform_marked | 3 | 0 | 0 | 100.0% | 100.0% | n/a | 6.333 |
| 0,5 | 4 | 4 | shot_batch | 1 | none | mismatch | learned | 3 | 0 | 0 | 0.0% | 0.0% | 0.062 | 5.000 |
| 0,5 | 4 | 4 | shot_batch | 3 | none | mismatch | learned | 3 | 0 | 0 | 33.3% | 33.3% | 0.188 | 7.000 |
| 0,5 | 4 | 4 | single_shot_repeated | 1 | none | mismatch | learned | 3 | 0 | 0 | 33.3% | 33.3% | 0.396 | 10.333 |
| 0,5 | 4 | 4 | single_shot_repeated | 3 | none | mismatch | learned | 3 | 0 | 0 | 33.3% | 33.3% | 0.396 | 10.333 |
| 0,5 | 4 | 8 | classical_uniform_marked_diagnostic | 1 | none | hidden_reference | hidden_perfect_uniform_marked | 3 | 0 | 0 | 100.0% | 100.0% | n/a | 10.000 |
| 0,5 | 4 | 8 | classical_uniform_marked_diagnostic | 3 | none | hidden_reference | hidden_perfect_uniform_marked | 3 | 0 | 0 | 100.0% | 100.0% | n/a | 11.667 |
| 0,5 | 4 | 8 | shot_batch | 1 | none | mismatch | learned | 3 | 0 | 0 | 0.0% | 0.0% | 0.062 | 9.000 |
| 0,5 | 4 | 8 | shot_batch | 3 | none | mismatch | learned | 3 | 0 | 0 | 33.3% | 33.3% | 0.167 | 10.667 |
| 0,5 | 4 | 8 | single_shot_repeated | 1 | none | mismatch | learned | 3 | 0 | 0 | 0.0% | 0.0% | 0.229 | 11.667 |
| 0,5 | 4 | 8 | single_shot_repeated | 3 | none | mismatch | learned | 3 | 0 | 0 | 0.0% | 0.0% | 0.229 | 11.667 |

## Random Baseline Comparison

Same-budget random baselines are diagnostic comparisons using hidden reference distributions, not algorithmic training signals.

| selected_generators | qubits | train_n | avg search budget | observed exact | random exact | observed 1pct | random 1pct | observed 3pct | random 3pct | observed 5pct | random 5pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,1,2 | 6 | 8 | 7.000 | 0.0% | 0.109 | 0.0% | 0.109 | 0.0% | 0.109 | 0.0% | 0.109 |
| 0,5 | 4 | 4 | 4.167 | 25.0% | 0.260 | 25.0% | 0.260 | 25.0% | 0.260 | 25.0% | 0.260 |
| 0,5 | 4 | 8 | 2.750 | 8.3% | 0.172 | 8.3% | 0.172 | 8.3% | 0.172 | 8.3% | 0.172 |

## Equal-Shot Measurement Comparison

Shot-batch sampling should be interpreted as repeated sampling plus classical candidate extraction, not as a single quantum measurement. single_shot_repeated uses one shot per circuit execution and repeats executions until the configured total-shot or circuit-execution limit.

The average total shots are not in the same range (shot_batch=200.0, single_shot_repeated=14.9); these rows should not be used to claim shot-batch superiority.

| measurement_policy | shots_per_circuit | avg_total_shots | avg_circuit_executions | top_k | runs | exact_success | within_3_percent_success | avg_algorithmic_ed_lp_calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shot_batch | 200 | 200.000 | 1.000 | 1 | 9 | 0.0% | 0.0% | 7.889 |
| shot_batch | 200 | 200.000 | 1.000 | 3 | 9 | 22.2% | 22.2% | 9.778 |
| single_shot_repeated | 1 | 14.889 | 14.889 | 1 | 9 | 11.1% | 11.1% | 13.778 |
| single_shot_repeated | 1 | 14.889 | 14.889 | 3 | 9 | 11.1% | 11.1% | 13.778 |

## Candidate Budget Comparison

Larger top-k budgets can improve recovery, but they spend more search verification ED/LP calls.

| top_k | avg verified candidates | runs | exact | within_3pct | avg ED/LP |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.944 | 18 | 5.6% | 5.6% | 10.833 |
| 3 | 8.944 | 18 | 16.7% | 16.7% | 11.778 |

## Refit Comparison

The accepted refit policy updates the surrogate only from training samples plus measured and verified candidates.

| refit_policy | runs | exact | within_3pct | avg ED/LP |
|---|---:|---:|---:|---:|
| none | 36 | 11.1% | 11.1% | 11.306 |

## Learner Comparison

The pairwise_ranking learner is an ablation against the existing mismatch learner; neither uses hidden enumeration for training.

| learner | runs | exact | within_3pct | avg ED/LP |
|---|---:|---:|---:|---:|
| mismatch | 36 | 11.1% | 11.1% | 11.306 |

## Perfect Oracle Diagnostic

hidden_perfect_uniform_marked uses hidden reference for diagnostic sampling and is not an algorithmic or gate-level result. This diagnostic estimates whether correct oracle labels plus the same candidate budget would be sufficient to reach the hidden optimum.

| oracle_mode | runs | exact | within_3pct | avg ED/LP |
|---|---:|---:|---:|---:|
| hidden_perfect_uniform_marked | 18 | 100.0% | 100.0% | 10.056 |
| learned | 36 | 11.1% | 11.1% | 11.306 |

## Cautious Conclusions

- The current prototype should be read as selected-generator subspace optimum recovery, not full 12-bit case14 T=2 global optimization.
- Non-random enrichment in small 4q subspaces is visible when observed success exceeds same-budget random probabilities, but this does not establish quantum advantage.
- Equal-shot comparison is needed before claiming shot-batch superiority.
- Top-k candidate verification improves recovery but increases ED/LP verification calls.
- hidden_perfect_uniform_marked is diagnostic-only and cannot be reported as algorithmic performance.
- If learned remains far below hidden_perfect_uniform_marked, the likely bottleneck is surrogate quality.

## Notes

- Random baseline and hidden oracle diagnostics use hidden reference only for diagnostic comparison.
- hidden_oracle_trivial_upper_bound is a trivial upper bound and should not be used as GAS/sampling evidence.
- hidden_perfect_uniform_marked is not an algorithmic or gate-level result.
- This is selected-generator subspace optimum recovery, not full 12-bit case14 T=2 global optimization.
- Shot-batch sampling uses repeated circuit sampling and must be reported with total_shots.
