# GAS Surrogate-Focused Sweep Summary

## Purpose

Prior diagnostics suggest learned surrogate quality is the main bottleneck. This sweep fixes shot-batch sampling and top-k candidate budget, then compares surrogate learners and accepted refit.

## Fixed Settings

- backend: qasm
- shots: 2000
- measurement_policy: shot_batch
- max_candidates_per_shotbatch: 3
- oracle_mode: learned
- exclude_hidden_optimum_from_training: True
- exclude_hidden_optimum_from_initial: True

## Overall Results

- total runs: 360
- ok runs: 193
- skipped runs: 167
- skipped invalid config runs: 120
- skipped resource limit runs: 47
- error runs: 0
- exact success: 66.3%
- within 1 percent success: 66.3%
- within 3 percent success: 66.3%
- within 5 percent success: 66.3%
- avg algorithmic ED/LP calls: 22.658
- avg total shots: 31554.404
- avg circuit executions: 15.777
- avg max qubits: 19.326
- avg max transpiled depth: 517.668
- all learned ok runs exclude hidden optimum from training and initial: True

## Learner Comparison

| learner | num_ok_runs | exact_success | within_3_percent_success | avg_pairwise_order_accuracy | avg_pairwise_hinge_loss | avg_algorithmic_ed_lp_calls | avg_total_shots | avg_max_transpiled_depth | fallback_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mismatch | 80 | 62.5% | 62.5% | 0.656 | 28.150 | 27.525 | 35825.000 | 357.750 | 0.0% |
| pairwise_ranking | 52 | 75.0% | 75.0% | 0.731 | 11.923 | 19.654 | 30653.846 | 776.712 | 0.0% |
| rank_hinge | 61 | 63.9% | 63.9% | 0.829 | 7.033 | 18.836 | 26721.311 | 506.574 | 0.0% |

## Refit Comparison

| refit_policy | num_ok_runs | exact_success | within_3_percent_success | avg_refit_count | avg_observed_sample_count | avg_algorithmic_ed_lp_calls |
|---|---:|---:|---:|---:|---:|---:|
| accepted | 94 | 64.9% | 64.9% | 1.649 | 8.936 | 21.660 |
| none | 99 | 67.7% | 67.7% | 0.000 | 7.838 | 23.606 |

## Learner × Refit

| learner | refit_policy | num_ok_runs | exact_success | within_3_percent_success | avg_algorithmic_ed_lp_calls |
|---|---|---:|---:|---:|---:|
| mismatch | accepted | 40 | 60.0% | 60.0% | 27.425 |
| mismatch | none | 40 | 65.0% | 65.0% | 27.625 |
| pairwise_ranking | accepted | 24 | 75.0% | 75.0% | 17.667 |
| pairwise_ranking | none | 28 | 75.0% | 75.0% | 21.357 |
| rank_hinge | accepted | 30 | 63.3% | 63.3% | 17.167 |
| rank_hinge | none | 31 | 64.5% | 64.5% | 20.452 |

## 4q vs 6q

| qubits | learner | num_ok_runs | exact_success | within_3_percent_success | avg_pairwise_order_accuracy |
|---:|---|---:|---:|---:|---:|
| 4 | mismatch | 40 | 92.5% | 92.5% | 0.737 |
| 4 | pairwise_ranking | 40 | 87.5% | 87.5% | 0.752 |
| 4 | rank_hinge | 40 | 92.5% | 92.5% | 0.876 |
| 6 | mismatch | 40 | 32.5% | 32.5% | 0.575 |
| 6 | pairwise_ranking | 12 | 33.3% | 33.3% | 0.663 |
| 6 | rank_hinge | 21 | 9.5% | 9.5% | 0.739 |

## Same-Budget Random Baseline

Same-budget random baseline is retained as a diagnostic comparison only.

- observed exact success: 66.3%
- average random exact probability: 0.490

## Formal-Sweep Decision

- rank_hinge overall exact gain vs mismatch: 1.4%
- rank_hinge 6q exact gain vs mismatch: -23.0%
- best accepted-refit exact gain vs none with the same learner: 0.0%
- The smoke results did not show a sufficiently clear improvement to justify a 20-seed focused formal sweep.

## Cautious Conclusion

The result suggests that simply improving pairwise ranking within the current integer max-affine surrogate class is insufficient; a richer surrogate class may be needed.

This is selected-generator subspace optimum recovery, not full 12-bit case14 T=2 global optimization.
This does not establish quantum advantage.
