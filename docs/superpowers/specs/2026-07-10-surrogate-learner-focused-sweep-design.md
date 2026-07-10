# Surrogate Learner Focused Sweep Design

## Scope

This is the final surrogate-focused ablation for the case14 T=2 selected-generator-subspace experiment. It fixes `shot_batch`, 2000 shots, three verified candidates per batch, the learned oracle, and exclusion of the hidden optimum from both training and initialisation. It does not add shot, perfect-oracle, or broad top-k ablations.

## Learner

Add `rank_hinge` beside `mismatch` and `pairwise_ranking`. It starts from the lightweight pairwise-ranking integer max-affine pieces, then performs deterministic seeded coordinate local search over integer weights and biases. The objective combines ordered-pair hinge loss (margin one), a penalty when the best training sample is not predicted lowest, and small L1 regularisation. Weights remain in `[0, max_weight]`; biases and weights remain integers, so every result remains convertible to the existing gate-level max-affine oracle.

If fewer than two training samples are available, or coordinate search cannot improve the initialization, the learner returns the original pairwise-ranking pieces and reports a stable `learner_fallback` value.

## Diagnostics and refit

Every learner reports its name and fallback, ordered-pair accuracy, pair count, pairwise violations, hinge loss, best training rank/bitstring/cost, predictions and truth, and surrogate piece and weight statistics. The `accepted` policy remains unchanged: only a measured, ED/LP-verified candidate that becomes the incumbent is added to the observed data and triggers retraining. Each such refit records its count, observed sample count and indices, learner, and post-refit ordering accuracy. No hidden-reference point is added to the observed data.

## Focused sweep

The surrogate sweep script evaluates valid 4q and 6q training-set sizes for `mismatch`, `pairwise_ranking`, and `rank_hinge`, crossed with `none` and `accepted`. Invalid combinations are recorded as `skipped_invalid_config`; resource exceptions are retained as `skipped_resource_limit`. The JSON, CSV, and Markdown outputs include overall, learner, refit, qubit-count, and same-budget random-baseline aggregates.

The smoke run uses five seeds. A twenty-seed focused formal run is performed only if smoke shows a clear rank-hinge or accepted-refit improvement; otherwise the smoke results are the final empirical artifact.

## Tests and verification

Tests cover integer and oracle-compatible rank-hinge output, synthetic pairwise diagnostics, stable fallback reporting, accepted-refit provenance, and learner/refit/skip grouped summaries. Run the exact smoke command, then `pytest -q`. The final report will state the fresh command outputs and explicitly describe whether the formal sweep was warranted.

## Interpretation

The Markdown conclusion is data-dependent. An improvement supports learned-surrogate quality as a key bottleneck; no improvement indicates that pairwise ranking alone within the integer max-affine class is insufficient and a richer class may be needed. Results are subspace optimum recovery only and do not establish quantum advantage.
