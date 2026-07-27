# Stage A VQC oracle–Grover amplification diagnostic design

## Purpose and boundary

This is a proposed small-scale diagnostic, not a completed experiment. It would isolate whether the existing trained sparse-VQC joint phase oracle and the diffuser amplify the states marked by that oracle. It does not claim end-to-end quantum speedup, quadratic speedup, or better ED/LP optimization performance.

No diagnostic described here may run without explicit user approval. It must reuse a persisted formal trained model; it must not retrain VQC or call ED/LP.

## Fixed-oracle experiment

For representative saved formal scenarios, select one fixed threshold each with joint marked count (M=1,2,3,4), if those models and threshold tables have been persisted. For each fixed oracle, execute full gate-level circuits at (k=0,1,2,3), with 1024 or 4096 shots per circuit.

Each run must measure the search register and the auxiliary registers. Report:

- the search-register distribution;
- total probability assigned to the precomputed joint-marked set;
- (M/16) uniform reference;
- \(\sin^2((2k+1)\arcsin\sqrt{M/16})\) ideal Grover reference;
- phase-marking, uncomputation, auxiliary-zero syndrome, diffuser, circuit resources, wall time, and memory separately.

The marked set is calculated offline from the persisted quantized VQC integer table and identical hard-logic predicate. It is never fed back into online search.

## Required safeguards

- Reuse a saved trained model and quantized 16-state value table; do not recreate it by retraining.
- Do not invoke `FixedCommitmentEvaluator`, ED, LP, or true-cost enumeration.
- Keep each oracle fixed while varying only (k); BBHT's dynamic threshold/window policy is not part of this fixed-oracle diagnostic.
- Treat (k=0) as a same-workflow no-amplitude-amplification reference, not as a fully randomized control experiment.
- Use multiple shots to estimate probabilities. A single-shot 0/1 outcome is not a circuit probability estimate.
- Report shots, seeds, transpilation/simulator settings, circuit depth/gate counts, elapsed time, and memory.

## Interpretation

Agreement with the ideal curve supports only the gate-level oracle/diffuser behavior for that fixed small circuit. A discrepancy can arise from implementation semantics, finite shots, transpilation/simulation details, or an unsuitable (k). It does not itself assess VQC true-cost accuracy. That requires a separately labelled offline validation analysis of marked precision and recall.

## Persistence requirement before approval

The current formal completed traces do not persist each scenario's 16-state quantized VQC integer value table. Future approved diagnostics must first have a read-only source of that already-trained model/table; no audit may infer missing marked counts from sampled candidates.
