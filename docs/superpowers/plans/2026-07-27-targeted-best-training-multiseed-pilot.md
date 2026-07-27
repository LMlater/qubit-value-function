# Targeted best-training multi-seed pilot — implementation plan

**Goal:** Add a safe diagnostic-only runner plus read-only audit/validation for
the specified best-training pilot, without changing search semantics.

1. Add failing unit tests for oracle-specific Grover probabilities, strict
   marked-set predicates, outcome classification, run-level ED/LP statistics,
   and nontraining-improvement success metadata.
2. Implement the pure audit helpers in
   `qubit_value_function/targeted_pilot_diagnostics.py`; rerun those tests.
3. Add failing tests for manifest roles, artifact-only preflight selection,
   missing snapshot/empty-candidate failures, deterministic configuration, and
   existing-output refusal.
4. Implement an explicit pilot manifest and CLI adapter in
   `experiments/stage1_targeted_best_training_pilot_cli.py`, reusing the
   existing scenario builder and closed-loop runner; rerun CLI tests.
5. Add tests for synthetic persisted-run aggregation and read-only validation;
   implement their serializers and validators without calling RNG, ED/LP,
   training, or global-optimum readers during audit.
6. Add concise README usage notes; run targeted tests, full pytest suite,
   `git diff --check`, inspect the diff/status, then stage only task files and
   create the requested local commit.
