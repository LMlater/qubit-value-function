# Stage B Fixed Fit/Validation Implementation Plan

> **For Hermes:** Execute locally with TDD. The user explicitly prohibits
> commits, pushes, staging results, and modification of Stage A evidence.

**Goal:** Provide deterministic, leakage-free fit/validation partitions,
normalization, model selection, metrics, ED/LP truth caching, and a one-unit
classical smoke path for the approved Stage B generalization protocol.

**Architecture:** `stage2_generalization.py` owns protocol constants,
canonical partitions, normalizers, selection, metrics, and truth-cache keys.
`stage2_baselines.py` owns constant/linear/quadratic classical regressors.
The smoke CLI composes those modules for one pair/window/split only; it does
not launch the 1,008-row pilot or QNN grid.

**Tech Stack:** Python 3, NumPy, existing case14 ED/LP evaluator, pytest.

---

### Task 1: Freeze and document the protocol

**Files:**

- Create: `docs/research_content1_stageB_generalization_protocol.md`

**Steps:** Record the seven loads, train/interpolation/extrapolation roles,
8-state split, per-training-load 6/2 deterministic partition, canonical JSON
and SHA-256, fit-only normalization, validation-only selection ordering, test
one-shot rule, and the later model roster. State that this implementation does
not yet run the full model/QNN grid.

### Task 2: Add the partition and selection contract tests

**Files:**

- Create: `tests/test_stage2_generalization.py`

**Steps:** Write failing tests for fixed state splits; 18 fit/6 validation
sample pairs; union/disjointness; all training load participation; same-seed
canonical hash; fit-only normalization; test-label independence; deterministic
MAE/regret/parameter/order tie-break; finite regression/regret metrics;
deterministic no-duplicate cache key; and imports of the first pilot.

Run: `python -m pytest tests/test_stage2_generalization.py -q`

Expected before implementation: import failure for `stage2_generalization`.

### Task 3: Implement protocol primitives

**Files:**

- Create: `qubit_value_function/stage2_generalization.py`

**Steps:** Implement validated protocol constants, state-split construction,
per-load stable 6/2 partition using canonical SHA-256-derived randomness,
canonical JSON/hash storage, fit-only load/target normalizers, finite metrics,
regret, candidate selection, and an in-memory deterministic ED/LP truth cache.
Run the new test module until green.

### Task 4: Add classical baselines

**Files:**

- Create: `qubit_value_function/stage2_baselines.py`
- Modify: `tests/test_stage2_generalization.py`

**Steps:** Add constant, linear Ridge, and quadratic Ridge feature builders
and regressors. Fit only passed fit arrays; expose finite prediction and
parameter-count metadata. Add their unit tests and rerun the test module.

### Task 5: Add a one-unit classical smoke runner

**Files:**

- Create: `experiments/stage2_generalization_benchmark_cli.py`
- Modify: `tests/test_stage2_generalization.py`

**Steps:** Generate exactly one selected pair/window/split's 112 ED/LP truth
rows through the cache. Build fit/validation/test rows, select from a fixed
classical candidate list exclusively on validation, and emit finite summary
JSON only to an explicitly new output directory. Keep QNN selection and the
full 1,008-row protocol out of this smoke runner.

### Task 6: Verify and estimate

**Steps:** Run the new test module, `python -m pytest -q`, and exactly one
temporary-directory smoke. Report the saved partition example/hash, cache
solve count, selection result, and finite metrics. Estimate full-pilot ED/LP
calls, classical fits, planned QNN optimizations, duration, and output size;
stop rather than launch if the estimate exceeds 90 minutes.
