# Handoff

## What this work is about

Complete the final **surrogate-focused** experiment round for the case14 T=2 selected-generator-subspace gate-level GAS workflow. The question is whether a better learned integer max-affine surrogate can materially improve learned-oracle recovery.

Fixed experiment settings:

- `backend=qasm`, `shots=2000`, `measurement_policy=shot_batch`
- `max_candidates_per_shotbatch=3`, `oracle_mode=learned`
- exclude the hidden optimum from both training and the initial candidate
- compare `mismatch`, `pairwise_ranking`, and `rank_hinge`, with `refit_policy=none,accepted`

Do not reopen single-shot, perfect-oracle, broad top-k, or shot-count ablations.

## Repository state

- Repository: `LMlater/qubit-value-function`
- Branch: `main`, tracking `origin/main`
- `origin/main` base verified as `f7e323d Fix GAS diagnostics and add equal-shot smoke`
- Current local HEAD: `e21b54d Document surrogate learner sweep design` (one commit ahead of origin; not pushed)
- The working tree contains uncommitted task changes. Preserve them; they are the in-progress implementation described below.

## Completed so far

1. Verified `git pull`: already up to date at the requested base before the local design commit.
2. Ran the test suite successfully with permission to write test artifacts: `33 passed` (warnings are Qiskit deprecations and a non-fatal pytest-cache permission warning).
3. Reviewed the existing in-progress changes:
   - `experiments/stage1_case14_t2_small_sample_gate_level_max_affine_gas.py` adds `rank_hinge`, pairwise hinge diagnostics, local integer coordinate search, and extra refit diagnostics.
   - `experiments/stage1_case14_t2_small_sample_gate_level_gas_surrogate_sweep.py` is new and produces JSON, CSV, and Markdown grouped summaries for the focused sweep.
   - `tests/test_stage1.py` adds tests for rank-hinge output, diagnostics, fallback, and grouped sweep results.
4. Created and committed the approved design specification: `docs/superpowers/specs/2026-07-10-surrogate-learner-focused-sweep-design.md` in commit `e21b54d`.

## Current uncommitted files

- Modified: `experiments/stage1_case14_t2_small_sample_gate_level_max_affine_gas.py`
- Modified: `tests/test_stage1.py`
- New: `experiments/stage1_case14_t2_small_sample_gate_level_gas_surrogate_sweep.py`
- New temporary test artifacts: `results/test_diagnostics_tiny.csv`, `results/test_diagnostics_tiny.json`, `results/test_diagnostics_tiny.md`

The temporary `results/test_diagnostics_tiny.*` files should not be included in the final experiment commit. Do not delete them without confirming they are generated test artifacts and no user data has been placed there.

## Current status / blocker

There is no technical blocker. Work was intentionally paused after writing the design spec and before implementation review, per the design workflow. The user has indicated to continue after this handoff is created.

## Next steps

1. Inspect the complete in-progress diff and compare every required field in the task specification against it.
2. Use test-driven development for any missing or incorrect behavior: add a targeted test, run it to observe the expected failure, then make the smallest production change.
3. Check `rank_hinge` semantics carefully:
   - ordered-pair hinge uses `margin + pred(low-cost) - pred(high-cost)`;
   - weights must stay integer and within `[0, max_weight]`;
   - output must remain valid for the existing max-affine gate-level oracle;
   - a stable `learner_fallback` must always be present.
4. Verify accepted refit uses only initial training points plus measured and ED/LP-verified candidates that became incumbents; it must never include hidden-reference enumeration data.
5. Run the required smoke command (five seeds, four specified configs, valid 4q/6q training sizes, three learners, two refit policies). Preserve generated smoke JSON/CSV/Markdown.
6. Decide whether to run the 20-seed focused formal sweep only from smoke evidence of a clear rank-hinge or accepted-refit improvement. If no clear improvement, do not run formal.
7. Run fresh `pytest -q`, review `git diff --check` and the result summaries, stage only requested source/tests/results, commit with `Add surrogate learner focused GAS sweep`, and push.

## Important pitfalls

- The sandboxed `pytest -q` initially failed seven tests because tests could not write under `results/` and `.pytest_cache`; retrying with elevated permission produced `33 passed`. Do not interpret those first failures as functional regressions.
- The root working tree is already dirty with task-related changes. Do not use `git reset --hard`, `git checkout --`, or any bulk cleanup; preserve and build on the edits.
- The only committed work after the requested base is the design document. The learner/sweep implementation is currently uncommitted.
- Keep `accepted` and `none` only; do not add an `every_round` refit policy.
- The conclusion must be cautious: this is selected-generator subspace optimum recovery, not full 12-bit case14 T=2 global optimization, and it does not establish quantum advantage.
- A formal sweep is conditional on smoke evidence. Avoid running it merely because the script exists.
