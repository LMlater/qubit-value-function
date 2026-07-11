# Handoff

## Current task

Finish the final surrogate-focused case14 T=2 gate-level GAS experiment round. Accepted-refit provenance and per-round oracle-version logging are fixed; the five-seed smoke has completed. Remaining work is final verification and committing/pushing the source, tests, and smoke artifacts. A 20-seed formal sweep is not warranted by the smoke threshold.

Fixed scope: qasm backend, 2000 shots, `shot_batch`, three candidates per batch, learned oracle, hidden optimum excluded from training and initialisation, learners `mismatch`, `pairwise_ranking`, and `rank_hinge`, and refit policies `none` and `accepted`. Do not add other learners, shots, oracle modes, measurement policies, or top-k ablations.

## Repository state

- Branch: `main`, currently based on and tracking `origin/main` at `f911954 Add surrogate learner focused GAS sweep`.
- Earlier relevant commits: `e21b54d Document surrogate learner sweep design`, `f7e323d Fix GAS diagnostics and add equal-shot smoke`.
- `f911954` contains the rank-hinge implementation, focused-sweep script, tests, and this handoff document.
- Current uncommitted work: accepted-refit, round-oracle-version, and no-marked-state handling in `experiments/stage1_case14_t2_small_sample_gate_level_max_affine_gas.py`; regression and summary tests in `tests/test_stage1.py`; updated focused-sweep summary generation; and smoke artifacts.
- Untracked `results/test_diagnostics_tiny.*` are generated test artifacts; do not commit them.

## Completed verification

- Latest baseline before the fixes: `pytest -q` reported `33 passed`.
- Two regression tests were added and observed failing against the original behavior:
  - verified-but-rejected candidates incorrectly entered accepted-refit data;
  - round logging had no pre-refit snapshot and used the post-refit learner for allocation.
- After the minimal fix, the focused regression tests passed.
- Latest complete verification before the no-marked-state and summary additions: `pytest -q` reported `35 passed`; `git diff --check` returned no whitespace errors. Run both again before committing.

## Implemented but uncommitted fix

- `observed_by_index` now starts with initial training data and receives a candidate only after that measured, ED/LP-verified candidate becomes the new incumbent under `refit_policy="accepted"`.
- All verified candidates remain counted and are recorded in `verified_candidate_indices`; accepted refit additions are separately recorded in `accepted_refit_indices`.
- Each search round freezes `round_learned`, its calibration, integer threshold, oracle spec, and register allocation. A refit replaces `current_learned` only for the next round.
- Round rows now record pre/post refit versions, the learner/fallback and pieces used before execution, whether a refit was triggered, and next-round version.
- A calibration threshold with no marked state now ends that run cleanly with `stop_reason=no_marked_state_at_threshold` rather than raising a Grover exception.

## Smoke result

- Artifacts: `results/stage1_case14_t2_small_sample_gate_level_gas_surrogate_smoke.json`, `.csv`, and `_summary.md`.
- Total / OK / invalid skip / resource skip / error: `360 / 193 / 120 / 47 / 0`.
- Overall exact and within 3%: `66.3%`; average algorithmic ED/LP calls: `22.658`; average total shots: `31554.404`.
- All learned OK runs have `training_contains_hidden_optimum=false` and `initial_matches_hidden_optimum=false`.
- Exact success by learner: mismatch `62.5%`, pairwise_ranking `75.0%`, rank_hinge `63.9%`.
- Exact success by refit policy: none `67.7%`, accepted `64.9%`.
- rank_hinge vs mismatch: overall `+1.4` percentage points; 6q `-23.0` points. No accepted-refit comparison improved by 10 points.
- Therefore, do not run the 20-seed formal sweep. The smoke evidence does not meet the configured 10 percentage-point threshold.

## Next steps

1. Run fresh `pytest -q` and `git diff --check`.
2. Stage only source, tests, this handoff, and the three smoke artifacts. Do not stage `results/test_diagnostics_tiny.*`.
3. Commit with `Fix surrogate refit semantics and add smoke results` and push `main`. Do not omit the add/commit/push cycle.

## Pitfalls

- Never add all top-k verified candidates to accepted-refit training data; only accepted new incumbents belong there.
- Never use a refit model to describe a circuit already executed with the old model.
- Do not use `git reset --hard` or `git checkout -- .` in this working tree.
- Sandboxed pytest can fail to write `results/` and `.pytest_cache`; elevated execution has previously produced valid results, though the cache warning can remain non-fatal.
- The final conclusion must say this is selected-generator subspace optimum recovery, not full 12-bit global optimisation, and does not establish quantum advantage.
