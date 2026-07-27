# Targeted best-training multi-seed pilot design

## Purpose and scope

Add a diagnostic-only, non-formal pilot for the existing case14 closed-loop
VQC--Grover SCUC search.  It increases the number of observed dynamic-oracle
trials without changing the VQC, fixed-point model, strict comparator, hard
logic oracle, BBHT behavior, caching, ED/LP budget, stopping rule, incumbent
rule, or threshold rule.

The pilot is not a random scenario sample and must not support a claim of
quantum advantage or unbiased population performance.  Aer MPS remains a
classical simulation backend.

## Selected approach

Implement a narrow adapter around the existing closed-loop runner rather than
adding another baseline preset or rewriting the experiment framework.  The
adapter will:

1. Load an explicit, version-controlled scenario manifest.
2. Reuse `build_case14_closed_loop_scenario` and `run_closed_loop_method` with
   `initial_incumbent_policy=best_training` and persisted dynamic-oracle
   metadata.
3. Refuse an output directory that already exists.
4. Write a new pilot root containing a manifest, completed-run JSON files,
   trial-level audit records, summaries, and read-only consistency validation.

This keeps online decisions inside existing code and confines new behavior to
configuration, scheduling, serialization adapters, and post-run readers.

## Scenario selection

The manifest contains explicit scenario identifiers, roles, and reasons, never
a directory-order-derived selection.  Preflight records each scenario's
`initial_cost_marked_count`, `initial_joint_marked_count`,
`initial_nontraining_cost_marked_count`,
`initial_nontraining_joint_marked_count`, and
`initial_training_joint_marked_count`.  These values are derived only from a
saved or recoverable quantized VQC model, best-training encoded threshold,
hard-logic feasibility, training indices, and cache membership.

`case14-g0g5-w2-s0` is retained with role
`grover_calibration_training_marked`: its initial joint `M=1` state is the
training incumbent, so it is a probability-calibration and marked-training
cache control, not a high-potential nontraining-improvement scenario.
`case14-g1g3-w0-s0` has role `joint_marked_empty_control`.  Remaining main
targets have role `nontraining_improvement_candidate` and require
`initial_nontraining_joint_marked_count > 0`; preflight prefers two or more
such states, complete snapshots, and distinct `M_joint` values.  Every role
states explicitly whether an initial nontraining joint-marked state exists.
A missing snapshot is represented as `snapshot_unavailable`; it is never
converted to `M=0`.

If manifest requirements cannot be demonstrated from those saved artifacts,
the preflight command fails with the inspected candidates and no output is
created.  It never enumerates true costs, runs ED/LP, or retrains VQC to make a
selection.  If no candidate has an initial nontraining joint-marked state,
preflight fails with `no_nontraining_joint_marked_candidate` and reports the
eligible artifact-derived counts.  Global optima and true-cost landscapes are
never used to select a scenario or pre-register seeds.

## CLI and output safety

The new CLI accepts a manifest/config path, a required new output directory,
an optional comma-separated seed list or inclusive seed range, an optional
method subset, `--preflight`, `--summarize`, and `--validate`.  Its default run
seeds are 0 through 19.  The primary methods are exactly `joint_bbht`,
`cost_only_bbht`, `full_space_random`, and `logic_rejection_random`.

The runner creates only a previously nonexistent output directory.  It does
not clear, append to, resume, or overwrite an existing directory; it does not
write under the source tree or any existing pilot/formal directory.  It does
not fetch, mutate Git, or automatically stage files.

The root `pilot_manifest.json` records schema version, diagnostic/formal
flags, code identity, Python and dependency versions, configuration digest,
scenario roles and reasons, their initial marked-set/count membership facts,
methods, run seeds, best-training policy, dynamic metadata setting, fixed-point
configuration, creation time, and the non-advantage disclaimer.

## Audit and statistics

Only records already persisted during each online run feed the audit layer.
The layer never calls RNG, ED/LP, training, global-optimum readers, cache
mutation, or threshold mutation.

For quantum methods it produces flat CSV and JSONL trial records.  They include
the run identity, dynamic marked-set counts and indices, sampled `k`, measured
index and predicates, cache source, real-cost availability, improvement and
update flags, stopping information, oracle calls, and existing resource fields.

Grouped probability statistics are:

- `joint_bbht`, grouped by scenario, `M_joint`, and `k`, with hit rate,
  `sin^2((2k+1) asin(sqrt(M_joint/N)))`, absolute error, and Wilson 95% CI.
- `cost_only_bbht`, grouped by scenario, `M_cost`, and `k`, using the same
  formula with `M_cost`, never `M_joint`.
- a separate cost-only intersection diagnostic grouped by scenario, `M_cost`,
  `M_joint`, and `k`, with cost-marked, joint-marked, cost-marked/joint-unmarked,
  and hard-logic-infeasible measurement counts.

The random baselines receive a separate operational summary and no Grover
probability calibration.  It reports run/proposal/candidate/cache/EDLP/
improvement counts and stopping reasons without claiming that either baseline
uses VQC.  For every method, run-level search-stage new-EDLP counts produce a
total, mean, median, q25, q75, and (when defined) sample standard deviation.
Training ED/LP and training-plus-search totals are emitted only where persisted
fields reliably distinguish them; they are never inferred.  All conditional
means first filter their run records and then recompute their denominator.

The primary success metric is `nontraining_strict_improvement_success`: a
measured state outside training indices whose true cost comes from a legal new
ED/LP solve or legal search cache, is strictly below the best-training
incumbent, and triggers the unchanged normal incumbent/threshold update.  A
post-run-only validation may additionally emit
`nontraining_global_optimum_success` from an already available validation
landscape; it is forbidden in scenario selection and online decisions.
Run summaries report success counts and probabilities, first success trial and
new-EDLP count, successful state index, and whether it is a global optimum.
Single successes are labeled case-study evidence, not generalization proof.

## Trial outcomes and validation

`joint_bbht` outcomes are `unmarked_measurement`,
`marked_training_cache`, `marked_search_cache`,
`marked_new_edlp_no_improvement`, and
`marked_new_edlp_nontraining_improvement`; pretrial empty-oracle stops are
`pretrial_joint_marked_empty_stop`.

`cost_only_bbht` additionally has
`cost_marked_hard_logic_infeasible` and
`pretrial_cost_marked_empty_stop`.  Its records retain both
`joint_marked_empty_at_trial` and `measured_joint_marked`, so an empty joint
set does not prevent a nonempty cost-only oracle from running.

Read-only validation checks persisted marked-set sizes and intersection,
strict threshold boundary behavior, point-predicate agreement, cache and
new-EDLP exclusivity, real-cost prerequisites for improvements and updates,
best-training cache semantics, nontraining-improvement membership, required
metadata/snapshot presence, outcome exhaustiveness, and random-baseline
nonuse of quantum metadata.  Failures identify run and trial and return a
nonzero status without changing results.

## Testing and documentation

Tests use synthetic persisted traces and `tmp_path`; they do not depend on the
repository `results/` directory.  They cover Grover probabilities and edge
cases, every classification, cache distinctions, empty-oracle distinction,
strict threshold bounds, refusal of existing outputs, stable serialization,
read-only aggregation, metadata/snapshot failures, and baseline isolation.

The README/CLI help will state the diagnostic-only scope, default 20 seeds,
methods, new-output requirement, non-overwrite guarantee, MPS simulation
status, correct oracle-specific probability denominators, and that users run
the complete pilot later themselves.
