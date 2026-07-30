# Stage 1 evidence fixtures and Stage B generalization design

## Status and scope

This document records the approved design for two sequenced pieces of work on
`feature/research-content1-stage-b`.  It is deliberately uncommitted while the
work is under review.  Task A must pass the default test suite in an isolated
checkout before any Task B implementation begins.

Task A makes four historical Stage 1 evidence-integration tests reproducible
without relying on untracked `results/` files.  It does not change Stage 1
scientific conclusions, evidence denominators, assertions, or numerical
tolerances.  Task B is a later Stage B generalization benchmark; only its
protocol is specified here.

## Task A: frozen, minimal evidence mirror

### Inputs and frozen layout

The repository will contain `tests/fixtures/stage1_evidence/`, preserving the
directory and JSON-reading contracts used by the current audit functions:

- one reduced formal-methods `batch_manifest.json`;
- twelve reduced formal scenario snapshots;
- one reduced targeted-pilot summary and 80 reduced completed runs;
- one reduced selected-split `batch_manifest.json`, one reduced summary, and
  360 reduced completed runs.

The fixture paths mirror the historical result-directory layout.  Tests refer
only to the repository fixture root via paths derived from the test file, never
from the process working directory or a Windows absolute path.  Production
audit functions retain their existing paths and file-reading interfaces; no
test-only production reader adapter is introduced.

The fixture keeps the original 12 snapshots, 80 targeted runs, 360
selected-split runs, denominators, assertions, and tolerances.  It is a
minimal structural mirror, not a substitute for the complete official result
delivery.

### Field minimization

The extractor has explicit, versioned whitelist functions for each input type.
It copies only fields read by `preflight_selected_split_manifest`,
`audit_targeted_best_training_candidate_pilot`, and
`audit_selected_split_provenance`, together with the unchanged test
assertions:

- formal manifest: formal methods and their budget configuration;
- snapshot: identity, generator/window facts, training labels and indices,
  initial-cache/incumbent facts, scale/unit/seed/hash facts, and the 16-state
  proxy values needed for the reachability/marked-count checks;
- targeted run: scenario identity/training indices, the trial-trace fields
  used by the audit, and ED/LP solve count; targeted summary: its three read
  counters;
- selected-split manifest: environment and completed/failed counts;
- selected run: method, scenario/run seed, final and initial incumbent,
  reachability/stop reason, required counters, and the two trial-trace flags;
  selected summary: the audit's complete recomputed summary object.

No prediction tables, network/line data, trial intermediates, absolute local
paths, or unrelated historical metadata are copied.

### Provenance and deterministic regeneration

`scripts/build_stage1_evidence_fixtures.py` accepts the full historical
results root and a destination that does not already exist.  It writes through
a newly created sibling temporary directory, refuses to overwrite an existing
fixture, and atomically publishes only after all files are ready.  JSON is
serialized with fixed indentation, UTF-8, sorted keys, and a trailing newline.

The manifest records each source path relative to the supplied results root,
the complete SHA-256 of every source and generated fixture file, the official
source result-directory names, extractor/whitelist version, a supplied
generation timestamp, sizes, and an explicit limitation statement.  The
generation timestamp is an explicit extractor input: identical inputs,
including that timestamp, yield byte-identical fixtures and SHA-256 values.
This resolves the otherwise incompatible requirements to record generation
time and to reproduce byte-identical output.

The known source hashes must be verified and included in full:

- formal manifest:
  `2f225f75fb000a216ea3835dee6b231588bbdcf589010d9f89335c6408ffac03`;
- targeted summary:
  `3e6f8a9b6963ea3a87f17e19cbda1d6fa807a3744694959b8728d7406fef469e`;
- selected-split manifest:
  `5cd8732eeef15870145fecdd3208e27cecc7071641851b1f9b8ea0e5db1f787b`;
- selected-split summary:
  `0257f91af4a747e2504881ae6e58bc0d323a726671aecd19de9f73c31c54e1ac`.

`README.md`, `manifest.json`, and
`docs/testing_and_evidence_fixtures.md` document fixture purpose, sources,
extraction, validation, and the explicit full-results integration workflow.

### Test protections and verification

The four historical tests receive a fixture-root helper that fails with a
clear missing-fixture message.  It never falls back to workspace `results/`,
including when those untracked directories exist.  A fixture-integrity test
checks the expected inventory and manifest hashes.  Default pytest neither
executes the extractor nor accesses historical results outside the repository.

After fixture generation, test execution is performed in a separate clean
checkout populated only with the tracked base plus this uncommitted patch.
`python -m pytest -q` must pass there before Task B code is started.  The
report will include file count, total bytes, source bytes, reduction ratio,
hash validation, and test output.

## Task B: approved protocol, implementation held

Task B compares generalization across generator pairs `(0, 1)`, `(0, 5)`, and
`(1, 5)`; windows `0`, `1`, and `2`; and state split seeds `1`, `7`, and `19`.
Each scenario uses load multipliers `.80`, `.85`, `.925`, `1.00`, `1.075`,
`1.15`, and `1.20`.  Training loads are `.85`, `1.00`, and `1.15`; `.925` and
`1.075` are interpolation-only tests; `.80` and `1.20` are
extrapolation-only tests.  The ED/LP truth cache has 1,008 rows.

Each scenario/split's 24 training samples use a frozen stratified partition:
for each training load, a deterministic algorithm selects two of its eight
`training_indices` for validation and six for fit.  Therefore each split has
18 fit samples and six validation samples.  The algorithm must be independent
of dictionary/file order, state its protocol version, and store a hash of the
partition content.  `fit_indices`, `validation_indices`, and `unseen_indices`
are pairwise disjoint; fit plus validation is exactly `training_indices`; every
training load contributes according to the protocol.  The eight unseen states
never enter validation.

Only fit/validation samples can influence normalization, hyperparameter
selection, regularization, early stopping, model/cutoff choice, or seed
selection.  Test-load labels cannot alter selected hyperparameters.  The
selection order is frozen before test evaluation: validation MAE, then
validation regret, then fewer parameters, then fixed candidate order.  A
threshold-conditioned QNN cutoff is fit/validation-derived only.  Test loads
are evaluated once after selection.

The later benchmark will compare constant, linear Ridge, quadratic Ridge,
small MLP, simplified QNN (four single-Z readouts), full QNN (15 readouts),
and random quantum features plus Ridge.  Planned implementation boundaries
are `qubit_value_function/stage2_generalization.py`,
`qubit_value_function/stage2_baselines.py`,
`experiments/stage2_generalization_benchmark_cli.py`,
`tests/test_stage2_generalization.py`, and
`docs/research_content1_stageB_generalization_protocol.md`.  These files are
not created or modified until Task A passes the isolated clean-checkout suite.
