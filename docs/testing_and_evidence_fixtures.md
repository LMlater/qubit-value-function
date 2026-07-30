# Testing and Stage 1 evidence fixtures

## Purpose

Four tests are historical evidence-integration tests, rather than new Stage B
experiments:

- `tests/test_selected_split_best_training_benchmark.py` verifies the frozen
  12-snapshot selected-split preflight and the 80-run targeted-pilot audit.
- `tests/test_selected_split_provenance.py` verifies the 360-run
  selected-split provenance audit and recomputed summary.

Their checked-in input is the intentionally minimal mirror at
`tests/fixtures/stage1_evidence/`. It tests parsing, recomputation,
completeness, hash provenance, and abnormal-summary detection while preserving
the original 12/80/360 evidence denominators. It is not a complete official
Stage 1 result delivery and must not be used to replace the historical formal
results.

## Default test behavior

Tests find the fixture from `tests/stage1_evidence_fixture.py`, which derives
its path from the test source location. It does not depend on the current
working directory or on a Windows absolute fixture path. A missing
`manifest.json` raises a clear missing-fixture error. There is deliberately no
fallback to `results/`, so local untracked results cannot change default pytest
behavior.

Run the default suite with:

```powershell
python -m pytest -q
```

The fixture extractor is never run by pytest.

## Windows sandbox test-environment note

Some restricted Windows sandbox sessions cannot enumerate the user-level
`pytest-of-<user>` temporary directory because of its ACL. This is an
environment-permission failure, not a fixture or test failure: the tests and
their assertions must remain unchanged. Run the same command in an environment
with normal access to pytest's temporary directory and report that distinction
with the test result; do not weaken the evidence tests to work around it.

## Provenance and regeneration

`tests/fixtures/stage1_evidence/manifest.json` contains the full SHA-256 for
every 456 source JSON and reduced fixture JSON. It also records the source
result-directory names, extractor and whitelist versions, generation timestamp,
and byte reduction figures. The four canonical source values are verified by
the extractor before it writes anything:

- formal manifest:
  `2f225f75fb000a216ea3835dee6b231588bbdcf589010d9f89335c6408ffac03`;
- targeted-pilot summary:
  `3e6f8a9b6963ea3a87f17e19cbda1d6fa807a3744694959b8728d7406fef469e`;
- selected-split manifest:
  `5cd8732eeef15870145fecdd3208e27cecc7071641851b1f9b8ea0e5db1f787b`;
- selected-split summary:
  `0257f91af4a747e2504881ae6e58bc0d323a726671aecd19de9f73c31c54e1ac`.

For an explicit regeneration review, use a destination that does not exist:

```powershell
python scripts/build_stage1_evidence_fixtures.py `
  --source-results-root results `
  --output-dir tests/fixtures/stage1_evidence_rebuild `
  --generated-at 2026-07-30T14:34:52Z
```

The timestamp is an explicit input. Reusing the identical results root,
extractor version, and timestamp produces byte-identical data JSON and
manifest content. This avoids making a wall-clock timestamp silently break
reproducibility. Inspect the two manifests' relative source and fixture
SHA-256 maps before accepting a rebuild.

## Explicit complete-results evidence review

Complete historical results remain a separate opt-in integration workflow.
They are not read by default pytest. With a verified local `results/` tree,
run the original read-only audits against their full directories, for example:

```powershell
python -c "from pathlib import Path; from qubit_value_function.selected_split_best_training_audit import audit_targeted_best_training_candidate_pilot as audit; print(audit(Path('results/stage1_targeted_best_training_candidate_pilot_20260727_221524')))"
```

For the selected-split provenance audit, choose a new explicit sidecar output
directory; the audit refuses an existing output directory and does not alter
the result set. Any such full-results review is evidence maintenance and
should report its source hashes separately from the repository fixture.
