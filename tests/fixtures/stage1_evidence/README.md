# Stage 1 minimal evidence fixture

This directory is a frozen, minimal mirror for four historical Stage 1
evidence-integration tests. It contains 456 reduced JSON artifacts: one
formal manifest, 12 snapshots, one targeted-pilot summary with 80 completed
runs, and one selected-split manifest and summary with 360 completed runs.
The fixture preserves the original evidence counts and denominators; it is not
the complete formal experimental result delivery.

`manifest.json` records the official source result-directory names, every
source-relative JSON path, each source SHA-256, each reduced-fixture SHA-256,
the field-whitelist version, extractor version, source/fixture sizes, and the
fixed generation timestamp. It covers data JSON files only; the README and
manifest are explanatory metadata rather than reduced experimental artifacts.

Default pytest reads only this directory. If `manifest.json` is absent, the
tests fail explicitly and never fall back to a local `results/` directory.

To regenerate for an explicit evidence-maintenance review, start with an
empty target directory and supply a fixed timestamp as part of the reproducible
input:

```powershell
python scripts/build_stage1_evidence_fixtures.py `
  --source-results-root results `
  --output-dir tests/fixtures/stage1_evidence_rebuild `
  --generated-at 2026-07-30T14:34:52Z
```

Compare the relative JSON SHA-256 values in the rebuilt `manifest.json` with
this one before replacing any fixture. The extractor refuses to overwrite an
existing destination.
