# Stage 1 Evidence Fixtures Implementation Plan

> **For Hermes:** Execute this plan locally, task by task.  The user explicitly
> prohibits commits and pushes for this round.

**Goal:** Make the four Stage 1 historical evidence tests reproducible from a
small repository fixture, with complete provenance and no dependency on local
untracked results.

**Architecture:** Preserve the production audit APIs and historical directory
shape.  A deterministic offline extractor creates a minimal JSON mirror from
verified full results; tests bind to that mirror through a test-only path
helper and verify its inventory.  A clean worktree receives the uncommitted
patch only for the final default-pytest verification.

**Tech Stack:** Python standard library (`argparse`, `hashlib`, `json`,
`pathlib`, `tempfile`), pytest, Git worktrees.

---

### Task 1: Specify fixture paths and missing-fixture failure

**Objective:** Redirect the four historical tests to a repository-relative
fixture root without changing any audit assertion or production reader.

**Files:**

- Modify: `tests/test_selected_split_best_training_benchmark.py`
- Modify: `tests/test_selected_split_provenance.py`
- Create: `tests/test_stage1_evidence_fixtures.py`

**Step 1: Write failing tests.** Add a fixture-root helper based on
`Path(__file__).resolve().parent / "fixtures" / "stage1_evidence"` and a test
that requires `manifest.json`; its error must name the missing fixture path.
Point the two audit tests and the selected-split manifest test at the mirror.

**Step 2: Verify the red state.**

Run: `python -m pytest tests/test_stage1_evidence_fixtures.py -q`

Expected: failure that explicitly names the absent fixture manifest.

**Step 3: Keep only test-side routing.** Do not alter
`qubit_value_function/selected_split_*_audit.py` or create a production
adapter.  Retain all existing counts, fractions, and tolerances.

### Task 2: Build the deterministic minimal extractor

**Objective:** Generate only the JSON fields consumed by existing audits and
tests, with source and output SHA-256 provenance.

**Files:**

- Create: `scripts/build_stage1_evidence_fixtures.py`
- Test: `tests/test_stage1_evidence_fixtures.py`

**Step 1: Write the failing extractor-contract tests.** Test that the script
refuses an existing destination, creates the expected 456 evidence JSON files
(12 snapshots + 80 targeted runs + 360 selected-split runs + 4
manifests/summaries), writes stable JSON, and stores no
absolute source path in its manifest.

**Step 2: Verify the red state.**

Run: `python -m pytest tests/test_stage1_evidence_fixtures.py -q`

Expected: failure because the extractor and fixture manifest do not exist.

**Step 3: Implement explicit whitelists.** The script must:

```python
EXTRACTOR_VERSION = "stage1-evidence-fixture-v1"
WHITELIST_VERSION = "stage1-evidence-fields-v1"

def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
```

Use per-type projection functions, relative source paths, sorted source
enumeration, full SHA-256 values, an explicit `--generated-at` argument, a
new sibling temporary output directory, and failure when the destination
already exists.  Verify the four supplied canonical hashes before writing.

### Task 3: Generate and inspect the frozen mirror

**Objective:** Create `tests/fixtures/stage1_evidence/` from the verified local
historical results without copying full results.

**Files:**

- Create: `tests/fixtures/stage1_evidence/**`

**Step 1: Run the extractor once with the full results root and a fixed,
documented UTC generation timestamp.** The output target must not exist.

**Step 2: Verify inventory and provenance.** Confirm 12 snapshots, 80
targeted runs, 360 selected-split runs, the four canonical full source hashes,
all per-file source/output hashes, sizes, and no absolute paths.

**Step 3: Verify byte determinism.** Generate a second temporary output using
the same arguments and compare relative file hashes with the frozen fixture.

### Task 4: Document fixture limits and explicit full-results audit

**Objective:** Explain why the mirror exists and how to regenerate or perform
an opt-in historical evidence review.

**Files:**

- Create: `tests/fixtures/stage1_evidence/README.md`
- Create: `docs/testing_and_evidence_fixtures.md`

**Step 1: Document the four tests, source result directory names, field
minimization, manifest verification, and the statement that the fixture is not
the complete formal delivery.**

**Step 2: Provide exact commands for an explicit local full-results audit and
for regeneration.** Default pytest must not run either operation.

### Task 5: Run focused and isolated clean-checkout verification

**Objective:** Prove default pytest works without local untracked results.

**Files:** none (verification only).

**Step 1: Run the fixture test and affected test modules in the primary
checkout.**

Run: `python -m pytest tests/test_stage1_evidence_fixtures.py tests/test_selected_split_best_training_benchmark.py tests/test_selected_split_provenance.py -q`

Expected: all pass.

**Step 2: Create a detached clean worktree at current `HEAD`; apply only the
uncommitted tracked and fixture diff to it; do not copy `results/`.**

**Step 3: Run default pytest there.**

Run: `python -m pytest -q`

Expected: all tests pass and no access to a repository-external or untracked
historical `results/` directory occurs.

**Step 4: Report complete evidence.** Include fixture file count, bytes,
source bytes, reduction ratio, source/output hash verification, focused/full
test results, `git status --short`, `git diff --check`, and `git diff --stat`.
Do not commit, push, add results, or implement Task B.
