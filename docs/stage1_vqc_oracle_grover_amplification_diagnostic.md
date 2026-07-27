# Stage A sparse-VQC oracle semantic validation and fixed-oracle Grover integration

## Purpose and boundary

The primary evidence for this stage is deterministic, exact simulation of the
custom sparse-VQC value, threshold-comparator, hard-logic, and joint phase
oracle circuits.  Fixed-oracle multi-shot Grover is only an end-to-end
integration check for one representative marked-set size; it is not a claim of
quantum speedup or the principal correctness evidence.

## Primary exact semantic checks

For each selected saved scenario, reconstruct the VQC only from its persisted
training labels and stored training configuration, then use exact statevector
simulation to check all 16 commitment states:

- the reversible value register equals the classical quantized VQC integer;
- the strict predicate is `integer_value < encoded_threshold`, including
  threshold minus one, threshold, and threshold plus one;
- the joint relative-phase truth table equals that predicate AND the hard-logic
  feasibility predicate;
- after compute--phase-mark--uncompute, every non-search register is zero and
  unentangled with the search register;
- the actual output agrees with the theoretical diagonal phase oracle up to one
  global phase only.

The comparison is deterministic and exact; 1024-shot frequencies are not used
as evidence for these circuit semantics.

## Fixed-oracle Grover integration check

For one representative natural `M=3` joint-marked scenario, keep the oracle and
threshold fixed and run the complete circuit at `k=0,1,2,3`.  Report exact
statevector marked probabilities and an independent 1024-shot Aer-MPS result,
including a Wilson interval, search-register counts, auxiliary-zero
probability, circuit resources, and seeds.  Compare the exact result with
`sin^2((2k+1) asin(sqrt(M/16)))`.

The marked set is calculated offline from the reconstructed quantized VQC
integer table and identical hard-logic predicate.  It is never fed back into
the online search procedure.

## Safeguards

- No `FixedCommitmentEvaluator`, ED, LP, true-cost enumeration, closed-loop
  search, or formal/pilot/smoke batch is invoked.
- Persisted labels are used only for the stored training indices; all 16-state
  values come from the reconstructed surrogate circuit/model.
- Natural fixed thresholds are retained.  If an unavailable marked-set size
  would require altering a threshold, it is reported as unavailable rather
  than manufactured.
- This diagnostic makes no end-to-end optimization or speedup claim.
