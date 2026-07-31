# Stage B supplemental ED/LP component audit

The formal Stage B benchmark directory is immutable and is the sole source of
the 1,008 formal truth keys. Its truth table did not store hard-logic labels or
cost components. This supplemental audit re-evaluates exactly those keys in a
new results directory; it never writes to the formal benchmark.

Each row uses the original case14 input, two-period window, load scaling,
selected-generator embedding, state-bit order, and `FixedCommitmentEvaluator`.
It records production hard logic and independently compiled hard logic. Any
disagreement aborts the audit and blocks later attribution.

The recomputed total must satisfy `abs_diff <= 1e-6 + 1e-9 * abs(formal_cost)`
for every formal row. A failed gate does not replace formal truth and blocks
attribution and alternative-model experiments. Only a 100% pass permits a
keyed join to the formal predictions. Formal-benchmark and supplemental-audit
results must be cited separately.
