# Active hard-logic oracle comparison for sparse-VQC BBHT

## Purpose

The previous 2×2 case14 joint-oracle experiment compiled zero retained Boolean
logic clauses. In that selected subspace the hard-feasibility function was
identically true, so the joint oracle was operationally identical to the
cost-only threshold oracle.

This stage adds a deterministic logic-only scan and then runs a fair cost-only
versus joint-oracle BBHT comparison in a selected subspace where:

- retained forbidden patterns are nonzero;
- logic-feasible and logic-infeasible states both exist;
- enough logic-feasible states exist for the configured training set;
- selection does not use cost, ED/LP, VQC fit, BBHT outcomes, or a hidden optimum.

## BBHT hardening

`BBHTConfig` now includes:

```text
max_auxiliary_syndrome_rejections = 8
```

A trial whose measured auxiliary-zero probability is below
`minimum_auxiliary_zero_probability` is rejected without cache lookup, exact
candidate evaluation, ED/LP, threshold update, or BBHT-window growth. The
independent syndrome counter stops the run with:

```text
max_auxiliary_syndrome_rejections_reached
```

When a supplied compiled feasibility specification has
`always_infeasible=True`, BBHT performs no circuit and stops with:

```text
no_hard_logic_feasible_state_by_compiled_constraints
```

This is a deterministic compiled-constraint conclusion rather than a
probabilistic search failure.

## Logic-safe base commitment

The scan fixes unselected generators to a deterministic high-capacity Boolean
schedule that satisfies the existing `is_logic_feasible` semantics:

- must-run generators remain online;
- initially online generators remain online;
- initially offline generators respect any remaining minimum downtime, then
  start and remain online;
- no load, cost, ED/LP, VQC, or optimal-solution information is used.

The selected generators are replaced by the search-register bits.

## Subspace scan

The default scan considers:

```text
horizon = 2, 3
selected generator count = 2
window start = 0
```

For every candidate subspace it records:

- selected generator indices and names;
- search-register size;
- retained forbidden-pattern count;
- source and retained pattern-family counts;
- logic-feasible and logic-infeasible indices;
- feasible ratio;
- compiled-spec versus classical-logic agreement;
- estimated violation-ancilla count.

Enumeration is limited to this small diagnostic selected subspace. It is not
called by the Grover or BBHT builders and is not used during quantum search.

## Deterministic selection rule

A candidate must satisfy:

1. `always_infeasible=False`;
2. retained forbidden patterns greater than zero;
3. both feasible and infeasible states exist;
4. at least the requested number of feasible training states exists;
5. compiled and classical logic agree on every diagnostic state.

Ranking is:

1. prefer feasible ratio in `[0.25, 0.75]`;
2. maximize the number of excluded logic-infeasible states;
3. minimize retained forbidden patterns and therefore violation ancillas;
4. prefer the smaller horizon;
5. generator-index lexicographic tie break.

The selection metadata explicitly records that it does not use costs or hidden
optimization results.

## Fair comparison

For each load window, both methods share exactly the same:

- selected subspace and base commitment;
- training commitment indices;
- ED/LP training labels;
- trained sparse phase VQC;
- quantized integer value model;
- fixed-point configuration;
- initial incumbent;
- BBHT budgets and random seed.

The only intended quantum-oracle difference is:

### Cost-only

```math
\widehat C_{\theta,\mathrm{int}}(x) < \tau.
```

A measured surrogate-better but logic-infeasible state reaches the classical
logic precheck and is rejected without an LP solve.

### Joint

```math
F_{\mathrm{logic}}(x)
\land
[\widehat C_{\theta,\mathrm{int}}(x) < \tau].
```

The same classical precheck remains as a defensive integration check, but the
quantum oracle should exclude those states from the marked set.

The default comparison uses `worst-training` initialization. This relies only
on shared training labels, applies equally to both methods, and creates a broad
initial threshold that makes filtering effects easier to observe without using
hidden validation labels.

## Reported metrics

Each method reports:

- final incumbent and true cost;
- threshold history;
- trials, shots, and oracle calls;
- new exact-evaluation attempts;
- actual ED/LP solves;
- logic-precheck rejections;
- cache lookups;
- hard-infeasible measurements;
- hard-infeasible and surrogate-better measurements;
- auxiliary syndrome rejections;
- marked and repeated candidate hits;
- maximum trial qubits and depth;
- MPS elapsed time;
- stop reason.

Post-search validation reports, for every threshold:

- cost-only surrogate-better indices;
- joint hard-feasible-and-better indices;
- hard-infeasible better indices removed;
- true improving logic-feasible indices;
- joint surrogate false positives and false negatives.

The complete exact landscape and true global optimum are evaluated only after
both searches finish.

## Scope boundary

The hard oracle in this stage encodes only:

- must-run;
- initial remaining minimum uptime;
- initial remaining minimum downtime;
- minimum uptime following startup;
- minimum downtime following shutdown.

It does not yet encode:

- capacity adequacy;
- reserve adequacy;
- minimum-output adequacy;
- exact continuous ramp feasibility;
- network power flow;
- N-1 security.

The result must therefore be described as an active Boolean UC logic oracle,
not a complete SCUC physical-feasibility oracle.

## Commands

Logic-only scan:

```powershell
$scan = Join-Path $env:TEMP "scan_case14_logic_feasibility_subspaces.json"
D:\pytorch\anaconda_develop\envs\QubitValueFunction\python.exe `
  experiments\scan_case14_logic_feasibility_subspaces.py `
  --horizons 2,3 `
  --minimum-feasible-states 8 `
  --results $scan
```

Seed-0 fair comparison:

```powershell
$result = Join-Path $env:TEMP "stage1_case14_active_logic_joint_bbht_seed0.json"
D:\pytorch\anaconda_develop\envs\QubitValueFunction\python.exe `
  experiments\stage1_case14_active_logic_joint_bbht.py `
  --scan-horizons 2,3 `
  --window-starts 0,1,2 `
  --train-sample-count 8 `
  --initialization-policy worst-training `
  --seed 0 `
  --max-auxiliary-syndrome-rejections 8 `
  --results $result
```
