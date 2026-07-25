# Stage 2: active-logic surrogate-margin BBHT

## Scope

Stage 2 keeps the accepted active Boolean UC logic oracle unchanged and targets
the dominant remaining failure mode observed in Stage 1: sparse-VQC surrogate
false negatives.

The coherent marked condition is widened from

```math
F_{\mathrm{logic}}(x)\land
[\widehat C_{\theta,\mathrm{int}}(x)<k_\tau]
```

to

```math
F_{\mathrm{logic}}(x)\land
[\widehat C_{\theta,\mathrm{int}}(x)<k_\tau+\Delta],
```

where `Delta` is a nonnegative integer in the same fixed-point code used by the
value register.

The exact acceptance rule does **not** change. An incumbent and its base
threshold are updated only after a strict true ED/LP cost improvement.

## BBHT preflight semantics

When a compiled hard-logic specification has `always_infeasible=True`, BBHT now
returns before requiring an initial incumbent, exact cache, evaluator, or MPS
executor.

The result contains:

```json
{
  "initial_incumbent_index": null,
  "final_incumbent_index": null,
  "final_encoded_threshold": null,
  "threshold_history": [],
  "preflight": {
    "hard_logic_space_empty": true,
    "reason": "compiled_constraints",
    "circuit_skipped": true
  }
}
```

All circuit, shot, oracle, exact-evaluation, cache-lookup, and LP counters are
zero.

## Integer margin semantics

`BBHTConfig` includes:

```text
surrogate_integer_margin = 0
```

For a true incumbent cost `C_inc`, the two thresholds are reported separately:

```text
encoded_threshold = encode(C_inc)
effective_oracle_threshold = encoded_threshold + surrogate_integer_margin
```

Only `effective_oracle_threshold` is passed to the coherent comparator. The
following continue to use the exact incumbent:

- strict true improvement;
- incumbent replacement;
- `true_threshold`;
- base `encoded_threshold`;
- exact-cost optimality reporting.

The default margin is zero, preserving the Stage 1 behavior.

## Diagnostic scan guard

Logic-only selected-subspace enumeration now has:

```text
max_scan_qubits = 12
```

Before any `2**num_x_qubits` diagnostic enumeration, the scan checks:

```text
selected_generator_count * horizon <= max_scan_qubits
```

This guard applies only to the experiment-selection diagnostic. It is not a
marked-count input to Grover or BBHT.

## Experiment contract

The Stage 2 experiment is:

```text
experiments/stage2_case14_active_logic_margin_sweep.py
```

The experiment requires `selected generators` and `horizon` to be either both
specified or both omitted:

- both omitted: use the deterministic logic-only scan;
- both specified: validate that exact pair and horizon through the guarded scan;
- only one specified: fail immediately.

Requested windows are never silently discarded. The JSON records:

```json
{
  "window_execution": {
    "requested_window_starts": [0, 1, 2],
    "executed_window_starts": [0, 1],
    "skipped_windows": [
      {
        "window_start": 2,
        "reason": "window_start + horizon exceeds source horizon"
      }
    ]
  }
}
```

## Fair margin sweep

For each valid load window, the experiment performs the following exactly once:

1. build the same logic-safe base commitment;
2. collect the same logic-feasible training commitments;
3. solve the same ED/LP training labels;
4. train one sparse phase VQC;
5. quantize one integer value model;
6. choose one initial incumbent from the training cache.

It then runs cost-only and joint BBHT for:

```text
Delta = 0, 1, 2, 4, 8
```

All methods and margins share:

- training indices and labels;
- phase and integer value models;
- initial incumbent;
- fixed-point configuration;
- BBHT budgets;
- random seed.

Only the effective coherent threshold and the presence or absence of the hard
logic oracle differ.

The complete true landscape is evaluated only after all margin searches have
returned.

## Reported surrogate metrics

For each threshold history item the experiment reports:

- marked indices and count;
- zero-margin marked set at the same incumbent;
- candidate inflation;
- true improving logic-feasible indices;
- surrogate true positives;
- false positives;
- false negatives;
- recall;
- precision;
- cost-only better states;
- joint feasible-and-better states;
- hard-infeasible better states removed;
- logic filtering ratio.

A larger margin is expected to trade lower false-negative risk for more false
positives and exact candidate checks. It is not assumed to improve every seed.

## Split timing

Each actual MPS BBHT trial now reports:

```text
circuit_build_seconds
transpile_seconds
backend_run_seconds
total_trial_seconds
```

The legacy `elapsed_seconds` remains the sum of transpilation and backend
execution for compatibility.

This separates the cost of constructing and decomposing the deeper joint
oracle from the tensor-network backend runtime.

## Commands

Logic scan with the explicit guard:

```powershell
$scan = Join-Path $env:TEMP "scan_case14_logic_guarded.json"
D:\pytorch\anaconda_develop\envs\QubitValueFunction\python.exe `
  experiments\scan_case14_logic_feasibility_subspaces.py `
  --horizons 2,3 `
  --selected-generator-count 2 `
  --max-scan-qubits 12 `
  --minimum-feasible-states 8 `
  --results $scan
```

Seed-0 Stage 2 sweep:

```powershell
$result = Join-Path $env:TEMP "stage2_case14_margin_seed0.json"
D:\pytorch\anaconda_develop\envs\QubitValueFunction\python.exe `
  experiments\stage2_case14_active_logic_margin_sweep.py `
  --scan-horizons 2,3 `
  --window-starts 0,1,2 `
  --margins 0,1,2,4,8 `
  --max-scan-qubits 12 `
  --train-sample-count 8 `
  --initialization-policy worst-training `
  --seed 0 `
  --results $result
```

After seed 0 is accepted, later seeds should explicitly reuse the selected pair
and horizon reported by seed 0:

```powershell
--selected-generators 0,1 --horizon 3
```

## Scope boundary

The hard oracle still encodes only:

- must-run;
- initial remaining minimum uptime;
- initial remaining minimum downtime;
- minimum uptime after startup;
- minimum downtime after shutdown.

It does not encode capacity adequacy, reserve adequacy, minimum-output
adequacy, continuous ramp feasibility, network power flow, or N-1 security.

Stage 2 therefore remains an active Boolean UC logic oracle with an approximate
value-function threshold, not a complete SCUC physical-feasibility oracle.
