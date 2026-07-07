# Small-Sample Gate-Level Max-Affine Grover Adaptive Search for UC

This repository is now focused on a compact quantum-potential validation path for unit commitment (UC):

1. sample a small selected-generator commitment subregister,
2. evaluate only those sampled commitments with exact ED/LP,
3. learn an integer max-affine surrogate oracle from the samples,
4. build a Qiskit gate-level value-register comparator oracle,
5. run BBHT/Grover adaptive search with shot-based circuit execution,
6. verify measured candidates classically with exact ED/LP before accepting incumbent updates.

The current experiment does **not** use full-enumeration training. A hidden exact reference may be computed for reporting quality, but it is kept under `hidden_reference_not_used_by_algorithm` and is not used to train the oracle, calibrate thresholds, update incumbents, or stop the search.

This is still an integer value-register comparator prototype. It should not be described as a complete QFT-style signed fixed-point encoding, nor as a full 12-bit case14 T=2 global optimum verification. The experiment evaluates whether shot-based gate-level GAS can recover the hidden optimum inside a selected-generator subspace under limited ED/LP supervision.

The default demonstration is a 4-qubit selected-subregister case:

```text
selected generators = 0,5
search bits = g1_t0, g1_t1, g6_t0, g6_t1
```

Earlier non-gate-level exploration code is preserved in the Git branch:

```text
archive/pre-gate-exploration
```

## Main Experiment

Run the current main experiment:

```powershell
python experiments/stage1_case14_t2_small_sample_gate_level_max_affine_gas.py `
  --backend qasm `
  --shots 2000 `
  --selected-generators 0,5 `
  --train-sample-count 8 `
  --max-rounds 10 `
  --max-trials-per-threshold 10
```

The result is written to:

```text
results/stage1_case14_t2_small_sample_gate_level_max_affine_gas.json
```

For diagnostic sweeps, the main experiment also supports:

```powershell
--exclude-hidden-optimum-from-training
```

That option first computes the hidden subspace optimum for evaluation, then excludes that index from the training sample pool. It is a diagnostic stress test, not an algorithmic assumption.

The JSON summary records both algorithmic ED/LP calls and hidden-reference ED/LP calls. Algorithmic calls include only training samples plus measured candidates checked by ED/LP.

If `--save-qasm true` is enabled, a reference QASM/text circuit dump is written to:

```text
results/stage1_case14_t2_small_sample_gate_level_max_affine_gas.qasm
```

## Sweep Experiments

Run a smoke sweep:

```powershell
python experiments/stage1_case14_t2_small_sample_gate_level_gas_sweep.py `
  --backend qasm `
  --shots 1000 `
  --seed-start 0 `
  --seed-count 3 `
  --configs "0,5;0,1;0,1,5" `
  --train-sample-counts "4,8,12" `
  --max-rounds 8 `
  --max-trials-per-threshold 8 `
  --max-candidates-per-shotbatch 1 `
  --output-json results/stage1_case14_t2_small_sample_gate_level_gas_sweep_smoke.json `
  --output-csv results/stage1_case14_t2_small_sample_gate_level_gas_sweep_smoke.csv
```

Run the hidden-optimum-exclusion smoke sweep by adding:

```powershell
--exclude-hidden-optimum-from-training
```

The sweep output contains individual run rows plus grouped summaries by selected generators, search-qubit count, train sample count, and exclusion mode. The key diagnostic metric is `success_rate_when_hidden_optimum_not_in_training`.

## Current Structure

Core package files:

```text
qubit_value_function/
  commitment.py
  ed.py
  experiment_utils.py
  gate_level_oracle.py
  qft_weighted_sum_oracle.py
  uc_loader.py
```

Current experiments:

```text
experiments/
  stage1_case14_t2_small_sample_gate_level_max_affine_gas.py
  stage1_case14_t2_small_sample_gate_level_gas_sweep.py
  stage1_case14_t2_gate_level_grover_oracle.py
  stage1_case14_t2_gate_level_max_affine_oracle.py
  stage1_case14_t2_learned_small_max_affine_gate_level_oracle.py
```

Tests:

```text
tests/test_stage1.py
```

Data:

```text
data/case14.json.gz
data/aelmp_simple.json.gz
```

## Validation

Run:

```powershell
pytest -q
```

The tests cover:

- UC instance loading and ED/LP evaluation,
- selected-subregister embedding helpers,
- Qiskit gate-level affine and max-affine phase oracles,
- QFT weighted-sum oracle consistency,
- small-sample integer max-affine learning,
- qasm shot execution and bitstring mapping,
- adaptive search incumbent updates only after true ED/LP improvement,
- hidden reference accounting outside algorithmic ED/LP calls,
- hidden-optimum exclusion from training samples,
- sweep config parsing and grouped success-rate metrics.
