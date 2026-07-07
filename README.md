# Small-Sample Gate-Level Max-Affine Grover Adaptive Search for UC

This repository is now focused on a compact quantum-potential validation path for unit commitment (UC):

1. sample a small selected-generator commitment subregister,
2. evaluate only those sampled commitments with exact ED/LP,
3. learn an integer max-affine surrogate oracle from the samples,
4. build a Qiskit gate-level value-register comparator oracle,
5. run BBHT/Grover adaptive search with shot-based circuit execution,
6. verify measured candidates classically with exact ED/LP before accepting incumbent updates.

The current experiment does **not** use full-enumeration training. A hidden exact reference may be computed for reporting quality, but it is kept under `hidden_reference_not_used_by_algorithm` and is not used to train the oracle, calibrate thresholds, update incumbents, or stop the search.

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

If `--save-qasm true` is enabled, a reference QASM/text circuit dump is written to:

```text
results/stage1_case14_t2_small_sample_gate_level_max_affine_gas.qasm
```

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
- hidden reference accounting outside algorithmic ED/LP calls.
