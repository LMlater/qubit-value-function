# Formal Exclude-Hidden Small-Sample Gate-Level GAS Sweep Summary

## Experiment Setup

- backend: qasm
- shots: 2000
- seed range: 0-19
- 4q configs: 0,5; 0,1; 1,5; 2,5; 0,2
- 6q configs: 0,1,5; 0,2,5; 1,2,5; 0,1,2
- 4q train sample counts: 4, 6, 8, 12
- 6q train sample counts: 8, 12, 16
- max rounds: 12
- max trials per threshold: 12
- max candidates per shot batch: 1
- training hidden optimum exclusion: enabled
- random initial hidden optimum exclusion: enabled

The results are a small-scale gate-level quantum-potential validation under limited ED/LP supervision. Hidden full subspace enumeration is used only for evaluation. Algorithmic ED/LP calls include only training samples and measured candidates. The sweep evaluates selected-generator subspace optimum recovery, not full 12-bit case14 T=2 global optimization.

## Overall Statistics

- total runs: 640
- ok runs: 640
- error runs: 0
- overall success rate over ok runs: 35.5%
- overall average algorithmic ED/LP calls: 15.622
- overall average circuit executions: 17
- overall average total shots: 34000
- overall average max qubits: 16.869
- overall average max transpiled depth: 320.219

## Grouped Summary

| selected_generators | num_search_qubits | train_sample_count | num_ok_runs | num_success | success_rate_hidden_not_train_not_initial | avg_algorithmic_ed_lp_calls | avg_circuit_executions | avg_total_shots | avg_max_qubits | avg_max_transpiled_depth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,1 | 4 | 4 | 20 | 10 | 50.0% | 10.150 | 16.150 | 32300.000 | 17.600 | 336.900 |
| 0,1 | 4 | 6 | 20 | 8 | 40.0% | 11.250 | 15.350 | 30700.000 | 15.800 | 251.100 |
| 0,1 | 4 | 8 | 20 | 10 | 50.0% | 11.950 | 15.400 | 30800.000 | 15.000 | 187.150 |
| 0,1 | 4 | 12 | 20 | 9 | 45.0% | 14.050 | 16.150 | 32300.000 | 15.200 | 203.450 |
| 0,2 | 4 | 4 | 20 | 10 | 50.0% | 9.950 | 16.050 | 32100.000 | 17.600 | 350.550 |
| 0,2 | 4 | 6 | 20 | 9 | 45.0% | 10.800 | 15.000 | 30000.000 | 15.200 | 189.700 |
| 0,2 | 4 | 8 | 20 | 9 | 45.0% | 11.550 | 15.900 | 31800.000 | 14.800 | 158.350 |
| 0,2 | 4 | 12 | 20 | 9 | 45.0% | 13.950 | 15.100 | 30200.000 | 14.000 | 91.050 |
| 0,5 | 4 | 4 | 20 | 8 | 40.0% | 10.300 | 16.500 | 33000.000 | 16.400 | 298.850 |
| 0,5 | 4 | 6 | 20 | 8 | 40.0% | 10.500 | 17.500 | 35000.000 | 14.800 | 161.800 |
| 0,5 | 4 | 8 | 20 | 8 | 40.0% | 11.850 | 17.200 | 34400.000 | 14.200 | 103.700 |
| 0,5 | 4 | 12 | 20 | 8 | 40.0% | 13.900 | 17.700 | 35400.000 | 14.000 | 75.850 |
| 1,5 | 4 | 4 | 20 | 10 | 50.0% | 10.250 | 16.400 | 32800.000 | 16.200 | 268.450 |
| 1,5 | 4 | 6 | 20 | 8 | 40.0% | 10.700 | 15.650 | 31300.000 | 15.200 | 181.450 |
| 1,5 | 4 | 8 | 20 | 8 | 40.0% | 11.600 | 15.750 | 31500.000 | 14.600 | 139.900 |
| 1,5 | 4 | 12 | 20 | 8 | 40.0% | 13.750 | 16.400 | 32800.000 | 14.000 | 79.400 |
| 2,5 | 4 | 4 | 20 | 12 | 60.0% | 10.650 | 17.700 | 35400.000 | 16.000 | 264.250 |
| 2,5 | 4 | 6 | 20 | 10 | 50.0% | 10.650 | 16.850 | 33700.000 | 16.000 | 228.750 |
| 2,5 | 4 | 8 | 20 | 7 | 35.0% | 11.650 | 16.350 | 32700.000 | 14.800 | 154.100 |
| 2,5 | 4 | 12 | 20 | 9 | 45.0% | 13.650 | 15.900 | 31800.000 | 14.000 | 77.800 |
| 0,1,2 | 6 | 8 | 20 | 4 | 20.0% | 19.250 | 18.550 | 37100.000 | 20.400 | 586.200 |
| 0,1,2 | 6 | 12 | 20 | 0 | 0.0% | 21.600 | 18.500 | 37000.000 | 19.800 | 627.000 |
| 0,1,2 | 6 | 16 | 20 | 5 | 25.0% | 25.050 | 18.750 | 37500.000 | 18.400 | 437.500 |
| 0,1,5 | 6 | 8 | 20 | 3 | 15.0% | 19.500 | 18.650 | 37300.000 | 21.000 | 594.250 |
| 0,1,5 | 6 | 12 | 20 | 4 | 20.0% | 21.150 | 16.650 | 33300.000 | 20.000 | 672.800 |
| 0,1,5 | 6 | 16 | 20 | 4 | 20.0% | 25.750 | 18.950 | 37900.000 | 18.200 | 353.250 |
| 0,2,5 | 6 | 8 | 20 | 5 | 25.0% | 19.700 | 17.650 | 35300.000 | 21.800 | 701.200 |
| 0,2,5 | 6 | 12 | 20 | 5 | 25.0% | 22.200 | 17.950 | 35900.000 | 19.600 | 608.700 |
| 0,2,5 | 6 | 16 | 20 | 5 | 25.0% | 24.750 | 17.150 | 34300.000 | 18.200 | 427.750 |
| 1,2,5 | 6 | 8 | 20 | 5 | 25.0% | 21.400 | 20.150 | 40300.000 | 20.000 | 518.550 |
| 1,2,5 | 6 | 12 | 20 | 4 | 20.0% | 21.400 | 17.000 | 34000.000 | 18.800 | 511.950 |
| 1,2,5 | 6 | 16 | 20 | 5 | 25.0% | 25.050 | 19.050 | 38100.000 | 18.200 | 405.300 |

## Observations

- The 4-qubit subspaces show success rates from 35.0% to 60.0%, with an average grouped rate of 44.5%. The strongest 4q row is selected generators 2,5 with train_sample_count=4.
- The 6-qubit subspaces are clearly harder in this prototype, with success rates from 0.0% to 25.0% and an average grouped rate of 20.4%. The strongest 6q row is selected generators 0,1,2 with train_sample_count=16.
- Increasing train_sample_count does not monotonically improve success rate in this sweep. The learned integer max-affine oracle and shot-based GAS behavior remain sensitive to the selected subspace and random seed.
- Resource use rises from 4q to 6q: average max qubits increase from 15.270 to 19.533, and average max transpiled depth increases from 190.127 to 537.038.
- There is a practical tradeoff between success rate and resource/call budget. Larger 6q runs can use more circuit depth and ED/LP checks without consistently improving recovery in this limited-supervision setting.
