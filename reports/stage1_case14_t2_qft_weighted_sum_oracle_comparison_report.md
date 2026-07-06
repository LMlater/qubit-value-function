# QFT Weighted-Sum 与 WeightedAdder Grover Oracle 对照实验

## 1. 实验目的

本实验固定 case14 T=2 负荷曲线，在 g1/g6 小子空间上比较两种整数加权求和电路：当前主线使用的 Qiskit WeightedAdder，以及借鉴文献思路实现的 QFT-based weighted sum。

两条路线实现同一个 max-affine 阈值 oracle：

```text
V_hat_int(x) = max_r L_r(x)
O_tau |x> = (-1)^[V_hat_int(x) <= tau] |x>
```

## 2. 样本与学习到的整数片段

- selected generators: ['g1', 'g6']
- bit order: selected generator-major order: g_i_t0,g_i_t1,...
- selected-subspace optimum: 1100
- embedded full optimum: 110011111100
- true cost: 20578.215260

- L0_g1_learned_mismatch: 7*(1-g1_t0) + 6*(1-g1_t1)
- L1_g6_learned_mismatch: 1*g6_t0 + 1*g6_t1

## 3. 电路设计

| 路线 | 求和方式 | 值寄存器含义 | 特点 |
|---|---|---|---|
| WeightedAdder | 用 carry/control 辅助位完成整数加法 | 计算基中的 value register | 当前主线，结构直观，依赖 Qiskit 算术模块 |
| QFT weighted sum | QFT 后用受控相位旋转累加权重，再 inverse QFT | inverse QFT 后的计算基 value register | 更接近文献中的 QFT 加权求和思想，可能节省辅助位 |

## 4. 结果对比

| 目标 | 路线 | tau | marked states | marked probability | aux 回零概率 | phase error | qubits | depth | decomposed depth | argmax true cost |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| top-1 | WeightedAdder | 0 | 1100 | 0.961319 | 1.000000000000 | 4.44e-16 | 21 | 56 | 309 | 20578.215260 |
| top-1 | QFT weighted sum | 0 | 1100 | 0.961319 | 1.000000000000 | 2.44e-15 | 16 | 49 | 319 | 20578.215260 |
| top-3 | WeightedAdder | 1 | 1100, 1110, 1101 | 0.949219 | 1.000000000000 | 4.44e-16 | 21 | 20 | 103 | 20578.215260 |
| top-3 | QFT weighted sum | 1 | 1100, 1110, 1101 | 0.949219 | 1.000000000000 | 2.44e-15 | 16 | 17 | 109 | 20578.215260 |

## 5. 门类型与操作统计

下面列出 Grover 电路的高层 operation counts；更细的分解门统计可查看 JSON 结果中的 resources.grover_circuit.decomposed_operations。

| 目标 | 路线 | 高层 operations |
|---:|---|---|
| top-1 | WeightedAdder | `{'x': 48, 'h': 40, 'adder': 6, 'cmp': 6, 'cmp_dg': 6, 'adder_dg': 6, 'cx': 3, 'mcx': 3}` |
| top-1 | QFT weighted sum | `{'h': 40, 'x': 24, 'cmp': 6, 'cmp_dg': 6, 'qft_sum_L0_g1_learned_mismatch': 3, 'qft_sum_L1_g6_learned_mismatch': 3, 'cx': 3, 'qft_sum_L1_g6_learned_mismatch_dg': 3, 'qft_sum_L0_g1_learned_mismatch_dg': 3, 'mcx': 3}` |
| top-3 | WeightedAdder | `{'h': 16, 'x': 16, 'adder': 2, 'cmp': 2, 'cmp_dg': 2, 'adder_dg': 2, 'cx': 1, 'mcx': 1}` |
| top-3 | QFT weighted sum | `{'h': 16, 'x': 8, 'cmp': 2, 'cmp_dg': 2, 'qft_sum_L0_g1_learned_mismatch': 1, 'qft_sum_L1_g6_learned_mismatch': 1, 'cx': 1, 'qft_sum_L1_g6_learned_mismatch_dg': 1, 'qft_sum_L0_g1_learned_mismatch_dg': 1, 'mcx': 1}` |

## 6. QFT oracle 的 compute-phase-uncompute

- Apply Hadamard gates on all x qubits to create the Grover uniform superposition.
- For each affine piece L_r, temporarily flip inverted x inputs for (1-x_i) terms.
- Apply QFT to the piece value register.
- For every nonzero integer coefficient w_i, apply controlled phase rotations from x_i to every value-register Fourier qubit.
- Apply inverse QFT so the computational-basis value register stores L_r(x).
- Use IntegerComparator to write flag_r = [L_r(x) <= tau].
- Apply a multi-controlled phase on all flag qubits, implementing max_r L_r(x) <= tau as AND_r flag_r.
- Run inverse comparators and inverse QFT weighted-sum gates, restoring all value and ancilla registers to zero.
- Apply the standard Grover diffuser on x only.

## 7. 小结

两种加权求和方式实现了同一个 max-affine threshold oracle，得到相同的 marked set、相近的 Grover marked probability，并都能通过真实 UC 成本验证最优启停状态。在该小样本中，QFT weighted sum 使用更少量子比特，但引入 QFT/IQFT 和受控相位旋转；WeightedAdder 路线更直观，适合作为当前主线基准。
