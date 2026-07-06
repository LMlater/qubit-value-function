# T=2 固定负荷 case14 小样本门级 Grover Oracle 批量验证报告

## 1. 实验目标

在固定负荷曲线 d 下，批量验证多个 T=2 小子空间中的 Grover value-function oracle。搜索变量是选定机组的启停承诺 x，负荷 d 和未选机组承诺只作为条件量/嵌入验证量。

核心 oracle 形式：

```text
V_hat_int(x) = max_r L_r(x)
O_tau |x> = (-1)^[V_hat_int(x) <= tau] |x>
```

## 2. 数据与全局参照

- 算例：case14_T2，来源：`data\case14.json.gz`
- 固定负荷 MW：[310.50480000000005, 291.96720999999997]
- 完整 T=2 commitment bits：12
- 完整枚举有限可行状态数：768
- 完整 T=2 穷举最优：110011111100，真实成本 20578.215260

## 3. 批量筛选结果

- 筛选子空间数：15
- 请求目标集合大小：[1, 3]
- 各目标集合精确分离数量：{'1': 15, '3': 3}
- 同时精确分离全部请求目标的子空间数：3

| 子空间 | 最优 selected bitstring | 整数权重 | top-1 | top-3 |
|---|---:|---:|---:|---:|
| g1/g2 | 1100 | [7, 6, 7, 1] | 精确 | 不精确 |
| g1/g3 | 1111 | [7, 6, 7, 7] | 精确 | 不精确 |
| g1/g4 | 1111 | [7, 6, 5, 4] | 精确 | 精确 |
| g1/g5 | 1111 | [7, 6, 5, 4] | 精确 | 精确 |
| g1/g6 | 1100 | [7, 6, 1, 1] | 精确 | 精确 |
| g2/g3 | 0011 | [7, 7, 7, 7] | 精确 | 不精确 |
| g2/g4 | 0011 | [7, 1, 7, 6] | 精确 | 不精确 |
| g2/g5 | 0011 | [7, 1, 7, 6] | 精确 | 不精确 |
| g2/g6 | 0000 | [7, 5, 7, 7] | 精确 | 不精确 |
| g3/g4 | 1111 | [7, 7, 7, 6] | 精确 | 不精确 |
| g3/g5 | 1111 | [7, 7, 7, 6] | 精确 | 不精确 |
| g3/g6 | 1100 | [7, 7, 7, 7] | 精确 | 不精确 |
| g4/g5 | 1111 | [7, 6, 7, 6] | 精确 | 不精确 |
| g4/g6 | 1100 | [7, 6, 1, 1] | 精确 | 不精确 |
| g5/g6 | 1100 | [7, 6, 1, 1] | 精确 | 不精确 |

## 4. 门级 Grover 仿真结果

下面这些子空间完成了 Qiskit statevector 门级 oracle + Grover diffuser 仿真。

| 子空间 | 目标 | tau | 标记 selected states | Grover 后标记概率 | aux 回零概率 | qubits | depth | argmax 真实成本 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| g1/g6 | top-1 | 0 | 1100 | 0.961319 | 1.000000000000 | 21 | 56 | 20578.215260 |
| g4/g6 | top-1 | 0 | 1100 | 0.961319 | 1.000000000000 | 21 | 56 | 20578.215260 |
| g5/g6 | top-1 | 0 | 1100 | 0.961319 | 1.000000000000 | 21 | 56 | 20578.215260 |

## 5. 每个门级实验的电路与 oracle 说明

### 5.1 子空间 g1/g6

- 搜索寄存器 x：['g1_t0', 'g1_t1', 'g6_t0', 'g6_t1']
- 学得整数权重：[7, 6, 1, 1]
- 值函数代理：V_hat_int(x) = max_r L_r(x)
- 阈值 oracle：O_tau |x> = (-1)^[V_hat_int(x) <= tau] |x>
- 仿射片段：
  - L0_g1_learned_mismatch: 7*(1-g1_t0) + 6*(1-g1_t1)
  - L1_g6_learned_mismatch: 1*g6_t0 + 1*g6_t1
- 电路寄存器：
  - x：['g1_t0', 'g1_t1', 'g6_t0', 'g6_t1']
  - value registers：['v0 stores L0(x) before comparison', 'v1 stores L1(x) before comparison']
  - flag qubits：['flag0 stores [L0(x) <= tau]', 'flag1 stores [L1(x) <= tau]']
- compute-phase-uncompute 顺序：
  - Apply Hadamard gates on all x qubits to create the Grover uniform superposition.
  - For each learned affine piece L_r, temporarily flip inverted x inputs so (1-x_i) terms become addable x_i terms.
  - Use WeightedAdder to write the integer piece value L_r(x) into value register v_r.
  - Use IntegerComparator to write flag_r = [L_r(x) <= tau].
  - Apply a multi-controlled phase on all flag qubits, implementing max_r L_r(x) <= tau as AND_r flag_r.
  - Run inverse comparators and inverse adders, restoring all value and ancilla registers to zero.
  - Apply the standard Grover diffuser on x only.
- 可逆性检查：The oracle is U_f^dagger Z_flags U_f, so it is unitary and leaves only a phase on x.
- 真实成本验证：The most amplified selected bitstring is embedded back into the full T=2 commitment and checked with the exact UC evaluator.

### 5.2 子空间 g4/g6

- 搜索寄存器 x：['g4_t0', 'g4_t1', 'g6_t0', 'g6_t1']
- 学得整数权重：[7, 6, 1, 1]
- 值函数代理：V_hat_int(x) = max_r L_r(x)
- 阈值 oracle：O_tau |x> = (-1)^[V_hat_int(x) <= tau] |x>
- 仿射片段：
  - L0_g4_learned_mismatch: 7*(1-g4_t0) + 6*(1-g4_t1)
  - L1_g6_learned_mismatch: 1*g6_t0 + 1*g6_t1
- 电路寄存器：
  - x：['g4_t0', 'g4_t1', 'g6_t0', 'g6_t1']
  - value registers：['v0 stores L0(x) before comparison', 'v1 stores L1(x) before comparison']
  - flag qubits：['flag0 stores [L0(x) <= tau]', 'flag1 stores [L1(x) <= tau]']
- compute-phase-uncompute 顺序：
  - Apply Hadamard gates on all x qubits to create the Grover uniform superposition.
  - For each learned affine piece L_r, temporarily flip inverted x inputs so (1-x_i) terms become addable x_i terms.
  - Use WeightedAdder to write the integer piece value L_r(x) into value register v_r.
  - Use IntegerComparator to write flag_r = [L_r(x) <= tau].
  - Apply a multi-controlled phase on all flag qubits, implementing max_r L_r(x) <= tau as AND_r flag_r.
  - Run inverse comparators and inverse adders, restoring all value and ancilla registers to zero.
  - Apply the standard Grover diffuser on x only.
- 可逆性检查：The oracle is U_f^dagger Z_flags U_f, so it is unitary and leaves only a phase on x.
- 真实成本验证：The most amplified selected bitstring is embedded back into the full T=2 commitment and checked with the exact UC evaluator.

### 5.3 子空间 g5/g6

- 搜索寄存器 x：['g5_t0', 'g5_t1', 'g6_t0', 'g6_t1']
- 学得整数权重：[7, 6, 1, 1]
- 值函数代理：V_hat_int(x) = max_r L_r(x)
- 阈值 oracle：O_tau |x> = (-1)^[V_hat_int(x) <= tau] |x>
- 仿射片段：
  - L0_g5_learned_mismatch: 7*(1-g5_t0) + 6*(1-g5_t1)
  - L1_g6_learned_mismatch: 1*g6_t0 + 1*g6_t1
- 电路寄存器：
  - x：['g5_t0', 'g5_t1', 'g6_t0', 'g6_t1']
  - value registers：['v0 stores L0(x) before comparison', 'v1 stores L1(x) before comparison']
  - flag qubits：['flag0 stores [L0(x) <= tau]', 'flag1 stores [L1(x) <= tau]']
- compute-phase-uncompute 顺序：
  - Apply Hadamard gates on all x qubits to create the Grover uniform superposition.
  - For each learned affine piece L_r, temporarily flip inverted x inputs so (1-x_i) terms become addable x_i terms.
  - Use WeightedAdder to write the integer piece value L_r(x) into value register v_r.
  - Use IntegerComparator to write flag_r = [L_r(x) <= tau].
  - Apply a multi-controlled phase on all flag qubits, implementing max_r L_r(x) <= tau as AND_r flag_r.
  - Run inverse comparators and inverse adders, restoring all value and ancilla registers to zero.
  - Apply the standard Grover diffuser on x only.
- 可逆性检查：The oracle is U_f^dagger Z_flags U_f, so it is unitary and leaves only a phase on x.
- 真实成本验证：The most amplified selected bitstring is embedded back into the full T=2 commitment and checked with the exact UC evaluator.

## 6. 当前结论

当前批量结果说明，在 T=2 固定负荷 case14 的多个小子空间中，可以从真实 UC 成本局部敏感性构造非查表的整数 max-affine oracle，并在门级 Qiskit 电路中完成 compute -> compare -> phase -> uncompute -> diffuse。
该结果仍是小样本 proof-of-concept，不声称已经完成完整 12-bit、207-feature、32-piece T=2 门级综合；它的作用是证明研究内容1的 oracle 架构、可逆性、Grover 放大和真实 UC 成本验证链条是闭合的。
