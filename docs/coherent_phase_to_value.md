# 稀疏相位 VQC 的相干 phase-to-value 编码

## 目标

普通 VQC 回归常以测量期望值作为输出。若在 Grover 叠加态内部直接测量成本，搜索寄存器会坍缩，因此不能把普通测量式回归器直接放入 Grover oracle。

本阶段使用已经训练并验证的稀疏对角相位 VQC，将预测成本表示成计算基态的本征相位，再通过无中间测量的相位估计结构写入固定点值寄存器：

```text
|x>|0_value>
  -> |x>|shifted_encoded_cost(x)>
```

随后使用 `IntegerComparator` 构造：

```text
phase-to-value
-> threshold comparator
-> Z phase marking
-> comparator inverse
-> phase-to-value inverse
```

当前阶段只验证相干值寄存器、比较器、相位标记和反计算，尚未接入 Grover 或自适应 Grover。

## 从相位模型到成本系数

稀疏相位模型为：

```text
phi(x) = phase_intercept + sum_j phase_weight[j] * feature_j(x)
```

成本反归一化关系为：

```text
cost(x)
  = cost_center
  + cost_scale * (phi(x) - phase_center) / phase_scale
```

因此真实成本系数为：

```text
cost_intercept
  = cost_center
  + cost_scale * (phase_intercept - phase_center) / phase_scale

cost_weight[j]
  = cost_scale * phase_weight[j] / phase_scale
```

使用统一 `FixedPointConfig` 对截距和每个局部项系数分别量化：

```text
integer_value(x)
  = integer_intercept
  + sum_j integer_weight[j] * feature_j(x)
```

默认配置仍为：

```text
cost_unit = 1000
fractional_bits = 2
one integer code = 250 USD
```

## 无枚举范围保护

所有局部特征均为布尔量，因此不需要遍历 `2^(G*T)` 个输入即可得到安全边界：

```text
lower_bound
  = integer_intercept + sum_j min(0, integer_weight[j])

upper_bound
  = integer_intercept + sum_j max(0, integer_weight[j])
```

设置：

```text
value_shift = -lower_bound
```

则保守范围满足：

```text
0 <= integer_value(x) + value_shift <= upper_bound - lower_bound
```

值寄存器宽度为：

```text
num_value_qubits = bit_length(upper_bound - lower_bound)
```

至少使用 1 个 value qubit。构造时会检查：

```text
shifted_upper_bound < 2**num_value_qubits
```

如果不满足则拒绝构造，避免相位模溢出。

这些边界可能因局部特征相关性而偏保守，但复杂度只与稀疏项数有关，不依赖全状态表。

## 整数相位单元

定义：

```text
U|x>
  = exp(
      2*pi*i*shifted_integer_value(x)
      / 2**num_value_qubits
    ) |x>
```

其中：

```text
shifted_integer_value(x)
  = integer_value(x) + value_shift
```

局部门结构：

- 常数项：在独立相位线路中是整体相位；在受控相位估计中显式施加到 evaluation qubit，成为可观测相对相位；
- 单变量项：`P`，受控幂中使用 `CP`；
- 二变量项：`CP`，受控幂中使用双控制 `PhaseGate`；
- 正负整数权重均通过正负旋转角直接支持。

核心线路构造只遍历稀疏局部项，不生成按状态索引的 lookup。

## 相干 phase-to-value

value register 首先制备均匀叠加。对第 `r` 个 evaluation qubit，施加整数相位单元的 `2**r` 次幂。由于所有相位项对易，不需要重复整个 VQC 电路，只需把每个相位角乘以 `2**r`。

流程：

```text
H on value register
-> controlled U^(1), U^(2), U^(4), ...
-> inverse QFTGate
```

当相位严格位于 `m` 位网格：

```text
phase = shifted_integer_value / 2**m
```

输出应为唯一整数代码：

```text
|x>|0> -> |x>|shifted_integer_value(x)>
```

线路不使用中间测量，可以在 Grover oracle 内完整反计算。

## QFT 的作用边界

这里的 inverse `QFTGate` 只用于：

```text
相位本征值 -> 二进制固定点值寄存器
```

原有固定点算术基线仍然使用：

```text
WeightedAdder + IntegerComparator
```

而不使用 QFT 做加法。两者作用不同。

## threshold comparator

真实 incumbent 成本先通过相同的 `FixedPointConfig` 编码为：

```text
encoded_threshold
```

value register 存储的是平移后的值，因此比较常数为：

```text
shifted_compare_value = encoded_threshold + value_shift
```

严格比较使用：

```text
shifted_value < shifted_compare_value
```

非严格比较通过将比较常数加 1 实现：

```text
shifted_value < shifted_compare_value + 1
```

完整 oracle：

```text
compute phase-to-value
-> IntegerComparator(geq=False)
-> Z on flag
-> comparator inverse
-> phase-to-value inverse
```

结束后要求 value register、flag 和 comparator ancillas 全部回到零态。

## 复杂度

设：

- 搜索位数为 `n = G*T`；
- 稀疏局部特征数为 `K`；
- value register 位数为 `m`。

受控相位项数量约为：

```text
O(m*K)
```

inverse QFT 的标准门数约为：

```text
O(m^2)
```

模型构造与线路构造均不需要 `O(2^(G*T))` lookup。

当前 2×2 实验遍历 16 个状态，只用于验证每个 basis code、叠加态配对和 comparator phase，不参与模型训练或电路构造。

## 三个 2×2 验证场景

默认仍使用 case14 派生的三个独立窗口：

- g1、g6，源时段 0–1；
- g1、g6，源时段 1–2；
- g1、g6，源时段 2–3。

每个场景：

1. 使用 8 个真实 ED/LP 样本训练稀疏相位 VQC；
2. 将训练模型转换为固定点整数局部模型；
3. 对全部 16 个状态验证 exact phase-to-value code；
4. 在搜索叠加态中验证 `(x, value(x))` 配对；
5. 验证 compute/inverse 后 value register 回零；
6. 使用第一个训练 incumbent 和最佳训练 incumbent 验证 threshold oracle；
7. 不运行 Grover。

## 下一步

本阶段通过后，普通 Grover 将直接复用：

```text
phase-to-value
-> IntegerComparator
-> phase marking
-> uncompute
-> diffuser
```

随后实现 BBHT 风格自适应 Grover：

```text
随机 Grover 迭代次数
-> MPS shots
-> measured candidate
-> exact ED/LP verification
-> only true improvement updates threshold
```

真实 threshold 始终来自 ED/LP incumbent，不由 VQC 预测值直接更新。

## 当前限制

- 当前固定点系数是逐项量化，整数模型与“先求总预测成本再量化”可能存在少量差异；实验会报告该差异；
- 保守上下界可能比真实可达范围宽，增加 value qubit 数；
- 当前只在 2×2 上执行完整 Statevector 验证；
- 3×2、3×3 暂时只构造资源，不运行大状态模拟；
- 尚未接入普通 Grover、BBHT 自适应 Grover和完整网络安全约束；
- 尚未证明量子优势。
