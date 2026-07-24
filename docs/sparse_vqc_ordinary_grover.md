# 稀疏 VQC 普通 Grover 验证

## 阶段目标

本阶段把已经通过验证的相干值函数 oracle 接入普通 Grover：

```text
sparse phase VQC
-> coherent fixed-point value register
-> IntegerComparator
-> phase marking
-> uncompute
-> search-register diffuser
-> Aer MPS shots
-> measured candidate
-> exact ED/LP verification
```

本阶段只验证固定 threshold 下的振幅放大。它不实现 BBHT，不在同一次实验中更新 threshold，也不声称已经解决未知目标态数量问题。

## 正式 oracle 成本语义

量子 oracle 使用逐项固定点量化后的稀疏整数模型：

```text
k_sparse(x) = a0 + sum_j a_j f_j(x)
```

其中 `f_j(x)` 是单变量、同机组相邻时间耦合和稀疏机组边耦合等局部布尔特征。

浮点 VQC 的完整预测值再整体取整：

```text
encode(C_float(x))
```

只作为量化诊断。它不能修正量子电路、候选或测量概率，否则会重新引入逐状态 lookup。

每个 threshold 同时记录：

- `sparse_integer_marked_indices`；
- `direct_rounded_float_marked_indices`；
- 交集和对称差；
- integer-only 与 direct-float-only 状态；
- 分类分歧数量。

## 普通 Grover 电路

`build_sparse_vqc_grover_circuit` 的 `iterations` 必须显式传入。核心 builder 不枚举搜索空间，也不自行计算 marked count。

电路为：

```text
H on x
repeat iterations times:
    coherent sparse-VQC threshold phase oracle
    H-X-multi-controlled-Z-X-H diffuser on x only
```

value、flag 和 comparator ancillas 只在 oracle 内部使用。diffuser 仅作用于搜索寄存器。

支持 `iterations=0`，此时电路只制备均匀搜索叠加态。

## 2×2 验证中的已知 marked count

当前 2 台机组×2 时间步只有 16 个搜索状态。为了验证普通 Grover，可以在实验层枚举稀疏整数模型并计算：

```text
M = marked count
iterations = floor(pi/4 * sqrt(16/M))
```

这个枚举只用于：

- 小规模验收；
- 选择普通 Grover 的迭代次数；
- 计算初始 marked probability；
- 检查量化分类差异。

它不参与 VQC 训练、固定点 bounds 或量子电路构造。JSON 明确记录：

```text
marked_count_known_for_validation_only = true
```

当 `M=0` 时，实验直接记录 `no_marked_state`，不启动 MPS。

## Aer MPS 测量

`execute_sparse_vqc_grover_mps` 使用：

```python
AerSimulator(method="matrix_product_state")
```

执行完整门级 Grover 电路并测量全部量子比特。返回：

- 原始全寄存器 `raw_counts`；
- 项目 little-endian 表示的 `x_counts`；
- 搜索状态概率；
- 所有辅助寄存器测量为零的概率；
- 运行时间；
- 总 qubits 和 Statevector 最低内存估算。

概率必须来自实际 Aer counts，不允许根据经典 marked 集合合成。

## 候选选择

候选必须满足：

1. 在实际 `x_counts` 中出现且 count 大于零；
2. 属于稀疏整数 oracle 的 marked 集合；
3. 优先选择尚未出现在训练 ED/LP 集合中的状态；
4. 在同一优先级内选择实际 count 最大的状态。

结果记录：

```text
candidate_selection_source = measured_gate_level_counts
```

经典 marked 集合只用于过滤 oracle 预测为改进的实际测量结果，不能直接指定一个未测量状态。

## ED/LP 校验与固定 threshold

候选会重新执行真实 ED/LP，并记录：

- `candidate_true_ed_lp_cost`；
- `incumbent_true_cost`；
- `would_improve_incumbent`。

本阶段即使候选真实成本更低，也保持：

```text
threshold_updated = false
encoded_threshold_before = encoded_threshold_after
```

普通 Grover 通过后，下一阶段才把真实 ED/LP 改进接入 BBHT 自适应 threshold 更新。

## 默认实验

```powershell
python experiments/stage1_case14_2x2_sparse_vqc_grover.py `
  --selected-generators 0,5 `
  --window-starts 0,1,2 `
  --train-sample-count 8 `
  --fractional-bits 2 `
  --cost-unit 1000 `
  --shots 4096 `
  --seed 0
```

结果路径：

```text
results/stage1_case14_2x2_sparse_vqc_grover.json
```

每个窗口验证：

- 第一个有限训练样本作为 incumbent；
- 最佳训练样本作为 incumbent。

当前三个窗口预期覆盖多个 marked、唯一 marked 和无 marked 三类情况。

## 测试范围

专项测试包括：

- builder 显式接收 iterations 且不枚举状态；
- diffuser 不作用于辅助位；
- 唯一 marked state 的精确 Statevector 放大；
- 三个 marked states 的精确 Statevector 放大；
- 无 marked validation plan；
- Aer MPS 与精确概率的一致性；
- 实际 counts 候选选择；
- ED/LP 校验不更新 threshold；
- 3×2、3×3只构造稀疏资源，不运行大 Statevector。

## 与 BBHT 的关系

普通 Grover 验证通过后，BBHT 阶段将增加：

```text
random iteration count
-> MPS measurement
-> measured candidate
-> exact ED/LP
-> true-improvement threshold update
-> retry and budget rules
```

BBHT 不会预先使用 marked count。

## 当前限制

- 真实运行仍是 Aer MPS 模拟，不是真实量子硬件；
- 2×2 中已知 marked count 只用于普通 Grover 验收；
- threshold 在本阶段固定；
- 浮点 VQC 与逐项量化整数模型可能产生少量分类分歧；
- 尚未实现 BBHT、搜索预算和自适应停止；
- 尚未证明量子优势。
