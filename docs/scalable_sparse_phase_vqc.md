# 可扩展稀疏对角相位 VQC：2×2 训练与相位编码阶段

## 当前目标

本阶段在 case14 派生的三个独立 2台机组×2时间步场景中验证：

1. 稀疏局部相位 VQC 能否拟合固定启停方案的 ED/LP 总成本；
2. 学得的参数能否被真实 Qiskit 对角相位门编码；
3. 计算基态相位和 Hadamard readout 是否与解析模型一致；
4. 模型结构是否按局部特征数增长，而不是依赖 `2^(G*T)` 状态表。

本阶段尚未实现 phase-to-value 固定点寄存器，也尚未接入 Grover 或自适应 Grover。

## 场景定义

默认实例为 `data/case14.json.gz`，可变机组为索引 `(0, 5)`，即 `g1` 和 `g6`。其余机组在两个局部时间步内固定开机。

默认验证三个窗口：

- `start=0`：源负荷/备用序列的第 0、1 个时间步；
- `start=1`：第 1、2 个时间步；
- `start=2`：第 2、3 个时间步。

每个 shifted window 都保留源实例的机组 `initial_status` 和 `initial_power`。因此它们是三个独立负荷/备用场景，不是同一条启停轨迹的连续滚动窗口。

## 模型

对 `G` 台可变机组和 `T` 个时间步，搜索位采用机组优先顺序：

```text
q(g,t) = g*T + t
```

相位模型为：

```text
phi_theta(x)
  = b
  + sum_i w_i x_i
  + sum_(g,t) w_time[g,t] x[g,t] x[g,t+1]
  + sum_((g,h),t) w_edge[g,h,t] x[g,t] x[h,t]
```

默认机组关系图使用最近邻链。参数和门数量为：

```text
O(GT + G(T-1) + |E_G|T)
```

而不是 `O(2^(GT))`。

量子线路使用：

- 单变量项：`P(2*pi*w)`；
- 双变量项：`CP(2*pi*w)`；
- 截距：作为模型整体相位；在 Hadamard readout 中转化为 readout qubit 上的相位。

## 训练

训练样本只包含按代表性顺序取得的有限 ED/LP 结果。默认每个场景使用 8 个训练状态。

成本先映射到远离模 1 边界的相位区间。优化目标为无噪声 Hadamard test 期望的解析等价形式：

```text
mean |exp(2*pi*i*phi_theta(x)) - exp(2*pi*i*phi_target(x))|^2
```

这一步使用解析梯度减少小规模验证的运行时间。训练后再用真实 Qiskit `Statevector` 分别验证：

1. 计算基态经过相位线路后的复相位；
2. readout ancilla 的 `<X> + i<Y>` Hadamard 期望。

隐藏状态的 ED/LP 标签在 `fit_sparse_phase_vqc` 返回之后才计算，不参与参数训练。

## 运行

```powershell
python experiments/stage1_case14_2x2_sparse_phase_vqc.py `
  --selected-generators 0,5 `
  --window-starts 0,1,2 `
  --horizon 2 `
  --train-sample-count 8 `
  --seed 0
```

结果写入：

```text
results/stage1_case14_2x2_sparse_phase_vqc.json
```

## 输出指标

每个窗口记录：

- 训练初始/最终 phase loss；
- train、hidden 和全部有限评估状态的 MAE、RMSE；
- 常数基线 MAE；
- pairwise ranking accuracy；
- true/predicted top-1；
- 解析相位与 basis-Statevector 相位最大误差；
- 解析相位与 Hadamard readout 最大误差；
- 相位线路及 readout 线路的 qubit、depth 和 gate counts；
- 算法训练 ED/LP 调用数与隐藏评估 ED/LP 调用数。

## 与后续 Grover 的关系

本阶段产生的是一个可按 `G`、`T` 和稀疏机组图构造的对角相位线路族。下一阶段需要将其相干地转换为固定点值寄存器，再接入：

```text
phase-to-value
-> IntegerComparator
-> phase marking
-> uncompute
-> Grover / adaptive Grover
-> measured candidate
-> exact ED/LP verification
```

Grover threshold 仍只能由真实 ED/LP incumbent 更新。

## 当前限制

- 当前实验脚本只运行 2×2 场景，但核心模型 builder 支持其他 `G×T`；
- 三个时间窗口分别训练独立模型，尚未把负荷作为同一个 VQC 的连续输入；
- 当前训练使用精确解析期望，尚未使用有限 shots；
- 尚未处理相位周期到无符号固定点值代码的相干解码；
- 尚未接入 Grover、自适应 Grover或完整网络安全约束；
- 当前结果不能用于声称量子优势。
