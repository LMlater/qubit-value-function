# 稀疏 VQC 的 BBHT 自适应 Grover 搜索

## 目标

本阶段把已经验证的稀疏 VQC 相干值函数 oracle、普通 Grover 和真实 ED/LP 校验连接成多轮自适应搜索：

```text
训练稀疏相位 VQC
-> 固定点稀疏值模型
-> coherent phase-to-value
-> threshold comparator oracle
-> BBHT 随机 Grover 迭代
-> Aer MPS 单次测量
-> exact ED/LP cache / 新 ED/LP
-> 真实改善时更新 incumbent 和 threshold
```

与上一阶段固定 threshold 的普通 Grover 不同，本阶段会在真实候选严格改善后不断缩小阈值。

## 正式成本语义

量子 oracle 使用逐项固定点量化后的稀疏整数模型：

```text
k_theta(x) = a0 + sum_j a_j f_j(x)
```

比较条件为：

```text
k_theta(x) < encode(C_ED(incumbent))
```

浮点 VQC 总预测值再整体取整只用于搜索结束后的量化诊断，不控制电路、候选或 threshold 更新。

## BBHT 随机迭代

搜索空间大小为：

```text
N = 2^(G*T)
```

初始化：

```text
m = 1
lambda = 1.2
m_cap = ceil(sqrt(N))
```

第 r 个 trial 随机选择：

```text
j ~ Uniform{0, ..., m-1}
```

随后执行 j 次普通 Grover 迭代并测量一次。正式循环固定：

```text
shots_per_trial = 1
```

多 shots 概率诊断仍保留在普通 Grover 阶段，不用于 BBHT 主循环。

trial 失败后：

```text
m = min(m_cap, ceil(lambda*m))
```

代码同时保证 m 至少增加 1，避免整数取整后停滞。真实 incumbent 改善后：

```text
m = 1
```

重新从较小随机窗口搜索新 threshold 下的候选。

## 不使用 marked count

`run_sparse_vqc_bbht` 不调用：

```text
ordinary_grover_validation_plan
marked count
完整状态 marked mask
```

每轮只根据当前真实 incumbent 生成 comparator threshold，再随机选择 Grover 次数。

2×2 实验在算法结束后允许枚举16个状态，用于报告：

- 实际 sparse-integer marked 集合；
- direct-rounded-float 诊断集合；
- 真实全局最优；
- 最终真实最优性差距。

这些结果不允许反馈到 BBHT 主循环。

普通 Grover 的 validation enumeration 还设置：

```text
max_validation_qubits = 12
```

超过限制会拒绝执行，防止未来误用于较大模型。

## Exact cache

训练阶段取得的 ED/LP 标签直接进入 exact cache：

```text
index -> success, true cost, source, message
```

BBHT 测量到候选后：

1. 先用稀疏整数模型判断该单一候选是否 marked；
2. unmarked 候选不调用 ED/LP；
3. 已存在于 cache 的 marked 候选直接复用真实结果；
4. 新 marked 候选执行一次 ED/LP 并写入 cache；
5. 后续重复命中不重复调用 ED/LP。

第一版不在量子 oracle 中逐状态排除已验证候选，因为那会使线路深度随 cache 大小增长。重复候选通过 exact cache 和预算控制。

## 真实 threshold 更新

当前真实 incumbent 为 x_inc 时：

```text
true_threshold = C_ED(x_inc)
encoded_threshold = fixed_point.encode(true_threshold)
```

只有候选 exact ED/LP 满足：

```text
C_ED(x_candidate) < C_ED(x_inc)
```

才执行：

```text
incumbent = candidate
true_threshold = candidate true ED/LP cost
encoded_threshold = encode(true_threshold)
m = 1
```

VQC 参数不重新训练，只改变 comparator threshold。

代理模型判为 marked 但真实成本不改善时：

```text
threshold_updated = false
m 继续增长
```

这正是上一阶段 window 2 / best-training 所暴露的代理 false positive 所需处理方式。

## 固定点量化停滞

可能出现真实成本严格下降，但两次成本编码为同一整数：

```text
C_new < C_old
encode(C_new) == encode(C_old)
```

此时真实 incumbent 仍然更新，但量子 comparator 没有变窄。记录：

```text
encoded_threshold_changed = false
quantization_stagnation = true
```

连续同码更新达到：

```text
max_same_encoded_threshold_updates
```

后停止，避免在相同 oracle threshold 下无限循环。

## 无枚举确定性停止

稀疏整数模型的保守下界为：

```text
L = a0 + sum_j min(0, a_j)
```

若：

```text
L >= encoded_threshold
```

则所有输入都不可能满足严格比较条件，可以不运行任何量子电路而停止：

```text
no_surrogate_marked_state_by_conservative_lower_bound
```

该判断只遍历稀疏局部项，适用于后续更大 G×T 构造。

## 预算

默认预算：

```text
max_trials = 64
max_oracle_calls = 128
max_new_ed_lp_calls = 16
max_threshold_updates = 8
max_consecutive_nonimproving_marked = 16
max_same_encoded_threshold_updates = 3
```

分别统计：

- circuit executions；
- total shots；
- Grover oracle calls；
- diffuser calls；
- new ED/LP calls；
- cached exact lookups；
- threshold updates；
- 重复候选命中次数。

总 oracle 调用量为所有 trial 的随机迭代次数之和：

```text
Q = sum_r j_r
```

`j=0` 的 trial 仍是一次实际电路执行和一次测量，但不增加 oracle / diffuser 调用数。

## 停止原因

每次运行必须返回一个明确停止原因：

```text
no_surrogate_marked_state_by_conservative_lower_bound
max_trials_reached
max_oracle_calls_reached
max_new_ed_lp_calls_reached
max_threshold_updates_reached
max_consecutive_nonimproving_marked_reached
same_encoded_threshold_update_limit
```

## 2×2 实验

默认验证 case14 派生的三个独立窗口：

```text
window 0: source t0-t1
window 1: source t1-t2
window 2: source t2-t3
```

可变机组仍为 g1 和 g6，训练样本数为8，初始策略默认为 `first`，以便观察多次 threshold 下降。

运行：

```powershell
python experiments/stage1_case14_2x2_sparse_vqc_bbht.py `
  --selected-generators 0,5 `
  --window-starts 0,1,2 `
  --train-sample-count 8 `
  --initialization-policy first `
  --fractional-bits 2 `
  --cost-unit 1000 `
  --seed 0
```

输出：

```text
results/stage1_case14_2x2_sparse_vqc_bbht.json
```

每个窗口记录完整 trial trace、threshold history、exact cache、预算计数、停止原因，以及搜索结束后的16状态验证结果。

## 当前限制

- 当前使用 Aer MPS，不是量子硬件；
- 当前 VQC 在搜索期间固定，不做在线增量训练；
- exact cache 只做经典去重，没有量子 tabu/exclusion oracle；
- 保守下界安全但可能偏松；
- 2×2 全局最优只作为搜索后验证；
- 3×2和3×3尚未运行完整 BBHT 实验；
- 当前结果不能用于声称量子优势。
