# 研究内容 1：阶段 A 证据完成

## 执行摘要

在冻结 selected-split 的 12 个四比特、16 状态子空间中，训练集外真值审计覆盖 96 个状态，得到 9 个真实严格改善状态。初始 joint oracle 覆盖 4 个正例场景；四个发现状态均是各自所选 16 状态子空间中的真实最优状态。这不是完整 UC 的全局最优声明。

## 技术链路

稀疏相位 VQC → 固定点系数/值寄存器 → 严格 `<` 比较器 → hard-logic joint oracle → Grover/BBHT → Aer MPS 门级模拟测量 → ED/LP 真值验证 → 正常阈值更新。真值只用于事后审计，不参与在线选择。

## N=16、M=1 固定 oracle

`case14-g0g1-w2-s1` 在阈值 136 下只标记 index 3。理论/精确电路/五个 4096-shot Aer MPS 均值的 marked 概率为：k=0 `0.062500 / 0.062500 / 0.062305`，k=1 `0.472656 / 0.472656 / 0.472656`，k=2 `0.908447 / 0.908447 / 0.909424`，k=3 `0.961319 / 0.961319 / 0.959619`。相位真值表只翻转目标状态，辅助寄存器归零概率为 1（数值误差内）。这是门级经典 MPS 模拟中的 oracle-query 证据，不是量子硬件、墙钟时间或端到端加速证据。

## 训练集外候选质量

必须区分三个不同的分母。

- 20/20：四个 initial joint-marked 场景各有 5 个 selected-split joint-BBHT run；每个 run 都获得训练集外严格改善，故 run 级成功率为 20/20。
- 4/4：这四个初始 unique joint-marked 状态都是真实严格改善，故 unique-state joint precision 为 4/4。
- 4/5：真实存在训练集外改善的五个场景中，joint oracle 覆盖四个，故场景覆盖率为 4/5。
- 4/9：按状态计，joint oracle 召回 9 个真实改善状态中的 4 个，state-level recall 为 44.44%。

状态级 cost oracle 为 TP/FP/FN/TN=`5/2/4/85`，precision=`71.43%`、recall=`55.56%`；joint oracle 为 `4/0/5/87`，precision=`100.00%`、recall=`44.44%`。差异来自 `case14-g1g5-w0-s1`, index 1：它是 cost TP，但 hard logic 不可行，故 joint-marked=False；这正是 cost TP=5 而 joint TP=4 的唯一原因。

八个 joint-unreachable 场景中，六个没有训练集外真实严格改善；`case14-g0g1-w1-s1` 有 cost 假阳性；`case14-g1g3-w0-s1` 有一个真实改善但被 cost oracle 漏标。

## 资源口径：平均每 run 为主

前期 1080 run 和后期 360 run 均按 run JSON 重聚合；括号中的总数仅用于核验。

| 批次 / 方法 | 严格改善率 | 平均候选 | 平均 oracle calls | 平均 MPS 执行 | 平均新增 ED/LP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 前期 joint-BBHT | 50.00% (90/180) | 14.0667 | 17.5611 | 14.0667 | 0.2111 (38/180) |
| 前期 logic-rejection random | 58.33% (105/180) | 21.6667 | 0 | 0 | 3.6111 (650/180) |
| 后期 joint-BBHT | 33.33% (20/60) | 10.5833 | 14.4833 | 10.5833 | 0.3333 (20/60) |
| 后期 logic-rejection random | 33.33% (20/60) | 21.2000 | 0 | 0 | 3.7500 (225/60) |

两批的初始化与成功定义不同：前期允许训练缓存改善；后期 best-training 初始化排除了训练集内严格改善。因此不应只用全部 run 成功率直接比较，也不应将资源均值脱离成功率解读。

## 查询诊断与 posthoc margin

均匀经典基线从 16 状态均匀抽样后才调用同一 Boolean predicate，不预知目标 index；这只是 oracle-query 模型诊断。fixed-point margin sweep 使用已知训练集外真值，明确为 **posthoc diagnostic**：delta=0 时 joint precision=100.00%、state-level recall=44.44%；正 margin 没有提高召回却降低 precision，不能作为独立泛化结论。

## 局限与后续工作

- Aer MPS 是经典门级模拟器。
- 当前搜索空间只有四比特、16 状态，selected-split 只代表这个子空间。
- VQC 仍存在假阴性；hard logic 也会过滤部分 cost-level 真正改善。
- same-candidate-set 经典抽样的表现可能接近 BBHT。
- 未证明真实量子硬件速度、端到端量子优势或完整 UC 全局最优。
- posthoc threshold sweep 不是独立泛化证据。
