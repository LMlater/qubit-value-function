# 研究内容 1：阶段 A 证据完成

## 执行摘要

冻结的 selected-split（12 个四比特、16 状态子空间）真值审计补齐了 96 个训练集外 ED/LP 值：9 个为严格改善。初始 joint oracle 的训练集外候选有 4 个，均为该 16 状态子空间内的真实最优状态；这不是完整 UC 的全局最优声明。

## 技术链路

稀疏相位 VQC → 固定点系数/值寄存器 → 严格 `<` 比较器 → hard-logic joint oracle → Grover/BBHT → Aer MPS 门级模拟测量 → ED/LP 真值验证 → 正常阈值更新。所有真值仅在后验审计中使用。

## N=16、M=1 固定 oracle

`case14-g0g1-w2-s1` 的阈值 136 仅标记 index 3。精确概率与五个 4096-shot Aer MPS 均值分别为：k=0：0.062500 / 0.062305；k=1：0.472656 / 0.472656；k=2：0.908447 / 0.909424；k=3：0.961319 / 0.959619。相位真值表只翻转目标状态，辅助寄存器归零概率为 1（数值误差内）。这是门级经典 MPS 模拟的 oracle-query 证据，不是硬件或端到端加速证据。

## 训练集外 oracle 真值

状态级 cost oracle：TP/FP/FN/TN=5/2/4/85，precision=0.7143，recall=0.5556。joint oracle：4/0/5/87，precision=1.0000，recall=0.4444。八个 joint-unreachable 场景中，6 个没有训练集外真实严格改善；`case14-g0g1-w1-s1` 另有 cost 假阳性；`case14-g1g3-w0-s1` 有一个真实改善但被 cost oracle 漏标。没有“cost 标记真实改善、却被 hard logic 排除”的案例。

四个正例 `g0g1-w2`、`g0g5-w0`、`g0g5-w2`、`g1g5-w0` 的发现 index 均为所选 16 状态子空间最优。

## 前后批次资源口径

对不可变 run JSON 重聚合：前期 1080，后期 360，均覆盖六种方法。前期 joint BBHT 的平均候选数/ oracle calls / 新 EDLP 分别为 14.0667 / 17.5611 / 0.2111；logic-rejection random 为 21.6667 / 0 / 3.6111。后期 joint BBHT 为 10.5833 / 14.4833 / 0.3333；logic-rejection random 为 21.2 / 0 / 3.75。资源均与成功率并列解释，不能脱离成功率单独比较。

## 查询诊断与 margin

均匀经典基线每次从 16 状态抽样后调用相同 Boolean predicate，不预知目标 index；这是 oracle-query 模型的诊断。margin sweep 使用已知训练集外真值，明确属于 **posthoc diagnostic**：delta=0 的 joint precision=1、recall=0.4444；正 margin 未提高召回而降低 precision，不能作为独立泛化结论。

## 局限

MPS 仍是经典模拟器；搜索空间仅四比特/16 状态；selected-split 只代表该子空间；VQC 仍有假阴性；same-candidate-set 经典抽样与 BBHT 可能产生相近结果；未证明硬件速度、端到端量子优势或完整 UC 全局最优。
