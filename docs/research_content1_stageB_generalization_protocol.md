# 研究内容 1：阶段 B 泛化基准协议

## 范围

本协议定义阶段 B 的后续泛化基准，目标是比较负荷条件价值模型，而不是主张量子优势或将 QNN 接入 Grover。本轮只实现可复现的划分、归一化、选择、指标、缓存和一项经典基线 smoke；不运行完整 1,008 行 ED/LP 真值表或 QNN 网格。

## 场景和负荷

场景单元为机组对 `(0,1)`、`(0,5)`、`(1,5)` 与窗口 `0`、`1`、`2` 的笛卡尔积。原始状态 split seed 为 `1`、`7`、`19`。每个单元有 16 个承诺状态：8 个 `training_indices` 和互补的 8 个 `unseen_indices`。

负荷倍率固定为 `.80`、`.85`、`.925`、`1.00`、`1.075`、`1.15`、`1.20`：

- 拟合和验证负荷：`.85`、`1.00`、`1.15`；
- 仅插值测试负荷：`.925`、`1.075`；
- 仅外推测试负荷：`.80`、`1.20`。

完整真值缓存因此为 `3 × 3 × 3 × 7 × 16 = 1,008` 条 ED/LP 求值。

## 固定分层 fit/validation

每个场景单元、每个原始 split 的每个训练负荷独立但确定性地将其 8 个训练状态划为 6 个 fit 和 2 个 validation。因此总计 18 个 fit 样本和 6 个 validation 样本。`unseen_indices` 永远不进入 fit 或 validation。

划分算法以排序后的训练状态、split seed、负荷倍率和算法版本构造 canonical JSON，再由 SHA-256 派生无序无关的选择顺序。输出必须保存 split seed、训练/未见状态、每个训练负荷的 fit/validation 状态、算法版本、canonical JSON 与划分 SHA-256。每个负荷上 fit 与 validation 不相交且并集恰为 `training_indices`。

## 泄漏隔离与选择

输入归一化、目标归一化、模型拟合、正则化、early stopping、cutoff 与随机种子选择只读取 fit/validation 数据。`.925`、`1.075`、`.80`、`1.20` 测试负荷的标签或真实成本绝不进入这些步骤。测试集仅在配置选定后评估一次。

候选配置按以下固定字典序选择：validation MAE、validation regret、参数量、配置列表顺序。regret 定义为每个评估组中“预测最低成本状态”的真实成本减去该组真实最小成本；报告其跨组均值和最大值。任何 NaN 或 Infinity 都使计算失败。

threshold-conditioned QNN 的 quantile、目标归一化与概率 cutoff 同样只能由 fit/validation 数据构建；本轮不运行其网格。

## 模型与本轮 smoke

完整后续比较包括常数、线性 Ridge、二次 Ridge、小型 MLP、简化 QNN、当前 QNN，以及随机量子特征加 Ridge。本轮 smoke 只使用固定的经典候选列表，限一个机组对、一个窗口、一个 split，验证 ED/LP 缓存与验证集选择链路。它不覆盖或替代第一轮 Stage B pilot，也不写入其目录。
