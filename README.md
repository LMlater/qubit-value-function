# Qubit Value Function for Unit Commitment

本仓库当前整理完成的是“研究内容1阶段A：固定负荷、commitment-only 的模拟器原型”，并不表示整个研究内容1已经完成。阶段A以固定负荷窗口下的机组开停状态为输入，构建并验证稀疏 VQC 值函数、相干成本寄存器、严格阈值 oracle 和自适应 Grover 搜索的模拟器闭环。

## 当前正式主线

当前正式路径为 sparse-VQC + phase-to-value + joint BBHT：

- 在固定负荷窗口 u_w 下学习值函数切片 Q_{u_w}(x)；
- 用稀疏 VQC 输出代理成本，再通过相干 phase-to-value 电路写入固定点整数成本寄存器；
- 以真实 incumbent 的 ED/LP 成本编码为阈值，只采用严格比较：

    C_hat_integer(x) < tau

    tau = encode(C_incumbent_true)

- 将成本阈值条件与可选的硬 UC 启停逻辑可行性条件组成联合 oracle；
- 用普通 Grover 或 BBHT 从实际门级测量结果中选择候选，并以真实 ED/LP 复核；只有真实成本改善时才更新 incumbent。

阶段A不包含负荷输入泛化、真实量子硬件结果或对量子加速的结论。

## 阶段A范围与主要模块

- 稀疏 VQC 代理模型：以低阶、稀疏相位项表示固定负荷下的 commitment-only 成本切片。
- 训练标签与真实验证：由 ED/LP 求解得到训练标签和候选的真实成本。
- 相干成本编码：相位编码、phase-to-value、固定点量化、WeightedAdder 和 IntegerComparator。
- Oracle：严格成本 oracle，以及与 UC 启停逻辑共同工作的 joint oracle。
- 搜索：完整门级 Grover/BBHT 电路、Aer MPS 测量、候选来源和真实成本闭环诊断。
- 验证：单元测试、静态电路检查与此前已验收的 case14 模拟器实验。

更完整的研究演进、已验收实验口径、资源代价和当前边界见 [RESEARCH_CONTENT_1_SUMMARY.md](RESEARCH_CONTENT_1_SUMMARY.md)。

## 历史 max-affine 原型

max-affine 路线是早期的门级可逆值寄存器与 Grover/GAS 探索。它为后续固定点寄存器、比较器、候选真实验证等工作提供了原型基础，具有阶段性研究意义。该历史原型采用自身的模型、实验设置和阈值语义；它不是当前 sparse-VQC + phase-to-value + joint BBHT 的正式主线，历史结论亦不因此被删除或否定。
