# 2台机组×2时间步固定点门级 GAS 原型

## 研究定位

本原型属于研究内容1的门级算术基线。它保留“学习启停方案到经济调度最小成本的映射，再由 Grover Adaptive Search 更新真实成本阈值”的方向，但当前只使用一个经典 ridge 仿射模型，不把它表述为 VQC。

## 运行方式

```powershell
python experiments/stage1_case14_t2_fixed_point_affine_gas.py `
  --selected-generators 0,5 `
  --train-sample-count 6 `
  --max-rounds 3 `
  --fractional-bits 2 `
  --cost-unit 1000
```

默认结果写入：

```text
results/stage1_case14_t2_fixed_point_affine_gas.json
```

## 定点数编码

统一使用

```text
encoded(value) = round(value / cost_unit * 2**fractional_bits)
```

模型系数、bias、预测成本和 incumbent threshold 必须使用相同配置。默认 `cost_unit=1000`、`fractional_bits=2`，一个整数码对应 250 美元，最近舍入的单次最大误差为 125 美元。

负系数使用

```text
-a*x = -a + a*(1-x)
```

转换为常数 offset、非负整数权重和反相输入，因此可直接使用 Qiskit `WeightedAdder`。

## 门级 oracle

```text
compute WeightedAdder
→ IntegerComparator
→ phase mark
→ comparator inverse
→ WeightedAdder inverse
```

不使用 QFT。`Statevector.from_instruction` 只负责执行完整门级电路和读取概率，不允许根据经典 `marked_mask` 直接修改振幅。

## 小规模验证

实验固定为 2 台机组、2 个时间步，共 4 个搜索 qubits。训练样本按照代表性索引顺序逐个进行 ED/LP 计算，达到指定数量后停止，不运行多学习器或多 seed sweep。

输出记录：

- 每个代表性样本的逐次 ED/LP 状态；
- 仿射模型的真实成本系数；
- 固定点 offset、weights、反相输入和量化误差；
- 每轮真实 incumbent threshold；
- phase oracle 误差、辅助寄存器回零概率和 Grover marked probability；
- 门级电路 qubit、depth 和操作计数。

## 验证命令

```powershell
pytest -q tests/test_fixed_point_oracle.py
pytest -q
```

## 尚未完成

- VQC 值函数近似；
- VQC 输出写入固定点值寄存器；
- 多实例与大规模验证；
- 旧实验文件清理。
