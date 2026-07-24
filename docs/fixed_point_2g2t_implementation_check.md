# 固定点 2机组×2时间步原型实施检查

## 已完成检查

1. 对以下新增 Python 文件运行 `python -m py_compile`，均通过：
   - `qubit_value_function/fixed_point_oracle.py`
   - `experiments/stage1_case14_t2_fixed_point_affine_gas.py`
   - `tests/test_fixed_point_oracle.py`
2. 使用 Python AST 再次解析上述文件，均通过。
3. 检查新增 Python 文件不存在超过 119 字符的代码行。
4. 在不执行 Qiskit 电路的条件下，使用 stub 模块加载固定点模块并验证：
   - 成本缩放和解码；
   - 负系数到 offset、非负 weights 和反相输入的转换；
   - 单一 ridge 仿射模型可恢复合成数据系数；
   - 严格阈值比较的代数条件与 `encoded_cost < encoded_threshold` 对所有合成状态一致。
5. 重新从 GitHub 分支读取新增文件，确认远端内容与本地静态检查版本一致。

## 尚未完成的动态检查

当前可执行容器没有安装 Qiskit，且仓库没有返回自动 CI status，因此本轮无法在该容器中真实执行：

- `Statevector.from_instruction`；
- `WeightedAdder` 和 `IntegerComparator` 的门级模拟；
- `pytest -q tests/test_fixed_point_oracle.py`；
- 完整 `pytest -q`；
- case14 典型实验和结果 JSON。

这部分不能标记为通过。应在项目原有 Python/Qiskit 环境中运行：

```powershell
pytest -q tests/test_fixed_point_oracle.py
pytest -q
python experiments/stage1_case14_t2_fixed_point_affine_gas.py `
  --selected-generators 0,5 `
  --train-sample-count 6 `
  --max-rounds 3 `
  --fractional-bits 2 `
  --cost-unit 1000
```

## 动态验收标准

- 所有专项测试通过；
- 原有测试无回归；
- phase oracle 的 `max_phase_error` 接近 0；
- `auxiliary_zero_probability` 接近 1；
- 唯一 marked state 的 Grover probability 明显放大；
- 实验只逐个计算指定数量的代表性 ED/LP 样本；
- 输出 threshold 来自真实 incumbent 成本并使用统一固定点尺度；
- 不生成或使用隐藏全枚举训练数据。
