# 硬 UC 逻辑可行性与稀疏 VQC 成本阈值联合 BBHT

## 1. 研究定位

本阶段把项目已有的经典 `is_logic_feasible` 语义编译为可逆量子线路，并与已经验证的稀疏 VQC 成本阈值相位 oracle 联合。

联合标记条件为

\[
F_{\mathrm{logic}}(x)=1
\quad\land\quad
\widehat C_{\theta,\mathrm{int}}(x)<\tau_{\mathrm{enc}}.
\]

对固定输入状态 \(x\)，逻辑可行性是确定的 0/1 判断，不是概率型软判断。稀疏 VQC 成本比较也是确定的整数比较，但被比较的代理成本可能存在模型误差。

本阶段没有声称量子化完整 SCUC 连续可行域。连续经济调度、连续爬坡和网络安全等仍由测量后的真实 ED/LP 最终验证。

## 2. 已编码的硬逻辑约束

量子逻辑可行性 spec 与 `qubit_value_function.commitment.is_logic_feasible` 对齐，编码：

- must-run；
- 初始停机状态遗留的最小停机时间；
- 初始开机状态遗留的最小开机时间；
- 启动后的最小开机时间；
- 停机后的最小停机时间。

对于选定机组子空间，未选机组由 `base_commitment` 固定替换。若某条违反模式与固定值矛盾，该模式不可能发生并被删除；若某条违反模式完全由固定值满足，则整个选定子空间被标记为 `always_infeasible`。

## 3. 局部 forbidden-pattern 编译

逻辑不可行性表示为若干局部禁止模式的 OR：

\[
\neg F_{\mathrm{logic}}(x)
=
V_1(x)\lor V_2(x)\lor\cdots\lor V_K(x).
\]

每个 \(V_k\) 是少量 Boolean literals 的合取，例如：

- must-run 违反：\(x_{g,t}=0\)；
- 启动后提前停机：\(x_{g,t-1}=0, x_{g,t}=1, x_{g,s}=0\)；
- 停机后提前启动：\(x_{g,t-1}=1, x_{g,t}=0, x_{g,s}=1\)。

每个禁止模式使用一个 violation ancilla：

```text
search literals
→ X-normalize zero controls
→ CX / MCX into violation[k]
→ undo X-normalization
```

该结构随 UC 逻辑 clause 数增长，不建立 `2**num_x_qubits` 状态查找表。

## 4. 硬逻辑可行性相位 oracle

先计算全部 violation ancillas：

\[
|x\rangle|0\cdots0\rangle_v
\mapsto
|x\rangle|V_1(x)\cdots V_K(x)\rangle_v.
\]

随后将 violation 位反相，并对“全部为 1”施加多控相位，因此只有全部 violation 为 0 的状态获得负相位。最后反计算全部 violation ancillas：

\[
|x\rangle|0\rangle_v
\mapsto
(-1)^{F_{\mathrm{logic}}(x)}|x\rangle|0\rangle_v.
\]

## 5. 联合 feasible-and-better 相位 oracle

完整联合 oracle 为：

```text
phase-to-value compute
→ IntegerComparator 得到 better flag
→ 计算全部 logic violation flags
→ 对 better=1 且全部 violation=0 施加相位
→ 反计算 logic violation flags
→ comparator inverse
→ phase-to-value inverse
```

最终作用为

\[
|x\rangle|0\rangle_{aux}
\mapsto
(-1)^{F_{\mathrm{logic}}(x)\land[\widehat C_{\theta,\mathrm{int}}(x)<\tau]}
|x\rangle|0\rangle_{aux}.
\]

该 oracle 位于每次 Grover iteration 内部，后接只作用于搜索寄存器的 diffuser。

## 6. BBHT 集成

BBHT 主循环仍然不知道 marked count：

1. 设 \(m=1\)，随机选 \(j\in\{0,\ldots,m-1\}\)；
2. 执行 \(j\) 次“联合 oracle + diffuser”；
3. Aer MPS 单 shot 测量；
4. 辅助位 syndrome 不为零时拒绝该测量，不验证候选；
5. 测得硬不可行状态时不调用 ED/LP；
6. 只有硬可行且代理成本低于阈值的状态进入 exact cache / 新 ED/LP；
7. 只有真实 ED/LP 成本严格改善时更新 incumbent 和 threshold，并把 \(m\) 重置为 1。

辅助 syndrome 的默认接受阈值为 `1 - 1e-12`。在理想单-shot MPS 中，辅助位归零概率应为 1。

## 7. 计数语义加固

结果现在区分：

- `new_exact_evaluation_attempts`：首次遇到未缓存候选后进行的 exact 验证尝试；
- `actual_ed_lp_solves`：真正调用连续 ED/LP 求解器的次数；
- `logic_precheck_rejections`：在 LP 前被经典防御性逻辑检查拒绝的次数；
- `cached_exact_lookups`：命中 exact cache 的次数；
- `marked_candidate_hits`：每个联合 marked 状态的总命中次数；
- `repeated_candidate_hits`：扣除第一次后的真实重复命中次数；
- `auxiliary_syndrome_rejections`：因辅助位未归零而拒绝的 trial 数。

为兼容旧结果，`new_ed_lp_calls` 暂时保留为 `new_exact_evaluation_attempts` 的别名。报告真实 LP 工作量时应使用 `actual_ed_lp_solves`。

## 8. 端到端实验

```powershell
D:\pytorch\anaconda_develop\envs\QubitValueFunction\python.exe `
  experiments\stage1_case14_2x2_joint_feasible_sparse_vqc_bbht.py `
  --selected-generators 0,5 `
  --window-starts 0,1,2 `
  --train-sample-count 8 `
  --fractional-bits 2 `
  --cost-unit 1000 `
  --initialization-policy first `
  --seed 0 `
  --results "$env:TEMP\stage1_case14_2x2_joint_feasible_sparse_vqc_bbht.json"
```

完整 16 状态逻辑可行集合、联合 marked 集和真实全局最优仅在 BBHT 返回后计算，不能控制搜索轨迹。

## 9. 专项测试

```powershell
D:\pytorch\anaconda_develop\envs\QubitValueFunction\python.exe -m pytest `
  tests\test_logic_feasibility_oracle.py `
  tests\test_logic_feasibility_initial_conditions.py `
  tests\test_joint_feasible_sparse_vqc_grover.py `
  tests\test_joint_feasible_sparse_vqc_bbht.py -vv
```

测试验证：

- 编译后的 spec 与经典 `is_logic_feasible` 对所有小实例状态逐一一致；
- 初始状态、must-run、最小开停机条件均被覆盖；
- 硬可行性相位符号正确且辅助位反计算为零；
- 联合 oracle 只标记 feasible AND better；
- 普通 Grover 只放大联合 marked 状态；
- BBHT 不对硬不可行或 auxiliary-syndrome 异常状态调用 exact evaluator；
- exact 尝试、LP solve、逻辑预检和重复命中计数相互独立。

## 10. 尚未量子化的约束

当前硬逻辑 oracle 不是完整物理可行性证明。后续阶段仍需单独研究：

- 在线最大容量与负荷/备用的整数预检查；
- 在线最小出力与负荷关系；
- 更精确的连续爬坡存在性；
- 网络潮流与安全约束；
- 噪声下的辅助 syndrome、读出误差和重试策略。

这些扩展不得通过完整状态表或逐状态 phase mask 实现，应继续采用局部 Boolean clause、可逆整数算术和比较器。