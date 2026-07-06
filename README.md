# 量子值函数 Oracle：阶段 1

本项目是面向 SCUC/UC 问题的第一阶段固定负荷值函数 oracle 原型。实验使用真实 UnitCommitment.jl JSON 实例，不人工编造机组数据或负荷数据。

研究内容1当前阶段性总结见：

```text
RESEARCH_CONTENT_1_SUMMARY.md
```

阶段 1 固定负荷曲线 `d`，研究如下值函数：

<div align="center">
  <b>V<sub>d</sub>(x) = startup_cost(x) + min<sub>y</sub> ED_cost(y; x, d)</b>
</div>

其中量子搜索空间是机组承诺向量 `x`。负荷曲线当前作为固定条件量，而不是 Grover 搜索变量。

早期阶段 1 实验包括：

- `stage1_fixed_load.py`：测量读出型 VQC 基线。它可以说明值函数有一定可学习性，但不是 Grover 可嵌入的可逆 oracle。
- `stage1_phase_vqc_oracle.py`：阈值条件对角相位 VQC。它学习值函数子水平集，并直接实现不依赖测量读出的相位 oracle：

<div align="center">
  <b>O<sub>θ</sub>(τ)|x⟩ ≈ (-1)<sup>[V<sub>d</sub>(x) ≤ τ]</sup>|x⟩</b>
</div>

## 数据

默认小规模实验使用：

```text
https://axavier.org/UnitCommitment.jl/0.3/instances/test/aelmp_simple.json.gz
```

这是一个单时段、三机组的小型基准算例，便于穷举验证值函数和阈值 oracle。

后续主线实验使用 `data/case14.json.gz` 中的 `case14` 算例，并取其中 6 台机组进行 T=2、T=3 时间步实验。

## 运行

常用实验命令如下：

```powershell
python experiments/stage1_fixed_load.py
python experiments/stage1_phase_vqc_oracle.py
python experiments/stage1_phase_vqc_generalization.py
python experiments/stage1_case14_single_period.py
python experiments/stage1_case14_sparse_phase.py
python experiments/stage1_case14_physics_features_phase.py
python experiments/stage1_case14_hamming_phase.py
python experiments/stage1_case14_threshold_oracle_library.py
python experiments/stage1_case14_sparse_term_analysis.py
python experiments/stage1_case14_bundle_phase.py
python experiments/stage1_case14_partial_signed_phase.py
python experiments/stage1_case14_hierarchical_oracle.py
python experiments/stage1_ancilla_vqc_oracle.py
python experiments/stage1_case14_t2_ancilla_vqc.py
python experiments/stage1_case14_t2_separated_oracle.py
python experiments/stage1_case14_t2_explicit_two_ancilla_oracle.py
python experiments/stage1_case14_t2_leakage_reweighted_training.py
python experiments/stage1_case14_t2_joint_oracle_training.py
python experiments/stage1_case14_t2_threshold_conditioned_ancilla_vqc.py
python experiments/stage1_case14_t2_threshold_conditioned_ancilla_vqc.py --train-target-counts 1,4,16,32,64 --holdout-target-counts 8,48 --results results/stage1_case14_t2_threshold_conditioned_dense_tau.json
python experiments/stage1_case14_t2_threshold_conditioned_ancilla_vqc.py --boundary-weight 2 --results results/stage1_case14_t2_threshold_conditioned_boundary_w2.json
python experiments/stage1_case14_t2_threshold_conditioned_ancilla_vqc.py --tau-basis piecewise_linear --results results/stage1_case14_t2_threshold_conditioned_piecewise_tau.json
python experiments/stage1_case14_t2_threshold_conditioned_ancilla_vqc.py --tau-basis piecewise_linear --train-target-counts 1,4,16,32,64 --holdout-target-counts 8,48 --results results/stage1_case14_t2_threshold_conditioned_piecewise_dense_tau.json
python experiments/stage1_case14_t2_value_register_comparator.py
python experiments/stage1_case14_t2_value_register_comparator.py --register-bits 8,10,12,14,16 --results results/stage1_case14_t2_value_register_comparator_bits.json
python experiments/stage1_case14_t2_value_register_comparator.py --max-order 12 --register-bits 12,16,20 --tie-tolerance 1e-6 --results results/stage1_case14_t2_value_register_comparator_tie_tolerant.json
python experiments/stage1_case14_t2_structured_value_surrogate.py --max-same-time-order 6 --max-adjacent-time-order 3 --register-bits 12,16,20 --results results/stage1_case14_t2_structured_value_surrogate.json
python experiments/stage1_case14_t2_structured_value_surrogate.py --horizon 3 --target-counts 1,4,8,16,32,64,128 --max-same-time-order 6 --max-adjacent-time-order 3 --register-bits 16,20 --results results/stage1_case14_t3_structured_value_surrogate.json
python experiments/stage1_case14_t2_max_affine_value_surrogate.py --piece-counts 2,4,8,16,32,64 --candidate-count 128 --register-bits 16,20 --results results/stage1_case14_t2_max_affine_value_surrogate.json
python experiments/stage1_case14_t2_max_affine_value_surrogate.py --horizon 3 --target-counts 1,4,8,16,32,64,128 --same-time-order 4 --adjacent-time-order 2 --piece-counts 2,4,8,16,32 --candidate-count 256 --initializations least_squares --register-bits 16,20 --results results/stage1_case14_t3_max_affine_value_surrogate.json
python experiments/stage1_case14_t2_max_affine_value_surrogate.py --horizon 3 --target-counts 1,4,8,16,32,64,128 --same-time-order 4 --adjacent-time-order 2 --piece-counts 32 --candidate-count 256 --initializations least_squares --register-bits 16,20 --boundary-target-counts 128 --boundary-rank-window 32 --boundary-weight 16 --boundary-target-side-weight 1 --boundary-nontarget-side-weight 3 --boundary-rounds 4 --boundary-misorder-boost 4 --results results/stage1_case14_t3_boundary_aware_max_affine_value_surrogate.json
pytest -q
```

## 结果文件

实验结果主要写入：

```text
results/stage1_aelmp_simple.json
results/stage1_phase_vqc_oracle.json
results/stage1_phase_vqc_generalization.json
results/stage1_case14_single_period.json
results/stage1_case14_sparse_phase.json
results/stage1_case14_physics_features_phase.json
results/stage1_case14_hamming_phase.json
results/stage1_case14_threshold_oracle_library.json
results/stage1_case14_sparse_term_analysis.json
results/stage1_case14_bundle_phase.json
results/stage1_case14_partial_signed_phase.json
results/stage1_case14_hierarchical_oracle.json
results/stage1_ancilla_vqc_oracle.json
results/stage1_case14_t2_ancilla_vqc.json
results/stage1_case14_t2_separated_oracle.json
results/stage1_case14_t2_explicit_two_ancilla_oracle.json
results/stage1_case14_t2_leakage_reweighted_training.json
results/stage1_case14_t2_joint_oracle_training.json
results/stage1_case14_t2_threshold_conditioned_ancilla_vqc.json
results/stage1_case14_t2_threshold_conditioned_dense_tau.json
results/stage1_case14_t2_threshold_conditioned_boundary_w2.json
results/stage1_case14_t2_threshold_conditioned_dense_boundary_w2.json
results/stage1_case14_t2_threshold_conditioned_piecewise_tau.json
results/stage1_case14_t2_threshold_conditioned_piecewise_dense_tau.json
results/stage1_case14_t2_value_register_comparator.json
results/stage1_case14_t2_value_register_comparator_bits.json
results/stage1_case14_t2_value_register_comparator_tie_tolerant.json
results/stage1_case14_t2_structured_value_surrogate.json
results/stage1_case14_t3_structured_value_surrogate.json
results/stage1_case14_t2_max_affine_value_surrogate.json
results/stage1_case14_t3_max_affine_value_surrogate.json
results/stage1_case14_t3_boundary_aware_max_affine_value_surrogate.json
```

## 实验说明

`stage1_case14_t2_ancilla_vqc.py` 固定 `case14` 的 6 台机组，并使用前两个负荷时段。搜索变量包含 12 个承诺比特，共 4096 个状态；逻辑不可行承诺赋为无穷大值。该脚本使用状态向量 Grover 仿真，而不是显式构造完整的 `8192 x 8192` oracle 矩阵。

`stage1_case14_t2_separated_oracle.py` 将精确布尔可行性 oracle 与可行域上的 ancilla VQC 值函数 oracle 分离，贴近“先物理约束表示、再值函数嵌入”的研究任务路线。

`stage1_case14_t2_explicit_two_ancilla_oracle.py` 显式仿真寄存器序列：

<div align="center">
  <b>A<sub>f</sub> → U<sub>θ</sub> → CCZ(f,a) → U<sub>θ</sub><sup>†</sup> → A<sub>f</sub><sup>†</sup></b>
</div>

其中包含一个可行性辅助比特和一个值函数辅助比特。

`stage1_case14_t2_leakage_reweighted_training.py` 比较普通角度最小二乘与 oracle 导向重加权拟合，后者强调 value-ancilla 泄漏较大的状态。

`stage1_case14_t2_joint_oracle_training.py` 在普通和重加权 VQC 候选中选择联合评分更好的模型，评分项包括 Grover 目标概率、value-ancilla 泄漏和标记集合错误。

`stage1_case14_t2_threshold_conditioned_ancilla_vqc.py` 拟合一个跨多个阈值的 `U_theta(x,tau)` 模型，使值函数 oracle 被检验为阈值条件子水平集族，而不是多个互相独立的固定阈值分类器。脚本还报告单调性诊断：如果模型确实对应一个相干的隐式值函数，预测标记集应随 `tau` 增大而只扩张不收缩。

该阈值条件脚本还支持两个诊断方向：`--boundary-weight` 增加阈值边界附近样本权重；`--tau-basis piecewise_linear` 用局部分段线性 hat function 替代全局多项式 `tau` 特征。目前在 `case14` T=2 结果中，两者更多是诊断而不是最终修复：它们能保持训练阈值上的正确性，但不能稳定修正 held-out 阈值。

`stage1_case14_t2_value_register_comparator.py` 检验更结构化的路线：先拟合标量值函数 surrogate `V_theta(x,u)`，再量化到定点值寄存器，与阈值寄存器比较，最后反算。该实验仍是状态向量/经典仿真，但 oracle 分解比直接阈值角度分类更接近可逆比较器结构。

高阶 value-register 运行使用 `--tie-tolerance 1e-6`，避免在阈值边界处强行切分数值上几乎相等的 UC 成本。order 12 时，浮点比较器和 16/20-bit 定点比较器能精确标记 `case14` T=2 的全部评估目标集。这个结果是强基准，但 order 12 使用 4096 个 Boolean 特征，因此是指数级上界参考，不是可扩展最终 ansatz。

`stage1_case14_t2_structured_value_surrogate.py` 从全 Boolean 单项式转向结构化 value-register 特征，包括承诺比特、容量和备用聚合、启停/转移项、同一时段机组局部交互、相邻时段交互和确定性 merit-order 调度代理。这些特征只依赖承诺寄存器和实例常数，不使用值函数查表作为输入。

在 `case14` T=2 上，当前 207 特征结构化模型（`same_time_order=4`）经过阈值寄存器校准后，用 20-bit 值寄存器可精确标记所有评估目标集。一个 257 特征的相邻时段 pair 变体在不校准时可标记 6/7 个目标集，校准后 7/7 全部正确。这里的校准只是改变比较器加载的阈值数值，不改变可逆 oracle 分解。

同一结构化脚本支持 `--horizon 3`。在 `case14` T=3 中，搜索空间有 18 个承诺比特、262144 个状态；精确 ED 评估得到 16384 个有限逻辑可行承诺。当前 761 特征局部结构化模型（`same_time_order=6`，`adjacent_time_order=3`）在 20-bit 值寄存器和阈值校准后能精确标记 top-1/top-4/top-8/top-16/top-32/top-64，但 top-128 尚不可 rank-separable。T=3 精确 ED 值缓存于：

```text
results/value_cache_case14.json_h3.npz
```

## Max-Affine 值函数 Oracle

`stage1_case14_t2_max_affine_value_surrogate.py` 将标量 surrogate 从单个仿射函数升级为凸 max-affine 形式：

<div align="center">
  <b>V̂<sub>θ</sub>(x) = max<sub>r</sub> (b<sub>r</sub> + θ<sub>r</sub><sup>T</sup> f(x))</b>
</div>

这更接近 ED/Benders 视角下值函数是多个仿射函数逐点最大值的结构。对应可逆 oracle 分解为：可逆计算结构化特征、可逆计算每个仿射片段寄存器 `L_r(x)`、用可逆比较器树得到 `max_r L_r(x)`、与阈值寄存器比较、相位标记、再反算所有辅助寄存器。

在 `case14` T=2、207 个结构化特征（`same_time_order=4`）下，测试了两种初始化：

- `floor`：严格 supporting-cut 下包络构造，lower-bound violation 为 0；64 个片段时 MAE 约 287.61，校准后 16/20-bit 值寄存器均能精确标记所有测试目标集。
- `least_squares`：更实用的回归优先初始化；32 个片段时 MAE 约 12.97，max error 约 137.96，校准后 16/20-bit 值寄存器同样能精确标记所有测试目标集。

这说明 T=2 已经不需要回退到 4096 项 Boolean 插值上界，也可以用分段仿射 value-register oracle 覆盖整个测试目标集族。

在 `case14` T=3、380 个结构化特征（`same_time_order=4`，`adjacent_time_order=2`）下，32 片段 max-affine 模型 MAE 约 21.70，max error 约 139.38。校准后 20-bit 比较器能精确标记 top-1/top-4/top-8/top-16/top-32/top-64，但 top-128 失败，calibration margin 约为 `-31.84`，出现 6 个 false positive 和 1 个 false negative。

同一脚本现在支持 boundary-aware 训练模式。它不改变可逆 oracle 分解，而是在训练阶段做两件事：对指定阈值边界附近样本进行 weighted least squares；对校准比较后仍错序的状态进行迭代重加权。

在 `case14` T=3 上，使用：

```text
--boundary-target-counts 128 --boundary-rank-window 32 --boundary-weight 16
```

并进行 4 轮重加权后，top-128 的 calibrated margin 从约 `-31.84` 提升到约 `+1.85`。

这是项目中第一次在 `case14` T=3 上，用同一套 380 个结构化特征和 32 个 max-affine 片段，在 20-bit 值寄存器下精确标记完整目标集族：

```text
top-1/top-4/top-8/top-16/top-32/top-64/top-128
```

当前剩余限制是量化裕量：同一 boundary-aware 模型在 16-bit 值寄存器下仍不能精确标记 top-128，因为最终正 margin 较小，定点舍入会让少量非目标状态重新跨过阈值。
