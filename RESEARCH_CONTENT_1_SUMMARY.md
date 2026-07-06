# 研究内容1阶段性总结

## 目标

研究内容1目前聚焦于固定负荷曲线下的机组组合值函数 oracle。给定机组承诺变量 `x`，固定负荷曲线 `d` 作为条件量，不作为 Grover 搜索变量，定义

<div align="center">
  <b>V<sub>d</sub>(x) = startup(x) + min<sub>y</sub> C(y; x, d)</b>
</div>

其中 `y` 是给定承诺 `x` 后的经济调度变量。量子搜索空间只包含承诺比特 `x`，目标是构造可逆的值函数子电路，并嵌入 Grover oracle：

<div align="center">
  <b>O<sub>τ</sub>|x⟩ = (-1)<sup>[V̂<sub>θ</sub>(x) ≤ τ]</sup>|x⟩</b>
</div>

## 技术路线

最初的测量读出型 VQC 不适合直接嵌入 Grover，因为测量会破坏相干性。当前路线改为值寄存器型 oracle：

<div align="center">
  <b>|x⟩|0<sub>f</sub>⟩|0<sub>L</sub>⟩|0<sub>m</sub>⟩|0<sub>c</sub>⟩</b><br>
  <b>→ |x⟩|f(x)⟩|L<sub>r</sub>(x)⟩|max<sub>r</sub> L<sub>r</sub>(x)⟩|[V̂<sub>θ</sub>(x) ≤ τ]⟩</b>
</div>

随后对比较辅助比特 `c` 做相位标记，再按相反顺序 uncompute：

<div align="center">
  <b>compute value → compare → phase mark → uncompute</b>
</div>

这样最终所有辅助寄存器恢复，只在承诺态 `|x⟩` 上留下相位，因此是 Grover 可嵌入的可逆 oracle。

## 值函数表示

固定承诺后的 ED 子问题具有分段线性凸值函数结构。为贴近这一理论性质，当前采用 max-affine surrogate：

<div align="center">
  <b>V̂<sub>θ</sub>(x) = max<sub>r=1,...,R</sub> (b<sub>r</sub> + θ<sub>r</sub><sup>T</sup> f(x))</b>
</div>

其中 `f(x)` 是结构化特征，包括承诺比特、容量和备用聚合、启停/转移项、同一时段机组交互、相邻时段交互、merit-order 调度代理等。该表示不是查表；特征只依赖承诺变量和机组物理参数。

每个仿射片段 `L_r(x)=b_r+theta_r^T f(x)` 可由定点可逆加法器计算；`max_r L_r(x)` 可由可逆比较器树实现；最后与阈值寄存器比较，得到标记辅助比特。

## 阈值比较与辅助比特

阈值比较采用定点寄存器：

<div align="center">
  <b>|V̂<sub>θ</sub>(x)⟩|τ̂⟩|0<sub>c</sub>⟩</b><br>
  <b>→ |V̂<sub>θ</sub>(x)⟩|τ̂⟩|[V̂<sub>θ</sub>(x) ≤ τ̂]⟩</b>
</div>

如果 `c=1`，说明该承诺被判定为目标解，随后对 `c` 做相位翻转。比较完成后必须反算比较器、max 寄存器、仿射片段寄存器和特征寄存器，保证 oracle 结束时辅助比特不残留信息。

实验中使用 calibrated threshold，即比较器中加载预测值空间内的 `tau_hat`，而不是直接加载原始成本 `tau`。这不改变 oracle 结构，只改变阈值寄存器的数值。

## 实验结果

数据来自 UnitCommitment.jl 的 `case14` 实例，使用其中 6 台机组。已验证 `T=2` 和 `T=3` 两个时间步规模。

| 模型 | 时间步 | 承诺比特 | 状态数 | 有限可行状态 | 特征数 | 片段数 | 20-bit 精确标记目标集 | 失败目标集 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| structured linear | 2 | 12 | 4096 | 768 | 207 | 1 | top-1/4/8/16/32/48/64 | 无 |
| max-affine | 2 | 12 | 4096 | 768 | 207 | 32 | top-1/4/8/16/32/48/64 | 无 |
| structured linear | 3 | 18 | 262144 | 16384 | 761 | 1 | top-1/4/8/16/32/64 | top-128 |
| max-affine | 3 | 18 | 262144 | 16384 | 380 | 32 | top-1/4/8/16/32/64 | top-128 |
| boundary-aware max-affine | 3 | 18 | 262144 | 16384 | 380 | 32 | top-1/4/8/16/32/64/128 | 无 |

关键数值：

- `T=2 max-affine`: MAE `12.96`，max error `137.96`，20-bit 下全部测试目标集精确标记。
- `T=3 max-affine`: MAE `21.70`，max error `139.38`，20-bit 下 top-128 失败，边界 margin 为 `-31.84`。
- `T=3 boundary-aware max-affine`: MAE `21.90`，max error `147.24`，20-bit 下 top-1/4/8/16/32/64/128 全部精确标记，top-128 margin 提升到 `+1.85`。

boundary-aware 训练只改变训练权重，不改变 oracle 结构。成功运行中 top-128 边界的 margin 变化为：

<div align="center">
  <b>-26.28 → -18.01 → -2.93 → +1.85</b>
</div>

错分数量从 `6` 个 false positive 和 `1` 个 false negative 收敛到 `0/0`。

## 资源量级

以当前最好的 `case14 T=3 boundary-aware max-affine` 为例：

- 结构化特征数：380；
- max-affine 片段数：32；
- 仿射累加器数量：32；
- 加权加法项数量约 `380 x 32 = 12160`；
- max 比较器数量约 `32 - 1 = 31`；
- 阈值比较器数量：1；
- 由于需要 uncompute，主要计算模块需要正向计算一次、反向计算一次。

这还不是门级综合结果，但已经给出可逆 oracle 的资源主导项。

## 当前结论

目前已经得到一条可支撑的研究内容1主线：用结构化 max-affine VQC 隐式表示 UC 值函数，将其写入定点值寄存器，通过可逆比较器与阈值寄存器比较，再做相位标记和反算，从而构造 Grover 可嵌入 oracle。

与查表或全布尔高阶插值相比，该路线保留了值函数的分段线性凸结构，并且在 `case14 T=3` 上已经实现 18 个承诺比特、262144 个状态空间中的目标集精确标记验证。

## 局限与下一步

当前 `T=3 boundary-aware max-affine` 在 20-bit 值寄存器下成功，但 16-bit 在 top-128 仍失败，说明边界分离裕量仍偏小，量化精度是资源优化瓶颈。

下一步可选方向：

- 资源优化：继续增大 top-128 边界 margin，争取 16-bit 寄存器也精确标记；
- 规模扩展：在保持 6 台机组的前提下，将时间步从 `T=3` 扩展到 `T=4`；
- 文本收束：把本路线写成申报书中“研究内容1”的正式技术方案和阶段性可行性论证。
