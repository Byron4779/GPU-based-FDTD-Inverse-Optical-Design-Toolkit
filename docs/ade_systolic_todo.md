# 色散介质（ADE）接入 systolic 内核 —— 待办清单

本清单只列"要做哪些事"及其顺序，不代表已经开工。目标是让 `fdtdz_jax` 的
color dispersion（ADE / Drude / Lorentz / Multipole）从当前的 `lax.scan` 路径
迁移到 CUDA systolic 内核上。

## 阅读约定

- 每一项都带 `[依赖]` / `[文件]` / `[验收]` / `[理由]`，缺省表示依赖前一项。
- `【B→A】` 标记表示"方案 B（状态放 shared）中能够反哺方案 A 的部分"，
  已按其真正需要发生的时刻插进序列，而不是堆在末尾。
- 阶段 0/1 是**决策与前置优化**；阶段 2 之后才是方案 A 正式开工。
- **阶段 1.6 是关卡**：若 JAX 侧优化后已达目标，阶段 2 之后可整体搁置。

## 结论先行

| 场景 | 建议 |
| --- | --- |
| 极点 ≤ 2（NIR 金可只用 1 个 Drude 极点） | **方案 A**：ADE 状态做成 `diamond::Cell` 里"额外的 E-like 平面"，由 systolic 带进去 |
| 极点 ≥ 3（Rakić 6 极点全覆盖） | 寄存器放不下，**方案 B**：状态放 shared，只保留"驻留摊薄" |
| 不想动 CUDA | 方案 C 不可取（见"已否决的方案"），改为优化 JAX 侧 |

核心判断：**systolic 化 ADE 的唯一真实收益来自"节点在钻石内驻留 N=4 步"**。
这 4 步里 ADE 的递推是纯点态（pointwise）的，状态不必进出内存；如果放弃这一点、
把状态放到全域数组里逐步读写，那 systolic 就退化成带宽受限的点态循环，
还不如直接写单独的 elementwise kernel。

## 三个可以利用的既有事实

### 1. 时间-空间 skew 天然提供了 4 步的"免费寄存器缓存"

```cpp cuda/scanner.h
return WrapDomainPos(StartDomainPos(warppos, blockpos, rs) + XYT(step, 0, step), rs);
//                                                         ^x 与 ^t 同步推进
```

`DomainPos` 让 x 与 t 同步前进，加上 `diamond::Shift` 把槽位沿 -x 搬运，因此
同一个物理节点在同一线程的寄存器里连续驻留 `N = 4` 步，且每步 `t` 递增 1。
这意味着 `E^{n-1}`、`P^{n-1}` 在更新 `E^n` 的那一刻已经在同一线程的相邻槽位里，
不需要任何通信；每次和内存交互只需 `1/4` 的频次（与 E/H 完全同构）。

### 2. ADE 不需要 halo，但需要跟 E 走同一套 ring

差分需要邻居 E/H（要 cross-thread 通信）；ADE 递推只读本节点的
`E^{n-1}`、`P^n`、`P^{n-1}`。所以：

- **静态系数**：无时间语义，可以像 `C` 那样做成全域数组 +
  `cbuf::NearestNode` 边界推断（`cbuf.h: GlobalValue`）。
- **动态状态**：不能用 `NearestNode` 推断（会把别人的历史当自己的历史），
  必须跟 E 一起走 `buffer::LoadGlobal` / `StoreGlobal` 的 ring，
  否则 subvolume / 多卡边界会读到未初始化的状态。

### 3. `Cell` 已经是按 `Ehc` 参数化的

```cpp cuda/diamond.h
T values_[N + 2][N + 1][Nz][kNumEhc][kNumXyz];
```

`kNumEhc` 从 3 放大到 `3 + 2 + 2 * N_poles` 之后，`ExternalElems`、
`field::ExternalElems`、`Node` 索引等都会自动跟上。真正要手写的是
`buffer_ops.h` 里按平面手写的 `UE/UH/VE/VH` 存取 pattern（`C2`/`C4`）。

## 方案 A 的状态最小化：只实现 Multipole 分支

`yee_ade_3d.py` 里 Multipole 分支完全不用 `a0..b2`：

```python
known = (q_current * electric + q_previous * electric_previous +
         r_current * polarization_current + r_previous * polarization_previous)
electric_next = (displacement_next - known.sum(-1)) / (eps_inf + q_next.sum(-1))
polarization_next = known + q_next * electric_next[..., None]
```

所以金只需要：`D`（当前）、`E^{n-1}`、每个极点 `P^n / P^{n-1}`。
不需要 `D^{n-1}`（那是 Lossless / Debye / Drude / Lorentz 单极点分支才要的）。
新增平面数 `ΔkNumEhc = 2 + 2 * N_poles`：

| 极点 | 新增平面 | Cell 体积相对今天 | 判断 |
| --- | --- | --- | --- |
| 1（Drude-only 金） | 4 | ×2.3 | 可试 |
| 2 | 6 | ×3.0 | 很可能 spill |
| 6（Rakić） | 14 | ×5.7 | 不可行 → 方案 B |

---

## 阶段 0 · 决策与基线（不写实现代码）

- [ ] **0.0 抓取 golden 参考数据**
  - 用现有 `yee_ade_3d` 路径跑 3 个小算例并存档：1D 单 Drude、1D 单 Lorentz、
    3D 均匀金膜（单极点拟合）。
  - 存档内容：E 场时间序列、`displacement`/`polarization` 末态、
    以及 `ADEState` 递推的逐步中间量。
  - `[文件]` 在 `tests/` 旁新增落盘脚本（不进入 CI，只做本地对照）。
  - `[理由]` 阶段 4 的所有验收都靠它；现在不存，后面只能靠"看起来收敛了"判断。

- [ ] **0.1 基线性能与资源测量**
  - 测三组：(a) 纯 ε、`n_poles=0`；(b) 1 极点；(c) 6 极点。
    记录 wall time / 每步耗时的**斜率**，确认 JAX 路径对 `n_poles` 是否严格线性。
  - 用 `nvcc --ptxas-options=-v` 记录当前 kernel 的 registers / spill / occupancy
    （fp32 与 fp16 各一份）。
  - `[理由]` 后续所有"做得成/做不成"的判断都基于这两个数。

- [ ] **0.2 寄存器预算决策（A vs B 的岔路口）**
  - 用 `ΔkNumEhc = 2 + 2N` 叠加到 0.1 的基线，写出判据：
    `spill == 0` 且 `occupancy ≥ 2 blocks/SM` → 走 A；否则走 B。
  - `[产出]` 一页结论：`N_poles ≤ ?` 时 A 可行。这个数字决定阶段 1 里
    极点裁剪的目标值。

## 阶段 1 · JAX 侧优化（方案 A 正式开工之前）

> 这一阶段的目的是**先确认是否真的需要动 CUDA**。如果 1.6 的关卡通过，
> 阶段 2 之后可以整体搁置。

- [ ] **1.1 极点数裁剪**
  - 按波长裁剪极点表：NIR 金降到单 Drude；给出"波长 → 最少极点数"的映射
    函数与误差界。
  - `[文件]` `material_ade_3d.py`（极点拟合/裁剪工具）
  - `[产出]` 目标场景的 `N_poles`，直接回填 0.2 的决策。
  - `[理由]` 这是最便宜的一步，而且同时决定 CUDA 侧的通道数。

- [ ] **1.2 `roll` → 切片拼接**
  - 把 `_forward` / `_backward` 中的 `jnp.roll` 换成 `concatenate` + slice，
    避免整块读+写。
  - `[文件]` `yee_ade_3d.py`、`yee_ade_2d_tez.py`、`yee_ade_2d.py`
  - `[验收]` 改动前后输出在 fp 误差内一致；XLA 生成的 HLO 里
    `slice` / `concatenate` 可被融合。

- [ ] **1.3 状态原地复用**
  - 对 `lax.scan` 的 carry 加 `donate_argnums`，让 `displacement` /
    `polarization` / `electric` 系列不产生每步全量拷贝。
  - `[验收]` HLO 里每步的 `parameter` / `copy` 数量下降；内存峰值下降。

- [ ] **1.4 系数归一化 + 强制 fp32**
  - 极点表归一化，使 `eps_inf + Σ q_next ≡ 1`（分母恒定，消掉共振附近的
    除法放大）。
  - 色散路径强制单精度（禁止 half）。
  - `[理由]` 这两条对方案 A 同样必需（阶段 4.2 会直接复用同一套归一化），
    提前在 Python 侧定死接口可以避免 CUDA 侧返工。

- [ ] **1.5 补 `dt < 2 / ω0_max` 校验**
  - 在 `fdtdz_jax()` 内对 `pole_table` 做前缀校验并抛明确异常；
    `prepare_material_grid_3d` 现有的 `min(eps_inf)` CFL 检查保留，
    但不再作为唯一依据。
  - `[验收]` 构造 `dt → 2/ω0` 的用例，确认是"明确报错"而不是"5000 步后 NaN"。

- [ ] **1.6 关卡（Gate）**
  - 若 1.1 + 1.2 + 1.3 之后，目标场景（≥ 1e7 cell、单极点）已经落到可接受
    耗时，则**暂停阶段 2 之后全部工作**。
  - 否则进入阶段 2，并把 1.1 / 1.4 的产出作为 CUDA 侧的输入契约。

## 阶段 2 · 前置重构：CPML 从 E 挪到 D

> 色散介质必须走 D-CPML（Roden–Gedney）。这一步在常 ε 下物理等价，
> 所以可以独立验收，是后面所有改动的安全网。

- [ ] **2.1 `Ehc` 引入 `D` 平面**
  - `enum Ehc` 加 `D`，`kNumEhc` 相应 +1；`IsActive` 让 `D` 与 `E` 同 active 集。
  - `[文件]` `cuda/diamond.h`

- [ ] **2.2 更新顺序改为 D-驱动**
  - `update::Scale(...)` 作用对象 E → D；新增 `update::UpdateD`
    （`D += (1/eps_inf) * curl(H)`）；保留 E 平面仅供 H 的 curl 使用。
  - `[文件]` `cuda/update.h`、`cuda/kernel.h`、`cuda/zcoeff.h`
  - `[验收]` `tests/test_fdtdz_jax.py::test_point_source` 全组通过，
    误差不劣于基线；fp16 路径同样通过。

- [ ] **2.3 `cbuffer` 语义改为 `1/eps_inf`**
  - `fdtdz_jax()` 中 `cbuffer = 1/epsilon` → `1/eps_inf`；
    非色散时 `eps_inf == epsilon`，行为不变。
  - `[文件]` `src/fdtdz_jax/fdtdz_jax.py`

- [ ] **2.4 输出 / 子体积回归**
  - 跑 `test_large_sim`（subvolume）与 `test_raises_correct_exception` 全组。
  - `[理由]` D 是新增平面，`field::ExternalElems` / `SharedElems` 的大小都变了，
    这里最容易悄悄越界。

## 阶段 3 · 打通状态通道（哑通道，ADE = 恒等）

> 目标是只验证存储拓扑，不验证物理。这是整个方案里最容易出错、也最难 debug
> 的部分，所以必须单独成一个里程碑。

- [ ] **3.1 扩展 `Ehc` 与 `Cell` 布局**
  - 按 `2 + 2N` 扩 `kNumEhc`；先只加 `Eprev` 一个平面，
    ADE 用 `E = D * eps_inf_inv` 的恒等映射。

- [ ] **3.2 `buffer_ops.h` 的存取 pattern**
  - 为 `Eprev` 复制一套 UE-like / VE-like（`C2` / `C4`）pattern；`D` 同理。
  - `[文件]` `cuda/buffer_ops.h`

- [ ] **3.3 索引与容量**
  - `GlobalBlockIndexing::NumElems`、`SharedBlockIndexing::NumElems`、
    `buffer::GlobalElemsUV`、`cbuf::SharedIndexing` 逐个核对平面数。
  - `[文件]` `cuda/buffer_indexing.h`、`cuda/buffer.h`

- [ ] **3.4 Python 侧形状与算子签名**
  - `_internal_shapes["buffer"][0]`（现在是 6，"larger than needed"）
    按 `buffer::GlobalElems` **重新推导上界**，不要凭经验加一；
    `fdtdz()` 新增 `pole_ids` / `pole_table` 占位参数；
    `_fdtdz_lowering` 的 `operands` / `operand_layouts` 同步。
  - `[文件]` `src/fdtdz_jax/fdtdz_jax.py`、`src/fdtdz_jax/gpu_ops.cpp`
    （descriptor 若需扩字段）

- [ ] **3.5 `Shift` 一致性**
  - 确认 `Eprev` 与 `E` 在 `diamond::Shift` 里走完全相同的搬运轴（co-located）。
  - `[验收]` **bit-identical**：与阶段 2 结尾的输出逐位相同。
    只要有一位不同，就是拓扑接错了。

- [ ] **3.6 subvolume 一致性**
  - 在 `test_large_sim` 的小子体积上跑阶段 3 版本，确认跨 subvolume 边界的
    状态被正确交接（而不是被 `NearestNode` 推断成别人的历史）。
  - `[验收]` 子体积结果与全域结果在 fp 误差内一致。
  - `[理由]` 这是"状态必须走 ring、不能走全域名数组"的实际检验点。

## 阶段 4 · 单极点 ADE（Drude）

- [ ] **4.1 `update::ADEStep`**
  - 只实现 Multipole 分支：`E = (D - Σ known) / (eps_inf + Σ q_next)`，
    `P ← known + q_next * E`；`known` 用 `E`、`Eprev`、`P_c`、`P_p`。
  - `[文件]` `cuda/update.h`
  - `[理由]` 不实现 `a0..b2` 单极点分支可以省掉 `D^{n-1}` 平面
    （`yee_ade_3d` 的 multipole 分支就不用它）。

- [ ] **4.2 静态系数通道**
  - `pole_ids`（逐节点材料/极点索引）+ `pole_table`（`n_poles × 5`）
    → 逐节点系数平面；转换逻辑照抄 `cbuf::Convert`（全域名逐节点写一次），
    只 load 不 store。
  - `[文件]` `cuda/cbuf.h`（新增 `ConvertPoles`）、`cuda/kernel.h`
  - `[注意]` 系数沿用 `NearestNode` 边界推断（静态、无时间语义，安全）；
    **状态绝不能用这个推断**。

- [ ] **4.3 验收（三重）**
  - (a) 与 0.0 存档的 `yee_ade_3d` 逐点对比；
  - (b) 均匀金膜反解 ε，对照 `-20.2777 + 2.0706i`（800 nm）；
  - (c) `dt` 边界扫描与 1.5 的 Python 校验行为一致。

- [ ] **4.4 性能对照**
  - 与阶段 1 优化后的 JAX 路径同网格对比；同时给出"每节点每步摊薄后有效流量"
    的理论值 vs 实测值，确认 4 步摊薄真的生效
    （若没生效，说明状态退化成每步读写，方案 A 的价值就没了）。

## 阶段 5 · 多极点 + 【B→A】shared staging 补充

> 从这里开始方案 B 才真正介入，但它有两块内容值得提前借用，见 5.0。

- [ ] **5.0【B→A】提前借用：状态分块生命周期约定**
  - 把"状态只在钻石 leading/trailing 边缘进出全局"的约定写成一张
    **状态生命周期表**（进入/驻留/退出各发生什么、谁负责 store、谁负责 load），
    先在阶段 3 的哑通道上验证。
  - `[理由]` 这张表是 A 和 B 的公共契约。先定下来，5.2 换成 shared 时
    不用重写 `update` 逻辑，只换存储层。

- [ ] **5.1【B→A】寄存器实测与岔路确认**
  - 按 `N = 1, 2, 3` 逐个实测 spill / occupancy，回填 0.2 的判据。
    若 `N ≥ 2` 就 spill，直接进 5.2。

- [ ] **5.2【B→A】shared 版状态存储**
  - 每 warp 一块 `4 × 4 × Nz × 3 × (2 + 2N)` 的 shared 状态；
    只在钻石边缘与 ring 交换。
  - `[验收]` 与 5.1 之前的寄存器版结果一致；shared 用量与估算吻合
    （1 极点 ≈ 1.5 KB/warp，6 极点 ≈ 5.4 KB/warp）。
  - `[注意]` 5.2 不改变存储拓扑（ring 仍是跨块 / 跨 subvolume 的唯一通道），
    所以阶段 3.6 的测试必须继续通过。

- [ ] **5.3 N 泛化**
  - 把 4.1 / 4.2 从单极点模板化到 `N`；极点表的 `(n_poles, 5)` 维度
    贯通 Python 与 kernel descriptor。

- [ ] **5.4 多极点验收**
  - 与 0.0 的 6 极点存档对比；给出"极点数目 vs 耗时"曲线，
    与 0.1 的 JAX 曲线放在同一张图上。

## 阶段 6 · 加固与收尾

- [ ] **6.1 吸收层限制的硬断言**
  - `absorption_mask > 0` 的节点上强制 `eps_inf == 1`（含 subvolume 补齐区域），
    并在 `fdtdz()` docstring 里改写那句 "dispersion are not implemented"。
  - `[理由]` `σ(x, y) ∝ ε` 对负介电常数会翻符号发散，必须在入口拦住。

- [ ] **6.2 half 精度路径**
  - 色散 + `use_reduced_precision=True` 显式拒绝
    （或提供"仅状态通道 fp32"的混合精度方案，作为独立可选项）。
  - `[理由]` `eps_inf + Σ q_next` 在共振附近的相对误差会被放大。

- [ ] **6.3 Bloch / 周期边界**
  - 明确列为"暂不支持"；`WrapX` / `WrapY` 加相位是另一条独立工作线，
    不要和 ADE 混在一个 PR 里。

- [ ] **6.4 文档与回归基线**
  - 更新 `fdtdz()` docstring、`paper/` 里性能一节、CI 里新增色散小算例
    （用 0.0 的存档做 golden）。
  - 记录最终的 `dt` / 极点 / 校验矩阵。

---

## 依赖关系速查

```text
0.0 ─┬─> 4.3(a)
0.1 ─┴─> 0.2 ─> 5.1 ─> 5.2
1.1 ─────────> 0.2
1.4 ─────────> 4.2
1.5 ─────────> 4.3(c)
1.6 ─>[gate]──> 2.x（可整体搁置）
2.x ─> 3.x ─> 4.x ─> 5.x ─> 6.x
5.0 <─ 必须在 3.x 之前写好表，在 5.2 之前不改接口
```

## 风险最高的三件事

排期上建议留出余量：

1. **3.2 的 pattern 复制** —— 错一位就静默错，且没有任何报错。
2. **3.4 的 buffer 上界推导** —— 越界不报错。
3. **2.2 的 D-CPML 改写** —— 唯一会安静地改变物理的一步。

## 已否决的方案

**方案 C：算子分裂（两次 launch）** —— "systolic 更新 D/H（`eps_inf`）→
单独 elementwise kernel 做 `D → E` + `P` 递推"看起来最省事，但 `fdtdz` 的
全部价值在于整个仿真是一次 cooperative launch（`grid.sync()` 在
`step % rs.spacing == 0` 时同步）。拆成两个 kernel 意味着每个 timestep
两次 launch + 一轮 D/E/P 的 round-trip：

- 网格 < 1e7 cell 时 launch 开销直接吃掉收益；
- 大网格下多出来的流量恰好是方案 A 摊薄掉的那 4 倍。

只有当方案 A/B 都失败、且需要 ≥ 6 极点时才值得重新考虑。

## 预期收益（用于验收时的目标值）

作为对照，现有 JAX 路径每步要做约 10–20 次全阵列 `roll`（读 + 写）。
估算下来方案 A 的"每节点每步 ~24 B、摊薄后约 6 floats"大约快 5–10 倍。
这个倍数是阶段 4.4 的验收目标，不是保证值。
