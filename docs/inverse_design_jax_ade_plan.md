# JAX ADE 色散介质逆向设计接口 — 分步实现计划

状态：M0–M4 已完成；CUDA 黑盒优化与显式伴随 VJP 作为后续里程碑
依据：FDTDZ 现有 JAX ADE 求解器，以及 Meep `adjoint` 的公开接口和已发表公式
验证环境：Python 3.10–3.12、JAX 0.4.32–0.6.x、SciPy、NumPy 与 pytest

## 1. 目标与范围

- 在 `fdtdz/src/fdtdz_jax/` 下新增 `inverse_design/` 模块，提供对齐 Meep `adjoint` 的三层接口：参数化、目标、问题与优化。
- 首个里程碑（M1）：无损 2D TMz 端到端可微——SIMP 密度参数化 + 透射探针能量目标 + `jax.value_and_grad` + `jax.jit` + scipy L-BFGS-B。完全基于现有 JAX ADE 求解器，不改 CUDA 内核。
- 色散预留（M3）：实现 JAX 可微系数构造器 `coefficients_from_design(rho, material_spec)`，无损退化为 `(1/ε, 0, 0, 0, 0)`，Debye/Lorentz/Drude/Multipole 时输出完整 ADE 系数与极点系数。设计变量始终是几何/密度 `rho`，材料模型固定，梯度只流经 `rho`。
- 不在本计划内：CUDA 内核色散（内核级障碍，只能接口层预留）、CUDA 伴随 VJP（等黑盒线验证价值后另立里程碑）。

## Meep 参照映射

原则：接口与数学公式照搬 Meep，fdtdz 里只写薄适配层；梯度机制用 JAX autodiff 替代 Meep 的显式伴随（ceviche 式）。因 fdtdz 是 MIT 许可，不复刻 Meep 的 GPL 代码，只对齐接口与已发表公式。

| fdtdz 模块 | Meep 参照 | 状态 |
| --- | --- | --- |
| `inverse_design/parameterization.py` | `python/adjoint/filters.py`（`conic_filter`/`gaussian_filter`/`cylindrical_filter`/`tanh_projection`，含周期轴）、`python/geom.py` `MaterialGrid`（weights∈[0,1]、tanh 投影、SIMP 插值、damping） | 已落地并逐点对拍（误差 ~2e-7；周期卷积误差 ~1e-16） |
| `inverse_design/problem.py` | `python/adjoint/optimization_problem.py` `OptimizationProblem`（`__call__`/`get_fdf_funcs`/`calculate_fd_gradient`/`update_design`）、`utils.py` `DesignRegion` | 已按接口对齐 |
| `inverse_design/objectives.py` | `python/adjoint/objective.py` `ObjectiveQuantity` 基类 + 具体量 | 骨架已落地（`ProbeTrace`/`ProbeEnergy`） |
| `inverse_design/optimize.py` | Meep 示例中的 scipy L-BFGS-B 驱动 | 已落地 |
| 色散可微系数构造器（M3） | `geom.py` 对 `LorentzianSusceptibility.sigma` 的插值 + `materials.ade_coefficients`/`multipole_reference.pole_ade_coefficients` | 已完成（Lossless/Debye/Lorentz/Drude + Multipole 极点） |
| CUDA 黑盒线（可选） | Meep 的 nevergrad/Optuna/BoTorch 用法 | 待做 |
| 显式伴随 VJP（远期） | `adjoint/wrapper.py` `MeepJaxWrapper`（`jax.custom_vjp`） | 待做 |

照搬的关键公式：

- tanh 投影：`(tanh(βη) + tanh(β(u−η))) / (tanh(βη) + tanh(β(1−η)))`
- SIMP 插值：`ε = medium1 + u**p * (medium2 − medium1)`（Meep 为 p=1 线性，本仓库默认 p=3）
- 灰度惩罚：`σ = damping · u · (1 − u)`

## 2. 现状核对（已确认，作为实现前提）

- 求解器 `jax_yee_ade_2d.py` 是纯 `jnp + lax.scan`，端到端可微；`simulate_jax_yee_ade_2d_probes` 只返回两个 x 平均探针 `(time,)`（内存友好）；`simulate_jax_yee_ade_2d_line_probes` 返回完整 x 线 `(time, nx)`。
- NumPy 侧 `prepare_material_grid_2d` 产出静态 `MaterialGrid2D`，`coefficients` 形状 `(nx, ny, 5)`，无损材料 `a0 = 1/eps_r`，其余为 0；CFL 用 `min(eps_inf)` 校验；CPML/导电海绵仅允许 Lossless 边界节点。
- `ade_coefficients(material, dt)`（`materials.py`）与 `pole_ade_coefficients(pole, dt)`（`multipole_reference.py`）是纯 NumPy。
- 已验证的关键 JAX 模式：
  - `jax.scipy.signal.convolve2d` 存在，可用于高斯模糊；
  - `grid.coefficients.at[design_mask, 0].set(a0_design)` 能按设计区逐点写入 a0；
  - `grid._replace(coefficients=...)` 在 `jax.jit` 内可用（`JAXMaterialGrid2D` 是 `typing.NamedTuple`）。
- 仓库目前没有任何逆向设计 / 伴随 / 优化代码。

## 3. 模块布局

```text
fdtdz/src/fdtdz_jax/inverse_design/
  __init__.py          # 公开导出
  parameterization.py  # blur + projection + SIMP（纯 jnp）
  materials_jax.py     # jax_ade_coefficients / jax_pole_ade_coefficients / coefficients_from_design
  objectives.py        # ObjectiveQuantity 风格的标量目标
  problem.py           # InverseDesignProblem（Meep 式抽象）
  optimize.py          # value_and_grad + jit + L-BFGS-B / Adam
```

## 4. 分步实施

### 步骤 0 — 基线与环境（不新增功能）✅ 已完成

目标：确认 CPU JAX 能跑现有 2D 求解器，留一条可复现基线。

动作：

1. `python -c "import jax; print(jax.devices())"` 确认 CPU 后端。
2. 用 `examples/metasurface_2d_tmz.py` 或 `run_tmz_scattering_2d` 跑一个最小 TMz 散射，确认导入与数值正常。
3. 新增 `tests/test_inverse_design_0_baseline.py`：构造 `(nx, ny) = (64, 96)` 的 `Lossless(1.0)` 网格，跑 `simulate_jax_yee_ade_2d_probes`，断言透射探针形状为 `(num_steps,)` 且全部有限。

验收：`pytest tests/test_inverse_design_0_baseline.py` 通过；CPU 基线可复现。

### 步骤 1 — 无损端到端 autodiff（首个里程碑 M1）✅ 已完成

目标：`rho -> eps(rho_tilde) -> 透射能量目标` 整条链可 `jax.value_and_grad`，梯度经有限差分校验。

文件：`parameterization.py`、`objectives.py`、`optimize.py` 初版；`tests/test_inverse_design_grad.py`。

关键函数：

```python
def gaussian_blur(rho, sigma, kernel_width):          # jax.scipy.signal.convolve2d, "same"
    ...
def threshold_project(rho, beta, eta=0.5):            # 0.5*(tanh(beta*(rho-eta))/tanh(beta/2)+1)
    ...
def simp_interpolate(rho_tilde, eps_min, eps_max, p): # eps_min + rho_tilde**p * (eps_max - eps_min)
    ...
def transmission_energy_objective(probes):            # -mean(transmission_probe ** 2)，最小化取负
    ...
```

组装管线（整体可 jit）：

1. 用 `prepare_material_grid_2d` 建立固定背景（背景材料取 `eps = eps_min >= 1`，保证 CFL 全程成立），`jax_material_grid_2d` 转 JAX。
2. `design_mask`（`(nx, ny)` 布尔）落在内部区域，避开 CPML/海绵。
3. `a0_design = 1.0 / simp_interpolate(...)`；`coefficients = grid.coefficients.at[design_mask, 0].set(a0_design)`，设计区其余四系数保持 0（无损）。
4. `grid = grid._replace(coefficients=coefficients)`。
5. `simulate_jax_yee_ade_2d_probes(...)` 跑源，取透射探针，代入目标函数。

梯度校验：对 `rho` 中 3~5 个分量做中心差分，与 `jax.grad(jax.jit(f))(rho)` 对比，相对误差 `< 1e-4`。

验收：梯度校验通过；在一个 32×32 设计区上跑 10 步 L-BFGS-B，目标单调下降。

### 步骤 2 — 问题抽象层（Meep 式三层接口）

目标：把步骤 1 的手写管线收敛成可复用接口。

文件：`problem.py`；`objectives.py` 完善；`__init__.py` 惰性导出。

设计：

- `ObjectiveQuantity`（dataclass）：目标种类（`transmission_energy` / `field_match` / `mode_overlap`）、探针 y 位置、权重、可选频段窗口。
- `InverseDesignProblem`：持有背景 grid、`design_mask`、参数化超参（`sigma/beta/p/eps_min/eps_max`）、`material_spec`（先 `Lossless`）、源与探针配置、目标列表；提供 `loss(rho)`、`grad(rho)`、`solve(rho0)`。
- `optimize.py`：
  - `minimize_lbfgsb(problem, rho0, maxiter)`：开 `jax_enable_x64`，值/梯度转 `float64` 交给 scipy；
  - `minimize_adam(problem, rho0, steps, lr)`：走 optax；未安装时给出友好报错（当前环境未装 optax）。

验收：把步骤 1 的 demo 改写为“用 `problem` 类跑同一次优化”，结果一致；新名字加入 `fdtdz_jax/__init__.py` 的懒加载路由。

### 步骤 3 — 色散可微系数构造器（JAX ADE 色散扩展）✅ 已完成（五系数 + Multipole 极点）

目标：设计区从“只改 ε”升级为“固定色散模型下按 `rho` 插值出完整 ADE 系数”，无损退化为 `(1/ε, 0, 0, 0, 0)`。

文件：`materials_jax.py`；`tests/test_inverse_design_materials_jax.py`。

关键函数：

```python
def jax_ade_coefficients(material_spec, rho_tilde, dt):   # 纯 jnp，Lossless/Debye/Lorentz/Drude
    ...
def jax_pole_ade_coefficients(pole, dt):                   # jnp 版 q/r 五元组，对应 Multipole
    ...
def coefficients_from_design(rho, material_spec, dt, ...): # 返回可与静态背景合并的完整系数字段
    ...
```

插值方式采用“系数插值”：`c = c_bg + rho_tilde ** p * (c_mat - c_bg)`（对 a0..b2 与极点 q/r 都成立），设计变量仍是 `rho`，材料模型固定。

回归校验：`rho_tilde ∈ {0, 1}` 两个角点上，`coefficients_from_design` 与 NumPy `prepare_material_grid_2d` 输出逐元素一致（`1e-6`）；中间值有限且 `eps_inf >= eps_min` 满足 CFL。

示例：Debye 色散设计区，梯度校验 + 小规模 L-BFGS-B 优化一个色散滤波结构。

### 步骤 4 — 收尾：测试、文档、导出 ✅ 已完成（CUDA 黑盒线暂缓）

- 测试齐备：`tests/test_inverse_design_*.py` 全绿；端到端“50 步优化 + 梯度校验”标注 `@pytest.mark.slow`。
- 文档：`docs/inverse_design_jax_ade.md`（用户手册 + 可运行示例）。
- 导出：在 `fdtdz_jax/__init__.py` 的 `__all__` 与 `__getattr__` 中新增 `inverse_design` 相关名字，保持现有懒加载风格。
- 可选里程碑 M-CUDA（不阻塞主计划）：用 nevergrad / Optuna / BoTorch 对 `fdtdz` CUDA 内核做黑盒优化，验证薄片 / 小 z 问题价值；确认价值后再规划伴随 VJP。

## 5. 风险与坑（文档 + 代码核对）

- `eps_min >= 1`，背景就用 `eps_min`，保证与 `prepare_material_grid_2d` 的 CFL 校验一致。
- 设计区必须落在 CPML / 导电海绵之外（现有校验会拒绝色散或导电节点进入边界层）。
- `_probes` 只给 x 平均 Ez；需要空间分辨的目标改用 `_line_probes` 或全量 `simulate_jax_yee_ade_2d`。
- 大设计区内存：scan body 内 `jax.checkpoint`（remat）换内存；不够再上显式伴随 custom VJP（后续里程碑）。
- scipy 走 float64：开 `jax_enable_x64`，值和梯度统一 `float64`；当前环境无 optax，优先 scipy L-BFGS-B。
- 色散是“固定模型 + 几何/密度 `rho`”，梯度只流经 `rho`；材料参数不要当变量（除非后续明确需要）。

## 6. 验收口径（Definition of Done）

- M1：无损 2D TMz 设计区梯度通过有限差分校验；L-BFGS-B 能把透射目标从基线提升且收敛。
- M2：`inverse_design` 模块可用 `problem.loss / grad / solve` 复现 M1 结果。
- M3：`coefficients_from_design` 在无损 / 色散角点与 NumPy 一致；Debye 示例梯度校验通过。
- M4：`pytest tests/test_inverse_design_*.py` 全绿；文档与导出齐备。
