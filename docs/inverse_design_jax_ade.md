# JAX ADE 逆向设计（inverse_design）用户手册

`fdtdz_jax.inverse_design` 把现有的可微 JAX ADE 求解器封装成逆向设计接口，
接口形状对齐 Meep 的 `meep.adjoint`（`OptimizationProblem` / `ObjectiveQuantity`
/ `DesignRegion` / `MaterialGrid` 风格），但梯度机制用的是 JAX 端到端自动微分
（等价于离散伴随），而不是 Meep 的显式伴随。许可上只对齐接口与已发表公式，
不包含 Meep 的 GPL 代码。

## 环境

- Python 3.10+，`jax`（CPU 或 GPU 均可）、`numpy`、`scipy`（L-BFGS-B 用）。
- L-BFGS-B 走 float64：模块内部会打开 `jax_enable_x64`。
- 运行测试时若报 `ModuleNotFoundError: No module named 'anyio.pytest_plugin'`，
  用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests` 绕过损坏的插件入口。

## 快速开始（无损 2D TMz，最大化透射能量）

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from fdtdz_jax import (
    DesignParameters,
    DesignRegion,
    InverseDesignProblem,
    ProbeTrace,
    minimize_lbfgsb,
)
from fdtdz_jax.materials import Lossless
from fdtdz_jax.metasurface_2d import gaussian_sine_pulse, huygens_current_waveforms
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d, prepare_y_cpml

nx, ny = 32, 64
dt, dx, dy = 0.5, 1.0, 1.0
eps_min, eps_max = 1.0, 9.0

cpml = prepare_y_cpml((nx, ny), 8, dt, dy, dtype=np.float64)
grid = prepare_material_grid_2d(
    [Lossless(eps_min)], np.zeros((nx, ny), dtype=int),
    dt, dx, dy, dtype=np.float64, y_cpml=cpml,
)

num_steps = 240
waveform = gaussian_sine_pulse(num_steps, dt, 20.0, 10.0, 0.3, dtype=np.float64)
electric, magnetic = huygens_current_waveforms(
    waveform, direction=1, wave_impedance=1.0, dtype=np.float64
)

grid_size = (12, 8)
params = DesignParameters(
    grid_size=grid_size,
    medium1=eps_min,          # 密度 0 -> eps_min
    medium2=eps_max,          # 密度 1 -> eps_max
    weights=np.full(grid_size[0] * grid_size[1], 0.5),
    beta=2.0,                 # tanh 投影强度（0 表示不投影）
    p=3.0,                    # SIMP 指数
    filter_type="conic",      # None / "conic" / "gaussian"
    radius=1.5,
)
region = DesignRegion(
    params, x0=10, x1=10 + grid_size[0], y0=28, y1=28 + grid_size[1]
)

problem = InverseDesignProblem(
    grid,
    region,
    electric,
    magnetic,
    np.ones(nx, dtype=np.float64),
    source_y=14,
    reflection_probe_y=18,
    transmission_probe_y=46,
    objective_arguments=[ProbeTrace(probe="transmission")],
    objective_functions=[lambda t: -jnp.mean(jnp.square(t))],
)

f0 = problem(need_value=True, need_gradient=False)[0]
result = minimize_lbfgsb(
    problem, np.full(grid_size[0] * grid_size[1], 0.5), maxiter=30
)
f1 = problem(need_value=True, need_gradient=False)[0]
print(f0, f1)  # f1 应小于 f0（透射能量增大）
```

完整可执行脚本见 `examples/inverse_design_2d.py`。

## 色散设计区（Debye / Lorentz / Drude）

把 `medium1` / `medium2` 换成材料对象即可。此时按完整五元组 ADE 系数
`(a0,a1,a2,b1,b2)` 做密度插值：

```python
from fdtdz_jax.materials import Debye

params = DesignParameters(
    grid_size=grid_size,
    medium1=Lossless(1.0),
    medium2=Debye(eps_inf=1.0, eps_static=3.0, tau=0.4),
    weights=np.full(grid_size[0] * grid_size[1], 0.5),
    p=3.0,
)
```

## Multipole（多极点）设计区

`Multipole` 走极点 q/r 递推。关键要求：背景网格的 `materials` 列表必须包含该
`Multipole` 材料（即使 `ids` 全是背景），这样 `maximum_poles` 才正确：

```python
from fdtdz_jax.materials import Multipole, LorentzPole, DrudePole

design = Multipole(1.0, (LorentzPole(1.2, 3.0, 0.2), DrudePole(2.0, 0.4)))
grid = prepare_material_grid_2d(
    [Lossless(1.0), design], np.zeros((nx, ny), dtype=int),
    dt, dx, dy, dtype=np.float64, y_cpml=cpml,
)
params = DesignParameters(
    grid_size=grid_size,
    medium1=Lossless(1.0),
    medium2=design,
    weights=np.full(grid_size[0] * grid_size[1], 0.5),
    p=3.0,
)
```

## 核心 API

- `DesignParameters(grid_size, medium1, medium2, weights, beta, eta, damping, p,
  filter_type, radius, sigma, dx, dy, periodic_axes)`：设计变量与材料插值；
  `medium1`/`medium2` 接受数字、`Lossless`、`Debye`、`Lorentz`、`Drude` 或
  `Multipole`。`periodic_axes=(0,)` 表示滤波沿横向 x 周期卷绕，`(0, 1)`
  表示两个方向都周期；`None` 保持非周期边缘填充。
- `DesignRegion(design_parameters, x0, x1, y0, y1)`：Yee 网格上的矩形设计区。
- `InverseDesignProblem(grid, design_regions, electric_waveform,
  magnetic_waveform, x_profile, source_y, reflection_probe_y,
  transmission_probe_y, objective_arguments, objective_functions)`：
  提供 `__call__(rho, need_value, need_gradient, beta)`、`get_fdf_funcs()`、
  `calculate_fd_gradient(...)`、`update_design(...)`。
- `ObjectiveQuantity` / `ProbeTrace` / `ProbeEnergy`：目标量；目标函数接收
  各目标量的取值并返回标量。
- `coefficients_from_design(density, background_material, design_material, dt,
  p, maximum_poles, dtype)`：返回 `DesignCoefficients`（`coefficients` /
  `multipole_mask` / `multipole_eps_inf` / `pole_coefficients`），供角点回归等。
- `minimize_lbfgsb(problem, rho0, maxiter, bounds)`：scipy L-BFGS-B 驱动。
- `minimize_binary_ste(problem, latent0, steps, learning_rate,
  constraint_penalty)`：严格二值前向的 Adam/STE 优化器；目标和约束由实际
  0/1 材料计算，代理梯度仅用于更新 latent。返回结果中的 `latent` 不是器
  件，`binary_densities` 才是物理解。
- `binary_flip_local_search(problem, latent0, iterations, candidate_count)`：
  对真实二值候选逐个仿真，只接受罚函数目标实测下降的翻点。
- `conic_filter` / `gaussian_filter` / `cylindrical_filter` / `tanh_projection`：
  与 Meep `filters.py` 同公式的滤波/投影算子，支持周期与非周期轴混合卷积。

## 注意事项

- 设计区必须在 CPML / 导电海绵之外（现有校验会拒绝色散或导电节点进入边界）。
- 设计区覆盖仿真网格的完整横向周期时，应设置 `periodic_axes=(0,)`；局部设计窗
  不应开启该轴，否则会把窗口边界按同一周期首尾相连。
- `eps_min >= 1`，背景用 `eps_min`，保证 CFL 全程成立。
- `_probes` 只返回 x 平均的 Ez；需要空间分辨的目标改用 `_line_probes` 或全量
  `simulate_jax_yee_ade_2d`。
- 端到端优化慢测见 `tests/test_inverse_design_optimize.py`（`@pytest.mark.slow`）。
