# IMAGINARY-FDTDZ：GPU 加速的 FDTD 与光学逆向设计工具包

[![Quality](https://github.com/Byron4779/GPU-based-FDTD-Inverse-Optical-Design-Toolkit/actions/workflows/quality.yml/badge.svg)](https://github.com/Byron4779/GPU-based-FDTD-Inverse-Optical-Design-Toolkit/actions/workflows/quality.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

IMAGINARY-FDTDZ （暂命名）是一个面向光子学、超表面与纳米光学研究的开源时域有限差分（FDTD）
工具包。项目保留原始的 CUDA 吞吐量路线，同时提供基于 NumPy/JAX 的 ADE
（辅助微分方程）求解器，用于二维和三维电磁正向仿真、色散材料建模、
散射谱分析及可微逆向设计。

本仓库重点验证**复介电常数材料**的电磁响应：时域场保持实数，频域复数
介电常数通过 Debye、Lorentz、Drude 或多极点 ADE 模型映射到时间推进中。
这使工具包能够处理有损介质、共振介质、金属以及由实测光学数据拟合得到的
多极点材料。

> 当前状态：正向仿真与材料工具已形成完整测试链路；二维 TMz JAX ADE
> 逆向设计接口处于实验阶段。连续密度与单频相位约束已有收敛案例，严格
> 二值金属结构仍不是稳定求解器，详见[逆向设计成熟度](#逆向设计接口与收敛状态)。

## 主要能力

- **GPU 与可移植后端**：原始 CUDA/PTX 求解器面向高吞吐计算；NumPy/JAX
  ADE 路径可在 CPU 或 GPU 上运行。
- **二维与三维电磁仿真**：支持 2D TMz、2D TEz 和 3D 全矢量 FDTD。
- **复介电常数材料**：支持无损耗介质、Debye 弛豫、Lorentz 共振、
  Drude 自由电子金属及混合多极点材料。
- **定量散射工作流**：参考场/器件场成对仿真，输出反射、透射、吸收谱；
  二维和三维工作流支持周期衍射级次分解。
- **周期与斜入射**：支持周期单元、Bloch 相位、CPML 吸收层及三维斜入射
  薄片/超表面算例。
- **材料数据互操作**：可从版本化 JSON/HDF5 文件加载材料，也可从
  `n–k` 或复介电常数表拟合被动 ADE 多极点模型。
- **可微逆向设计**：JAX 自动微分贯穿 FDTD 时间推进，提供密度滤波、
  投影、SIMP/ADE 系数插值、约束优化和二值直线估计器（STE）。
- **工程质量检查**：包含梯度有限差分校验、数值收敛研究、物理单位检查、
  结果 I/O、材料回归测试和 GitHub Actions CI。

## 复介电常数正向仿真

工具包不是把复数 `epsilon_r` 直接写入实时时域场，而是将频域材料响应写成
被动因果材料的辅助微分方程，再在 Yee 网格上推进实值时域场。常用模型为

| 模型 | 适用场景 | 频域相对介电常数 |
| --- | --- | --- |
| `Lossless` | 空气、玻璃、低损耗近似介质 | `epsilon_r`（常数） |
| `Debye` | 弛豫损耗、部分生物/聚合物介质 | `epsilon_inf + (epsilon_s - epsilon_inf)/(1 - j*omega*tau)` |
| `Lorentz` | 共振吸收、激子或介质谐振 | 单极点 Lorentz 响应 |
| `Drude` | 自由电子气、简单金属近似 | `epsilon_inf - omega_p^2/(omega^2 + j*gamma*omega)` |
| `Multipole` | 金属、实测材料及宽频拟合 | `epsilon_inf` 与任意 Debye/Lorentz/Drude 极点之和 |

材料参数使用求解器的归一化角频率和时间单位。若输入来自波长、Hz、eV 或
SI 材料表，应先通过 `PhysicalScale`、`read_material_table` 和
`fit_passive_material` 完成单位转换与拟合。仓库附带 Rakić-1998 金材料的
多极点 HDF5 示例，在 800 nm 处可用于金属超表面正向算例；它是示例数据，
不是对所有波长和加工条件的通用材料证书。

二维复介电常数超表面的典型流程如下：

```python
from fdtdz_jax import read_material, run_tmz_scattering_2d
from fdtdz_jax.materials import Lossless

metal = read_material("examples/materials.json", "metal")
materials = [Lossless(1.0), metal]

result = run_tmz_scattering_2d(
    materials,
    reference_material_ids,
    device_material_ids,
    dx,
    dy,
    dt,
    waveform,
    source_y=14,
    reflection_probe_y=18,
    transmission_probe_y=46,
    cpml_width=8,
    backend="jax",
)
```

完整参数与定量结果要求见
[二维复介电常数工作流](docs/complex_permittivity_2d.md)。做定量结论前，
应至少检查材料拟合误差、设计区与 CPML 的间距、参考场归一化，以及网格、
时间步和 CPML 的收敛性。

## 逆向设计接口与收敛状态

当前逆向设计接口位于 `fdtdz_jax.inverse_design`，主要面向 **2D TMz JAX
ADE**。接口采用 `DesignParameters`、`DesignRegion`、`ObjectiveQuantity`
和 `InverseDesignProblem` 组织问题，并用 JAX 反向自动微分计算离散伴随
梯度。二维设计区可以使用 `Lossless`、`Debye`、`Lorentz`、`Drude` 和
`Multipole` 材料。

已实现的目标量和优化入口包括：

- `ProbeTrace`：直接使用透射或反射探针的时域轨迹，可构造能量、窗口
  能量或自定义标量目标。
- `ProbeEnergy`：窗口内探针的均方幅值。
- `SingleFrequencyTransmission`：用 Hann 窗 DFT 和空单元参考场归一化的
  单频复透射系数 `t(f0)`。
- `phase_180_constraints`：控制复透射相位差达到 180°。
- `minimize_lbfgsb`：无约束可微目标。
- `minimize_slsqp`、`minimize_slsqp_beta_continuation`：含物理约束和
  设计域约束的连续密度优化。
- `minimize_binary_ste`、`binary_flip_local_search`：严格 0/1 端点材料的
  二值优化及实测翻转清理。

当前收敛证据应按工作流区分：

| 工作流 | 目标 | 当前已验证状态 |
| --- | --- | --- |
| 无损耗连续密度 | 最小化负时间积分透射能量 | `L-BFGS-B` 能完成迭代并降低目标值；梯度与有限差分一致 |
| 全周期无损耗连续密度 | 固定 `f0` 相位 180°，最大化 `|t(f0)|^2` | `SLSQP` 测试可满足相位和约束容差，并相对可行初值提升透射率 |
| 800 nm 金连续松弛 | 单频 180° 相位、幅值、透射功率、体积和有效厚度联合约束 | 30% 金体积上限的案例满足该连续松弛的全部硬约束；20% 和 40% 案例失败，不能视为通用可解性 |
| 800 nm 金严格二值设计 | 与上相同的目标，但每次正向仿真只使用 `{0,1}` 端点材料 | 优化器可以严格执行二值和独立验证，但默认案例尚未通过目标容差，当前**未收敛为有效器件** |

注：和tidy3d结果比较（不在本项目内），现有成熟软件依旧无法收敛为有效器件，所以是算例的问题

严格二值 800 nm 金代理梯度实验的实测结果如下。表中失败结果被保留，用于区分
“优化器正在工作”和“已经找到合格器件”：

| 最大 Au 体积分数 | `T` | 相位误差 | 有效 Au 体积 | 有效厚度 | 有效器件 |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0.20 | 0.074835 | 42.017° | 0.1641 | 1.312 | 否 |
| 0.30 | 0.003961 | 12.330° | 0.2461 | 1.969 | 否 |
| 0.40 | 0.001154 | 1.099° | 0.2930 | 2.344 | 否 |

因此，现阶段可以把该接口用于连续密度、单频相位约束和色散材料梯度研究，
但不应把严格二值金超表面结果用于器件指标承诺。接口使用说明、限制和完整
示例见 [JAX ADE 逆向设计手册](docs/inverse_design_jax_ade.md)，约束构造见
[设计域约束](docs/design_domain_constraints.md)，金二值实验记录见
[800 nm 严格二值结果](docs/gold_binary_800nm_results.md)。

## 支持的材料

正向求解器在 JSON/HDF5、二维/三维网格、仿真对象和逆向设计区中使用一致的
材料类型：

| 类别 | 支持情况 |
| --- | --- |
| 无损耗各向同性介质 | `Lossless(epsilon_r)`；可用于二维、三维和逆向设计 |
| 单极点弛豫材料 | `Debye(epsilon_inf, epsilon_static, tau)` |
| 单极点共振材料 | `Lorentz(epsilon_inf, epsilon_static, omega_0, damping)` |
| 自由电子金属 | `Drude(epsilon_inf, plasma_frequency, collision_frequency)` |
| 混合/拟合多极点材料 | `Multipole(epsilon_inf, poles)`，极点为 Debye、Lorentz 或 Drude |
| 外部材料数据 | 版本化 JSON/HDF5 schema；可从 `n–k` 或复介电常数表拟合固定极点基 |
| 示例金属 | Rakić-1998 金多极点 HDF5 文件及可复现生成脚本 |

材料拟合器约束非负的 Debye/Lorentz 强度和非负的 Drude 等离子体频率平方，
并让 Lorentz 阻尼采用差分方程中 `2*delta*d/dt` 的 `delta` 约定。它完成的是
固定极点形状的强度拟合，不自动发现极点位置、线宽或全局因果模型；拟合后仍需
检查 `material_accuracy_report`、留出频段和 FDTD 收敛性。

## 安装

要求 Python 3.10+。若只需 NumPy/JAX 求解器，无需 CUDA 编译器：

```bash
git clone https://github.com/Byron4779/GPU-based-FDTD-Inverse-Optical-Design-Toolkit.git
cd GPU-based-FDTD-Inverse-Optical-Design-Toolkit
FDTDZ_BUILD_CUDA=0 python -m pip install "."
```

Windows PowerShell：

```powershell
$env:FDTDZ_BUILD_CUDA = "0"
python -m pip install .
```

可选依赖按需安装：

```bash
python -m pip install ".[hdf5]"   # HDF5 材料文件，包括示例金材料
python -m pip install ".[fit]"    # 从光学材料表拟合 ADE 极点
python -m pip install ".[plot]"   # 绘图
```

需要 JAX GPU 时，请先按本机 CUDA 环境安装对应 JAX/JAXLIB 版本，再安装
FDTDZ。完整安装方式：

```bash
python -m pip install "jax[cuda12]"
python -m pip install "."
```

构建原始 CUDA 扩展还需要 CMake、C++ 编译器、NVIDIA CUDA Toolkit 和兼容的
GPU。可用 `FDTDZ_CUDA_ARCHITECTURES` 指定目标架构，例如 `75;80;86;89`。
安装后运行 `examples/validate_jax_gpu_2d.py` 或
`examples/validate_jax_gpu_3d.py`，确认 JAX 实际使用的是 GPU。

## 快速开始

直接从源码检查二维 TEz 超表面算例：

```bash
python examples/metasurface_2d_tez.py
```

运行二维复介电常数 TMz 算例：

```bash
python examples/metasurface_2d_tmz.py
```

运行无损耗连续密度透射优化：

```bash
python examples/inverse_design_2d.py
```

运行全周期 180° 相位约束优化：

```bash
python examples/inverse_design_2d_phase_constraint.py
```

其他可用入口：

- [`complex-permittivity-metasurface-3d.ipynb`](complex-permittivity-metasurface-3d.ipynb)：
  三维复介电常数算例。
- [`inverse_design_2d_gold_hdf5.ipynb`](inverse_design_2d_gold_hdf5.ipynb)：
  800 nm 金材料的连续松弛和严格二值验证。
- `examples/gold_rakic_1998.h5`：示例金材料定义。
- `examples/fit_material_3d.py`：材料表读取、被动 ADE 拟合和三维薄膜检查。
- `examples/oblique_film_3d.py`：三维斜入射与 Bloch 边界示例。
- `examples/convergence_3d.py`：三维数值收敛研究。

## 测试与验证

安装测试依赖并运行 CPU 测试：

```bash
FDTDZ_BUILD_CUDA=0 python -m pip install -e ".[test]"
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q
```

逆向设计的无损耗透射优化测试标记为 `slow`：

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest \
  tests/test_inverse_design_optimize.py -q -m slow
```

梯度正确性通过后向自动微分与有限差分对照检查。数值精度结论必须同时给出
网格、时间步、CPML 厚度和运行时长；GPU 测试还需记录 JAX/JAXLIB、驱动、
CUDA Toolkit 和 GPU 型号。相关说明见
[GPU 验证](docs/gpu_validation.md) 与
[JAX GPU 性能分析](docs/jax_gpu_profiling.md)。

## 当前限制

- 逆向设计目前只覆盖二维 TMz。
- 宽带同时约束相位与透射、多目标 Pareto 优化、鲁棒性和加工公差优化尚未
  形成稳定工作流。
- 严格二值优化使用直线估计器生成候选方向，真实有限差分仿真仍是最终判据；
  当前 800 nm 金默认算例没有找到满足全部容差的二值器件。
- 时域求解器只推进实值场；频域复介电常数必须映射为受支持的并行 ADE 模型。
- 旧版 CUDA systolic 内核与 JAX ADE 网格/历史布局不同，二维逆向设计不会
  自动加速三维 CUDA 内核。
- 当前材料拟合基于固定极点基，不能替代完整的非线性色散模型发现或广泛的
  实验材料数据库。

## 目录结构

```text
src/fdtdz_jax/             Python、NumPy 与 JAX 求解器、材料、前后处理
src/fdtdz_jax/inverse_design/
                           密度参数化、目标量、约束、优化器与问题抽象
cuda/                      原始 CUDA/PTX 内核与 JAX FFI 绑定
tests/                     正向、材料、收敛、梯度和逆设计回归测试
examples/                  可运行算例、材料文件和 notebook
docs/                      材料、GPU、斜入射、约束和逆设计说明
tools/                     构建、分发检查、GPU 验证和 notebook 工具
```

## 参与贡献

GitHub 上的缺陷报告、功能建议和拉取请求均欢迎。提交前请阅读
[贡献指南](CONTRIBUTING.md)、[行为准则](CODE_OF_CONDUCT.md) 和
[安全策略](SECURITY.md)。行为变化应附带测试和文档；涉及 CUDA/JAX 的变更
请注明实际硬件与版本环境。功能建议和缺陷报告可使用仓库中的 Issue 模板。

## 许可与来源

本项目采用 [MIT License](LICENSE)。FDTDZ 原始项目由
[SPINS Photonics](https://github.com/spinsphotonics/fdtdz) 维护；本仓库在其
基础上继续开发材料工具、JAX ADE 求解器和逆向设计实验。逆向设计接口参考
Meep `adjoint` 的公开接口与已发表公式，但梯度实现基于 JAX 自动微分，不包含
Meep 的 GPL 源码。

## 文档索引

- [二维复介电常数超表面](docs/complex_permittivity_2d.md)
- [JAX ADE 逆向设计](docs/inverse_design_jax_ade.md)
- [设计域约束与二值验证](docs/design_domain_constraints.md)
- [材料文件 schema](docs/material_files.md)
- [光学材料表与 ADE 拟合](docs/material_fitting.md)
- [三维斜入射](docs/oblique_incidence_3d.md)
- [GPU 环境验证](docs/gpu_validation.md)
- [JAX GPU 性能分析](docs/jax_gpu_profiling.md)
