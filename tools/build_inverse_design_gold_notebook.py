"""Build the 800 nm gold strict-binary inverse-design notebook."""

import json
from pathlib import Path
import textwrap
import warnings

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "inverse_design_2d_gold_hdf5.ipynb"


CELLS = (
("markdown", r"""
# 800 nm 金材料的严格二值逆向设计

本 notebook 使用 HDF5 Rakic 金材料。设计区中的每一次 FDTD/ADE 仿真都只使用
空气或金两个端点材料，不会把连续混合材料当作物理器件。

优化器保留连续 latent 作为翻转提案变量。实际进入求解器的材料密度始终为：

```text
rho_material = 0 或 1
```

目标、相位、透射率、体积和厚度全部由该二值材料计算。latent 只用于生成直通
梯度，不会作为最终设计保存。
"""),
("code", r"""
import os
import sys
import warnings
from pathlib import Path

candidate_roots = (Path.cwd(), Path.cwd().parent)
repo_root = next(
    path for path in candidate_roots
    if (path / "examples" / "gold_rakic_1998.h5").exists()
)
sys.path.insert(0, str(repo_root / "src"))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from fdtdz_jax import (
    DesignGeometryLimits,
    DesignParameters,
    DesignRegion,
    InverseDesignProblem,
    PhysicalScale,
    SingleFrequencyTransmission,
    binary_flip_local_search,
    binary_topology_metrics,
    density_metrics,
    effective_thickness_constraints,
    metal_volume_fraction_constraints,
    minimize_binary_ste,
    phase_180_constraints,
    read_material,
    sparse_random_design,
    transmission_metrics,
)
from fdtdz_jax.materials import Lossless
from fdtdz_jax.metasurface_2d import gaussian_sine_pulse, huygens_current_waveforms
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d, prepare_y_cpml
"""),
("markdown", r"""
## 1. 800 nm 光源与 HDF5 金材料

背景为空气，y 方向使用 8 层 CPML。设计区覆盖完整的横向周期。
"""),
("code", r"""
nx, ny = 32, 64
dt, dx, dy = 0.5, 1.0, 1.0
cpml_width = 8

gold_material_file = repo_root / "examples" / "gold_rakic_1998.h5"
gold = read_material(gold_material_file, "gold")

physical_scale = PhysicalScale(length_unit_m=48e-9)
target_wavelength_m = 800e-9
normalized_wavelength = float(physical_scale.wavelength(target_wavelength_m))
source_frequency = 1.0 / normalized_wavelength
source_omega = 2.0 * np.pi * source_frequency
gold_eps_at_source = complex(gold.relative_permittivity(source_omega))

num_steps = int(np.ceil(8.0 * 2.0 / source_frequency / dt))
pulse_sigma = 2.0 / source_frequency
pulse_center_time = 100.0

cpml = prepare_y_cpml((nx, ny), cpml_width, dt, dy, dtype=np.float64)
base_grid = prepare_material_grid_2d(
    [Lossless(1.0), gold],
    np.zeros((nx, ny), dtype=int),
    dt,
    dx,
    dy,
    dtype=np.float64,
    y_cpml=cpml,
)

waveform = gaussian_sine_pulse(
    num_steps,
    dt,
    center_time=pulse_center_time,
    width=pulse_sigma,
    frequency=source_frequency,
    dtype=np.float64,
)
electric, magnetic = huygens_current_waveforms(
    waveform, direction=1, wave_impedance=1.0, dtype=np.float64
)

print("Loaded Au material:", gold_material_file)
print(f"Normalized wavelength: {normalized_wavelength:.6f}")
print(f"Normalized frequency:  {source_frequency:.6f}")
print(
    f"Au epsilon at source: {gold_eps_at_source.real:.4f}"
    f"{gold_eps_at_source.imag:+.4f}i"
)
print("Pulse samples:", num_steps)
"""),
("markdown", r"""
## 2. 固定几何与严格二值问题

`binary_material=True` 让 conic filter 只作用于 latent，随后立即硬阈值到 0/1。
体积和有效厚度约束读取的密度与 FDTD/ADE 求解器读取的密度完全相同。
"""),
("code", r"""
grid_size = (32, 8)
design_y0 = 28
source_y = 14
reflection_probe_y = 18
transmission_probe_y = 46

geometry_limits = DesignGeometryLimits(
    min_width=32,
    max_width=32,
    min_height=8,
    max_height=8,
    cpml_clearance=0,
    source_clearance=4,
    reflection_probe_clearance=2,
    transmission_probe_clearance=4,
)

target_frequency = float(source_frequency)
transmission = SingleFrequencyTransmission(
    frequency=target_frequency,
    dt=dt,
    window="hann",
    reference_floor=1e-10,
)
physical_constraints = phase_180_constraints(
    amplitude_floor=1e-4,
    power_limit=1.0,
)

VOLUME_LIMITS = tuple(
    float(value) for value in os.environ.get(
        "FDTDZ_GOLD_VOLUME_LIMITS", "0.2,0.3,0.4"
    ).split(",")
)
SPARSE_SEEDS = tuple(
    int(value) for value in os.environ.get(
        "FDTDZ_GOLD_SEEDS", "11,23,37"
    ).split(",")
)
BINARY_STEPS = int(os.environ.get("FDTDZ_GOLD_BINARY_STEPS", "120"))
BINARY_LEARNING_RATE = float(
    os.environ.get("FDTDZ_GOLD_BINARY_LR", "0.01")
)
CONSTRAINT_PENALTY = float(
    os.environ.get("FDTDZ_GOLD_BINARY_PENALTY", "20")
)
THRESHOLD_MARGIN = float(
    os.environ.get("FDTDZ_GOLD_BINARY_MARGIN", "0.002")
)
LOCAL_SEARCH_ITERATIONS = int(
    os.environ.get("FDTDZ_GOLD_LOCAL_SEARCH_ITERATIONS", "0")
)
LOCAL_SEARCH_CANDIDATES = int(
    os.environ.get("FDTDZ_GOLD_LOCAL_SEARCH_CANDIDATES", "32")
)

PHASE_TOLERANCE_DEG = 0.1
PHASE_RESIDUAL_TOLERANCE = 1e-4
AMPLITUDE_FLOOR = 1e-4
MIN_FEATURE_CELLS = 1


def make_problem(volume_max, rho0, reference_transmission=None):
    params = DesignParameters(
        grid_size=grid_size,
        medium1=1.0,
        medium2=gold,
        weights=np.asarray(rho0, dtype=np.float64).reshape(-1),
        beta=0.0,
        eta=0.5,
        p=3.0,
        filter_type="conic",
        radius=1.5,
        periodic_axes=(0,),
        binary_material=True,
    )
    region = DesignRegion(
        params,
        x0=0,
        x1=grid_size[0],
        y0=design_y0,
        y1=design_y0 + grid_size[1],
    )
    design_constraints = (
        *metal_volume_fraction_constraints(minimum=0.0, maximum=volume_max),
        *effective_thickness_constraints(minimum=2.0, maximum=4.0),
    )
    problem = InverseDesignProblem(
        base_grid,
        region,
        electric,
        magnetic,
        np.ones(nx, dtype=np.float64),
        source_y=source_y,
        reflection_probe_y=reflection_probe_y,
        transmission_probe_y=transmission_probe_y,
        objective_arguments=[transmission],
        objective_functions=[
            lambda coefficient: -jnp.abs(coefficient) ** 2
        ],
        constraints=physical_constraints,
        reference_transmission=reference_transmission,
        design_constraints=design_constraints,
        geometry_limits=geometry_limits,
    )
    return problem, params, region


seed_problem, _, _ = make_problem(
    volume_max=VOLUME_LIMITS[-1],
    rho0=np.zeros(grid_size[0] * grid_size[1]),
)
empty_reference_transmission = seed_problem.reference_transmission
print("Empty-cell reference transmission precomputed.")
print("Geometry checks passed for the fixed design box.")
"""),
("markdown", r"""
## 3. Bernoulli 种子与严格二值优化

每个候选种子本身是 Bernoulli 0/1 图。优化器使用二值前向和直通梯度更新
latent；每轮记录的实际材料都是 0/1。可选局部搜索会逐个仿真真实二值翻点，
只接受实测罚函数目标下降的结构。
"""),
("code", r"""
def find_sparse_phase_seed(problem, volume_max):
    best = None
    target_fraction = min(0.75 * volume_max, 0.4)
    rows = []
    for seed in SPARSE_SEEDS:
        candidate = sparse_random_design(
            grid_size,
            target_fraction=target_fraction,
            seed=seed,
        ).reshape(-1)
        coefficient = complex(problem.evaluate_quantities(candidate)[0])
        amplitude = abs(coefficient)
        residual = (
            float("inf")
            if amplitude < AMPLITUDE_FLOOR
            else abs(1.0 + coefficient.real / amplitude)
        )
        rows.append((seed, float(np.mean(candidate)), amplitude, residual))
        if best is None or residual < best[0]:
            best = (residual, candidate, coefficient, seed)
    if best is None:
        raise RuntimeError("No sparse Bernoulli phase candidate was evaluated.")
    for seed, fraction, amplitude, residual in rows:
        print(
            f"  seed={seed:2d}, initial gold={fraction:.3f}, "
            f"|t|={amplitude:.4f}, phase residual={residual:.4e}"
        )
    return best[1], best[2], best[3]


def run_binary_case(volume_max):
    initial_problem, _, _ = make_problem(
        volume_max,
        np.zeros(grid_size[0] * grid_size[1]),
        reference_transmission=empty_reference_transmission,
    )
    rho_seed, initial_coefficient, selected_seed = find_sparse_phase_seed(
        initial_problem, volume_max
    )
    problem, params, region = make_problem(
        volume_max,
        rho_seed,
        reference_transmission=empty_reference_transmission,
    )
    print(f"\nvolume_max={volume_max:.2f}, selected sparse seed={selected_seed}")

    optimization = minimize_binary_ste(
        problem,
        rho_seed,
        steps=BINARY_STEPS,
        learning_rate=BINARY_LEARNING_RATE,
        constraint_penalty=CONSTRAINT_PENALTY,
        threshold_margin=THRESHOLD_MARGIN,
        tolerance=PHASE_RESIDUAL_TOLERANCE,
    )
    latent = optimization.latent
    rho_material = optimization.binary_densities[0].copy()
    local_search = None
    if LOCAL_SEARCH_ITERATIONS:
        local_search = binary_flip_local_search(
            problem,
            latent,
            iterations=LOCAL_SEARCH_ITERATIONS,
            candidate_count=LOCAL_SEARCH_CANDIDATES,
            constraint_penalty=CONSTRAINT_PENALTY,
            tolerance=PHASE_RESIDUAL_TOLERANCE,
        )
        latent = local_search.latent
        rho_material = local_search.binary_densities[0].copy()

    coefficient = complex(problem.evaluate_quantities(latent)[0])
    raw = problem.get_constrained_fdfuncs()(latent)
    objective_value = float(np.asarray(raw[0]))
    max_constraint_violation = 0.0
    equality = np.asarray(raw[2])
    inequality = np.asarray(raw[4])
    if equality.size:
        max_constraint_violation = max(
            max_constraint_violation,
            float(np.max(np.abs(equality))),
        )
    if inequality.size:
        max_constraint_violation = max(
            max_constraint_violation,
            float(np.max(np.maximum(-inequality, 0.0))),
        )
    metrics = transmission_metrics(
        coefficient,
        amplitude_floor=AMPLITUDE_FLOOR,
    )
    print(
        f"  final objective={objective_value:+.6f}, "
        f"T={metrics.transmittance:.6f}, "
        f"phase={metrics.phase_degrees:+.3f} deg, "
        f"max_violation={max_constraint_violation:.3e}, "
        f"Au={np.mean(rho_material):.4f}"
    )
    return {
        "volume_max": volume_max,
        "seed": selected_seed,
        "problem": problem,
        "params": params,
        "region": region,
        "rho_seed": rho_seed,
        "initial_coefficient": initial_coefficient,
        "optimization": optimization,
        "local_search": local_search,
        "latent": latent.copy(),
        "objective_value": objective_value,
        "max_constraint_violation": max_constraint_violation,
        "coefficient": coefficient,
        "rho_material": rho_material,
    }


binary_cases = []
for volume_max in VOLUME_LIMITS:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        case = run_binary_case(volume_max)
    for warning_message in caught:
        print("optimizer warning:", warning_message.message)
    binary_cases.append(case)
"""),
("markdown", r"""
## 4. 独立端点材料验收

优化器返回的密度必须已经只有 0/1。这里再次建立全新的、无滤波的端点材料
问题并重跑 ADE 求解，避免把 latent 或混合材料结果当作器件。
"""),
("code", r"""
def validate_binary_case(case):
    rho_binary = np.asarray(case["rho_material"], dtype=np.float64)
    if not np.all((rho_binary == 0.0) | (rho_binary == 1.0)):
        raise RuntimeError("optimizer returned non-binary material density.")

    binary_params = DesignParameters(
        grid_size=grid_size,
        medium1=1.0,
        medium2=gold,
        weights=rho_binary.reshape(-1),
        beta=0.0,
        p=1.0,
        filter_type=None,
        periodic_axes=(0,),
        binary_material=True,
    )
    binary_region = DesignRegion(
        binary_params,
        x0=case["region"].x0,
        x1=case["region"].x1,
        y0=case["region"].y0,
        y1=case["region"].y1,
    )
    binary_problem = InverseDesignProblem(
        base_grid,
        binary_region,
        electric,
        magnetic,
        np.ones(nx, dtype=np.float64),
        source_y=source_y,
        reflection_probe_y=reflection_probe_y,
        transmission_probe_y=transmission_probe_y,
        objective_arguments=[transmission],
        objective_functions=case["problem"].objective_functions,
        constraints=physical_constraints,
        reference_transmission=empty_reference_transmission,
        geometry_limits=geometry_limits,
    )
    binary_coefficient = complex(binary_problem.evaluate_quantities()[0])
    binary_transmission = transmission_metrics(
        binary_coefficient,
        amplitude_floor=AMPLITUDE_FLOOR,
    )
    binary_density = density_metrics(rho_binary)
    binary_topology = binary_topology_metrics(rho_binary)

    volume_ok = (
        binary_topology.metal_fraction
        <= case["volume_max"] + 1e-12
    )
    thickness_ok = 2.0 <= binary_density.effective_thickness <= 4.0
    phase_ok = binary_transmission.phase_error_degrees <= PHASE_TOLERANCE_DEG
    residual_ok = (
        case["max_constraint_violation"] <= PHASE_RESIDUAL_TOLERANCE
    )
    feature_ok = (
        binary_topology.minimum_metal_feature >= MIN_FEATURE_CELLS
    )
    valid = all(
        (
            phase_ok,
            residual_ok,
            binary_transmission.amplitude_ok,
            binary_transmission.power_ok,
            volume_ok,
            thickness_ok,
            feature_ok,
            binary_density.gray_fraction == 0.0,
        )
    )

    print(
        f"binary f_max={case['volume_max']:.2f}: "
        f"T={binary_transmission.transmittance:.6f}, "
        f"phase={binary_transmission.phase_degrees:+.3f} deg, "
        f"phase_error={binary_transmission.phase_error_degrees:.3f} deg, "
        f"Au={binary_topology.metal_fraction:.4f}, "
        f"H_eff={binary_density.effective_thickness:.3f}, "
        f"min_feature={binary_topology.minimum_metal_feature} cells, "
        f"Au_components={binary_topology.metal_components}, "
        f"air_components={binary_topology.air_components}, "
        f"VALID={valid}"
    )
    return {
        "rho_binary": rho_binary,
        "problem": binary_problem,
        "coefficient": binary_coefficient,
        "transmission": binary_transmission,
        "density": binary_density,
        "topology": binary_topology,
        "volume_ok": volume_ok,
        "thickness_ok": thickness_ok,
        "residual_ok": residual_ok,
        "feature_ok": feature_ok,
        "valid": valid,
    }


for case in binary_cases:
    case["binary"] = validate_binary_case(case)
"""),
("markdown", r"""
## 5. 二值结果汇总

不降低相位、体积或厚度标准。任何不可行结果都保留为失败，不把 latent 或
连续材料解报告为成功器件。
"""),
("code", r"""
binary_rows = []
for case in binary_cases:
    binary = case["binary"]
    binary_rows.append(
        (
            case["volume_max"],
            binary["transmission"].transmittance,
            binary["transmission"].amplitude,
            binary["transmission"].phase_degrees,
            binary["transmission"].phase_error_degrees,
            case["max_constraint_violation"],
            binary["topology"].metal_fraction,
            binary["density"].gray_fraction,
            binary["topology"].minimum_metal_feature,
            binary["topology"].metal_components,
            binary["valid"],
        )
    )

print(
    "f_max  T          |t|        phase(deg)  phase_err  violation  "
    "Au     gray    min_feature  components  PASS"
)
for row in binary_rows:
    print(
        f"{row[0]:.2f}   {row[1]:.6f}   {row[2]:.6f}   "
        f"{row[3]:+9.4f}   {row[4]:.4f}     {row[5]:.2e}   "
        f"{row[6]:.4f} {row[7]:.1f}   {row[8]:>11d}  "
        f"{row[9]:>10d}  {row[10]}"
    )

if not any(row[-1] for row in binary_rows):
    warnings.warn(
        "No strict-binary gold design passed the requested validation.",
        RuntimeWarning,
    )
"""),
("code", r"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

ax = axes[0]
ax.plot(
    [row[0] for row in binary_rows],
    [row[1] for row in binary_rows],
    marker="o",
)
ax.set_xlabel("maximum gold volume fraction")
ax.set_ylabel("transmittance T")
ax.set_title("strict binary result")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(
    [row[0] for row in binary_rows],
    [row[5] for row in binary_rows],
    marker="o",
)
ax.axhline(PHASE_RESIDUAL_TOLERANCE, color="k", linestyle="--")
ax.set_yscale("log")
ax.set_xlabel("maximum gold volume fraction")
ax.set_ylabel("max constraint violation")
ax.set_title("final binary constraint residual")
ax.grid(alpha=0.3)

selected_case = binary_cases[-1]
ax = axes[2]
image = ax.imshow(
    selected_case["rho_material"].T,
    origin="lower",
    cmap="magma",
    vmin=0.0,
    vmax=1.0,
)
ax.set_title(f"binary material, f_max={selected_case['volume_max']:.2f}")
ax.set_xlabel("x cell")
ax.set_ylabel("y cell")
fig.colorbar(image, ax=ax, label="material density")

ax = axes[3]
history = selected_case["optimization"].history
ax.plot(
    [record.step for record in history],
    [record.objective_value for record in history],
)
ax.set_xlabel("STE evaluation")
ax.set_ylabel("binary objective value")
ax.set_title("strict binary objective history")
ax.grid(alpha=0.3)

fig.tight_layout()
"""),
("code", r"""
output_path = repo_root / os.environ.get(
    "FDTDZ_GOLD_OUTPUT",
    "inverse_design_2d_gold_binary_800nm.npz",
)
np.savez(
    output_path,
    volume_limits=np.asarray(VOLUME_LIMITS),
    selected_sparse_seeds=np.asarray([
        case["seed"] for case in binary_cases
    ]),
    binary_t=np.asarray([
        case["binary"]["coefficient"] for case in binary_cases
    ]),
    material_density=np.stack([
        case["rho_material"] for case in binary_cases
    ]),
    validated_binary_density=np.stack([
        case["binary"]["rho_binary"] for case in binary_cases
    ]),
    final_latent=np.stack([
        case["latent"].reshape(-1) for case in binary_cases
    ]),
    binary_valid=np.asarray([row[-1] for row in binary_rows]),
    objective_history=np.asarray([
        [record.objective_value for record in case["optimization"].history]
        for case in binary_cases
    ]),
    constraint_violation_history=np.asarray([
        [
            record.max_constraint_violation
            for record in case["optimization"].history
        ]
        for case in binary_cases
    ]),
    material_history=np.asarray([
        [
            record.binary_densities[0]
            for record in case["optimization"].history
        ]
        for case in binary_cases
    ]),
)
print("Saved:", output_path)

if all(row[-1] for row in binary_rows):
    print("All strict-binary designs passed independent validation.")
else:
    print("At least one strict-binary design failed independent validation.")
"""),
("markdown", r"""
## 6. 配置与复现

默认扫描 `(0.2, 0.3, 0.4)` 三个金体积上限。以下环境变量可缩短 smoke test：

- `FDTDZ_GOLD_VOLUME_LIMITS`
- `FDTDZ_GOLD_SEEDS`
- `FDTDZ_GOLD_BINARY_STEPS`
- `FDTDZ_GOLD_BINARY_LR`
- `FDTDZ_GOLD_BINARY_PENALTY`
- `FDTDZ_GOLD_BINARY_MARGIN`
- `FDTDZ_GOLD_LOCAL_SEARCH_ITERATIONS`
- `FDTDZ_GOLD_LOCAL_SEARCH_CANDIDATES`
- `FDTDZ_GOLD_OUTPUT`

局部搜索默认关闭，因为每个候选都需要一次完整 FDTD/ADE 仿真。
"""),
)


def _source(text):
    return textwrap.dedent(text).lstrip("\n")


def main():
    existing = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook = nbformat.v4.new_notebook(metadata=existing.get("metadata", {}))
    for cell_type, text in CELLS:
        source = _source(text)
        if cell_type == "code":
            notebook.cells.append(nbformat.v4.new_code_cell(source))
        else:
            notebook.cells.append(nbformat.v4.new_markdown_cell(source))
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    print("Created:", NOTEBOOK)


if __name__ == "__main__":
    main()
