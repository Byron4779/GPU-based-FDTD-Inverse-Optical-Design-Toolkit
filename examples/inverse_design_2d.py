"""2D TMz inverse-design example (Meep-aligned ``fdtdz_jax.inverse_design``).

Maximizes the time-integrated transmission-probe energy through a lossless
design slab using scipy L-BFGS-B with JAX-computed gradients.

Run from the repository root::

    python examples/inverse_design_2d.py
"""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdtdz_jax import (  # noqa: E402
    DesignParameters,
    DesignRegion,
    InverseDesignProblem,
    ProbeTrace,
    minimize_lbfgsb,
)
from fdtdz_jax.materials import Lossless  # noqa: E402
from fdtdz_jax.metasurface_2d import (  # noqa: E402
    gaussian_sine_pulse,
    huygens_current_waveforms,
)
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d  # noqa: E402
from fdtdz_jax.yee_ade_2d import prepare_y_cpml  # noqa: E402


def main() -> None:
    nx, ny = 32, 64
    dt, dx, dy = 0.5, 1.0, 1.0
    eps_min, eps_max = 1.0, 9.0
    cpml_width = 8

    cpml = prepare_y_cpml((nx, ny), cpml_width, dt, dy, dtype=np.float64)
    grid = prepare_material_grid_2d(
        [Lossless(eps_min)],
        np.zeros((nx, ny), dtype=int),
        dt,
        dx,
        dy,
        dtype=np.float64,
        y_cpml=cpml,
    )

    num_steps = 240
    waveform = gaussian_sine_pulse(
        num_steps, dt, center_time=20.0, width=10.0, frequency=0.3,
        dtype=np.float64,
    )
    electric, magnetic = huygens_current_waveforms(
        waveform, direction=1, wave_impedance=1.0, dtype=np.float64
    )

    grid_size = (12, 8)
    params = DesignParameters(
        grid_size=grid_size,
        medium1=eps_min,
        medium2=eps_max,
        weights=np.full(grid_size[0] * grid_size[1], 0.5),
        beta=2.0,
        p=3.0,
        filter_type="conic",
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

    density = problem.design_regions[0].design_parameters.weights.reshape(
        grid_size
    )
    np.save("inverse_design_2d_density.npy", density)

    print(f"initial objective (negative transmission energy): {f0:.6f}")
    print(f"final   objective (negative transmission energy): {f1:.6f}")
    print(f"L-BFGS-B iterations: {result.nit}, converged: {result.success}")
    print("optimized density saved to inverse_design_2d_density.npy")


if __name__ == "__main__":
    main()
