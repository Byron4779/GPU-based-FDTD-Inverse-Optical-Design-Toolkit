"""Constrained 180-degree phase inverse design in a 2D TMz unit cell.

The complex transmission coefficient is normalized by an empty-cell reference
recorded at the same transmission monitor. The optimizer maximizes |t|^2 while
requiring the transmission phase to be 180 degrees modulo one full period.

Run from the repository root::

    python examples/inverse_design_2d_phase_constraint.py
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
    SingleFrequencyTransmission,
    minimize_slsqp,
    phase_180_constraints,
)
from fdtdz_jax.materials import Lossless  # noqa: E402
from fdtdz_jax.metasurface_2d import (  # noqa: E402
    gaussian_sine_pulse,
    huygens_current_waveforms,
)
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d  # noqa: E402
from fdtdz_jax.yee_ade_2d import prepare_y_cpml  # noqa: E402


def _phase_degrees(transmission):
    return float(np.degrees(np.angle(transmission)))


def build_problem(
    design_size=(32, 8),
    x_start=0,
    eps_max=9.0,
    target_frequency=0.3,
    num_steps=240,
    pulse_sigma=10.0,
    pulse_center_time=20.0,
):
    nx, ny = 32, 64
    dt, dx, dy = 0.5, 1.0, 1.0
    eps_min = 1.0
    cpml_width = 8

    cpml = prepare_y_cpml(
        (nx, ny), cpml_width, dt, dy, dtype=np.float64
    )
    grid = prepare_material_grid_2d(
        [Lossless(eps_min)],
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
        frequency=target_frequency,
        dtype=np.float64,
    )
    electric, magnetic = huygens_current_waveforms(
        waveform, direction=1, wave_impedance=1.0, dtype=np.float64
    )

    params = DesignParameters(
        grid_size=design_size,
        medium1=eps_min,
        medium2=eps_max,
        weights=np.full(design_size[0] * design_size[1], 0.5),
        beta=2.0,
        p=3.0,
        filter_type="conic",
        radius=1.5,
        periodic_axes=(
            (0,) if x_start == 0 and design_size[0] == nx else None
        ),
    )
    region = DesignRegion(
        params,
        x0=x_start,
        x1=x_start + design_size[0],
        y0=28,
        y1=28 + design_size[1],
    )
    transmission = SingleFrequencyTransmission(
        frequency=target_frequency,
        dt=dt,
        window="hann",
        reference_floor=1e-10,
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
        objective_arguments=[transmission],
        objective_functions=[
            lambda coefficient: -jnp.abs(coefficient) ** 2
        ],
        constraints=phase_180_constraints(
            amplitude_floor=1e-4,
            power_limit=1.0,
        ),
    )
    return problem, target_frequency, design_size


def find_phase_seed(
    problem,
    design_size,
    samples=256,
    seed=11,
    amplitude_floor=1e-3,
):
    """Find a deterministic random design near the 180-degree phase basin."""

    rng = np.random.default_rng(seed)
    size = design_size[0] * design_size[1]
    best = None
    for _ in range(int(samples)):
        candidate = rng.random(size)
        transmission = complex(problem.evaluate_quantities(candidate)[0])
        amplitude = abs(transmission)
        if amplitude < amplitude_floor:
            continue
        residual = abs(1.0 + transmission.real / amplitude)
        if best is None or residual < best[0]:
            best = (residual, candidate, transmission)
    if best is None:
        raise RuntimeError(
            "random seed search did not find a transmission above the "
            "amplitude floor"
        )
    return best[1], best[2]


def main() -> None:
    problem, target_frequency, grid_size = build_problem()
    phase_seed, phase_seed_transmission = find_phase_seed(
        problem, grid_size
    )

    initial_t = phase_seed_transmission
    initial_objective, _, initial_eq, _, initial_ineq, _ = (
        problem.get_constrained_fdfuncs()(phase_seed)
    )
    result = minimize_slsqp(
        problem,
        phase_seed,
        maxiter=400,
        bounds=(0.0, 1.0),
        tol=1e-4,
        ftol=1e-6,
    )
    final_t = complex(problem.evaluate_quantities()[0])

    density = problem.design_regions[0].design_parameters.weights.reshape(
        grid_size
    )
    np.savez(
        "inverse_design_2d_phase_constraint.npz",
        density=density,
        phase_seed_density=phase_seed.reshape(grid_size),
        initial_transmission=np.asarray(initial_t),
        final_transmission=np.asarray(final_t),
        design_history=result.design_history,
        objective_history=result.objective_history,
        eq_constraints=result.constraint_values["eq"],
        ineq_constraints=result.constraint_values["ineq"],
    )

    print(
        "initial: "
        f"t={initial_t:.6g}, |t|^2={abs(initial_t) ** 2:.6f}, "
        f"phase={_phase_degrees(initial_t):.3f} deg, "
        f"objective={float(initial_objective):.6f}"
    )
    print(
        "final:   "
        f"t={final_t:.6g}, |t|^2={abs(final_t) ** 2:.6f}, "
        f"phase={_phase_degrees(final_t):.3f} deg"
    )
    print(
        f"constraints: eq={np.asarray(result.constraint_values['eq'])}, "
        f"ineq={np.asarray(result.constraint_values['ineq'])}, "
        f"max_violation={result.max_constraint_violation:.3e}"
    )
    print(
        f"SLSQP iterations: {result.nit}, "
        f"converged: {result.success}, message: {result.message}"
    )
    print(
        "initial inequality values: "
        f"{np.asarray(initial_ineq)}; initial equality: "
        f"{np.asarray(initial_eq)}"
    )
    print("results saved to inverse_design_2d_phase_constraint.npz")


if __name__ == "__main__":
    main()
