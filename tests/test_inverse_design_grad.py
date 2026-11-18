import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from fdtdz_jax.inverse_design import DesignParameters
from fdtdz_jax.inverse_design import DesignRegion
from fdtdz_jax.inverse_design import InverseDesignProblem
from fdtdz_jax.inverse_design import ProbeTrace
from fdtdz_jax.materials import Debye
from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import LorentzPole
from fdtdz_jax.materials import Lossless
from fdtdz_jax.materials import Multipole
from fdtdz_jax.metasurface_2d import gaussian_sine_pulse
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d
from fdtdz_jax.yee_ade_2d import prepare_y_cpml


def _build_problem(
    grid_size=(8, 4),
    beta=0.0,
    filter_type=None,
    binary_material=False,
):
    nx, ny = 16, 32
    eps_min, eps_max = 1.0, 4.0
    dt, dx, dy = 0.2, 1.0, 1.0
    cpml = prepare_y_cpml((nx, ny), 6, dt, dy, dtype=np.float64)
    grid = prepare_material_grid_2d(
        [Lossless(eps_min)],
        np.zeros((nx, ny), dtype=int),
        dt,
        dx,
        dy,
        dtype=np.float64,
        y_cpml=cpml,
    )

    num_steps = 180
    electric_waveform = gaussian_sine_pulse(
        num_steps, dt, center_time=15.0, width=8.0, frequency=0.3,
        dtype=np.float64,
    )
    magnetic_waveform = np.zeros(num_steps, dtype=np.float64)

    params = DesignParameters(
        grid_size=grid_size,
        medium1=eps_min,
        medium2=eps_max,
        weights=np.full(grid_size[0] * grid_size[1], 0.5),
        beta=beta,
        p=3.0,
        filter_type=filter_type,
        radius=1.0,
        binary_material=binary_material,
    )
    region = DesignRegion(params, x0=4, x1=4 + grid_size[0], y0=14, y1=18)
    problem = InverseDesignProblem(
        grid,
        region,
        electric_waveform,
        magnetic_waveform,
        np.ones(nx, dtype=np.float64),
        source_y=8,
        reflection_probe_y=10,
        transmission_probe_y=22,
        objective_arguments=[ProbeTrace(probe="transmission")],
        objective_functions=[lambda t: -jnp.mean(jnp.square(t))],
    )
    return problem


def _check_gradient(problem, atol=1e-5, rtol=1e-4, db=1e-5):
    rho0 = problem._rho_vector()
    analytic = np.asarray(problem.get_fdf_funcs()[1](rho0)).reshape(-1)
    fd, idx = problem.calculate_fd_gradient(
        num_gradients=4, db=db, seed=7
    )
    assert np.max(np.abs(analytic[idx])) > 1e-8
    for k, fd_value in zip(idx, fd):
        np.testing.assert_allclose(
            analytic[k], fd_value, atol=atol, rtol=rtol
        )


def test_lossless_end_to_end_gradient_simp_only():
    problem = _build_problem(beta=0.0, filter_type=None)
    _check_gradient(problem)


def test_lossless_end_to_end_gradient_filter_and_projection():
    problem = _build_problem(beta=4.0, filter_type="conic")
    _check_gradient(problem, atol=1e-4, rtol=1e-4)


def test_debye_end_to_end_gradient():
    nx, ny = 16, 32
    dt, dx, dy = 0.2, 1.0, 1.0
    background = Lossless(1.0)
    design = Debye(1.0, 3.0, 0.4)
    cpml = prepare_y_cpml((nx, ny), 6, dt, dy, dtype=np.float64)
    grid = prepare_material_grid_2d(
        [background],
        np.zeros((nx, ny), dtype=int),
        dt,
        dx,
        dy,
        dtype=np.float64,
        y_cpml=cpml,
    )
    num_steps = 180
    electric = gaussian_sine_pulse(
        num_steps, dt, center_time=15.0, width=8.0, frequency=0.3,
        dtype=np.float64,
    )
    magnetic = np.zeros(num_steps, dtype=np.float64)
    grid_size = (8, 4)
    params = DesignParameters(
        grid_size=grid_size,
        medium1=background,
        medium2=design,
        weights=np.full(grid_size[0] * grid_size[1], 0.5),
        beta=2.0,
        p=3.0,
        filter_type="conic",
        radius=1.0,
    )
    region = DesignRegion(
        params, x0=4, x1=4 + grid_size[0], y0=14, y1=18
    )
    problem = InverseDesignProblem(
        grid,
        region,
        electric,
        magnetic,
        np.ones(nx, dtype=np.float64),
        source_y=8,
        reflection_probe_y=10,
        transmission_probe_y=22,
        objective_arguments=[ProbeTrace(probe="transmission")],
        objective_functions=[lambda t: -jnp.mean(jnp.square(t))],
    )
    _check_gradient(problem, atol=1e-4, rtol=1e-4)


def test_multipole_end_to_end_gradient():
    nx, ny = 16, 32
    dt, dx, dy = 0.2, 1.0, 1.0
    background = Lossless(1.0)
    design = Multipole(
        1.0, (LorentzPole(1.2, 3.0, 0.2), DrudePole(2.0, 0.4))
    )
    cpml = prepare_y_cpml((nx, ny), 6, dt, dy, dtype=np.float64)
    grid = prepare_material_grid_2d(
        [background, design],
        np.zeros((nx, ny), dtype=int),
        dt,
        dx,
        dy,
        dtype=np.float64,
        y_cpml=cpml,
    )
    num_steps = 180
    electric = gaussian_sine_pulse(
        num_steps, dt, center_time=15.0, width=8.0, frequency=0.3,
        dtype=np.float64,
    )
    magnetic = np.zeros(num_steps, dtype=np.float64)
    grid_size = (8, 4)
    params = DesignParameters(
        grid_size=grid_size,
        medium1=background,
        medium2=design,
        weights=np.full(grid_size[0] * grid_size[1], 0.5),
        beta=2.0,
        p=3.0,
        filter_type="conic",
        radius=1.0,
    )
    region = DesignRegion(
        params, x0=4, x1=4 + grid_size[0], y0=14, y1=18
    )
    problem = InverseDesignProblem(
        grid,
        region,
        electric,
        magnetic,
        np.ones(nx, dtype=np.float64),
        source_y=8,
        reflection_probe_y=10,
        transmission_probe_y=22,
        objective_arguments=[ProbeTrace(probe="transmission")],
        objective_functions=[lambda t: -jnp.mean(jnp.square(t))],
    )
    _check_gradient(problem, atol=1e-4, rtol=1e-4)

    problem.design_regions[0].design_parameters.binary_material = True
    problem._rebuild()
    binary_rho = np.linspace(
        0.25,
        0.75,
        grid_size[0] * grid_size[1],
        dtype=np.float64,
    )
    projected = problem.evaluate_projected_densities(binary_rho)[0]
    np.testing.assert_array_equal(
        np.unique(projected), np.asarray([0.0, 1.0])
    )
    value, gradient = problem(
        binary_rho, need_value=True, need_gradient=True
    )
    assert np.isfinite(value)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.linalg.norm(np.asarray(gradient)) > 0.0


def test_call_returns_value_and_gradient():
    problem = _build_problem(beta=2.0)
    value, gradient = problem(need_value=True, need_gradient=True)
    assert np.isfinite(value)
    assert np.asarray(gradient).shape == (8 * 4,)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_binary_lossless_end_to_end_surrogate_gradient():
    problem = _build_problem(
        beta=8.0,
        filter_type="conic",
        binary_material=True,
    )
    rho = np.linspace(
        0.25,
        0.75,
        problem.num_design_params[0],
        dtype=np.float64,
    )
    projected = problem.evaluate_projected_densities(rho)[0]
    np.testing.assert_array_equal(
        np.unique(projected), np.asarray([0.0, 1.0])
    )
    value, gradient = problem(
        rho, need_value=True, need_gradient=True
    )
    assert np.isfinite(value)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.linalg.norm(np.asarray(gradient)) > 0.0
