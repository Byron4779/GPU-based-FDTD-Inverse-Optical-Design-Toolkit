import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdz_jax.inverse_design import DesignParameters
from fdtdz_jax.inverse_design import DesignRegion
from fdtdz_jax.inverse_design import InverseDesignProblem
from fdtdz_jax.inverse_design import ProbeTrace
from fdtdz_jax.inverse_design import binary_flip_local_search
from fdtdz_jax.inverse_design import binary_straight_through
from fdtdz_jax.inverse_design import minimize_binary_ste
from fdtdz_jax.inverse_design import minimize_lbfgsb
from fdtdz_jax.materials import Lossless
from fdtdz_jax.metasurface_2d import gaussian_sine_pulse
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d
from fdtdz_jax.yee_ade_2d import prepare_y_cpml


class _BinaryToyProblem:
    def __init__(self, target):
        self.target = np.asarray(target, dtype=np.float64)
        params = DesignParameters(
            grid_size=(self.target.size, 1),
            medium1=1.0,
            medium2=2.0,
            weights=np.full(self.target.size, 0.5),
            binary_material=True,
        )
        self.design_regions = [
            DesignRegion(params, x0=0, x1=self.target.size, y0=0, y1=1)
        ]
        self.num_design_params = [self.target.size]
        self.design = params.weights.copy()

    def get_constrained_fdfuncs(self):
        target = jnp.asarray(self.target)

        def evaluate(latent):
            material = binary_straight_through(latent)
            objective, objective_gradient = jax.value_and_grad(
                lambda value: jnp.sum(
                    (binary_straight_through(value) - target) ** 2
                )
            )(latent)
            return (
                objective,
                objective_gradient,
                jnp.zeros((0,), dtype=jnp.float64),
                jnp.zeros((0, latent.size), dtype=jnp.float64),
                jnp.zeros((0,), dtype=jnp.float64),
                jnp.zeros((0, latent.size), dtype=jnp.float64),
            )

        return evaluate

    def evaluate_projected_densities(self, latent=None):
        if latent is None:
            latent = self.design
        material = np.asarray(
            binary_straight_through(jnp.asarray(latent, dtype=jnp.float64))
        )
        return (material.reshape(self.target.size, 1),)

    def update_design(self, latent):
        self.design = np.asarray(latent, dtype=np.float64).copy()


def test_binary_ste_optimizer_returns_exact_endpoint_material():
    target = np.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    problem = _BinaryToyProblem(target)
    result = minimize_binary_ste(
        problem,
        np.full(target.size, 0.5),
        steps=120,
        learning_rate=0.02,
        threshold_margin=0.01,
    )

    actual = result.binary_densities[0].reshape(-1)
    np.testing.assert_array_equal(actual, target)
    assert result.feasible
    assert result.objective_value == pytest.approx(0.0)
    assert result.history[0].step == 0
    assert all(
        np.all((record.binary_densities[0] == 0.0)
               | (record.binary_densities[0] == 1.0))
        for record in result.history
    )


def test_binary_flip_local_search_accepts_only_measured_improvements():
    target = np.asarray([1.0, 0.0, 1.0, 0.0])
    problem = _BinaryToyProblem(target)
    result = binary_flip_local_search(
        problem,
        np.zeros(target.size),
        iterations=6,
        candidate_count=target.size,
        seed=3,
    )

    np.testing.assert_array_equal(
        result.binary_densities[0].reshape(-1), target
    )
    assert result.improved
    assert result.objective_value == pytest.approx(0.0)
    assert len(result.accepted_flips) == 2


@pytest.mark.slow
def test_lbfgsb_improves_lossless_transmission():
    nx, ny = 32, 64
    dt, dx, dy = 0.5, 1.0, 1.0
    eps_min, eps_max = 1.0, 9.0
    cpml = prepare_y_cpml((nx, ny), 8, dt, dy, dtype=np.float64)
    grid = prepare_material_grid_2d(
        [Lossless(eps_min)],
        np.zeros((nx, ny), dtype=int),
        dt,
        dx,
        dy,
        dtype=np.float64,
        y_cpml=cpml,
    )
    num_steps = 200
    electric = gaussian_sine_pulse(
        num_steps, dt, center_time=20.0, width=10.0, frequency=0.3,
        dtype=np.float64,
    )
    magnetic = np.zeros(num_steps, dtype=np.float64)

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
        problem, np.full(grid_size[0] * grid_size[1], 0.5), maxiter=20
    )
    f1 = problem(need_value=True, need_gradient=False)[0]

    assert result.nit > 0
    assert f1 < f0
