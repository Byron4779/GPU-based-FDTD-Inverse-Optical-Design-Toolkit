import json
from pathlib import Path
import runpy

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdz_jax.inverse_design import DesignParameters
from fdtdz_jax.inverse_design import DesignGeometryLimits
from fdtdz_jax.inverse_design import DesignRegion
from fdtdz_jax.inverse_design import InverseDesignProblem
from fdtdz_jax.inverse_design import SingleFrequencyTransmission
from fdtdz_jax.inverse_design import binary_topology_metrics
from fdtdz_jax.inverse_design import density_metrics
from fdtdz_jax.inverse_design import effective_thickness_constraints
from fdtdz_jax.inverse_design import metal_volume_fraction_constraints
from fdtdz_jax.inverse_design import minimize_slsqp
from fdtdz_jax.inverse_design import minimize_slsqp_beta_continuation
from fdtdz_jax.inverse_design import phase_180_constraints
from fdtdz_jax.inverse_design import single_frequency_amplitude
from fdtdz_jax.inverse_design import sparse_random_design
from fdtdz_jax.inverse_design import transmission_metrics
from fdtdz_jax.materials import Lossless
from fdtdz_jax.metasurface_2d import gaussian_sine_pulse
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d
from fdtdz_jax.yee_ade_2d import prepare_y_cpml


def _build_reference_problem(design_constraints=(), geometry_limits=None):
    nx, ny = 8, 20
    dt, dx, dy = 0.2, 1.0, 1.0
    cpml = prepare_y_cpml((nx, ny), 4, dt, dy, dtype=np.float64)
    grid = prepare_material_grid_2d(
        [Lossless(1.0)],
        np.zeros((nx, ny), dtype=int),
        dt,
        dx,
        dy,
        dtype=np.float64,
        y_cpml=cpml,
    )
    num_steps = 80
    electric = gaussian_sine_pulse(
        num_steps,
        dt,
        center_time=10.0,
        width=6.0,
        frequency=0.3,
        dtype=np.float64,
    )
    magnetic = np.zeros(num_steps, dtype=np.float64)
    design_size = (4, 2)
    params = DesignParameters(
        grid_size=design_size,
        medium1=1.0,
        medium2=4.0,
        weights=np.full(design_size, 0.5).reshape(-1),
        beta=0.0,
        p=3.0,
    )
    region = DesignRegion(params, x0=2, x1=6, y0=8, y1=10)
    response = SingleFrequencyTransmission(
        frequency=0.3,
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
        source_y=4,
        reflection_probe_y=6,
        transmission_probe_y=14,
        objective_arguments=[response],
        objective_functions=[lambda transmission: -jnp.abs(transmission) ** 2],
        constraints=phase_180_constraints(amplitude_floor=1e-4),
        design_constraints=design_constraints,
        geometry_limits=geometry_limits,
    )
    return problem


def test_single_frequency_amplitude_matches_numpy_fft():
    num_steps, dt = 64, 0.1
    frequency = 1.0 / (num_steps * dt)
    trace = np.sin(2.0 * np.pi * frequency * np.arange(num_steps) * dt)
    window = np.hanning(num_steps)
    expected = dt * np.fft.rfft(trace * window)[1]

    actual = single_frequency_amplitude(
        jnp.asarray(trace), frequency, dt, window=window
    )

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12)


def test_single_frequency_transmission_normalizes_device_by_reference():
    trace = np.linspace(-1.0, 1.0, 32)
    response = SingleFrequencyTransmission(
        frequency=0.25,
        dt=0.125,
        window=None,
        reference_floor=1e-12,
    )
    actual = response(
        {
            "transmission": 0.5 * trace,
            "reference_transmission": trace,
        }
    )
    np.testing.assert_allclose(np.asarray(actual), 0.5 + 0.0j, atol=1e-12)


def test_phase_180_constraints_reject_wrong_phase_zero_field_and_gain():
    constraints = phase_180_constraints(
        amplitude_floor=1e-3, power_limit=1.0
    )

    def values(transmission):
        return np.asarray(
            [constraint.function(transmission) for constraint in constraints]
        )

    np.testing.assert_allclose(
        values(-0.9 + 0.0j),
        [0.0, 0.899, 0.19],
        atol=1e-11,
    )

    wrong_phase = values(0.9 + 0.0j)
    assert wrong_phase[0] == pytest.approx(2.0)
    assert wrong_phase[1] > 0.0

    zero_field = values(0.0 + 0.0j)
    assert zero_field[0] == pytest.approx(1.0)
    assert zero_field[1] < 0.0

    imaginary_residual = values(-0.9 + 0.01j)
    assert imaginary_residual[0] == pytest.approx(6.17e-5, rel=2e-2)

    too_much_power = values(-1.01 + 0.0j)
    assert too_much_power[2] < 0.0


def test_slsqp_preserves_phase_constraint_while_maximizing_reward():
    class ToyProblem:
        def __init__(self):
            self.design = None

        def get_constrained_fdfuncs(self):
            def evaluate(x):
                rho = jnp.asarray(x)[0]
                transmission = -rho - 1j * (rho - 1.0)
                objective = -jnp.abs(transmission) ** 2
                objective_gradient = jnp.asarray(
                    [-2.0 * rho - 2.0 * (rho - 1.0)],
                    dtype=jnp.float64,
                )
                equality = jnp.asarray([rho - 1.0], dtype=jnp.float64)
                equality_jacobian = jnp.ones((1, 1), dtype=jnp.float64)
                inequality = jnp.asarray(
                    [rho - 0.9], dtype=jnp.float64
                )
                inequality_jacobian = jnp.ones(
                    (1, 1), dtype=jnp.float64
                )
                return (
                    objective,
                    objective_gradient,
                    equality,
                    equality_jacobian,
                    inequality,
                    inequality_jacobian,
                )

            return evaluate

        def update_design(self, design):
            self.design = np.asarray(design)

    problem = ToyProblem()
    result = minimize_slsqp(
        problem,
        np.asarray([0.5]),
        bounds=(0.0, 2.0),
        maxiter=50,
        tol=1e-10,
        ftol=1e-10,
    )

    assert result.success
    np.testing.assert_allclose(result.x, [1.0], atol=1e-5)
    np.testing.assert_allclose(problem.design, [1.0], atol=1e-5)
    assert result.max_constraint_violation <= 1e-6


def test_problem_precomputes_reference_and_exposes_constraint_jacobians():
    problem = _build_reference_problem()

    assert problem.reference_transmission is not None
    assert np.asarray(problem.reference_transmission).shape == (
        problem.electric_waveform.shape[0],
    )

    rho = np.asarray(problem._rho_vector())
    (
        objective,
        objective_gradient,
        equality,
        equality_jacobian,
        inequality,
        inequality_jacobian,
    ) = problem.get_constrained_fdfuncs()(rho)

    assert np.isfinite(np.asarray(objective))
    assert np.asarray(objective_gradient).shape == rho.shape
    assert np.asarray(equality).shape == (1,)
    assert np.asarray(equality_jacobian).shape == (1, rho.size)
    assert np.asarray(inequality).shape == (2,)
    assert np.asarray(inequality_jacobian).shape == (2, rho.size)
    assert np.all(np.isfinite(np.asarray(objective_gradient)))
    assert np.all(np.isfinite(np.asarray(equality_jacobian)))
    assert np.all(np.isfinite(np.asarray(inequality_jacobian)))


def test_projected_density_constraints_are_appended_to_jacobian():
    constraints = (
        *metal_volume_fraction_constraints(minimum=0.1, maximum=0.6),
        *effective_thickness_constraints(minimum=0.2, maximum=2.0),
    )
    problem = _build_reference_problem(design_constraints=constraints)
    rho = np.asarray(problem._rho_vector())
    (
        _,
        _,
        equality,
        equality_jacobian,
        inequality,
        inequality_jacobian,
    ) = problem.get_constrained_fdfuncs()(rho)

    assert np.asarray(equality).shape == (1,)
    assert np.asarray(equality_jacobian).shape == (1, rho.size)
    assert np.asarray(inequality).shape == (2 + 4,)
    assert np.asarray(inequality_jacobian).shape == (6, rho.size)
    assert problem.constraint_names("ineq")[-4:] == (
        "metal_volume_fraction_min[region=0]",
        "metal_volume_fraction_max[region=0]",
        "effective_thickness_min[region=0]",
        "effective_thickness_max[region=0]",
    )
    projected = problem.evaluate_projected_densities(rho)[0]
    np.testing.assert_allclose(inequality[2], np.mean(projected) - 0.1)
    np.testing.assert_allclose(inequality[5], 2.0 - np.sum(np.mean(projected, axis=0)))


def test_geometry_limits_reject_changed_design_size_and_clearance():
    with pytest.raises(ValueError, match="width"):
        _build_reference_problem(
            geometry_limits=DesignGeometryLimits(min_width=5, max_width=5)
        )
    with pytest.raises(ValueError, match="transmission probe"):
        _build_reference_problem(
            geometry_limits=DesignGeometryLimits(
                transmission_probe_clearance=5
            )
        )


def test_sparse_seed_and_validation_metrics():
    design = sparse_random_design((32, 8), target_fraction=0.3, seed=7)
    assert design.shape == (32, 8)
    assert 0.15 < np.mean(design) < 0.45
    density = density_metrics(design)
    assert density.metal_fraction == pytest.approx(np.mean(design))
    assert density.effective_thickness == pytest.approx(
        np.sum(np.mean(design, axis=0))
    )

    transmission = transmission_metrics(-0.4 + 0.0j)
    assert transmission.phase_error_degrees == pytest.approx(0.0)
    assert transmission.phase_residual == pytest.approx(0.0)
    assert transmission.transmittance == pytest.approx(0.16)

    metal_block = np.zeros((6, 4))
    metal_block[1:3, 1:3] = 1.0
    topology = binary_topology_metrics(metal_block)
    assert topology.metal_components == 1
    assert topology.minimum_metal_feature >= 1


def test_beta_continuation_retains_stage_history():
    class ContinuationToyProblem:
        def __init__(self):
            self.beta = 0.0
            self.design = np.asarray([0.25])

        def update_beta(self, beta):
            self.beta = beta

        def get_constrained_fdfuncs(self):
            def evaluate(x):
                rho = jnp.asarray(x)
                objective = rho[0] ** 2
                return (
                    objective,
                    jnp.asarray([2.0 * rho[0]]),
                    jnp.zeros((0,)),
                    jnp.zeros((0, 1)),
                    jnp.asarray([rho[0] - 0.5]),
                    jnp.ones((1, 1)),
                )

            return evaluate

        def update_design(self, design):
            self.design = np.asarray(design)

        def evaluate_projected_densities(self, rho):
            return (np.asarray(rho).reshape(1, 1),)

        def constraint_names(self, kind):
            return ("volume[region=0]",) if kind == "ineq" else ()

    problem = ContinuationToyProblem()
    continuation = minimize_slsqp_beta_continuation(
        problem,
        problem.design,
        betas=(2, 4, 8),
        maxiter=30,
        tol=1e-9,
        ftol=1e-9,
    )

    assert [stage.beta for stage in continuation.stages] == [2.0, 4.0, 8.0]
    assert continuation.final_optimization_result.success
    assert continuation.final_stage.metal_volume_fraction == pytest.approx(0.5)


def test_constrained_jacobian_matches_finite_difference():
    problem = _build_reference_problem()
    rho = np.asarray(problem._rho_vector())
    _, objective_gradient, _, equality_jacobian, _, inequality_jacobian = (
        problem.get_constrained_fdfuncs()(rho)
    )
    objective_gradient = np.asarray(objective_gradient)
    equality_jacobian = np.asarray(equality_jacobian)
    inequality_jacobian = np.asarray(inequality_jacobian)

    db = 1e-5
    indices = (0, rho.size - 1)
    for index in indices:
        rho_plus = rho.copy()
        rho_minus = rho.copy()
        rho_plus[index] += db
        rho_minus[index] -= db
        transmission_plus = problem.evaluate_quantities(rho_plus)[0]
        transmission_minus = problem.evaluate_quantities(rho_minus)[0]
        objective_plus = -abs(transmission_plus) ** 2
        objective_minus = -abs(transmission_minus) ** 2
        equality_plus = (
            1.0 + np.real(transmission_plus) / abs(transmission_plus)
        )
        equality_minus = (
            1.0 + np.real(transmission_minus) / abs(transmission_minus)
        )
        inequality_plus = abs(transmission_plus) - 1e-4
        inequality_minus = abs(transmission_minus) - 1e-4

        np.testing.assert_allclose(
            objective_gradient[index],
            (objective_plus - objective_minus) / (2.0 * db),
            rtol=2e-3,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            equality_jacobian[0, index],
            (equality_plus - equality_minus) / (2.0 * db),
            rtol=2e-3,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            inequality_jacobian[0, index],
            (inequality_plus - inequality_minus) / (2.0 * db),
            rtol=2e-3,
            atol=2e-5,
        )


@pytest.mark.slow
def test_full_period_example_improves_transmission_at_180_degrees():
    example = Path(__file__).resolve().parents[1] / (
        "examples/inverse_design_2d_phase_constraint.py"
    )
    namespace = runpy.run_path(str(example))
    problem, _, design_size = namespace["build_problem"]()
    phase_seed, initial_transmission = namespace["find_phase_seed"](
        problem, design_size
    )
    initial_objective = -abs(initial_transmission) ** 2

    result = minimize_slsqp(
        problem,
        phase_seed,
        maxiter=40,
        bounds=(0.0, 1.0),
        tol=1e-4,
        ftol=1e-6,
    )
    final_transmission = complex(problem.evaluate_quantities()[0])
    phase_residual = 1.0 + final_transmission.real / abs(
        final_transmission
    )

    assert abs(phase_residual) < 1e-4
    assert result.max_constraint_violation < 1e-4
    assert -abs(final_transmission) ** 2 < initial_objective


def test_800nm_notebook_uses_constrained_phase_objective():
    notebook_path = Path(__file__).resolve().parents[1] / (
        "inverse_design_2d_gold_hdf5.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert "target_wavelength_m = 800e-9" in source
    assert 'read_material(gold_material_file, "gold")' in source
    assert "SingleFrequencyTransmission" in source
    assert "phase_180_constraints" in source
    assert "minimize_binary_ste" in source
    assert "binary_material=True" in source
    assert "binary_flip_local_search" in source
    assert "PHASE_TOLERANCE_DEG" in source
    assert "metal_volume_fraction_constraints" in source
    assert "effective_thickness_constraints" in source
    assert "binary_topology_metrics" in source
    assert "periodic_axes=(0,)" in source
