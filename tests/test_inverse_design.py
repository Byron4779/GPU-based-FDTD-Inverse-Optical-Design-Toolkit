"""Regression tests for the differentiable inverse-design module."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from fdtdz_jax.inverse_design import (
    DesignResult,
    InverseDesign2D,
    adjoint_gradient,
    band_power,
    build_design_grid,
    field_overlap,
    interpolate_coefficients,
    level_set_density,
    make_objective,
    project_density,
    run_inverse_design,
    sigmoid_projection,
    time_integrated_power,
    zero_state_2d,
)
from fdtdz_jax.materials import Lossless
from fdtdz_jax.metasurface_2d import gaussian_sine_pulse
from fdtdz_jax.metasurface_2d import huygens_current_waveforms
from fdtdz_jax.yee_ade_2d import prepare_y_cpml


def _make_problem(nx=12, ny=36, num_steps=60, mask_slice=(14, 18),
                  design_eps=4.0, dtype=np.float32):
  """A transmission problem with every probe clear of the CPML region.

  ``source_y=8 < reflection_y=11 < mask < transmission_y=26`` all sit inside
  the interior ``[6, ny - 6)`` so the transmission probe sees a real signal.
  """
  dx = dy = 1.0
  dt = 0.5
  waveform = gaussian_sine_pulse(
      num_steps, dt, center_time=12.0, width=5.0, frequency=0.2,
      dtype=dtype)
  electric, magnetic = huygens_current_waveforms(
      waveform, direction=1, wave_impedance=1.0, dtype=dtype)
  cpml = prepare_y_cpml((nx, ny), width=6, dt=dt, dy=dy, dtype=dtype)
  mask = np.zeros((nx, ny), dtype=bool)
  mask[:, mask_slice[0]:mask_slice[1]] = True
  return InverseDesign2D.from_materials(
      (nx, ny), mask, Lossless(1.0), Lossless(design_eps), dt, dx, dy,
      electric, magnetic, np.ones(nx, dtype=dtype),
      source_y=8, reflection_y=11, transmission_y=26, cpml=cpml,
      dtype=dtype)


def test_interpolate_coefficients_endpoints():
  background = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
  design = np.array([0.25, 0.1, 0.0, 0.2, 0.0], dtype=np.float32)
  density = np.array([0.0, 1.0], dtype=np.float32)
  result = np.asarray(interpolate_coefficients(background, design, density))
  np.testing.assert_allclose(result[0], background, atol=1e-6)
  np.testing.assert_allclose(result[1], design, atol=1e-6)


def test_interpolate_coefficients_penalty_is_binary_bias():
  background = np.array([1.0, 0, 0, 0, 0], dtype=np.float32)
  design = np.array([0.0, 0, 0, 0, 0], dtype=np.float32)
  density = np.array([0.5], dtype=np.float32)
  a0_linear = np.asarray(
      interpolate_coefficients(background, design, density, penalty=1.0))[0, 0]
  a0_penalized = np.asarray(
      interpolate_coefficients(background, design, density, penalty=3.0))[0, 0]
  # Higher penalty pushes the intermediate density toward the background end.
  assert a0_penalized > a0_linear


def test_sigmoid_projection_range_and_endpoints():
  density = np.linspace(0.0, 1.0, 101, dtype=np.float32)
  projected = np.asarray(sigmoid_projection(density, beta=16.0))
  assert np.all(projected >= 0.0) and np.all(projected <= 1.0)
  assert projected[0] < 1e-3
  assert projected[-1] > 1.0 - 1e-3


def test_level_set_density_is_smooth_sigmoid():
  phi = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
  density = np.asarray(level_set_density(phi, width=1.0))
  assert density[1] == pytest.approx(0.5, abs=1e-6)
  assert density[0] < density[1] < density[2]


def test_time_integrated_power_and_band_power():
  dt = 0.5
  signal = np.sin(2 * np.pi * 0.25 * np.arange(64) * dt).astype(np.float32)
  total = float(time_integrated_power(jnp.asarray(signal), dt))
  assert total == pytest.approx(dt * float(np.sum(signal**2)), rel=1e-5)
  # A single-tone signal concentrates its band power near its frequency.
  in_band = float(band_power(jnp.asarray(signal), dt, 0.24, 0.26))
  out_band = float(band_power(jnp.asarray(signal), dt, 0.4, 0.5))
  assert in_band > 100 * out_band


def test_build_design_grid_shapes():
  coefficients = np.ones((6, 8, 5), dtype=np.float32)
  grid = build_design_grid(coefficients, dt=0.5, dx=1.0, dy=1.0)
  assert grid.coefficients.shape == (6, 8, 5)
  assert grid.electric_conductivity.shape == (6, 8)
  assert grid.pole_coefficients.shape == (6, 8, 1, 5)
  assert not bool(grid.multipole_mask.any())


def test_zero_state_2d_shapes():
  state = zero_state_2d((6, 8), dtype=jnp.float32)
  assert state.electric_z.shape == (6, 8)
  assert state.polarization_current.shape == (6, 8, 1)
  assert int(state.step) == 0


def test_project_density_clips_and_masks():
  mask = np.array([[True, False], [False, True]])
  rho = jnp.asarray(np.array([[1.5, 0.7], [-0.5, 0.3]], dtype=np.float32))
  projected = np.asarray(project_density(rho, jnp.asarray(mask)))
  np.testing.assert_allclose(projected, [[1.0, 0.0], [0.0, 0.3]], atol=1e-6)


def test_objective_and_gradient_are_finite_and_masked():
  problem = _make_problem()
  objective = make_objective(problem, penalty=3.0, mode="transmission")
  rho = np.full(problem.shape, 0.5, dtype=np.float32)
  loss = float(objective(jnp.asarray(rho)))
  assert np.isfinite(loss)
  gradient = np.asarray(adjoint_gradient(problem, rho))
  assert np.all(np.isfinite(gradient))
  # Cells outside the design region receive exactly zero gradient.
  np.testing.assert_allclose(gradient[~problem.design_mask], 0.0, atol=1e-12)


def test_adjoint_gradient_matches_finite_difference():
  problem = _make_problem(num_steps=40)
  objective = make_objective(problem, penalty=2.0, mode="transmission")
  rng = np.random.default_rng(7)
  rho = np.full(problem.shape, 0.5, dtype=np.float32)
  rho[problem.design_mask] = rng.uniform(
      0.3, 0.7, size=int(problem.design_mask.sum())).astype(np.float32)
  gradient = np.asarray(jax.grad(objective)(jnp.asarray(rho)))

  i, j = np.argwhere(problem.design_mask)[3]
  step = 1e-2
  rho_plus = rho.copy()
  rho_minus = rho.copy()
  rho_plus[i, j] += step
  rho_minus[i, j] -= step
  fd = (float(objective(jnp.asarray(rho_plus))) -
        float(objective(jnp.asarray(rho_minus)))) / (2 * step)
  # float32 central differences are noisy; a correct adjoint is still within
  # a few percent of the finite-difference estimate.
  assert fd == pytest.approx(float(gradient[i, j]), rel=2e-2)


def test_run_inverse_design_improves_transmission():
  problem = _make_problem(num_steps=70, design_eps=9.0)
  rho0 = np.zeros(problem.shape, dtype=np.float32)
  rho0[problem.design_mask] = 0.5
  result = run_inverse_design(
      problem, rho0, iterations=60, learning_rate=0.1, penalty=3.0,
      mode="transmission", binarize=False)
  assert isinstance(result, DesignResult)
  assert result.objective_history.shape == (60,)
  assert result.gradient_norm_history.shape == (60,)
  # Maximizing transmission thins the blocking slab, raising transmitted power.
  assert result.objective_history[-1] > result.objective_history[0]
  # The gradient norm should decay as the design converges.
  assert result.gradient_norm_history[-1] < result.gradient_norm_history[0]
  assert np.all(np.asarray(result.density) >= 0.0)
  assert np.all(np.asarray(result.density) <= 1.0)


def test_field_overlap_prefers_matching_mode():
  dt = 0.5
  frequency = 0.25
  time = np.arange(64, dtype=np.float32) * dt
  nx = 8
  # A pure travelling mode matching the target profile.
  target = np.sin(np.linspace(0.0, np.pi, nx, dtype=np.float32))
  matching = np.sin(2 * np.pi * frequency * time)[:, None] * target[None, :]
  mismatched = np.sin(2 * np.pi * frequency * time)[:, None] * np.ones(
      (1, nx), dtype=np.float32)
  match = float(field_overlap(
      jnp.asarray(matching), jnp.asarray(target), dt, frequency))
  mismatch = float(field_overlap(
      jnp.asarray(mismatched), jnp.asarray(target), dt, frequency))
  assert 0.0 <= match <= 1.0
  assert match > mismatch


def test_band_limited_objective_is_wavelength_selective():
  problem = _make_problem(num_steps=70, design_eps=9.0)
  band = make_objective(
      problem, penalty=1.0, mode="transmission",
      frequency_min=0.15, frequency_max=0.25)
  rho = np.full(problem.shape, 0.5, dtype=np.float32)
  loss = float(band(jnp.asarray(rho)))
  assert np.isfinite(loss)
  gradient = np.asarray(jax.grad(band)(jnp.asarray(rho)))
  assert np.all(np.isfinite(gradient))
  # A fully opaque slab transmits essentially nothing in band; vacuum does not.
  opaque = np.asarray(jnp.full(problem.shape, 1.0, dtype=np.float32))
  clear = np.asarray(jnp.zeros(problem.shape, dtype=np.float32))
  assert float(band(jnp.asarray(opaque))) < float(band(jnp.asarray(clear)))


def test_run_inverse_design_minimize_reflection():
  problem = _make_problem(num_steps=70, design_eps=9.0)
  rho0 = np.full(problem.shape, 0.5, dtype=np.float32)
  result = run_inverse_design(
      problem, rho0, iterations=40, learning_rate=0.1, penalty=3.0,
      mode="reflection", maximize=False, binarize=False)
  # Minimizing reflection should lower the reflected power below its start.
  assert result.objective_history[-1] < result.objective_history[0]
