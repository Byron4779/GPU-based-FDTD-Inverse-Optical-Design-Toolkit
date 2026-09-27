"""Differentiable inverse-design primitives for the JAX FDTD solvers.

This module turns the existing JAX ADE solvers into an adjoint-method
inverse-design engine.  Because :mod:`fdtdz_jax.jax_yee_ade_2d` is written as
pure, JIT-compiled JAX functions, the entire forward time integration is
automatically differentiable.  Differentiating a scalar objective through the
solver with reverse-mode autodiff is exactly the discrete adjoint method: one
forward simulation and one adjoint (backward) simulation yield the gradient of
the objective with respect to every design variable at once.

The design variables are a per-node density field ``rho(x, y)`` in ``[0, 1]``.
Each node interpolates between a background material and a design material
using a SIMP-style penalty, so ``rho -> ADE coefficients -> FDTD update`` is a
single differentiable map.  Two parameterizations are provided:

* density optimization (``rho`` directly), and
* level-set optimization (``rho = sigmoid(phi / width)`` over a level-set
  function ``phi``), which produces smooth, manufacturable boundaries.

Objectives are scalar and self-contained (no reference simulation needed):

* :func:`time_integrated_power` -- broadband transmitted/reflected energy, and
* :func:`band_power` -- energy in a target frequency band via ``jnp.fft.rfft``.

A small dependency-free Adam optimizer and an end-to-end driver
:func:`run_inverse_design` complete the toolkit.  This module depends only on
JAX and NumPy, matching the package's existing ``jax``-only requirements.
"""

from dataclasses import dataclass
import math

import numpy as np

import jax
from jax import numpy as jnp

from .jax_yee_ade_2d import JAXMaterialGrid2D
from .jax_yee_ade_2d import JAXYeeADE2DState
from .jax_yee_ade_2d import simulate_jax_yee_ade_2d_probes
from .materials import ade_coefficients


# ---------------------------------------------------------------------------
# Density parameterizations
# ---------------------------------------------------------------------------

def interpolate_coefficients(background, design, density, penalty=3.0):
  """SIMP-style interpolation between two ADE coefficient vectors.

  ``background`` and ``design`` are the five-vector ADE coefficients
  ``(a0, a1, a2, b1, b2)`` returned by
  :func:`fdtdz_jax.materials.ade_coefficients`.  ``density`` is a ``(...,)``
  array of values in ``[0, 1]``.  The result has shape ``(..., 5)``::

      coeff = background + density**penalty * (design - background)

  For lossless materials ``(a0, 0, 0, 0, 0)`` this is linear interpolation of
  ``1 / eps_r`` (inverse-permittivity mixing).  ``penalty >= 1`` biases the
  optimizer toward binary 0/1 designs.
  """
  background = jnp.asarray(background)
  design = jnp.asarray(design)
  density = jnp.asarray(density)
  if background.shape != (5,) or design.shape != (5,):
    raise ValueError(
        "background and design must be (5,) ADE coefficient arrays.")
  if penalty < 1.0 or not math.isfinite(penalty):
    raise ValueError("penalty must be a finite value >= 1.0.")
  weight = jnp.clip(density, 0.0, 1.0) ** penalty
  return background + weight[..., None] * (design - background)


def sigmoid_projection(density, beta=32.0, eta=0.5):
  """Soft threshold toward binary 0/1 density with sharpness ``beta``.

  This is the ``tanh`` projection popularized for robust topology
  optimization.  ``eta`` is the density value mapped to 0.5.  Larger ``beta``
  gives sharper, more binary boundaries at the cost of a stiffer optimization.
  """
  density = jnp.asarray(density)
  if beta < 0.0 or not math.isfinite(beta):
    raise ValueError("beta must be finite and non-negative.")
  if not 0.0 < eta < 1.0:
    raise ValueError("eta must be strictly between zero and one.")
  numerator = jnp.tanh(beta * eta) + jnp.tanh(beta * (density - eta))
  denominator = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
  return numerator / denominator


def level_set_density(phi, width=1.0):
  """Map a level-set function ``phi`` to density ``sigmoid(phi / width)``.

  ``width`` controls the interface thickness.  Optimizing ``phi`` instead of
  ``rho`` keeps boundaries smooth (a form of level-set inverse design).
  """
  if width <= 0.0 or not math.isfinite(width):
    raise ValueError("width must be finite and positive.")
  return jax.nn.sigmoid(jnp.asarray(phi) / width)


# ---------------------------------------------------------------------------
# Differentiable grid and state construction
# ---------------------------------------------------------------------------

def build_design_grid(coefficients, dt, dx, dy, cpml=None):
  """Build a :class:`JAXMaterialGrid2D` from a ``(nx, ny, 5)`` coefficient map.

  The grid has no conductivity and no multipole poles (the design materials
  are Lossless, Debye, Lorentz, or Drude, which are all captured by the five
  ADE coefficients).  ``cpml`` is an optional :class:`YCPML2D` whose y-directed
  absorbing coefficients are embedded as constants.
  """
  coefficients = jnp.asarray(coefficients)
  if coefficients.ndim != 3 or coefficients.shape[-1] != 5:
    raise ValueError("coefficients must have shape (nx, ny, 5).")
  nx, ny = coefficients.shape[:2]
  dtype = coefficients.dtype
  if cpml is None:
    inverse_kappa_e = jnp.ones(ny, dtype=dtype)
    inverse_kappa_h = jnp.ones(ny, dtype=dtype)
    b_e = c_e = b_h = c_h = jnp.zeros(ny, dtype=dtype)
  else:
    inverse_kappa_e = jnp.asarray(cpml.inverse_kappa_e, dtype=dtype)
    inverse_kappa_h = jnp.asarray(cpml.inverse_kappa_h, dtype=dtype)
    b_e = jnp.asarray(cpml.b_e, dtype=dtype)
    c_e = jnp.asarray(cpml.c_e, dtype=dtype)
    b_h = jnp.asarray(cpml.b_h, dtype=dtype)
    c_h = jnp.asarray(cpml.c_h, dtype=dtype)
  zeros = jnp.zeros((nx, ny), dtype=dtype)
  # A single all-zero pole keeps the multipole update numerically inert while
  # avoiding zero-sized axes, which break reverse-mode autodiff through
  # ``lax.scan``.  The multipole path is masked off by ``multipole_mask``.
  return JAXMaterialGrid2D(
      coefficients=coefficients,
      electric_conductivity=zeros,
      magnetic_conductivity=zeros,
      multipole_mask=jnp.zeros((nx, ny), dtype=bool),
      multipole_eps_inf=jnp.ones((nx, ny), dtype=dtype),
      pole_coefficients=jnp.zeros((nx, ny, 1, 5), dtype=dtype),
      inverse_kappa_e=inverse_kappa_e,
      b_e=b_e,
      c_e=c_e,
      inverse_kappa_h=inverse_kappa_h,
      b_h=b_h,
      c_h=c_h,
      dt=jnp.asarray(dt, dtype=dtype),
      dx=jnp.asarray(dx, dtype=dtype),
      dy=jnp.asarray(dy, dtype=dtype))


def zero_state_2d(shape, dtype=jnp.float32):
  """Create a zero-field :class:`JAXYeeADE2DState` with one inert pole slot.

  The single all-zero polarization slot matches :func:`build_design_grid` and
  keeps every axis non-empty so reverse-mode autodiff through ``lax.scan``
  remains valid.
  """
  nx, ny = (int(value) for value in shape)
  zeros = jnp.zeros((nx, ny), dtype=dtype)
  pole_zeros = jnp.zeros((nx, ny, 1), dtype=dtype)
  return JAXYeeADE2DState(
      electric_z=zeros,
      magnetic_x=zeros,
      magnetic_y=zeros,
      displacement_z=zeros,
      displacement_z_previous=zeros,
      electric_z_previous=zeros,
      cpml_hx_y=zeros,
      cpml_dz_y=zeros,
      polarization_current=pole_zeros,
      polarization_previous=pole_zeros,
      step=jnp.asarray(0, dtype=jnp.int32))


def project_density(density, mask):
  """Clip ``density`` to ``[0, 1]`` and zero any cell outside ``mask``."""
  density = jnp.clip(density, 0.0, 1.0)
  return jnp.where(mask, density, 0.0)


# ---------------------------------------------------------------------------
# Scalar objectives
# ---------------------------------------------------------------------------

def time_integrated_power(signal, dt):
  """Time-integrated energy of a scalar probe signal: ``dt * sum(signal**2)``.

  For a transmitted or reflected electric-field probe this is a monotonically
  increasing measure of output power for a fixed incident pulse.
  """
  signal = jnp.asarray(signal)
  return dt * jnp.sum(signal * signal)


def band_power(signal, dt, frequency_min, frequency_max):
  """Energy of a probe signal inside ``[frequency_min, frequency_max]``.

  The magnitude-squared ``rfft`` is summed over the requested band, giving a
  differentiable wavelength-selective objective without a reference run.
  """
  signal = jnp.asarray(signal)
  if signal.ndim != 1:
    raise ValueError("signal must be one-dimensional.")
  if frequency_max <= frequency_min:
    raise ValueError("frequency_max must exceed frequency_min.")
  spectrum = jnp.fft.rfft(signal)
  frequencies = jnp.fft.rfftfreq(signal.shape[0], d=dt)
  power = (spectrum * jnp.conj(spectrum)).real
  band = (frequencies >= frequency_min) & (frequencies <= frequency_max)
  return jnp.sum(power * band.astype(power.dtype))


def field_overlap(field, target, dt, frequency, bandwidth=None):
  """Normalized mode-overlap integral between a field and a target profile.

  ``field`` has shape ``(time, nx)``: the real electric field recorded on a
  plane normal to propagation (for example, a line probe from
  :func:`simulate_jax_yee_ade_2d_line_probes`).  ``target`` is the ``(nx,)``
  spatial profile to match.  The field is projected onto ``frequency`` with a
  Gaussian spectral window (``bandwidth``), and the overlap

      |sum(E_f * conj(target))|**2 / (sum(|E_f|**2) * sum(|target|**2))

  is returned in ``[0, 1]``.  Maximizing it steers the design toward a desired
  output mode (focusing, mode conversion, beam shaping) instead of a scalar
  power.  ``target`` may be complex for phase control.
  """
  field = jnp.asarray(field)
  target = jnp.asarray(target)
  if field.ndim != 2 or target.ndim != 1 or field.shape[1] != target.shape[0]:
    raise ValueError(
        f"field must be (time, nx) and target (nx,), got {field.shape} and "
        f"{target.shape}.")
  if frequency <= 0.0:
    raise ValueError("frequency must be positive.")
  spectrum = jnp.fft.rfft(field, axis=0)
  frequencies = jnp.fft.rfftfreq(field.shape[0], d=dt)
  if bandwidth is None:
    bandwidth = max(float(frequencies[1] - frequencies[0]),
                    float(frequency) * 0.05)
  weight = jnp.exp(-0.5 * ((frequencies - frequency) / bandwidth)**2)
  field_frequency = jnp.sum(spectrum * weight[:, None], axis=0)
  numerator = jnp.abs(jnp.sum(field_frequency * jnp.conj(target)))**2
  denominator = (
      jnp.sum(jnp.abs(field_frequency)**2) *
      jnp.sum(jnp.abs(target)**2))
  return numerator / (denominator + 1e-12)


# ---------------------------------------------------------------------------
# Problem specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InverseDesign2D:
  """Static configuration of a 2D TMz transmission inverse-design problem.

  The setup follows :func:`fdtdz_jax.run_tmz_scattering_2d`: a directional
  Huygens line source at ``source_y``, reflection and transmission probes, and
  an optional y-directed CPML.  ``design_mask`` marks the free cells; every
  other cell is fixed to the background material.

  Attributes:
    shape: ``(nx, ny)`` grid shape.
    design_mask: boolean ``(nx, ny)`` mask of free cells.
    background_coefficients: ``(5,)`` ADE coefficients of the fixed material.
    design_coefficients: ``(5,)`` ADE coefficients of the design material.
    dt: time step in normalized units.
    dx: x cell size in normalized units.
    dy: y cell size in normalized units.
    electric_waveform: ``(num_steps,)`` Jz waveform (Huygens pair).
    magnetic_waveform: ``(num_steps,)`` Mx waveform (Huygens pair).
    x_profile: ``(nx,)`` lateral source profile.
    source_y: y index of the source line.
    reflection_y: y index of the reflection probe.
    transmission_y: y index of the transmission probe.
    cpml: optional :class:`YCPML2D` for y-absorbing boundaries.
  """

  shape: tuple
  design_mask: np.ndarray
  background_coefficients: np.ndarray
  design_coefficients: np.ndarray
  dt: float
  dx: float
  dy: float
  electric_waveform: np.ndarray
  magnetic_waveform: np.ndarray
  x_profile: np.ndarray
  source_y: int
  reflection_y: int
  transmission_y: int
  cpml: object = None

  @classmethod
  def from_materials(cls, shape, design_mask, background, design_material,
                     dt, dx, dy, electric_waveform, magnetic_waveform,
                     x_profile, source_y, reflection_y, transmission_y,
                     cpml=None, dtype=np.float64):
    """Build the problem from material objects instead of raw coefficients."""
    shape = tuple(int(value) for value in shape)
    design_mask = np.asarray(design_mask)
    if design_mask.shape != shape or design_mask.dtype != np.bool_:
      raise ValueError("design_mask must be boolean with shape == shape.")
    x_profile = np.asarray(x_profile)
    if x_profile.shape != (shape[0],):
      raise ValueError(f"x_profile must have shape ({shape[0]},).")
    background_coefficients = ade_coefficients(
        background, dt).as_array(dtype)
    design_coefficients = ade_coefficients(design_material, dt).as_array(dtype)
    return cls(
        shape=shape,
        design_mask=np.array(design_mask, dtype=bool, copy=True),
        background_coefficients=background_coefficients,
        design_coefficients=design_coefficients,
        dt=float(dt),
        dx=float(dx),
        dy=float(dy),
        electric_waveform=np.asarray(electric_waveform, dtype=dtype),
        magnetic_waveform=np.asarray(magnetic_waveform, dtype=dtype),
        x_profile=np.asarray(x_profile, dtype=dtype),
        source_y=int(source_y),
        reflection_y=int(reflection_y),
        transmission_y=int(transmission_y),
        cpml=cpml)


# ---------------------------------------------------------------------------
# Objective, adjoint gradient, and driver
# ---------------------------------------------------------------------------

def make_objective(problem, penalty=3.0, projection_beta=None,
                   mode="transmission", frequency_min=None, frequency_max=None):
  """Return a pure scalar objective ``rho -> float`` for ``problem``.

  The returned function is differentiable with :func:`jax.grad` /
  :func:`jax.value_and_grad`.  ``mode`` selects the probe whose energy is
  returned: ``"transmission"`` or ``"reflection"``.  When both
  ``frequency_min`` and ``frequency_max`` are given, the objective is the
  band-limited power :func:`band_power` over that band (a wavelength-selective
  transmission or reflection); otherwise it is the broadband
  :func:`time_integrated_power`.

  The returned value is a quantity to be *maximized*; pass the result to
  ``-make_objective(...)`` or use the ``maximize`` flag of
  :func:`run_inverse_design` to minimize instead.
  """
  if not isinstance(problem, InverseDesign2D):
    raise TypeError("problem must be an InverseDesign2D instance.")
  if mode not in ("transmission", "reflection"):
    raise ValueError("mode must be 'transmission' or 'reflection'.")
  if (frequency_min is None) != (frequency_max is None):
    raise ValueError(
        "frequency_min and frequency_max must be provided together.")
  mask = jnp.asarray(problem.design_mask)
  background = jnp.asarray(problem.background_coefficients)
  design = jnp.asarray(problem.design_coefficients)
  electric_waveform = jnp.asarray(problem.electric_waveform)
  magnetic_waveform = jnp.asarray(problem.magnetic_waveform)
  x_profile = jnp.asarray(problem.x_profile)
  dt = problem.dt
  dx = problem.dx
  dy = problem.dy
  source_y = problem.source_y
  reflection_y = problem.reflection_y
  transmission_y = problem.transmission_y
  cpml = problem.cpml

  def objective(rho):
    density = jnp.clip(rho, 0.0, 1.0)
    if projection_beta is not None:
      density = sigmoid_projection(density, projection_beta)
    coefficients = interpolate_coefficients(
        background, design, density, penalty)
    coefficients = jnp.where(mask[..., None], coefficients, background)
    grid = build_design_grid(coefficients, dt, dx, dy, cpml)
    state = zero_state_2d(problem.shape, dtype=coefficients.dtype)
    # ``simulate_jax_yee_ade_2d_probes`` returns
    # ``(final_state, (reflection, transmission))`` from ``lax.scan``.
    _, (reflection, transmission) = simulate_jax_yee_ade_2d_probes(
        grid, state, electric_waveform, magnetic_waveform, x_profile,
        source_y, reflection_y, transmission_y)
    signal = transmission if mode == "transmission" else reflection
    if frequency_min is not None:
      return band_power(signal, dt, frequency_min, frequency_max)
    return time_integrated_power(signal, dt)

  return objective


def adjoint_gradient(problem, rho, penalty=3.0, projection_beta=None,
                     mode="transmission"):
  """Return the adjoint gradient of the objective with respect to ``rho``.

  This is the discrete adjoint: reverse-mode autodiff through the full FDTD
  time integration, producing one gradient per free cell in a single backward
  pass.
  """
  objective = make_objective(problem, penalty, projection_beta, mode)
  return jax.grad(objective)(jnp.asarray(rho))


@dataclass(frozen=True)
class DesignResult:
  """Result of an inverse-design run."""

  density: np.ndarray
  objective_history: np.ndarray
  gradient_norm_history: np.ndarray
  mode: str


@dataclass
class _AdamState:
  step: int
  first_moment: object
  second_moment: object


def _adam_init(params):
  zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
  return _AdamState(0, zeros, zeros)


def _adam_update(params, grads, state, learning_rate=0.05, beta1=0.9,
                 beta2=0.999, eps=1e-8):
  step = state.step + 1
  first = jax.tree_util.tree_map(
      lambda m, g: beta1 * m + (1.0 - beta1) * g, state.first_moment, grads)
  second = jax.tree_util.tree_map(
      lambda v, g: beta2 * v + (1.0 - beta2) * g * g,
      state.second_moment, grads)
  first_hat = jax.tree_util.tree_map(
      lambda m: m / (1.0 - beta1 ** step), first)
  second_hat = jax.tree_util.tree_map(
      lambda v: v / (1.0 - beta2 ** step), second)
  update = jax.tree_util.tree_map(
      lambda m, v: m / (jnp.sqrt(v) + eps), first_hat, second_hat)
  params = jax.tree_util.tree_map(
      lambda p, u: p - learning_rate * u, params, update)
  return params, _AdamState(step, first, second)


def run_inverse_design(problem, rho0, iterations=100, learning_rate=0.05,
                       penalty=3.0, projection_beta=None, mode="transmission",
                       maximize=True, binarize=True, binarize_threshold=0.5,
                       frequency_min=None, frequency_max=None):
  """Run Adam-based adjoint optimization of the design density.

  ``rho0`` is the initial ``(nx, ny)`` density (only cells inside
  ``problem.design_mask`` are optimized).  The objective is maximized by
  default; pass ``maximize=False`` to minimize it (for example, to minimize
  reflected power with ``mode="reflection"``).  ``frequency_min`` and
  ``frequency_max`` select a wavelength-selective objective.  Returns a
  :class:`DesignResult` with the final density and per-iteration objective and
  gradient norms.
  """
  if not isinstance(problem, InverseDesign2D):
    raise TypeError("problem must be an InverseDesign2D instance.")
  rho0 = np.asarray(rho0)
  if rho0.shape != problem.shape:
    raise ValueError(
        f"rho0 must have shape {problem.shape}, got {rho0.shape}.")
  mask = jnp.asarray(problem.design_mask)
  value_and_grad = jax.jit(jax.value_and_grad(
      make_objective(problem, penalty, projection_beta, mode,
                     frequency_min, frequency_max)))
  rho = project_density(jnp.asarray(rho0), mask)
  adam = _adam_init(rho)
  losses = []
  gradient_norms = []
  for _ in range(int(iterations)):
    loss, gradient = value_and_grad(rho)
    # ``_adam_update`` is a descent step.  Negate the gradient to maximize.
    step_gradient = -gradient if maximize else gradient
    rho, adam = _adam_update(
        rho, step_gradient, adam, learning_rate=learning_rate)
    rho = project_density(rho, mask)
    losses.append(float(loss))
    gradient_norms.append(float(jnp.sqrt(jnp.sum(gradient * gradient))))
  density = np.asarray(rho)
  if binarize:
    density = np.where(density >= binarize_threshold, 1.0, 0.0)
  return DesignResult(
      density=density,
      objective_history=np.asarray(losses, dtype=np.float64),
      gradient_norm_history=np.asarray(gradient_norms, dtype=np.float64),
      mode=mode)


__all__ = [
    "DesignResult",
    "InverseDesign2D",
    "adjoint_gradient",
    "band_power",
    "build_design_grid",
    "field_overlap",
    "interpolate_coefficients",
    "level_set_density",
    "make_objective",
    "project_density",
    "run_inverse_design",
    "sigmoid_projection",
    "time_integrated_power",
    "zero_state_2d",
]
