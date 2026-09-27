# Inverse design with the adjoint method

`fdtdz_jax.inverse_design` turns the differentiable JAX FDTD solvers into an
adjoint-method inverse-design engine.  This document explains the idea, the
API, and how to drive it.

## How it works

The forward solver (`simulate_jax_yee_ade_2d_probes` and friends) is written as
pure, JIT-compiled JAX.  A scalar objective `L(rho)` that sums a probe signal
over time is therefore differentiable with respect to every design variable at
once:

```python
gradient = jax.grad(objective)(rho)
```

One reverse-mode pass through the time loop is exactly the **discrete adjoint
method**: a single forward simulation plus a single backward (adjoint)
simulation yields the gradient of `L` with respect to all `nx * ny` design
variables.  This is the same gradient that the continuous adjoint-variable
method produces, obtained here by automatic differentiation rather than by
hand-derived adjoint equations.

The design variables are a per-node density `rho(x, y) in [0, 1]`.  Each cell
interpolates between a background and a design material through the five ADE
coefficients `(a0, a1, a2, b1, b2)`:

```text
coeff(rho) = coeff_bg + rho**penalty * (coeff_design - coeff_bg)
```

so the map `rho -> coefficients -> FDTD update` is one differentiable chain.
For lossless materials this is inverse-permittivity mixing; the same formula
works unchanged for `Drude` (metal), `Lorentz`, and `Debye` design materials,
giving complex-permittivity inverse design for free.

## Quick start

```python
import numpy as np
from fdtdz_jax import (
    InverseDesign2D, Lossless, prepare_y_cpml, gaussian_sine_pulse,
    huygens_current_waveforms, run_inverse_design,
)

nx, ny, num_steps = 32, 64, 200
dt, dx, dy = 0.5, 1.0, 1.0
waveform = gaussian_sine_pulse(num_steps, dt, 30.0, 12.0, 0.2)
electric, magnetic = huygens_current_waveforms(waveform, 1, 1.0)
cpml = prepare_y_cpml((nx, ny), 12, dt, dy)

design_mask = np.zeros((nx, ny), dtype=bool)
design_mask[:, 28:36] = True              # a slab of cells to design

problem = InverseDesign2D.from_materials(
    (nx, ny), design_mask, Lossless(1.0), Lossless(9.0),
    dt, dx, dy, electric, magnetic, np.ones(nx),
    source_y=14, reflection_y=18, transmission_y=46, cpml=cpml)

rho0 = np.zeros((nx, ny)); rho0[design_mask] = 0.5
result = run_inverse_design(
    problem, rho0, iterations=150, learning_rate=0.1,
    mode="transmission", maximize=True)
```

`result.density` is the optimized design, and
`result.objective_history` / `result.gradient_norm_history` record convergence.
A complete, runnable script is in
[`examples/inverse_design_2d.py`](../examples/inverse_design_2d.py).

## Parameterizations

* **Density** (`rho` directly) -- standard topology optimization.
* **Level set** (`level_set_density(phi, width)`) -- parameterize
  `rho = sigmoid(phi / width)` over a level-set function `phi` for smooth,
  manufacturable boundaries.
* **Projection** (`sigmoid_projection(rho, beta, eta)`) -- a tanh threshold that
  pushes intermediate densities toward 0/1 to suppress gray designs.

## Objectives

All objectives are scalar, self-contained (no reference simulation), and
differentiable:

| Function | Meaning |
| --- | --- |
| `time_integrated_power(signal, dt)` | broadband transmitted/reflected energy |
| `band_power(signal, dt, fmin, fmax)` | energy in a target frequency band |
| `field_overlap(field, target, dt, freq)` | normalized mode-overlap integral `[0, 1]` |

`make_objective` exposes the first two through `mode="transmission"` /
`"reflection"` and optional `frequency_min` / `frequency_max`.  `field_overlap`
matches a spatial profile (focusing, mode conversion, beam shaping) and is
used through a custom objective or `adjoint_gradient`.

## Optimization

`run_inverse_design` runs a small dependency-free Adam optimizer.  Set
`maximize=False` to minimize (e.g. antireflection via `mode="reflection"`).
The gradient is a plain `jax.grad`, so any JAX-compatible optimizer (Optax,
custom schedules, PyTorch via `dlpack`) can drive the same objective:

```python
import optax
objective = make_objective(problem, mode="transmission")
optimizer = optax.adam(0.1)
opt_state = optimizer.init(rho0)
# ... loss, grad = jax.value_and_grad(objective)(rho); updates, opt_state = optimizer.update(grad, opt_state)
```

## GPU and memory

The module is backend-agnostic: run on GPU by installing `jax[cuda12]` and
letting JAX place the solver there -- no CUDA-kernel changes are required for
the complex-permittivity path.  Reverse-mode autodiff through `lax.scan`
stores intermediate fields for the backward pass; for very large
`num_steps`, wrap the scan body with `jax.checkpoint` to trade recomputation
for memory, exactly the "time-reversal / memory-pooling" strategy noted in the
project plan.

## Limitations and extensions

* The current objective set is power/overlap based.  A fixed-material-volume or
  minimum-linewidth constraint can be added by penalizing `sum(rho)` or the
  density gradient in the objective.
* The 2D TMz path is the reference implementation; the identical recipe applies
  to the 3D solver in `jax_yee_ade_3d` (`JAXMaterialGrid3D`).
* Design materials use the five-coefficient path (Lossless/Debye/Lorentz/
  Drude).  `Multipole` materials need per-pole interpolation, which is not yet
  wired into the density map.
