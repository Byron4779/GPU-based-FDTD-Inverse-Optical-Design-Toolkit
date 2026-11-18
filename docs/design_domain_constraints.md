# Design-domain constraints and binary validation

`InverseDesignProblem` accepts two independent constraint collections:

- `constraints`: physical constraints evaluated from objective quantities, such
  as the normalized complex transmission.
- `design_constraints`: scalar constraints evaluated from each region's
  `rho_projected` field with shape `(nx, ny)`.

Both collections use the SciPy sign convention: equality constraints target
zero and inequality constraints must be nonnegative. Their values and JAX
Jacobians are appended to the same SLSQP equality/inequality vectors.

## Volume and effective dimensions

```python
from fdtdz_jax import (
    effective_thickness_constraints,
    metal_volume_fraction_constraints,
)

design_constraints = (
    *metal_volume_fraction_constraints(minimum=0.0, maximum=0.4),
    *effective_thickness_constraints(minimum=2.0, maximum=4.0),
)
```

The volume fraction is `mean(rho_projected)`. Effective thickness is
`sum_y(mean_x(rho_projected))`, in grid cells. Effective area and aspect-ratio
constraint factories are also available. No constraint is added to the
objective as a weighted penalty.

With multiple design regions, each design constraint is evaluated once per
region. Constraint names therefore include a region suffix in optimizer
diagnostics and results.

## Static geometry checks

`DesignGeometryLimits` validates fixed region dimensions and clearances when
the problem is constructed:

```python
from fdtdz_jax import DesignGeometryLimits

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
```

The design box must lie inside the simulation grid, outside CPML, and clear of
the source and both probe lines. The density array shape remains fixed.

## Beta continuation

`minimize_slsqp_beta_continuation` rebuilds the differentiable material and
constraint evaluator at each beta, continues from the previous density, and
records objective, constraint violation, gray fraction, volume, and effective
thickness:

```python
continuation = minimize_slsqp_beta_continuation(
    problem,
    rho0,
    betas=(2.0, 4.0, 8.0, 16.0, 32.0),
    maxiter=120,
)
```

Optimizer failures are preserved per stage. They are not converted into
success by relaxing phase, amplitude, volume, or thickness tolerances.

## Endpoint-material validation

`density_metrics`, `transmission_metrics`, and `binary_topology_metrics`
provide the final report fields. `binary_topology_metrics` requires endpoint
values `{0, 1}` and reports:

- actual metal volume fraction;
- approximate minimum metal and air feature sizes in cells, using square
  morphological opening;
- metal and air connected-component counts, with periodic connectivity along
  x.

The canonical gold notebook thresholds the final projected density, creates a
new `beta=0`, unfiltered endpoint-material problem, and reruns the ADE solver.
Continuous-material success and binary endpoint success are reported and saved
separately.
