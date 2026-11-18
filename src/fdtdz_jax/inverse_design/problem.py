"""Optimization-problem abstraction mirroring Meep's ``OptimizationProblem``."""

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

import jax
import jax.numpy as jnp

from ..jax_yee_ade_2d import (
    jax_material_grid_2d,
    jax_yee_ade_2d_state,
    simulate_jax_yee_ade_2d_probes,
)
from ..yee_ade_2d import initialize_yee_ade_2d
from .constraints import DesignConstraint, ScalarConstraint
from .materials_jax import as_eps
from .materials_jax import eps_inf_of
from .materials_jax import interpolate_coefficients
from .materials_jax import interpolate_multipole
from .materials_jax import material_coefficients
from .materials_jax import pole_coefficients
from .materials_jax import pole_count
from .objectives import SingleFrequencyTransmission
from .parameterization import DesignParameters, filtered_projected


class DesignSpec(NamedTuple):
    """Hashable snapshot of a design region, safe to capture inside ``jax.jit``."""

    x0: int
    x1: int
    y0: int
    y1: int
    grid_size: Tuple[int, int]
    filter_type: Optional[str]
    radius: float
    sigma: float
    periodic_axes: Optional[Tuple[int, ...]]
    binary_material: bool
    dx: float
    dy: float
    beta: float
    eta: float
    dispersive: bool
    multipole: bool
    medium1: float
    medium2: float
    background_coeff: Tuple[float, float, float, float, float]
    design_coeff: Tuple[float, float, float, float, float]
    background_eps_inf: float
    design_eps_inf: float
    background_pole: Tuple[Tuple[float, float, float, float, float], ...]
    design_pole: Tuple[Tuple[float, float, float, float, float], ...]
    p: float
    damping: float


@dataclass(frozen=True)
class DesignGeometryLimits:
    """Static size and clearance checks applied when a problem is created."""

    min_width: Optional[int] = None
    max_width: Optional[int] = None
    min_height: Optional[int] = None
    max_height: Optional[int] = None
    cpml_clearance: int = 0
    source_clearance: int = 0
    reflection_probe_clearance: int = 0
    transmission_probe_clearance: int = 0

    def __post_init__(self):
        for name in ("min_width", "max_width", "min_height", "max_height"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, Integral) or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        for pair in (("min_width", "max_width"), ("min_height", "max_height")):
            minimum = getattr(self, pair[0])
            maximum = getattr(self, pair[1])
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(f"{pair[0]} must not exceed {pair[1]}.")
        for name in (
            "cpml_clearance",
            "source_clearance",
            "reflection_probe_clearance",
            "transmission_probe_clearance",
        ):
            value = getattr(self, name)
            if not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")


@dataclass(frozen=True)
class DesignRegion:
    """A rectangular design region overlaid on the Yee grid."""

    design_parameters: DesignParameters
    x0: int
    x1: int
    y0: int
    y1: int

    def __post_init__(self):
        for name in ("x0", "x1", "y0", "y1"):
            if not isinstance(getattr(self, name), Integral):
                raise TypeError(f"design region {name} must be an integer.")
        nx = self.x1 - self.x0
        ny = self.y1 - self.y0
        if nx <= 0 or ny <= 0:
            raise ValueError("design region must have positive width and height.")
        if (nx, ny) != tuple(self.design_parameters.grid_size):
            raise ValueError(
                f"design region slice ({nx}, {ny}) does not match grid_size "
                f"{self.design_parameters.grid_size}."
            )

    def spec(self, dt, dtype=np.float64, maximum_poles=0) -> DesignSpec:
        dp = self.design_parameters
        if dp.is_multipole:
            background = (0.0, 0.0, 0.0, 0.0, 0.0)
            design = (0.0, 0.0, 0.0, 0.0, 0.0)
            medium1 = 0.0
            medium2 = 0.0
            background_eps_inf = eps_inf_of(dp.medium1)
            design_eps_inf = eps_inf_of(dp.medium2)
            background_pole = tuple(
                tuple(row) for row in pole_coefficients(
                    dp.medium1, dt, maximum_poles, dtype
                )
            )
            design_pole = tuple(
                tuple(row) for row in pole_coefficients(
                    dp.medium2, dt, maximum_poles, dtype
                )
            )
        elif dp.is_dispersive:
            background = tuple(material_coefficients(dp.medium1, dt, dtype))
            design = tuple(material_coefficients(dp.medium2, dt, dtype))
            medium1 = 0.0
            medium2 = 0.0
            background_eps_inf = 0.0
            design_eps_inf = 0.0
            background_pole = ()
            design_pole = ()
        else:
            background = (0.0, 0.0, 0.0, 0.0, 0.0)
            design = (0.0, 0.0, 0.0, 0.0, 0.0)
            medium1 = as_eps(dp.medium1)
            medium2 = as_eps(dp.medium2)
            background_eps_inf = 0.0
            design_eps_inf = 0.0
            background_pole = ()
            design_pole = ()
        return DesignSpec(
            x0=self.x0,
            x1=self.x1,
            y0=self.y0,
            y1=self.y1,
            grid_size=tuple(dp.grid_size),
            filter_type=dp.filter_type,
            radius=dp.radius,
            sigma=dp.sigma,
            periodic_axes=dp.periodic_axes,
            binary_material=dp.binary_material,
            dx=dp.dx,
            dy=dp.dy,
            beta=dp.beta,
            eta=dp.eta,
            dispersive=dp.is_dispersive,
            multipole=dp.is_multipole,
            medium1=medium1,
            medium2=medium2,
            background_coeff=background,
            design_coeff=design,
            background_eps_inf=background_eps_inf,
            design_eps_inf=design_eps_inf,
            background_pole=background_pole,
            design_pole=design_pole,
            p=dp.p,
            damping=dp.damping,
        )


def _projected_densities(rho, specs):
    """Apply filtering and projection once to each design region."""

    rho = jnp.asarray(rho)
    densities = []
    offset = 0
    for spec in specs:
        n = (spec.x1 - spec.x0) * (spec.y1 - spec.y0)
        weights = rho[offset : offset + n]
        offset += n
        densities.append(
            filtered_projected(
                weights,
                spec.grid_size,
                spec.filter_type,
                spec.radius,
                spec.sigma,
                spec.dx,
                spec.dy,
                spec.beta,
                spec.eta,
                spec.periodic_axes,
                spec.binary_material,
            )
        )
    return tuple(densities)


def _apply_design(grid0, rho, specs):
    """Overlay differentiable design densities onto a background JAX grid."""

    rho = jnp.asarray(rho, dtype=grid0.coefficients.dtype)
    projected = _projected_densities(rho, specs)
    coefficients = grid0.coefficients
    conductivity = grid0.electric_conductivity
    multipole_mask = grid0.multipole_mask
    multipole_eps_inf = grid0.multipole_eps_inf
    pole_grid = grid0.pole_coefficients
    for spec, density in zip(specs, projected):
        interpolation_p = 1.0 if spec.binary_material else spec.p
        if spec.multipole:
            eps_inf, pole = interpolate_multipole(
                spec.background_eps_inf,
                spec.design_eps_inf,
                spec.background_pole,
                spec.design_pole,
                density,
                p=interpolation_p,
            )
            a0 = 1.0 / eps_inf
            coefficients = coefficients.at[
                spec.x0 : spec.x1, spec.y0 : spec.y1, 0
            ].set(a0)
            coefficients = coefficients.at[
                spec.x0 : spec.x1, spec.y0 : spec.y1, 1:
            ].set(0.0)
            multipole_eps_inf = multipole_eps_inf.at[
                spec.x0 : spec.x1, spec.y0 : spec.y1
            ].set(eps_inf)
            pole_grid = pole_grid.at[
                spec.x0 : spec.x1, spec.y0 : spec.y1, :, :
            ].set(pole)
            multipole_mask = multipole_mask.at[
                spec.x0 : spec.x1, spec.y0 : spec.y1
            ].set(True)
        elif spec.dispersive:
            background = jnp.asarray(
                spec.background_coeff, dtype=coefficients.dtype
            )
            design = jnp.asarray(spec.design_coeff, dtype=coefficients.dtype)
            coeff = interpolate_coefficients(
                background, design, density, p=interpolation_p
            )
            coefficients = coefficients.at[
                spec.x0 : spec.x1, spec.y0 : spec.y1, :
            ].set(coeff)
        else:
            eps = (
                spec.medium1
                + density**interpolation_p * (spec.medium2 - spec.medium1)
            )
            a0 = 1.0 / eps
            coefficients = coefficients.at[
                spec.x0 : spec.x1, spec.y0 : spec.y1, 0
            ].set(a0)
            coefficients = coefficients.at[
                spec.x0 : spec.x1, spec.y0 : spec.y1, 1:
            ].set(0.0)
        if spec.damping:
            conductivity = conductivity.at[
                spec.x0 : spec.x1, spec.y0 : spec.y1
            ].set(spec.damping * density * (1.0 - density))
    return grid0._replace(
        coefficients=coefficients,
        electric_conductivity=conductivity,
        multipole_mask=multipole_mask,
        multipole_eps_inf=multipole_eps_inf,
        pole_coefficients=pole_grid,
    )


class InverseDesignProblem:
    """Top-level inverse-design problem for the JAX ADE 2D TMz solver.

    Mirrors the public surface of Meep's ``OptimizationProblem``: it is
    constructed from objective arguments and objective functions plus design
    regions, and ``__call__(rho)`` returns ``(objective, gradient)``.  The
    gradient comes from ``jax.value_and_grad`` through the differentiable
    solver rather than from an explicit adjoint run. Optional scalar
    constraints and projected-density constraints are exposed through
    ``get_constrained_fdfuncs``.

    A complex transmission objective automatically precomputes the empty-design
    transmission probe unless ``reference_transmission`` is supplied.
    """

    def __init__(
        self,
        grid,
        design_regions,
        electric_waveform,
        magnetic_waveform,
        x_profile,
        source_y,
        reflection_probe_y,
        transmission_probe_y,
        objective_arguments,
        objective_functions,
        constraints=(),
        reference_transmission=None,
        design_constraints=(),
        geometry_limits: Optional[DesignGeometryLimits] = None,
    ):
        self.design_regions = (
            list(design_regions)
            if isinstance(design_regions, (list, tuple))
            else [design_regions]
        )
        self.num_design_params = [
            dr.design_parameters.num_params for dr in self.design_regions
        ]
        self.num_design_regions = len(self.design_regions)
        self.objective_arguments = list(objective_arguments)
        self.objective_functions = (
            list(objective_functions)
            if isinstance(objective_functions, (list, tuple))
            else [objective_functions]
        )
        self.constraints = (
            [constraints]
            if isinstance(constraints, ScalarConstraint)
            else list(constraints)
        )
        for constraint in self.constraints:
            if not isinstance(constraint, ScalarConstraint):
                raise TypeError(
                    "constraints must contain ScalarConstraint instances."
                )
        self.design_constraints = (
            [design_constraints]
            if isinstance(design_constraints, DesignConstraint)
            else list(design_constraints)
        )
        for constraint in self.design_constraints:
            if not isinstance(constraint, DesignConstraint):
                raise TypeError(
                    "design_constraints must contain DesignConstraint instances."
                )
        if geometry_limits is not None and not isinstance(
            geometry_limits, DesignGeometryLimits
        ):
            raise TypeError(
                "geometry_limits must be a DesignGeometryLimits instance."
            )
        self.geometry_limits = geometry_limits

        self.grid = grid
        dtype = grid.coefficients.dtype
        self._grid0 = jax_material_grid_2d(grid, dtype=dtype)
        self._state0 = jax_yee_ade_2d_state(
            initialize_yee_ade_2d(grid), grid, dtype=dtype
        )

        self.electric_waveform = jnp.asarray(electric_waveform, dtype=dtype)
        self.magnetic_waveform = jnp.asarray(magnetic_waveform, dtype=dtype)
        self.x_profile = jnp.asarray(x_profile, dtype=dtype)
        self.source_y = int(source_y)
        self.reflection_probe_y = int(reflection_probe_y)
        self.transmission_probe_y = int(transmission_probe_y)
        ny = self.grid.shape[1]
        for name, value in (
            ("source_y", self.source_y),
            ("reflection_probe_y", self.reflection_probe_y),
            ("transmission_probe_y", self.transmission_probe_y),
        ):
            if not 0 <= value < ny:
                raise ValueError(f"{name} must lie inside the grid.")
        self._validate_regions()
        if (
            reference_transmission is None
            and any(
                isinstance(quantity, SingleFrequencyTransmission)
                for quantity in self.objective_arguments
            )
        ):
            _, reference_transmission = self._solve_reference_probes()
        if reference_transmission is not None:
            reference_array = np.asarray(reference_transmission)
            if reference_array.ndim != 1:
                raise ValueError("reference_transmission must be one-dimensional.")
            if reference_array.shape != self.electric_waveform.shape:
                raise ValueError(
                    "reference_transmission must have one sample per time step."
                )
            if (
                not np.issubdtype(reference_array.dtype, np.number)
                or not np.all(np.isfinite(reference_array))
            ):
                raise ValueError(
                    "reference_transmission must contain finite numeric values."
                )
            reference_transmission = jnp.asarray(
                reference_array, dtype=dtype
            )
        self.reference_transmission = reference_transmission

        self.f_bank: List[float] = []
        self._rebuild()

    def _solve_reference_probes(self):
        """Run the empty-design reference problem at both probe planes."""

        _, probes = simulate_jax_yee_ade_2d_probes(
            self._grid0,
            self._state0,
            self.electric_waveform,
            self.magnetic_waveform,
            self.x_profile,
            self.source_y,
            self.reflection_probe_y,
            self.transmission_probe_y,
        )
        return probes

    def _validate_regions(self):
        nx, ny = self.grid.shape
        limits = self.geometry_limits

        def clearance(y0, y1, index):
            if y1 <= index:
                return index - y1
            if y0 >= index:
                return y0 - index
            return -1

        for dr in self.design_regions:
            if not (0 <= dr.x0 < dr.x1 <= nx and 0 <= dr.y0 < dr.y1 <= ny):
                raise ValueError(
                    f"design region {dr} is outside the grid shape {(nx, ny)}."
                )
            width = dr.x1 - dr.x0
            height = dr.y1 - dr.y0
            if limits is not None:
                if limits.min_width is not None and width < limits.min_width:
                    raise ValueError(
                        f"design region width {width} is below "
                        f"minimum {limits.min_width}."
                    )
                if limits.max_width is not None and width > limits.max_width:
                    raise ValueError(
                        f"design region width {width} exceeds "
                        f"maximum {limits.max_width}."
                    )
                if limits.min_height is not None and height < limits.min_height:
                    raise ValueError(
                        f"design region height {height} is below "
                        f"minimum {limits.min_height}."
                    )
                if limits.max_height is not None and height > limits.max_height:
                    raise ValueError(
                        f"design region height {height} exceeds "
                        f"maximum {limits.max_height}."
                    )
            cpml = self.grid.y_cpml
            if cpml is not None and (
                dr.y0 < cpml.width or dr.y1 > ny - cpml.width
            ):
                raise ValueError(
                    "design region must lie outside the CPML boundary layers."
                )
            if cpml is not None and limits is not None:
                margin = cpml.width + limits.cpml_clearance
                if dr.y0 < margin or dr.y1 > ny - margin:
                    raise ValueError(
                        "design region does not keep the requested clearance "
                        "from the CPML."
                    )
            if limits is not None:
                for name, index, required in (
                    ("source", self.source_y, limits.source_clearance),
                    (
                        "reflection probe",
                        self.reflection_probe_y,
                        limits.reflection_probe_clearance,
                    ),
                    (
                        "transmission probe",
                        self.transmission_probe_y,
                        limits.transmission_probe_clearance,
                    ),
                ):
                    gap = clearance(dr.y0, dr.y1, index)
                    if gap < required:
                        raise ValueError(
                            f"design region clearance from {name} is {gap} "
                            f"cells; at least {required} are required."
                        )
            if dr.design_parameters.is_multipole:
                required = max(
                    pole_count(dr.design_parameters.medium1),
                    pole_count(dr.design_parameters.medium2),
                )
                if required > self.grid.maximum_poles:
                    raise ValueError(
                        f"design region needs {required} poles but the grid was "
                        f"built with maximum_poles={self.grid.maximum_poles}; "
                        "include the Multipole material in "
                        "prepare_material_grid_2d()'s materials list."
                    )

    def _rebuild(self):
        self._specs = tuple(
            dr.spec(
                self.grid.dt,
                self.grid.coefficients.dtype,
                self.grid.maximum_poles,
            )
            for dr in self.design_regions
        )
        self._quantities_fn = self._build_quantities()
        self._loss_fn = self._build_loss()
        self._value_and_grad = jax.jit(jax.value_and_grad(self._loss_fn))
        self._constrained_evaluator = self._build_constrained_evaluator()

    def _build_quantities(self):
        grid0 = self._grid0
        state0 = self._state0
        electric_waveform = self.electric_waveform
        magnetic_waveform = self.magnetic_waveform
        x_profile = self.x_profile
        source_y = self.source_y
        reflection_probe_y = self.reflection_probe_y
        transmission_probe_y = self.transmission_probe_y
        specs = tuple(self._specs)
        objective_arguments = self.objective_arguments
        reference_transmission = self.reference_transmission

        def quantities(rho):
            grid = _apply_design(grid0, rho, specs)
            _, probes = simulate_jax_yee_ade_2d_probes(
                grid,
                state0,
                electric_waveform,
                magnetic_waveform,
                x_profile,
                source_y,
                reflection_probe_y,
                transmission_probe_y,
            )
            outputs = {"reflection": probes[0], "transmission": probes[1]}
            if reference_transmission is not None:
                outputs["reference_transmission"] = reference_transmission
            return tuple(q(outputs) for q in objective_arguments)

        return quantities

    def _objective_from_quantities(self, quantities):
        objective_functions = self.objective_functions
        values = [f(*quantities) for f in objective_functions]
        return jnp.sum(jnp.stack([jnp.asarray(v) for v in values]))

    def _constraint_values(self, quantities, kind):
        values = []
        for constraint in self.constraints:
            if constraint.kind != kind:
                continue
            value = jnp.asarray(constraint.function(*quantities))
            if value.shape != ():
                name = constraint.name or "<unnamed>"
                raise ValueError(
                    f"constraint {name!r} must return a scalar."
                )
            values.append(value)
        if not values:
            return jnp.zeros((0,), dtype=jnp.float64)
        return jnp.stack(values)

    def _design_constraint_values(self, projected, kind):
        values = []
        for constraint in self.design_constraints:
            if constraint.kind != kind:
                continue
            for density in projected:
                value = jnp.asarray(constraint.function(density))
                if value.shape != ():
                    name = constraint.name or "<unnamed>"
                    raise ValueError(
                        f"design constraint {name!r} must return a scalar."
                    )
                values.append(value)
        if not values:
            return jnp.zeros((0,), dtype=jnp.float64)
        return jnp.stack(values)

    def _build_loss(self):
        quantities_fn = self._quantities_fn

        def loss(rho):
            quantities = quantities_fn(rho)
            return self._objective_from_quantities(quantities)

        return loss

    def _build_constrained_evaluator(self):
        quantities_fn = self._quantities_fn
        specs = tuple(self._specs)
        num_physical_eq = sum(
            constraint.kind == "eq" for constraint in self.constraints
        )
        num_physical_ineq = len(self.constraints) - num_physical_eq
        num_design_eq = sum(
            constraint.kind == "eq" for constraint in self.design_constraints
        ) * self.num_design_regions
        num_design_ineq = (
            len(self.design_constraints) * self.num_design_regions
            - num_design_eq
        )

        def values(rho):
            quantities = quantities_fn(rho)
            projected = _projected_densities(rho, specs)
            objective = jnp.asarray(self._objective_from_quantities(quantities))
            if objective.shape != ():
                raise ValueError("the objective must evaluate to a scalar.")
            equality = jnp.concatenate(
                (
                    self._constraint_values(quantities, "eq"),
                    self._design_constraint_values(projected, "eq"),
                )
            )
            inequality = jnp.concatenate(
                (
                    self._constraint_values(quantities, "ineq"),
                    self._design_constraint_values(projected, "ineq"),
                )
            )
            return jnp.concatenate(
                (objective.reshape((1,)), equality, inequality)
            )

        @jax.jit
        def evaluate(rho):
            values_at_rho, pullback = jax.vjp(values, rho)
            cotangents = jnp.eye(values_at_rho.shape[0], dtype=rho.dtype)
            jacobian = jax.vmap(pullback)(cotangents)[0]
            objective = values_at_rho[0]
            objective_gradient = jacobian[0]
            eq_start = 1
            eq_stop = eq_start + num_physical_eq + num_design_eq
            ineq_start = eq_stop
            ineq_stop = (
                ineq_start + num_physical_ineq + num_design_ineq
            )
            return (
                objective,
                objective_gradient,
                values_at_rho[eq_start:eq_stop],
                jacobian[eq_start:eq_stop],
                values_at_rho[ineq_start:ineq_stop],
                jacobian[ineq_start:ineq_stop],
            )

        return evaluate

    def _rho_vector(self):
        dtype = self._grid0.coefficients.dtype
        parts = [
            jnp.asarray(dr.design_parameters.weights, dtype=dtype)
            for dr in self.design_regions
        ]
        return parts[0] if len(parts) == 1 else jnp.concatenate(parts)

    def _split_gradient(self, gradient):
        parts = []
        start = 0
        for n in self.num_design_params:
            parts.append(gradient[start : start + n])
            start += n
        return parts[0] if len(parts) == 1 else parts

    def update_design(self, rho_vector, beta: Optional[float] = None) -> None:
        rho_vector = list(rho_vector)
        if self.num_design_regions == 1 and np.asarray(rho_vector).ndim == 1:
            rho_vector = [rho_vector]
        if len(rho_vector) != self.num_design_regions:
            raise ValueError(
                "rho_vector must contain one array per design region."
            )
        beta_changed = False
        for bi, b in enumerate(self.design_regions):
            b.design_parameters.update_weights(rho_vector[bi])
            if beta is not None and beta != b.design_parameters.beta:
                b.design_parameters.update_beta(beta)
                beta_changed = True
        if beta_changed:
            self._rebuild()

    def update_beta(self, beta: float) -> None:
        changed = False
        for b in self.design_regions:
            if beta != b.design_parameters.beta:
                b.design_parameters.update_beta(beta)
                changed = True
        if changed:
            self._rebuild()

    def __call__(
        self,
        rho_vector=None,
        need_value: bool = True,
        need_gradient: bool = True,
        beta: Optional[float] = None,
    ):
        """Evaluate objective value and/or gradient (Meep-compatible API)."""
        if rho_vector is not None:
            self.update_design(rho_vector, beta=beta)
        rho = self._rho_vector()

        if need_value and need_gradient:
            value, gradient = self._value_and_grad(rho)
        elif need_value:
            value = self._loss_fn(rho)
            gradient = None
        elif need_gradient:
            value = None
            gradient = jax.grad(self._loss_fn)(rho)
        else:
            raise ValueError("need_value and need_gradient cannot both be False.")

        if need_value:
            value = float(np.asarray(value))
            self.f_bank.append(value)
        if need_gradient:
            gradient = self._split_gradient(gradient)
        return value, gradient

    def get_fdf_funcs(self) -> Tuple[Callable, Callable]:
        """Return standalone ``(f, df)`` callables for scipy-style optimizers."""

        def f(x=None):
            rho = self._rho_vector() if x is None else jnp.asarray(x)
            return self._loss_fn(rho)

        def df(x=None):
            rho = self._rho_vector() if x is None else jnp.asarray(x)
            return self._split_gradient(jax.grad(self._loss_fn)(rho))

        return f, df

    def get_constrained_fdfuncs(self):
        """Return one JAX evaluator for objective and constraint data.

        The returned function accepts a flat design vector and returns
        ``(objective, objective_gradient, eq_values, eq_jacobian,
        ineq_values, ineq_jacobian)``.
        """

        def evaluate(x=None):
            rho = self._rho_vector() if x is None else jnp.asarray(x)
            return self._constrained_evaluator(rho)

        return evaluate

    def evaluate_quantities(self, rho_vector=None):
        """Evaluate objective quantities without changing stored design data."""

        rho = self._rho_vector() if rho_vector is None else jnp.asarray(
            rho_vector
        )
        return tuple(
            np.asarray(value) for value in self._quantities_fn(rho)
        )

    def evaluate_projected_densities(self, rho_vector=None):
        """Return each region's filtered and projected density as NumPy arrays."""

        rho = self._rho_vector() if rho_vector is None else jnp.asarray(
            rho_vector
        )
        return tuple(
            np.asarray(density)
            for density in _projected_densities(rho, tuple(self._specs))
        )

    def constraint_names(self, kind: str) -> Tuple[str, ...]:
        """Return physical and design constraint names in evaluator order."""

        if kind not in ("eq", "ineq"):
            raise ValueError("constraint kind must be 'eq' or 'ineq'.")
        names = []
        for constraint in self.constraints:
            if constraint.kind == kind:
                names.append(constraint.name or "<unnamed>")
        for constraint in self.design_constraints:
            if constraint.kind == kind:
                for index in range(self.num_design_regions):
                    name = constraint.name or "<unnamed>"
                    names.append(f"{name}[region={index}]")
        return tuple(names)

    def calculate_fd_gradient(
        self,
        num_gradients: int = 1,
        db: float = 1e-4,
        design_variables_idx: int = 0,
        seed: Optional[int] = 0,
    ):
        """Estimate central-difference gradients on a few random parameters."""
        if self.num_design_regions != 1:
            raise NotImplementedError(
                "finite-difference checks currently support a single design region."
            )
        rho0 = np.asarray(
            self.design_regions[design_variables_idx].design_parameters.weights
        ).copy()
        total = rho0.size
        num_gradients = int(num_gradients)
        if not (0 < num_gradients <= total):
            raise ValueError(
                "num_gradients must be in [1, num_design_params]."
            )
        idx = np.random.default_rng(seed).choice(
            total, num_gradients, replace=False
        )
        fd_gradient = []
        for k in idx:
            rho_plus = rho0.copy()
            rho_minus = rho0.copy()
            rho_plus[k] += db
            rho_minus[k] -= db
            fp = float(
                np.asarray(self._loss_fn(jnp.asarray(rho_plus)))
            )
            fm = float(
                np.asarray(self._loss_fn(jnp.asarray(rho_minus)))
            )
            fd_gradient.append((fp - fm) / (2.0 * db))
        return fd_gradient, idx


__all__ = ["DesignGeometryLimits", "DesignRegion", "InverseDesignProblem"]
