"""Scalar and projected-density constraints for inverse design."""

from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple

import jax.numpy as jnp


@dataclass(frozen=True)
class ScalarConstraint:
    """One scalar constraint evaluated from the objective quantities.

    ``function`` receives the same positional arguments as entries in
    ``InverseDesignProblem.objective_functions``. Equality constraints target
    zero. Inequality constraints use the SciPy convention ``>= 0``.
    """

    function: Callable
    kind: str
    name: str = ""

    def __post_init__(self):
        if not callable(self.function):
            raise TypeError("constraint function must be callable.")
        if self.kind not in ("eq", "ineq"):
            raise ValueError("constraint kind must be 'eq' or 'ineq'.")
        if not isinstance(self.name, str):
            raise TypeError("constraint name must be a string.")


@dataclass(frozen=True)
class DesignConstraint:
    """One scalar constraint evaluated from a projected density field.

    The constraint is evaluated once per design region. ``function`` receives
    that region's ``rho_projected`` with shape ``(nx, ny)``. Equality
    constraints target zero; inequality constraints use the SciPy convention
    ``>= 0``.
    """

    function: Callable
    kind: str
    name: str = ""

    def __post_init__(self):
        if not callable(self.function):
            raise TypeError("design constraint function must be callable.")
        if self.kind not in ("eq", "ineq"):
            raise ValueError("design constraint kind must be 'eq' or 'ineq'.")
        if not isinstance(self.name, str):
            raise TypeError("design constraint name must be a string.")


def phase_180_constraints(
    amplitude_floor: float = 1e-6,
    power_limit: float = 1.0,
    numerical_epsilon: float = 1e-12,
) -> Tuple[ScalarConstraint, ScalarConstraint, ScalarConstraint]:
    """Return constraints for a 180-degree, nonzero, passive transmission.

    The normalized equality ``1 + Re(t)/(|t|+eps)`` is zero only when the
    complex transmission coefficient lies on the negative real axis. Its scale
    does not shrink with transmission amplitude, so reducing transmission
    cannot make the phase residual look smaller. A separate minimum-amplitude
    inequality rejects ``t=0``, where phase is undefined. The third constraint
    bounds transmittance by ``power_limit``.

    Delay and advance by 180 degrees are intentionally indistinguishable at a
    single frequency because they differ by one full optical period.
    """

    amplitude_floor = float(amplitude_floor)
    power_limit = float(power_limit)
    numerical_epsilon = float(numerical_epsilon)
    if not math.isfinite(amplitude_floor) or amplitude_floor <= 0.0:
        raise ValueError("amplitude_floor must be finite and positive.")
    if not math.isfinite(power_limit) or power_limit <= 0.0:
        raise ValueError("power_limit must be finite and positive.")
    if not math.isfinite(numerical_epsilon) or numerical_epsilon <= 0.0:
        raise ValueError("numerical_epsilon must be finite and positive.")

    return (
        ScalarConstraint(
            function=lambda transmission, *_quantities: (
                1.0
                + jnp.real(transmission)
                / (jnp.abs(transmission) + jnp.asarray(numerical_epsilon))
            ),
            kind="eq",
            name="phase_180",
        ),
        ScalarConstraint(
            function=lambda transmission, *_quantities: (
                jnp.abs(transmission) - amplitude_floor
            ),
            kind="ineq",
            name="transmission_amplitude_floor",
        ),
        ScalarConstraint(
            function=lambda transmission, *_quantities: (
                power_limit - jnp.abs(transmission) ** 2
            ),
            kind="ineq",
            name="transmission_power_upper_bound",
        ),
    )


def metal_volume_fraction_constraints(
    minimum: float = 0.0,
    maximum: Optional[float] = 1.0,
) -> Tuple[DesignConstraint, ...]:
    """Constrain ``mean(rho_projected)`` to a closed interval."""

    minimum = float(minimum)
    maximum = 1.0 if maximum is None else float(maximum)
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum volume fraction must be in [0, 1].")
    if not math.isfinite(maximum) or not 0.0 <= maximum <= 1.0:
        raise ValueError("maximum volume fraction must be in [0, 1].")
    if minimum > maximum:
        raise ValueError("minimum volume fraction exceeds maximum.")

    constraints = [
        DesignConstraint(
            function=lambda rho_projected: (
                jnp.mean(rho_projected) - minimum
            ),
            kind="ineq",
            name="metal_volume_fraction_min",
        )
    ]
    if maximum < 1.0:
        constraints.append(
            DesignConstraint(
                function=lambda rho_projected: (
                    maximum - jnp.mean(rho_projected)
                ),
                kind="ineq",
                name="metal_volume_fraction_max",
            )
        )
    return tuple(constraints)


def effective_thickness_constraints(
    minimum: float,
    maximum: float,
) -> Tuple[DesignConstraint, ...]:
    """Constrain ``sum_y(mean_x(rho_projected))`` in grid cells."""

    minimum = float(minimum)
    maximum = float(maximum)
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum < 0.0
        or minimum > maximum
    ):
        raise ValueError(
            "effective thickness bounds must be finite with 0 <= minimum <= maximum."
        )
    return (
        DesignConstraint(
            function=lambda rho_projected: (
                jnp.sum(jnp.mean(rho_projected, axis=0)) - minimum
            ),
            kind="ineq",
            name="effective_thickness_min",
        ),
        DesignConstraint(
            function=lambda rho_projected: (
                maximum - jnp.sum(jnp.mean(rho_projected, axis=0))
            ),
            kind="ineq",
            name="effective_thickness_max",
        ),
    )


def effective_area_constraints(
    minimum: float,
    maximum: float,
) -> Tuple[DesignConstraint, ...]:
    """Constrain ``sum(rho_projected)`` in grid-cell units."""

    minimum = float(minimum)
    maximum = float(maximum)
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum < 0.0
        or minimum > maximum
    ):
        raise ValueError(
            "effective area bounds must be finite with 0 <= minimum <= maximum."
        )
    return (
        DesignConstraint(
            function=lambda rho_projected: (
                jnp.sum(rho_projected) - minimum
            ),
            kind="ineq",
            name="effective_area_min",
        ),
        DesignConstraint(
            function=lambda rho_projected: (
                maximum - jnp.sum(rho_projected)
            ),
            kind="ineq",
            name="effective_area_max",
        ),
    )


def aspect_ratio_constraints(
    minimum: float,
    maximum: float,
) -> Tuple[DesignConstraint, ...]:
    """Constrain effective width divided by effective thickness."""

    minimum = float(minimum)
    maximum = float(maximum)
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum <= 0.0
        or minimum > maximum
    ):
        raise ValueError(
            "aspect ratio bounds must be finite and 0 < minimum <= maximum."
        )

    def aspect_ratio(rho_projected):
        thickness = jnp.sum(jnp.mean(rho_projected, axis=0))
        width = jnp.sum(jnp.mean(rho_projected, axis=1))
        return width / jnp.maximum(thickness, jnp.asarray(1e-12))

    return (
        DesignConstraint(
            function=lambda rho_projected: (
                aspect_ratio(rho_projected) - minimum
            ),
            kind="ineq",
            name="aspect_ratio_min",
        ),
        DesignConstraint(
            function=lambda rho_projected: (
                maximum - aspect_ratio(rho_projected)
            ),
            kind="ineq",
            name="aspect_ratio_max",
        ),
    )


__all__ = [
    "DesignConstraint",
    "ScalarConstraint",
    "aspect_ratio_constraints",
    "effective_area_constraints",
    "effective_thickness_constraints",
    "metal_volume_fraction_constraints",
    "phase_180_constraints",
]
