"""JAX-differentiable ADE-coefficient construction for design regions.

This mirrors the dispersive design-region recipe in Meep's ``MaterialGrid``
(which interpolates ``LorentzianSusceptibility.sigma`` between two media): the
design density ``rho`` interpolates the ADE recurrence between a background
material and a design material.

Two representations are supported:

* the five-coefficient recurrence ``(a0, a1, a2, b1, b2)`` for ``Lossless``,
  ``Debye``, ``Lorentz``, and ``Drude``; and
* the pole-based recurrence (``multipole_mask`` / ``multipole_eps_inf`` /
  per-pole ``q``, ``r`` coefficients) for ``Multipole``.

For lossless materials the five-coefficient path degenerates to inverse-
permittivity mixing.
"""

from typing import NamedTuple

import numpy as np

import jax.numpy as jnp

from ..materials import Debye
from ..materials import Drude
from ..materials import Lorentz
from ..materials import Lossless
from ..materials import Multipole
from ..materials import ade_coefficients
from ..multipole_reference import pole_ade_coefficients


_DISPERSIVE_TYPES = (Debye, Lorentz, Drude, Multipole)


def is_dispersive(material) -> bool:
    """Return True when ``material`` has frequency-dependent ADE coefficients."""
    return isinstance(material, _DISPERSIVE_TYPES)


def is_multipole(material) -> bool:
    """Return True when ``material`` uses the pole-based ADE recurrence."""
    return isinstance(material, Multipole)


def as_eps(material) -> float:
    """Return the relative permittivity of a lossless material (or a number)."""
    if isinstance(material, Lossless):
        return float(material.eps_r)
    return float(material)


def eps_inf_of(material) -> float:
    """Return the high-frequency relative permittivity of any material/number."""
    if isinstance(material, Lossless):
        return float(material.eps_r)
    if hasattr(material, "eps_inf"):
        return float(material.eps_inf)
    return float(material)


def pole_count(material) -> int:
    """Return the number of ADE poles of a material (0 for non-Multipole)."""
    return len(material.poles) if isinstance(material, Multipole) else 0


def material_coefficients(material, dt, dtype=np.float64) -> np.ndarray:
    """Return the five ADE coefficients ``(a0, a1, a2, b1, b2)`` of a material.

    A plain number is treated as shorthand for ``Lossless(eps_r)``.  Multipole
    materials do not have a five-coefficient representation.
    """
    if isinstance(material, (int, float, np.integer, np.floating)):
        material = Lossless(float(material))
    if isinstance(material, Multipole):
        raise NotImplementedError(
            "Multipole materials use pole-based coefficients; use "
            "coefficients_from_design() instead."
        )
    return ade_coefficients(material, dt).as_array(dtype)


def pole_coefficients(material, dt, maximum_poles, dtype=np.float64) -> np.ndarray:
    """Return the ``(maximum_poles, 5)`` per-pole ``(q, r)`` matrix of a material.

    Non-Multipole materials return zeros; ``Multipole`` rows follow
    ``(q_next, q_current, q_previous, r_current, r_previous)``.
    """
    matrix = np.zeros((int(maximum_poles), 5), dtype=dtype)
    if isinstance(material, Multipole):
        for index, pole in enumerate(material.poles[: int(maximum_poles)]):
            coefficient = pole_ade_coefficients(pole, dt)
            matrix[index] = (
                coefficient.q_next,
                coefficient.q_current,
                coefficient.q_previous,
                coefficient.r_current,
                coefficient.r_previous,
            )
    return matrix


def interpolate_coefficients(background, design, density, p=3.0):
    """SIMP interpolation between two ``(5,)`` ADE coefficient vectors.

    ``density`` may be any shape; the result has shape ``density.shape + (5,)``.
    """
    if p < 1.0 or not np.isfinite(p):
        raise ValueError("penalty must be a finite value >= 1.0.")
    background = jnp.asarray(background)
    design = jnp.asarray(design)
    density = jnp.asarray(density)
    weight = jnp.clip(density, 0.0, 1.0) ** p
    return background + weight[..., None] * (design - background)


def interpolate_multipole(
    background_eps_inf,
    design_eps_inf,
    background_pole,
    design_pole,
    density,
    p=3.0,
):
    """Interpolate the pole-based recurrence between two materials.

    Returns ``(eps_inf, pole_coefficients)`` where ``eps_inf`` has
    ``density.shape`` and ``pole_coefficients`` has
    ``density.shape + (maximum_poles, 5)``.
    """
    weight = jnp.clip(jnp.asarray(density), 0.0, 1.0) ** p
    eps_inf = background_eps_inf + weight * (design_eps_inf - background_eps_inf)
    background_pole = jnp.asarray(background_pole)
    design_pole = jnp.asarray(design_pole)
    pole = background_pole + weight[..., None, None] * (
        design_pole - background_pole
    )
    return eps_inf, pole


class DesignCoefficients(NamedTuple):
    """All material fields produced for a design region."""

    coefficients: object  # (..., 5)
    multipole_mask: object  # (...,)
    multipole_eps_inf: object  # (...,)
    pole_coefficients: object  # (..., maximum_poles, 5)


def coefficients_from_design(
    density,
    background_material,
    design_material,
    dt,
    p=3.0,
    maximum_poles=None,
    dtype=np.float64,
):
    """Return all ADE material fields interpolated between two materials.

    ``background_material`` and ``design_material`` may be a ``Lossless``,
    ``Debye``, ``Lorentz``, ``Drude``, or ``Multipole`` instance, or a plain
    number (shorthand for ``Lossless``).  ``density`` maps to
    ``background_material`` at 0 and to ``design_material`` at 1.

    The result is a :class:`DesignCoefficients` namedtuple whose
    ``coefficients`` field has shape ``density.shape + (5,)`` and whose
    ``pole_coefficients`` field has shape
    ``density.shape + (maximum_poles, 5)``.
    """
    density = jnp.asarray(density)
    if is_multipole(background_material) or is_multipole(design_material):
        if maximum_poles is None:
            maximum_poles = max(
                pole_count(background_material), pole_count(design_material)
            )
        background_eps_inf = eps_inf_of(background_material)
        design_eps_inf = eps_inf_of(design_material)
        background_pole = pole_coefficients(
            background_material, dt, maximum_poles, dtype
        )
        design_pole = pole_coefficients(
            design_material, dt, maximum_poles, dtype
        )
        eps_inf, pole = interpolate_multipole(
            background_eps_inf,
            design_eps_inf,
            background_pole,
            design_pole,
            density,
            p=p,
        )
        a0 = 1.0 / eps_inf
        zeros = jnp.zeros(density.shape + (4,), dtype=a0.dtype)
        coefficients = jnp.concatenate([a0[..., None], zeros], axis=-1)
        mask = jnp.clip(density, 0.0, 1.0) ** p > 0.0
        return DesignCoefficients(coefficients, mask, eps_inf, pole)

    background = material_coefficients(background_material, dt, dtype)
    design = material_coefficients(design_material, dt, dtype)
    coefficients = interpolate_coefficients(background, design, density, p=p)
    weight = jnp.clip(density, 0.0, 1.0) ** p
    eps_inf = eps_inf_of(background_material) + weight * (
        eps_inf_of(design_material) - eps_inf_of(background_material)
    )
    maximum_poles = 0 if maximum_poles is None else int(maximum_poles)
    pole = jnp.zeros(density.shape + (maximum_poles, 5), dtype=coefficients.dtype)
    mask = jnp.zeros(density.shape, dtype=bool)
    return DesignCoefficients(coefficients, mask, eps_inf, pole)


__all__ = [
    "DesignCoefficients",
    "as_eps",
    "coefficients_from_design",
    "eps_inf_of",
    "interpolate_coefficients",
    "interpolate_multipole",
    "is_dispersive",
    "is_multipole",
    "material_coefficients",
    "pole_coefficients",
    "pole_count",
]
