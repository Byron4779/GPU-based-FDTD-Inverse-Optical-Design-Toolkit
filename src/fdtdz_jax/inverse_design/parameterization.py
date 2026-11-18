"""Density-field parameterization for inverse design.

This is a JAX port of the convolution filters, projection operator, and
``MaterialGrid`` semantics from Meep's ``python/adjoint/filters.py`` and
``python/geom.py``.  The formulas follow Lazarov, Wang & Sigmund (2011/2016)
and Wang, Sell, Phan & Fan (2019); the convolution/projection layout is kept
identical to Meep so outputs match Meep bit-for-bit in float32/float64.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

import jax
import jax.numpy as jnp

from .materials_jax import as_eps
from .materials_jax import is_dispersive
from .materials_jax import is_multipole


def _normalize_periodic_axes(periodic_axes):
    """Validate and return periodic design-grid axes as an immutable tuple."""
    if periodic_axes is None:
        return None
    if isinstance(periodic_axes, (bool, np.bool_)):
        raise TypeError("periodic_axes must be an iterable of integer axis labels.")
    if isinstance(periodic_axes, (int, np.integer)):
        periodic_axes = (periodic_axes,)
    try:
        raw_axes = tuple(periodic_axes)
    except TypeError as error:
        raise TypeError(
            "periodic_axes must be an iterable of integer axis labels."
        ) from error
    if any(
        isinstance(axis, (bool, np.bool_))
        or not isinstance(axis, (int, np.integer))
        for axis in raw_axes
    ):
        raise TypeError(
            "periodic_axes must be an iterable of integer axis labels."
        )
    axes = tuple(int(axis) for axis in raw_axes)
    if any(axis not in (0, 1) for axis in axes):
        raise ValueError("periodic_axes entries must be 0 (x) or 1 (y).")
    return tuple(sorted(set(axes)))


def _centered(arr, newshape):
    """Crop the central ``newshape`` region of ``arr`` (FFT-shift helper)."""
    newshape = tuple(int(n) for n in newshape)
    startind = tuple((c - n) // 2 for c, n in zip(arr.shape, newshape))
    endind = tuple(s + n for s, n in zip(startind, newshape))
    return arr[tuple(slice(s, e) for s, e in zip(startind, endind))]


def _quarter_to_full_kernel(arr, pad_to):
    """Build the full C4v-symmetric kernel from its nonnegative quadrant."""
    pad_to = np.asarray(pad_to)
    arr = np.asarray(arr)
    pad_size = pad_to - 2 * np.asarray(arr.shape) + 1

    top = np.zeros((pad_size[0], arr.shape[1]))
    bottom = np.zeros((pad_size[0], arr.shape[1] - 1))
    middle = np.zeros((pad_to[0], pad_size[1]))

    top_left = arr[:, :]
    top_right = np.flipud(arr[1:, :])
    bottom_left = np.fliplr(arr[:, 1:])
    bottom_right = np.flipud(np.fliplr(arr[1:, 1:]))

    return np.concatenate(
        (
            np.concatenate((top_left, top, top_right)),
            middle,
            np.concatenate((bottom_left, bottom, bottom_right)),
        ),
        axis=1,
    )


def _edge_pad(arr, pad):
    """Border-pad an array with its edge values (feature preserving)."""
    (x0, x1), (y0, y1) = pad
    left = jnp.tile(arr[0, :], (x0, 1))
    right = jnp.tile(arr[-1, :], (x1, 1))
    top = jnp.tile(arr[:, 0], (y0, 1)).transpose()
    bottom = jnp.tile(arr[:, -1], (y1, 1)).transpose()

    top_left = jnp.tile(arr[0, 0], (x0, y0))
    top_right = jnp.tile(arr[-1, 0], (x1, y0))
    bottom_left = jnp.tile(arr[0, -1], (x0, y1))
    bottom_right = jnp.tile(arr[-1, -1], (x1, y1))

    if x0 > 0 and x1 > 0 and y0 > 0 and y1 > 0:
        return jnp.concatenate(
            (
                jnp.concatenate((top_left, top, top_right)),
                jnp.concatenate((left, arr, right)),
                jnp.concatenate((bottom_left, bottom, bottom_right)),
            ),
            axis=1,
        )
    if x0 == 0 and x1 == 0 and y0 > 0 and y1 > 0:
        return jnp.concatenate((top, arr, bottom), axis=1)
    if x0 > 0 and x1 > 0 and y0 == 0 and y1 == 0:
        return jnp.concatenate((left, arr, right), axis=0)
    if x0 == 0 and x1 == 0 and y0 == 0 and y1 == 0:
        return arr
    raise ValueError("At least one of the padding numbers is invalid.")


def convolve_design_weights_and_kernel(x, h, periodic_axes=None):
    """Convolve design weights with a filter kernel using an FFT.

    ``periodic_axes`` contains design-grid axes (x=0, y=1) whose convolution
    wraps around the design-region boundary. Nonperiodic axes retain the
    feature-preserving edge extension used by Meep's adjoint filters.
    """
    (sx, sy) = x.shape
    periodic_axes = _normalize_periodic_axes(periodic_axes)
    if periodic_axes is None:
        h_full = _quarter_to_full_kernel(h, 3 * np.array([sx, sy]))
        x = _edge_pad(x, ((sx, sx), (sy, sy)))
    else:
        (kx, ky) = h.shape
        npx = int(np.ceil((2 * kx - 1) / sx))
        npy = int(np.ceil((2 * ky - 1) / sy))
        if npx % 2 == 0:
            npx += 1
        if npy % 2 == 0:
            npy += 1

        x = jnp.tile(
            x,
            (
                npx if 0 in periodic_axes else 1,
                npy if 1 in periodic_axes else 1,
            ),
        )
        pad_x = 0 if 0 in periodic_axes else sx
        pad_y = 0 if 1 in periodic_axes else sy
        x = _edge_pad(x, ((pad_x, pad_x), (pad_y, pad_y)))
        h_full = _quarter_to_full_kernel(
            h,
            np.array(
                [
                    npx * sx if 0 in periodic_axes else 3 * sx,
                    npy * sy if 1 in periodic_axes else 3 * sy,
                ]
            ),
        )
    h_full = jnp.asarray(h_full, dtype=x.dtype)
    h_full = h_full / jnp.sum(h_full)
    return _centered(
        jnp.real(jnp.fft.ifft2(jnp.fft.fft2(x) * jnp.fft.fft2(h_full))),
        (sx, sy),
    )


def _get_resolution(resolution):
    if isinstance(resolution, (tuple, list, np.ndarray)):
        if len(resolution) == 2:
            return resolution
        if len(resolution) == 1:
            return resolution[0], resolution[0]
        raise ValueError("The dimension of the design-grid resolution is incorrect.")
    if isinstance(resolution, (int, float)):
        return resolution, resolution
    raise ValueError("The input for design-grid resolution is invalid.")


def mesh_grid(radius, Lx, Ly, resolution, periodic_axes=None):
    """Return design-grid point counts and the positive-quadrant coordinates."""
    periodic_axes = _normalize_periodic_axes(periodic_axes)
    resolution = _get_resolution(resolution)
    Nx = int(round(Lx * resolution[0])) + 1
    Ny = int(round(Ly * resolution[1])) + 1
    if Nx <= 1 and Ny <= 1:
        raise AssertionError(
            "The grid size is improper. Check the size and resolution of the "
            "design region."
        )
    xv = np.arange(0, Lx / 2, 1 / resolution[0]) if resolution[0] > 0 else [0]
    yv = np.arange(0, Ly / 2, 1 / resolution[1]) if resolution[1] > 0 else [0]
    if periodic_axes is not None:
        if 0 in periodic_axes:
            xv = (
                np.arange(
                    0,
                    np.ceil(2 * radius / Lx) * Lx / 2,
                    1 / resolution[0],
                )
                if resolution[0] > 0
                else [0]
            )
        if 1 in periodic_axes:
            yv = (
                np.arange(
                    0,
                    np.ceil(2 * radius / Ly) * Ly / 2,
                    1 / resolution[1],
                )
                if resolution[1] > 0
                else [0]
            )
    X, Y = np.meshgrid(xv, yv, sparse=True, indexing="ij")
    return Nx, Ny, X, Y


def cylindrical_filter(x, radius, Lx, Ly, resolution, periodic_axes=None):
    """Cylindrical convolution filter (Meep port)."""
    Nx, Ny, X, Y = mesh_grid(radius, Lx, Ly, resolution, periodic_axes)
    x = jnp.asarray(x).reshape(Nx, Ny)
    h = np.where(X**2 + Y**2 < radius**2, 1, 0)
    return convolve_design_weights_and_kernel(x, h, periodic_axes)


def conic_filter(x, radius, Lx, Ly, resolution, periodic_axes=None):
    """Linear conic (hat) convolution filter (Meep port)."""
    Nx, Ny, X, Y = mesh_grid(radius, Lx, Ly, resolution, periodic_axes)
    x = jnp.asarray(x).reshape(Nx, Ny)
    h = np.where(
        X**2 + Y**2 < radius**2,
        1 - np.sqrt(np.abs(X**2 + Y**2)) / radius,
        0,
    )
    return convolve_design_weights_and_kernel(x, h, periodic_axes)


def gaussian_filter(x, sigma, Lx, Ly, resolution, periodic_axes=None):
    """Gaussian convolution filter (Meep port)."""
    Nx, Ny, X, Y = mesh_grid(3 * sigma, Lx, Ly, resolution, periodic_axes)
    x = jnp.asarray(x).reshape(Nx, Ny)
    h = np.exp(-(X**2 + Y**2) / sigma**2)
    return convolve_design_weights_and_kernel(x, h, periodic_axes)


def tanh_projection(x, beta, eta):
    """Sigmoid (tanh) projection filter (Meep port)."""
    if beta == float("inf"):
        return jnp.where(x > eta, 1.0, 0.0)
    return (jnp.tanh(beta * eta) + jnp.tanh(beta * (x - eta))) / (
        jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta))
    )


def binary_straight_through(value, eta=0.5):
    """Return an exact binary forward value with an identity surrogate gradient.

    The forward value is 0 below ``eta`` and 1 at or above it. The backward
    pass treats the threshold as an identity so latent variables can receive
    proposal directions. The latent is not a physical material.
    """
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError("binary_straight_through requires a floating-point value.")
    eta = float(eta)
    if not np.isfinite(eta) or not 0.0 < eta < 1.0:
        raise ValueError("binary threshold eta must be finite and in (0, 1).")
    hard = (value >= eta).astype(value.dtype)
    return value + jax.lax.stop_gradient(hard - value)


def filtered_projected(
    weights,
    grid_size,
    filter_type,
    radius,
    sigma,
    dx,
    dy,
    beta,
    eta,
    periodic_axes=None,
    binary_material=False,
):
    """Filter a latent field and return the material density.

    With ``binary_material=True``, filtering is applied to the latent and the
    material density is hard-thresholded. No intermediate material is returned.
    """
    if not isinstance(binary_material, (bool, np.bool_)):
        raise TypeError("binary_material must be a boolean.")
    x = jnp.asarray(weights).reshape(grid_size)
    Lx = (grid_size[0] - 1) * dx
    Ly = (grid_size[1] - 1) * dy
    resolution = (1.0 / dx, 1.0 / dy)
    if filter_type == "conic":
        x = conic_filter(x, radius, Lx, Ly, resolution, periodic_axes)
    elif filter_type == "gaussian":
        x = gaussian_filter(x, sigma, Lx, Ly, resolution, periodic_axes)
    elif filter_type is not None:
        raise ValueError(f"Unknown filter_type: {filter_type!r}.")
    if binary_material:
        return binary_straight_through(x, eta)
    if beta:
        x = tanh_projection(x, beta, eta)
    return x


def sparse_random_design(
    grid_size: Tuple[int, int],
    target_fraction: float = 0.3,
    seed: int = 0,
    dtype=np.float64,
):
    """Return a Bernoulli air/gold seed with the requested metal fraction."""

    nx, ny = (int(value) for value in grid_size)
    target_fraction = float(target_fraction)
    if nx < 1 or ny < 1:
        raise ValueError("grid_size entries must be positive.")
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must be strictly between zero and one.")
    rng = np.random.default_rng(int(seed))
    return (rng.random((nx, ny)) < target_fraction).astype(dtype)


@dataclass
class DesignParameters:
    """Design variables and their material interpolation.

    Mirrors Meep's ``meep.MaterialGrid``: ``weights`` live in ``[0, 1]`` where
    ``0`` maps to ``medium1`` and ``1`` maps to ``medium2``.  Each ``medium``
    may be a plain number (shorthand for ``Lossless``), a ``Lossless`` instance,
    or a dispersive ``Debye``/``Lorentz``/``Drude`` instance.

    Lossless design regions map the projected density to a permittivity via SIMP
    interpolation ``eps = medium1 + u**p * (medium2 - medium1)``.  When either
    medium is dispersive, the full five-coefficient ADE recurrence is
    interpolated instead; see :mod:`fdtdz_jax.inverse_design.materials_jax`.
    """

    grid_size: Tuple[int, int]
    medium1: object
    medium2: object
    weights: Optional[np.ndarray] = None
    beta: float = 0.0
    eta: float = 0.5
    damping: float = 0.0
    p: float = 3.0
    filter_type: Optional[str] = None
    radius: float = 0.0
    sigma: float = 0.0
    dx: float = 1.0
    dy: float = 1.0
    periodic_axes: Optional[Sequence[int]] = None
    binary_material: bool = False

    def __post_init__(self):
        self.num_params = int(self.grid_size[0]) * int(self.grid_size[1])
        self.periodic_axes = _normalize_periodic_axes(self.periodic_axes)
        if not isinstance(self.binary_material, (bool, np.bool_)):
            raise TypeError("binary_material must be a boolean.")
        self.binary_material = bool(self.binary_material)
        if self.weights is None:
            self.weights = np.zeros(self.num_params, dtype=np.float64)
        else:
            self.update_weights(self.weights)

    def update_weights(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.size != self.num_params:
            raise ValueError(
                f"weights of shape {x.size} do not match user specified grid "
                f"dimension: {self.grid_size}"
            )
        self.weights = np.clip(x.reshape(-1), 0.0, 1.0)

    def update_beta(self, beta):
        self.beta = beta

    @property
    def is_dispersive(self) -> bool:
        return is_dispersive(self.medium1) or is_dispersive(self.medium2)

    @property
    def is_multipole(self) -> bool:
        return is_multipole(self.medium1) or is_multipole(self.medium2)

    def projected_weights(self, weights=None):
        if weights is None:
            weights = jnp.asarray(self.weights)
        else:
            weights = jnp.asarray(weights)
        return filtered_projected(
            weights,
            self.grid_size,
            self.filter_type,
            self.radius,
            self.sigma,
            self.dx,
            self.dy,
            self.beta,
            self.eta,
            self.periodic_axes,
            self.binary_material,
        )

    def material_grid(self, weights=None):
        if self.is_dispersive:
            raise ValueError(
                "material_grid() is for lossless design regions; use "
                "coefficients_from_design() for dispersive materials."
            )
        u = self.projected_weights(weights)
        eps1 = as_eps(self.medium1)
        eps2 = as_eps(self.medium2)
        p = 1.0 if self.binary_material else self.p
        return eps1 + u**p * (eps2 - eps1)


__all__ = [
    "DesignParameters",
    "binary_straight_through",
    "conic_filter",
    "cylindrical_filter",
    "gaussian_filter",
    "sparse_random_design",
    "tanh_projection",
]
