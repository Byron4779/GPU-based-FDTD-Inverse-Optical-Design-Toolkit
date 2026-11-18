"""Validation metrics for continuous and endpoint-material designs."""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class TransmissionMetrics:
    amplitude: float
    transmittance: float
    phase_degrees: float
    phase_error_degrees: float
    phase_residual: float
    amplitude_ok: bool
    power_ok: bool


@dataclass(frozen=True)
class DensityMetrics:
    metal_fraction: float
    gray_fraction: float
    effective_area: float
    effective_width: float
    effective_thickness: float
    aspect_ratio: float


@dataclass(frozen=True)
class BinaryTopologyMetrics:
    metal_fraction: float
    air_fraction: float
    minimum_metal_feature: int
    minimum_air_feature: int
    metal_components: int
    air_components: int


def transmission_metrics(
    coefficient,
    amplitude_floor: float = 1e-4,
    target_phase_degrees: float = 180.0,
) -> TransmissionMetrics:
    """Report phase and power without hiding failed binary endpoints."""

    coefficient = complex(np.asarray(coefficient))
    amplitude = abs(coefficient)
    transmittance = amplitude**2
    phase = float(np.degrees(np.angle(coefficient)))
    phase_error = abs(
        float(
            np.degrees(
                np.angle(
                    np.exp(
                        1j
                        * (
                            np.radians(phase)
                            - np.radians(float(target_phase_degrees))
                        )
                    )
                )
            )
        )
    )
    residual = (
        math.inf
        if amplitude == 0.0
        else abs(1.0 + coefficient.real / amplitude)
    )
    return TransmissionMetrics(
        amplitude=amplitude,
        transmittance=transmittance,
        phase_degrees=phase,
        phase_error_degrees=phase_error,
        phase_residual=residual,
        amplitude_ok=amplitude >= float(amplitude_floor),
        power_ok=0.0 <= transmittance <= 1.0,
    )


def density_metrics(
    rho_projected,
    gray_bounds=(0.1, 0.9),
) -> DensityMetrics:
    """Compute volume, effective dimensions, and gray-fraction metrics."""

    density = np.asarray(rho_projected, dtype=np.float64)
    if density.ndim != 2:
        raise ValueError("rho_projected must be a two-dimensional array.")
    low, high = (float(value) for value in gray_bounds)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("gray_bounds must satisfy 0 <= low < high <= 1.")
    width = float(np.sum(np.mean(density, axis=1)))
    thickness = float(np.sum(np.mean(density, axis=0)))
    return DensityMetrics(
        metal_fraction=float(np.mean(density)),
        gray_fraction=float(np.mean((density > low) & (density < high))),
        effective_area=float(np.sum(density)),
        effective_width=width,
        effective_thickness=thickness,
        aspect_ratio=width / max(thickness, 1e-12),
    )


def _component_count(mask, periodic_x=True):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0
    if periodic_x:
        padded = np.pad(mask, ((1, 1), (0, 0)), mode="wrap")
        labels, _ = ndimage.label(padded)
        center_labels = labels[1:-1, :]
        return int(np.unique(center_labels[center_labels > 0]).size)
    _, count = ndimage.label(mask)
    return int(count)


def _largest_opening_scale(mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0
    limit = min(mask.shape)
    for scale in range(limit, 0, -1):
        structure = np.ones((scale, scale), dtype=bool)
        padded = np.pad(mask, ((scale, scale), (0, 0)), mode="wrap")
        padded = np.pad(padded, ((0, 0), (scale, scale)), mode="edge")
        opened = ndimage.binary_opening(padded, structure=structure)
        opened = opened[scale:-scale, scale:-scale]
        if np.array_equal(opened, mask):
            return int(scale)
    return 1


def binary_topology_metrics(
    rho_binary,
    metal_threshold: float = 0.5,
) -> BinaryTopologyMetrics:
    """Approximate minimum square feature size and periodic-x connectivity."""

    density = np.asarray(rho_binary, dtype=np.float64)
    if density.ndim != 2:
        raise ValueError("rho_binary must be a two-dimensional array.")
    if not np.all(np.isin(density, (0.0, 1.0))):
        raise ValueError("rho_binary must contain only endpoint values 0 and 1.")
    metal = density >= float(metal_threshold)
    air = ~metal
    return BinaryTopologyMetrics(
        metal_fraction=float(np.mean(metal)),
        air_fraction=float(np.mean(air)),
        minimum_metal_feature=_largest_opening_scale(metal),
        minimum_air_feature=_largest_opening_scale(air),
        metal_components=_component_count(metal),
        air_components=_component_count(air),
    )


def phase_error_degrees(
    coefficient,
    target_phase_degrees: float = 180.0,
) -> float:
    """Return the smallest wrapped phase error in degrees."""

    return transmission_metrics(
        coefficient, target_phase_degrees=target_phase_degrees
    ).phase_error_degrees


__all__ = [
    "BinaryTopologyMetrics",
    "DensityMetrics",
    "TransmissionMetrics",
    "binary_topology_metrics",
    "density_metrics",
    "phase_error_degrees",
    "transmission_metrics",
]
