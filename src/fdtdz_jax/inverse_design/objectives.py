"""Objective quantities for the JAX ADE inverse-design module.

Mirrors Meep's ``meep.adjoint.objective.ObjectiveQuantity`` hierarchy: a
quantity extracts a value (or array) from simulation outputs, and user-provided
objective functions consume those quantities to produce scalar objectives.
"""

import abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax.numpy as jnp


class ObjectiveQuantity(abc.ABC):
    """A quantity extracted from JAX ADE simulation outputs."""

    @abc.abstractmethod
    def __call__(self, outputs):
        """Extract the quantity from a simulation-outputs mapping."""


@dataclass(frozen=True)
class ProbeTrace(ObjectiveQuantity):
    """The time-domain trace of one of the probe lines."""

    probe: str = "transmission"

    def __call__(self, outputs):
        return outputs[self.probe]


@dataclass(frozen=True)
class ProbeEnergy(ObjectiveQuantity):
    """Mean squared magnitude of a probe trace (a scalar quantity)."""

    probe: str = "transmission"
    window: Optional[Tuple[int, int]] = None

    def __call__(self, outputs):
        trace = outputs[self.probe]
        if self.window is not None:
            start, stop = self.window
            trace = trace[start:stop]
        return jnp.mean(jnp.square(jnp.abs(trace)))


def single_frequency_amplitude(
    trace,
    frequency: float,
    dt: float,
    window: Optional[Any] = None,
):
    """Evaluate a one-sided Fourier amplitude at one positive frequency.

    The transform uses ``exp(-i * 2*pi*frequency*t)`` and includes the ``dt``
    factor. ``window`` may be ``None``, ``"hann"``, or a matching one-dimensional
    weight array.
    """

    trace = jnp.asarray(trace)
    if trace.ndim != 1:
        raise ValueError("trace must be one-dimensional.")
    if trace.shape[0] < 2:
        raise ValueError("trace must contain at least two samples.")

    frequency = float(frequency)
    dt = float(dt)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be finite and positive.")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")
    if frequency > 0.5 / dt:
        raise ValueError("frequency must not exceed the Nyquist frequency.")

    if window is None:
        weights = jnp.ones(trace.shape, dtype=jnp.real(trace).dtype)
    elif isinstance(window, str):
        if window != "hann":
            raise ValueError("window string must be 'hann'.")
        weights = jnp.hanning(trace.shape[0]).astype(jnp.real(trace).dtype)
    else:
        weights = jnp.asarray(window, dtype=jnp.real(trace).dtype)
        if weights.shape != trace.shape:
            raise ValueError("window must have the same shape as trace.")

    time = jnp.arange(trace.shape[0], dtype=jnp.real(trace).dtype) * dt
    phase = jnp.exp(-2j * jnp.pi * frequency * time)
    return dt * jnp.sum(weights * trace * phase)


@dataclass(frozen=True)
class SingleFrequencyTransmission(ObjectiveQuantity):
    """Complex transmission coefficient at one frequency.

    The device probe is divided by a reference probe recorded for the same
    source, monitor plane, and time window. Samples whose reference amplitude
    is below ``reference_floor`` return zero and therefore fail the
    :func:`phase_180_constraints` amplitude condition.
    """

    frequency: float
    dt: float
    probe: str = "transmission"
    reference_probe: str = "reference_transmission"
    window: Optional[Any] = "hann"
    reference_floor: float = 1e-12

    def __call__(self, outputs):
        missing = [
            key for key in (self.probe, self.reference_probe)
            if key not in outputs
        ]
        if missing:
            raise ValueError(
                "simulation outputs are missing required probe(s): "
                + ", ".join(missing)
            )
        reference_floor = float(self.reference_floor)
        if not math.isfinite(reference_floor) or reference_floor <= 0.0:
            raise ValueError("reference_floor must be finite and positive.")

        device = single_frequency_amplitude(
            outputs[self.probe], self.frequency, self.dt, self.window
        )
        reference = single_frequency_amplitude(
            outputs[self.reference_probe], self.frequency, self.dt, self.window
        )
        valid = jnp.abs(reference) >= reference_floor
        safe_reference = jnp.where(valid, reference, 1.0 + 0.0j)
        return jnp.where(valid, device / safe_reference, 0.0 + 0.0j)


__all__ = [
    "ObjectiveQuantity",
    "ProbeEnergy",
    "ProbeTrace",
    "SingleFrequencyTransmission",
    "single_frequency_amplitude",
]
