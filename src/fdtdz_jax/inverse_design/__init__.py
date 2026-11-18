"""Inverse-design interface for the JAX ADE solvers.

The public API mirrors Meep's ``meep.adjoint`` module (``OptimizationProblem``,
``ObjectiveQuantity``, ``DesignRegion``, ``MaterialGrid``-style design
parameters) so scripts written against Meep's adjoint interface map directly
onto this package.  The underlying gradient mechanism is JAX autodiff through
the existing differentiable ``fdtdz_jax`` Yee-ADE solvers.
"""

from .materials_jax import coefficients_from_design
from .materials_jax import interpolate_coefficients
from .constraints import DesignConstraint
from .constraints import ScalarConstraint
from .constraints import aspect_ratio_constraints
from .constraints import effective_area_constraints
from .constraints import effective_thickness_constraints
from .constraints import metal_volume_fraction_constraints
from .constraints import phase_180_constraints
from .objectives import ObjectiveQuantity
from .objectives import ProbeEnergy
from .objectives import ProbeTrace
from .objectives import SingleFrequencyTransmission
from .objectives import single_frequency_amplitude
from .optimize import minimize_lbfgsb
from .optimize import binary_flip_local_search
from .optimize import minimize_binary_ste
from .optimize import minimize_slsqp
from .optimize import minimize_slsqp_beta_continuation
from .parameterization import DesignParameters
from .parameterization import binary_straight_through
from .parameterization import conic_filter
from .parameterization import cylindrical_filter
from .parameterization import gaussian_filter
from .parameterization import sparse_random_design
from .parameterization import tanh_projection
from .problem import DesignGeometryLimits
from .problem import DesignRegion
from .problem import InverseDesignProblem
from .validation import binary_topology_metrics
from .validation import density_metrics
from .validation import phase_error_degrees
from .validation import transmission_metrics

__all__ = [
    "DesignConstraint",
    "DesignParameters",
    "DesignGeometryLimits",
    "DesignRegion",
    "InverseDesignProblem",
    "ObjectiveQuantity",
    "ProbeEnergy",
    "ProbeTrace",
    "ScalarConstraint",
    "SingleFrequencyTransmission",
    "aspect_ratio_constraints",
    "binary_straight_through",
    "binary_topology_metrics",
    "binary_flip_local_search",
    "coefficients_from_design",
    "conic_filter",
    "cylindrical_filter",
    "density_metrics",
    "effective_area_constraints",
    "effective_thickness_constraints",
    "gaussian_filter",
    "interpolate_coefficients",
    "minimize_lbfgsb",
    "minimize_binary_ste",
    "minimize_slsqp",
    "minimize_slsqp_beta_continuation",
    "metal_volume_fraction_constraints",
    "phase_180_constraints",
    "phase_error_degrees",
    "single_frequency_amplitude",
    "sparse_random_design",
    "tanh_projection",
    "transmission_metrics",
]
