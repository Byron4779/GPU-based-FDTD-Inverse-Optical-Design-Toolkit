"""Optimizer drivers for :class:`InverseDesignProblem`."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

import jax
import jax.numpy as jnp
from scipy import optimize as scipy_optimize


def minimize_lbfgsb(problem, rho0, maxiter=100, bounds=(0.0, 1.0), **kwargs):
    """Minimize a problem with scipy L-BFGS-B using JAX-computed gradients.

    ``rho0`` is the initial flat design vector (one array per design region for
    multi-region problems).  Returns the scipy ``OptimizeResult`` and writes the
    final design back into the problem.
    """
    jax.config.update("jax_enable_x64", True)
    f, df = problem.get_fdf_funcs()

    rho0 = np.asarray(rho0, dtype=np.float64).reshape(-1)

    def fun(x):
        return float(np.asarray(f(x), dtype=np.float64))

    def jac(x):
        return np.asarray(df(x), dtype=np.float64).reshape(-1)

    result = scipy_optimize.minimize(
        fun,
        rho0,
        jac=jac,
        method="L-BFGS-B",
        bounds=[(float(bounds[0]), float(bounds[1]))] * rho0.size,
        options={"maxiter": int(maxiter)},
        **kwargs,
    )
    problem.update_design(result.x)
    return result


def minimize_slsqp(
    problem,
    rho0,
    maxiter=100,
    bounds=(0.0, 1.0),
    tol=1e-6,
    ftol=1e-6,
    **kwargs,
):
    """Minimize a scalar objective with nonlinear constraints using SLSQP.

    Equality constraints target zero. Inequality constraints use the SciPy
    convention ``>= 0``. Objective and constraint gradients are evaluated from
    one JAX Jacobian and cached when SciPy requests several values at the same
    design vector.
    """

    jax.config.update("jax_enable_x64", True)
    evaluator = problem.get_constrained_fdfuncs()
    rho0 = np.asarray(rho0, dtype=np.float64).reshape(-1)
    if rho0.size == 0:
        raise ValueError("rho0 must contain at least one design variable.")
    lower, upper = (float(value) for value in bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("bounds must contain finite values with lower <= upper.")

    cached_x = None
    cached_result = None
    evaluation_history = []

    def evaluate(x):
        nonlocal cached_x, cached_result
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if cached_x is not None and np.array_equal(x, cached_x):
            return cached_result
        raw = evaluator(jnp.asarray(x))
        cached_x = x.copy()
        cached_result = tuple(np.asarray(value) for value in raw)
        evaluation_history.append((cached_x.copy(), cached_result))
        return cached_result

    def objective(x):
        return float(evaluate(x)[0])

    def objective_jacobian(x):
        return evaluate(x)[1].reshape(-1)

    def equality(x):
        return evaluate(x)[2].reshape(-1)

    def equality_jacobian(x):
        return evaluate(x)[3]

    def inequality(x):
        return evaluate(x)[4].reshape(-1)

    def inequality_jacobian(x):
        return evaluate(x)[5]

    scipy_constraints = []
    num_equalities = int(evaluate(rho0)[2].size)
    num_inequalities = int(evaluate(rho0)[4].size)
    if num_equalities:
        scipy_constraints.append(
            {
                "type": "eq",
                "fun": equality,
                "jac": equality_jacobian,
            }
        )
    if num_inequalities:
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": inequality,
                "jac": inequality_jacobian,
            }
        )

    options = dict(kwargs.pop("options", {}))
    options.setdefault("maxiter", int(maxiter))
    options.setdefault("ftol", float(ftol))
    result = scipy_optimize.minimize(
        objective,
        rho0,
        jac=objective_jacobian,
        method="SLSQP",
        bounds=[(lower, upper)] * rho0.size,
        constraints=scipy_constraints,
        tol=float(tol),
        options=options,
        **kwargs,
    )
    problem.update_design(result.x)
    final = evaluate(result.x)
    result.objective_value = float(final[0])
    result.constraint_values = {
        "eq": final[2].copy(),
        "ineq": final[4].copy(),
    }
    if hasattr(problem, "constraint_names"):
        result.constraint_names = {
            "eq": problem.constraint_names("eq"),
            "ineq": problem.constraint_names("ineq"),
        }
    max_violation = 0.0
    if final[2].size:
        max_violation = max(
            max_violation, float(np.max(np.abs(final[2])))
        )
    if final[4].size:
        max_violation = max(
            max_violation, float(np.max(np.maximum(-final[4], 0.0)))
        )
    result.max_constraint_violation = max_violation
    result.design_history = np.stack(
        [entry[0] for entry in evaluation_history]
    )
    result.objective_history = np.asarray(
        [float(entry[1][0]) for entry in evaluation_history]
    )
    return result


@dataclass(frozen=True)
class BetaStageResult:
    """Metrics recorded after one differentiable-projection stage."""

    beta: float
    optimization_result: object
    projected_densities: tuple
    gray_fraction: float
    metal_volume_fraction: float
    effective_thickness: float
    max_constraint_violation: float
    objective_value: float


@dataclass(frozen=True)
class BetaContinuationResult:
    """Stages produced by a beta continuation, in increasing-beta order."""

    stages: tuple

    @property
    def final_stage(self) -> BetaStageResult:
        if not self.stages:
            raise ValueError("continuation contains no stages.")
        return self.stages[-1]

    @property
    def final_optimization_result(self):
        return self.final_stage.optimization_result


@dataclass(frozen=True)
class BinaryStepRecord:
    """One hard-forward binary optimization evaluation."""

    step: int
    latent: np.ndarray
    binary_densities: tuple
    objective_value: float
    penalized_objective: float
    max_constraint_violation: float


@dataclass(frozen=True)
class BinaryOptimizationResult:
    """Result from binary STE optimization.

    ``latent`` is an optimization proposal. ``binary_densities`` are the exact
    endpoint materials used by the selected forward simulation.
    """

    latent: np.ndarray
    binary_densities: tuple
    best_step: int
    objective_value: float
    max_constraint_violation: float
    feasible: bool
    history: tuple


@dataclass(frozen=True)
class BinaryFlipResult:
    """Result from exact binary latent-flip local search."""

    latent: np.ndarray
    binary_densities: tuple
    objective_value: float
    max_constraint_violation: float
    penalized_objective: float
    accepted_flips: tuple
    improved: bool


def _binary_threshold_vector(problem):
    """Return one threshold per latent variable for binary design regions."""
    thresholds = []
    for region in problem.design_regions:
        params = region.design_parameters
        if not params.binary_material:
            raise ValueError(
                "binary optimizers require every design region to set "
                "binary_material=True."
            )
        thresholds.extend([float(params.eta)] * params.num_params)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.size == 0:
        raise ValueError("problem must contain at least one design variable.")
    return thresholds


def _binary_projected_densities(problem, latent):
    """Evaluate and validate the exact endpoint material seen by the solver."""
    if not hasattr(problem, "evaluate_projected_densities"):
        raise TypeError(
            "binary optimization requires evaluate_projected_densities()."
        )
    densities = tuple(
        np.asarray(density, dtype=np.float64)
        for density in problem.evaluate_projected_densities(latent)
    )
    if not densities:
        raise ValueError("problem returned no projected densities.")
    for density in densities:
        if not np.all((density == 0.0) | (density == 1.0)):
            raise RuntimeError(
                "binary forward produced a material density outside {0, 1}."
            )
    return densities


def _constraint_penalty_terms(objective, equality, inequality, penalty):
    equality = np.asarray(equality, dtype=np.float64).reshape(-1)
    inequality = np.asarray(inequality, dtype=np.float64).reshape(-1)
    inequality_violation = np.minimum(inequality, 0.0)
    violation = 0.0
    if equality.size:
        violation = max(violation, float(np.max(np.abs(equality))))
    if inequality.size:
        violation = max(violation, float(np.max(-inequality_violation)))
    auxiliary = (
        float(np.sum(equality**2))
        + float(np.sum(inequality_violation**2))
    )
    return (
        float(objective) + float(penalty) * auxiliary,
        violation,
    )


def _penalty_objective_and_gradient(
    objective,
    objective_gradient,
    equality,
    equality_jacobian,
    inequality,
    inequality_jacobian,
    penalty,
):
    equality = np.asarray(equality, dtype=np.float64).reshape(-1)
    equality_jacobian = np.asarray(equality_jacobian, dtype=np.float64)
    inequality = np.asarray(inequality, dtype=np.float64).reshape(-1)
    inequality_jacobian = np.asarray(inequality_jacobian, dtype=np.float64)
    violation = np.minimum(inequality, 0.0)
    loss, max_violation = _constraint_penalty_terms(
        objective, equality, inequality, penalty
    )
    gradient = np.asarray(
        objective_gradient, dtype=np.float64
    ).reshape(-1).copy()
    if equality.size:
        gradient += float(penalty) * 2.0 * equality @ equality_jacobian
    if inequality.size:
        gradient += float(penalty) * 2.0 * violation @ inequality_jacobian
    return loss, max_violation, gradient


def _enforce_threshold_margin(latent, threshold, gradient, margin):
    """Push latent values just outside a narrow threshold dead zone."""
    if margin <= 0.0:
        return latent
    distance = latent - threshold
    near_threshold = np.abs(distance) < margin
    if not np.any(near_threshold):
        return latent
    direction = np.where(gradient < 0.0, -1.0, 1.0)
    adjusted = latent.copy()
    adjusted[near_threshold] = (
        threshold[near_threshold] + direction[near_threshold] * margin
    )
    return np.clip(adjusted, 0.0, 1.0)


def minimize_binary_ste(
    problem,
    latent0,
    *,
    steps=300,
    learning_rate=1e-2,
    constraint_penalty=10.0,
    beta1=0.9,
    beta2=0.999,
    adam_epsilon=1e-8,
    threshold_margin=1e-3,
    tolerance=1e-4,
):
    """Optimize a strictly binary forward problem with STE proposal gradients.

    Every objective and constraint evaluation uses the hard endpoint material
    supplied by ``problem.evaluate_projected_densities()``. The returned latent
    is not a physical material; the returned densities are the actual binary
    materials.
    """
    steps = int(steps)
    learning_rate = float(learning_rate)
    constraint_penalty = float(constraint_penalty)
    threshold_margin = float(threshold_margin)
    tolerance = float(tolerance)
    for name, value in (
        ("learning_rate", learning_rate),
        ("constraint_penalty", constraint_penalty),
        ("threshold_margin", threshold_margin),
        ("tolerance", tolerance),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if steps < 1:
        raise ValueError("steps must be a positive integer.")
    if not 0.0 < learning_rate:
        raise ValueError("learning_rate must be positive.")
    if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
        raise ValueError("beta1 and beta2 must lie in (0, 1).")
    if not np.isfinite(adam_epsilon) or adam_epsilon <= 0.0:
        raise ValueError("adam_epsilon must be finite and positive.")

    threshold = _binary_threshold_vector(problem)
    if threshold_margin >= float(np.min(
        np.minimum(threshold, 1.0 - threshold)
    )):
        raise ValueError(
            "threshold_margin must be smaller than each threshold's distance "
            "to 0 and 1."
        )
    latent = np.asarray(latent0, dtype=np.float64).reshape(-1)
    if latent.size != threshold.size:
        raise ValueError(
            "latent0 must contain one value per binary design variable."
        )
    latent = np.clip(latent, 0.0, 1.0)
    evaluator = problem.get_constrained_fdfuncs()
    first_moment = np.zeros_like(latent)
    second_moment = np.zeros_like(latent)
    history = []
    best = None

    def evaluate_and_record(step):
        raw = evaluator(latent)
        objective = float(np.asarray(raw[0], dtype=np.float64))
        objective_gradient = np.asarray(
            raw[1], dtype=np.float64
        ).reshape(-1)
        equality = np.asarray(raw[2], dtype=np.float64).reshape(-1)
        equality_jacobian = np.asarray(raw[3], dtype=np.float64)
        inequality = np.asarray(raw[4], dtype=np.float64).reshape(-1)
        inequality_jacobian = np.asarray(raw[5], dtype=np.float64)
        densities = _binary_projected_densities(problem, latent)
        loss, violation, gradient = _penalty_objective_and_gradient(
            objective,
            objective_gradient,
            equality,
            equality_jacobian,
            inequality,
            inequality_jacobian,
            constraint_penalty,
        )
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError(
                "binary STE evaluation produced a nonfinite objective or gradient."
            )
        record = BinaryStepRecord(
            step=step,
            latent=latent.copy(),
            binary_densities=tuple(density.copy() for density in densities),
            objective_value=objective,
            penalized_objective=loss,
            max_constraint_violation=violation,
        )
        history.append(record)
        return record, gradient

    for step in range(steps + 1):
        record, gradient = evaluate_and_record(step)
        if best is None or (
            record.penalized_objective,
            record.max_constraint_violation,
        ) < (
            best.penalized_objective,
            best.max_constraint_violation,
        ):
            best = record
        if step == steps:
            break
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = (
            beta2 * second_moment + (1.0 - beta2) * gradient**2
        )
        first_hat = first_moment / (1.0 - beta1 ** (step + 1))
        second_hat = second_moment / (1.0 - beta2 ** (step + 1))
        latent = latent - learning_rate * first_hat / (
            np.sqrt(second_hat) + adam_epsilon
        )
        latent = np.clip(latent, 0.0, 1.0)
        latent = _enforce_threshold_margin(
            latent, threshold, gradient, threshold_margin
        )

    problem.update_design(best.latent)
    return BinaryOptimizationResult(
        latent=best.latent.copy(),
        binary_densities=tuple(
            density.copy() for density in best.binary_densities
        ),
        best_step=best.step,
        objective_value=best.objective_value,
        max_constraint_violation=best.max_constraint_violation,
        feasible=best.max_constraint_violation <= tolerance,
        history=tuple(history),
    )


def binary_flip_local_search(
    problem,
    latent0,
    *,
    iterations=20,
    candidate_count=None,
    constraint_penalty=10.0,
    tolerance=1e-4,
    seed=0,
):
    """Greedily flip latent coordinates and accept only measured improvements.

    Every candidate is evaluated with the hard forward binary material. If a
    filter is active, one latent flip can change a local neighborhood; the
    accepted material is therefore recorded from the projected density rather
    than inferred from the flipped coordinate.
    """
    iterations = int(iterations)
    constraint_penalty = float(constraint_penalty)
    tolerance = float(tolerance)
    if iterations < 0:
        raise ValueError("iterations must be nonnegative.")
    if not np.isfinite(constraint_penalty) or constraint_penalty < 0.0:
        raise ValueError("constraint_penalty must be finite and nonnegative.")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")

    threshold = _binary_threshold_vector(problem)
    latent = np.asarray(latent0, dtype=np.float64).reshape(-1)
    if latent.size != threshold.size:
        raise ValueError(
            "latent0 must contain one value per binary design variable."
        )
    latent = np.clip(latent, 0.0, 1.0)
    if candidate_count is None:
        candidate_count = latent.size
    candidate_count = int(candidate_count)
    if not 1 <= candidate_count <= latent.size:
        raise ValueError(
            "candidate_count must be in [1, number of design variables]."
        )
    evaluator = problem.get_constrained_fdfuncs()
    rng = np.random.default_rng(int(seed))
    accepted_flips = []

    def evaluate(candidate):
        raw = evaluator(candidate)
        objective = float(np.asarray(raw[0], dtype=np.float64))
        densities = _binary_projected_densities(problem, candidate)
        loss, violation = _constraint_penalty_terms(
            objective,
            raw[2],
            raw[4],
            constraint_penalty,
        )
        if not np.isfinite(loss):
            raise FloatingPointError(
                "binary local search produced a nonfinite objective."
            )
        return loss, objective, violation, densities

    current_loss, current_objective, current_violation, current_densities = (
        evaluate(latent)
    )
    improved = False
    for _ in range(iterations):
        indices = rng.choice(
            latent.size,
            size=min(candidate_count, latent.size),
            replace=False,
        )
        best_candidate = None
        best_index = None
        for index in indices:
            candidate = latent.copy()
            candidate[index] = (
                0.0 if latent[index] >= threshold[index] else 1.0
            )
            result = evaluate(candidate)
            if best_candidate is None or result[0] < best_candidate[1][0]:
                best_candidate = (candidate, result)
                best_index = int(index)
        if best_candidate is None or best_candidate[1][0] >= current_loss:
            break
        latent = best_candidate[0]
        (
            current_loss,
            current_objective,
            current_violation,
            current_densities,
        ) = best_candidate[1]
        accepted_flips.append(best_index)
        improved = True

    problem.update_design(latent)
    return BinaryFlipResult(
        latent=latent.copy(),
        binary_densities=tuple(
            density.copy() for density in current_densities
        ),
        objective_value=current_objective,
        max_constraint_violation=current_violation,
        penalized_objective=current_loss,
        accepted_flips=tuple(accepted_flips),
        improved=improved,
    )


def minimize_slsqp_beta_continuation(
    problem,
    rho0,
    betas: Sequence[float],
    *,
    callback: Optional[Callable[[BetaStageResult], None]] = None,
    **slsqp_kwargs,
) -> BetaContinuationResult:
    """Run hard-constrained SLSQP through increasing projection strengths.

    Each stage starts from the previous stage's density. The problem is rebuilt
    at the new beta so both the material interpolation and density constraints
    use the same differentiable projection. Failure at one stage is preserved
    in its optimization result and does not silently change tolerances.
    """

    beta_values = tuple(float(beta) for beta in betas)
    if not beta_values or any(
        not np.isfinite(beta) or beta < 0.0 for beta in beta_values
    ):
        raise ValueError("betas must be a nonempty sequence of finite values >= 0.")
    if any(b2 <= b1 for b1, b2 in zip(beta_values, beta_values[1:])):
        raise ValueError("betas must be strictly increasing.")

    rho = np.asarray(rho0, dtype=np.float64).reshape(-1)
    stages = []
    for beta in beta_values:
        problem.update_beta(beta)
        result = minimize_slsqp(problem, rho, **slsqp_kwargs)
        projected = problem.evaluate_projected_densities(result.x)
        if not projected:
            raise ValueError("problem has no design regions.")
        gray_fraction = float(np.mean([
            np.mean((density > 0.1) & (density < 0.9))
            for density in projected
        ]))
        metal_fraction = float(np.mean([
            np.mean(density) for density in projected
        ]))
        thickness = float(np.mean([
            np.sum(np.mean(density, axis=0)) for density in projected
        ]))
        stage = BetaStageResult(
            beta=beta,
            optimization_result=result,
            projected_densities=projected,
            gray_fraction=gray_fraction,
            metal_volume_fraction=metal_fraction,
            effective_thickness=thickness,
            max_constraint_violation=result.max_constraint_violation,
            objective_value=result.objective_value,
        )
        stages.append(stage)
        if callback is not None:
            callback(stage)
        rho = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    return BetaContinuationResult(stages=tuple(stages))


__all__ = [
    "BetaContinuationResult",
    "BetaStageResult",
    "BinaryFlipResult",
    "BinaryOptimizationResult",
    "BinaryStepRecord",
    "binary_flip_local_search",
    "minimize_binary_ste",
    "minimize_lbfgsb",
    "minimize_slsqp",
    "minimize_slsqp_beta_continuation",
]
