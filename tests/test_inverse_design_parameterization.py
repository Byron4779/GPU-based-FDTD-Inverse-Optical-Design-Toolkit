import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdz_jax.inverse_design import DesignParameters
from fdtdz_jax.inverse_design import DesignRegion
from fdtdz_jax.inverse_design import binary_straight_through
from fdtdz_jax.inverse_design import conic_filter
from fdtdz_jax.inverse_design import tanh_projection


def test_tanh_projection_matches_closed_form():
    x = np.linspace(0.0, 1.0, 9)
    beta, eta = 8.0, 0.5
    expected = (np.tanh(beta * eta) + np.tanh(beta * (x - eta))) / (
        np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    )
    actual = np.asarray(tanh_projection(x, beta, eta))
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_conic_filter_preserves_constant_field():
    x = np.full((7, 9), 0.25)
    actual = np.asarray(conic_filter(x, radius=1.5, Lx=6.0, Ly=8.0, resolution=1.0))
    np.testing.assert_allclose(actual, np.full((7, 9), 0.25), atol=1e-12)


def test_conic_filter_preserves_mass():
    rng = np.random.default_rng(3)
    x = rng.random((7, 9))
    actual = np.asarray(conic_filter(x, radius=1.5, Lx=6.0, Ly=8.0, resolution=1.0))
    np.testing.assert_allclose(actual.sum(), x.sum(), rtol=1e-10)


def test_conic_filter_periodic_x_is_shift_equivariant():
    x = np.zeros((8, 5))
    x[1, 2] = 1.0
    shifted = np.roll(x, 3, axis=0)
    actual = np.asarray(
        conic_filter(
            x,
            radius=1.5,
            Lx=7.0,
            Ly=4.0,
            resolution=1.0,
            periodic_axes=(0,),
        )
    )
    shifted_actual = np.asarray(
        conic_filter(
            shifted,
            radius=1.5,
            Lx=7.0,
            Ly=4.0,
            resolution=1.0,
            periodic_axes=(0,),
        )
    )
    np.testing.assert_allclose(
        shifted_actual, np.roll(actual, 3, axis=0), atol=1e-12
    )
    np.testing.assert_allclose(actual.sum(), x.sum(), rtol=1e-12)


def test_conic_filter_periodic_x_wraps_impulse_across_boundary():
    x = np.zeros((8, 5))
    x[0, 2] = 1.0
    actual = np.asarray(
        conic_filter(
            x,
            radius=1.5,
            Lx=7.0,
            Ly=4.0,
            resolution=1.0,
            periodic_axes=(0,),
        )
    )
    np.testing.assert_allclose(actual[-1, 2], actual[1, 2], atol=1e-12)


def test_design_parameters_normalize_and_forward_periodic_axes():
    rng = np.random.default_rng(7)
    x = rng.random((8, 5))
    params = DesignParameters(
        grid_size=x.shape,
        medium1=1.0,
        medium2=4.0,
        weights=x.reshape(-1),
        beta=0.0,
        filter_type="conic",
        radius=1.5,
        periodic_axes=[0, 0],
    )
    expected = conic_filter(
        x,
        radius=1.5,
        Lx=7.0,
        Ly=4.0,
        resolution=1.0,
        periodic_axes=(0,),
    )
    np.testing.assert_allclose(
        np.asarray(params.projected_weights()), np.asarray(expected), atol=1e-12
    )
    region = DesignRegion(params, x0=0, x1=8, y0=0, y1=5)
    assert region.spec(dt=0.2).periodic_axes == (0,)


def test_periodic_axes_reject_invalid_labels():
    with pytest.raises(ValueError, match="0 .* or 1"):
        conic_filter(
            np.ones((7, 9)),
            radius=1.5,
            Lx=6.0,
            Ly=8.0,
            resolution=1.0,
            periodic_axes=(2,),
        )


def test_periodic_conic_filter_is_differentiable():
    x = jnp.linspace(0.0, 1.0, 40, dtype=jnp.float64).reshape(8, 5)

    def objective(weights):
        filtered = conic_filter(
            weights,
            radius=1.5,
            Lx=7.0,
            Ly=4.0,
            resolution=1.0,
            periodic_axes=(0,),
        )
        return jnp.sum(filtered**2)

    gradient = jax.grad(objective)(x)
    assert gradient.shape == x.shape
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_binary_straight_through_has_exact_forward_and_identity_gradient():
    latent = jnp.asarray([0.2, 0.5, 0.8], dtype=jnp.float64)
    actual = binary_straight_through(latent, eta=0.5)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray([0.0, 1.0, 1.0]))

    weights = jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float64)
    gradient = jax.grad(
        lambda value: jnp.sum(binary_straight_through(value) * weights)
    )(latent)
    np.testing.assert_allclose(gradient, weights, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("eta", [0.0, 1.0, -0.1, float("nan")])
def test_binary_straight_through_rejects_invalid_threshold(eta):
    with pytest.raises(ValueError, match="threshold"):
        binary_straight_through(jnp.asarray([0.2, 0.8]), eta=eta)


def test_binary_design_parameterization_feeds_only_endpoint_materials():
    rng = np.random.default_rng(19)
    latent = rng.random((8, 5))
    params = DesignParameters(
        grid_size=latent.shape,
        medium1=1.0,
        medium2=4.0,
        weights=latent.reshape(-1),
        p=3.0,
        filter_type="conic",
        radius=1.5,
        periodic_axes=(0,),
        binary_material=True,
    )
    projected = np.asarray(params.projected_weights())
    np.testing.assert_array_equal(
        np.unique(projected), np.asarray([0.0, 1.0])
    )
    material = np.asarray(params.material_grid())
    np.testing.assert_array_equal(
        np.unique(material), np.asarray([1.0, 4.0])
    )

    pixel_weights = jnp.asarray(rng.normal(size=latent.shape), dtype=jnp.float64)
    gradient = jax.grad(
        lambda value: jnp.sum(params.material_grid(value) * pixel_weights)
    )(jnp.asarray(latent.reshape(-1), dtype=jnp.float64))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.linalg.norm(np.asarray(gradient)) > 0.0


def test_design_parameters_linear_interpolation():
    params = DesignParameters(
        grid_size=(3, 3),
        medium1=1.0,
        medium2=4.0,
        weights=np.full(9, 0.5),
        p=1.0,
    )
    eps = np.asarray(params.material_grid())
    np.testing.assert_allclose(eps, np.full((3, 3), 2.5))
