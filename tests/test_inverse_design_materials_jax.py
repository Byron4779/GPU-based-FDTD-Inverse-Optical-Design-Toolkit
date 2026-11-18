import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from fdtdz_jax.inverse_design import coefficients_from_design
from fdtdz_jax.inverse_design import interpolate_coefficients
from fdtdz_jax.materials import Debye
from fdtdz_jax.materials import Drude
from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import Lorentz
from fdtdz_jax.materials import LorentzPole
from fdtdz_jax.materials import Lossless
from fdtdz_jax.materials import Multipole
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d


def test_interpolate_coefficients_endpoints_and_midpoint():
    background = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    design = np.array([0.25, 0.1, 0.2, 0.3, 0.4])
    density = np.array([0.0, 0.5, 1.0])
    out = np.asarray(interpolate_coefficients(background, design, density, p=2.0))
    np.testing.assert_allclose(out[0], background, atol=1e-14)
    np.testing.assert_allclose(out[2], design, atol=1e-14)
    np.testing.assert_allclose(out[1], background + 0.25 * (design - background), atol=1e-14)


@pytest.mark.parametrize(
    "design",
    [
        Debye(1.0, 3.0, 0.4),
        Lorentz(1.0, 3.0, 0.6, 0.1),
        Drude(1.0, 1.0, 0.1),
    ],
)
def test_coefficients_from_design_matches_numpy_grid(design):
    background = Lossless(1.0)
    dt = 0.2
    ids = np.array([[0, 1], [0, 1]])
    grid = prepare_material_grid_2d(
        [background, design], ids, dt, 1.0, 1.0, dtype=np.float64
    )
    result = coefficients_from_design(
        np.array([0.0, 1.0]), background, design, dt, p=3.0, dtype=np.float64
    )
    coeffs = np.asarray(result.coefficients)
    np.testing.assert_allclose(coeffs[0], grid.coefficients[0, 0], atol=1e-14)
    np.testing.assert_allclose(coeffs[1], grid.coefficients[0, 1], atol=1e-14)


def test_multipole_from_design_matches_numpy_grid():
    background = Lossless(1.0)
    design = Multipole(
        1.0, (LorentzPole(1.2, 3.0, 0.2), DrudePole(2.0, 0.4))
    )
    dt = 0.25
    ids = np.array([[0, 1], [0, 1]])
    grid = prepare_material_grid_2d(
        [background, design], ids, dt, 1.0, 1.0, dtype=np.float64
    )
    result = coefficients_from_design(
        np.array([0.0, 1.0]),
        background,
        design,
        dt,
        p=3.0,
        maximum_poles=grid.maximum_poles,
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        np.asarray(result.coefficients[0]), grid.coefficients[0, 0], atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(result.coefficients[1]), grid.coefficients[0, 1], atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(result.multipole_eps_inf[0]), grid.multipole_eps_inf[0, 0], atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(result.multipole_eps_inf[1]), grid.multipole_eps_inf[0, 1], atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(result.pole_coefficients[1]), grid.pole_coefficients[0, 1], atol=1e-14
    )
    assert not bool(np.asarray(result.multipole_mask[0]))
    assert bool(np.asarray(result.multipole_mask[1]))
