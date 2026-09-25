"""Compatibility facade for material models and external material files.

The canonical model module is :mod:`fdtdz_jax.materials` (plural).  This
facade keeps the intuitive singular import path working and also exposes the
JSON/HDF5 readers from :mod:`fdtdz_jax.material_io`.
"""

from .material_io import read_material
from .material_io import read_materials
from .materials import ADECoefficients
from .materials import Debye
from .materials import DebyePole
from .materials import Drude
from .materials import DrudePole
from .materials import Lorentz
from .materials import LorentzPole
from .materials import Lossless
from .materials import Material
from .materials import MaterialModel
from .materials import Multipole
from .materials import Pole
from .materials import ade_coefficients


__all__ = [
    "ADECoefficients",
    "Debye",
    "DebyePole",
    "Drude",
    "DrudePole",
    "Lorentz",
    "LorentzPole",
    "Lossless",
    "Material",
    "MaterialModel",
    "Multipole",
    "Pole",
    "ade_coefficients",
    "read_material",
    "read_materials",
]
