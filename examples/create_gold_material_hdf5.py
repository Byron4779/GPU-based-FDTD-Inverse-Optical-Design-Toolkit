"""Create the normalized Rakić-1998 Lorentz-Drude gold material file."""

from pathlib import Path
import sys

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(REPO_ROOT / "src")
if SOURCE_ROOT not in sys.path:
  sys.path.insert(0, SOURCE_ROOT)

from fdtdz_jax import PhysicalScale  # noqa: E402


LENGTH_UNIT_M = 48e-9
GOLD_EPS_INF = 1.0
GOLD_PLASMA_ENERGY_EV = 9.03
GOLD_DRUDE_STRENGTH = 0.760
GOLD_DRUDE_WIDTH_EV = 0.053
GOLD_LORENTZ_PARAMETERS = (
    # (oscillator strength f_j, resonance energy hbar*omega_j in eV,
    #  full width hbar*Gamma_j in eV)
    (0.024, 0.415, 0.241),
    (0.010, 0.830, 0.345),
    (0.071, 2.969, 0.870),
    (0.601, 4.304, 2.494),
    (4.384, 13.32, 2.214),
)


def create_gold_material_file(path):
  """Write one ``gold`` Multipole material in fdtdz.materials schema v1."""
  path = Path(path)
  scale = PhysicalScale(length_unit_m=LENGTH_UNIT_M)
  normalized_omega = scale.angular_frequency_from_ev

  with h5py.File(path, "w") as archive:
    archive.attrs["schema"] = "fdtdz.materials"
    archive.attrs["schema_version"] = 1
    archive.attrs["units"] = "normalized"

    material = archive.create_group("materials/000000")
    material.attrs["name"] = "gold"
    material.attrs["model"] = "multipole"
    material.create_dataset("eps_inf", data=GOLD_EPS_INF)
    poles = material.create_group("poles")

    drude = poles.create_group("000000")
    drude.attrs["model"] = "drude"
    drude.create_dataset(
        "plasma_frequency",
        data=float(normalized_omega(
            np.sqrt(GOLD_DRUDE_STRENGTH) * GOLD_PLASMA_ENERGY_EV)))
    drude.create_dataset(
        "collision_frequency",
        data=float(normalized_omega(GOLD_DRUDE_WIDTH_EV)))

    for index, (strength, resonance_ev, width_ev) in enumerate(
        GOLD_LORENTZ_PARAMETERS, start=1):
      pole = poles.create_group(f"{index:06d}")
      pole.attrs["model"] = "lorentz"
      pole.create_dataset(
          "delta_eps",
          data=strength * (GOLD_PLASMA_ENERGY_EV / resonance_ev)**2)
      pole.create_dataset(
          "omega_0", data=float(normalized_omega(resonance_ev)))
      # The solver uses delta = Gamma / 2 for Lorentz damping.
      pole.create_dataset(
          "damping", data=float(normalized_omega(width_ev / 2.0)))
  return path


if __name__ == "__main__":
  output = create_gold_material_file(REPO_ROOT / "examples" /
                                     "gold_rakic_1998.h5")
  print("Created:", output)
