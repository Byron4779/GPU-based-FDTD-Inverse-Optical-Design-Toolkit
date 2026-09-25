import json
from pathlib import Path

import numpy as np
import pytest

from fdtdz_jax.material_io import read_material
from fdtdz_jax.material_io import read_materials
from fdtdz_jax.materials import Debye
from fdtdz_jax.materials import DebyePole
from fdtdz_jax.materials import Drude
from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import Lorentz
from fdtdz_jax.materials import LorentzPole
from fdtdz_jax.materials import Lossless
from fdtdz_jax.materials import Multipole


def _document():
  return {
      "schema": "fdtdz.materials",
      "schema_version": 1,
      "units": "normalized",
      "materials": [
          {"name": "air", "model": "lossless", "eps_r": 1.0},
          {"name": "debye", "model": "debye", "eps_inf": 2.0,
           "eps_static": 4.0, "tau": 0.5},
          {"name": "lorentz", "model": "lorentz", "eps_inf": 1.0,
           "eps_static": 2.25, "omega_0": 4.0, "damping": 0.28},
          {"name": "drude", "model": "drude", "eps_inf": 1.0,
           "plasma_frequency": 5.0, "collision_frequency": 0.2},
          {"name": "复合材料", "model": "multipole", "eps_inf": 1.2,
           "poles": [
               {"model": "debye", "delta_eps": 0.7, "tau": 0.4},
               {"model": "lorentz", "delta_eps": 1.1,
                "omega_0": 4.0, "damping": 0.2},
               {"model": "drude", "plasma_frequency": 2.5,
                "collision_frequency": 0.3},
           ]},
      ],
  }


def _expected():
  return {
      "air": Lossless(1.0),
      "debye": Debye(2.0, 4.0, 0.5),
      "lorentz": Lorentz(1.0, 2.25, 4.0, 0.28),
      "drude": Drude(1.0, 5.0, 0.2),
      "复合材料": Multipole(1.2, (
          DebyePole(0.7, 0.4),
          LorentzPole(1.1, 4.0, 0.2),
          DrudePole(2.5, 0.3),
      )),
  }


def test_read_json_materials_preserves_order_and_all_constants(tmp_path):
  path = tmp_path / "materials.JSON"
  path.write_text(
      json.dumps(_document(), ensure_ascii=False), encoding="utf-8")

  materials = read_materials(path)

  assert materials == _expected()
  assert list(materials) == list(_expected())
  assert read_material(path, "复合材料") == _expected()["复合材料"]


def test_read_single_material_without_name(tmp_path):
  document = _document()
  document["materials"] = document["materials"][:1]
  path = tmp_path / "one.json"
  path.write_text(json.dumps(document), encoding="utf-8")

  assert read_material(path) == Lossless(1.0)


def test_read_material_requires_name_for_library(tmp_path):
  path = tmp_path / "materials.json"
  path.write_text(json.dumps(_document()), encoding="utf-8")

  with pytest.raises(ValueError, match="name is required"):
    read_material(path)
  with pytest.raises(KeyError, match="Unknown material"):
    read_material(path, "missing")


@pytest.mark.parametrize("change,match", [
    (lambda document: document.update(schema_version=2), "schema_version"),
    (lambda document: document.update(schema_version=1.0), "schema_version"),
    (lambda document: document.update(units="SI"), "units"),
    (lambda document: document["materials"][0].update(epsilon=1.0),
     "unknown field"),
    (lambda document: document["materials"][0].pop("eps_r"),
     "missing required"),
    (lambda document: document["materials"].append(
        {"name": "air", "model": "lossless", "eps_r": 2.0}),
     "Duplicate material name"),
])
def test_json_schema_errors_identify_invalid_location(tmp_path, change, match):
  document = _document()
  change(document)
  path = tmp_path / "invalid.json"
  path.write_text(json.dumps(document), encoding="utf-8")

  with pytest.raises(ValueError, match=match):
    read_materials(path)


def test_material_constructor_validation_applies_to_external_values(tmp_path):
  document = _document()
  document["materials"][1]["tau"] = -1.0
  path = tmp_path / "invalid-constant.json"
  path.write_text(json.dumps(document), encoding="utf-8")

  with pytest.raises(ValueError, match="tau"):
    read_materials(path)


def _write_scalar(group, name, value):
  group.create_dataset(name, data=value)


def test_read_hdf5_materials_preserves_order_and_all_constants(tmp_path):
  h5py = pytest.importorskip("h5py")
  path = tmp_path / "materials.hdf5"
  document = _document()
  with h5py.File(path, "w") as archive:
    archive.attrs["schema"] = document["schema"]
    archive.attrs["schema_version"] = document["schema_version"]
    archive.attrs["units"] = document["units"]
    materials = archive.create_group("materials")
    for index, record in enumerate(document["materials"]):
      group = materials.create_group(f"{index:06d}")
      group.attrs["name"] = record["name"]
      group.attrs["model"] = record["model"]
      for name, value in record.items():
        if name in ("name", "model", "poles"):
          continue
        _write_scalar(group, name, value)
      if record["model"] == "multipole":
        poles = group.create_group("poles")
        for pole_index, pole_record in enumerate(record["poles"]):
          pole = poles.create_group(f"{pole_index:06d}")
          pole.attrs["model"] = pole_record["model"]
          for name, value in pole_record.items():
            if name != "model":
              _write_scalar(pole, name, value)

  assert read_materials(path) == _expected()


def test_hdf5_rejects_noncontiguous_material_groups(tmp_path):
  h5py = pytest.importorskip("h5py")
  path = tmp_path / "invalid.h5"
  with h5py.File(path, "w") as archive:
    archive.attrs["schema"] = "fdtdz.materials"
    archive.attrs["schema_version"] = 1
    archive.attrs["units"] = "normalized"
    group = archive.create_group("materials/000001")
    group.attrs["name"] = "air"
    group.attrs["model"] = "lossless"
    group.create_dataset("eps_r", data=1.0)

  with pytest.raises(ValueError, match="contiguous six-digit groups"):
    read_materials(path)


def test_reader_rejects_unsupported_extension(tmp_path):
  with pytest.raises(ValueError, match="Unsupported material file extension"):
    read_materials(tmp_path / "materials.txt")


def test_public_package_exports_material_readers():
  import fdtdz_jax

  assert fdtdz_jax.read_material is read_material
  assert fdtdz_jax.read_materials is read_materials


def test_bundled_rakic_gold_hdf5_matches_notebook_reference_value():
  pytest.importorskip("h5py")
  path = Path(__file__).parents[1] / "examples" / "gold_rakic_1998.h5"

  gold = read_material(path, "gold")

  assert isinstance(gold, Multipole)
  assert len(gold.poles) == 6
  epsilon = complex(gold.relative_permittivity(2.0 * np.pi * 0.06))
  assert epsilon.real == pytest.approx(-20.2777305895)
  assert epsilon.imag == pytest.approx(2.0706139183)


def test_singular_material_module_is_a_supported_compatibility_import():
  from fdtdz_jax.material import Lossless as CompatibilityLossless
  from fdtdz_jax.material import read_materials as compatibility_reader

  assert CompatibilityLossless is Lossless
  assert compatibility_reader is read_materials
