"""Read material model constants from JSON or HDF5 files.

Both formats implement version 1 of the ``fdtdz.materials`` schema documented
in ``docs/material_files.md``.  Values are passed through the material class
constructors, so external input receives the same finite-value, positivity,
and passivity validation as materials created directly in Python.
"""

import json
import numbers
import os
from pathlib import Path

from .materials import Debye
from .materials import DebyePole
from .materials import Drude
from .materials import DrudePole
from .materials import Lorentz
from .materials import LorentzPole
from .materials import Lossless
from .materials import Multipole


_SCHEMA = "fdtdz.materials"
_SCHEMA_VERSION = 1
_UNITS = "normalized"

_MATERIAL_FIELDS = {
    "lossless": ("eps_r",),
    "debye": ("eps_inf", "eps_static", "tau"),
    "lorentz": ("eps_inf", "eps_static", "omega_0", "damping"),
    "drude": ("eps_inf", "plasma_frequency", "collision_frequency"),
    "multipole": ("eps_inf", "poles"),
}
_POLE_FIELDS = {
    "debye": ("delta_eps", "tau"),
    "lorentz": ("delta_eps", "omega_0", "damping"),
    "drude": ("plasma_frequency", "collision_frequency"),
}


def _require_mapping(value, location):
  if not isinstance(value, dict):
    raise ValueError(f"{location} must be an object/group.")
  return value


def _require_exact_fields(value, required, location):
  missing = set(required) - set(value)
  extra = set(value) - set(required)
  if missing:
    raise ValueError(
        f"{location} is missing required field(s): {', '.join(sorted(missing))}.")
  if extra:
    raise ValueError(
        f"{location} has unknown field(s): {', '.join(sorted(extra))}.")


def _model_name(value, location, supported):
  if not isinstance(value, str):
    raise ValueError(f"{location} must be a string.")
  value = value.lower()
  if value not in supported:
    raise ValueError(
        f"{location} must be one of {', '.join(supported)}, got {value!r}.")
  return value


def _pole_from_record(record, location):
  record = _require_mapping(record, location)
  if "model" not in record:
    raise ValueError(f"{location} is missing required field: model.")
  model = _model_name(record["model"], f"{location}.model", _POLE_FIELDS)
  fields = _POLE_FIELDS[model]
  _require_exact_fields(record, ("model",) + fields, location)
  values = {field: record[field] for field in fields}
  constructors = {
      "debye": DebyePole,
      "lorentz": LorentzPole,
      "drude": DrudePole,
  }
  return constructors[model](**values)


def _material_from_record(record, location):
  record = _require_mapping(record, location)
  if "model" not in record:
    raise ValueError(f"{location} is missing required field: model.")
  model = _model_name(
      record["model"], f"{location}.model", _MATERIAL_FIELDS)
  fields = _MATERIAL_FIELDS[model]
  _require_exact_fields(record, ("name", "model") + fields, location)
  name = record["name"]
  if not isinstance(name, str) or not name:
    raise ValueError(f"{location}.name must be a non-empty string.")

  if model == "multipole":
    pole_records = record["poles"]
    if not isinstance(pole_records, list):
      raise ValueError(f"{location}.poles must be an array/group.")
    poles = tuple(
        _pole_from_record(pole, f"{location}.poles[{index}]")
        for index, pole in enumerate(pole_records))
    material = Multipole(eps_inf=record["eps_inf"], poles=poles)
  else:
    values = {field: record[field] for field in fields}
    constructors = {
        "lossless": Lossless,
        "debye": Debye,
        "lorentz": Lorentz,
        "drude": Drude,
    }
    material = constructors[model](**values)
  return name, material


def _validate_header(schema, version, units, location):
  if schema != _SCHEMA:
    raise ValueError(
        f"{location}.schema must be {_SCHEMA!r}, got {schema!r}.")
  if (isinstance(version, bool) or
      not isinstance(version, numbers.Integral) or
      version != _SCHEMA_VERSION):
    raise ValueError(
        f"{location}.schema_version must be {_SCHEMA_VERSION}, got {version!r}.")
  if units != _UNITS:
    raise ValueError(
        f"{location}.units must be {_UNITS!r}, got {units!r}.")


def _records_to_materials(records, location):
  if not isinstance(records, list):
    raise ValueError(f"{location} must be an array/group.")
  if not records:
    raise ValueError(f"{location} must contain at least one material.")
  result = {}
  for index, record in enumerate(records):
    name, material = _material_from_record(record, f"{location}[{index}]")
    if name in result:
      raise ValueError(f"Duplicate material name {name!r} at {location}[{index}].")
    result[name] = material
  return result


def _read_json(path):
  try:
    with path.open("r", encoding="utf-8") as stream:
      document = json.load(stream)
  except json.JSONDecodeError as error:
    raise ValueError(
        f"Invalid JSON in {path}: line {error.lineno}, column {error.colno}: "
        f"{error.msg}.") from error
  document = _require_mapping(document, "root")
  _require_exact_fields(
      document, ("schema", "schema_version", "units", "materials"), "root")
  _validate_header(
      document["schema"], document["schema_version"], document["units"],
      "root")
  return _records_to_materials(document["materials"], "root.materials")


def _decode_hdf5_string(value, location):
  if isinstance(value, bytes):
    try:
      value = value.decode("utf-8")
    except UnicodeDecodeError as error:
      raise ValueError(f"{location} must contain valid UTF-8.") from error
  if not isinstance(value, str):
    raise ValueError(f"{location} must be a UTF-8 string attribute.")
  return value


def _hdf5_scalar(group, name, location, h5py):
  if name not in group:
    raise ValueError(f"{location} is missing required dataset: {name}.")
  dataset = group[name]
  if not isinstance(dataset, h5py.Dataset) or dataset.shape != ():
    raise ValueError(f"{location}/{name} must be a scalar dataset.")
  return dataset[()]


def _ordered_hdf5_groups(group, location, h5py):
  if not isinstance(group, h5py.Group):
    raise ValueError(f"{location} must be a group.")
  keys = list(group.keys())
  expected = [f"{index:06d}" for index in range(len(keys))]
  if sorted(keys) != expected:
    raise ValueError(
        f"{location} children must be contiguous six-digit groups "
        f"000000, 000001, ..., got {sorted(keys)!r}.")
  for key in expected:
    child = group[key]
    if not isinstance(child, h5py.Group):
      raise ValueError(f"{location}/{key} must be a group.")
    yield key, child


def _hdf5_record(group, location, h5py, *, pole=False):
  allowed_models = _POLE_FIELDS if pole else _MATERIAL_FIELDS
  required_attributes = {"model"} if pole else {"name", "model"}
  attributes = set(group.attrs.keys())
  if attributes != required_attributes:
    missing = required_attributes - attributes
    extra = attributes - required_attributes
    details = []
    if missing:
      details.append(f"missing attributes {sorted(missing)!r}")
    if extra:
      details.append(f"unknown attributes {sorted(extra)!r}")
    raise ValueError(f"{location} has " + " and ".join(details) + ".")

  model = _model_name(
      _decode_hdf5_string(group.attrs["model"], f"{location}@model"),
      f"{location}@model", allowed_models)
  fields = allowed_models[model]
  expected_children = set(fields)
  if not pole and model == "multipole":
    expected_children = {"eps_inf", "poles"}
  actual_children = set(group.keys())
  if actual_children != expected_children:
    missing = expected_children - actual_children
    extra = actual_children - expected_children
    details = []
    if missing:
      details.append(f"missing children {sorted(missing)!r}")
    if extra:
      details.append(f"unknown children {sorted(extra)!r}")
    raise ValueError(f"{location} has " + " and ".join(details) + ".")

  record = {"model": model}
  if not pole:
    record["name"] = _decode_hdf5_string(
        group.attrs["name"], f"{location}@name")
  for field in fields:
    if field == "poles":
      record["poles"] = [
          _hdf5_record(child, f"{location}/poles/{key}", h5py, pole=True)
          for key, child in _ordered_hdf5_groups(
              group["poles"], f"{location}/poles", h5py)
      ]
    else:
      record[field] = _hdf5_scalar(group, field, location, h5py)
  return record


def _read_hdf5(path):
  try:
    import h5py
  except ImportError as error:
    raise ImportError(
        "Reading HDF5 material files requires h5py; install fdtdz[hdf5].") \
        from error

  with h5py.File(path, "r") as archive:
    required_attributes = {"schema", "schema_version", "units"}
    if set(archive.attrs.keys()) != required_attributes:
      raise ValueError(
          "HDF5 root attributes must be exactly schema, schema_version, and "
          "units.")
    _validate_header(
        _decode_hdf5_string(archive.attrs["schema"], "/@schema"),
        archive.attrs["schema_version"],
        _decode_hdf5_string(archive.attrs["units"], "/@units"), "/")
    if set(archive.keys()) != {"materials"}:
      raise ValueError("HDF5 root must contain exactly the /materials group.")
    records = [
        _hdf5_record(child, f"/materials/{key}", h5py)
        for key, child in _ordered_hdf5_groups(
            archive["materials"], "/materials", h5py)
    ]
  return _records_to_materials(records, "/materials")


def read_materials(path):
  """Read an ordered mapping of material names to material model instances.

  The format is selected from the case-insensitive filename extension:
  ``.json`` selects JSON and ``.h5``/``.hdf5`` select HDF5.  See
  ``docs/material_files.md`` for the versioned external-file schema and units.
  """
  if not isinstance(path, (str, os.PathLike)):
    raise TypeError("path must be a string or path-like object.")
  path = Path(path)
  suffix = path.suffix.lower()
  if suffix == ".json":
    return _read_json(path)
  if suffix in (".h5", ".hdf5"):
    return _read_hdf5(path)
  raise ValueError(
      f"Unsupported material file extension {path.suffix!r}; expected .json, "
      ".h5, or .hdf5.")


def read_material(path, name=None):
  """Read one material, optionally selecting it by name from the file."""
  materials = read_materials(path)
  if name is None:
    if len(materials) != 1:
      raise ValueError(
          "name is required when the material file contains more than one "
          f"material; available names: {', '.join(materials)}.")
    return next(iter(materials.values()))
  if not isinstance(name, str):
    raise TypeError("name must be a string or None.")
  try:
    return materials[name]
  except KeyError as error:
    raise KeyError(
        f"Unknown material {name!r}; available names: {', '.join(materials)}.") \
        from error


__all__ = ["read_material", "read_materials"]
