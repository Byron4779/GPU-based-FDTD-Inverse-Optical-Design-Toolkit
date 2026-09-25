# External material files

`read_materials(path)` loads every material in a JSON or HDF5 file and returns
an insertion-ordered `dict[str, Material]`. `read_material(path, name)` selects
one entry; its `name` argument may be omitted only when the file contains one
material. The filename extension selects the reader: `.json`, `.h5`, or
`.hdf5` (case-insensitive).

```python
from fdtdz_jax import read_material, read_materials

materials_by_name = read_materials("materials.json")
materials = list(materials_by_name.values())
metal = read_material("materials.h5", "metal")
```

The package-level imports above are preferred. Direct imports from
`fdtdz_jax.material_io` are also supported. For compatibility,
`fdtdz_jax.material` (singular) re-exports both the readers and all material
model classes; the canonical model module remains `fdtdz_jax.materials`.

Both file types use schema `fdtdz.materials`, version `1`, and normalized
solver units. Relative permittivities (`eps_r`, `eps_inf`, `eps_static`, and
`delta_eps`) are dimensionless. `tau` uses the same normalized time unit as
the simulation `dt`. `omega_0`, `plasma_frequency`, `collision_frequency`, and
`damping` are angular frequencies in inverse normalized time. In particular,
`damping` is the Lorentz delta for a differential term `2 * damping * d/dt`,
and `collision_frequency` is the Drude gamma. The loader does not convert SI,
Hz, eV, or wavelength values; use `PhysicalScale` before writing the file when
conversion is needed.

Material names are case-sensitive. Model strings are case-insensitive and the
loader normalizes them to lower case. Unknown or missing fields are rejected so
a misspelled constant cannot silently change a simulation. Loaded values also
pass through the normal material validation rules.

## JSON layout

The root object has exactly these four keys:

- `schema`: the string `"fdtdz.materials"`.
- `schema_version`: the integer `1`.
- `units`: the string `"normalized"`.
- `materials`: a non-empty array. Array position is the material order used by
  `list(read_materials(path).values())` and therefore can define material IDs.

Each material is an object with `name`, `model`, and the constants shown below.
A multipole material also contains a `poles` array; its array order is the pole
order.

```json
{
  "schema": "fdtdz.materials",
  "schema_version": 1,
  "units": "normalized",
  "materials": [
    {"name": "air", "model": "lossless", "eps_r": 1.0},
    {
      "name": "water_like",
      "model": "debye",
      "eps_inf": 2.0,
      "eps_static": 4.0,
      "tau": 0.5
    },
    {
      "name": "resonator",
      "model": "lorentz",
      "eps_inf": 1.0,
      "eps_static": 2.25,
      "omega_0": 4.0,
      "damping": 0.28
    },
    {
      "name": "simple_metal",
      "model": "drude",
      "eps_inf": 1.0,
      "plasma_frequency": 5.0,
      "collision_frequency": 0.2
    },
    {
      "name": "fitted_metal",
      "model": "multipole",
      "eps_inf": 1.2,
      "poles": [
        {"model": "debye", "delta_eps": 0.7, "tau": 0.4},
        {
          "model": "lorentz",
          "delta_eps": 1.1,
          "omega_0": 4.0,
          "damping": 0.2
        },
        {
          "model": "drude",
          "plasma_frequency": 2.5,
          "collision_frequency": 0.3
        }
      ]
    }
  ]
}
```

The exact constants required for each `model` are:

| Model | Constants at the material-object level |
|---|---|
| `lossless` | `eps_r` |
| `debye` | `eps_inf`, `eps_static`, `tau` |
| `lorentz` | `eps_inf`, `eps_static`, `omega_0`, `damping` |
| `drude` | `eps_inf`, `plasma_frequency`, `collision_frequency` |
| `multipole` | `eps_inf`, `poles` |

Within a multipole `poles` entry:

| Pole model | Constants at the pole-object level |
|---|---|
| `debye` | `delta_eps`, `tau` |
| `lorentz` | `delta_eps`, `omega_0`, `damping` |
| `drude` | `plasma_frequency`, `collision_frequency` |

## HDF5 layout

The HDF5 root has exactly three attributes and one child group:

```text
/@schema = "fdtdz.materials"           (UTF-8 string attribute)
/@schema_version = 1                   (integer attribute)
/@units = "normalized"                (UTF-8 string attribute)
/materials                             (group)
```

Material entries are contiguous, zero-based, six-digit groups. This makes the
order explicit and stable: `/materials/000000`, `/materials/000001`, and so
on. Each entry stores `name` and `model` as UTF-8 attributes, while every
constant is a numeric scalar dataset:

```text
/materials/000000@name = "air"
/materials/000000@model = "lossless"
/materials/000000/eps_r = 1.0

/materials/000001@name = "simple_metal"
/materials/000001@model = "drude"
/materials/000001/eps_inf = 1.0
/materials/000001/plasma_frequency = 5.0
/materials/000001/collision_frequency = 0.2
```

For a multipole entry, `eps_inf` is a scalar dataset and `poles` is a group.
Pole groups use the same contiguous six-digit numbering. A pole has only the
`model` attribute (no `name`) and its constants are scalar datasets:

```text
/materials/000002@name = "fitted_metal"
/materials/000002@model = "multipole"
/materials/000002/eps_inf = 1.2
/materials/000002/poles/000000@model = "lorentz"
/materials/000002/poles/000000/delta_eps = 1.1
/materials/000002/poles/000000/omega_0 = 4.0
/materials/000002/poles/000000/damping = 0.2
```

The material and pole tables in the JSON section define the exact scalar
datasets for every HDF5 model as well. HDF5 support is optional; install it
with `python -m pip install ".[hdf5]"`.

This example creates a valid one-material HDF5 file:

```python
import h5py

with h5py.File("material.h5", "w") as f:
    f.attrs["schema"] = "fdtdz.materials"
    f.attrs["schema_version"] = 1
    f.attrs["units"] = "normalized"
    material = f.create_group("materials/000000")
    material.attrs["name"] = "glass"
    material.attrs["model"] = "lossless"
    material.create_dataset("eps_r", data=2.25)
```

The repository also includes a complete dispersive example:

- `examples/gold_rakic_1998.h5` contains normalized Rakić-1998 gold data.
- `examples/create_gold_material_hdf5.py` reproducibly rebuilds that file.
- `complex-permittivity-metasurface-3d-hdf5.ipynb` loads the file and plots Ex
  using the same source, geometry, and monitor planes as the original
  `complex-permittivity-metasurface-3d.ipynb`.
