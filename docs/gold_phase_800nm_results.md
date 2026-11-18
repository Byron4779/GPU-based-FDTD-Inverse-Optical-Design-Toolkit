# Historical 800 nm gold phase result: continuous-material scan

This document records the pre-strict-binary beta-continuation experiment.
Those continuous results are not valid binary devices and are superseded by the
strict-binary workflow in `inverse_design_2d_gold_hdf5.ipynb`. The current
notebook always evaluates exact 0/1 endpoint materials.

Configuration:

- 2D TMz JAX ADE solver;
- 48 nm physical length unit;
- 800 nm wavelength, normalized frequency 0.06;
- HDF5 Rakic gold, `epsilon_r = -20.2777 + 2.0706i`;
- fixed `32 x 8` full-period design canvas;
- beta continuation `(2, 4, 8, 16, 32)`;
- volume limits `(0.2, 0.3, 0.4)`;
- effective thickness hard bounds `2 <= H_eff <= 4`;
- 180-degree hard phase constraint with `0.1` degree validation tolerance.

## Continuous projected-density results

| Maximum Au fraction | T | Phase | Maximum hard-constraint violation | Validation |
| ---: | ---: | ---: | ---: | --- |
| 0.20 | 0.143010 | +113.124 degrees | 6.07e-1 | failed |
| 0.30 | 0.010799 | +179.945 degrees | 4.69e-7 | passed |
| 0.40 | 0.012558 | +177.197 degrees | 1.20e-3 | failed |

The 30% case satisfies volume, effective-thickness, phase, amplitude, and power
constraints at the final beta. The 20% case is infeasible with the selected
local solution and effective thickness near 1.61 cells. The 40% case misses the
phase threshold after endpoint-oriented continuation.

## Independent binary endpoint results

Each projected density was thresholded at `eta=0.5`, copied into a new
`beta=0`, unfiltered endpoint-material problem, and simulated again.

| Maximum Au fraction | Actual Au fraction | T | Phase error | Minimum feature | Metal components | Validation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.20 | 0.2031 | 0.155931 | 70.044 degrees | 1 cell | 5 | failed |
| 0.30 | 0.2852 | 0.202723 | 87.440 degrees | 1 cell | 6 | failed |
| 0.40 | 0.3945 | 0.098183 | 46.339 degrees | 1 cell | 3 | failed |

No binary gold endpoint design passed the requested 180-degree validation.
The result is therefore a feasibility result, not a successful device. The
notebook preserves both the passing 30% continuous relaxation and all failing
endpoint simulations without relaxing the hard tolerances.

The machine-readable output is
`inverse_design_2d_gold_phase_800nm.npz`.
