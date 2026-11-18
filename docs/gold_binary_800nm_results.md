# 800 nm gold strict-binary result

Configuration:

- 2D TMz JAX ADE;
- Rakic HDF5 gold with `epsilon_r = -20.2777 + 2.0706i` at 800 nm;
- fixed `32 x 8` full-period design region;
- periodic x conic filtering on a latent field;
- material density hard-thresholded to exactly 0 or 1 before every solve;
- 120 STE/Adam evaluations per volume limit;
- phase tolerance `0.1` degrees and maximum hard-constraint violation
  tolerance `1e-4`;
- local exact flip search disabled by default.

## Result

| Maximum Au fraction | T | Phase error | Maximum violation | Actual Au | H_eff | Valid |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.20 | 0.074835 | 42.017 degrees | 6.88e-01 | 0.1641 | 1.312 | false |
| 0.30 | 0.003961 | 12.330 degrees | 3.12e-02 | 0.2461 | 1.969 | false |
| 0.40 | 0.001154 | 1.099 degrees | 1.84e-04 | 0.2930 | 2.344 | false |

All reported material densities and every recorded material-history frame
contain only `{0, 1}`. No result is reported as a successful device.

The best case is the 40% volume limit, but it still misses the requested phase
and residual tolerances. The optimizer is therefore functioning as a strict
binary optimizer, while the current search is not yet sufficient to find a
device that passes the requested validation.

Machine-readable output:
`inverse_design_2d_gold_binary_800nm.npz`.
