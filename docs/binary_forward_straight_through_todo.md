# Strict Binary Forward with Straight-Through Gradients

Status: M1-M6 implementation complete. The default strict-binary scan is
feasible as software but does not yet meet the phase target.

## 1. Contract

The optimization must never feed an interpolated material into the FDTD/ADE
solver inside a binary design region. Every forward simulation uses only the
declared endpoint materials:

```text
rho_material in {0, 1}
```

The optimizer keeps a continuous latent variable because a strictly binary
variable has zero or undefined derivative almost everywhere. The latent is
used only to generate update directions. It is not a physical material.

```python
rho_hard = (latent >= eta).astype(latent.dtype)
rho_forward = latent + jax.lax.stop_gradient(rho_hard - latent)
```

The forward value of `rho_forward` is exactly `rho_hard`; the backward
derivative is the identity surrogate from the straight-through estimator
(STE). Every objective, phase, transmission, volume, and topology result must
be evaluated from the strictly binary forward material.

## 2. Non-negotiable invariants

- [ ] A binary design region contains no SIMP interpolation in the forward
  material fields.
- [ ] Filters may smooth latent variables, but the material passed to the
  electromagnetic solver is thresholded to exactly 0 or 1 after filtering.
- [ ] `beta` continuation is not treated as physical binarization. It may only
  guide latent updates if explicitly supported.
- [ ] Lossless, Debye/Lorentz/Drude, and Multipole materials all use exact
  endpoint fields in the forward solve.
- [ ] The air end of a dispersive/Multipole region remains on a differentiable
  path during backward evaluation so air-to-metal flips can receive proposals.
- [ ] A candidate is reported as binary only after an independent endpoint
  validation.
- [ ] The optimizer's latent is never reported as the device geometry.

## 3. Implementation checklist

### M1: Binary STE primitive

- [x] Add `binary_straight_through(value, eta)` as a JIT-compatible JAX
  primitive helper.
- [x] Validate the threshold and floating-point dtype.
- [x] Verify exact forward values of 0 and 1.
- [x] Verify a nonzero finite surrogate gradient.
- [x] Export the helper through `fdtdz_jax.inverse_design` and the top-level
  lazy API.

### M2: Binary design parameterization

- [x] Add an explicit `binary_material` option to `DesignParameters`.
- [x] Apply optional filtering to the latent, then hard-threshold the filtered
  field.
- [x] Preserve `periodic_axes` behavior through the binary path.
- [x] Forward `binary_material` through `DesignSpec` into
  `InverseDesignProblem`.
- [x] Ensure all projected-density constraints see the same hard material that
  the solver sees.
- [x] Add tests proving that `projected_weights()` and `material_grid()` contain
  only the two endpoint materials.

### M3: Endpoint material and gradient routing

- [x] Set the design-region Multipole recurrence active for both endpoint
  materials; a zero-pole air cell is equivalent to lossless air.
- [x] Use `p=1` for the binary STE surrogate. Forward values are unchanged,
  while air-to-metal updates retain a nonzero proposal gradient.
- [x] Test binary lossless end-to-end gradients.
- [x] Test binary multipole end-to-end gradients, including air cells.
- [x] Confirm that normal continuous/SIMP optimization remains unchanged.

### M4: Binary-aware optimizer

- [x] Add a dedicated optimizer entry point for hard-forward binary designs.
- [x] Use surrogate gradients only to propose latent updates.
- [x] Handle phase and other constraints with an explicit loss/penalty or a
  clearly documented constrained update strategy; do not claim exact
  differentiable SLSQP constraints for a discontinuous forward response.
- [x] Add latent dead-zone handling so a cell near the threshold cannot
  oscillate indefinitely without a structure change.
- [x] Record latent history, hard material history, physical response, and
  constraint violations separately.
- [x] Add an exact binary local-search cleanup that evaluates real FDTD
  candidates and accepts only measured improvements.

### M5: Gold notebook integration

- [x] Replace physical beta continuation with the binary-aware optimizer.
- [x] Keep periodic x filtering on the latent, not on the physical material.
- [x] Always initialize and save a strictly binary `rho_material`.
- [x] Evaluate phase, transmittance, gold fraction, thickness, feature size,
  and connectivity from that same binary array.
- [x] Rerun the default 800 nm gold case and preserve failures rather than
  relaxing tolerances.
- [x] Refresh notebook outputs and the generated `.npz` only after the binary
  pipeline is complete.

### M6: Validation and documentation

- [x] Add a regression test that fails if any forward material density is not
  exactly 0 or 1.
- [x] Add a nearest-neighbor flip test using exact binary forward evaluations.
- [x] Document that surrogate gradients are heuristic proposal signals.
- [x] Document the difference between latent, material density, and validated
  endpoint result.
- [x] Run all `inverse_design` tests, including slow constrained examples.

## 4. Acceptance criteria

- Every FDTD/ADE call in a binary region receives material density values from
  `{0, 1}` only.
- Every accepted design is backed by an independent exact binary simulation.
- The optimizer can change a cell between air and metal even though the hard
  forward operation is discontinuous.
- Continuous/SIMP paths and their existing gradient tests still pass.
- Phase and transmittance are reported from the exact binary material, never
  from the latent or a mixed endpoint interpolation.

## 5. Main risks

- STE gradients can be biased and may recommend a flip whose true response is
  worse.
- Hard thresholding can create a dead zone or rapid oscillation around `eta`.
- Volume and thickness constraints are discrete even though their values are
  easy to evaluate.
- A single-frequency objective can overfit a fragile one-cell feature.
- A binary local search is expensive and must batch exact simulations or reduce
  the discrete search space.

The forward electromagnetic result is authoritative. Surrogate gradients and
latent values are optimization diagnostics only.
