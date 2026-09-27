"""2D TMz transmission inverse-design example.

This example optimizes the density of a metasurface unit cell to maximize the
power transmitted through it for a Huygens line-source pulse.  The design
region starts as a uniform, partly blocking dielectric slab; adjoint gradients
(``jax.grad`` through the FDTD time loop) drive an Adam update toward a
structure that transmits more light.

Run from a source checkout with::

    PYTHONPATH=src python examples/inverse_design_2d.py

An optional ``--plot`` flag writes ``inverse_design_2d.png`` if matplotlib is
available.
"""

import argparse
import os

import numpy as np

from fdtdz_jax.inverse_design import (
    InverseDesign2D,
    make_objective,
    run_inverse_design,
)
from fdtdz_jax.materials import Lossless
from fdtdz_jax.metasurface_2d import gaussian_sine_pulse
from fdtdz_jax.metasurface_2d import huygens_current_waveforms
from fdtdz_jax.yee_ade_2d import prepare_y_cpml


def build_problem(nx=32, ny=64, num_steps=200, design_eps=9.0,
                  frequency=0.2, dtype=np.float32):
  """Build a transmission problem with the design slab clear of the CPML."""
  dx = dy = 1.0
  dt = 0.5
  waveform = gaussian_sine_pulse(
      num_steps, dt, center_time=0.15 * num_steps, width=0.06 * num_steps,
      frequency=frequency, dtype=dtype)
  electric, magnetic = huygens_current_waveforms(
      waveform, direction=1, wave_impedance=1.0, dtype=dtype)
  cpml = prepare_y_cpml((nx, ny), width=12, dt=dt, dy=dy, dtype=dtype)
  design_mask = np.zeros((nx, ny), dtype=bool)
  design_mask[:, 28:36] = True  # a slab of cells in the middle of the cell
  return InverseDesign2D.from_materials(
      (nx, ny), design_mask, Lossless(1.0), Lossless(design_eps), dt, dx, dy,
      electric, magnetic, np.ones(nx, dtype=dtype),
      source_y=14, reflection_y=18, transmission_y=46, cpml=cpml,
      dtype=dtype), frequency


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--iterations", type=int, default=150)
  parser.add_argument("--learning-rate", type=float, default=0.1)
  parser.add_argument("--plot", action="store_true")
  args = parser.parse_args()

  problem, frequency = build_problem()
  rho0 = np.zeros(problem.shape, dtype=np.float32)
  rho0[problem.design_mask] = 0.5

  # Band-limited transmission at the source frequency (wavelength-selective).
  band = make_objective(
      problem, penalty=3.0, mode="transmission",
      frequency_min=0.8 * frequency, frequency_max=1.2 * frequency)

  def evaluate(density):
    return float(band(density))

  empty = evaluate(np.zeros(problem.shape, dtype=np.float32))
  initial = evaluate(rho0.astype(np.float32))
  print(f"band transmission @ f={frequency}:")
  print(f"  vacuum        : {empty:.4e}")
  print(f"  initial rho=0.5: {initial:.4e}")

  result = run_inverse_design(
      problem, rho0, iterations=args.iterations,
      learning_rate=args.learning_rate, penalty=3.0, mode="transmission",
      maximize=True, binarize=False,
      frequency_min=0.8 * frequency, frequency_max=1.2 * frequency)

  final = evaluate(result.density.astype(np.float32))
  print(f"  optimized     : {final:.4e}")
  print(f"  improvement   : {final / max(initial, 1e-30):.2f}x over initial")
  print(f"  objective     : {result.objective_history[0]:.4e} -> "
        f"{result.objective_history[-1]:.4e}")

  if args.plot:
    try:
      import matplotlib
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
    except ImportError:
      print("matplotlib is not installed; skipping plot.")
      return
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(result.density.T, origin="lower", cmap="viridis",
                   vmin=0.0, vmax=1.0, aspect="auto")
    axes[0].set_title("Optimized density")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y (propagation)")
    axes[1].plot(result.objective_history)
    axes[1].set_title("Objective history")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("band transmission")
    figure.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "inverse_design_2d.png")
    figure.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
  main()
