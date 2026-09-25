"""Execute plain-Python notebook code cells in the current process.

This lightweight validator is useful with ``run_wsl_gpu.py``, whose explicit
CUDA driver initialization must remain in the same process as JAX. Notebooks
containing IPython magics or shell escapes are intentionally unsupported.
"""

import argparse
import base64
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from io import BytesIO
from io import StringIO
import json
import os
from pathlib import Path
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("notebook", type=Path)
  parser.add_argument(
      "--write-outputs", action="store_true",
      help="Store captured stdout/stderr and newly created figures in place.")
  args = parser.parse_args()
  notebook_path = args.notebook.resolve()
  notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
  namespace = {"__name__": "__main__"}
  previous_directory = Path.cwd()
  os.chdir(notebook_path.parent)
  try:
    execution_count = 0
    for index, cell in enumerate(notebook["cells"]):
      if cell["cell_type"] != "code":
        continue
      execution_count += 1
      source = "".join(cell["source"])
      print(f"Executing code cell {index}...", flush=True)
      figures_before = set()
      if "matplotlib.pyplot" in sys.modules:
        import matplotlib.pyplot as plt
        figures_before = set(plt.get_fignums())
      stdout = StringIO()
      stderr = StringIO()
      with redirect_stdout(stdout), redirect_stderr(stderr):
        exec(compile(source, f"{notebook_path.name}:cell-{index}", "exec"),
             namespace)
      print(stdout.getvalue(), end="", flush=True)
      print(stderr.getvalue(), end="", file=sys.stderr, flush=True)

      if args.write_outputs:
        outputs = []
        if stdout.getvalue():
          outputs.append({
              "name": "stdout", "output_type": "stream",
              "text": stdout.getvalue().splitlines(keepends=True),
          })
        if stderr.getvalue():
          outputs.append({
              "name": "stderr", "output_type": "stream",
              "text": stderr.getvalue().splitlines(keepends=True),
          })
        if "matplotlib.pyplot" in sys.modules:
          import matplotlib.pyplot as plt
          for figure_number in sorted(set(plt.get_fignums()) - figures_before):
            image = BytesIO()
            plt.figure(figure_number).savefig(
                image, format="png", bbox_inches="tight", dpi=100)
            encoded = base64.b64encode(image.getvalue()).decode("ascii")
            outputs.append({
                "data": {"image/png": encoded},
                "metadata": {},
                "output_type": "display_data",
            })
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs
  finally:
    os.chdir(previous_directory)
  if args.write_outputs:
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print("Updated notebook outputs:", notebook_path)


if __name__ == "__main__":
  main()
