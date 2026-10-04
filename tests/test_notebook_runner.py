"""Native notebook execution preserves displayed figures in the saved output."""

import base64
import importlib.util
import os
import sys
from pathlib import Path

import pytest


def test_runner_retains_matplotlib_display_with_headless_parent(tmp_path, monkeypatch):
    """An inherited Agg backend cannot silently discard a notebook's plt.show output."""
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    pytest.importorskip("matplotlib_inline")
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("run_notebooks", scripts / "run_notebooks.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts/run_notebooks.py"))
    source = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([0, 1], [0, 1])\n"
        "ax.set_xlabel('position')\n"
        "plt.show()\n"
    )
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    path = tmp_path / "notebooks/02_display.ipynb"
    path.parent.mkdir()
    nbformat.write(notebook, path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py", str(path), "--timeout", "30"])
    runner.main()
    assert os.environ["MPLBACKEND"] == "Agg"
    executed = nbformat.read(tmp_path / "build/notebooks/02_display.ipynb", as_version=4)
    assert executed.cells[0].source == source
    outputs = executed.cells[0].outputs
    assert all(output.output_type != "error" for output in outputs)
    images = [
        output.data["image/png"] for output in outputs if "image/png" in output.get("data", {})
    ]
    assert len(images) == 1
    assert base64.b64decode(images[0]).startswith(b"\x89PNG\r\n\x1a\n")
