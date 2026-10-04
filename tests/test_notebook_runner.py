"""Native notebook execution preserves displayed figures in the saved output."""

import base64
import importlib.util
import os
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "relative_path",
    ["notebooks/foundations/general/02_display.ipynb", "external/14_display.ipynb"],
)
def test_runner_retains_matplotlib_display_with_headless_parent(
    tmp_path, monkeypatch, relative_path
):
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
        "from pathlib import Path\n"
        f"assert Path.cwd() == Path({str(tmp_path)!r})\n"
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([0, 1], [0, 1])\n"
        "ax.set_xlabel('position')\n"
        "plt.show()\n"
    )
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    nbformat.write(notebook, path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py", str(path), "--timeout", "30"])
    runner.main()
    assert os.environ["MPLBACKEND"] == "Agg"
    executed = nbformat.read(runner.output_notebook_path(tmp_path, path), as_version=4)
    assert executed.cells[0].source == source
    outputs = executed.cells[0].outputs
    assert all(output.output_type != "error" for output in outputs)
    images = [
        output.data["image/png"] for output in outputs if "image/png" in output.get("data", {})
    ]
    assert len(images) == 1
    assert base64.b64decode(images[0]).startswith(b"\x89PNG\r\n\x1a\n")


@pytest.fixture
def runner(monkeypatch):
    """Load the CLI with native notebook dependencies when the environment has them."""
    pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("run_notebooks", scripts / "run_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recursive_runner_preserves_sources_and_equal_basenames(runner, tmp_path, monkeypatch):
    """A group execution retains separate outputs and runs each source once at repo root."""
    sources = [
        tmp_path / "notebooks/waves/helmholtz/tutorial.ipynb",
        tmp_path / "notebooks/waves/maxwell/tutorial.ipynb",
    ]
    before = {}
    for path in sources:
        path.parent.mkdir(parents=True)
        notebook = runner.nbformat.v4.new_notebook(
            cells=[runner.nbformat.v4.new_code_cell(f"print({path.parent.name!r})")]
        )
        runner.nbformat.write(notebook, path)
        before[path] = path.read_bytes()
    calls = []

    class Client:
        """Record executions without launching additional native kernels."""

        def __init__(self, notebook, **kwargs):
            """Retain the exact notebook and kernel working-directory contract."""
            self.notebook = notebook
            calls.append(kwargs)

        def execute(self, **kwargs):
            """Provide an output to distinguish executed and source notebooks."""
            self.notebook.cells[0].execution_count = 1

    monkeypatch.setattr(runner, "NotebookClient", Client)
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts/run_notebooks.py"))
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py", "waves", str(sources[0])])
    runner.main()
    assert len(calls) == 2
    assert all(call["resources"]["metadata"]["path"] == str(tmp_path) for call in calls)
    for source in sources:
        assert source.read_bytes() == before[source]
        destination = tmp_path / "build/notebooks" / source.relative_to(tmp_path / "notebooks")
        assert runner.nbformat.read(destination, as_version=4).cells[0].execution_count == 1


def test_external_notebook_outputs_are_distinct_and_bounded(runner, tmp_path):
    """External sources with equal basenames neither overwrite nor escape output storage."""
    root = tmp_path / "checkout"
    first = tmp_path / "other/tutorial.ipynb"
    second = tmp_path / "another/tutorial.ipynb"
    destinations = [runner.output_notebook_path(root, source) for source in (first, second)]
    assert destinations[0] != destinations[1]
    for destination in destinations:
        assert destination.is_relative_to(root / "build/notebooks/external")
        assert destination.name == "tutorial.ipynb"
        assert len(destination.parent.name) == 64
    assert runner.output_notebook_path(root, first) == destinations[0]


@pytest.mark.parametrize(
    "arguments,message",
    [
        ([], "No notebooks found"),
        (["unknown"], "Unknown notebook"),
        (["--timeout", "0"], "positive"),
    ],
)
def test_runner_rejects_invalid_selection_before_execution(
    runner, tmp_path, monkeypatch, arguments, message
):
    """The CLI reports selection and timeout errors before starting a kernel."""
    if "--timeout" in arguments:
        path = tmp_path / "notebooks/tutorial.ipynb"
        path.parent.mkdir()
        path.write_text("{}")
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts/run_notebooks.py"))
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py", *arguments])
    with pytest.raises(SystemExit, match=message):
        runner.main()
