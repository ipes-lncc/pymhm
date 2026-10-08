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
        "import sys\n"
        f"assert Path.cwd() == Path({str(tmp_path)!r})\n"
        f"assert Path(sys.executable).resolve() == Path({sys.executable!r}).resolve()\n"
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


def test_plan_and_auto_preparation_of_clean_inputs(runner, tmp_path, monkeypatch, capsys):
    """A clean notebook selection prepares its declared fields before launching a kernel."""
    import json

    import notebook_reproduction as reproduction

    manifest_path = tmp_path / "reproduction.json"
    manifest_path.write_text(
        json.dumps(
            {
                "notebooks": {
                    "notebooks/darcy/33_darcy_rt.ipynb": {
                        "environment": "notebooks",
                        "requires": [],
                        "preparation": [
                            {
                                "environment": "notebooks",
                                "argv": [
                                    "python",
                                    "-m",
                                    "examples.solve_darcy_rt",
                                ],
                            }
                        ],
                    },
                }
            }
        )
    )
    monkeypatch.setattr(reproduction, "REPRODUCTION_MANIFEST", manifest_path)
    path = tmp_path / "notebooks/darcy/33_darcy_rt.ipynb"
    path.parent.mkdir(parents=True)
    runner.nbformat.write(runner.nbformat.v4.new_notebook(), path)
    fields = tmp_path / "examples/results"
    fields.mkdir(parents=True)
    (fields / "darcy-rt.json").write_text(json.dumps({"rows": [{"fields": "field.npz"}]}))
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts/run_notebooks.py"))
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py", "33", "--plan"])
    runner.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["missing"] == ["examples/results/field.npz"]
    assert plan["preparation"][0]["argv"] == ["python", "-m", "examples.solve_darcy_rt"]
    assert not (tmp_path / "build/notebooks").exists()
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py", "33", "--no-prepare"])
    with pytest.raises(SystemExit, match="Missing 1 computed"):
        runner.main()
    calls = []

    def prepare(root, acquisition):
        calls.append(acquisition)
        (fields / "field.npz").write_bytes(b"PK-produced")
        for image in runner.execution_inputs(root, [path])[1]["33"]:
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"produced-png")

    class Client:
        def __init__(self, notebook, **kwargs):
            self.notebook = notebook
            assert kwargs["km"].kernel_spec.argv[0] == sys.executable

        def execute(self, **kwargs):
            assert kwargs["cleanup_kc"] is True

    monkeypatch.setattr(runner, "prepare_notebook_inputs", prepare)
    monkeypatch.setattr(runner, "NotebookClient", Client)
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py", "33"])
    runner.main()
    assert len(calls) == 1
    assert (tmp_path / "build/notebooks/darcy/33_darcy_rt.ipynb").is_file()
    receipt = json.loads((tmp_path / "build/notebooks/execution.json").read_text())
    assert receipt["python_executable"] == sys.executable
    assert receipt["notebooks"][0]["source_sha256"]
