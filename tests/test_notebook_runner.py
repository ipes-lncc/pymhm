"""Native notebook execution preserves displayed figures in the saved output."""

import base64
import importlib
import os
import sys

import pytest

pytestmark = pytest.mark.notebooks


@pytest.mark.visualization
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
    runner = importlib.import_module("scripts.run_notebooks")
    monkeypatch.setattr(runner, "case_workspace", lambda: tmp_path)
    monkeypatch.setattr(runner, "notebook_workspace", lambda selector: tmp_path)
    monkeypatch.setattr(runner, "stage_notebook_resources", lambda selector: tmp_path)
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
    return importlib.import_module("scripts.run_notebooks")


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
    monkeypatch.setattr(runner, "case_workspace", lambda: tmp_path)
    monkeypatch.setattr(runner, "notebook_workspace", lambda selector: tmp_path)
    monkeypatch.setattr(runner, "stage_notebook_resources", lambda selector: tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py", "waves", str(sources[0])])
    runner.main()
    assert len(calls) == 2
    assert all(call["resources"]["metadata"]["path"] == str(tmp_path) for call in calls)
    for source in sources:
        assert source.read_bytes() == before[source]
        destination = tmp_path / "build/notebooks" / source.relative_to(tmp_path / "notebooks")
        assert runner.nbformat.read(destination, as_version=4).cells[0].execution_count == 1


def test_plan_and_auto_preparation_of_clean_inputs(runner, tmp_path, monkeypatch, capsys):
    """A clean notebook selection prepares its declared fields before launching a kernel."""
    import json

    import scripts.notebook_reproduction as reproduction

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
    monkeypatch.setattr(runner, "case_workspace", lambda: tmp_path)
    monkeypatch.setattr(runner, "notebook_workspace", lambda selector: tmp_path)
    monkeypatch.setattr(runner, "stage_notebook_resources", lambda selector: tmp_path)
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


def test_selected_resource_is_staged_before_archive_validation(runner, tmp_path, monkeypatch):
    """A declared downloadable input with no numerical producer works with --no-prepare."""
    import scripts.notebook_data as data

    source = tmp_path / "notebooks/darcy/21_spe10_data.ipynb"
    source.parent.mkdir(parents=True)
    runner.nbformat.write(runner.nbformat.v4.new_notebook(), source)
    field = tmp_path / "examples/results/spe10/layer-36.npz"
    monkeypatch.setattr(runner, "case_workspace", lambda: tmp_path)
    monkeypatch.setattr(data, "required_images", lambda *args: {})
    monkeypatch.setattr(runner, "notebook_workspace", lambda selector: tmp_path)
    monkeypatch.setattr(runner, "prepare_notebook_inputs", lambda *args: pytest.fail("producer"))
    stages = []

    def stage(selector):
        assert selector == "darcy/21_spe10_data.ipynb"
        stages.append(selector)
        field.parent.mkdir(parents=True)
        field.write_bytes(b"PK-verified-resource-fixture")
        return tmp_path

    class Client:
        """Avoid launching a kernel for this resource ordering regression."""

        def __init__(self, notebook, **kwargs):
            assert field.is_file()

        def execute(self, **kwargs):
            pass

    monkeypatch.setattr(runner, "stage_notebook_resources", stage)
    monkeypatch.setattr(runner, "NotebookClient", Client)
    monkeypatch.setattr(sys, "argv", ["notebooks", "21", "--no-prepare"])
    runner.main()
    assert stages == ["darcy/21_spe10_data.ipynb"]


def test_downloaded_companion_runs_real_public_provider_and_lazy_data(tmp_path):
    """A downloaded notebook prepares one real problem outside any checkout."""
    import hashlib
    import json
    import shutil
    import subprocess
    import threading
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path

    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    from pymhm.io.workspace import workspace_from_archive
    from scripts.build_notebook_companions import NOTEBOOK_TOOLS, build_companion

    repository = Path(__file__).resolve().parents[1]
    source_tree, public, work = (tmp_path / name for name in ("sources", "public", "work"))
    public.mkdir()
    source_files = {
        *NOTEBOOK_TOOLS,
        "LICENSE",
        "examples/__init__.py",
        "examples/tutorial_local_provider.py",
    }
    for name in source_files:
        destination = source_tree / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, destination)
    selector = "foundations/bundle_smoke.ipynb"
    step = {"environment": "notebooks", "argv": ["python", "-m", "examples.acquire_smoke"]}
    contract = {
        "kind": "standalone",
        "environment": "notebooks",
        "requires": [],
        "preparation": [step],
    }
    (source_tree / "scripts/notebook_reproduction.json").write_text(
        json.dumps({"notebooks": {"notebooks/" + selector: contract}})
    )
    producer = source_tree / "examples/acquire_smoke.py"
    producer.write_text(
        "import json\nfrom pymhm.io.workspace import case_workspace, read_resource_text\n"
        "from examples.tutorial_local_provider import run_tutorial, diagnostics\n"
        "root = case_workspace()\n"
        "options = json.loads(read_resource_text(root / 'inputs/control.json'))\n"
        "result = run_tutorial(element_backend='portable', **options)\n"
        "(root / 'acquisition.json').write_text(json.dumps(diagnostics(result)))\n"
        "with (root / 'producer-count.txt').open('a') as stream: stream.write('1\\n')\n"
    )
    source_files.add("examples/acquire_smoke.py")
    control = public / "control.json"
    control.write_text(json.dumps({"formulation": "mixed", "boundary": "neumann"}))
    handler = partial(SimpleHTTPRequestHandler, directory=str(public))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = f"http://127.0.0.1:{server.server_port}"
        scopes = {
            key: {selector: ["inputs/control.json"]}
            for key in (
                "notebook_resources",
                "notebook_historical_resources",
                "notebook_study_resources",
            )
        }
        descriptor = build_companion(
            source_tree,
            source_files,
            public,
            name="smoke-companion.zip",
            scopes=scopes,
            resources={
                "inputs/control.json": {
                    "url": origin + "/control.json",
                    "sha256": hashlib.sha256(control.read_bytes()).hexdigest(),
                    "size_bytes": control.stat().st_size,
                }
            },
        )
        url = origin + "/downloads/" + descriptor["sha256"] + "/smoke-companion.zip"
        workspace_from_archive(url, sha256=descriptor["sha256"], directory=work)
        assert not (work / "inputs/control.json").exists()
        downloaded = tmp_path / "my-notebook.ipynb"
        code = (
            "from scripts.notebook_reproduction import notebook_workspace\n"
            f"ROOT = notebook_workspace({selector!r})\n"
            "import json\nfrom pathlib import Path\n"
            "import examples.tutorial_local_provider as provider\n"
            "assert Path(provider.__file__).resolve().is_relative_to(ROOT)\n"
            "record = json.loads((ROOT / 'acquisition.json').read_text())\n"
            "assert record['boundary'] == 'neumann'\n"
            "assert record['global_original_relative_residual'] < 1e-10\n"
            "print(record['spaces'])\n"
        )
        nbformat.write(
            nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), downloaded
        )
        original = downloaded.read_bytes()
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "PYTHONPATH",
                "PYMHM_WORKSPACE",
                "PYMHM_NOTEBOOK_PREPARED",
                "PYMHM_RESOURCE_ORIGIN",
            }
        }
        environment.update(PYMHM_WORKSPACE=str(work), PYMHM_CACHE_DIR=str(tmp_path / "cache"))
        subprocess.run(
            [sys.executable, "-m", "scripts.run_notebooks", str(downloaded), "--study", "--check"],
            cwd=work,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert downloaded.read_bytes() == original
        assert (work / "producer-count.txt").read_text() == "1\n"
        assert (work / "inputs/control.json").read_bytes() == control.read_bytes()
        receipt = json.loads((work / "build/notebooks/execution.json").read_text())
        assert receipt["python_executable"] == sys.executable
        assert receipt["notebooks"][0]["source_sha256"] == hashlib.sha256(original).hexdigest()
        assert receipt["preparation"][0]["argv"] == [sys.executable, "-m", "examples.acquire_smoke"]
        assert not (work / "src").exists() and not (work / "pixi.lock").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
