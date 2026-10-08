"""Clean-checkout recipes preserve study scope and explicit native execution."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def reproduction(monkeypatch):
    """Load the portable planning layer without starting a notebook kernel."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "notebook_reproduction", scripts / "notebook_reproduction.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("PIXI_EXE", raising=False)
    return module


def source(tmp_path, name, code="print('executed')"):
    """Write a minimal standard notebook without requiring nbformat."""
    path = tmp_path / "notebooks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": code}]}))
    return path


def manifest(module, tmp_path, contracts):
    """Install a fixture catalogue with no unrelated publication inputs."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"notebooks": contracts}))
    module.REPRODUCTION_MANIFEST = path


def test_native_guard_cannot_silently_skip_required_demonstration(
    reproduction, tmp_path, monkeypatch
):
    """A guarded UFL import remains a required executed mathematical example."""
    path = source(
        tmp_path,
        "ufl.ipynb",
        "if False:\n    from dolfinx import fem\n    import ufl\n",
    )
    assert reproduction.native_requirements(path) == {"dolfinx", "ufl"}
    manifest(reproduction, tmp_path, {})
    monkeypatch.setattr(reproduction.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="pixi run --locked -e introduction"):
        reproduction.validate_native_requirements(tmp_path, [path])
    monkeypatch.setattr(reproduction.importlib.util, "find_spec", lambda name: object())
    reproduction.validate_native_requirements(tmp_path, [path])


def test_external_notebook_has_no_checkout_producer(reproduction, tmp_path):
    """External notebooks receive native checks without inheriting same-ID data recipes."""
    root = tmp_path / "checkout"
    root.mkdir()
    path = source(tmp_path, "33_external.ipynb", "import cupy\n!echo optional\n")
    manifest(reproduction, tmp_path, {})
    assert reproduction.native_requirements(path) == set()
    assert reproduction.notebook_contract(root, path)["preparation"] == []


def test_preparation_selects_missing_inputs_and_deduplicates_steps(reproduction, tmp_path):
    """Only missing selected inputs trigger their complete acquisition recipe."""
    paths = [source(tmp_path, f"{number}_study.ipynb") for number in (33, 41, 69)]
    step = {"environment": "notebooks", "argv": ["python", "examples/acquire.py", "--full"]}
    manifest(
        reproduction,
        tmp_path,
        {path.relative_to(tmp_path).as_posix(): {"preparation": [step]} for path in paths},
    )
    data = {
        "notebooks": {"33": ["examples/results/field.npz"], "41": [], "69": []},
        "notebook_images": {"41": ["docs/figures/field.png"], "69": []},
        "missing": ["examples/results/field.npz"],
        "missing_images": ["docs/figures/field.png"],
    }
    plan = reproduction.preparation_plan(tmp_path, paths, data)
    assert plan["preparation"] == [step]
    assert plan["unresolved"] == []
    assert plan["notebooks"][-1]["missing"] == []


def test_unknown_historical_archives_fail_before_any_acquisition(
    reproduction, tmp_path, monkeypatch
):
    """A fresh solver cannot manufacture the checksum of an absent external acquisition."""
    path = source(tmp_path, "14_reference.ipynb")
    manifest(
        reproduction,
        tmp_path,
        {
            path.relative_to(tmp_path).as_posix(): {
                "limitation": "Original NeoPZ archive not published"
            }
        },
    )
    data = {
        "notebooks": {"14": ["examples/results/external.npz"]},
        "missing": ["examples/results/external.npz"],
    }
    plan = reproduction.preparation_plan(tmp_path, [path], data)
    monkeypatch.setattr(reproduction.subprocess, "run", lambda *a, **k: pytest.fail("producer ran"))
    with pytest.raises(ValueError, match="Original NeoPZ archive not published"):
        reproduction.prepare_notebook_inputs(tmp_path, plan)


def test_prepare_uses_locked_pixi_argument_vectors_and_root(reproduction, tmp_path, monkeypatch):
    """Spaces and shell metacharacters stay literal subprocess arguments."""
    calls = []
    monkeypatch.setattr(reproduction.shutil, "which", lambda name: "/opt/Pixi Tools/pixi")
    monkeypatch.setattr(reproduction.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    step = {
        "environment": "notebooks",
        "argv": ["python", "examples/case.py", "--output", "literal $value; archive"],
    }
    reproduction.prepare_notebook_inputs(tmp_path, {"preparation": [step], "unresolved": []})
    args, kwargs = calls[0]
    assert args[0] == ["/opt/Pixi Tools/pixi", "run", "--locked", "-e", "notebooks", *step["argv"]]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["check"]
    assert kwargs["env"]["MPLBACKEND"] == "Agg"
    assert kwargs["env"]["PYVISTA_OFF_SCREEN"] == "true"
    assert "shell" not in kwargs
    reproduction.prepare_notebook_inputs(tmp_path, {"preparation": [], "unresolved": []})
    assert len(calls) == 1
    monkeypatch.setattr(reproduction.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="Pixi on PATH"):
        reproduction.prepare_notebook_inputs(tmp_path, {"preparation": [step], "unresolved": []})


def test_manifest_covers_every_source_and_executable_producer(reproduction):
    """The repository contract has no unclassified or stale source/producer paths."""
    root = Path(__file__).resolve().parents[1]
    reproduction.validate_reproduction_manifest(root)


@pytest.mark.parametrize(
    "contract,message",
    [
        ({"kind": "unknown", "preparation": []}, "reproducibility kind"),
        ({"kind": "standalone", "preparation": [{"argv": "python case.py"}]}, "argument vector"),
        (
            {
                "kind": "generated-study",
                "preparation": [{"argv": ["python", "missing.py"], "environment": "notebooks"}],
            },
            "producer source",
        ),
    ],
)
def test_invalid_manifest_contract_is_rejected(reproduction, tmp_path, contract, message):
    """A source catalogue cannot claim a usable recipe without a real public producer."""
    path = source(tmp_path, "study.ipynb")
    manifest(reproduction, tmp_path, {path.relative_to(tmp_path).as_posix(): contract})
    with pytest.raises(ValueError, match=message):
        reproduction.validate_reproduction_manifest(tmp_path)


def test_catalogue_drift_and_cli(reproduction, tmp_path, monkeypatch, capsys):
    """Clean source audits diagnose unregistered tutorials without acquiring any data."""
    path = source(tmp_path, "study.ipynb")
    manifest(reproduction, tmp_path, {})
    monkeypatch.setattr(
        reproduction, "__file__", str(tmp_path / "scripts/notebook_reproduction.py")
    )
    monkeypatch.setattr(sys, "argv", ["notebook_reproduction.py", "--check"])
    with pytest.raises(SystemExit, match="2"):
        reproduction.main()
    manifest(
        reproduction,
        tmp_path,
        {path.relative_to(tmp_path).as_posix(): {"kind": "standalone", "preparation": []}},
    )
    reproduction.main()
    assert json.loads(capsys.readouterr().out)["classifications"]["standalone"] == 1


def test_public_import_audit_survives_magics(reproduction, tmp_path):
    """A shell line cannot conceal a private package import or a required native example."""
    path = source(
        tmp_path,
        "private.ipynb",
        "!echo display\nfrom pymhm._legacy.models import operator\n"
        "from pymhm.meshes.hexahedron import _geometry\nimport dolfinx.fem\n",
    )
    imports, private = reproduction.notebook_imports(path)
    assert "dolfinx.fem" in imports
    assert private == {"pymhm._legacy.models.operator", "pymhm.meshes.hexahedron._geometry"}
    assert reproduction.native_requirements(path) == {"dolfinx", "ufl"}
    manifest(
        reproduction,
        tmp_path,
        {path.relative_to(tmp_path).as_posix(): {"kind": "standalone", "preparation": []}},
    )
    with pytest.raises(ValueError, match="private package interfaces"):
        reproduction.validate_reproduction_manifest(tmp_path)


def test_execution_modes_preserve_original_inventory(reproduction, tmp_path, monkeypatch):
    """Default controls, complete studies and exact historical replay have distinct inputs."""
    import notebook_data

    paths = [source(tmp_path, f"{n}_study.ipynb") for n in (14, 33, 71)]
    contracts = [
        {"historical_inputs": True},
        {"study_inputs": True},
        {"historical_images": ["docs/figures/native.png"]},
    ]
    manifest(
        reproduction,
        tmp_path,
        {
            path.relative_to(tmp_path).as_posix(): contract
            for path, contract in zip(paths, contracts, strict=True)
        },
    )
    archives = {str(n): {tmp_path / f"examples/results/{n}.npz"} for n in (14, 33, 71)}
    images = {
        "14": {tmp_path / "docs/figures/external.png"},
        "33": {tmp_path / "docs/figures/study.png"},
        "71": {tmp_path / "docs/figures/native.png", tmp_path / "docs/figures/current.png"},
    }
    monkeypatch.setattr(
        notebook_data,
        "required_archives",
        lambda *args: {key: set(value) for key, value in archives.items()},
    )
    monkeypatch.setattr(
        notebook_data,
        "required_images",
        lambda *args: {key: set(value) for key, value in images.items()},
    )
    actual, figures = reproduction.execution_inputs(tmp_path, paths)
    assert set(actual) == {"71"}
    assert figures == {"71": {tmp_path / "docs/figures/current.png"}}
    actual, figures = reproduction.execution_inputs(tmp_path, paths, study=True)
    assert set(actual) == {"33", "71"}
    assert set(figures) == {"33", "71"}
    actual, figures = reproduction.execution_inputs(tmp_path, paths, historical=True)
    assert actual == archives
    assert figures == images
    assert "14" in archives and "33" in archives


def test_native_dispatch_pins_locked_profile_and_literal_flags(reproduction, tmp_path, monkeypatch):
    """Unavailable native modules select a checked environment without shell interpolation."""
    path = source(tmp_path, "ufl.ipynb", "import ufl")
    manifest(reproduction, tmp_path, {})
    monkeypatch.setattr(reproduction.importlib.util, "find_spec", lambda name: None)
    assert reproduction.required_environment(tmp_path, path) == "introduction"
    monkeypatch.setattr(reproduction.importlib.util, "find_spec", lambda name: object())
    assert reproduction.required_environment(tmp_path, path) is None
    calls = []
    monkeypatch.setattr(reproduction.shutil, "which", lambda name: "/opt/pixi")
    monkeypatch.setattr(
        reproduction.subprocess, "run", lambda *args, **kw: calls.append((args, kw))
    )
    reproduction.execute_in_environment(tmp_path, path, "introduction", ["--study", "--no-prepare"])
    args, kwargs = calls[0]
    assert args[0] == [
        "/opt/pixi",
        "run",
        "--locked",
        "-e",
        "introduction",
        "python",
        str(tmp_path / "scripts/run_notebooks.py"),
        str(path),
        "--no-dispatch",
        "--study",
        "--no-prepare",
    ]
    assert kwargs == {"cwd": tmp_path, "check": True}
    monkeypatch.setattr(reproduction.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="dispatch requires Pixi"):
        reproduction.execute_in_environment(tmp_path, path, "introduction", [])


def test_explicit_study_forces_complete_public_acquisition(reproduction, tmp_path, monkeypatch):
    """A --study acquisition executes declared levels even when previous inputs exist."""
    path = source(tmp_path, "study.ipynb")
    step = {
        "environment": "notebooks",
        "argv": ["python", "-m", "examples.acquire", "--output", "build/results/{acquisition}"],
    }
    manifest(
        reproduction, tmp_path, {path.relative_to(tmp_path).as_posix(): {"current_study": [step]}}
    )
    data = {"notebooks": {}, "missing": []}
    assert reproduction.preparation_plan(tmp_path, [path], data)["preparation"] == []
    plan = reproduction.preparation_plan(tmp_path, [path], data, study=True)
    assert plan["preparation"] == [step]
    calls = []
    monkeypatch.setattr(reproduction, "uuid4", lambda: "new-execution")
    monkeypatch.setattr(reproduction.shutil, "which", lambda name: "/opt/pixi")
    monkeypatch.setattr(reproduction.subprocess, "run", lambda *args, **kw: calls.append(args[0]))
    reproduction.prepare_notebook_inputs(tmp_path, plan)
    assert calls[0][-1] == "build/results/new-execution"
    assert plan["acquisition_id"] == "new-execution"
    assert plan["executed_preparation"][0]["argv"][-1] == "build/results/new-execution"
    assert plan["preparation"][0]["argv"][-1] == "build/results/{acquisition}"


def test_absolute_pixi_invocation_can_reenter_locked_environments(
    reproduction, tmp_path, monkeypatch
):
    """An active Pixi executable need not be installed on the invoking user's PATH."""
    executable = tmp_path / "Pixi Tools/pixi"
    executable.parent.mkdir()
    executable.write_text("executable fixture")
    monkeypatch.setattr(reproduction.shutil, "which", lambda name: None)
    monkeypatch.setenv("PIXI_EXE", str(executable))
    assert reproduction.pixi_executable() == str(executable)
    executable.unlink()
    assert reproduction.pixi_executable() is None


def test_historical_identity_is_not_replaced_by_current_producer(reproduction, tmp_path):
    """Known fresh producers do not recreate an absent original field checksum."""
    path = source(tmp_path, "71_original.ipynb")
    step = {"environment": "notebooks", "argv": ["python", "-m", "examples.acquire"]}
    manifest(
        reproduction, tmp_path, {path.relative_to(tmp_path).as_posix(): {"preparation": [step]}}
    )
    data = {
        "notebooks": {"71": ["examples/results/old.npz"]},
        "missing": ["examples/results/old.npz"],
    }
    plan = reproduction.preparation_plan(tmp_path, [path], data, historical=True)
    assert plan["preparation"] == []
    assert len(plan["unresolved"]) == 1
    assert "Original historical payload identities" in plan["unresolved"][0]["reason"]
