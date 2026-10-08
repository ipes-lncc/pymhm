"""Clean-checkout recipes preserve study scope and explicit native execution."""

import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def reproduction(monkeypatch):
    """Load downloaded portable planning without a notebook kernel or checkout."""
    module = importlib.import_module("scripts.notebook_reproduction")
    monkeypatch.setattr(module, "REPRODUCTION_MANIFEST", module.REPRODUCTION_MANIFEST)
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
    with pytest.raises(ValueError, match="Install compatible DOLFINx/UFL"):
        reproduction.validate_native_requirements(tmp_path, [path])
    monkeypatch.setattr(reproduction.importlib.util, "find_spec", lambda name: object())
    reproduction.validate_native_requirements(tmp_path, [path])


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


def test_prepare_uses_current_python_argument_vectors_and_workspace(
    reproduction, tmp_path, monkeypatch
):
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
    assert args[0] == [
        sys.executable,
        str(reproduction.source_file("examples/case.py", root=tmp_path)),
        *step["argv"][2:],
    ]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["check"]
    assert kwargs["env"]["MPLBACKEND"] == "Agg"
    assert kwargs["env"]["PYVISTA_OFF_SCREEN"] == "true"
    assert "shell" not in kwargs
    reproduction.prepare_notebook_inputs(tmp_path, {"preparation": [], "unresolved": []})
    assert len(calls) == 1
    monkeypatch.setattr(reproduction.shutil, "which", lambda name: None)
    reproduction.prepare_notebook_inputs(tmp_path, {"preparation": [step], "unresolved": []})
    assert len(calls) == 2
    assert calls[1][1]["env"]["PYMHM_WORKSPACE"] == str(tmp_path)


def test_manifest_covers_every_source_and_executable_producer(reproduction):
    """The repository contract has no unclassified or stale source/producer paths."""
    root = Path(__file__).resolve().parents[1]
    reproduction.validate_reproduction_manifest(root)


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
    import scripts.notebook_data as notebook_data

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
        lambda root, selected: {
            key: set(value) for key, value in archives.items() if key in selected
        },
    )
    monkeypatch.setattr(
        notebook_data,
        "required_images",
        lambda root, selected: {
            key: set(value) for key, value in images.items() if key in selected
        },
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
        "-m",
        "scripts.run_notebooks",
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


def test_default_scope_filters_before_opening_historical_manifests(
    reproduction, tmp_path, monkeypatch
):
    """Absent original comparison manifests are never read for a current default run."""
    import scripts.notebook_data as data

    paths = [source(tmp_path, "14_external.ipynb"), source(tmp_path, "33_study.ipynb")]
    manifest(
        reproduction,
        tmp_path,
        {
            paths[0].relative_to(tmp_path).as_posix(): {"historical_inputs": True},
            paths[1].relative_to(tmp_path).as_posix(): {"study_inputs": True},
        },
    )
    seen = []

    def inventory(root, selected):
        assert selected == set()
        seen.append(selected)
        return {}

    monkeypatch.setattr(data, "required_archives", inventory)
    monkeypatch.setattr(data, "required_images", inventory)
    assert reproduction.execution_inputs(tmp_path, paths) == ({}, {})
    assert len(seen) == 2


def test_historical_archives_keep_current_figures_and_public_plot_recipe(
    reproduction, tmp_path, monkeypatch
):
    """Original native fields cannot block a default scalar-record figure render."""
    import scripts.notebook_data as data

    path = source(tmp_path, "45_conditioning.ipynb")
    step = {
        "environment": "notebooks",
        "argv": ["python", "-m", "examples.plot_rad_conditioning"],
    }
    manifest(
        reproduction,
        tmp_path,
        {
            path.relative_to(tmp_path).as_posix(): {
                "historical_archives": True,
                "preparation": [step],
            }
        },
    )
    archive = tmp_path / "examples/results/rad-native/original.npz"
    image = tmp_path / "docs/figures/rad-conditioning/epsilon.png"
    lookups = []

    def archives(root, selected):
        lookups.append(set(selected))
        return {"45": {archive}} if "45" in selected else {}

    monkeypatch.setattr(data, "required_archives", archives)
    monkeypatch.setattr(data, "required_images", lambda root, selected: {"45": {image}})
    actual, figures = reproduction.execution_inputs(tmp_path, [path])
    assert actual == {} and figures == {"45": {image}}
    assert lookups == [set()]
    plan = reproduction.preparation_plan(
        tmp_path, [path], data.dependency_plan(tmp_path, actual, figures)
    )
    assert plan["preparation"] == [step]
    assert plan["unresolved"] == []
    actual, figures = reproduction.execution_inputs(tmp_path, [path], study=True)
    assert actual == {} and figures == {"45": {image}}
    actual, figures = reproduction.execution_inputs(tmp_path, [path], historical=True)
    assert actual == {"45": {archive}}
    assert lookups[-1] == {"45"}


def test_tetra_pk_public_resolution_producer_precedes_its_profile_plot(reproduction):
    """The downloaded recipe acquires the unchanged P4 control before plotting its fields."""
    contracts = json.loads(reproduction.REPRODUCTION_MANIFEST.read_text())["notebooks"]
    contract = contracts["notebooks/darcy/64_tetra_pk.ipynb"]
    for key in ("preparation", "current_study"):
        modules = [step["argv"][2] for step in contract[key]]
        assert modules.index("examples.reconstruction3d_resolution") < modules.index(
            "examples.plot_tetra_pk"
        )
        producer = next(
            step
            for step in contract[key]
            if step["argv"][2] == "examples.reconstruction3d_resolution"
        )
        assert producer["argv"] == ["python", "-m", "examples.reconstruction3d_resolution"]
