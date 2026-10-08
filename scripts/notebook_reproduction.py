"""Declared companion preparation and native requirements for notebooks.

Recipes execute public acquisition and plotting modules with the current Python
installation. They regenerate complete declared studies rather than replacing
them with smaller analytical controls. Historical external acquisitions retain
their provenance and cannot be recreated by merely rerunning a PyMHM solve.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from pymhm.core.validation import positive_int
from pymhm.io.workspace import (
    case_workspace,
    materialize_resources,
    register_resources,
    source_file,
)

REPRODUCTION_MANIFEST = Path(__file__).with_name("notebook_reproduction.json")


def stage_notebook_resources(selector: str, directory: str | Path | None = None) -> Path:
    """Stage explicitly selected companion inputs without executing producers.

    The downloaded companion supplies its selection metadata separately from the
    generic resource catalogue. The checkout uses ``examples/resource_manifest.json``.
    Study and historical scopes are opt-in; existing local inputs are preserved.
    """
    relative = selector.removeprefix("notebooks/")
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in relative
        or path.suffix != ".ipynb"
    ):
        raise ValueError(f"Invalid notebook workspace selector: {selector!r}")
    root = case_workspace(directory)
    metadata = root / ".pymhm-notebook-resources.json"
    if not metadata.is_file():
        metadata = root / "examples/resource_manifest.json"
    inventory = json.loads(metadata.read_text(encoding="utf-8"))
    if not (root / ".pymhm-resources.json").is_file():
        resources = {
            name: {"sha256": digest} for name, digest in inventory.get("bundled", {}).items()
        }
        resources.update(inventory.get("remote", {}))
        register_resources(
            root, {"resources": resources, "unavailable": inventory.get("unavailable", {})}
        )
    if relative not in inventory["notebook_resources"]:
        raise FileNotFoundError(f"Companion does not declare notebook: {relative}")
    names = list(inventory["notebook_resources"][relative])
    if os.environ.get("PYMHM_NOTEBOOK_HISTORICAL", "0") == "1":
        names.extend(inventory["notebook_historical_resources"][relative])
    if os.environ.get("PYMHM_NOTEBOOK_STUDY", "0") == "1":
        names.extend(inventory["notebook_study_resources"][relative])
    materialize_resources(names, root)
    return root


def notebook_workspace(selector: str, directory: str | Path | None = None) -> Path:
    """Prepare this companion's declared notebook inputs with the current Python.

    This explicit call may execute the printed public preparation commands from
    ``scripts/notebook_reproduction.json``. It never runs notebook mathematical
    cells. Historical coefficient replay and complete studies remain opt-in.
    No installed example package, checkout or Pixi executable is required.
    """
    from scripts import notebook_data as data

    relative = selector.removeprefix("notebooks/")
    root = stage_notebook_resources(relative, directory)
    # A logical path selects the contract; it is not claimed as an executed file.
    source = root / "notebooks" / relative
    historical = os.environ.get("PYMHM_NOTEBOOK_HISTORICAL", "0") == "1"
    study = os.environ.get("PYMHM_NOTEBOOK_STUDY", "0") == "1"
    inputs = data.dependency_plan(
        root, *execution_inputs(root, [source], historical=historical, study=study)
    )
    plan = preparation_plan(root, [source], inputs, historical=historical, study=study)
    if os.environ.get("PYMHM_NOTEBOOK_PREPARED") != relative:
        prepare_notebook_inputs(root, plan)
    maximum_bytes = positive_int(
        int(os.environ.get("PYMHM_NOTEBOOK_MAX_DATA_BYTES", data.DEFAULT_MAX_BYTES)),
        "maximum data bytes",
    )
    data.validate_archives(
        root,
        data.dependency_plan(
            root, *execution_inputs(root, [source], historical=historical, study=study)
        ),
        maximum_bytes,
    )
    return root


def pixi_executable() -> str | None:
    """Find Pixi through PATH or its explicit executable in an active Pixi task.

    Pixi can be invoked by absolute path without being installed on PATH. Its
    ``PIXI_EXE`` identifies that same executable for nested locked commands.
    """
    executable = shutil.which("pixi")
    if executable is not None:
        return executable
    active = os.environ.get("PIXI_EXE")
    return active if active is not None and Path(active).is_file() else None


def notebook_contract(root: Path, source: Path) -> dict[str, Any]:
    """Read a source's execution contract without opening numerical payloads.

    Downloaded sources retain their declared ``notebook_workspace`` identity.
    Unrelated external notebooks receive only their actual import requirements.
    """
    from scripts.notebook_data import notebook_selector

    selector = notebook_selector(root, source)
    relative = f"notebooks/{selector}" if selector is not None else None
    manifest = json.loads(REPRODUCTION_MANIFEST.read_text(encoding="utf-8"))
    contract = dict(manifest["notebooks"].get(relative, {}))
    contract.setdefault("environment", "notebooks")
    contract.setdefault("preparation", [])
    contract.setdefault("requires", [])
    return contract


def notebook_imports(source: Path) -> tuple[set[str], set[str]]:
    """Inspect actual code imports, including guarded native and private package imports.

    Line magics and shell escapes do not obscure Python imports in other lines.
    The second set identifies private PyMHM modules or explicitly imported
    private symbols; such imports cannot be part of a public API example.
    """
    notebook = json.loads(source.read_text(encoding="utf-8"))
    imports: set[str] = set()
    private: set[str] = set()
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source_text = cell.get("source", "")
        text = "".join(source_text) if isinstance(source_text, list) else source_text
        text = "\n".join(
            "" if line.lstrip().startswith(("%", "!")) else line for line in text.splitlines()
        )
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
                private.update(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("pymhm.")
                    and any(part.startswith("_") for part in alias.name.split(".")[1:])
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
                if node.module == "pymhm" or node.module.startswith("pymhm."):
                    private.update(
                        f"{node.module}.{alias.name}"
                        for alias in node.names
                        if alias.name.startswith("_")
                        or any(part.startswith("_") for part in node.module.split(".")[1:])
                    )
    return imports, private


def native_requirements(source: Path) -> set[str]:
    """Find required UFL/DOLFINx execution without honoring silent import guards.

    A tutorial importing UFL demonstrates native assembly, so both UFL and
    DOLFINx must run. Optional alternative solvers and accelerators are governed
    by their explicitly declared notebook controls, not incidental imports.
    """
    imports, _ = notebook_imports(source)
    return (
        {"dolfinx", "ufl"}
        if {name.split(".", 1)[0] for name in imports}.intersection({"dolfinx", "ufl"})
        else set()
    )


def validate_native_requirements(root: Path, notebooks: list[Path]) -> None:
    """Reject unavailable required backends before kernels or studies start."""
    for source in notebooks:
        contract = notebook_contract(root, source)
        requirements = native_requirements(source) | set(contract["requires"])
        missing = sorted(name for name in requirements if importlib.util.find_spec(name) is None)
        if missing:
            selector = (
                source.relative_to(root).as_posix() if source.is_relative_to(root) else str(source)
            )
            raise ValueError(
                f"Notebook {selector} requires executed native modules: {', '.join(missing)}. "
                "Optional cells are not accepted as silently skipped demonstrations. "
                "Install compatible DOLFINx/UFL native dependencies for this Python "
                "as described at https://ipes-lncc.github.io/pymhm/installation/, "
                "and install the notebook extras with python -m pip install "
                "'pymhm[notebooks,visualization]'. Then run: "
                f"python -m scripts.run_notebooks {selector}"
            )


def execution_inputs(
    root: Path, notebooks: list[Path], *, historical: bool = False, study: bool = False
) -> tuple[dict[str, set[Path]], dict[str, set[Path]]]:
    """Separate current execution inputs from explicitly optional archived replay.

    The complete historical inventory remains available through notebook_data.
    ``historical=True`` requires all original payloads and their notebook checks;
    new acquisitions cannot stand in for the historical coefficient vectors.
    A ``historical_archives`` contract keeps current image production available
    while requiring original field arrays only for explicit historical replay.
    """
    from scripts.notebook_data import (
        notebook_identifier,
        notebook_selector,
        required_archives,
        required_images,
        selected_notebook_ids,
    )

    selected = selected_notebook_ids(root, notebooks)
    archive_selected = set(selected)
    image_selected = set(selected)
    identifiers = {}
    contracts = {}
    for source in notebooks:
        selector = notebook_selector(root, source)
        identifier = notebook_identifier(Path(selector)) if selector is not None else None
        contract = notebook_contract(root, source)
        identifiers[source] = identifier
        contracts[source] = contract
        if not historical and (
            contract.get("historical_inputs", False)
            or (contract.get("study_inputs", False) and not study)
        ):
            archive_selected.discard(identifier)
            image_selected.discard(identifier)
        elif not historical and contract.get("historical_archives", False):
            archive_selected.discard(identifier)
    archives = required_archives(root, archive_selected)
    images = required_images(root, image_selected)
    if not historical:
        for source in notebooks:
            contract = contracts[source]
            identifier = identifiers[source]
            if identifier is None:
                continue
            if contract.get("historical_inputs", False) or (
                contract.get("study_inputs", False) and not study
            ):
                archives.pop(identifier, None)
                images.pop(identifier, None)
            else:
                images.get(identifier, set()).difference_update(
                    root / name for name in contract.get("historical_images", [])
                )
    return archives, images


def required_environment(root: Path, source: Path) -> str | None:
    """Select the locked execution profile when required native modules are absent."""
    contract = notebook_contract(root, source)
    requirements = native_requirements(source) | set(contract["requires"])
    missing = {name for name in requirements if importlib.util.find_spec(name) is None}
    if not missing:
        return None
    return "introduction" if "dolfinx" in requirements else contract["environment"]


def execute_in_environment(root: Path, source: Path, environment: str, flags: list[str]) -> None:
    """Run one notebook in its declared locked profile without recursive dispatch."""
    executable = pixi_executable()
    if executable is None:
        raise ValueError("Native notebook dispatch requires Pixi on PATH")
    argv = [
        executable,
        "run",
        "--locked",
        "-e",
        environment,
        "python",
        "-m",
        "scripts.run_notebooks",
        str(source),
        "--no-dispatch",
        *flags,
    ]
    print(f"Executing notebook in its required environment: {shlex.join(argv)}", flush=True)
    subprocess.run(argv, cwd=root, check=True)


def preparation_plan(
    root: Path,
    notebooks: list[Path],
    data: dict[str, Any],
    *,
    study: bool = False,
    historical: bool = False,
) -> dict[str, Any]:
    """Select complete acquisition recipes only for sources missing declared inputs.

    Commands are argument vectors, never shell fragments. Shared steps execute
    once, in catalogue order; the requested study's numerical levels and spaces
    remain those specified by its public producer.
    """
    from scripts.notebook_data import notebook_identifier, notebook_selector

    steps: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    missing = set(data["missing"]) | set(data.get("missing_images", []))
    for source in notebooks:
        selector = notebook_selector(root, source)
        identifier = (
            notebook_identifier(Path(selector)) if selector else notebook_identifier(source)
        )
        dependencies = set(data["notebooks"].get(identifier, [])) | set(
            data.get("notebook_images", {}).get(identifier, [])
        )
        absent = sorted(dependencies & missing)
        contract = notebook_contract(root, source)
        source_name = (
            source.relative_to(root).as_posix() if source.is_relative_to(root) else str(source)
        )
        recipe = (
            contract.get("current_study", contract["preparation"])
            if study
            else contract["preparation"]
        )
        selected.append(
            dict(
                path=source_name,
                environment=contract["environment"],
                missing=absent,
                current_study=contract.get("current_study", contract["preparation"]),
                limitation=contract.get("limitation"),
                resources=contract.get("resources"),
            )
        )
        if historical and absent:
            unresolved.append(
                dict(
                    notebook=source_name,
                    missing=absent,
                    reason="Original historical payload identities require the original files. "
                    + contract.get("limitation", "Fresh acquisitions have separate provenance."),
                )
            )
            continue
        if not absent and not study:
            continue
        if not recipe:
            if absent:
                unresolved.append(
                    dict(
                        notebook=source_name,
                        missing=absent,
                        reason=contract.get(
                            "limitation",
                            "This notebook has no declared producer for its missing inputs.",
                        ),
                    )
                )
            continue
        for step in recipe:
            if step not in steps:
                steps.append(step)
    return dict(notebooks=selected, preparation=steps, unresolved=unresolved)


def prepare_notebook_inputs(root: Path, plan: dict[str, Any]) -> None:
    """Execute complete downloaded public producers with the current Python.

    No remote field archives, private reference sources or arbitrary notebook
    shell cells are acquired. Existing dataset helpers own URL/checksum checks.
    """
    if plan["unresolved"]:
        details = "\n".join(
            f"{item['notebook']}: {item['reason']} Missing: {', '.join(item['missing'][:3])}"
            for item in plan["unresolved"]
        )
        raise ValueError(
            f"Notebook inputs cannot be prepared from the declared public sources:\n{details}"
        )
    if not plan["preparation"]:
        return
    environment = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYVISTA_OFF_SCREEN": "true",
        "PYMHM_WORKSPACE": str(root),
    }
    acquisition = str(uuid4())
    plan["acquisition_id"] = acquisition
    plan["executed_preparation"] = []
    for step in plan["preparation"]:
        arguments = [value.replace("{acquisition}", acquisition) for value in step["argv"]]
        argv = [sys.executable, *arguments[1:]]
        if len(argv) >= 2 and argv[1] != "-m":
            argv[1] = str(source_file(argv[1], root=root))
        print(f"Preparing notebook inputs: {shlex.join(argv)}", flush=True)
        subprocess.run(argv, cwd=root, env=environment, check=True)
        plan["executed_preparation"].append(
            dict(
                declared_environment=step["environment"],
                environment=os.environ.get("PIXI_ENVIRONMENT_NAME"),
                python_executable=sys.executable,
                argv=argv,
            )
        )


def validate_reproduction_manifest(root: Path, sources: list[Path] | None = None) -> None:
    """Validate actual notebook sources and their downloadable producer contracts.

    Without ``sources`` the checkout catalogue must match all local notebooks.
    An explicit downloaded selection validates its real files and contracts only;
    absent unrelated notebook sources are not claimed as checked.
    """
    from scripts.notebook_data import discover_notebooks, notebook_selector

    manifest = json.loads(REPRODUCTION_MANIFEST.read_text(encoding="utf-8"))
    declared = manifest["notebooks"]
    actual_sources = discover_notebooks(root) if sources is None else sources
    selected = {
        "notebooks/" + selector: path
        for path in actual_sources
        if (selector := notebook_selector(root, path)) is not None
    }
    found = set(selected)
    if set(declared) != found if sources is None else not found <= set(declared):
        raise ValueError(
            "Notebook reproduction catalogue differs from sources: "
            f"missing={sorted(found - set(declared))}, obsolete={sorted(set(declared) - found)}"
        )
    for name in sorted(found):
        contract = declared[name]
        _, private = notebook_imports(selected[name])
        if private:
            raise ValueError(
                f"Notebook imports private package interfaces: {name}: {', '.join(sorted(private))}"
            )
        if contract.get("kind") not in {"standalone", "generated-study", "archived-comparison"}:
            raise ValueError(f"Notebook has no declared reproducibility kind: {name}")
        for step in [*contract["preparation"], *contract.get("current_study", [])]:
            if (
                not isinstance(step["argv"], list)
                or len(step["argv"]) < 2
                or step["argv"][0] != "python"
                or any(not isinstance(value, str) or not value for value in step["argv"])
                or step["environment"]
                not in {"notebooks", "introduction", "fem", "intel", "meshing", "remeshing"}
            ):
                raise ValueError(f"Notebook producer must declare a Python argument vector: {name}")
            argv = step["argv"]
            if argv[1] == "-m":
                if len(argv) < 3:
                    raise ValueError(f"Notebook producer must declare a Python module: {name}")
                try:
                    spec = importlib.util.find_spec(argv[2])
                except (ImportError, ValueError):
                    spec = None
                target = Path(spec.origin) if spec is not None and spec.origin is not None else None
            else:
                target = source_file(argv[1], root=root)
            if target is None or not target.is_file():
                raise ValueError(
                    f"Notebook producer source is not available in this companion: {name}: {argv}"
                )


def main() -> None:
    """Validate every declared notebook without acquiring data or importing FEM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Validate the complete source catalogue"
    )
    parser.parse_args()
    root = case_workspace()
    try:
        validate_reproduction_manifest(root)
    except ValueError as error:
        parser.error(str(error))
    manifest = json.loads(REPRODUCTION_MANIFEST.read_text(encoding="utf-8"))
    counts = {
        kind: sum(contract["kind"] == kind for contract in manifest["notebooks"].values())
        for kind in ("standalone", "generated-study", "archived-comparison")
    }
    print(json.dumps(dict(notebooks=len(manifest["notebooks"]), classifications=counts), indent=2))


if __name__ == "__main__":
    main()
