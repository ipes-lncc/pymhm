"""Build checksum-addressed notebook companions outside the library distribution."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import json
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

DOWNLOAD_BASE = "https://ipes-lncc.github.io/pymhm/downloads/"
SMALL_CONFIGS = frozenset(
    {
        "examples/data/reconstruction3d-macro.json",
        "examples/data/three-layer-2017/case.json",
        "examples/data/three-layer-2017/macro-mesh.json",
        "examples/results/hpc4e/dataset.json",
        "examples/results/marmousi/dataset.json",
        "examples/results/spe10/dataset.json",
    }
)
NOTEBOOK_TOOLS = frozenset(
    {
        "scripts/__init__.py",
        "scripts/run_notebooks.py",
        "scripts/notebook_data.py",
        "scripts/notebook_reproduction.py",
        "scripts/notebook_images.json",
        "scripts/notebook_reproduction.json",
    }
)
RESOURCE_SCOPE_KEYS = (
    "notebook_resources",
    "notebook_historical_resources",
    "notebook_study_resources",
)


def _label(name: str) -> PurePosixPath:
    """Require a regular relative POSIX label before accessing a source file."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Invalid companion source path: {name}")
    return path


def _read(root: Path, name: str) -> bytes:
    """Read a declared regular repository file without following source symlinks."""
    _label(name)
    path = root / name
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Companion source must be a regular repository file: {name}")
    return path.read_bytes()


def _module_file(root: Path, name: str) -> str | None:
    """Resolve only repository example and notebook-tool Python modules."""
    if name.split(".", 1)[0] not in {"examples", "scripts"}:
        return None
    path = name.replace(".", "/")
    for relative in (path + ".py", path + "/__init__.py"):
        if (root / relative).is_file():
            return relative
    return None


def _dependencies(root: Path, source: str, module: str) -> set[str]:
    """Inspect imports, literal dynamic imports and explicit companion source identities."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parts = module.split(".")[:-1]
            base = ".".join(parts[: len(parts) - node.level + 1]) if node.level else ""
            imported = ".".join(part for part in (base, node.module) if part)
            names.add(imported)
            names.update(imported + "." + alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            called = (
                function.id if isinstance(function, ast.Name) else getattr(function, "attr", "")
            )
            argument = node.args[0]
            if (
                called == "import_module"
                and isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
            ):
                names.add(argument.value)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("examples/")
            and node.value.endswith(".py")
        ):
            names.add(node.value.removesuffix(".py").replace("/", "."))
    return {relative for name in names if (relative := _module_file(root, name)) is not None}


def module_sources(root: Path, modules: Iterable[str]) -> list[str]:
    """Resolve the transitive public companion imports for the stated modules.

    Installed ``pymhm`` algorithms and third-party imports stay outside the ZIP.
    Every example/script package initializer participates. Literal dynamic
    imports and example source labels used for scientific provenance participate
    as well; computed dependency names require their own explicit seed module.
    """
    pending = set()
    for module in modules:
        relative = _module_file(root, module)
        if relative is None:
            raise ValueError(f"Companion module is unavailable: {module}")
        pending.add(relative)
    found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        payload = _read(root, name)
        found.add(name)
        path = PurePosixPath(name)
        for parent in path.parents:
            initial = (parent / "__init__.py").as_posix()
            if str(parent) != "." and (root / initial).is_file():
                pending.add(initial)
        module = name.removesuffix(".py").replace("/", ".")
        pending.update(_dependencies(root, payload.decode("utf-8"), module) - found)
    return sorted(found)


def build_companion(
    root: Path,
    source_files: Iterable[str],
    output: Path,
    *,
    name: str,
    resources: Mapping[str, Mapping[str, Any]] | None = None,
    unavailable: Mapping[str, Mapping[str, Any]] | None = None,
    scopes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export exact source/configuration bytes in a deterministic, verified companion ZIP.

    The archive includes local SHA-256 identities and selected remote resource
    descriptions, never the remote payloads themselves. Notebook sources,
    generated records, datasets, fields, images, installed library sources and
    external reference tools are excluded from the companion. Extraction alone
    does not execute its Python code. ``output`` receives checksum-addressed
    downloads; the returned descriptor contains the public URL and literal bytes.
    """
    label = _label(name)
    if len(label.parts) != 1 or label.suffix != ".zip":
        raise ValueError("Companion download name must be a ZIP basename")
    payloads = {}
    for relative in sorted(set(source_files)):
        path = _label(relative)
        allowed = (
            path.parts[0] in {"examples", "scripts"}
            and path.suffix == ".py"
            or relative in SMALL_CONFIGS | NOTEBOOK_TOOLS | {"LICENSE"}
        )
        if not allowed:
            raise ValueError(f"Unsupported companion payload: {relative}")
        payloads[relative] = _read(root, relative)
    local = {
        relative: {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        for relative, payload in payloads.items()
    }
    selected = {relative: dict(record) for relative, record in (resources or {}).items()}
    for relative in selected:
        _label(relative)
    if scopes is not None:
        payload = (json.dumps(scopes, sort_keys=True, indent=2) + "\n").encode()
        relative = ".pymhm-notebook-resources.json"
        payloads[relative] = payload
        local[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    conflicts = set(local) & set(selected)
    if any(local[relative]["sha256"] != selected[relative]["sha256"] for relative in conflicts):
        raise ValueError("Companion local and remote identities conflict")
    registry = {"resources": {**selected, **local}, "unavailable": dict(unavailable or {})}
    payloads[".pymhm-resources.json"] = (
        json.dumps(registry, sort_keys=True, indent=2) + "\n"
    ).encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, payload in sorted(payloads.items()):
            member = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, payload, compresslevel=9)
    payload = stream.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    destination = output / "downloads" / digest / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "url": DOWNLOAD_BASE + digest + "/" + name,
        "sha256": digest,
        "size_bytes": len(payload),
        "source_files": {relative: record["sha256"] for relative, record in sorted(local.items())},
    }


def _referenced_resources(root: Path, files: Iterable[str], available: set[str]) -> set[str]:
    """Describe available literal inputs referenced by the selected case helper closure.

    Literal paths and filename templates can identify a transitive CSV/JSON input
    that the notebook does not read directly. They add external catalogue entries,
    never payloads or eager staging. Computed directory/filename dependencies need
    explicit notebook selection metadata. General notebook-tool selectors do not
    expand a case's input catalogue.
    """

    def literal_path(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(
                part.value
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
                else "*"
                for part in node.values
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left, right = literal_path(node.left), literal_path(node.right)
            if right is not None:
                return left.rstrip("/") + "/" + right.lstrip("/") if left else right
        return None

    names: set[str] = set()
    for file in files:
        if not file.startswith("examples/") or not file.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(_read(root, file).decode("utf-8"))):
            pattern = literal_path(node)
            if pattern in available:
                names.add(pattern)
            elif pattern is not None and "*" in pattern and "/" in pattern:
                names.update(name for name in available if fnmatch.fnmatchcase(name, pattern))
    return names


def _selected_resources(manifest: Mapping[str, Any], names: set[str]) -> dict[str, dict[str, Any]]:
    """Include the selected remote inputs and the complete closure of their notices."""
    result = {}
    pending = set(names)
    while pending:
        name = pending.pop()
        if name in result or name not in manifest["remote"]:
            continue
        record = dict(manifest["remote"][name])
        result[name] = record
        pending.update(record.get("related_resources", []))
    return dict(sorted(result.items()))


def build_notebook_companions(root: Path, output: Path) -> dict[str, dict[str, Any]]:
    """Build every declared notebook's helper closure without reading its computed inputs."""
    manifest = json.loads((root / "examples/resource_manifest.json").read_text())
    recipes = json.loads((root / "scripts/notebook_reproduction.json").read_text())["notebooks"]
    descriptors = {}
    for selector in sorted(manifest["notebook_resources"]):
        notebook = json.loads(_read(root, "notebooks/" + selector))
        modules = {
            "scripts.run_notebooks",
            "scripts.notebook_data",
            "scripts.notebook_reproduction",
        }
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            source = "".join(source) if isinstance(source, list) else source
            source = "\n".join(
                "" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines()
            )
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(
                        alias.name for alias in node.names if alias.name.startswith("examples")
                    )
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and (node.module == "examples" or node.module.startswith("examples."))
                ):
                    modules.add(node.module)
                    modules.update(
                        node.module + "." + alias.name
                        for alias in node.names
                        if _module_file(root, node.module + "." + alias.name)
                    )
        contract = recipes.get("notebooks/" + selector, {})
        for key in ("preparation", "current_study"):
            for command in contract.get(key, []):
                for token in command["argv"]:
                    if token.startswith("examples.") and _module_file(root, token):
                        modules.add(token)
                    elif token.startswith("examples/") and token.endswith(".py"):
                        modules.add(token.removesuffix(".py").replace("/", "."))
        scopes = {
            key: {selector: manifest.get(key, {}).get(selector, [])} for key in RESOURCE_SCOPE_KEYS
        }
        names = {name for selected in scopes.values() for name in selected[selector]}
        files = (
            set(module_sources(root, modules))
            | set(SMALL_CONFIGS)
            | set(NOTEBOOK_TOOLS)
            | {"LICENSE"}
        )
        names.update(_referenced_resources(root, files, set(manifest["remote"])))
        resources = _selected_resources(manifest, names)
        unavailable = {
            name: manifest["unavailable"][name] for name in names & set(manifest["unavailable"])
        }
        descriptors[selector] = build_companion(
            root,
            files,
            output,
            name=PurePosixPath(selector).stem + "-companion.zip",
            resources=resources,
            unavailable=unavailable,
            scopes=scopes,
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "notebook-companions.json").write_text(
        json.dumps({"schema_version": 1, "companions": descriptors}, sort_keys=True, indent=2)
        + "\n"
    )
    return descriptors


def validate_notebook_companion_pins(
    root: Path, companions: Mapping[str, Mapping[str, Any]], *, require_pins: bool = False
) -> None:
    """Check literal notebook download declarations against the built companion bytes.

    AST literals include multiline and parenthesized assignments. Missing pins
    can be allowed while drafting a source; website exports require both pins
    for every declared companion before publishing the notebook files.
    """
    for selector, descriptor in companions.items():
        notebook = json.loads(_read(root, "notebooks/" + selector))
        pins = {}
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            source = "".join(source) if isinstance(source, list) else source
            source = "\n".join(
                "" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines()
            )
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in {
                            "COMPANION_URL",
                            "COMPANION_SHA256",
                        }:
                            pins[target.id] = ast.literal_eval(node.value)
        expected = {"COMPANION_URL": descriptor["url"], "COMPANION_SHA256": descriptor["sha256"]}
        if (pins or require_pins) and pins != expected:
            raise ValueError(f"Notebook companion pins differ from the built archive: {selector}")


def refresh_resource_manifest(root: Path) -> None:
    """Refresh the repository-only source/configuration identities after notebook pins change.

    Remote field, figure and dataset identities remain untouched. This operation
    does not change notebook cells, acquire data or install a case registry.
    """
    path = root / "examples/resource_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["bundled"] = {
        name: hashlib.sha256(_read(root, name)).hexdigest() for name in sorted(manifest["bundled"])
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    """Build website companions without changing notebook source cells or installing cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/docs"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = build_notebook_companions(root, arguments.output)
    print(f"Built {len(result)} separate notebook companions")


if __name__ == "__main__":
    main()
