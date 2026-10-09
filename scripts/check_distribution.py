"""Verify library-only release archives preserve every runtime and typing file."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Collection, Mapping
from email.parser import Parser
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

MAX_ARTIFACT_BYTES = 100_000_000
SOURCE_BUILD_FILES = frozenset({"pyproject.toml", "README.md", "LICENSE"})
WHEEL_METADATA_FILES = frozenset({"METADATA", "WHEEL", "RECORD", "licenses/LICENSE"})


def package_sources(root: Path) -> dict[str, bytes]:
    """Read library modules and typing under their installed ``pymhm`` paths."""
    return {
        path.relative_to(root / "src").as_posix(): path.read_bytes()
        for path in sorted((root / "src/pymhm").rglob("*"))
        if path.is_file() and (path.suffix in {".py", ".pyi"} or path.name == "py.typed")
    }


def validate_library_build(configuration: Mapping[str, Any]) -> None:
    """Require the library-only wheel and independently rebuildable source allowlists.

    Notebook sources, companion modules, case registries, fields and datasets
    belong to repository and documentation downloads. Forced resource mappings
    are forbidden at every Hatch build level, including unused custom targets.
    """
    build = configuration.get("tool", {}).get("hatch", {}).get("build", {})
    targets = build.get("targets", {})
    if "force-include" in build or any("force-include" in value for value in targets.values()):
        raise SystemExit("Library releases must not declare force-include mappings")
    if targets.get("wheel", {}).get("packages") != ["src/pymhm"]:
        raise SystemExit("Wheel must contain only the src/pymhm package")
    source_names = targets.get("sdist", {}).get("only-include", [])
    required = {"src/pymhm", *SOURCE_BUILD_FILES}
    if set(source_names) not in (required, required | {".gitignore"}) or len(source_names) != len(
        set(source_names)
    ):
        raise SystemExit("Source archive must contain only the library and build metadata")


def validate_archive_names(names: Collection[str], expected: Collection[str], label: str) -> None:
    """Require the exact file allowlist, including generated distribution metadata."""
    missing, extra = set(expected) - set(names), set(names) - set(expected)
    if len(names) != len(set(names)):
        raise SystemExit(f"{label} contains duplicate archive members")
    if missing or extra:
        raise SystemExit(
            f"{label} file names differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def validate_package_sources(
    archived: Mapping[str, bytes], expected: Mapping[str, bytes], label: str
) -> None:
    """Reject omitted, extra or changed runtime modules, stubs and typing markers.

    Both mappings use installed names such as ``pymhm/core/assembly.py``. Every
    file inside the archived package participates, so undeclared data and caches
    cannot enter the distribution alongside the verified implementation.
    """
    if not expected:
        raise SystemExit("No runtime sources were found for distribution validation")
    validate_archive_names(list(archived), list(expected), f"{label} runtime source")
    _validate_payloads(archived, expected, label)


def _validate_payloads(
    archived: Mapping[str, bytes], expected: Mapping[str, bytes], label: str
) -> None:
    """Compare bytes after the archive's file names have been validated."""
    changed = sorted(name for name in expected if archived[name] != expected[name])
    if changed:
        raise SystemExit(f"{label} file bytes differ: {changed}")


def expected_requirements(project: Mapping[str, Any]) -> set[str]:
    """Normalize mandatory and optional PEP 508 requirements with their extra markers."""
    requirements = {str(Requirement(entry)) for entry in project.get("dependencies", [])}
    for extra, entries in project.get("optional-dependencies", {}).items():
        for entry in entries:
            requirement = Requirement(entry)
            marker = requirement.marker
            requirement.marker = None
            condition = f'({marker}) and extra == "{extra}"' if marker else f'extra == "{extra}"'
            requirements.add(str(Requirement(f"{requirement}; {condition}")))
    return requirements


def validate_metadata(payload: bytes, project: Mapping[str, Any], label: str) -> None:
    """Keep package identity, license, supported Python and every extra in both archives."""
    metadata = Parser().parsestr(payload.decode("utf-8"))
    for field, expected in (
        ("Name", project["name"]),
        ("Version", project["version"]),
        ("License-Expression", project["license"]),
    ):
        if metadata[field] != expected:
            raise SystemExit(f"{label} {field} differs from source metadata")
    if set(metadata.get_all("License-File", [])) != set(project["license-files"]):
        raise SystemExit(f"{label} license files differ from source metadata")
    if SpecifierSet(metadata.get("Requires-Python", "")) != SpecifierSet(
        project["requires-python"]
    ):
        raise SystemExit(f"{label} Python requirement differs from source metadata")
    if set(metadata.get_all("Provides-Extra", [])) != set(project.get("optional-dependencies", {})):
        raise SystemExit(f"{label} optional features differ from source metadata")
    requirements = {str(Requirement(entry)) for entry in metadata.get_all("Requires-Dist", [])}
    if requirements != expected_requirements(project):
        raise SystemExit(f"{label} dependencies differ from source metadata")


def source_archive_files(archive_path: Path, prefix: str) -> dict[str, bytes]:
    """Read regular sdist files without extraction, rejecting duplicate or misplaced members."""
    files = {}
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.name.startswith(prefix) or not member.isfile():
                raise SystemExit(f"Source archive contains an invalid member: {member.name}")
            name = member.name.removeprefix(prefix)
            if name in files:
                raise SystemExit(f"Source archive contains duplicate archive members: {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise SystemExit(f"Source archive lacks a regular file payload: {name}")
            with stream:
                files[name] = stream.read()
    return files


def validate_sdist_rebuild(sdist: Path, wheel: Path, version: str) -> None:
    """Rebuild the wheel from the sdist alone with locked build dependencies.

    Building runs outside the repository, without ``PYTHONPATH``. The installed
    payload and generated metadata must exactly match the wheel built from the
    checkout; ZIP compression and timestamps do not participate in comparison.
    """
    with tempfile.TemporaryDirectory(prefix="pymhm-sdist-check-") as directory:
        temporary = Path(directory)
        with tarfile.open(sdist) as archive:
            archive.extractall(temporary, filter="data")
        source = temporary / f"pymhm-{version}"
        output = temporary / "dist"
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(output),
            ],
            cwd=source,
            env=environment,
            check=True,
        )
        rebuilt = output / wheel.name
        with zipfile.ZipFile(wheel) as expected, zipfile.ZipFile(rebuilt) as actual:
            validate_archive_names(actual.namelist(), expected.namelist(), "Rebuilt wheel")
            _validate_payloads(
                {name: actual.read(name) for name in actual.namelist()},
                {name: expected.read(name) for name in expected.namelist()},
                "Rebuilt wheel",
            )


def main() -> None:
    """Validate exact wheel/sdist contents, features and independent sdist rebuilding."""
    root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = configuration["project"]
    version = project["version"]
    sources = package_sources(root)
    validate_library_build(configuration)
    if not {"pymhm/__init__.py", "pymhm/py.typed"}.issubset(sources):
        raise SystemExit("Runtime source lacks the package entry point or typing marker")
    wheel = root / "dist" / f"pymhm-{version}-py3-none-any.whl"
    sdist = root / "dist" / f"pymhm-{version}.tar.gz"
    if not wheel.is_file() or not sdist.is_file():
        raise SystemExit("Build the wheel and source archive before checking distributions")
    for artifact in (wheel, sdist):
        if artifact.stat().st_size > MAX_ARTIFACT_BYTES:
            raise SystemExit(
                f"{artifact.name} exceeds the {MAX_ARTIFACT_BYTES:,}-byte release limit"
            )
    metadata_prefix = f"pymhm-{version}.dist-info/"
    with zipfile.ZipFile(wheel) as archive:
        validate_archive_names(
            archive.namelist(),
            {*sources, *(metadata_prefix + name for name in WHEEL_METADATA_FILES)},
            "Wheel",
        )
        validate_package_sources({name: archive.read(name) for name in sources}, sources, "Wheel")
        validate_metadata(archive.read(metadata_prefix + "METADATA"), project, "Wheel")
        _validate_payloads(
            {"LICENSE": archive.read(metadata_prefix + "licenses/LICENSE")},
            {"LICENSE": (root / "LICENSE").read_bytes()},
            "Wheel license",
        )
    files = source_archive_files(sdist, f"pymhm-{version}/")
    build_files = {name: (root / name).read_bytes() for name in SOURCE_BUILD_FILES}
    if (root / ".gitignore").is_file():
        build_files[".gitignore"] = (root / ".gitignore").read_bytes()
    validate_archive_names(
        list(files),
        {
            *build_files,
            *("src/" + name for name in sources),
            "PKG-INFO",
        },
        "Source archive",
    )
    validate_package_sources(
        {
            name.removeprefix("src/"): payload
            for name, payload in files.items()
            if name.removeprefix("src/") in sources
        },
        sources,
        "Source archive",
    )
    _validate_payloads(files, build_files, "Source archive build inputs")
    validate_metadata(files["PKG-INFO"], project, "Source archive")
    validate_sdist_rebuild(sdist, wheel, version)
    subprocess.run(
        [sys.executable, "-m", "twine", "check", "--strict", str(wheel), str(sdist)],
        check=True,
    )
    print(
        f"Validated pymhm {version}: {len(sources)} library Python/typing files; "
        "no case payloads; independent sdist rebuild"
    )


if __name__ == "__main__":
    main()
