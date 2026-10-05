"""Release archives retain every runtime feature without repository-only material."""

from __future__ import annotations

import io
import os
import tarfile
import zipfile
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest

_CHECKS = run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_distribution.py"))
package_sources = _CHECKS["package_sources"]
validate_package_sources = _CHECKS["validate_package_sources"]
validate_metadata = _CHECKS["validate_metadata"]
validate_sdist_rebuild = _CHECKS["validate_sdist_rebuild"]

ReleaseFixture = tuple[Path, dict[str, bytes], dict[str, bytes], list[list[str]]]


def _write_release_archives(
    root: Path, wheel_sources: dict[str, bytes], archive_sources: dict[str, bytes]
) -> None:
    """Construct regular wheel and sdist members without extracting archive inputs."""
    (root / "dist").mkdir(exist_ok=True)
    with zipfile.ZipFile(root / "dist/pymhm-0.1.0-py3-none-any.whl", "w") as archive:
        for name, payload in wheel_sources.items():
            archive.writestr(name, payload)
    with tarfile.open(root / "dist/pymhm-0.1.0.tar.gz", "w:gz") as archive:
        for name, payload in archive_sources.items():
            member = tarfile.TarInfo("pymhm-0.1.0/" + name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


@pytest.fixture
def release_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReleaseFixture:
    """Supply independent release archives and capture rebuilding/Twine gate ordering."""
    source_payloads = {
        "src/pymhm/__init__.py": b"entry point",
        "src/pymhm/py.typed": b"",
        "src/pymhm/core/assembly.py": b"assembly implementation",
        "src/pymhm/core/assembly.pyi": b"assembly typing stub",
    }
    build_payloads = {
        "pyproject.toml": (
            b'[project]\nname = "pymhm"\nversion = "0.1.0"\nlicense = "LGPL-2.1-only"\n'
            b'license-files = ["LICENSE"]\nrequires-python = ">=3.11,<3.14"\n'
            b'dependencies = ["numpy>=1.26"]\n'
            b'[project.optional-dependencies]\namg = ["pyamg>=5.3"]\n'
            b"intel = [\"pypardiso>=0.4; sys_platform == 'win32'\"]\n"
        ),
        "README.md": b"package readme",
        "LICENSE": b"license",
        ".gitignore": b"__pycache__/\n",
    }
    for name, payload in {**source_payloads, **build_payloads}.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    metadata = (
        b"Name: pymhm\nVersion: 0.1.0\nLicense-Expression: LGPL-2.1-only\n"
        b"License-File: LICENSE\nRequires-Python: <3.14,>=3.11\n"
        b"Requires-Dist: numpy>=1.26\nProvides-Extra: amg\nProvides-Extra: intel\n"
        b"Requires-Dist: pyamg>=5.3; extra == 'amg'\n"
        b"Requires-Dist: pypardiso>=0.4; sys_platform == 'win32' and extra == 'intel'\n"
    )
    wheel_sources = {
        **{name.removeprefix("src/"): payload for name, payload in source_payloads.items()},
        "pymhm-0.1.0.dist-info/METADATA": metadata,
        "pymhm-0.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
        "pymhm-0.1.0.dist-info/RECORD": b"record",
        "pymhm-0.1.0.dist-info/licenses/LICENSE": build_payloads["LICENSE"],
    }
    archive_sources = {**source_payloads, **build_payloads, "PKG-INFO": metadata}
    invocations: list[list[str]] = []

    def capture_rebuild(sdist: Path, wheel: Path, version: str) -> None:
        """Record independent rebuilding after all content validation succeeds."""
        invocations.append(["rebuild", str(sdist), str(wheel), version])

    def capture_metadata_check(command: list[str], *, check: bool) -> None:
        """Record the final Twine command after the rebuild contract succeeds."""
        assert check is True
        invocations.append(command)

    globals_ = _CHECKS["main"].__globals__
    monkeypatch.setitem(globals_, "__file__", str(tmp_path / "scripts/check_distribution.py"))
    monkeypatch.setitem(globals_, "validate_sdist_rebuild", capture_rebuild)
    monkeypatch.setitem(globals_, "subprocess", SimpleNamespace(run=capture_metadata_check))
    return tmp_path, wheel_sources, archive_sources, invocations


def test_recursive_runtime_sources_include_typing_stubs_and_marker(tmp_path: Path) -> None:
    """Source collection includes nested implementation/typing while ignoring checkout caches."""
    package = tmp_path / "src/pymhm"
    (package / "core").mkdir(parents=True)
    sources = {
        "pymhm/__init__.py": b"module",
        "pymhm/__init__.pyi": b"stub",
        "pymhm/py.typed": b"",
        "pymhm/core/assembly.py": b"assembly",
        "pymhm/core/assembly.pyi": b"assembly stub",
    }
    for name, payload in {**sources, "pymhm/core/cached.pyc": b"cache"}.items():
        (tmp_path / "src" / name).write_bytes(payload)
    assert package_sources(tmp_path) == sources
    validate_package_sources(dict(reversed(list(sources.items()))), sources, "Wheel")


@pytest.mark.parametrize("problem", ["empty", "missing", "extra", "changed"])
def test_runtime_distribution_requires_exact_source_names_and_bytes(problem: str) -> None:
    """A smaller artifact cannot discard, add or replace implementation and typing files."""
    sources = {"pymhm/core/assembly.py": b"source", "pymhm/__init__.pyi": b"stub"}
    archived = dict(sources)
    if problem == "empty":
        sources.clear()
    elif problem == "missing":
        del archived["pymhm/__init__.pyi"]
    elif problem == "extra":
        archived["pymhm/obsolete.py"] = b"old"
    else:
        archived["pymhm/core/assembly.py"] = b"changed"
    with pytest.raises(SystemExit, match="No runtime sources|file names differ|file bytes differ"):
        validate_package_sources(archived, sources, "Wheel")


@pytest.mark.parametrize(
    "problem",
    [
        "valid",
        "missing_wheel_source",
        "changed_wheel_source",
        "missing_sdist_source",
        "changed_sdist_stub",
        "changed_readme",
        "changed_license",
        "missing_pkg_info",
        "wheel_asset",
        "wheel_metadata_extra",
        "docs",
        "scripts",
        "examples",
        "recipe",
        "notebooks",
        "tests",
        "benchmarks",
        "pixi",
        "roadmap",
        "external_source",
    ],
)
def test_release_gate_keeps_only_runtime_and_build_inputs(
    release_fixture: ReleaseFixture, problem: str
) -> None:
    """Exercise real ZIP/tar gates before permitting rebuilds or publication checks."""
    root, wheel_sources, archive_sources, invocations = release_fixture
    if problem == "missing_wheel_source":
        del wheel_sources["pymhm/core/assembly.py"]
    elif problem == "changed_wheel_source":
        wheel_sources["pymhm/core/assembly.py"] = b"changed"
    elif problem == "missing_sdist_source":
        del archive_sources["src/pymhm/core/assembly.py"]
    elif problem == "changed_sdist_stub":
        archive_sources["src/pymhm/core/assembly.pyi"] = b"changed"
    elif problem == "changed_readme":
        archive_sources["README.md"] = b"changed"
    elif problem == "changed_license":
        wheel_sources["pymhm-0.1.0.dist-info/licenses/LICENSE"] = b"changed"
    elif problem == "missing_pkg_info":
        del archive_sources["PKG-INFO"]
    elif problem == "wheel_asset":
        wheel_sources["pymhm/notebooks/demo.ipynb"] = b"notebook"
    elif problem == "wheel_metadata_extra":
        wheel_sources["pymhm-0.1.0.dist-info/extra.json"] = b"{}"
    elif problem != "valid":
        archive_sources[f"{problem}/unneeded.txt"] = b"repository-only payload"
    _write_release_archives(root, wheel_sources, archive_sources)
    if problem == "valid":
        _CHECKS["main"]()
        assert invocations[0] == [
            "rebuild",
            str(root / "dist/pymhm-0.1.0.tar.gz"),
            str(root / "dist/pymhm-0.1.0-py3-none-any.whl"),
            "0.1.0",
        ]
        assert invocations[1][1:5] == ["-m", "twine", "check", "--strict"]
        assert len(invocations) == 2
    else:
        with pytest.raises(SystemExit, match="file names differ|file bytes differ"):
            _CHECKS["main"]()
        assert not invocations


@pytest.mark.parametrize(
    "field",
    [
        "Name",
        "Version",
        "License-Expression",
        "License-File",
        "Requires-Python",
        "Provides-Extra",
        "Requires-Dist",
    ],
)
def test_metadata_cannot_lose_identity_or_any_dependency(
    release_fixture: ReleaseFixture, field: str
) -> None:
    """Omitted identity, Python support and optional dependencies fail in either metadata file."""
    root, wheel_sources, _, _ = release_fixture
    project = _CHECKS["tomllib"].loads((root / "pyproject.toml").read_text())["project"]
    metadata = wheel_sources["pymhm-0.1.0.dist-info/METADATA"]
    changed = b"\n".join(
        line for line in metadata.splitlines() if not line.startswith(field.encode() + b":")
    )
    with pytest.raises(SystemExit, match="differs|differ"):
        validate_metadata(changed, project, "Distribution")


@pytest.mark.parametrize("problem", ["duplicate", "link"])
def test_source_archive_rejects_ambiguous_or_nonregular_members(
    tmp_path: Path, problem: str
) -> None:
    """A filename allowlist cannot legitimize duplicate content or linked build inputs."""
    archive_path = tmp_path / "source.tar.gz"
    name = "pymhm-0.1.0/src/pymhm/__init__.py"
    with tarfile.open(archive_path, "w:gz") as archive:
        regular = tarfile.TarInfo(name)
        archive.addfile(regular, io.BytesIO(b""))
        member = tarfile.TarInfo(name if problem == "duplicate" else "pymhm-0.1.0/LICENSE")
        if problem == "link":
            member.type = tarfile.SYMTYPE
            member.linkname = "outside-license"
        archive.addfile(member, io.BytesIO(b""))
    with pytest.raises(SystemExit, match="duplicate archive members|invalid member"):
        _CHECKS["source_archive_files"](archive_path, "pymhm-0.1.0/")
    if problem == "duplicate":
        with pytest.raises(SystemExit, match="duplicate archive members"):
            _CHECKS["validate_archive_names"]([name, name], [name], "Wheel")


@pytest.mark.parametrize("problem", ["valid", "missing", "changed"])
def test_sdist_rebuild_uses_own_sources_and_preserves_entire_wheel(
    release_fixture: ReleaseFixture, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    """Run the rebuild adapter outside the checkout, then verify every ZIP member's bytes."""
    root, wheel_sources, archive_sources, _ = release_fixture
    _write_release_archives(root, wheel_sources, archive_sources)
    monkeypatch.setenv("PYTHONPATH", "/forbidden-checkout-import")
    invocations = []

    def build_from_extracted_source(
        command: list[str], *, cwd: Path, env: dict[str, str], check: bool
    ) -> None:
        """Emulate the build backend while asserting the clean-source subprocess contract."""
        assert cwd != root and not cwd.is_relative_to(root)
        assert {
            path.relative_to(cwd).as_posix(): path.read_bytes()
            for path in cwd.rglob("*")
            if path.is_file()
        } == archive_sources
        assert "PYTHONPATH" not in env and os.environ["PYTHONPATH"] == "/forbidden-checkout-import"
        assert check is True and command[1:5] == ["-m", "build", "--wheel", "--no-isolation"]
        output = Path(command[-1])
        output.mkdir()
        rebuilt_sources = dict(wheel_sources)
        if problem == "missing":
            del rebuilt_sources["pymhm/core/assembly.pyi"]
        elif problem == "changed":
            rebuilt_sources["pymhm-0.1.0.dist-info/METADATA"] = b"changed metadata"
        with zipfile.ZipFile(output / "pymhm-0.1.0-py3-none-any.whl", "w") as archive:
            for name, payload in rebuilt_sources.items():
                archive.writestr(name, payload)
        invocations.append(command)

    monkeypatch.setitem(
        validate_sdist_rebuild.__globals__,
        "subprocess",
        SimpleNamespace(run=build_from_extracted_source),
    )
    arguments = (
        root / "dist/pymhm-0.1.0.tar.gz",
        root / "dist/pymhm-0.1.0-py3-none-any.whl",
        "0.1.0",
    )
    if problem == "valid":
        validate_sdist_rebuild(*arguments)
        assert len(invocations) == 1
        assert not Path(invocations[0][-1]).exists()
    else:
        with pytest.raises(SystemExit, match="file names differ|file bytes differ"):
            validate_sdist_rebuild(*arguments)
