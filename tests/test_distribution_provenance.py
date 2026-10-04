"""Release metadata must preserve source digests without private filesystem paths."""

import io
import tarfile
import zipfile
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest

_CHECKS = run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "check_distribution.py"))
validate_json_provenance = _CHECKS["validate_json_provenance"]
validate_compact_layer = _CHECKS["validate_compact_layer"]
package_sources = _CHECKS["package_sources"]
validate_package_sources = _CHECKS["validate_package_sources"]
notebook_sources = _CHECKS["notebook_sources"]
validate_notebook_sources = _CHECKS["validate_notebook_sources"]
validate_wheel_notebooks = _CHECKS["validate_wheel_notebooks"]


def test_recursive_notebook_sources_include_catalogue_and_ignore_checkpoints(
    tmp_path: Path,
) -> None:
    """Notebook release inputs retain nested family paths without editor checkpoints."""
    sources = {
        "notebooks/darcy/primal.ipynb": b"primal notebook",
        "notebooks/waves/maxwell/cavity.ipynb": b"Maxwell notebook",
        "notebooks/catalogue.json": b'{"entries": []}',
        "notebooks/README.md": b"Notebook catalogue",
    }
    for name, payload in {
        **sources,
        "notebooks/darcy/.ipynb_checkpoints/primal-checkpoint.ipynb": b"editor copy",
        "notebooks/darcy/assets/figure.png": b"image",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    assert notebook_sources(tmp_path) == sources
    validate_notebook_sources(dict(reversed(list(sources.items()))), sources, "Source archive")


@pytest.mark.parametrize("name", sorted(_CHECKS["NOTEBOOK_METADATA_FILES"]))
def test_notebook_checkout_requires_both_catalogue_files(tmp_path: Path, name: str) -> None:
    """A missing public index cannot reduce the set of expected release inputs."""
    for required in _CHECKS["NOTEBOOK_METADATA_FILES"] - {name}:
        path = tmp_path / required
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"catalogue")
    with pytest.raises(SystemExit, match="catalogue files are missing"):
        notebook_sources(tmp_path)


@pytest.mark.parametrize("problem", ["empty", "missing", "extra", "changed"])
def test_notebook_distribution_requires_exact_nested_names_and_bytes(problem: str) -> None:
    """Nested omission or byte modification fails the same contract as runtime sources."""
    sources = {
        "notebooks/waves/maxwell/cavity.ipynb": b"notebook",
        "notebooks/catalogue.json": b"catalogue",
        "notebooks/README.md": b"index",
    }
    archived = dict(sources)
    if problem == "empty":
        sources.pop("notebooks/waves/maxwell/cavity.ipynb")
    elif problem == "missing":
        del archived["notebooks/waves/maxwell/cavity.ipynb"]
    elif problem == "extra":
        archived["examples/obsolete.ipynb"] = b"old"
    else:
        archived["notebooks/waves/maxwell/cavity.ipynb"] = b"changed"
    with pytest.raises(
        SystemExit, match="No source notebooks|source names differ|source bytes differ"
    ):
        validate_notebook_sources(archived, sources, "Source archive")


@pytest.mark.parametrize(
    "name", ["pymhm/examples/demo.ipynb", "notebooks/catalogue.json", "notebooks/README.md"]
)
def test_notebooks_and_their_catalogue_cannot_enter_runtime_wheel(name: str) -> None:
    """A notebook hidden inside the package is also outside the runtime payload contract."""
    validate_wheel_notebooks(["pymhm/__init__.py", "pymhm/py.typed"])
    with pytest.raises(SystemExit, match="Wheel contains notebook documents"):
        validate_wheel_notebooks(["pymhm/__init__.py", name])


def _write_release_archives(
    root: Path, wheel_sources: dict[str, bytes], archive_sources: dict[str, bytes]
) -> None:
    """Construct regular wheel and sdist members without extracting untrusted archives."""
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
def notebook_release_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, bytes], dict[str, bytes], list[tuple[list[str], bool]]]:
    """Provide independent release archives and capture the final Twine invocation."""
    repository = Path(__file__).resolve().parents[1]
    source_payloads = {
        "src/pymhm/__init__.py": b"",
        "src/pymhm/py.typed": b"",
        "notebooks/darcy/primal.ipynb": b'{"cells": []}',
        "notebooks/waves/maxwell/cavity.ipynb": b'{"cells": []}',
        "notebooks/catalogue.json": b'{"entries": []}',
        "notebooks/README.md": b"Notebook catalogue",
    }
    for name, payload in source_payloads.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
    wheel_sources = {
        "pymhm/__init__.py": b"",
        "pymhm/py.typed": b"",
        "pymhm-0.1.0.dist-info/METADATA": (
            b"Name: pymhm\nVersion: 0.1.0\nLicense-Expression: LGPL-2.1-only\n"
        ),
        "pymhm-0.1.0.dist-info/licenses/LICENSE": b"license",
    }
    archive_sources = {
        **source_payloads,
        "pyproject.toml": (tmp_path / "pyproject.toml").read_bytes(),
        "LICENSE": b"license",
        "README.md": b"package readme",
        "tests/test_elasticity_mixed_fenics.py": b"",
        "tests/test_darcy_bdm_fenics.py": b"",
        **{name: (repository / name).read_bytes() for name in _CHECKS["COMPACT_RESERVOIR_LAYERS"]},
    }
    invocations = []

    def capture_metadata_check(command: list[str], *, check: bool) -> None:
        """Record the command after all archive-content checks have succeeded."""
        invocations.append((command, check))

    globals_ = _CHECKS["main"].__globals__
    monkeypatch.setitem(globals_, "__file__", str(tmp_path / "scripts/check_distribution.py"))
    monkeypatch.setitem(globals_, "subprocess", SimpleNamespace(run=capture_metadata_check))
    return tmp_path, wheel_sources, archive_sources, invocations


@pytest.mark.parametrize(
    "problem",
    [
        "valid",
        "missing_nested",
        "tampered_nested",
        "missing_catalogue",
        "tampered_readme",
        "wheel_notebook",
    ],
)
def test_release_gate_validates_notebook_payloads_before_metadata_check(
    notebook_release_fixture: tuple[
        Path, dict[str, bytes], dict[str, bytes], list[tuple[list[str], bool]]
    ],
    problem: str,
) -> None:
    """Exercise the actual archive gate, including nested paths and an internal wheel leak."""
    root, wheel_sources, archive_sources, invocations = notebook_release_fixture
    if problem == "missing_nested":
        del archive_sources["notebooks/waves/maxwell/cavity.ipynb"]
    elif problem == "tampered_nested":
        archive_sources["notebooks/waves/maxwell/cavity.ipynb"] = b"changed"
    elif problem == "missing_catalogue":
        del archive_sources["notebooks/catalogue.json"]
    elif problem == "tampered_readme":
        archive_sources["notebooks/README.md"] = b"changed"
    elif problem == "wheel_notebook":
        wheel_sources["pymhm/examples/demo.ipynb"] = b"notebook"
    _write_release_archives(root, wheel_sources, archive_sources)
    if problem == "valid":
        _CHECKS["main"]()
        assert len(invocations) == 1
        assert invocations[0][0][-2:] == [
            str(root / "dist/pymhm-0.1.0-py3-none-any.whl"),
            str(root / "dist/pymhm-0.1.0.tar.gz"),
        ]
        assert invocations[0][1] is True
    else:
        with pytest.raises(
            SystemExit, match="source names differ|source bytes differ|notebook documents"
        ):
            _CHECKS["main"]()
        assert not invocations


def test_recursive_runtime_sources_include_typing_stubs_and_marker(tmp_path):
    package = tmp_path / "src/pymhm"
    core = package / "core"
    core.mkdir(parents=True)
    for name, payload in (
        ("__init__.py", b"module"),
        ("__init__.pyi", b"stub"),
        ("py.typed", b""),
        ("core/assembly.py", b"assembly"),
        ("core/assembly.pyi", b"assembly stub"),
        ("core/ignored.json", b"{}"),
        ("core/cached.pyc", b"cache"),
    ):
        (package / name).write_bytes(payload)
    sources = package_sources(tmp_path)
    assert sources == {
        "pymhm/__init__.py": b"module",
        "pymhm/__init__.pyi": b"stub",
        "pymhm/py.typed": b"",
        "pymhm/core/assembly.py": b"assembly",
        "pymhm/core/assembly.pyi": b"assembly stub",
    }
    validate_package_sources(dict(reversed(list(sources.items()))), sources, "Wheel")


@pytest.mark.parametrize("problem", ["empty", "missing", "extra", "changed"])
def test_runtime_distribution_requires_exact_source_names_and_bytes(problem):
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
    with pytest.raises(
        SystemExit, match="No runtime sources|source names differ|source bytes differ"
    ):
        validate_package_sources(archived, sources, "Wheel")


def test_compact_layers_are_real_npz_payloads():
    """The three small material inputs remain usable when checkout skips Git LFS."""
    root = Path(__file__).resolve().parents[1]
    for name in _CHECKS["COMPACT_RESERVOIR_LAYERS"]:
        validate_compact_layer((root / name).read_bytes(), name)


@pytest.mark.parametrize(
    "payload",
    [
        b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n",
        b"PK\x03\x04truncated",
    ],
)
def test_compact_layer_rejects_pointer_or_invalid_zip(payload):
    """A present filename cannot substitute a pointer or truncated archive for its arrays."""
    with pytest.raises(SystemExit, match="NPZ"):
        validate_compact_layer(payload, "layer-1.npz")


@pytest.mark.parametrize("missing", [True, False])
def test_compact_layer_requires_numpy_members(missing):
    """ZIP containers must contain each material input as an actual NPY member."""
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name in ("permeability.npy", "porosity.npy", "spacing.npy"):
            if not (missing and name == "spacing.npy"):
                archive.writestr(name, b"not an array")
    with pytest.raises(SystemExit, match="NPZ"):
        validate_compact_layer(payload.getvalue(), "layer-1.npz")


@pytest.mark.parametrize(
    "path",
    [
        "/home/researcher/work/compare.py",
        "/Users/researcher/work/compare.py",
        r"C:\Users\researcher\work\compare.py",
        ".tmp/validation/compare.py",
        "file:///home/researcher/work/compare.py",
    ],
)
def test_private_paths_are_rejected_in_keys_and_nested_values(path):
    """Inspect both source-hash keys and recursively nested provenance values."""
    for record in ({"source_hashes": {path: "abc123"}}, {"runs": [{"source": path}]}):
        with pytest.raises(SystemExit, match="private filesystem path"):
            validate_json_provenance(record, "results.json")


def test_relative_sources_and_public_urls_preserve_actual_digests():
    """Keep portable source identities, ordinary metadata and public URL paths."""
    record = {
        "source_hashes": {
            "independent-unfitted-ufl-comparison-driver": "abc123",
            "src/pymhm/models/darcy/primal.py": "def456",
            "src/pymhm/_legacy/models/darcy/primal.py": "fed654",
        },
        "source_url": "https://example.org/home/project",
        "runs": [None, 1, False, {"residual": 1e-14}],
    }
    validate_json_provenance(record, "results.json")
    assert record["source_hashes"]["independent-unfitted-ufl-comparison-driver"] == "abc123"
