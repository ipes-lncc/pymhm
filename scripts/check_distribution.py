"""Inspect built wheels and source archives before publication."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Any

MAX_ARTIFACT_BYTES = 100_000_000
COMPACT_RESERVOIR_LAYERS = {
    "examples/results/spe10/layer-1.npz",
    "examples/results/spe10/layer-36.npz",
    "examples/results/spe10/layer-85.npz",
}


def validate_compact_layer(payload: bytes, artifact_name: str) -> None:
    """Require real NPZ input arrays, excluding LFS pointers and corrupt ZIP members."""
    if not payload.startswith(b"PK\x03\x04"):
        raise SystemExit(f"Compact reservoir layer is not an NPZ payload: {artifact_name}")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            required = {"permeability.npy", "porosity.npy", "spacing.npy"}
            if not required.issubset(archive.namelist()) or archive.testzip() is not None:
                raise ValueError("Missing or corrupt compact layer arrays")
            for name in required:
                if not archive.read(name).startswith(b"\x93NUMPY"):
                    raise ValueError("Compact layer member is not a NumPy array")
    except (ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Invalid compact reservoir NPZ: {artifact_name}") from error


def validate_json_provenance(value: Any, artifact_name: str) -> None:
    """Reject private filesystem paths in public JSON keys or nested values.

    Relative package sources and public URLs remain legitimate provenance.
    An executed comparison driver can instead use a neutral identifier paired
    with its unchanged source digest.
    """
    if isinstance(value, dict):
        for key, entry in value.items():
            validate_json_provenance(key, artifact_name)
            validate_json_provenance(entry, artifact_name)
    elif isinstance(value, list):
        for entry in value:
            validate_json_provenance(entry, artifact_name)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        personal = re.search(r"(?<!\w)(?:file://)?(?:/home/|/Users/|[A-Za-z]:/Users/)", normalized)
        if personal or ".tmp" in Path(normalized).parts:
            raise SystemExit(f"Public JSON contains a private filesystem path: {artifact_name}")


def main() -> None:
    """Validate archive contents, version, license, and rendered metadata."""
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    wheel = root / "dist" / f"pymhm-{version}-py3-none-any.whl"
    sdist = root / "dist" / f"pymhm-{version}.tar.gz"
    if not wheel.is_file() or not sdist.is_file():
        raise SystemExit("Build the wheel and source archive before checking distributions")
    for artifact in (wheel, sdist):
        if artifact.stat().st_size > MAX_ARTIFACT_BYTES:
            raise SystemExit(
                f"{artifact.name} exceeds the {MAX_ARTIFACT_BYTES:,}-byte release limit"
            )
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        required = {"pymhm/__init__.py", "pymhm/py.typed"}
        if not required.issubset(names):
            raise SystemExit("Wheel lacks package entry point or typing marker")
        metadata_path = f"pymhm-{version}.dist-info/METADATA"
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        if metadata["Name"] != "pymhm" or metadata["Version"] != version:
            raise SystemExit("Wheel name or version differs from source metadata")
        if metadata["License-Expression"] != "LGPL-2.1-only":
            raise SystemExit("Wheel lacks the expected license expression")
        if not any(name.endswith("/licenses/LICENSE") for name in names):
            raise SystemExit("Wheel lacks its license file")
        if any(not name.startswith(("pymhm/", f"pymhm-{version}.dist-info/")) for name in names):
            raise SystemExit("Wheel contains files outside the package and metadata")
    with tarfile.open(sdist) as archive:
        prefix = f"pymhm-{version}/"
        names = {name.removeprefix(prefix) for name in archive.getnames()}
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith(".json"):
                stream = archive.extractfile(member)
                if stream is not None:
                    with stream:
                        validate_json_provenance(json.load(stream), member.name)
        required = {
            "pyproject.toml",
            "LICENSE",
            "README.md",
            "src/pymhm/py.typed",
            "tests/test_elasticity_mixed_fenics.py",
            "tests/test_darcy_bdm_fenics.py",
        }
        if not required.issubset(names):
            raise SystemExit("Source archive lacks required build or license files")
        if not COMPACT_RESERVOIR_LAYERS.issubset(names):
            raise SystemExit("Source archive lacks the compact reservoir input layers")
        for name in COMPACT_RESERVOIR_LAYERS:
            stream = archive.extractfile(prefix + name)
            if stream is None:
                raise SystemExit(f"Compact reservoir layer is not a regular file: {name}")
            with stream:
                validate_compact_layer(stream.read(), name)
        for name in names:
            if name.startswith("docs/figures/"):
                raise SystemExit("Source archive contains a rendered documentation gallery")
            if (
                name.startswith("examples/results/")
                and Path(name).suffix in {".npz", ".vtu"}
                and name not in COMPACT_RESERVOIR_LAYERS
            ):
                raise SystemExit(
                    f"Source archive contains a large numerical output archive: {name}"
                )
        comparison_runners = {
            "generate_neopz_reference.py",
            "verify_neopz_darcy.py",
            "verify_reference_darcy.py",
            "audit_neopz_quadrature.py",
            "audit_darcy.py",
            "audit_flow.py",
            "reproduce_darcy_2013.py",
            "reproduce_stokes_2017.py",
            "test_darcy_independent.py",
        }
        external_sources = {
            "msl_core",
            "msl_cg",
            "msl_mhm",
            "msl_mfem",
            "mhm-mfem",
            "labmec-neopz",
            "labmec-MHM",
            "labmec-iMRS",
        }
        for name in names:
            path = Path(name)
            if (
                name.startswith("tools/neopz/")
                or path.name in comparison_runners
                or external_sources.intersection(path.parts)
            ):
                raise SystemExit(f"Source archive contains an external comparison tool: {name}")
    subprocess.run(
        [sys.executable, "-m", "twine", "check", "--strict", str(wheel), str(sdist)],
        check=True,
    )
    print(f"Validated wheel and source archive for pymhm {version}")


if __name__ == "__main__":
    main()
