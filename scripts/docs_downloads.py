"""Publish checksum-addressed case downloads separately from Python distributions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

# MkDocs loads hooks by filename with no repository package on sys.path.
_spec = importlib.util.spec_from_file_location(
    "pymhm_notebook_companion_builder", Path(__file__).with_name("build_notebook_companions.py")
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("The documentation companion builder is unavailable")
_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_builder)
build_notebook_companions = _builder.build_notebook_companions
validate_notebook_companion_pins = _builder.validate_notebook_companion_pins

DOWNLOAD_BASE = "https://ipes-lncc.github.io/pymhm/downloads/"


def export_downloads(root: Path, site: Path) -> dict[str, Any]:
    """Export only declared available resources with their literal SHA-256 bytes.

    The website keeps datasets, numerical records and generated figures outside
    the wheel/sdist. Content-addressed URLs do not change an old resource's
    identity when a newer record is published. Unavailable historical payloads
    retain source links in the catalogue and are never exported or substituted.
    Notebook sources are downloadable individually; their checksum-addressed
    companions contain the declared helper closure and selected input catalogue.
    No companion or downloadable source is included in the Python distribution.
    """
    manifest = json.loads((root / "examples/resource_manifest.json").read_text())
    if manifest["schema_version"] != 2:
        raise ValueError("Download exports require resource manifest schema2")
    companions = build_notebook_companions(root, site)
    validate_notebook_companion_pins(root, companions, require_pins=True)
    downloads = dict(manifest["remote"])
    for name, digest in manifest["bundled"].items():
        downloads[name] = {
            "sha256": digest,
            "size_bytes": (root / name).stat().st_size,
            "url": DOWNLOAD_BASE + digest + "/" + PurePosixPath(name).name,
            "kind": "notebook-source" if name.endswith(".ipynb") else "case-configuration",
            "provenance": "PyMHM source/configuration; numerical scope is stated in its notebook.",
        }
    destinations = set()
    for name, record in sorted(downloads.items()):
        label = PurePosixPath(name)
        if label.is_absolute() or ".." in label.parts or "\\" in name:
            raise ValueError(f"Invalid download source path: {name}")
        source = root / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Download source must be a regular file: {name}")
        expected = DOWNLOAD_BASE + record["sha256"] + "/" + label.name
        if record["url"] != expected:
            raise ValueError(f"Unexpected download URL: {name}")
        payload = source.read_bytes()
        if (
            len(payload) != record["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Download source differs from its identity: {name}")
        destination = site / "downloads" / record["sha256"] / label.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destinations.add(destination)
    for record in companions.values():
        destinations.add(site / "downloads" / record["sha256"] / record["url"].rsplit("/", 1)[1])
    catalogue = {
        "schema_version": 2,
        "downloads": dict(sorted(downloads.items())),
        "companions": companions,
        "unavailable": manifest["unavailable"],
        "primary_sources": {
            "spe10": "https://github.com/OPM/opm-data/tree/eaa2261683a97027e057c2bc49612ad1c86390b3/spe10model2",
            "marmousi": "https://ahay.org/data/marm2/",
            "hpc4e": "https://github.com/labmec/MHM/tree/f978f29d657d28fe58bcea20fabee68953093482/Data_13_Set",
        },
    }
    (site / "downloads").mkdir(parents=True, exist_ok=True)
    (site / "downloads/catalogue.json").write_text(json.dumps(catalogue, indent=2) + "\n")
    return {
        "resources": len(downloads),
        "companions": len(companions),
        "unique_files": len(destinations),
        "size_bytes": sum(path.stat().st_size for path in destinations),
        "unavailable": len(manifest["unavailable"]),
    }


def on_post_build(config: Mapping[str, Any]) -> None:
    """Export declared downloads after MkDocs builds its pages, without modifying sources."""
    root = Path(__file__).resolve().parents[1]
    result = export_downloads(root, Path(config["site_dir"]))
    print(f"Published download catalogue: {result}")
