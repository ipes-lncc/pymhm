"""Load suite contracts and prepare isolated, checksum-verified notebook companions."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import pytest

from scripts.build_notebook_companions import (
    build_notebook_companions,
    validate_notebook_companion_pins,
)

pytest_plugins = ["scripts.pytest_suite"]


@pytest.fixture(scope="session")
def notebook_companion_archives(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Build deterministic support ZIPs from tracked sources, without numerical inputs."""
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("notebook-companions")
    return output, build_notebook_companions(root, output)


@pytest.fixture
def prepare_notebook_companion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    notebook_companion_archives: tuple[Path, dict[str, dict[str, Any]]],
) -> Callable[[str], Path]:
    """Run unchanged notebook acquisition with exact local ZIPs and no external downloads."""
    output, descriptors = notebook_companion_archives
    root = Path(__file__).resolve().parents[1]
    cache, workspace = tmp_path / "cache", tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("PYMHM_CACHE_DIR", str(cache))
    monkeypatch.setenv("PYMHM_WORKSPACE", str(workspace))
    for name in ("PYMHM_NOTEBOOK_PREPARED", "PYMHM_NOTEBOOK_HISTORICAL", "PYMHM_NOTEBOOK_STUDY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.syspath_prepend(str(workspace))

    def forbid_download(*args: Any, **kwargs: Any) -> NoReturn:
        """Reject any undeclared dependency on the public website or a developer cache."""
        raise AssertionError(f"Notebook tests must use locally built resources: {args}")

    monkeypatch.setattr("pymhm.io.resources.urlopen", forbid_download)

    def prepare(selector: str) -> Path:
        """Seed the selected ZIP and its locally available, verified numerical inputs."""
        descriptor = descriptors[selector]
        validate_notebook_companion_pins(root, {selector: descriptor}, require_pins=True)
        digest = descriptor["sha256"]
        archive = output / "downloads" / digest / (Path(selector).stem + "-companion.zip")
        destination = cache / "archives" / digest / "resources.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, destination)
        with zipfile.ZipFile(archive) as zipped:
            resources = json.loads(zipped.read(".pymhm-resources.json"))["resources"]
        for name, identity in resources.items():
            source = root / name
            if "url" not in identity or not source.is_file():
                continue
            assert hashlib.sha256(source.read_bytes()).hexdigest() == identity["sha256"], name
            target = cache / "resources" / identity["sha256"] / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        return workspace

    return prepare
