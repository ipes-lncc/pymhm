"""Notebook downloads keep verifiable companions separate from the installed library."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.build_notebook_companions import (
    SMALL_CONFIGS,
    build_companion,
    build_notebook_companions,
    module_sources,
    validate_notebook_companion_pins,
)
from scripts.docs_downloads import DOWNLOAD_BASE, export_downloads


def _companion_checkout(root: Path) -> None:
    """Create one copied notebook with transitive helpers and independently acquired inputs."""
    payloads = {
        "examples/__init__.py": "",
        "examples/helper.py": "from .forms import value\n",
        "examples/forms.py": "value = 3\nREQUIRED = 'data/required.json'\n",
        "examples/producer.py": "from examples.helper import value\n",
        "scripts/__init__.py": "",
        "scripts/run_notebooks.py": "from scripts.notebook_data import value\n",
        "scripts/notebook_data.py": "value = 2\n",
        "scripts/notebook_reproduction.py": "",
        "scripts/notebook_images.json": "{}",
        "LICENSE": "own source license",
        **dict.fromkeys(SMALL_CONFIGS, "{}"),
    }
    for name, payload in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    notebook = root / "notebooks/darcy/example.ipynb"
    notebook.parent.mkdir(parents=True)
    notebook.write_text(
        json.dumps(
            {"cells": [{"cell_type": "code", "source": "from examples.helper import value"}]}
        )
    )
    manifest = {
        "schema_version": 2,
        "bundled": {"notebooks/darcy/example.ipynb": "not part of the companion"},
        "remote": {
            "data/field.npz": {
                "sha256": "a" * 64,
                "url": "https://source/field.npz",
                "related_resources": ["data/NOTICE"],
            },
            "data/NOTICE": {"sha256": "b" * 64, "url": "https://source/NOTICE"},
            "data/required.json": {"sha256": "d" * 64, "url": "https://source/required.json"},
            "data/unused.npz": {"sha256": "c" * 64, "url": "https://source/unused.npz"},
        },
        "unavailable": {"data/original.npz": {"reason": "Original archive unavailable"}},
        "notebook_resources": {"darcy/example.ipynb": ["data/field.npz"]},
        "notebook_historical_resources": {"darcy/example.ipynb": ["data/original.npz"]},
        "notebook_study_resources": {"darcy/example.ipynb": []},
    }
    (root / "examples/resource_manifest.json").write_text(json.dumps(manifest))
    (root / "scripts/notebook_reproduction.json").write_text(
        json.dumps(
            {
                "notebooks": {
                    "notebooks/darcy/example.ipynb": {
                        "preparation": [
                            {
                                "environment": "notebooks",
                                "argv": ["python", "-m", "examples.producer"],
                            }
                        ]
                    }
                }
            }
        )
    )


def test_companion_closure_and_identities_reproduce_without_bundling_numerical_inputs(
    tmp_path: Path,
) -> None:
    """Archives preserve exact helpers/notices and exclude unrelated or circular inputs."""
    root, output = tmp_path / "checkout", tmp_path / "downloads"
    _companion_checkout(root)
    assert module_sources(root, ["examples.helper"]) == [
        "examples/__init__.py",
        "examples/forms.py",
        "examples/helper.py",
    ]
    initial = build_notebook_companions(root, output)["darcy/example.ipynb"]
    archive = output / "downloads" / initial["sha256"] / "example-companion.zip"
    with zipfile.ZipFile(archive) as source:
        payloads = {name: source.read(name) for name in source.namelist()}
    assert "examples/producer.py" in payloads and "examples/forms.py" in payloads
    assert not any(name.startswith(("notebooks/", "src/", "data/")) for name in payloads)
    assert "examples/resource_manifest.json" not in payloads
    registry = json.loads(payloads[".pymhm-resources.json"])
    assert "data/unused.npz" not in registry["resources"]
    assert registry["resources"]["data/required.json"]["sha256"] == "d" * 64
    assert registry["resources"]["data/NOTICE"]["sha256"] == "b" * 64
    assert registry["unavailable"]["data/original.npz"]["reason"]
    for name, digest in initial["source_files"].items():
        assert hashlib.sha256(payloads[name]).hexdigest() == digest
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == initial["sha256"]
    os.utime(root / "examples/helper.py", (12345, 12345))
    (root / "notebooks/darcy/example.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": "PIN = 'archive SHA'\nfrom examples.helper import value",
                    }
                ]
            }
        )
    )
    assert build_notebook_companions(root, output)["darcy/example.ipynb"] == initial
    (root / "examples/forms.py").write_text("value = 4\nREQUIRED = 'data/required.json'\n")
    changed = build_notebook_companions(root, output)["darcy/example.ipynb"]
    assert changed["sha256"] != initial["sha256"]
    assert (
        changed["source_files"]["examples/forms.py"] != initial["source_files"]["examples/forms.py"]
    )

    index = root / "examples/resource_manifest.json"
    inventory = json.loads(index.read_text())
    for name, record in inventory["remote"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record.update(sha256=digest, size_bytes=path.stat().st_size)
        record["url"] = DOWNLOAD_BASE + digest + "/" + path.name
    index.write_text(json.dumps(inventory))
    descriptor = build_notebook_companions(root, output)["darcy/example.ipynb"]
    notebook = root / "notebooks/darcy/example.ipynb"
    source = (
        f"COMPANION_URL = (\n    {descriptor['url']!r}\n)\n"
        f"COMPANION_SHA256 = {descriptor['sha256']!r}\nfrom examples.helper import value"
    )
    notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": source}]}))
    validate_notebook_companion_pins(root, {"darcy/example.ipynb": descriptor}, require_pins=True)
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": source.replace(
                            descriptor["url"], "https://incorrect.example/companion.zip"
                        ),
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="pins differ"):
        validate_notebook_companion_pins(root, {"darcy/example.ipynb": descriptor})
    notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": source}]}))
    inventory["bundled"]["notebooks/darcy/example.ipynb"] = hashlib.sha256(
        notebook.read_bytes()
    ).hexdigest()
    index.write_text(json.dumps(inventory))
    site = tmp_path / "site"
    result = export_downloads(root, site)
    assert result["companions"] == 1 and result["resources"] == 5
    catalogue = json.loads((site / "downloads/catalogue.json").read_text())
    companion = catalogue["companions"]["darcy/example.ipynb"]
    archive = site / "downloads" / companion["sha256"] / "example-companion.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == companion["sha256"]
    with zipfile.ZipFile(archive) as source:
        assert all(not name.endswith(".npz") for name in source.namelist())
    hook = Path(__file__).resolve().parents[1] / "scripts/docs_downloads.py"
    isolated_site = tmp_path / "isolated-mkdocs-site"
    command = (
        "import importlib.util;from pathlib import Path;"
        f"spec=importlib.util.spec_from_file_location('mkdocs_hook',{str(hook)!r});"
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);"
        f"module.export_downloads(Path({str(root)!r}),Path({str(isolated_site)!r}))"
    )
    subprocess.run([sys.executable, "-I", "-c", command], cwd=tmp_path, check=True)
    assert json.loads((isolated_site / "downloads/catalogue.json").read_text()) == catalogue


@pytest.mark.parametrize(
    "source",
    [
        "../private.py",
        "notebooks/case.ipynb",
        "examples/results/field.npz",
        "examples/resource_manifest.json",
    ],
)
def test_companion_rejects_unsafe_or_case_payloads(tmp_path: Path, source: str) -> None:
    """Notebook pins cannot legitimize traversals, fields or the circular global case registry."""
    with pytest.raises(ValueError, match="Invalid|Unsupported"):
        build_companion(tmp_path, [source], tmp_path / "output", name="companion.zip")
