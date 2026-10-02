"""Lightweight acquisition-manifest checks for the periodic scientific example."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal


def _example(name):
    """Load example dependencies explicitly under the portable test entry point."""
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_example("periodic_norms")
_example("verify_periodic")
compare_periodic = _example("compare_periodic")


def _archive(tmp_path, monkeypatch):
    """Create distinct legacy/current fields so an implicit filename cannot pass."""
    monkeypatch.setattr(compare_periodic, "ROOT", tmp_path)
    monkeypatch.setattr(compare_periodic, "ARTIFACTS", tmp_path)
    folder = tmp_path / "examples/results"
    folder.mkdir(parents=True)
    path = tmp_path / "reference-q2-2-threads8.npz"
    np.savez(path, pressure=np.arange(25.0), residual=1e-12)
    np.savez(tmp_path / "reference-q2-2-order3-lor.npz", pressure=np.zeros(25), residual=0.0)
    record = {
        "degree": 2,
        "n": 2,
        "quadrature_order": 3,
        "solver": "low-order-refined-pyamg-cg",
        "archive": path.name,
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "relative_equation_residual": 1e-12,
    }
    metadata = folder / "periodic-reference-q2-2-order3.json"
    metadata.write_text(json.dumps(record))
    return path, metadata, record


def test_manifest_selects_and_verifies_the_executed_archive(tmp_path, monkeypatch):
    """The declared hash/filename selects current coefficients, with legacy syntax preserved."""
    path, metadata, _ = _archive(tmp_path, monkeypatch)
    field, actual = compare_periodic.load_reference("2:2:3:lor")
    assert actual == path
    assert_array_equal(field.pressure, np.arange(25.0))
    metadata.unlink()
    field, actual = compare_periodic.load_reference("2:2:3:lor")
    assert actual.name == "reference-q2-2-order3-lor.npz"
    assert_array_equal(field.pressure, np.zeros(25))


@pytest.mark.parametrize("mutation", ["hash", "degree", "solver", "parent", "residual"])
def test_manifest_mismatch_is_rejected(tmp_path, monkeypatch, mutation):
    """Never silently load stale, differently discretized or relocated acquisition data."""
    _, metadata, record = _archive(tmp_path, monkeypatch)
    key, value = {
        "hash": ("archive_sha256", "0" * 64),
        "degree": ("degree", 3),
        "solver": ("solver", "other"),
        "parent": ("archive", "../outside.npz"),
        "residual": ("relative_equation_residual", 0.0),
    }[mutation]
    record[key] = value
    metadata.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest|digest|arrays"):
        compare_periodic.load_reference("2:2:3:lor")


@pytest.mark.parametrize(
    "pressure,residual",
    [(np.zeros(24), 0), (np.full(25, np.nan), 0), (np.zeros(25), np.inf), (np.zeros(25), -1)],
)
def test_invalid_legacy_arrays_are_rejected(tmp_path, monkeypatch, pressure, residual):
    """Dimension and finiteness contracts apply even to archives without a manifest."""
    monkeypatch.setattr(compare_periodic, "ARTIFACTS", tmp_path)
    np.savez(tmp_path / "reference-q2-2.npz", pressure=pressure, residual=residual)
    with pytest.raises(ValueError, match="arrays"):
        compare_periodic.load_reference("2:2")


@pytest.mark.parametrize("name", ["verify_periodic", "periodic_reference", "compare_periodic"])
@pytest.mark.parametrize("module", [False, True])
def test_periodic_cli_supports_file_and_module_entrypoints(name, module):
    """A fresh interpreter resolves every example dependency in either supported form."""
    root = Path(__file__).resolve().parents[1]
    command = ["-m", f"examples.{name}"] if module else [str(root / "examples" / f"{name}.py")]
    result = subprocess.run(
        [sys.executable, *command, "--help"], cwd=root, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
