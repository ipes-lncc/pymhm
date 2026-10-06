"""Qualify native UFL assembly with PETSc imports forbidden in fresh CPU workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.fem, pytest.mark.serial]


@pytest.mark.parametrize("solver", ["scipy", "pypardiso"])
def test_native_ufl_and_existing_cpu_solvers_without_petsc(solver: str, tmp_path: Path) -> None:
    """Exercise real 2D/3D assembly, physical data and signed shared-face spawn solves."""
    pytest.importorskip("dolfinx")
    if solver == "pypardiso":
        pytest.importorskip("pypardiso")
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1")
    )
    report = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tests/fenics_portability_probe.py"),
            "--solver",
            solver,
            "--report",
            str(report),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert json.loads(report.read_text(encoding="utf8")) == receipt
    assert receipt["solver"] == solver
    assert len(receipt["source_revision"]) == 40
    assert len(receipt["executed_source_sha256"]) == 64
    assert receipt["tracked_source_files"] > 0
    assert "scipy" in receipt["solver_package_versions"]
    if solver == "pypardiso":
        assert "pypardiso" in receipt["solver_package_versions"]
    assert receipt["petsc_python_modules"] == []
    assert receipt["local_fields"] == [
        "scalar2",
        "vector2",
        "mixed2",
        "scalar3",
        "vector3",
        "mixed3",
    ]
    assert len(receipt["shared_face_cases"]) == 9
    assert {case["execution"] for case in receipt["shared_face_cases"]} == {
        "serial",
        "thread",
        "process",
    }
