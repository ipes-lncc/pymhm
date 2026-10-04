"""Acquire current anisotropic mixed-elasticity cases with executed field contracts.

These analytical unit-square checks retain the published mixed spaces, with
P1 interior traction and full fine normal spaces on exterior faces. They do
not reproduce an unidentified historical mesh or establish uniform inf-sup.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
from threadpoolctl import threadpool_limits

from examples import core_extension_data as exact
from examples.campaign_provenance import file_digest, positive_integers
from examples.core_elasticity_field_archive import (
    field_arrays,
    observe_system,
    read_field,
    replay,
    write_field,
)
from examples.solve_core_extensions import polygon_grid
from examples.transport_checkpoints import write_progress
from pymhm.elasticity_mixed import solve_elasticity_mixed
from pymhm.elasticity_tensor_rt import solve_elasticity_tensor_rt
from pymhm.mesh import TriangleMesh
from pymhm.polygon import solve_elasticity_mixed_polygons
from pymhm.quadrilateral import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[1]
CASES = ("triangle-bdm2", "rectangle-rt1", "polygon-bdm2")
LEVELS = (1, 2, 4, 8, 16)


def physical_errors(arrays: Mapping[str, np.ndarray], order: int) -> dict[str, float]:
    """Integrate displacement/full stress/div(stress)/rotation in their own saved bases.

    The independently differentiated manufactured source satisfies
    div(stress)=-force. Stress includes both symmetric and skew coordinates.
    Quadrature and physical geometry are literal acquisition data.
    """
    totals = np.zeros(4, dtype=np.longdouble)
    for cell in range(int(arrays["local_count"])):
        fields = replay(arrays, cell, order)
        points = arrays[f"q{order}_physical_points_{cell}"]
        flat = points.reshape(-1, 2)
        targets = (
            exact.displacement(flat).reshape(fields[0].shape),
            exact.stress(flat).reshape(fields[1].shape),
            -exact.force(flat).reshape(fields[2].shape),
            exact.rotation(flat).reshape(fields[3].shape),
        )
        measure = arrays[f"areas_{cell}"][:, None] * arrays[f"q{order}_weights"]
        for index, (value, target) in enumerate(zip(fields, targets, strict=True)):
            error = (value - target).reshape(*measure.shape, -1)
            totals[index] += np.sum(measure * np.sum(error**2, axis=-1), dtype=np.longdouble)
    return dict(
        zip(
            ("displacement_l2", "stress_l2", "divergence_l2", "rotation_l2"),
            np.sqrt(totals).astype(float).tolist(),
            strict=True,
        )
    )


def capture_sources(output: Path) -> dict[str, str]:
    """Archive exact source/lock bytes before acquiring any current field coefficients."""
    paths = [
        *sorted((ROOT / "src/pymhm").rglob("*.py")),
        Path(__file__),
        ROOT / "examples/core_elasticity_field_archive.py",
        ROOT / "examples/core_extension_data.py",
        ROOT / "examples/solve_core_extensions.py",
        ROOT / "examples/archive_precision.py",
        ROOT / "examples/campaign_provenance.py",
        ROOT / "examples/local_response_cache.py",
        ROOT / "examples/transport_checkpoints.py",
        ROOT / "pixi.lock",
        ROOT / "pixi.toml",
        ROOT / "pyproject.toml",
    ]
    hashes = {str(path.relative_to(ROOT)): file_digest(path) for path in paths}
    for name, expected in hashes.items():
        target = output / "executed-sources/files" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if file_digest(target) != expected:
            raise ValueError("Captured executed source bytes differ")
    write_progress(
        output / "executed-sources/manifest.json",
        {
            "source_sha256": hashes,
            "scope": "Exact current producer/core/lock bytes; no retrospective field provenance",
        },
    )
    return hashes


def solve_case(name: str, n: int, *, homogeneous: bool = False) -> Any:
    """Solve one declared finite space with unchanged shared operators and boundary signs.

    The default is the anisotropic polynomial manufactured problem. The zero
    force/displacement variant is an independent homogeneous contract control.
    """
    if name not in CASES or type(n) is not int or n not in LEVELS:
        raise ValueError("A selected elasticity family at dyadic levels1..16 is required")
    mesh: Any
    solve: Any
    if name == "triangle-bdm2":
        mesh, solve = TriangleMesh.unit_square(n), solve_elasticity_mixed
    elif name == "rectangle-rt1":
        mesh, solve = CartesianMacroMesh(n), solve_elasticity_tensor_rt
    else:
        mesh, solve = polygon_grid(n), solve_elasticity_mixed_polygons
    return solve(
        mesh,
        compliance=exact.compliance(),
        source=(0.0, 0.0) if homogeneous else exact.force,
        dirichlet=(0.0, 0.0) if homogeneous else exact.displacement,
        local_refinement=1,
        quadrature_order=8,
    )


def acquire_case(name: str, n: int, output: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """Acquire one fresh original operator/basis archive and own-field physical norms."""
    started = perf_counter()
    with observe_system() as observed:
        solution = solve_case(name, n)
        arrays = field_arrays(solution, observed)
    path = output / f"{name}-n{n}.npz"
    saved = write_field(
        path,
        arrays,
        acquisition_uuid=str(uuid4()),
        source_sha256=sources,
        configuration={
            "case": name,
            "resolution": n,
            "local_refinement": 1,
            "source": "independently differentiated -div(C eps(u))",
            "displacement": "(x^2*y^2,x^3*y)",
            "boundary": "full nonhomogeneous weak Dirichlet displacement",
            "compliance_coordinates": ["xx", "xy", "yx", "yy"],
            "multiplier": "negative Cauchy traction",
            "assembly_order": 8,
        },
    )
    restored, metadata = read_field(path)
    norms = {f"quadrature_{q}": physical_errors(restored, q) for q in (9, 10)}
    lower, higher = norms.values()
    changes = {
        key: abs(lower[key] - higher[key]) / max(higher[key], np.finfo(float).tiny)
        for key in higher
    }
    if max(changes.values()) > 1e-9:
        raise ArithmeticError("Physical norm quadrature exceeds unchanged relative criterion1e-9")
    row = {
        "case": name,
        "resolution": n,
        "macro_cells": len(solution.local_meshes),
        "fine_cells": sum(len(mesh.cells) for mesh in solution.local_meshes),
        "stress_space": "RT1" if name == "rectangle-rt1" else "BDM2",
        "displacement_space": "Q1" if name == "rectangle-rt1" else "cardinal P1",
        "rotation_space": "total P1",
        "interior_traction_space": "P1",
        "exterior_traction_space": "full fine P1" if name == "rectangle-rt1" else "full fine P2",
        "local_refinement": 1,
        "assembly_order": 8,
        "norm_orders": [9, 10],
        "norms": norms,
        "norm_quadrature_relative_changes": changes,
        **higher,
        "original_physical_checks": metadata["original_checks"],
        "basis_checks": metadata["basis_checks"],
        "archive": path.name,
        "sha256": saved["archive_sha256"],
        "acquisition_uuid": saved["acquisition_uuid"],
        "finite_case_accepted": True,
        "native_whole_field_agreement_verified": False,
        "elapsed_seconds": perf_counter() - started,
    }
    write_progress(output / f"{name}-n{n}-verification.json", row)
    print(json.dumps(row), flush=True)
    return row


def run(levels: Sequence[int], names: Sequence[str], output: Path) -> dict[str, Any]:
    """Acquire selected cases into fresh outputs with source hashes checked at both ends."""
    levels = positive_integers(levels, label="macro resolutions", increasing=True)
    if (
        output.exists()
        or not names
        or len(set(names)) != len(names)
        or any(name not in CASES for name in names)
        or any(n not in LEVELS for n in levels)
    ):
        raise ValueError("Fresh output, selected dyadic levels and distinct families required")
    output.mkdir(parents=True)
    sources = capture_sources(output)
    report: dict[str, Any] = {
        "suite": "elasticity",
        "reference": "analytical",
        "configuration": {"macro_resolutions": list(levels), "families": list(names)},
        "source_sha256": sources,
        "rows": [],
        "fields": {},
        "literal_literature_reproduction": False,
        "uniform_inf_sup_verified": False,
        "native_whole_field_agreement_verified": False,
    }
    for name in names:
        for n in levels:
            row = acquire_case(name, n, output, sources)
            report["rows"].append(row)
            report["fields"][f"{name}-n{n}"] = {"archive": row["archive"], "sha256": row["sha256"]}
            write_progress(output / "elasticity.json", report)
    if any(file_digest(ROOT / name) != expected for name, expected in sources.items()):
        raise ValueError("An executed source changed during acquisition")
    report["complete"], report["source_changed"] = True, False
    write_progress(output / "elasticity.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    parser.add_argument("--names", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/results/core-elasticity-current"
    )
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.levels, args.names, args.output)
