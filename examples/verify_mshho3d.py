"""Acquire finite P2/P0 MsHHO3D analytical controls with complete executed bases.

The source is projected into the actual constant volume-moment space. These
tetrahedral and cubic finite Galerkin cases measure analytical errors; they
are distinguished from exact-local literature equivalence and its estimates.
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
from examples.mshho3d_field_archive import original_checks, read_field, replay, write_field
from examples.mshho3d_ideal_p0 import ideal_p0_audit
from examples.transport_checkpoints import write_progress
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.methods.hho_3d import solve_mshho_3d

ROOT = Path(__file__).resolve().parents[1]


def physical_errors(arrays: Mapping[str, np.ndarray], order: int) -> dict[str, float]:
    """Integrate pressure and physical raw Darcy flux against independent sine fields."""
    pressure_error = flux_error = np.longdouble(0)
    for cell in range(int(arrays["local_count"])):
        pressure, _, flux = replay(arrays, cell, order)
        points = arrays[f"q{order}_physical_points_{cell}"]
        measure = arrays[f"volumes_{cell}"][:, None] * arrays[f"q{order}_weights"]
        target_pressure = exact.pressure3d(points.reshape(-1, 3)).reshape(pressure.shape)
        target_flux = exact.flux3d(points.reshape(-1, 3)).reshape(flux.shape)
        pressure_error += np.sum(measure * (pressure - target_pressure) ** 2, dtype=np.longdouble)
        flux_error += np.sum(
            measure * np.sum((flux - target_flux) ** 2, axis=-1), dtype=np.longdouble
        )
    return {
        "pressure_l2": float(np.sqrt(pressure_error)),
        "flux_l2": float(np.sqrt(flux_error)),
    }


def capture_sources(output: Path) -> dict[str, str]:
    """Archive exact executed sources and the lockfile before a fresh acquisition."""
    paths = [
        *sorted((ROOT / "src/pymhm").rglob("*.py")),
        Path(__file__),
        ROOT / "examples/mshho3d_field_archive.py",
        ROOT / "examples/archive_precision.py",
        ROOT / "examples/campaign_provenance.py",
        ROOT / "examples/local_response_cache.py",
        ROOT / "examples/transport_checkpoints.py",
        ROOT / "examples/core_extension_data.py",
        ROOT / "examples/mshho3d_sections.py",
        ROOT / "examples/mshho3d_ideal_p0.py",
        ROOT / "examples/sample_core_sections.py",
        ROOT / "examples/solve_core_extensions.py",
        ROOT / "pixi.lock",
        ROOT / "pixi.toml",
        ROOT / "pyproject.toml",
    ]
    hashes = current_source_manifest(
        {str(path.relative_to(ROOT)): file_digest(path) for path in paths}
    )
    for name, expected in hashes.items():
        target = output / "executed-sources/files" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if file_digest(target) != expected:
            raise RuntimeError("Captured numerical source differs from its executed bytes")
    write_progress(
        output / "executed-sources/manifest.json",
        {
            "source_sha256": hashes,
            "scope": "Exact source bytes; no retroactive provenance substitution",
        },
    )
    return hashes


def acquire_case(
    kind: str,
    n: int,
    output: Path,
    source_sha256: Mapping[str, str],
    *,
    display_refinement: int = 0,
) -> dict[str, Any]:
    """Acquire one unchanged finite analytical problem and verify its physical rows.

    Identity permeability and p=sin(pi*x)sin(pi*y)sin(pi*z) give f=3*pi^2*p
    by differentiating each of the three coordinate terms. The operator uses
    its P0 projection on each original macro, with zero Dirichlet pressure.
    """
    if kind not in {"tetra", "cube"} or type(n) is not int or n not in {1, 2, 3, 4, 5}:
        raise ValueError("Tetrahedral/cubic macros at levels1..5 required")
    started = perf_counter()
    tetra = kind == "tetra"
    mesh = TetraMesh.unit_cube(n) if tetra else PolyhedralMesh.cubes(n)
    solution = solve_mshho_3d(
        mesh,
        source=exact.source3d,
        quadrature_order=9,
        local_refinement=2 if tetra else 1,
        local_refinement_precision="extended",
    )
    path = output / f"{kind}-p0-n{n}.npz"
    archive_record = write_field(
        path,
        solution,
        {
            "case": f"{kind}-p0",
            "resolution": n,
            "local_refinement": 2 if tetra else 1,
            "permeability": "identity",
            "source": "P0 projection of3*pi^2*sin(pi*x)*sin(pi*y)*sin(pi*z)",
            "boundary": "Homogeneous physical Dirichlet pressure on every exterior macroface",
        },
        acquisition_uuid=str(uuid4()),
        source_sha256=source_sha256,
        assembly_order=9,
        source=exact.source3d,
        display_refinement=display_refinement,
    )
    arrays, _ = read_field(path)
    norms = {f"quadrature_{q}": physical_errors(arrays, q) for q in (10, 12)}
    first, last = norms.values()
    absolute = max(abs(first[k] - last[k]) for k in first)
    relative = max(abs(first[k] - last[k]) / max(abs(last[k]), np.finfo(float).tiny) for k in first)
    if relative > 1e-9 or absolute > 1e-8 * max(1.0, *last.values()):
        raise ArithmeticError("Independent physical error quadrature failed its unchanged criteria")
    checks = original_checks(arrays)
    ideal = ideal_p0_audit(arrays, kind)
    row = {
        "case": f"{kind}-p0",
        "resolution": n,
        "macro_cells": len(mesh.cells),
        "fine_cells": sum(len(local.mesh.cells) for local in solution.local),
        "degree": 2,
        "cell_degree": 0,
        "face_degree": 0,
        "local_refinement": 2 if tetra else 1,
        "source_variant": "projected",
        "local_refinement_precision": "extended",
        "coefficient_nmant": archive_record["coefficient_nmant"],
        "algebraic_residual": float(solution.residual),
        "original_physical_checks": checks,
        "ideal_constant_moment_space_audit": ideal,
        "error_quadrature": [10, 12],
        "error_quadrature_absolute_change": absolute,
        "error_quadrature_relative_change": relative,
        "norms": norms,
        **last,
        "elapsed_seconds": perf_counter() - started,
        "archive": path.name,
        "sha256": archive_record["archive_sha256"],
        "acquisition_uuid": archive_record["acquisition_uuid"],
        "finite_case_accepted": True,
        "native_whole_field_agreement_verified": False,
    }
    write_progress(output / f"{kind}-p0-n{n}-verification.json", row)
    return row


def run(levels: Sequence[int] = (1, 2, 3, 4, 5), *, output: Path) -> dict[str, Any]:
    """Acquire fresh complete field contracts, preserving any prior result generation."""
    selected = tuple(levels)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(type(n) is not int or n not in {1, 2, 3, 4, 5} for n in selected)
    ):
        raise ValueError("Distinct supported levels1..5 required")
    if output.exists():
        raise ValueError("A fresh acquisition directory is required")
    output.mkdir(parents=True)
    hashes = capture_sources(output)
    report: dict[str, Any] = {
        "schema": 2,
        "suite": "mshho3d",
        "reference": "analytical",
        "method": "Finite Galerkin P2 energy lifts with P0 volume and original-macroface moments",
        "source_sha256": hashes,
        "rows": [],
        "fields": {},
        "literature": "10.1051/m2an/2021082; ideal-local equivalence and Remark7.7 qualifications",
        "literal_literature_reproduction": False,
        "uniform_inf_sup_verified": False,
        "native_whole_field_agreement_verified": False,
        "scope": (
            "Analytical unit cube on regular Freudenthal tetrahedral "
            "and Cartesian cubic macro partitions"
        ),
        "status": "acquiring",
    }
    with threadpool_limits(1):
        for kind in ("tetra", "cube"):
            for n in selected:
                row = acquire_case(
                    kind,
                    n,
                    output,
                    hashes,
                    display_refinement=6 if n == selected[-1] else 0,
                )
                report["rows"].append(row)
                if n == selected[-1]:
                    report["fields"][row["case"]] = {
                        "archive": row["archive"],
                        "sha256": row["sha256"],
                    }
                write_progress(output / "mshho3d.json", report)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "case",
                                "resolution",
                                "pressure_l2",
                                "flux_l2",
                                "original_physical_checks",
                                "elapsed_seconds",
                            )
                        }
                    ),
                    flush=True,
                )
    if any(file_digest(ROOT / name) != expected for name, expected in hashes.items()):
        raise RuntimeError("Executed numerical sources changed during acquisition")
    report["source_changed"] = False
    report["status"] = (
        "verified-finite-analytical-cases; native-and-local-resolution-controls-pending"
    )
    write_progress(output / "mshho3d.json", report)
    return report


def main() -> None:
    """Acquire selected levels into a fresh directory, retaining every verified field."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "examples/results/core-extensions/mshho3d-current"
    )
    args = parser.parse_args()
    run(args.levels, output=args.output)


if __name__ == "__main__":
    main()
