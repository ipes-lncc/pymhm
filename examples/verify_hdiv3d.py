"""Acquire affine mixed Darcy cases with literal executed bases and physical rows.

These are analytical unit-cube checks, with independent normal and pressure
orders. They do not reproduce a historical mesh or establish uniform inf-sup.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
from examples.campaign_provenance import positive_integers
from examples.formulations.application import hdiv_darcy as solve_darcy_hdiv3d
from examples.hdiv3d_field_archive import (
    field_arrays,
    observe_system,
    read_field,
    replay,
    write_field,
)
from examples.transport_checkpoints import write_progress
from examples.verify_mshho3d import capture_sources as capture_shared_sources
from pymhm.io.provenance import file_digest
from pymhm.meshes.mixed import AffineMixedMesh

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "tetra-p1-k1": ("tetrahedron", 1, 1),
    "tetra-p2-k2": ("tetrahedron", 2, 2),
    "tetra-p3-k1": ("tetrahedron", 3, 1),
    "prism-p2-k2": ("prism", 2, 2),
}


def physical_errors(arrays: Mapping[str, np.ndarray], order: int) -> dict[str, float]:
    """Integrate p, physical H(div) flux and div(q) against independently derived sine fields."""
    total = np.zeros(3, dtype=np.longdouble)
    reference, weights = arrays[f"q{order}_points"], arrays[f"q{order}_weights"]
    for cell in range(int(arrays["local_count"])):
        pressure, flux, divergence = replay(arrays, cell, order)
        points = arrays[f"points_{cell}"][arrays[f"cells_{cell}"][:, 0], None]
        points = points + np.einsum("tab,qb->tqa", arrays[f"jacobian_{cell}"], reference)
        flat = points.reshape(-1, 3)
        pressure_error = pressure - exact.pressure3d(flat).reshape(pressure.shape)
        flux_error = flux - exact.flux3d(flat).reshape(flux.shape)
        divergence_error = divergence - exact.source3d(flat).reshape(divergence.shape)
        measure = arrays[f"determinants_{cell}"][:, None] * weights
        total += [
            np.sum(measure * pressure_error**2, dtype=np.longdouble),
            np.sum(measure * np.sum(flux_error**2, axis=-1), dtype=np.longdouble),
            np.sum(measure * divergence_error**2, dtype=np.longdouble),
        ]
    return dict(
        zip(
            ("pressure_l2", "flux_l2", "divergence_l2"),
            np.sqrt(total).astype(float).tolist(),
            strict=True,
        )
    )


def capture_sources(output: Path) -> dict[str, str]:
    """Capture exact shared-core bytes and this producer before any field acquisition."""
    hashes = capture_shared_sources(output)
    for path in (Path(__file__), ROOT / "examples/hdiv3d_field_archive.py"):
        name = path.relative_to(ROOT).as_posix()
        hashes[name] = file_digest(path)
        target = output / "executed-sources/files" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_digest(target) != hashes[name]:
            raise ValueError("Executed producer source capture differs")
    write_progress(
        output / "executed-sources/manifest.json",
        {
            "source_sha256": hashes,
            "scope": "Literal shared core and H(div) producer bytes before fresh acquisition",
        },
    )
    return hashes


def acquire_case(name: str, n: int, output: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """Acquire unchanged K=I, zero-pressure-boundary, sine-source data for one selected pair."""
    if name not in CASES or type(n) is not int or n not in range(1, 6):
        raise ValueError("A selected family at levels1..5 is required")
    started = perf_counter()
    kind, pressure_degree, normal_degree = CASES[name]
    with observe_system() as observed:
        solution = solve_darcy_hdiv3d(
            AffineMixedMesh.unit_cube(n, kind),
            pressure_degree=pressure_degree,
            normal_degree=normal_degree,
            trace_degree=normal_degree,
            local_refinement=1,
            source=exact.source3d,
            quadrature_order=12,
        )
        arrays = field_arrays(solution, observed, assembly_order=12)
    path = output / f"{name}-n{n}.npz"
    saved = write_field(
        path,
        arrays,
        acquisition_uuid=str(uuid4()),
        source_sha256=sources,
        configuration={
            "case": name,
            "resolution": n,
            "permeability": "identity",
            "source": "3*pi^2*sin(pi*x)*sin(pi*y)*sin(pi*z)",
            "boundary": "Homogeneous physical pressure weakly imposed on the complete exterior",
            "local_refinement": 1,
        },
    )
    restored, metadata = read_field(path)
    norms = {f"quadrature_{q}": physical_errors(restored, q) for q in (12, 13)}
    first, last = norms.values()
    difference = max(abs(first[key] - last[key]) for key in ("pressure_l2", "flux_l2"))
    if difference > 1e-8 * max(1.0, last["pressure_l2"], last["flux_l2"]):
        raise ArithmeticError("Physical error quadrature exceeds its unchanged criterion")
    divergence_change = abs(first["divergence_l2"] - last["divergence_l2"])
    if divergence_change > 1e-8 * max(1.0, last["divergence_l2"]):
        raise ArithmeticError("Physical divergence error quadrature is insufficient")
    row = {
        "case": name,
        "resolution": n,
        "macro_cells": len(solution.local_meshes),
        "fine_cells": sum(len(mesh.cells) for mesh in solution.local_meshes),
        "pressure_degree": pressure_degree,
        "normal_degree": normal_degree,
        "trace_degree": normal_degree,
        "local_refinement": 1,
        "assembly_order": 12,
        "error_quadrature": [12, 13],
        "error_quadrature_absolute_change": difference,
        "divergence_error_quadrature_absolute_change": divergence_change,
        "norms": norms,
        **last,
        "original_physical_checks": metadata["original_checks"],
        "orientation_checks": metadata["orientation_checks"],
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
    """Acquire all requested fields with immutable producer identities and per-case progress."""
    levels = positive_integers(levels, label="macro resolutions", increasing=True)
    if (
        not names
        or output.exists()
        or len(set(names)) != len(names)
        or any(name not in CASES for name in names)
        or any(level not in range(1, 6) for level in levels)
    ):
        raise ValueError("Fresh output and distinct selected families are required")
    output.mkdir(parents=True)
    sources = capture_sources(output)
    report: dict[str, Any] = {
        "suite": "hdiv3d",
        "reference": "analytical",
        "configuration": {"macro_resolutions": list(levels), "families": list(names)},
        "source_sha256": sources,
        "rows": [],
        "fields": {},
        "native_whole_field_agreement_verified": False,
        "literal_literature_reproduction": False,
        "uniform_inf_sup_verified": False,
    }
    for name in names:
        for n in levels:
            row = acquire_case(name, n, output, sources)
            report["rows"].append(row)
            report["fields"][name] = {"archive": row["archive"], "sha256": row["sha256"]}
            write_progress(output / "hdiv3d.json", report)
    if any(file_digest(ROOT / name) != value for name, value in sources.items()):
        raise ValueError("An executed source changed during acquisition")
    report["source_changed"] = False
    report["complete"] = True
    write_progress(output / "hdiv3d.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--names", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--output", type=Path, default=ROOT / "build/results/hdiv3d-current")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.levels, args.names, args.output)
