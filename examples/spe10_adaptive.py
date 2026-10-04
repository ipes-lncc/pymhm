"""Adaptive L09 Darcy spaces and a separately refined classical RT2 SPE10 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

import pymhm
from examples.solve_spe10 import load_layer, pressure_boundary
from pymhm._legacy.models.darcy.mixed_rt import pressure_basis, solve_darcy_rt_conforming
from pymhm.fem.hdiv.rt import RTField
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/spe10-adaptive"
DOMAIN = np.array([1200.0, 2200.0])


def source_path(name: str) -> Path:
    """Resolve canonical source names to the package actually imported by this process."""
    if name.startswith("src/pymhm/"):
        return Path(pymhm.__file__).resolve().parent / name.removeprefix("src/pymhm/")
    return ROOT / name


def mesh_rectangle(nx: int, ny: int) -> TriangleMesh:
    """Split a physical reservoir rectangle along southwest/northeast diagonals."""
    mesh = TriangleMesh.unit_square(nx, ny)
    return TriangleMesh(mesh.points * DOMAIN, mesh.cells)


def natural_faces(mesh: TriangleMesh) -> dict[int, float]:
    """Prescribe zero physical outward flux on the left and right sides."""
    normals = mesh.normals
    return {int(face): 0.0 for face in mesh.boundary_faces if abs(normals[face, 0]) > 0.5}


def hashes() -> dict[str, str]:
    """Fingerprint numerical source and unchanged physical layer data."""
    names = [
        f"src/pymhm/{name}.py"
        for name in (
            "_legacy/models/darcy/mixed_rt",
            "fem/hdiv/rt",
            "fem/quadrature/material",
            "linalg/linear",
            "meshes/triangle",
            "io/reservoir",
            "_legacy/models/darcy/primal",
            "adaptivity/darcy",
            "meshes/refinement",
            "estimators/darcy_energy",
            "recovery/moments",
        )
    ]
    names += ["examples/spe10_adaptive.py", "examples/results/spe10/layer-36.npz"]
    return current_source_manifest(
        {name: hashlib.sha256(source_path(name).read_bytes()).hexdigest() for name in names}
    )


class StructuredRT:
    """Archived global RT2/P2 fields on an explicitly indexed rectangular triangulation."""

    def __init__(self, nx: int, ny: int, pressure: np.ndarray, flux: np.ndarray) -> None:
        """Keep the original mesh and physical coefficients without projection."""
        self.nx, self.ny = nx, ny
        self.mesh = mesh_rectangle(nx, ny)
        self.areas = self.mesh.areas
        self.areas.setflags(write=False)
        self.pressure = pressure
        self._flux_field = RTField(self.mesh, flux, 2)
        self.flux = self._flux_field.coefficients

    @classmethod
    def load(cls, path: Path) -> StructuredRT:
        """Restore a numerical reference from its original coefficient arrays."""
        with np.load(path) as arrays:
            return cls(int(arrays["nx"]), int(arrays["ny"]), arrays["pressure"], arrays["flux"])

    def evaluate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the containing triangle, with a declared southwest-diagonal side rule."""
        scaled = points / DOMAIN * np.array([self.nx, self.ny])
        ij = np.minimum(np.floor(scaled).astype(int), [self.nx - 1, self.ny - 1])
        local = scaled - ij
        upper = local[:, 1] > local[:, 0]
        owners = 2 * (ij[:, 1] * self.nx + ij[:, 0]) + upper
        bary = np.empty((len(points), 3))
        x, y = local.T
        bary[~upper] = np.column_stack((1 - x, x - y, y))[~upper]
        bary[upper] = np.column_stack((1 - y, x, y - x))[upper]
        p = np.einsum("qi,qi->q", pressure_basis(2, bary), self.pressure[owners])
        q, div = self._flux_field.evaluate(points, owners)
        return p, q, div


def reference_squared_batch(
    current: StructuredRT, previous: StructuredRT | None, order: int, ids: np.ndarray, material: Any
) -> np.ndarray:
    """Integrate one ordered cell batch with the shared physical norm arithmetic."""
    mesh = current.mesh
    areas = current.areas
    vertices = mesh.points[mesh.cells[ids]]
    batch = TriangleMesh(vertices.reshape(-1, 2), np.arange(3 * len(ids)).reshape(-1, 3))
    bary, weights, values = material_triangle_quadrature(batch, material, order)
    points = np.einsum("tqi,tia->tqa", bary, vertices)
    p, q, div = current.evaluate(points.reshape(-1, 2))
    factor = (areas[ids, None] * weights).ravel().astype(np.longdouble)
    inverse = np.linalg.inv(values)
    totals = np.zeros(7, dtype=np.longdouble)
    totals[:3] = [
        _weighted_integral(factor, p**2),
        _weighted_integral(factor, np.sum(q * q, axis=1)),
        _weighted_integral(factor, div**2),
    ]
    totals[5] = _weighted_integral(factor, np.einsum("qa,qab,qb->q", q, inverse, q))
    if previous is not None:
        old_p, old_q, _ = previous.evaluate(points.reshape(-1, 2))
        totals[3:5] = [
            _weighted_integral(factor, (p - old_p) ** 2),
            _weighted_integral(factor, np.sum((q - old_q) ** 2, axis=1)),
        ]
        difference = q - old_q
        totals[6] = _weighted_integral(
            factor, np.einsum("qa,qab,qb->q", difference, inverse, difference)
        )
    return totals


def reference_norm_record(norms: np.ndarray, order: int) -> dict[str, float]:
    """Name physical norms and normalize increments by the finer reference."""
    return {
        "pressure_l2": float(norms[0]),
        "flux_l2": float(norms[1]),
        "divergence_l2": float(norms[2]),
        "previous_pressure_difference_l2": float(norms[3]),
        "previous_flux_difference_l2": float(norms[4]),
        "previous_pressure_relative_difference": float(norms[3] / norms[0]),
        "previous_flux_relative_difference": float(norms[4] / norms[1]),
        "flux_energy_norm": float(norms[5]),
        "previous_flux_energy_difference": float(norms[6]),
        "previous_flux_energy_relative_difference": float(norms[6] / norms[5]),
        "norm_quadrature_order": order,
    }


def reference_integrals(
    current: StructuredRT, previous: StructuredRT | None = None, order: int = 5
) -> dict[str, float]:
    """Integrate physical L2/energy norms on nested triangles and material intersections."""
    if previous is not None and (current.nx != 2 * previous.nx or current.ny != 2 * previous.ny):
        raise ValueError("successive reference integration requires the declared dyadic hierarchy")
    material = load_layer()
    partials = [
        reference_squared_batch(
            current,
            previous,
            order,
            np.arange(first, min(first + 512, len(current.mesh.cells))),
            material,
        )
        for first in range(0, len(current.mesh.cells), 512)
    ]
    norms = np.sqrt(np.sum(np.asarray(partials), axis=0, dtype=np.longdouble))
    return reference_norm_record(norms, order)


def _weighted_integral(weights: np.ndarray, values: np.ndarray) -> np.longdouble:
    """Reduce positive quadrature products pairwise, using wider precision where available."""
    return np.sum(weights * values, dtype=np.longdouble)


def acquire_references(resolutions: list[int], solver: str = "scipy") -> list[dict[str, Any]]:
    """Solve five global RT2 levels, preserving their own residual and refinement evidence."""
    records = []
    previous = None
    for nx in resolutions:
        ny = 7 * nx // 2
        mesh = mesh_rectangle(nx, ny)
        started = perf_counter()
        fingerprint = hashes()
        solution = solve_darcy_rt_conforming(
            mesh,
            degree=2,
            permeability=load_layer(),
            dirichlet=pressure_boundary,
            neumann=natural_faces(mesh),
            quadrature_order=5,
            solver=solver,
        )
        seconds = perf_counter() - started
        current = StructuredRT(nx, ny, solution.pressure[0], solution.flux[0])
        archive = f"reference-rt2-{nx}x{ny}.npz"
        np.savez_compressed(
            DATA / archive,
            nx=nx,
            ny=ny,
            pressure=current.pressure,
            flux=current.flux,
            points=mesh.points,
            cells=mesh.cells,
        )
        moments = current.flux[3 * mesh.cell_faces] * mesh.signs
        bottom = [
            face for face in mesh.boundary_faces if np.all(mesh.points[mesh.faces[face], 1] == 0)
        ]
        row = {
            "nx": nx,
            "ny": ny,
            "cells": len(mesh.cells),
            "dofs": current.pressure.size + current.flux.size,
            "degree": 2,
            "assembly_order": 5,
            "solver": solver,
            "solve_seconds": seconds,
            "residual": solution.residual,
            "archive": archive,
            "fine_balance_linf": float(np.max(abs(moments.sum(axis=1)))),
            "inflow": -float(current.flux[3 * np.array(bottom)].sum()),
            **reference_integrals(current, previous),
            "source_hashes": fingerprint,
            "source_changed_during_solve": fingerprint != hashes(),
        }
        if previous is not None:
            row["independent_order6"] = reference_integrals(current, previous, order=6)
        records.append(row)
        (DATA / "references.json").write_text(json.dumps(records, indent=2) + "\n")
        print("reference", row, flush=True)
        previous = current
    return records


def main() -> None:
    """Acquire a standalone numerical campaign, separately from the light test suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", action="store_true")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--solver", default="scipy")
    options = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        if options.references:
            acquire_references(options.resolutions, options.solver)


if __name__ == "__main__":
    main()
