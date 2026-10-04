"""Spatial and temporal Maxwell MHM verification with physical staggered error norms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from examples.maxwell_data import CavityMode
from examples.maxwell_norms import MaxwellNorms
from pymhm._legacy.models.waves.maxwell import MaxwellSolution, MaxwellStepper
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.fem.vector.curl import TangentialTraceSpace as MaxwellSkeleton
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/maxwell"


def source_hashes() -> dict[str, str]:
    """Identify all formulation, geometry, norm and acquisition owners used here."""
    names = (
        "_legacy/models/waves/maxwell",
        "fem/vector/curl",
        "fem/scalar/helmholtz",
        "meshes/triangle",
        "fem/scalar/triangle",
        "fem/scalar/tetrahedron",
        "fem/scalar/tetrahedron_topology",
        "_legacy/models/darcy/primal_3d",
        "fem/quadrature/material",
        "fem/quadrature/planar",
        "fem/scalar/operators",
        "core/contracts",
        "linalg/linear",
    )
    paths = [ROOT / f"src/pymhm/{name}.py" for name in names]
    paths += [Path(__file__), ROOT / "examples/maxwell_data.py", ROOT / "examples/maxwell_norms.py"]
    return current_source_manifest(
        {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    )


def archive(solution: MaxwellSolution, path: Path, mode: CavityMode) -> None:
    """Persist independent DG coordinates, times and actual macro geometry for field replay."""
    mesh = solution.skeleton.mesh
    np.savez_compressed(
        path,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        local_points=np.stack([local.mesh.points for local in solution.locals]),
        local_cells=np.stack([local.mesh.cells for local in solution.locals]),
        electric=np.stack(solution.electric),
        magnetic=np.stack(solution.magnetic),
        trace=solution.trace,
        degree=solution.locals[0].degree,
        dimension=mode.dimension,
        wavenumber=mode.wavenumber,
        electric_time=solution.electric_time,
        magnetic_time=solution.magnetic_time,
        time_step=solution.time_step,
    )


def cavity_row(dimension: int, resolution: int, ell: int, output: Path) -> dict[str, Any]:
    """Measure maximum-in-time errors, keeping every leapfrog sample in the maximum."""
    start = perf_counter()
    mesh = (
        TriangleMesh.unit_square(resolution) if dimension == 2 else TetraMesh.unit_cube(resolution)
    )
    base = (
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(ell) for _ in mesh.faces))
        if dimension == 2
        else TriangularSkeleton(mesh, degree=ell)
    )
    mode = CavityMode(dimension)
    with MaxwellStepper(
        mesh, time_step=0.0005, skeleton=MaxwellSkeleton(base), degree=ell + 2, quadrature_order=8
    ) as stepper:
        solution = stepper.initialize(mode.electric_shape)
        energy = solution.energy
        integrator = MaxwellNorms(solution, mode, order=8)
        maxima = integrator.measure(solution)
        drift, balance = 0.0, 0.0
        for _ in range(100):
            solution = stepper.advance()
            current = integrator.measure(solution)
            maxima = {name: max(value, current[name]) for name, value in maxima.items()}
            drift = max(drift, abs(solution.energy / energy - 1))
            balance = max(balance, abs(solution.energy_balance_residual))
        row = {
            "study": "cavity",
            "dimension": dimension,
            "resolution": resolution,
            "macro_cells": len(mesh.cells),
            "trace_degree": ell,
            "local_degree": ell + 2,
            "local_refinement": 1,
            "trace_dofs": stepper.skeleton.size,
            "time_step": stepper.time_step,
            "steps": 100,
            "electric_time": solution.electric_time,
            "magnetic_time": solution.magnetic_time,
            "cfl_product": stepper.time_step * stepper.frequency_bound,
            "initial_modified_energy": energy,
            "modified_energy_relative_drift": drift,
            "energy_balance_max": balance,
            "max_in_time": maxima,
            "final": current,
            "elapsed_seconds": perf_counter() - start,
        }
        if resolution == (8 if dimension == 2 else 4):
            path = output / f"cavity-{dimension}d-ell{ell}-n{resolution}.npz"
            archive(solution, path, mode)
            row.update(
                fields=path.name, fields_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
        return row


def temporal_row(time_step: float) -> dict[str, Any]:
    """Compare leapfrog with the exact exponential of the same constrained DG ODE.

    This isolates time error from the fixed spatial approximation. Electric
    and magnetic reference vectors are evaluated at their respective times.
    No continuum error is subtracted to infer the temporal order.
    """
    mesh, mode = TriangleMesh.unit_square(), CavityMode(2)
    with MaxwellStepper(mesh, time_step=time_step, quadrature_order=8) as stepper:
        initial = stepper.initialize(mode.electric_shape)
        me = sparse.block_diag([local.electric_mass for local in stepper.locals]).toarray()
        mh = sparse.block_diag([local.magnetic_mass for local in stepper.locals]).toarray()
        curl = sparse.block_diag([local.curl for local in stepper.locals]).toarray()
        coupling = np.zeros((len(me), stepper.skeleton.size))
        offset = 0
        for local in stepper.locals:
            rows = offset + np.arange(local.electric_mass.shape[0])
            coupling[np.ix_(rows, local.trace_dofs)] = local.coupling.toarray()
            offset += len(rows)
        inverse = np.linalg.solve(me, np.eye(len(me)))
        lift = inverse @ coupling
        restricted = inverse - lift @ np.linalg.solve(coupling.T @ lift, lift.T)
        operator = np.block(
            [
                [np.zeros_like(me), -restricted @ curl.T],
                [np.linalg.solve(mh, curl), np.zeros_like(mh)],
            ]
        )
        initial_state = np.r_[np.concatenate(initial.electric), np.concatenate(initial.magnetic)]
        # H(0)=0 makes the initial half kick exactly zero. Thus this stored E
        # is the constrained mass projection at t=0, not an interpolated E(dt/2).
        steps = round(0.04 / time_step)
        for _ in range(steps):
            result = stepper.advance()
        e_exact = (expm(result.electric_time * operator) @ initial_state)[: len(me)]
        h_exact = (expm(result.magnetic_time * operator) @ initial_state)[len(me) :]
        de, dh = (
            np.concatenate(result.electric) - e_exact,
            np.concatenate(result.magnetic) - h_exact,
        )
        return {
            "study": "time",
            "dimension": 2,
            "resolution": 1,
            "trace_degree": 1,
            "local_degree": 3,
            "time_step": time_step,
            "steps": steps,
            "electric_time": result.electric_time,
            "magnetic_time": result.magnetic_time,
            "electric_error_l2": float(np.sqrt(de @ me @ de)),
            "magnetic_error_l2": float(np.sqrt(dh @ mh @ dh)),
            "combined_error_l2": float(np.sqrt(de @ me @ de + dh @ mh @ dh)),
            "modified_energy_relative_drift": abs(result.energy / initial.energy - 1),
            "cfl_product": time_step * stepper.frequency_bound,
        }


def run(output: Path) -> None:
    """Acquire five spatial levels in each family and five independent time levels."""
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    record: dict[str, Any] = {
        "reference": "Lanteri, Paredes, Scheid, Valentin (2018), DOI 10.1137/16M110037X",
        "scope": (
            "Published 2D TM cavity data on declared triangular spaces; original full-vector "
            "3D PEC cavity; exact semidiscrete temporal reference"
        ),
        "permittivity": 1,
        "permeability": 1,
        "boundary": "homogeneous PEC",
        "assembly_quadrature": 8,
        "error_quadrature": 8,
        "norm_scope": "Physical L2 and broken DG curls, at actual staggered times",
        "source_sha256": hashes,
        "rows": [],
    }

    def checkpoint(row: dict[str, Any]) -> None:
        """Persist each complete case before starting the next calculation."""
        record["rows"].append(row)
        print(json.dumps(row), flush=True)
        (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")

    for dimension, ell, levels in (
        (2, 1, (2, 4, 8, 12, 16)),
        (2, 2, (2, 4, 8, 12, 16)),
        (3, 1, (1, 2, 3, 4, 5)),
    ):
        for resolution in levels:
            checkpoint(cavity_row(dimension, resolution, ell, output))
    for dt in (0.004, 0.002, 0.001, 0.0005, 0.00025):
        checkpoint(temporal_row(dt))
    record["source_changed_during_run"] = source_hashes() != hashes
    if record["source_changed_during_run"]:
        raise RuntimeError("a Maxwell formulation or acquisition source changed")
    (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    """Execute the numerical study outside the lightweight test suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output)


if __name__ == "__main__":
    main()
