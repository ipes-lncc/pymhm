"""Compare two explicit L11 diffusion regimes without changing any published axis."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import transport as solve_transport
from examples.pgmhm_campaign import crisscross
from examples.transport_campaign import layer, natural_horizontal
from examples.transport_checkpoints import checkpoint_field, checkpoint_norm
from examples.transport_mixed_campaign import SOURCES
from pymhm.execution.cpu import map_local
from pymhm.fem.scalar.operators import p1_geometry, triangle_quadrature
from pymhm.fem.scalar.triangle import nodal_space, reference_basis
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file
from pymhm.meshes.triangle import TriangleMesh

ROOT = case_workspace()
DATA = ROOT / "examples/results/transport"


class _NormData(NamedTuple):
    """Invariants shared only within one macrocell and quadrature evaluation."""

    dofs: np.ndarray
    basis: np.ndarray
    gradient: np.ndarray
    weights: np.ndarray
    areas: np.ndarray
    exact: np.ndarray
    exact_gradient_x: np.ndarray


def _norm_data(fine: TriangleMesh, epsilon: float, order: int) -> _NormData:
    """Prepare unchanged physical tabulations and analytical values once per call."""
    bary, weights = triangle_quadrature(order)
    points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
    dofs, _ = nodal_space(fine, 1)
    basis, derivative, _ = reference_basis(1, bary)
    geometry, _ = p1_geometry(fine)
    gradient = np.einsum("qin,tna->tqia", derivative, geometry)
    return _NormData(
        dofs,
        basis,
        gradient,
        weights,
        fine.areas,
        layer(points.reshape(-1, 2), epsilon).reshape(len(fine.cells), -1),
        1 - np.exp((points[:, :, 0] - 1) / epsilon) / (epsilon * -np.expm1(-1 / epsilon)),
    )


def _norm_field(data: _NormData, coefficients: np.ndarray) -> np.ndarray:
    """Preserve the original integrands and floating-point reduction for one field."""
    difference = coefficients[data.dofs] @ data.basis.T - data.exact
    g = np.einsum("ti,tqia->tqa", coefficients[data.dofs], data.gradient)
    g[:, :, 0] -= data.exact_gradient_x
    return np.array(
        [
            data.areas @ (difference**2 @ data.weights),
            data.areas @ (np.sum(g**2, axis=2) @ data.weights),
        ]
    )


def norm_contribution(task: tuple[TriangleMesh, np.ndarray, float, int]) -> np.ndarray:
    """Return one macrocell's two squared errors, with unchanged arithmetic."""
    fine, coefficients, epsilon, order = task
    return _norm_field(_norm_data(fine, epsilon, order), coefficients)


def norm_contributions(task: tuple[TriangleMesh, np.ndarray, float, int]) -> np.ndarray:
    """Evaluate P1 fields on the same macrocell, reusing only integration invariants.

    Each coefficient row is an independent field on the supplied local mesh.
    The returned rows preserve the separate-call integrands and reduction order;
    no averaging across fields or macrointerfaces is introduced.
    """
    fine, coefficients, epsilon, order = task
    if coefficients.ndim != 2 or coefficients.shape[1] != len(fine.points):
        raise ValueError("grouped P1 coefficients need one row per field and one column per node")
    data = _norm_data(fine, epsilon, order)
    return np.array([_norm_field(data, field) for field in coefficients]).reshape(-1, 2)


def acquire(
    configuration: tuple[int, float, int],
    *,
    local_workers: int = 1,
    output_directory: Path = DATA,
) -> dict:
    """Retain local constants algebraically while keeping physical P1/P0 spaces."""
    n, epsilon, refinement = configuration
    paths = (
        *SOURCES,
        "examples/transport_coefficient_controls.py",
        "examples/transport_checkpoints.py",
        "examples/campaign_provenance.py",
        "src/pymhm/execution/cpu.py",
    )
    hashes = current_source_manifest(
        {p: hashlib.sha256(source_file(p, root=ROOT).read_bytes()).hexdigest() for p in paths},
        packages=("pymhm", "examples"),
    )
    start = perf_counter()
    mesh = crisscross(n)
    with threadpool_limits(1):
        solution = solve_transport(
            mesh,
            diffusion=epsilon,
            velocity=(1, 0),
            source=1,
            dirichlet=0,
            diffusive_flux=natural_horizontal(mesh),
            dirichlet_enforcement="strong",
            degree=1,
            local_refinement=refinement,
            stabilization="galerkin",
            quadrature_order=5,
            coarse_space="constants",
            backend="process" if local_workers > 1 else "serial",
            workers=local_workers,
        )
        print("assembled", n, solution.hybrid.residual, flush=True)
        output_directory.mkdir(parents=True, exist_ok=True)
        archive = output_directory / f"mixed-coefficient-e{epsilon:g}-n{n}-r{refinement}.npz"
        checkpoint = checkpoint_field(
            archive,
            dict(
                macro_points=mesh.points,
                macro_cells=mesh.cells,
                local_points=np.stack([m.points for m in solution.local_meshes]),
                local_cells=np.stack([m.cells for m in solution.local_meshes]),
                coefficients=np.stack(solution.values),
                trace=solution.hybrid.trace,
            ),
            dict(
                epsilon=epsilon,
                macro_resolution=n,
                local_refinement=refinement,
                source_hashes=hashes,
                original_hybrid_residual=solution.hybrid.residual,
                retained_constant_coordinates=sum(len(v) for v in solution.hybrid.coarse),
            ),
        )
        measurements = {}
        for order in (8, 12):
            squared = np.zeros(2)
            tasks = [
                (fine, coefficients, epsilon, order)
                for fine, coefficients in zip(solution.local_meshes, solution.values, strict=True)
            ]
            contributions = map_local(
                norm_contribution,
                tasks,
                backend="process" if local_workers > 1 else "serial",
                workers=local_workers,
            )
            for contribution in contributions:
                squared += contribution
            measurements[str(order)] = dict(
                l2_error=float(np.sqrt(squared[0])), broken_h1_error=float(np.sqrt(squared[1]))
            )
            checkpoint_norm(archive, checkpoint, order, measurements[str(order)])
            print("errors", n, order, measurements[str(order)], flush=True)
    assert all(
        hashlib.sha256(source_file(p, root=ROOT).read_bytes()).hexdigest() == h
        for p, h in hashes.items()
    )
    record = dict(
        epsilon=epsilon,
        macro_resolution=n,
        macro_triangles=len(mesh.cells),
        local_refinement=refinement,
        local_degree=1,
        trace_degree=0,
        free_trace_dofs=int(np.sum(mesh.face_cells[:, 1] >= 0)),
        retained_constant_coordinates=sum(len(v) for v in solution.hybrid.coarse),
        physical_space="Continuous P1 within each macrocell, P0 per interior macroface",
        algebraic_retention=(
            "Local constants retained; this changes condensed coordinates, "
            "not the physical approximation space"
        ),
        execution_backend="process" if local_workers > 1 else "serial",
        local_workers=local_workers,
        quadrature=measurements,
        original_hybrid_residual=solution.hybrid.residual,
        archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_hashes=hashes,
        source_changed_during_run=False,
        seconds=perf_counter() - start,
    )
    archive.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps({k: record[k] for k in ("epsilon", "macro_resolution", "quadrature")}),
        flush=True,
    )
    return record


def main() -> None:
    """Acquire explicitly selected coefficient controls separately from the nominal case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--local-refinement", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--local-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DATA)
    options = parser.parse_args()
    if (
        min(options.resolutions) < 1
        or options.epsilon <= 0
        or options.local_refinement < 1
        or options.workers < 1
        or options.local_workers < 1
    ):
        parser.error("all resolutions, diffusion and worker counts must be positive")
    configurations = [(n, options.epsilon, options.local_refinement) for n in options.resolutions]
    DATA.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=options.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        records = list(
            pool.map(
                partial(
                    acquire, local_workers=options.local_workers, output_directory=options.output
                ),
                configurations,
            )
        )
    output = dict(
        method="L11 Galerkin P1/P0 mixed-wall coefficient controls",
        interpretation=(
            "epsilon=1 is the Figure7 regime; Figure12 prints epsilon=0.1. "
            "Agreement of a different regime is a compatibility observation, "
            "not identification of the historical data."
        ),
        records=records,
    )
    (options.output / f"mixed-coefficient-e{options.epsilon:g}.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.transport_coefficient_controls").main()
