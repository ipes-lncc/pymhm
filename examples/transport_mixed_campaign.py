"""L11 Figure 12 physical data with mixed walls and separate local-resolution checks."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from threadpoolctl import threadpool_limits

from examples.field_sampling import sample_field
from examples.formulations.application import transport as solve_transport
from examples.pgmhm_campaign import crisscross
from examples.transport_campaign import layer, natural_horizontal
from pymhm.adaptivity.transport import (
    TransportBounds,
    estimate_transport_faces,
    refine_skeleton_faces,
)
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import tabulate
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.scalar import prepare_scalar_trace
from pymhm.io.provenance import current_source_manifest
from pymhm.postprocessing.solutions import ScalarSolution

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/transport"
SOURCES = (
    "src/pymhm/adaptivity/transport.py",
    "src/pymhm/_legacy/models/transport/rad.py",
    "src/pymhm/_legacy/models/transport/solver.py",
    "src/pymhm/fem/traces/scalar.py",
    "src/pymhm/fem/scalar/triangle.py",
    "src/pymhm/core/contracts.py",
    "src/pymhm/linalg/linear.py",
    "examples/transport_mixed_campaign.py",
    "examples/transport_campaign.py",
    "examples/pgmhm_campaign.py",
)


def digest(path: Path) -> str:
    """Return a byte digest for acquisition and replay provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(points: np.ndarray) -> np.ndarray:
    """Evaluate the section 5.1 layer with the Figure 12 diffusion epsilon=0.1."""
    return layer(points, 0.1)


def norms(solution: ScalarSolution, order: int) -> dict[str, float]:
    """Integrate absolute L2 and unweighted broken H1 errors independently of assembly."""
    bary, weights = triangle_quadrature(order)
    errors = np.zeros(2)
    for mesh, coefficients in zip(solution.local_meshes, solution.values, strict=True):
        points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        dofs, _, basis, gradient, _ = tabulate(mesh, solution.degree, bary)
        difference = coefficients[dofs] @ basis.T - exact(points.reshape(-1, 2)).reshape(
            len(mesh.cells), -1
        )
        numerical_gradient = np.einsum("ti,tqia->tqa", coefficients[dofs], gradient)
        numerical_gradient[:, :, 0] -= 1 - np.exp((points[:, :, 0] - 1) / 0.1) / (
            0.1 * -np.expm1(-10)
        )
        errors[0] += mesh.areas @ (difference**2 @ weights)
        errors[1] += mesh.areas @ (np.sum(numerical_gradient**2, axis=2) @ weights)
    return dict(l2_error=float(np.sqrt(errors[0])), broken_h1_error=float(np.sqrt(errors[1])))


def continuity_moments(solution: ScalarSolution) -> float:
    """Check the normalized P0 continuity moments on the actual skeletal segments."""
    mesh, maximum = solution.skeleton.mesh, 0.0
    gauss, weights = leggauss(solution.degree + 2)
    for face, space in enumerate(solution.skeleton.faces):
        first, second = mesh.face_cells[face]
        if second < 0:
            continue
        traces = [
            prepare_scalar_trace(mesh, solution.local_meshes[cell], face, solution.degree)
            for cell in (first, second)
        ]
        for left, right in zip(space.breaks[:-1], space.breaks[1:], strict=True):
            # Uniform local boundary edges have parameter length 1/r. Split
            # explicitly to integrate each polynomial piece on both sides.
            cuts = sorted(
                {
                    left,
                    right,
                    *(
                        float(t)
                        for trace in traces
                        for positions, _ in trace.pieces
                        for t in positions
                        if left < t < right
                    ),
                }
            )
            moment = 0.0
            for low, high in zip(cuts[:-1], cuts[1:], strict=True):
                parameter = low + (1 + gauss) * (high - low) / 2
                values = [
                    trace.evaluate(solution.values[cell], parameter)
                    for cell, trace in zip((first, second), traces, strict=True)
                ]
                moment += weights @ (values[0] - values[1]) * (high - low) / 2
            maximum = max(maximum, abs(float(moment)) / (right - left))
    return maximum


def acquire(skeleton: SkeletonSpace, refinement: int) -> ScalarSolution:
    """Keep the paper's P1/Galerkin locals, P0 trace and physical mixed boundary data."""
    return solve_transport(
        skeleton.mesh,
        skeleton=skeleton,
        diffusion=0.1,
        velocity=(1, 0),
        source=1,
        dirichlet=0,
        diffusive_flux=natural_horizontal(skeleton.mesh),
        dirichlet_enforcement="strong",
        degree=1,
        local_refinement=refinement,
        stabilization="galerkin",
        quadrature_order=5,
        coarse_space="kernel",
    )


def record(solution: ScalarSolution, name: str, refinement: int) -> dict[str, Any]:
    """Persist canonical P1 coefficients, independent traces and error measurements."""
    skeleton = solution.skeleton
    indicator = estimate_transport_faces(solution, TransportBounds(0.1, 1, 0))
    archive = DATA / f"{name}.npz"
    samples = sample_field(solution.local_meshes, solution.values, 1, 2)
    np.savez_compressed(
        archive,
        **samples,
        exact=exact(samples["points"]),
        macro_points=skeleton.mesh.points,
        macro_cells=skeleton.mesh.cells,
        local_points=np.stack([mesh.points for mesh in solution.local_meshes]),
        local_cells=np.stack([mesh.cells for mesh in solution.local_meshes]),
        coefficients=np.stack(solution.values),
        trace=solution.hybrid.trace,
        trace_breaks=np.concatenate([np.asarray(face.breaks) for face in skeleton.faces]),
        trace_break_offsets=np.r_[0, np.cumsum([len(face.breaks) for face in skeleton.faces])],
    )
    natural = np.concatenate([skeleton.dofs(int(f)) for f in solution.natural_faces])
    return dict(
        name=name,
        archive=archive.name,
        sha256=digest(archive),
        macro_triangles=len(skeleton.mesh.cells),
        local_refinement=refinement,
        local_triangles=sum(len(mesh.cells) for mesh in solution.local_meshes),
        trace_dofs=skeleton.size,
        free_trace_dofs=sum(
            len(skeleton.dofs(i))
            for i, cells in enumerate(skeleton.mesh.face_cells)
            if cells[1] >= 0
        ),
        retained_coarse_dofs=sum(len(row) for row in solution.hybrid.coarse),
        indicator=indicator.total,
        maximum_face_segments=max(len(face.degrees) for face in skeleton.faces),
        physical_hybrid_residual=solution.hybrid.residual,
        maximum_continuity_moment=continuity_moments(solution),
        maximum_prescribed_wall_multiplier=float(np.max(abs(solution.hybrid.trace[natural]))),
        quadrature={str(order): norms(solution, order) for order in (8, 12)},
    )


def collect() -> dict[str, Any]:
    """Run bounded spatial, adaptive and same-skeleton local-resolution controls."""
    source_hashes = current_source_manifest({name: digest(ROOT / name) for name in SOURCES})
    start = perf_counter()
    report: dict[str, Any] = dict(
        source_hashes=source_hashes,
        doi="10.1137/130938499",
        target="Section 5.1, Figure 12: epsilon=0.1, a=1, sigma=0, f=1",
        boundaries="Strong u=0 at x=0,1; physical diffusive flux zero at y=0,1",
        method="P1 continuous local Galerkin, P0 discontinuous skeletal segments; no SUPG",
        theta=0.75,
        adaptive_macro_mesh="16 crisscross triangles from a 2x2 square grid",
        historical_scope=(
            "Physical data and published space family; the complete Figure 12 local mesh "
            "and initial adaptive connectivity are not specified. Own local-refinement "
            "controls retain the final skeletal space exactly."
        ),
        norm_convention="Absolute L2 error and unweighted broken H1 seminorm against Eq5.2",
        dof_convention=(
            "free_trace_dofs counts only unconstrained interior multipliers; prescribed "
            "wall/Dirichlet coordinates are excluded; no retained coarse modes"
        ),
        adaptive=[],
        spatial=[],
        local_controls=[],
    )
    skeleton = SkeletonSpace(crisscross(2))
    for iteration in range(8):
        solution = acquire(skeleton, 16)
        row = record(solution, f"mixed-adaptive-{iteration}", 16)
        row["iteration"] = iteration
        report["adaptive"].append(row)
        print("adaptive", row["free_trace_dofs"], row["quadrature"]["12"], flush=True)
        if iteration < 7:
            indicator = estimate_transport_faces(solution, TransportBounds(0.1, 1, 0))
            skeleton = refine_skeleton_faces(skeleton, indicator.mark(0.75))
    for refinement in (8, 32):
        row = record(acquire(skeleton, refinement), f"mixed-local-r{refinement}", refinement)
        report["local_controls"].append(row)
        print("local", refinement, row["quadrature"]["12"], flush=True)
    for resolution in (1, 2, 3, 4, 6):
        space = SkeletonSpace(crisscross(resolution))
        row = record(acquire(space, 16), f"mixed-spatial-n{resolution}", 16)
        row["resolution"] = resolution
        report["spatial"].append(row)
        print("spatial", resolution, row["quadrature"]["12"], flush=True)
    assert all(digest(ROOT / name) == value for name, value in source_hashes.items())
    report["source_changed_during_run"] = False
    report["total_seconds_including_errors_and_archives"] = perf_counter() - start
    return report


def spatial_control(resolution: int) -> dict[str, Any]:
    """Acquire one additional unchanged P1/P0/r16 macro-resolution control."""
    before = current_source_manifest({name: digest(ROOT / name) for name in SOURCES})
    start = perf_counter()
    with threadpool_limits(1):
        row = record(
            acquire(SkeletonSpace(crisscross(resolution)), 16),
            f"mixed-spatial-n{resolution}",
            16,
        )
    assert all(digest(ROOT / name) == value for name, value in before.items())
    row.update(
        resolution=resolution,
        acquisition_source_hashes=before,
        source_changed_during_run=False,
        seconds_including_errors_and_archive=perf_counter() - start,
    )
    print("spatial", resolution, row["quadrature"]["12"], flush=True)
    return row


def main() -> None:
    """Execute the explicit numerical campaign outside the portable CI suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append-spatial", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    options = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if options.append_spatial:
        if options.workers < 1 or min(options.append_spatial) < 1:
            parser.error("positive resolutions and worker count are required")
        result = json.loads((DATA / "mixed-campaign.json").read_text())
        present = {row["resolution"] for row in result["spatial"]}
        if present.intersection(options.append_spatial):
            parser.error("requested resolutions already exist in this record")
        with ProcessPoolExecutor(
            max_workers=options.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            result["spatial"].extend(pool.map(spatial_control, options.append_spatial))
        result["spatial"].sort(key=lambda row: row["resolution"])
    else:
        with threadpool_limits(1):
            result = collect()
    (DATA / "mixed-campaign.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
