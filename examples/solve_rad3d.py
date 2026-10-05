"""Acquire the L12 oscillatory three-dimensional RAD case with explicit mesh provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.transport.rad_3d import (
    RAD3DSolution,
    solve_rad_3d,
    solve_rad_3d_conforming,
)
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.tetrahedron import TetraMesh

ROOT = Path(__file__).resolve().parents[1]
WAVE = 2 * np.pi * np.array([3.0, 2.0, 1.0])


def exact(points: np.ndarray) -> np.ndarray:
    """Evaluate the literal smooth oscillatory field from section 5.2.1 of L12."""
    return np.sin(points * WAVE).prod(axis=1)


def gradient(points: np.ndarray) -> np.ndarray:
    """Differentiate the trigonometric exact scalar independently of finite elements."""
    sine, cosine = np.sin(points * WAVE), np.cos(points * WAVE)
    return np.column_stack(
        [
            WAVE[a] * cosine[:, a] * np.prod(sine[:, [b for b in range(3) if b != a]], axis=1)
            for a in range(3)
        ]
    )


def source(points: np.ndarray) -> np.ndarray:
    """Evaluate -.1 Laplacian(u)+partial_x(u) for the published constant coefficients."""
    return 0.1 * float(WAVE @ WAVE) * exact(points) + gradient(points)[:, 0]


def physical_flux(points: np.ndarray) -> np.ndarray:
    """Evaluate the conservative physical flux -.1 grad(u)+(u,0,0)."""
    result = -0.1 * gradient(points)
    result[:, 0] += exact(points)
    return result


def norms(solution: RAD3DSolution, order: int) -> dict[str, float]:
    """Evaluate independently integrated scalar, gradient and conservative flux errors."""
    scalar = solution.l2_error(exact, order=order)
    seminorm = solution.h1_seminorm_error(gradient, order=order)
    return dict(
        l2_error=scalar,
        h1_seminorm_error=seminorm,
        V_error=float(np.sqrt(seminorm**2 + scalar**2 / 3)),
        physical_flux_l2_error=solution.flux_l2_error(physical_flux, order=order),
        backward_residual=solution.residual,
    )


def main() -> None:
    """Run five original MHM and classical P2 solves without external reference programs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/rad3d.json")
    parser.add_argument("--refinement", type=int, default=2)
    args = parser.parse_args()
    paths = [
        Path(__file__),
        *(
            ROOT / f"src/pymhm/{name}"
            for name in (
                "_legacy/models/transport/rad_3d.py",
                "_legacy/models/darcy/primal_3d.py",
                "fem/scalar/tetrahedron.py",
                "fem/scalar/tetrahedron_topology.py",
                "core/contracts.py",
                "linalg/linear.py",
                "execution/cpu.py",
            )
        ),
    ]
    hashes = current_source_manifest(
        {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with threadpool_limits(limits=1):
        for n in (1, 2, 3, 4, 5):
            mesh = TetraMesh.unit_cube(n)
            skeleton = TriangularSkeleton(mesh, degree=1)
            solution = solve_rad_3d(
                mesh,
                diffusion=0.1,
                velocity=[1, 0, 0],
                source=source,
                degree=4,
                skeleton=skeleton,
                local_refinement=args.refinement,
                quadrature_order=8,
            )
            classical_mesh = TetraMesh.unit_cube(n * args.refinement)
            classical = solve_rad_3d_conforming(
                classical_mesh, degree=2, diffusion=0.1, velocity=[1, 0, 0], source=source, order=8
            )
            row = dict(
                n=n,
                macro_tetrahedra=len(mesh.cells),
                fine_tetrahedra=sum(len(m.cells) for m in solution.local_meshes),
                global_mhm_unknowns=skeleton.size + len(mesh.cells),
                classical_total_dofs=len(classical.values[0]),
                mhm=norms(solution, 8),
                classical=norms(classical, 8),
            )
            if max(row["mhm"]["backward_residual"], row["classical"]["backward_residual"]) > 1e-10:
                raise RuntimeError("physical algebraic check failed")
            if n == 5:
                row["quadrature_order10"] = dict(
                    mhm=norms(solution, 10), classical=norms(classical, 10)
                )
                archive = args.output.with_suffix(".npz")
                np.savez_compressed(
                    archive,
                    macro_points=mesh.points,
                    macro_cells=mesh.cells,
                    local_points=np.stack([m.points for m in solution.local_meshes]),
                    local_cells=np.stack([m.cells for m in solution.local_meshes]),
                    values=np.stack(solution.values),
                    trace=solution.hybrid.trace,
                    classical_points=classical_mesh.points,
                    classical_cells=classical_mesh.cells,
                    classical_values=classical.values[0],
                )
                row.update(
                    fields=archive.name,
                    fields_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                )
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = dict(
        literature="Araya et al., CMAME 428 (2024) 117089, section 5.2.1 and Figure 4",
        problem=(
            "Unit cube, K=.1 I, beta=(1,0,0), reaction=0; "
            "u=sin(6pi x)sin(4pi y)sin(2pi z); homogeneous Dirichlet"
        ),
        formulation=(
            "Conservative RAD skew form, Robin skeleton flux "
            "(-K grad(u)+beta*u/2).n; Galerkin local solve"
        ),
        mhm=dict(
            degree=4,
            face_degree=1,
            face_subdivisions=1,
            local_edge_refinement=args.refinement,
            dirichlet="weak face moments",
            local_retained="physical constant",
        ),
        classical=dict(
            degree=2, dirichlet="strong nodal", mesh="Freudenthal cube, same fine Cartesian spacing"
        ),
        quadrature=dict(
            assembly=8,
            errors=8,
            finest_sensitivity=10,
            V="sqrt(H1_seminorm_squared+L2_squared/diameter_squared), diameter_squared=3",
        ),
        reproduction_limit=(
            "Published PDE and polynomial degrees; deterministic Freudenthal/red-refined "
            "meshes differ from the irregular meshes in Figure 4. Analytical refinement "
            "comparison, not identical published mesh data."
        ),
        rows=rows,
        source_sha256=hashes,
        source_changed_during_run=any(
            hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p.relative_to(ROOT).as_posix()]
            for p in paths
        ),
        timestamp_utc=datetime.now(UTC).isoformat(),
        native_threads=1,
    )
    if report["source_changed_during_run"]:
        raise RuntimeError("sources changed during acquisition")
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
