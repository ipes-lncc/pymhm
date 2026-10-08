"""Verify five nonconvex polyhedral resolutions of the analytical three-dimensional RAD case."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from examples.field_archive import field_archive_arrays
from examples.formulations.transport_3d import transport_3d
from examples.polygon_meshes import polygon_partition
from examples.solve_rad3d import exact, gradient, physical_flux, source
from pymhm import PolyhedralMesh
from pymhm.fem.traces.polygon_3d import PolygonalSkeleton3D
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file, source_identity, source_label

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/star-polyhedra"


def partition(n: int) -> PolyhedralMesh:
    """Extrude the conforming L/square tiling; every reentrant macroface remains intact."""
    return PolyhedralMesh.extrude(polygon_partition(n, "L"), n)


def run(workers: int) -> None:
    """Acquire exact-field errors with geometric-kernel certificates and independent quadrature."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    owners = [
        source_file(f"src/pymhm/{name}.py", root=ROOT)
        for name in (
            "meshes/polyhedral",
            "meshes/geometry",
            "fem/traces/polygon_3d",
            "fem/traces/normal",
            "core/multiscale",
            "postprocessing/scalar_3d",
            "meshes/polygonal",
            "fem/scalar/tetrahedron",
            "fem/scalar/tetrahedron_topology",
            "fem/scalar/transport_3d",
            "core/contracts",
            "linalg/linear",
            "execution/cpu",
        )
    ] + [
        Path(__file__),
        source_file("examples/polygon_meshes.py", root=ROOT),
        source_file("examples/solve_rad3d.py", root=ROOT),
        source_file("examples/field_archive.py", root=ROOT),
        source_file("examples/formulations/scalar_3d.py", root=ROOT),
        source_file("examples/formulations/transport_3d.py", root=ROOT),
    ]
    hashes = current_source_manifest(source_identity(ROOT, owners), packages=("pymhm", "examples"))
    snapshot = ROOT / "build/results/star-polyhedra/acquisition-sources"
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in owners:
        destination = snapshot / f"{hashes[source_label(path, ROOT)]}-{path.name}"
        destination.write_bytes(path.read_bytes())
    record = dict(
        reference="10.1016/j.cma.2024.117089, section 5.2.1 analytical 3D problem",
        geometry="Original nonconvex star-shaped L-prism / cuboid tiling of the unit cube",
        local_degree=4,
        trace_degree=1,
        local_refinement=1,
        exact="sin(6*pi*x)*sin(4*pi*y)*sin(2*pi*z)",
        source="5.6*pi**2*u + partial_x(u)",
        diffusion=0.1,
        velocity=[1.0, 0.0, 0.0],
        reaction=0.0,
        source_sha256=hashes,
        rows=[],
        interpretation=(
            "Published PDE and polynomial degrees on an original nonconvex polyhedral family, "
            "not historical mesh connectivity. One P1 polynomial belongs to each original "
            "polygonal face. Convex-hull filling and independent traces on triangulation "
            "diagonals are not used. Each recorded kernel radius is verified against all faces."
        ),
    )
    for n in range(2, 7):
        started = perf_counter()
        mesh = partition(n)
        diameter = np.array(
            [
                np.linalg.norm(points[:, None] - points[None], axis=-1).max()
                for points in (
                    mesh.points[np.unique(np.concatenate([mesh.faces[f] for f in ids]))]
                    for ids in mesh.cells
                )
            ]
        )
        volumes = np.array([mesh.submesh(i).volumes.sum() for i in range(len(mesh.cells))])
        if not np.allclose(volumes, mesh.volumes, rtol=5e-13, atol=0):
            raise ArithmeticError("local tetrahedra do not reproduce their polyhedral volumes")
        assembly_order = 20 if n == 2 else 16 if n == 3 else 14
        error_orders = (20, 24) if n == 2 else (16, 20) if n == 3 else (14, 16)
        solution = transport_3d(
            mesh,
            degree=4,
            skeleton=PolygonalSkeleton3D(mesh),
            diffusion=0.1,
            velocity=(1.0, 0.0, 0.0),
            source=source,
            quadrature_order=assembly_order,
            backend="thread",
            workers=workers,
        )
        print(
            f"Solved n={n}, {len(mesh.cells)} macros, {solution.skeleton.size} trace DOFs",
            flush=True,
        )
        errors = []
        for order in error_orders:
            errors.append(
                np.array(
                    [
                        solution.l2_error(exact, order),
                        solution.h1_seminorm_error(gradient, order),
                        solution.flux_l2_error(physical_flux, order),
                    ]
                )
            )
        print(f"Integrated n={n}: {[a.tolist() for a in errors]}", flush=True)
        change = float(np.max(abs(errors[1] - errors[0])))
        if change > 2e-9 * max(float(np.max(errors[1])), 1):
            raise ArithmeticError("analytical error integration is unresolved")
        arrays = dict(
            macro_points=mesh.points,
            macro_centers=mesh.centers,
            kernel_radii=mesh.kernel_radii,
            local_degree=4,
            trace=solution.hybrid.trace,
        )
        for face, ids in enumerate(mesh.faces):
            arrays[f"macro_face_{face}"] = ids
        for cell, ids in enumerate(mesh.cells):
            arrays[f"macro_cell_{cell}"] = ids
            fine = solution.local_meshes[cell]
            arrays.update(
                {
                    f"points_{cell}": fine.points,
                    f"cells_{cell}": fine.cells,
                    f"coefficients_{cell}": solution.values[cell],
                }
            )
        arrays.update(field_archive_arrays(solution.hybrid.field("scalar")))
        path = OUTPUT / f"n{n}.npz"
        np.savez_compressed(path, **arrays)
        row = dict(
            n=n,
            macro_cells=len(mesh.cells),
            nonconvex_cells=int(np.sum(~mesh.convex_cells)),
            original_faces=len(mesh.faces),
            trace_dofs=solution.skeleton.size,
            fine_cells=sum(len(fine.cells) for fine in solution.local_meshes),
            kernel_radius_min=float(mesh.kernel_radii.min()),
            kernel_radius_over_diameter_min=float(np.min(mesh.kernel_radii / diameter)),
            total_volume=float(mesh.volumes.sum()),
            volume_relative_difference=float(np.max(abs(volumes - mesh.volumes) / mesh.volumes)),
            l2_error=float(errors[1][0]),
            h1_seminorm_error=float(errors[1][1]),
            V_error=float(np.sqrt(errors[1][1] ** 2 + errors[1][0] ** 2 / 3)),
            physical_flux_l2_error=float(errors[1][2]),
            assembly_order=assembly_order,
            error_orders=list(error_orders),
            errors_by_order=[a.tolist() for a in errors],
            error_quadrature_change=change,
            global_relative_residual=solution.hybrid.residual,
            archive=path.name,
            archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            elapsed_seconds=perf_counter() - started,
        )
        if hashes != current_source_manifest(
            source_identity(ROOT, owners), packages=("pymhm", "examples")
        ):
            raise RuntimeError("campaign sources changed during acquisition")
        record["rows"].append(row)
        record["source_changed_during_run"] = False
        (OUTPUT / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(row), flush=True)


def main() -> None:
    """Parse the declared CLI controls and run the original case with its thread limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    with threadpool_limits(1):
        run(parser.parse_args().workers)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_star_polyhedra").main()
