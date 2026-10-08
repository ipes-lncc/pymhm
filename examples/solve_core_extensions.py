"""Acquire five-level analytical checks for anisotropic stress, MsHHO3D and enriched H(div).

These are explicit capability checks of the corresponding mathematical
formulations, not historical figure reproductions. Exact physical fields
provide the common reference and independently selected quadrature checks.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples import core_extension_data as data
from examples.formulations.application import hdiv_darcy as solve_darcy_hdiv3d
from examples.formulations.application import weak_stress_elasticity as solve_elasticity_mixed
from examples.formulations.application import (
    weak_stress_elasticity as solve_elasticity_mixed_polygons,
)
from examples.formulations.application import weak_stress_elasticity as solve_elasticity_tensor_rt
from pymhm.fem.scalar.tetrahedron import tetra_basis, tetra_nodal_space
from pymhm.fem.scalar.triangle import reference_basis
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.mixed import AffineMixedMesh, hdiv3d_basis, hdiv3d_dofs
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/core-extensions"


def polygon_grid(n: int) -> PolygonMesh:
    """Tile the square by nonconvex L cells and small squares with matching original edges."""
    points = np.array(
        [(i / (2 * n), j / (2 * n)) for i in range(2 * n + 1) for j in range(2 * n + 1)]
    )
    shapes = (
        ((0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2), (0, 1)),
        ((1, 1), (2, 1), (2, 2), (1, 2)),
    )
    cells = tuple(
        np.array([(2 * i + a) * (2 * n + 1) + 2 * j + b for a, b in shape])
        for i in range(n)
        for j in range(n)
        for shape in shapes
    )
    return PolygonMesh(points, cells)


def digest(path: Path) -> str:
    """Hash an executed source or archived numerical field."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_elasticity(solution: Any, name: str) -> dict[str, str]:
    """Archive one-sided fine-cell samples and all original macro edges."""
    polygons, points, values = [], [], []
    tensor = hasattr(solution, "degree")
    for cell, fine in enumerate(solution.local_meshes):
        vertices = fine.points[fine.cells]
        if tensor:
            u, stress, _, _ = solution.evaluate(cell, np.array([[0.5, 0.5]]))
            u, stress = u[:, 0], stress[:, 0]
        else:
            bary = np.ones((1, 3)) / 3
            scalar = reference_basis(solution.displacement_degree, bary)[0][0]
            u = np.einsum("i,tia->ta", scalar, solution.displacement[cell])
            stress = solution.family.evaluate(fine, solution.stress[cell], bary)[0][:, 0]
        polygons.extend(vertices)
        points.extend(vertices.mean(axis=1))
        values.extend(np.column_stack((u, stress.reshape(-1, 4))))
    points = np.asarray(points)
    exact = np.column_stack((data.displacement(points), data.stress(points).reshape(-1, 4)))
    mesh = solution.skeleton.mesh
    path = OUTPUT / f"{name}.npz"
    np.savez_compressed(
        path,
        polygons=np.array(polygons),
        points=points,
        actual=np.array(values),
        exact=exact,
        macro_edges=mesh.points[mesh.faces],
        compliance=data.compliance(),
    )
    return {"archive": path.name, "sha256": digest(path)}


def archive_section(solution: Any, mesh: Any, name: str, height: float = 0.37) -> dict[str, str]:
    """Save exact/numerical section samples with original macro intersections and executed bases."""
    polygons, points, values, segments = [], [], [], []
    # The section of a convex prism/cube is the convex hull of its edge cuts;
    # duplicate section segments are harmless overlays of original macrofaces.
    from itertools import combinations

    from scipy.spatial import ConvexHull

    def polygon(vertices: np.ndarray) -> np.ndarray:
        """Intersect a convex macrocell with the horizontal display plane."""
        cut = [point for point in vertices if point[2] == height]
        for a, b in combinations(vertices, 2):
            if (a[2] - height) * (b[2] - height) < 0:
                cut.append(a + (height - a[2]) / (b[2] - a[2]) * (b - a))
        if len(cut) < 3:
            return np.empty((0, 3))
        pts = np.unique(np.asarray(cut), axis=0)
        if len(pts) < 3 or np.linalg.matrix_rank(pts[:, :2] - pts[0, :2]) < 2:
            return np.empty((0, 3))
        return pts[ConvexHull(pts[:, :2]).vertices]

    for ids in mesh.cells:
        if isinstance(mesh, PolyhedralMesh):
            ids = np.unique(np.concatenate([mesh.faces[face] for face in ids]))
        cut = polygon(mesh.points[ids])
        if len(cut):
            segments.extend(zip(cut[:, :2], np.roll(cut[:, :2], -1, axis=0), strict=True))
    mixed = hasattr(solution, "family")
    for macro, fine in enumerate(
        solution.local_meshes if mixed else (local.mesh for local in solution.local)
    ):
        if not mixed:
            dofs, _ = tetra_nodal_space(fine, solution.degree)
        for cell, vertices in enumerate(fine.points[fine.cells]):
            cut = polygon(vertices)
            if not len(cut):
                continue
            point = cut.mean(axis=0)
            if mixed:
                xi = (point - vertices[0]) @ fine.inverse[cell].T
                basis = hdiv3d_basis(fine, solution.family, xi[None])[0][cell, 0]
                scalar = solution.family.tabulate(xi[None])[2][0]
                p = solution.pressure[macro][cell] @ scalar
                q = solution.flux[macro][hdiv3d_dofs(fine, solution.family)[cell]] @ basis
            else:
                inverse = np.linalg.inv(np.column_stack((np.ones(4), vertices)))
                bary = (np.r_[1.0, point] @ inverse)[None]
                basis, derivative = tetra_basis(solution.degree, bary)
                coeff = solution.pressure[macro][dofs[cell]]
                p = basis[0] @ coeff
                q = -coeff @ derivative[0] @ inverse[1:].T
            polygons.append(cut[:, :2])
            points.append(point)
            values.append(np.r_[p, q])
    # Variable polygon arity is encoded with offsets, never object-array pickle.
    offsets = np.r_[0, np.cumsum([len(p) for p in polygons])]
    points = np.asarray(points)
    payload = dict(
        vertices=np.concatenate(polygons),
        offsets=offsets,
        points=points,
        actual=np.array(values),
        exact=np.column_stack((data.pressure3d(points), data.flux3d(points))),
        macro_edges=np.array(segments),
    )
    if mixed:
        payload.update(
            basis_coefficients=solution.family.coefficients,
            normal_degree=np.array(solution.family.normal_degree),
            pressure_degree=np.array(solution.family.pressure_degree),
            flux=np.array(solution.flux),
            pressure=np.array(solution.pressure),
            local_points=np.array([f.points for f in solution.local_meshes]),
            local_cells=np.array([f.cells for f in solution.local_meshes]),
        )
    path = OUTPUT / f"{name}.npz"
    np.savez_compressed(path, **payload)
    return {"archive": path.name, "sha256": digest(path)}


def measure(solution: Any, suite: str, name: str, order: int) -> dict[str, float]:
    """Evaluate physical analytical errors using an independently selected quadrature."""
    if suite == "elasticity":
        if name.startswith("rectangle"):
            return solution.errors(
                data.displacement, data.stress, lambda x: -data.force(x), data.rotation, order
            )
        return {
            "displacement_l2": solution.l2_error(data.displacement, order),
            "stress_l2": solution.stress_l2_error(data.stress, order),
            "rotation_l2": solution.rotation_l2_error(data.rotation, order),
            "divergence_l2": solution.divergence_l2_error(lambda x: -data.force(x), order),
        }
    if suite == "mshho3d":
        return {
            "pressure_l2": solution.l2_error(data.pressure3d, order),
            "flux_l2": solution.flux_l2_error(data.flux3d, order),
        }
    return solution.errors(data.pressure3d, data.flux3d, order)


def run(suite: str, levels: list[int] | None = None) -> None:
    """Acquire all selected formulations, saving every completed numerical record."""
    if suite in {"mshho3d", "hdiv3d"}:
        selected = levels or [1, 2, 3, 4, 5]
        output = OUTPUT / f"{suite}-current"
        if suite == "mshho3d":
            from examples.verify_mshho3d import run as acquire_mshho3d

            report = acquire_mshho3d(selected, output=output)
        else:
            from examples.verify_hdiv3d import CASES
            from examples.verify_hdiv3d import run as acquire_hdiv3d

            report = acquire_hdiv3d(selected, list(CASES), output)
        # Consumers resolve field names relative to the summary location.
        for field in report["fields"].values():
            field["archive"] = (Path(output.name) / field["archive"]).as_posix()
        (OUTPUT / f"{suite}.json").write_text(json.dumps(report, indent=2) + "\n")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    modules = (
        [
            "materials/elasticity",
            "_legacy/models/elasticity/stress",
            "_legacy/models/elasticity/stress_tensor",
            "meshes/polygonal",
        ]
        if suite == "elasticity"
        else [
            "methods/hho",
            "methods/hho_3d",
            "_legacy/models/darcy/primal_3d",
            "_legacy/models/transport/polyhedral",
        ]
        if suite == "mshho3d"
        else [
            "fem/hdiv/family_3d",
            "fem/hdiv/moments_3d",
            "meshes/mixed",
            "_legacy/models/darcy/hdiv_3d",
        ]
    )
    modules += [
        "meshes/triangle",
        "fem/scalar/operators",
        "fem/scalar/tetrahedron",
        "fem/scalar/tetrahedron_topology",
        "fem/scalar/triangle",
        "core/contracts",
        "linalg/linear",
    ]
    modules += (
        ["fem/hdiv/bdm_family", "fem/hdiv/bdm", "_legacy/models/darcy/cartesian"]
        if suite == "elasticity"
        else []
    )
    paths = [ROOT / f"src/pymhm/{name}.py" for name in modules] + [
        Path(__file__),
        ROOT / "examples/core_extension_data.py",
    ]
    hashes = current_source_manifest(
        {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}
    )
    report: dict[str, Any] = {
        "suite": suite,
        "reference": "analytical",
        "source_sha256": hashes,
        "rows": [],
        "fields": {},
    }
    configurations = (
        ["triangle-bdm2", "rectangle-rt1", "polygon-bdm2"]
        if suite == "elasticity"
        else ["tetra-p0", "cube-p0"]
        if suite == "mshho3d"
        else ["tetra-p1-k1", "tetra-p2-k2", "tetra-p3-k1", "prism-p2-k2"]
    )
    levels = levels or ([1, 2, 4, 8, 16] if suite == "elasticity" else [1, 2, 3, 4, 5])
    for name in configurations:
        for n in levels:
            start = perf_counter()
            if suite == "elasticity":
                mesh = (
                    TriangleMesh.unit_square(n)
                    if name.startswith("triangle")
                    else CartesianMacroMesh(n)
                    if name.startswith("rectangle")
                    else polygon_grid(n)
                )
                solver = (
                    solve_elasticity_mixed
                    if name.startswith("triangle")
                    else solve_elasticity_tensor_rt
                    if name.startswith("rectangle")
                    else solve_elasticity_mixed_polygons
                )
                solution = solver(
                    mesh,
                    compliance=data.compliance(),
                    source=data.force,
                    dirichlet=data.displacement,
                    local_refinement=1,
                    quadrature_order=8,
                )
                diagnostics = {
                    "macro_equilibrium": float(np.max(abs(solution.equilibrium_residuals())))
                }
            else:
                kind = "prism" if name.startswith("prism") else "tetrahedron"
                p, k = (2, 2) if kind == "prism" else (int(name[7]), int(name[10]))
                mesh = AffineMixedMesh.unit_cube(n, kind)
                solution = solve_darcy_hdiv3d(
                    mesh,
                    pressure_degree=p,
                    normal_degree=k,
                    trace_degree=k,
                    local_refinement=1,
                    source=data.source3d,
                    quadrature_order=12,
                )
                diagnostics = {
                    "fine_equilibrium": max(
                        float(np.max(abs(r))) for r in solution.equilibrium_residuals(13)
                    ),
                    "physical_block_residual": float(solution.physical_residuals.max()),
                }
            error_order = 9 if suite == "elasticity" else 10 if suite == "mshho3d" else 12
            errors = measure(solution, suite, name, error_order)
            check = measure(solution, suite, name, error_order + 1)
            quadrature_change = max(abs(check[key] - value) for key, value in errors.items())
            if quadrature_change > 1e-8 * max(1.0, max(errors.values())):
                raise ArithmeticError("physical error norms are not quadrature converged")
            row = dict(
                case=name,
                resolution=n,
                macro_cells=len(mesh.cells),
                error_quadrature=[error_order, error_order + 1],
                error_quadrature_absolute_change=quadrature_change,
                elapsed_seconds=perf_counter() - start,
                **{key: float(value) for key, value in errors.items()},
                **diagnostics,
            )
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            if n == levels[-1]:
                report["fields"][name] = (
                    archive_elasticity(solution, name)
                    if suite == "elasticity"
                    else archive_section(solution, mesh, name)
                )
            (OUTPUT / f"{suite}.json").write_text(json.dumps(report, indent=2) + "\n")
    report["source_changed"] = any(
        digest(ROOT / name) != expected for name, expected in hashes.items()
    )
    (OUTPUT / f"{suite}.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["source_changed"]:
        raise RuntimeError("a guarded numerical source changed during acquisition")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("elasticity", "mshho3d", "hdiv3d"))
    with threadpool_limits(1):
        run(parser.parse_args().suite)
