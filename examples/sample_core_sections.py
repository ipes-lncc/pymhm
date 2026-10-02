"""Sample full one-sided polynomials on physical sections of the 3D capability cases.

Every display triangle owns its vertices. Interpolation for display therefore
stays within one fine cell and never averages across a material or macro face.
Physical error checks integrate volume polynomials independently of the plot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull
from threadpoolctl import threadpool_limits

from examples import core_extension_data as exact
from pymhm.hdiv3d_family import HDiv3DFamily, cell_quadrature
from pymhm.hdiv3d_mesh import AffineMixedMesh, hdiv3d_dofs, hdiv3d_transform
from pymhm.mesh import TriangleMesh
from pymhm.mshho3d import solve_mshho_3d
from pymhm.polyhedral import PolyhedralMesh
from pymhm.tetrahedral import TetraMesh, tetra_basis, tetra_nodal_space

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/core-extensions"
HEIGHT = 0.37


def digest(path: Path) -> str:
    """Identify a numerical input, executed source or generated field archive."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cut_polygon(vertices: np.ndarray, height: float = HEIGHT) -> np.ndarray:
    """Intersect a convex affine fine cell with a horizontal plane, in cyclic order."""
    cut = [point for point in vertices if point[2] == height]
    for a, b in combinations(vertices, 2):
        if (a[2] - height) * (b[2] - height) < 0:
            cut.append(a + (height - a[2]) / (b[2] - a[2]) * (b - a))
    if len(cut) < 3:
        return np.empty((0, 3))
    points = np.unique(np.asarray(cut), axis=0)
    if len(points) < 3 or np.linalg.matrix_rank(points[:, :2] - points[0, :2]) < 2:
        return np.empty((0, 3))
    return points[ConvexHull(points[:, :2]).vertices]


def section_samples(
    meshes: tuple[Any, ...],
    evaluate: Callable[[int, int, np.ndarray], np.ndarray],
    refinement: int,
) -> dict[str, np.ndarray]:
    """Evaluate full fields on independent refined fans of each fine-cell section."""
    reference = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, refinement)
    bary = np.column_stack((1 - reference.points.sum(axis=1), reference.points))
    points, cells, values, owners = [], [], [], []
    count = 0
    for macro, fine in enumerate(meshes):
        for cell, vertices in enumerate(fine.points[fine.cells]):
            polygon = cut_polygon(vertices)
            if not len(polygon):
                continue
            center = polygon.mean(axis=0)
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
                physical = bary @ np.array([center, a, b])
                points.append(physical)
                cells.append(reference.cells + count)
                values.append(evaluate(macro, cell, physical))
                owners.extend([(macro, cell)] * len(physical))
                count += len(physical)
    xyz = np.concatenate(points)
    return dict(
        points=xyz,
        cells=np.concatenate(cells),
        owners=np.asarray(owners, dtype=np.int64),
        actual=np.concatenate(values),
        exact=np.column_stack((exact.pressure3d(xyz), exact.flux3d(xyz))),
    )


def mixed_fields(
    archive: dict[str, np.ndarray], kind: str, refinement: int, order: int
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Replay archived flux coordinates using the executed basis and independent volume norms."""
    family = HDiv3DFamily(kind, int(archive["pressure_degree"]), int(archive["normal_degree"]))
    basis_matrix = archive["basis_coefficients"]
    meshes = tuple(
        AffineMixedMesh(points, cells, kind)
        for points, cells in zip(archive["local_points"], archive["local_cells"], strict=True)
    )
    transforms = tuple(hdiv3d_transform(m, family, coefficients=basis_matrix) for m in meshes)
    dofs = tuple(hdiv3d_dofs(m, family) for m in meshes)

    def evaluate(macro: int, cell: int, points: np.ndarray) -> np.ndarray:
        """Use the owning affine map, physical Piola transform and stored polynomial basis."""
        fine = meshes[macro]
        xi = (points - fine.points[fine.cells[cell, 0]]) @ fine.inverse[cell].T
        values, _, pressure = family.tabulate(xi, coefficients=basis_matrix)
        reference = np.einsum(
            "qia,ij,j->qa",
            values,
            transforms[macro][cell],
            archive["flux"][macro][dofs[macro][cell]],
        )
        flux = reference @ fine.jacobian[cell].T / fine.determinants[cell]
        scalar = pressure @ archive["pressure"][macro][cell]
        return np.column_stack((scalar, flux))

    points, weights = cell_quadrature(kind, order)
    reference_values, _, scalar = family.tabulate(points, coefficients=basis_matrix)
    total = np.zeros(2)
    for macro, fine in enumerate(meshes):
        physical = fine.points[fine.cells[:, 0], None] + np.einsum(
            "tab,qb->tqa", fine.jacobian, points
        )
        pressure = archive["pressure"][macro] @ scalar.T
        reference_coefficients = np.einsum(
            "tij,tj->ti", transforms[macro], archive["flux"][macro][dofs[macro]]
        )
        reference_flux = np.einsum("ti,qib->tqb", reference_coefficients, reference_values)
        flux = np.einsum("tab,tqb->tqa", fine.jacobian, reference_flux)
        flux /= fine.determinants[:, None, None]
        target = physical.reshape(-1, 3)
        pe = pressure - exact.pressure3d(target).reshape(pressure.shape)
        qe = flux - exact.flux3d(target).reshape(flux.shape)
        total += np.array(
            [
                fine.determinants @ (pe**2 @ weights),
                fine.determinants @ (np.sum(qe**2, axis=-1) @ weights),
            ]
        )
    errors = dict(zip(("pressure_l2", "flux_l2"), np.sqrt(total).tolist(), strict=True))
    return section_samples(meshes, evaluate, refinement), errors


def mshho_fields(
    name: str, resolution: int, refinement: int
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    """Recompute a finest MsHHO state, retaining complete local pressure coefficients."""
    tetra = name.startswith("tetra")
    macro = TetraMesh.unit_cube(resolution) if tetra else PolyhedralMesh.cubes(resolution)
    solution = solve_mshho_3d(
        macro, source=exact.source3d, quadrature_order=9, local_refinement=2 if tetra else 1
    )
    meshes = tuple(local.mesh for local in solution.local)
    dofs = tuple(tetra_nodal_space(mesh, solution.degree)[0] for mesh in meshes)
    inverses = tuple(
        np.linalg.inv(np.concatenate((np.ones((len(m.cells), 4, 1)), m.points[m.cells]), axis=2))
        for m in meshes
    )

    def evaluate(cell: int, fine_cell: int, points: np.ndarray) -> np.ndarray:
        """Evaluate the original continuous local pressure and its broken physical gradient."""
        inverse = inverses[cell][fine_cell]
        bary = np.column_stack((np.ones(len(points)), points)) @ inverse
        basis, derivative = tetra_basis(solution.degree, bary)
        coefficients = solution.pressure[cell][dofs[cell][fine_cell]]
        pressure = basis @ coefficients
        flux = -np.einsum("i,qib,ba->qa", coefficients, derivative, inverse[1:].T)
        return np.column_stack((pressure, flux))

    errors = dict(
        pressure_l2=solution.l2_error(exact.pressure3d, 10),
        flux_l2=solution.flux_l2_error(exact.flux3d, 10),
    )
    payload = dict(
        degree=np.array(solution.degree),
        pressure=np.array(solution.pressure),
        local_points=np.array([m.points for m in meshes]),
        local_cells=np.array([m.cells for m in meshes]),
    )
    return section_samples(meshes, evaluate, refinement), errors, payload


def run(suite: str, refinement: int = 6) -> None:
    """Publish display samples only after coefficient replay reproduces the acquired norms."""
    record_path = DATA / f"{suite}.json"
    record = json.loads(record_path.read_text())
    names = ["hdiv3d_family", "hdiv3d_general", "hdiv3d_mesh", "tetrahedral", "tetra_lagrange"]
    if suite == "mshho3d":
        names += [
            "mshho",
            "mshho3d",
            "darcy3d",
            "polyhedral",
            "polyhedral_rad",
            "hybrid",
            "solvers",
        ]
    paths = [ROOT / f"src/pymhm/{name}.py" for name in names] + [
        Path(__file__),
        ROOT / "examples/core_extension_data.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    report = dict(
        suite=suite,
        section_height=HEIGHT,
        sampling_refinement=refinement,
        sampling=(
            "Full polynomial evaluation on private fine-cell section vertices; "
            "no averaging across interfaces"
        ),
        norm_check_rtol=2e-10,
        norm_check_atol=2e-13,
        acquisition_record_sha256=digest(record_path),
        source_sha256=hashes,
        cases={},
    )
    for name, original in record["fields"].items():
        path = DATA / original["archive"]
        if digest(path) != original["sha256"]:
            raise ValueError("a numerical input differs from its acquisition digest")
        with np.load(path) as source:
            archive = {key: source[key] for key in source.files}
        row = [r for r in record["rows"] if r["case"] == name][-1]
        if suite == "hdiv3d":
            fields, errors = mixed_fields(
                archive, "prism" if name.startswith("prism") else "tetrahedron", refinement, 12
            )
            coefficients = {}
        else:
            fields, errors, coefficients = mshho_fields(name, row["resolution"], refinement)
        for key, value in errors.items():
            if not np.isclose(value, row[key], rtol=2e-10, atol=2e-13):
                raise ArithmeticError(
                    f"{name}: replayed {key} does not reproduce the acquired norm"
                )
        fields["macro_edges"] = archive["macro_edges"]
        output = DATA / f"{name}-polynomial-fields.npz"
        np.savez_compressed(output, **fields, **coefficients)
        info = dict(
            archive=output.name,
            sha256=digest(output),
            original_archive=path.name,
            original_sha256=digest(path),
            resolution=row["resolution"],
            **errors,
        )
        report["cases"][name] = info
        print(json.dumps({name: info}), flush=True)
    report["source_changed"] = hashes != {str(p.relative_to(ROOT)): digest(p) for p in paths}
    if report["source_changed"]:
        raise RuntimeError("a polynomial replay source changed during acquisition")
    (DATA / f"{suite}-field-sampling.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("hdiv3d", "mshho3d"))
    parser.add_argument("--refinement", type=int, default=6)
    arguments = parser.parse_args()
    with threadpool_limits(1):
        run(arguments.suite, arguments.refinement)
