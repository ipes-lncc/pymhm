"""Acquire original general-tensor primal elasticity convergence separately from CI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymhm.postprocessing.primal_elasticity import PrimalElasticitySolution

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import elasticity as solve_elasticity
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.estimators.elasticity import estimate_primal_elasticity_error
from pymhm.fem.scalar.triangle import nodal_space, reference_basis
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file, source_identity

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/primal-elasticity"
STIFFNESS = np.array([[5.0, 1.0, 0.4], [1.0, 4.0, 0.3], [0.4, 0.3, 2.0]])


@dataclass(frozen=True)
class TensorData:
    """Analytical trigonometric displacement and consistent anisotropic body force."""

    variable: bool = True

    def factor(self, points: np.ndarray) -> np.ndarray:
        """Return the positive scalar multiplier of the fixed anisotropic Kelvin tensor."""
        return 1 + points[:, 0] + 2 * points[:, 1] if self.variable else np.ones(len(points))

    def constitutive(self, points: np.ndarray) -> np.ndarray:
        """Evaluate stiffness in the orthonormal (xx,yy,sqrt(2)xy) representation."""
        return self.factor(points)[:, None, None] * STIFFNESS

    def derivatives(self, points: np.ndarray) -> tuple:
        """Return exact displacement, gradient and Hessian using analytic sine derivatives."""
        x, y = points.T
        frequencies = np.array([1, 2]) * np.pi
        sx, cx = np.sin(x[:, None] * frequencies), np.cos(x[:, None] * frequencies)
        sy, cy = np.sin(np.pi * y), np.cos(np.pi * y)
        value = sx * sy[:, None]
        gradient = np.stack((cx * frequencies * sy[:, None], sx * (np.pi * cy[:, None])), axis=-1)
        hessian = np.empty((len(points), 2, 2, 2))
        hessian[:, :, 0, 0] = -value * frequencies**2
        hessian[:, :, 1, 1] = -(np.pi**2) * value
        hessian[:, :, 0, 1] = hessian[:, :, 1, 0] = cx * frequencies * (np.pi * cy[:, None])
        return value, gradient, hessian

    def displacement(self, points: np.ndarray) -> np.ndarray:
        """Return the exact homogeneous-boundary displacement."""
        return self.derivatives(points)[0]

    def stress_and_divergence(self, points: np.ndarray) -> tuple:
        """Differentiate the constitutive law, including derivatives of the stiffness."""
        _, gradient, hessian = self.derivatives(points)
        strain = np.column_stack(
            (
                gradient[:, 0, 0],
                gradient[:, 1, 1],
                (gradient[:, 0, 1] + gradient[:, 1, 0]) / np.sqrt(2),
            )
        )
        strain_gradient = np.stack(
            (
                hessian[:, 0, 0],
                hessian[:, 1, 1],
                (hessian[:, 0, 1] + hessian[:, 1, 0]) / np.sqrt(2),
            ),
            axis=1,
        )
        kelvin = strain @ STIFFNESS.T
        derivatives = np.einsum("ab,nbj->naj", STIFFNESS, strain_gradient)
        tensor = np.empty((len(points), 2, 2))
        tensor[:, 0, 0], tensor[:, 1, 1] = kelvin[:, 0], kelvin[:, 1]
        tensor[:, 0, 1] = tensor[:, 1, 0] = kelvin[:, 2] / np.sqrt(2)
        divergence = np.column_stack(
            (
                derivatives[:, 0, 0] + derivatives[:, 2, 1] / np.sqrt(2),
                derivatives[:, 2, 0] / np.sqrt(2) + derivatives[:, 1, 1],
            )
        )
        factor = self.factor(points)
        if self.variable:
            divergence = factor[:, None] * divergence + tensor @ np.array([1.0, 2.0])
        return factor[:, None, None] * tensor, divergence

    def stress(self, points: np.ndarray) -> np.ndarray:
        """Return the exact symmetric Cauchy stress."""
        return self.stress_and_divergence(points)[0]

    def source(self, points: np.ndarray) -> np.ndarray:
        """Return minus the exact stress divergence."""
        return -self.stress_and_divergence(points)[1]


def archive(solution: PrimalElasticitySolution, data: TensorData, path: Path) -> str:
    """Archive independent one-sided triangular display samples and exact fields."""
    reference = TriangleMesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]]).submesh(0, 5)
    bary = np.column_stack((1 - reference.points.sum(axis=1), reference.points))
    basis = reference_basis(solution.degree, bary)[0]
    points, values, stress, cells = [], [], [], []
    offset = 0
    for cell, mesh in enumerate(solution.local_meshes):
        physical = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 2)
        dofs = nodal_space(mesh, solution.degree)[0]
        points.append(physical)
        values.append(np.einsum("qi,tia->tqa", basis, solution.values[cell][dofs]).reshape(-1, 2))
        stress.append(solution.stress(cell, bary).reshape(-1, 2, 2))
        cells.extend(reference.cells + offset + i * len(bary) for i in range(len(mesh.cells)))
        offset += len(physical)
    coordinates = np.concatenate(points)
    np.savez_compressed(
        path,
        points=coordinates,
        cells=np.concatenate(cells),
        displacement=np.concatenate(values),
        stress=np.concatenate(stress),
        exact_displacement=data.displacement(coordinates),
        exact_stress=data.stress(coordinates),
        macro_points=solution.skeleton.mesh.points,
        macro_cells=solution.skeleton.mesh.cells,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Run five macro refinements with either P1, minimally enriched P2 or full P3 locals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--constant", action="store_true")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    data = TensorData(not args.constant)
    minimal = args.degree == 2
    trace = 1
    refinement = 4 if args.degree == 1 else 1
    rows = []
    with threadpool_limits(1):
        for resolution in args.resolutions:
            mesh = TriangleMesh.unit_square(resolution)
            skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(trace) for _ in mesh.faces), 2)
            solution = solve_elasticity(
                mesh,
                formulation="primal",
                degree=args.degree,
                minimal_enrichment=minimal,
                local_refinement=refinement,
                skeleton=skeleton,
                constitutive=data.constitutive,
                source=data.source,
                dirichlet=data.displacement,
                quadrature_order=10,
            )
            indicator = estimate_primal_elasticity_error(
                solution,
                dirichlet=data.displacement,
                full_dirichlet=True,
                c_min=float(np.sqrt(np.linalg.eigvalsh(STIFFNESS)[0])),
                quadrature_order=12,
            )
            row = dict(
                resolution=resolution,
                macro_cells=len(mesh.cells),
                degree=args.degree,
                minimal_enrichment=minimal,
                local_refinement=refinement,
                trace_degree=trace,
                trace_dofs=skeleton.size,
                local_coordinates=len(solution.hybrid.fields[0]),
                displacement_l2=solution.l2_error(data.displacement, 12),
                stress_l2=solution.stress_l2_error(data.stress, 12),
                face_indicator=indicator.eta,
                algebraic_residual=solution.hybrid.residual,
            )
            print(json.dumps(row), flush=True)
            rows.append(row)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = f"{'constant' if args.constant else 'variable'}-p{args.degree}"
    path = OUTPUT / (name + ".npz")
    digest = archive(solution, data, path)
    sources = [
        Path(__file__),
        source_file("src/pymhm/_legacy/models/elasticity/primal.py", root=ROOT),
        source_file("src/pymhm/estimators/elasticity.py", root=ROOT),
    ]
    report = dict(
        case="Original trigonometric anisotropic elasticity",
        variable_material=data.variable,
        tensor_kelvin=STIFFNESS.tolist(),
        tensor_factor="1+x+2y" if data.variable else "1",
        source="minus div(C epsilon(u)); analytic material derivatives included",
        indicator_scope="L17 face residual only; finite local error is not certified",
        assembly_quadrature=10,
        error_quadrature=12,
        rows=rows,
        archive=path.name,
        sha256=digest,
        source_hashes=current_source_manifest(
            source_identity(ROOT, sources), packages=("pymhm", "examples")
        ),
    )
    (OUTPUT / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_primal_elasticity").main()
