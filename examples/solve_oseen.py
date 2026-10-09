"""Acquire analytical Oseen convergence and fixed-macro adaptive experiments.

The data in L15 sections 5.1--5.3 are differentiated explicitly. The declared
crisscross mesh, quadrature, marking threshold and stopping limits make this
an independently reproducible experiment, not an assertion of identical
historical adaptive meshes or table values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymhm.estimators.flow import FlowEstimator
    from pymhm.postprocessing.solutions import VectorSolution

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial import Polynomial
from threadpoolctl import threadpool_limits

from examples.field_sampling import sample_field
from examples.formulations.application import flow as solve_flow
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.adaptivity.flow import adapt_flow
from pymhm.estimators.flow import estimate_flow_error
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file, source_identity

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/oseen"


@dataclass(frozen=True)
class OseenData:
    """Analytical velocity, pressure and independently differentiated force."""

    case: str = "smooth"
    viscosity: float = 1.0

    @property
    def drag(self) -> float:
        """Return the constant reaction from the selected manufactured problem."""
        return 0.0 if self.case == "internal" else (2.0 if self.case == "variable" else 1.0)

    def advection(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the published constant advection or the supplementary affine field."""
        if self.case == "variable":
            return points.copy()
        vector = (1.0, 0.0) if self.case == "internal" else (1 / np.sqrt(2), 1 / np.sqrt(2))
        return np.tile(vector, (len(points), 1))

    def fields(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate velocity, its Jacobian and Laplacian by analytic differentiation."""
        x, y = points.T
        if self.case == "boundary":
            nu = self.viscosity
            denominator = -np.expm1(-1 / nu)
            exponential = np.exp((points - 1) / nu)
            ratio = (exponential - np.exp(-1 / nu)) / denominator
            u = (points - ratio)[:, ::-1]
            gradient = np.zeros((len(points), 2, 2))
            gradient[:, 0, 1] = 1 - exponential[:, 1] / (nu * denominator)
            gradient[:, 1, 0] = 1 - exponential[:, 0] / (nu * denominator)
            laplacian = -exponential[:, ::-1] / (nu**2 * denominator)
            return u, gradient, laplacian
        polynomial = Polynomial([0, 0, 1, -2, 1])
        fx = [polynomial.deriv(i)(x) for i in range(4)]
        fy = [polynomial.deriv(i)(y) for i in range(4)]
        scale = -128.0
        if self.case == "internal":
            tanh = np.tanh(75 - 150 * x)
            sech2 = 1 - tanh**2
            derivatives = [
                1 - tanh,
                150 * sech2,
                45000 * tanh * sech2,
                6750000 * sech2 * (3 * tanh**2 - 1),
            ]
            fx = [
                fx[0] * derivatives[0],
                fx[1] * derivatives[0] + fx[0] * derivatives[1],
                fx[2] * derivatives[0] + 2 * fx[1] * derivatives[1] + fx[0] * derivatives[2],
                fx[3] * derivatives[0]
                + 3 * fx[2] * derivatives[1]
                + 3 * fx[1] * derivatives[2]
                + fx[0] * derivatives[3],
            ]
            scale = 1.0
        u = scale * np.column_stack((fx[0] * fy[1], -fx[1] * fy[0]))
        gradient = scale * np.array(
            [[fx[1] * fy[1], fx[0] * fy[2]], [-fx[2] * fy[0], -fx[1] * fy[1]]]
        ).transpose(2, 0, 1)
        laplacian = scale * np.column_stack(
            (fx[2] * fy[1] + fx[0] * fy[3], -fx[3] * fy[0] - fx[1] * fy[2])
        )
        return u, gradient, laplacian

    def velocity(self, points: np.ndarray) -> np.ndarray:
        """Return the analytical divergence-free velocity."""
        return self.fields(points)[0]

    def gradient(self, points: np.ndarray) -> np.ndarray:
        """Return the velocity Jacobian, indexed component then derivative."""
        return self.fields(points)[1]

    def pressure(self, points: np.ndarray) -> np.ndarray:
        """Return the polynomial pressure with exact zero domain mean."""
        power = 8 if self.case == "boundary" else 6
        return (points[:, 0] - points[:, 1]) ** power - 2 / ((power + 1) * (power + 2))

    def source(self, points: np.ndarray) -> np.ndarray:
        """Evaluate -nu Delta u + grad(u) beta + gamma u + grad p."""
        value, gradient, laplacian = self.fields(points)
        power = 8 if self.case == "boundary" else 6
        gradp = (
            power * (points[:, 0] - points[:, 1])[:, None] ** (power - 1) * np.array([1.0, -1.0])
        )
        return (
            -self.viscosity * laplacian
            + np.einsum("nij,nj->ni", gradient, self.advection(points))
            + self.drag * value
            + gradp
        )

    def options(self) -> dict:
        """Return picklable solver inputs with explicit coefficient contracts."""
        variable = self.case == "variable"
        return dict(
            viscosity=self.viscosity,
            drag=self.drag,
            source=self.source,
            dirichlet=self.velocity,
            advection=self.advection if variable else self.advection(np.zeros((1, 2)))[0],
            advection_divergence=2.0 if variable else None,
            advection_bound=np.sqrt(2) if variable else None,
        )


def crisscross(resolution: int) -> TriangleMesh:
    """Create four triangles per unit-square grid cell, with exact H=1/resolution."""
    n = resolution
    points = [[i / n, j / n] for j in range(n + 1) for i in range(n + 1)]
    cells = []
    for j in range(n):
        for i in range(n):
            first = j * (n + 1) + i
            corners = [first, first + 1, first + n + 2, first + n + 1]
            center = len(points)
            points.append([(i + 0.5) / n, (j + 0.5) / n])
            cells.extend([corners[k], corners[(k + 1) % 4], center] for k in range(4))
    return TriangleMesh(points, cells)


def record(solution: VectorSolution, estimator: FlowEstimator, data: OseenData, order: int) -> dict:
    """Record physical norms and estimator components on one solved state."""
    mixed = estimator.mixed_error(data.velocity, data.gradient, data.pressure, order=order)
    return dict(
        macro_cells=len(solution.skeleton.mesh.cells),
        fine_cells=sum(len(mesh.cells) for mesh in solution.local_meshes),
        trace_dofs=solution.skeleton.size,
        velocity_l2=solution.l2_error(data.velocity, order),
        pressure_l2=solution.pressure_l2_error(data.pressure, order),
        mixed_error=mixed,
        eta1=estimator.eta1,
        eta2=estimator.eta2,
        effectivity=estimator.total / mixed,
        divergence_l2=solution.divergence_l2(),
        algebraic_residual=solution.hybrid.residual,
    )


def archive(solution: VectorSolution, data: OseenData, path: Path) -> str:
    """Archive discontinuous display samples without gluing macro interfaces."""
    velocity = sample_field(solution.local_meshes, solution.values, solution.degree, 4)
    pressure = sample_field(solution.local_meshes, solution.pressure, solution.pressure_degree, 4)
    np.savez_compressed(
        path,
        points=velocity["points"],
        cells=velocity["cells"],
        velocity=velocity["values"],
        exact_velocity=data.velocity(velocity["points"]),
        pressure=pressure["values"],
        exact_pressure=data.pressure(pressure["points"]),
        macro_points=solution.skeleton.mesh.points,
        macro_cells=solution.skeleton.mesh.cells,
        divergence=np.trace(velocity["gradient"], axis1=-2, axis2=-1),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Run a selected study, keeping potentially expensive campaigns outside CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["smooth", "boundary", "internal", "variable"], default="smooth"
    )
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--trace-degree", type=int, default=1)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--order", type=int, default=10)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = OseenData(args.case, args.viscosity)
    name = f"{args.case}-nu{args.viscosity:g}-l{args.trace_degree}" + (
        "-adaptive" if args.adaptive else "-uniform"
    )
    rows = []
    with threadpool_limits(1):
        if args.adaptive:
            mesh = crisscross(args.levels[0])
            skeleton = SkeletonSpace(
                mesh, tuple(FaceSpace.uniform(args.trace_degree) for _ in mesh.faces), 2
            )
            result = adapt_flow(
                mesh,
                solve_step=solve_flow,
                skeleton=skeleton,
                iterations=args.iterations,
                theta=0.5,
                degree=3,
                quadrature_order=args.order,
                estimator_order=args.order,
                max_local_refinement=16,
                **data.options(),
            )
            for solution, estimator in zip(result.solutions, result.estimators, strict=True):
                rows.append(record(solution, estimator, data, args.order))
                print(json.dumps(rows[-1]), flush=True)
            stop = result.stop_reason
        else:
            for n in args.levels:
                mesh = crisscross(n)
                skeleton = SkeletonSpace(
                    mesh, tuple(FaceSpace.uniform(args.trace_degree) for _ in mesh.faces), 2
                )
                solution = solve_flow(
                    mesh,
                    formulation="oseen",
                    degree=3,
                    local_refinement=1,
                    skeleton=skeleton,
                    quadrature_order=args.order,
                    **data.options(),
                )
                estimator = estimate_flow_error(
                    solution,
                    full_dirichlet=True,
                    viscosity=data.viscosity,
                    drag=data.drag,
                    advection=data.advection,
                    source=data.source,
                    dirichlet=data.velocity,
                    quadrature_order=args.order,
                )
                rows.append(
                    dict(resolution=n, H=1 / n, **record(solution, estimator, data, args.order))
                )
                print(json.dumps(rows[-1]), flush=True)
            stop = "levels"
        filename = name + ".npz"
        checksum = archive(solution, data, OUTPUT / filename)
    sources = [
        Path(__file__),
        source_file("src/pymhm/_legacy/models/flow/solver.py", root=ROOT),
        source_file("src/pymhm/estimators/flow.py", root=ROOT),
        source_file("src/pymhm/adaptivity/flow.py", root=ROOT),
    ]
    report = dict(
        case=args.case,
        viscosity=data.viscosity,
        drag=data.drag,
        local_pair="P3/P3",
        trace_degree=args.trace_degree,
        mesh="unit-square structured crisscross; four triangles per square",
        quadrature_order=args.order,
        marking_theta=0.5 if args.adaptive else None,
        stop_reason=stop,
        literature=(
            "L15 sections 5.1–5.3; variable beta is supplementary; historical "
            "marking threshold/meshes not identified"
        ),
        reference_doi="10.1007/s10444-020-09833-8",
        rows=rows,
        archive=filename,
        sha256=checksum,
        versions=dict(
            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__
        ),
        source_hashes=current_source_manifest(
            source_identity(ROOT, sources), packages=("pymhm", "examples")
        ),
    )
    (OUTPUT / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_oseen").main()
