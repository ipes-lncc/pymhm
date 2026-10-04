"""Analytical lowest-order Darcy harmonic lifts on affine simplices.

For macrocell-constant SPD permeability, the RT0 normal flux has an explicit
quadratic potential. A variable source contributes its full harmonic moments
to the global equations and a separate zero-mean Neumann source lifting.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.contracts import HybridSolution, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.operators import boundary_data, triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_operators, tetra_tabulate, tetrahedron_quadrature
from pymhm.fem.scalar.triangle import scalar_operators, tabulate
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, _boundary
from pymhm.materials.evaluation import (
    scalar_values,
    scalar_values_3d,
    tensor_values,
    tensor_values_3d,
)
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


def _volumes(mesh: Any) -> FloatArray:
    """Read cell measures in the two supported simplex dimensions."""
    return mesh.areas if isinstance(mesh, TriangleMesh) else mesh.volumes


def _rule(dimension: int, order: int) -> tuple[FloatArray, FloatArray]:
    """Use positive, unit-measure simplex quadrature in the physical dimension."""
    return triangle_quadrature(order) if dimension == 2 else tetrahedron_quadrature(order)


@dataclass(frozen=True)
class AnalyticDarcySpace:
    """Constant, mean-zero affine and mean-zero metric-quadratic local basis."""

    center: FloatArray
    covariance: FloatArray
    permeability: FloatArray

    def evaluate(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate basis and gradients at physical points, preserving metric units."""
        points = np.asarray(points)
        d = len(self.center)
        if (
            np.iscomplexobj(points)
            or points.ndim != 2
            or points.shape[1] != d
            or not np.isfinite(points).all()
        ):
            raise ValueError("points must be finite physical coordinates of the simplex dimension")
        x = points - self.center
        inverse = np.linalg.inv(self.permeability)
        quadratic = (
            np.einsum("qi,ij,qj->q", x, inverse, x) - np.einsum("ij,ji", inverse, self.covariance)
        ) / 2
        basis = np.column_stack((np.ones(len(points)), x, quadratic))
        gradient = np.zeros((len(points), d + 2, d))
        gradient[:, 1 : d + 1] = np.eye(d)
        gradient[:, -1] = x @ inverse
        return basis, gradient

    def integrate_rt0(
        self, pressure_mean: float, centroid_flux: Any, divergence: float
    ) -> FloatArray:
        """Integrate an RT0 field into the unique potential with its prescribed mean.

        The field is ``q(c) + divergence/d * (x-c)``. The result contains the
        coefficients in this space, with ``-K grad(p)=q`` identically. A supplied
        classical RT0 pressure mean is preserved; this operation does not add
        the variable-source moment correction of the analytical MHM equations.
        """
        flux = np.asarray(centroid_flux)
        if (
            flux.shape != self.center.shape
            or np.iscomplexobj(flux)
            or not np.isfinite(flux).all()
            or not np.isfinite([pressure_mean, divergence]).all()
        ):
            raise ValueError(
                "RT0 integration requires a finite mean, divergence and centroid vector"
            )
        return np.r_[
            pressure_mean, -np.linalg.solve(self.permeability, flux), -divergence / len(self.center)
        ]


def analytic_darcy_local(
    mesh: TriangleMesh | TetraMesh,
    cell: int,
    permeability: Any = 1.0,
    source: Any = 0.0,
    *,
    quadrature_order: int = 8,
) -> tuple[LocalProblem, AnalyticDarcySpace]:
    """Build exact harmonic lift equations and all variable-source moments.

    Permeability is one constant SPD tensor on this macrocell. The local trial
    span is ``{1, x_i-center_i, (x-center).K^-1.(x-center)/2}``. Its gradients
    generate exactly the RT0 vector space after multiplication by K. Basis
    functions other than the constant have zero physical mean. The source
    response in this small polynomial space is used for the global load only;
    a full variable-source potential requires a separate local Neumann solve.
    """
    positive_int(cell, "cell", 0)
    if not isinstance(mesh, (TriangleMesh, TetraMesh)) or cell >= len(mesh.cells):
        raise ValueError("provide a valid cell of a triangular or tetrahedral mesh")
    order = positive_int(quadrature_order, "quadrature_order", 3)
    vertices = mesh.points[mesh.cells[cell]]
    d = vertices.shape[1]
    tensor = (tensor_values if d == 2 else tensor_values_3d)(permeability, vertices[:1])[0]
    if callable(permeability) or np.asarray(permeability).shape not in ((), (d, d)):
        raise ValueError("analytical harmonic lifts require a macrocell-constant SPD tensor")
    center = vertices.mean(axis=0)
    covariance = (vertices - center).T @ (vertices - center) / ((d + 1) * (d + 2))
    space = AnalyticDarcySpace(center, covariance, tensor)
    volume = _volumes(mesh)[cell]
    matrix = np.zeros((d + 2, d + 2))
    matrix[1 : d + 1, 1 : d + 1] = volume * tensor
    matrix[-1, -1] = volume * np.trace(np.linalg.solve(tensor, covariance))
    coupling = np.zeros((d + 2, d + 1))
    for side, face in enumerate(mesh.cell_faces[cell]):
        points = mesh.points[mesh.faces[face]]
        face_center = points.mean(axis=0)
        delta = face_center - center
        face_covariance = (points - face_center).T @ (points - face_center) / (d * (d + 1))
        means = np.r_[
            1.0,
            delta,
            0.5
            * np.trace(
                np.linalg.solve(tensor, face_covariance + np.outer(delta, delta) - covariance)
            ),
        ]
        measure = mesh.lengths[face] if isinstance(mesh, TriangleMesh) else mesh.areas[face]
        coupling[:, side] = mesh.signs[cell, side] * measure * means
    bary, weights = _rule(d, order)
    physical = bary @ vertices
    values = (scalar_values if d == 2 else scalar_values_3d)(source, physical)
    load = volume * space.evaluate(physical)[0].T @ (weights * values)
    kernel = np.eye(d + 2)[:, :1]
    problem = LocalProblem(matrix, coupling, load, mesh.cell_faces[cell], kernel, volume * kernel)
    return problem, space


@dataclass(frozen=True)
class AnalyticDarcySolution:
    """Analytical trace-driven pressure/RT0 flux plus a separate source potential."""

    mesh: TriangleMesh | TetraMesh
    spaces: tuple[AnalyticDarcySpace, ...]
    harmonic: tuple[FloatArray, ...]
    source_meshes: tuple[Any, ...]
    source_pressure: tuple[FloatArray | None, ...]
    source_degree: int
    hybrid: HybridSolution

    def pressure_update(self, cell: int, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return p0+p_lambda and its exact RT0 physical flux, without p_f."""
        positive_int(cell, "cell", 0)
        if cell >= len(self.spaces):
            raise ValueError("cell index outside mesh")
        basis, gradient = self.spaces[cell].evaluate(points)
        coefficients = self.harmonic[cell]
        return basis @ coefficients, -np.einsum(
            "ab,qib,i->qa", self.spaces[cell].permeability, gradient, coefficients
        )

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate the full pressure and flux on every source-mesh cell at barycentric points."""
        mesh = self.source_meshes[cell]
        points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
        d = points.shape[-1]
        p, q = self.pressure_update(cell, points.reshape(-1, d))
        p, q = p.reshape(points.shape[:2]), q.reshape(points.shape)
        coefficients = self.source_pressure[cell]
        if coefficients is not None:
            if d == 2:
                dofs, _, values, gradients, _ = tabulate(mesh, self.source_degree, bary)
            else:
                dofs, _, values, gradients = tetra_tabulate(mesh, self.source_degree, bary)
            local = coefficients[dofs]
            p += np.einsum("qi,ti->tq", values, local)
            q -= np.einsum("ab,tqib,ti->tqa", self.spaces[cell].permeability, gradients, local)
        return p, q

    def errors(self, pressure: Any, flux: Any, order: int = 8) -> tuple[float, float]:
        """Integrate pressure and physical flux errors on the actual source meshes."""
        bary, weights = _rule(self.mesh.points.shape[1], order)
        error = np.zeros(2)
        for cell, mesh in enumerate(self.source_meshes):
            points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
            p, q = self.evaluate(cell, bary)
            flat = points.reshape(-1, points.shape[-1])
            dp = p - pressure(flat).reshape(p.shape)
            dq = q - flux(flat).reshape(q.shape)
            error += [
                float(_volumes(mesh) @ (dp**2 @ weights)),
                float(_volumes(mesh) @ (np.sum(dq**2, axis=-1) @ weights)),
            ]
        return float(np.sqrt(error[0])), float(np.sqrt(error[1]))


def solve_darcy_analytic(
    mesh: TriangleMesh | TetraMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    source_degree: int = 2,
    source_refinement: int = 2,
    quadrature_order: int = 8,
    solver: str = "scipy",
) -> AnalyticDarcySolution:
    """Solve the lowest-order analytical MHM with optional variable-source reconstruction.

    Each face carries one constant normal-flux coefficient. K is one constant
    SPD tensor throughout the domain; different cell tensors can be supplied by
    assembling :func:`analytic_darcy_local` directly. Harmonic lifts are exact
    quadratics, independent of ``source_degree`` and ``source_refinement``.
    Constant sources have p_f=0. For callback sources, a separate continuous Pk
    zero-mean Neumann solve reconstructs p_f on each refined macrocell. The full
    source moments enter the global equations in both cases; replacing them by
    cell averages would change the method for a nonconstant source.
    """
    positive_int(source_degree, "source_degree", 1)
    positive_int(source_refinement, "source_refinement")
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    local = [
        analytic_darcy_local(mesh, i, permeability, source, quadrature_order=quadrature_order)
        for i in range(len(mesh.cells))
    ]
    natural = {} if neumann is None else neumann
    if isinstance(mesh, TriangleMesh):
        boundary, fixed = boundary_data(
            SkeletonSpace(mesh), dirichlet, natural, order=quadrature_order
        )
    else:
        boundary, fixed = _boundary(TriangularSkeleton(mesh), dirichlet, natural, quadrature_order)
    system = HybridSystem([pair[0] for pair in local], boundary_load=boundary)
    moments = [problem.constraints[:, 0] for problem, _ in local]
    gauges = (
        [system.mean_constraint(moments, mean_pressure * _volumes(mesh).sum())]
        if set(natural) == set(mesh.boundary_faces)
        else None
    )
    result = system.solve(solver=solver, fixed=fixed, constraints=gauges)
    harmonic = tuple(
        field - response.source
        for field, response in zip(result.fields, system.responses, strict=True)
    )
    source_meshes, sources = [], []
    for cell, (_, space) in enumerate(local):
        fine = mesh.submesh(cell, source_refinement if callable(source) else 1)
        values = None
        if callable(source):
            if isinstance(fine, TriangleMesh):
                matrix, mass, load = scalar_operators(
                    fine,
                    source_degree,
                    diffusion=space.permeability,
                    source=source,
                    order=quadrature_order,
                )
            else:
                matrix, mass, load = tetra_operators(
                    fine,
                    source_degree,
                    diffusion=space.permeability,
                    source=source,
                    order=quadrature_order,
                )
            kernel = np.ones((len(load), 1))
            values = (
                LocalProblem(
                    matrix,
                    np.empty((len(load), 0)),
                    load,
                    np.empty(0, dtype=int),
                    kernel,
                    mass @ kernel,
                )
                .condense()
                .source
            )
        source_meshes.append(fine)
        sources.append(values)
    return AnalyticDarcySolution(
        mesh,
        tuple(space for _, space in local),
        harmonic,
        tuple(source_meshes),
        tuple(sources),
        source_degree,
        result,
    )
