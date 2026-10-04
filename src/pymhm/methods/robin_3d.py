"""Three-dimensional Robin MH with physical mixed boundary conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm._legacy.models.darcy.primal_3d import Darcy3DSolution
from pymhm.core.contracts import HybridSolution, LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space, tetra_operators
from pymhm.fem.traces.pressure_3d import boundary_rules, broken_face_basis
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.materials.evaluation import scalar_values_3d, tensor_values_3d
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.methods.boundary import RobinBoundaryFace, solve_robin_neumann


def _robin_matrix(
    mesh: TetraMesh,
    cell: int,
    fine: TetraMesh,
    degree: int,
    order: int,
    nu: float,
    origin: FloatArray,
) -> Any:
    """Integrate sigma.n uv over every fine boundary triangle with sigma=nu(x-a)/3."""
    size = len(tetra_nodal_space(fine, degree)[1])
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for face, dofs, basis, points, weights, _ in boundary_rules(mesh, cell, fine, degree, order):
        side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
        outward = mesh.signs[cell, side] * mesh.normals[face]
        coefficient = nu * ((points - origin) @ outward) / 3
        block = basis.T @ ((weights * coefficient)[:, None] * basis)
        rows.extend(np.repeat(dofs, len(dofs)))
        columns.extend(np.tile(dofs, len(dofs)))
        values.extend(block.ravel())
    return sparse.coo_matrix((values, (rows, columns)), shape=(size, size)).tocsc()


@dataclass(frozen=True)
class _Factory:
    """Picklable assembly of one coercive tetrahedral Robin local problem."""

    mesh: TetraMesh
    skeleton: TriangularSkeleton
    degree: int
    refinement: int
    order: int
    material: Any
    source: Any
    parameter: float
    origin: FloatArray
    lower: float

    def __call__(self, cell: int) -> LocalAssembly:
        """Return the local operator, Robin correction and physical volume moments."""
        fine = self.mesh.submesh(cell, self.refinement)
        sampled = np.linalg.eigvalsh(tensor_values_3d(self.material, fine.points)).min()
        if sampled < self.lower * (1 - 64 * np.finfo(float).eps):
            raise ValueError("ellipticity_lower_bound exceeds a sampled material eigenvalue")
        stiffness, mass, load = tetra_operators(
            fine, self.degree, diffusion=self.material, source=self.source, order=self.order
        )
        robin = _robin_matrix(
            self.mesh, cell, fine, self.degree, self.order, self.parameter, self.origin
        )
        coupling = tetra_trace_coupling(self.mesh, cell, fine, self.skeleton, self.degree)
        problem = LocalProblem(stiffness + robin, coupling, load, self.skeleton.cell_dofs(cell))
        return LocalAssembly(problem, (fine, robin, np.asarray(mass.sum(axis=1)).ravel()))


@dataclass(frozen=True)
class MH3DSolution:
    """Tetrahedral MH pressure, Robin multiplier and physical boundary diagnostics.

    The multiplier is (q-p sigma).n_global, with sigma=nu(x-origin)/3.
    Raw volume flux is q=-K grad(p); normal_flux_moments restores p sigma.n.
    The Neumann boundary extension is symmetric indefinite, even though the
    local Robin matrices and the Dirichlet Schur operator are positive definite.
    """

    skeleton: TriangularSkeleton
    local_meshes: tuple[TetraMesh, ...]
    pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    system: HybridSystem
    degree: int
    permeability: Any
    source: Any
    quadrature_order: int
    robin_parameter: float
    origin: FloatArray
    boundary_pressure: dict[int, FloatArray]
    global_matrix: Any
    global_rhs: FloatArray

    def _fields(self) -> Darcy3DSolution:
        """Reuse only physical volume evaluation, not the Darcy trace convention."""
        return Darcy3DSolution(
            self.skeleton,
            self.local_meshes,
            self.pressure,
            self.hybrid,
            self.degree,
            self.permeability,
            self.source,
            self.quadrature_order,
        )

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate the full pressure and physical flux at fine-cell barycentric points."""
        return self._fields().evaluate(cell, bary)

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the broken pressure error using an independent volume rule."""
        return self._fields().l2_error(exact, order)

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the raw physical flux error, not the Robin multiplier error."""
        return self._fields().flux_l2_error(exact, order)

    @property
    def global_coefficients(self) -> FloatArray:
        """Order Robin lambda before auxiliary pressures on sorted Neumann faces."""
        return np.concatenate((self.hybrid.trace, *self.boundary_pressure.values()))

    def normal_flux_moments(self, cell: int, face: int) -> FloatArray:
        """Integrate outward physical flux against all conormal modes on one side."""
        mesh = self.skeleton.mesh
        if face not in mesh.cell_faces[cell]:
            raise ValueError("face must be incident to the supplied macrocell")
        side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
        normal = mesh.signs[cell, side] * mesh.normals[face]
        result = np.zeros(len(self.skeleton.dofs(face)))
        for found, dofs, values, points, weights, bary in boundary_rules(
            mesh, cell, self.local_meshes[cell], self.degree, self.quadrature_order
        ):
            if found != face:
                continue
            basis = broken_face_basis(self.skeleton, face, bary)
            multiplier = mesh.signs[cell, side] * (
                basis @ self.hybrid.trace[self.skeleton.dofs(face)]
            )
            pressure = values @ self.pressure[cell][dofs]
            coefficient = self.robin_parameter * ((points - self.origin) @ normal) / 3
            result += basis.T @ (weights * (multiplier + pressure * coefficient))
        return result

    def conservation_residuals(self) -> FloatArray:
        """Return integrated outward physical flux minus source in every macrocell."""
        return np.array(
            [
                np.sum(
                    response.problem.coupling @ self.hybrid.trace[response.problem.trace_dofs]
                    + metadata[1] @ pressure
                    - response.problem.load
                )
                for response, pressure, metadata in zip(
                    self.system.responses, self.pressure, self.system.local_metadata, strict=True
                )
            ]
        )


def solve_mh_3d(
    mesh: TetraMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    skeleton: TriangularSkeleton | None = None,
    degree: int = 2,
    local_refinement: int = 2,
    quadrature_order: int = 5,
    robin_parameter: float | None = None,
    origin: Any = None,
    ellipticity_lower_bound: float | None = None,
    solver: str = "scipy",
    local_solver: str = "scipy",
    local_refinement_precision: Literal["double", "extended"] = "double",
    refinement_precision: Literal["double", "extended"] = "double",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> MH3DSolution:
    """Solve tetrahedral Robin MH using sigma=nu(x-origin)/3 and physical q.n data.

    This dimensional extension retains the 2024 bilinear form and uses div(sigma)=nu.
    With C=max_vertex|x-origin|/3, nu<=K_min/(4*C**2) gives the same sufficient
    Young-inequality coercivity bound in three dimensions. The default is half
    that bound. Material callbacks require a certified ellipticity lower bound.
    Local Pk and triangular skeletal modes are independently chosen; the local
    refinement must align their face partitions. Rank checks do not establish a
    uniform inf-sup constant for arbitrary degree/subdivision pairs.

    Neumann values are physical outward fluxes, never Robin multipliers. Pure
    Neumann data require integral compatibility and a prescribed volume mean.
    Other exterior faces prescribe pressure weakly. Assembly uses tetrahedral
    Gaussian integration; discontinuous coefficients require adequate local
    geometric resolution rather than an implicit enrichment of the trial space.
    """
    if not isinstance(mesh, TetraMesh):
        raise TypeError("MH3D requires a tetrahedral mesh")
    degree = positive_int(degree, "degree")
    refinement = _dyadic(local_refinement, "local_refinement")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    skeleton = TriangularSkeleton(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or np.any(skeleton.subdivisions > refinement):
        raise ValueError(
            "skeleton must use this mesh and subdivisions resolved by local refinement"
        )
    natural = {} if neumann is None else dict(neumann)
    if any(face not in mesh.boundary_faces for face in natural):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    mean = np.asarray(mean_pressure)
    if mean.shape != () or np.iscomplexobj(mean) or not np.isfinite(mean):
        raise ValueError("mean_pressure must be finite and real")
    pure = len(natural) == len(mesh.boundary_faces)
    if not pure and mean_pressure != 0:
        raise ValueError("mean_pressure applies only to pure Neumann problems")
    if ellipticity_lower_bound is None:
        if callable(permeability):
            raise ValueError("material callbacks require a certified ellipticity_lower_bound")
        lower = float(np.linalg.eigvalsh(tensor_values_3d(permeability, np.zeros((1, 3)))).min())
    else:
        value = np.asarray(ellipticity_lower_bound)
        if value.shape != () or np.iscomplexobj(value) or not np.isfinite(value) or value <= 0:
            raise ValueError("ellipticity_lower_bound must be a finite positive scalar")
        lower = float(value)
    point = np.min(mesh.points, axis=0) if origin is None else np.asarray(origin)
    if point.shape != (3,) or np.iscomplexobj(point) or not np.isfinite(point).all():
        raise ValueError("origin must be a finite real three-dimensional point")
    point = np.array(point, dtype=float, copy=True)
    radius = float(np.max(np.linalg.norm(mesh.points - point, axis=1))) / 3
    upper = lower / (4 * radius**2)
    parameter = upper / 2 if robin_parameter is None else robin_parameter
    if (
        np.iscomplexobj(parameter)
        or not np.isfinite(parameter)
        or parameter <= 0
        or parameter > upper
    ):
        raise ValueError("robin_parameter must satisfy the positive certified coercivity bound")
    factory = _Factory(
        mesh,
        skeleton,
        degree,
        refinement,
        order,
        permeability,
        source,
        float(parameter),
        point,
        lower,
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        local_solver=local_solver,
        local_refinement_precision=local_refinement_precision,
        backend=backend,
        workers=workers,
    )
    local_meshes = tuple(data[0] for data in system.local_metadata)
    boundary = np.zeros(skeleton.size)
    masses = {face: np.zeros((len(skeleton.dofs(face)),) * 2) for face in natural}
    loads = {face: np.zeros(len(skeleton.dofs(face))) for face in natural}
    for cell, fine in enumerate(local_meshes):
        for face, _, _, points, weights, bary in boundary_rules(mesh, cell, fine, degree, order):
            if face not in mesh.boundary_faces:
                continue
            basis = broken_face_basis(skeleton, face, bary)
            data = scalar_values_3d(natural.get(face, dirichlet), points)
            moments = basis.T @ (weights * data)
            if face in natural:
                masses[face] += basis.T @ (weights[:, None] * basis)
                loads[face] += moments
            else:
                boundary[skeleton.dofs(face)] += moments
    system.rhs[: skeleton.size] -= boundary
    system.load_scale[: skeleton.size] += abs(boundary)
    if natural:
        forms = tuple(
            RobinBoundaryFace(
                face,
                skeleton.dofs(face),
                masses[face],
                float(
                    parameter
                    * ((mesh.points[mesh.faces[face]].mean(axis=0) - point) @ mesh.normals[face])
                    / 3
                ),
                loads[face],
                np.ones(len(skeleton.dofs(face))),
            )
            for face in sorted(natural)
        )
        hybrid, pressures, matrix, rhs = solve_robin_neumann(
            system,
            skeleton.size,
            forms,
            tuple(data[2] for data in system.local_metadata),
            float(mesh.volumes.sum()),
            mean_pressure,
            pure,
            solver,
            refinement_precision,
        )
    else:
        hybrid = system.solve(solver=solver, refinement_precision=refinement_precision)
        pressures, matrix, rhs = {}, system.matrix, system.rhs
    return MH3DSolution(
        skeleton,
        local_meshes,
        hybrid.fields,
        hybrid,
        system,
        degree,
        permeability,
        source,
        order,
        float(parameter),
        point,
        pressures,
        matrix,
        rhs,
    )
