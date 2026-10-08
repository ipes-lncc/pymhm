"""Three-dimensional Robin MH with physical mixed boundary conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.robin_3d import tetra_robin_boundary_operator as _robin_matrix
from pymhm.fem.scalar.tetrahedron import tetra_operators
from pymhm.fem.traces.pressure_3d import boundary_rules, broken_face_basis
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.materials.evaluation import scalar_values_3d, tensor_values_3d
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.methods.boundary import RobinBoundaryFace, solve_robin_neumann
from pymhm.postprocessing.solutions import MH3DSolution as MH3DSolution


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
