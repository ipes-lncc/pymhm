"""Native conservative RAD on star-shaped polyhedra with one polynomial per original face."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.transport.rad_3d import (
    RAD3DSolution,
    _coefficient_derivatives,
    tetra_rad_operators,
)
from pymhm.core.contracts import HybridSolution, LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.reference import monomial_tabulation
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_face_basis, tetra_nodal_space, tetrahedron_quadrature
from pymhm.materials.evaluation import scalar_values_3d, vector_values_3d
from pymhm.meshes.geometry import triangle_in_face
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic


class PolygonalSkeleton3D:
    """Independent P0/P1 Robin-flux densities on original polygonal macrofaces.

    A planar P1 trace has exactly three coefficients regardless of the number
    of face vertices or triangles. Its basis is ``(1, s, t)`` in orthonormal
    tangential coordinates centered at the area centroid and scaled by sqrt(area).
    Internal triangulation edges introduce no skeletal degrees of freedom.
    """

    def __init__(self, mesh: PolyhedralMesh, degree: Any = 1) -> None:
        """Assign one degree or one independently selected P0/P1 degree per face."""
        if not isinstance(mesh, PolyhedralMesh):
            raise TypeError("polygonal skeleton requires PolyhedralMesh")
        raw = np.asarray(degree)
        if raw.ndim == 0:
            raw = np.full(len(mesh.faces), raw)
        if raw.shape != (len(mesh.faces),):
            raise ValueError("degree must be scalar or one value per polygonal face")
        self.degrees = np.array([positive_int(value, "degree", 0) for value in raw])
        if np.any(self.degrees > 1):
            raise ValueError("polygonal skeleton supports degree 0 or 1")
        self.mesh = mesh
        self.offsets = np.r_[0, np.cumsum(1 + 2 * self.degrees)]
        self.size = int(self.offsets[-1])
        self.constant_coefficients = np.zeros(self.size)
        self.constant_coefficients[self.offsets[:-1]] = 1

    def dofs(self, face: int) -> IntArray:
        """Return the coefficients belonging to one original polygonal face."""
        face = positive_int(face, "face", 0)
        if face >= len(self.mesh.faces):
            raise ValueError("face outside polygonal skeleton")
        return np.arange(self.offsets[face], self.offsets[face + 1])

    def cell_dofs(self, cell: int) -> IntArray:
        """Concatenate the cell's complete original-face coefficient blocks."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.mesh.cells):
            raise ValueError("cell outside mesh")
        return np.concatenate([self.dofs(int(face)) for face in self.mesh.cell_faces[cell]])

    def basis(self, face: int, points: FloatArray) -> FloatArray:
        """Evaluate the single macroface polynomial in canonical physical coordinates."""
        self.dofs(face)
        points = np.asarray(points)
        if (
            np.iscomplexobj(points)
            or points.ndim != 2
            or points.shape[1] != 3
            or not np.isfinite(points).all()
        ):
            raise ValueError("face evaluation points must be finite real XYZ points")
        if not self.degrees[face]:
            return np.ones((len(points), 1))
        local = (points - self.mesh.face_origins[face]) @ self.mesh.face_tangents[face].T
        return monomial_tabulation(
            local / np.sqrt(self.mesh.areas[face]), ((0, 0), (1, 0), (0, 1)), nderiv=0
        )[0]


def polygonal_trace_coupling(
    mesh: PolyhedralMesh, cell: int, fine: TetraMesh, skeleton: PolygonalSkeleton3D, degree: int
) -> FloatArray:
    """Integrate oriented Pk/P0–P1 coupling through the conforming fine boundary triangles."""
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to the supplied polyhedral mesh")
    dofs, nodes = tetra_nodal_space(fine, degree)
    matrix = np.zeros((len(nodes), len(skeleton.cell_dofs(cell))))
    bary_face, weights = triangle_quadrature(max(3, degree + 1))
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        for fine_face in fine.boundary_faces:
            ids = fine.faces[fine_face]
            corners = fine.points[ids]
            if not triangle_in_face(
                corners, mesh.points[mesh.face_triangles[face]], mesh.normals[face]
            ):
                continue
            element = int(fine.face_cells[fine_face, 0])
            bary = np.zeros((len(weights), 4))
            for j, node in enumerate(ids):
                bary[:, np.flatnonzero(fine.cells[element] == node)[0]] = bary_face[:, j]
            physical = corners[0] + bary_face[:, 1:] @ (corners[1:] - corners[0])
            opposite = int(np.flatnonzero(fine.cell_faces[element] == fine_face)[0])
            values = tetra_face_basis(degree, bary, opposite_vertex=opposite)
            trace = skeleton.basis(int(face), physical)
            block = (
                mesh.signs[cell][side]
                * fine.areas[fine_face]
                * np.einsum("q,qi,qj->ij", weights, values, trace)
            )
            matrix[np.ix_(dofs[element], offset + np.arange(trace.shape[1]))] += block
        offset += len(skeleton.dofs(int(face)))
    return matrix


def _face_quadrature(mesh: PolyhedralMesh, face: int, order: int) -> tuple[FloatArray, FloatArray]:
    """Integrate the polygon by its canonical conforming triangulation with positive weights."""
    bary, weights = triangle_quadrature(order)
    corners = mesh.points[mesh.face_triangles[face]]
    area = (
        np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
        )
        / 2
    )
    return (
        (
            corners[:, :1] + np.einsum("qi,tia->tqa", bary[:, 1:], corners[:, 1:] - corners[:, :1])
        ).reshape(-1, 3),
        (area[:, None] * weights).ravel(),
    )


def _boundary(
    skeleton: PolygonalSkeleton3D, dirichlet: Any, natural: dict[int, Any], order: int
) -> tuple:
    """Form weak Dirichlet moments or L2-project outward Robin fluxes on original faces."""
    mesh = skeleton.mesh
    if any(face not in mesh.boundary_faces for face in natural):
        raise ValueError("Neumann keys must identify exterior polygonal faces")
    load = np.zeros(skeleton.size)
    fixed: dict[int, float] = {}
    for face in mesh.boundary_faces:
        points, weights = _face_quadrature(mesh, int(face), order)
        basis = skeleton.basis(int(face), points)
        data = scalar_values_3d(natural.get(int(face), dirichlet), points)
        moments = basis.T @ (weights * data)
        ids = skeleton.dofs(int(face))
        if face in natural:
            coefficients = np.linalg.solve(basis.T @ (weights[:, None] * basis), moments)
            fixed.update(zip(ids, coefficients, strict=True))
        else:
            load[ids] = moments
    return load, fixed


def _tangent(mesh: PolyhedralMesh, velocity: Any, order: int, faces: Any) -> bool:
    """Check velocity tangency at the same physical polygonal boundary quadrature points."""
    for face in faces:
        points, _ = _face_quadrature(mesh, int(face), max(order, 8))
        beta = vector_values_3d(velocity, points)
        normal = mesh.normals[face]
        scale = float(np.max(abs(beta) @ abs(normal)))
        if np.any(abs(beta @ normal) > 64 * np.finfo(float).eps * scale):
            return False
    return True


@dataclass(frozen=True)
class _PolyhedralFactory:
    """Picklable complete tetrahedral assembly inside one star-shaped polyhedral macrocell."""

    mesh: PolyhedralMesh
    skeleton: PolygonalSkeleton3D
    refinement: int
    degree: int
    coefficients: dict[str, Any]
    coarse_space: str

    def __call__(self, cell: int) -> LocalAssembly:
        """Retain constants through physical means without changing original macroface traces."""
        fine = self.mesh.submesh(cell, self.refinement)
        matrix, load, moments, pure = tetra_rad_operators(fine, self.degree, **self.coefficients)
        coupling = polygonal_trace_coupling(self.mesh, cell, fine, self.skeleton, self.degree)
        constant = np.ones((len(load), 1))
        retained = {"kernel": constant} if pure else {"coarse_basis": constant}
        constraints = moments[:, None]
        if self.coarse_space == "kernel" and not pure:
            bary, _ = tetrahedron_quadrature(self.coefficients["order"])
            points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(-1, 3)
            kernel = (
                not np.any(scalar_values_3d(self.coefficients["reaction"], points))
                and not np.any(scalar_values_3d(self.coefficients["velocity_divergence"], points))
                and _tangent(
                    self.mesh,
                    self.coefficients["velocity"],
                    self.coefficients["order"],
                    self.mesh.cell_faces[cell],
                )
            )
            retained = {"kernel": constant} if kernel else {}
            if not kernel:
                constraints = np.empty((len(load), 0))
        problem = LocalProblem(
            matrix,
            coupling,
            load,
            self.skeleton.cell_dofs(cell),
            constraints=constraints,
            **retained,
        )
        return LocalAssembly(problem, (fine, moments))


@dataclass(frozen=True)
class PolyhedralRADSolution:
    """Broken tetrahedral physical fields with their original polygonal skeleton."""

    field: RAD3DSolution
    skeleton: PolygonalSkeleton3D
    hybrid: HybridSolution

    @property
    def local_meshes(self) -> tuple[TetraMesh, ...]:
        """Return each independent conforming tetrahedral local discretization."""
        return self.field.local_meshes

    @property
    def values(self) -> tuple[FloatArray, ...]:
        """Return the unreconciled local scalar coefficients."""
        return self.field.values

    @property
    def degree(self) -> int:
        """Return the continuous local tetrahedral polynomial degree."""
        return self.field.degree

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate scalar and full physical flux -K grad(u)+beta u without averaging."""
        return self.field.evaluate(cell, bary)

    def l2_error(self, exact: Any, order: int = 7) -> float:
        """Integrate the scalar error through the common tetrahedral evaluator."""
        return self.field.l2_error(exact, order)

    def h1_seminorm_error(self, exact_gradient: Any, order: int = 7) -> float:
        """Integrate the complete broken physical gradient error."""
        return self.field.h1_seminorm_error(exact_gradient, order)

    def flux_l2_error(self, exact_flux: Any, order: int = 7) -> float:
        """Integrate the full conservative physical flux error."""
        return self.field.flux_l2_error(exact_flux, order)


def solve_polyhedral_rad(
    mesh: PolyhedralMesh,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: PolygonalSkeleton3D | None = None,
    local_refinement: int = 1,
    degree: int = 4,
    quadrature_order: int = 6,
    velocity_divergence: Any = None,
    diffusion_divergence: Any = None,
    stabilization: str = "galerkin",
    mean_value: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    coarse_space: Literal["constants", "kernel"] = "constants",
    global_refinement_precision: Literal["double", "extended"] = "double",
) -> PolyhedralRADSolution:
    """Solve conservative RAD with Pk local fields and original polygonal P0/P1 traces.

    Natural data prescribe outward ``(-K grad(u)+beta*u/2).n``; other exterior
    faces carry weak Dirichlet moments. Every macrocell is star-shaped and triangulated
    internally without changing its skeleton. The constant is retained by default
    for stability near singular local operators. ``coarse_space='kernel'`` retains
    only exact constant null modes. Pure Robin problems admit a mean only for zero
    reaction, divergence-free velocity tangent to the entire exterior.
    """
    if not isinstance(mesh, PolyhedralMesh):
        raise TypeError("polyhedral RAD requires PolyhedralMesh")
    if coarse_space not in ("constants", "kernel"):
        raise ValueError("coarse_space must be constants or kernel")
    refinement = _dyadic(local_refinement, "local_refinement")
    tetra_nodal_space(mesh.submesh(0), degree)
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    if not np.isfinite(mean_value):
        raise ValueError("mean_value must be finite")
    div_beta, div_tensor = _coefficient_derivatives(
        mesh.points, diffusion, velocity, velocity_divergence, diffusion_divergence, stabilization
    )
    skeleton = PolygonalSkeleton3D(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to the supplied polyhedral mesh")
    natural = {} if neumann is None else dict(neumann)
    boundary, fixed = _boundary(skeleton, dirichlet, natural, order)
    coefficients = dict(
        diffusion=diffusion,
        velocity=velocity,
        reaction=reaction,
        source=source,
        velocity_divergence=div_beta,
        diffusion_divergence=div_tensor,
        stabilization=stabilization,
        order=order,
    )
    factory = _PolyhedralFactory(mesh, skeleton, refinement, degree, coefficients, coarse_space)
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    constraints = []
    gauge = False
    if set(natural) == set(mesh.boundary_faces):
        bary, _ = tetrahedron_quadrature(order)
        pure = True
        for fine, _ in system.local_metadata:
            points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(-1, 3)
            pure = (
                pure
                and not np.any(scalar_values_3d(reaction, points))
                and not np.any(scalar_values_3d(div_beta, points))
            )
        gauge = pure and _tangent(mesh, velocity, order, mesh.boundary_faces)
    if gauge:
        constraints.append(
            system.mean_constraint(
                [data[1] for data in system.local_metadata], mean_value * float(mesh.volumes.sum())
            )
        )
    elif mean_value != 0:
        raise ValueError(
            "mean_value requires all-Robin zero-reaction divergence-free tangential data"
        )
    result = system.solve(
        solver=solver,
        fixed=fixed,
        constraints=constraints,
        refinement_precision=global_refinement_precision,
    )
    field = RAD3DSolution(
        tuple(data[0] for data in system.local_metadata),
        result.fields,
        degree,
        diffusion,
        velocity,
        result,
        residual=result.residual,
    )
    return PolyhedralRADSolution(field, skeleton, result)
