"""Three-dimensional primal MHM Darcy with P1–P4 tetrahedral local spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm.elements import triangle_quadrature
from pymhm.hybrid import HybridSolution, HybridSystem, LocalAssembly, LocalProblem
from pymhm.mesh import FloatArray, IntArray, TriangleMesh, positive_int
from pymhm.tetra_validation import validate_face_partition, validate_tetra_submesh
from pymhm.tetrahedral import (
    TetraMesh,
    _dyadic,
    _real,
    scalar_values_3d,
    tensor_values_3d,
    tetra_face_basis,
    tetra_nodal_space,
    tetra_operators,
    tetra_tabulate,
    tetrahedron_quadrature,
)


class TriangularSkeleton:
    """Independent Pk flux densities on subtriangles of each macroface.

    ``subdivisions`` counts uniform edge divisions, giving its square many
    subtriangles by default. Explicit ``face_partitions`` instead prescribe a
    conforming nonuniform triangulation of each original macroface. Each
    subtriangle has (k+1)(k+2)/2 Bernstein coefficients. Sequences select
    independent face degrees. Flux uses the canonical normal of ``TetraMesh``.
    """

    def __init__(
        self,
        mesh: TetraMesh,
        subdivisions: Any = 1,
        *,
        degree: Any = 0,
        face_partitions: tuple[FloatArray, ...] | None = None,
    ) -> None:
        """Create uniform or explicit conforming triangular partitions with Pk modes.

        Explicit vertices use barycentric coordinates in ``mesh.faces[face]``
        ordering. Every original face must be covered without holes, overlaps or
        hanging internal edges. ``subdivisions`` and ``partitions`` retain the
        uniform-grid configuration; ``face_partition`` returns the active geometry.
        """
        raw = np.asarray(subdivisions)
        if raw.ndim == 0:
            raw = np.full(len(mesh.faces), raw)
        if raw.shape != (len(mesh.faces),):
            raise ValueError("subdivisions must be one integer or one per macroface")
        self.mesh = mesh
        self.subdivisions = np.array([_dyadic(value, "subdivisions") for value in raw])
        degrees = np.asarray(degree)
        if degrees.ndim == 0:
            degrees = np.full(len(mesh.faces), degrees)
        if degrees.shape != (len(mesh.faces),):
            raise ValueError("degree must be one integer or one per macroface")
        self.degrees = np.array([positive_int(value, "degree", 0) for value in degrees])
        self.modes = (self.degrees + 1) * (self.degrees + 2) // 2
        reference = TriangleMesh(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
        )
        self.partitions: dict[int, FloatArray] = {}
        for count in np.unique(self.subdivisions):
            fine = reference.submesh(0, int(count))
            bary = np.column_stack((1 - fine.points.sum(axis=1), fine.points))
            self.partitions[int(count)] = bary[fine.cells]
        if face_partitions is None:
            self._face_partitions = tuple(self.partitions[int(n)] for n in self.subdivisions)
            self._face_weights = tuple(np.full(n * n, 1.0 / (n * n)) for n in self.subdivisions)
        else:
            if len(face_partitions) != len(mesh.faces):
                raise ValueError("face_partitions must contain one partition per macroface")
            validated = tuple(validate_face_partition(part) for part in face_partitions)
            self._face_partitions = tuple(item[0] for item in validated)
            self._face_weights = tuple(item[1] for item in validated)
        counts = np.array([len(part) for part in self._face_partitions])
        self.offsets = np.r_[0, np.cumsum(counts * self.modes)]
        self.size = int(self.offsets[-1])
        self.constant_coefficients = np.ones(self.size)

    def face_partition(self, face: int) -> FloatArray:
        """Return active subtriangles in original macroface barycentric coordinates."""
        self.dofs(face)
        return self._face_partitions[face]

    def face_weights(self, face: int) -> FloatArray:
        """Return positive subtriangle area fractions summing to one on a macroface."""
        self.dofs(face)
        return self._face_weights[face]

    def basis(self, face: int, barycentric: FloatArray) -> FloatArray:
        """Evaluate the Bernstein Pk density basis on a canonical subtriangle.

        Degrees zero and one retain their constant and barycentric coordinate
        conventions. All modes integrate to the same fraction of subface area,
        and coefficients equal to one represent a constant density exactly.
        """
        self.dofs(face)
        bary = np.asarray(barycentric)
        if (
            np.iscomplexobj(bary)
            or bary.ndim != 2
            or bary.shape[1] != 3
            or not np.isfinite(bary).all()
        ):
            raise ValueError("subface barycentric points must be finite real triples")
        degree = int(self.degrees[face])
        from pymhm.element_backends import bernstein_tabulation

        exponents = tuple(
            (i, j, degree - i - j) for i in range(degree, -1, -1) for j in range(degree - i, -1, -1)
        )
        return bernstein_tabulation(bary[:, 1:], exponents)[0]

    def dofs(self, face: int) -> IntArray:
        """Return the global coefficients for one validated macroface index."""
        face = positive_int(face, "face", 0)
        if face >= len(self.mesh.faces):
            raise ValueError("face outside mesh")
        return np.arange(self.offsets[face], self.offsets[face + 1])

    def cell_dofs(self, cell: int) -> IntArray:
        """Concatenate the four oriented macroface coefficient blocks of a cell."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.mesh.cells):
            raise ValueError("cell outside mesh")
        return np.concatenate([self.dofs(int(face)) for face in self.mesh.cell_faces[cell]])


def tetra_trace_coupling(
    coarse: TetraMesh, cell: int, fine: TetraMesh, skeleton: TriangularSkeleton, degree: int
) -> FloatArray:
    """Integrate signed P1–P4 traces against aligned Pk triangular face modes."""
    dofs, points = tetra_nodal_space(fine, degree)
    matrix = np.zeros((len(points), len(skeleton.cell_dofs(cell))))
    vertices = coarse.points[coarse.cells[cell]]
    jacobian = (vertices[1:] - vertices[:1]).T
    coordinates = (fine.points - vertices[0]) @ np.linalg.inv(jacobian).T
    macro_bary = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
    order = max(3, degree + 1, (degree + int(skeleton.degrees.max()) + 3) // 2)
    bary_face, weights = triangle_quadrature(order)
    offset = 0
    for side, face in enumerate(coarse.cell_faces[cell]):
        partitions = skeleton.face_partition(int(face))
        macro_vertices = [
            int(np.flatnonzero(coarse.cells[cell] == node)[0]) for node in coarse.faces[face]
        ]
        for fine_face in fine.boundary_faces:
            ids = fine.faces[fine_face]
            if np.max(np.abs(macro_bary[ids, side])) > 1e-10:
                continue
            local = int(fine.face_cells[fine_face, 0])
            face_bary = macro_bary[ids][:, macro_vertices]
            coordinates = np.einsum("i,sij->sj", face_bary.mean(axis=0), np.linalg.inv(partitions))
            candidates = np.flatnonzero(np.min(coordinates, axis=1) >= -1e-10)
            if len(candidates) != 1:
                raise ValueError("fine boundary face is not contained in one skeleton subtriangle")
            segment = int(candidates[0])
            if np.min(face_bary @ np.linalg.inv(partitions[segment])) < -1e-10:
                raise ValueError("skeleton partition must align with the tetrahedral boundary mesh")
            bary = np.zeros((len(weights), 4))
            for j, node in enumerate(ids):
                bary[:, np.flatnonzero(fine.cells[local] == node)[0]] = bary_face[:, j]
            opposite = int(np.flatnonzero(fine.cell_faces[local] == fine_face)[0])
            values = tetra_face_basis(degree, bary, opposite_vertex=opposite)
            if skeleton.degrees[face] == 0:
                matrix[dofs[local], offset + segment] += (
                    coarse.signs[cell, side] * fine.areas[fine_face] * (weights @ values)
                )
            else:
                trace_basis = skeleton.basis(
                    int(face), (bary_face @ face_bary) @ np.linalg.inv(partitions[segment])
                )
                block = (
                    coarse.signs[cell, side]
                    * fine.areas[fine_face]
                    * np.einsum("q,qi,qj->ij", weights, values, trace_basis)
                )
                modes = int(skeleton.modes[face])
                matrix[np.ix_(dofs[local], offset + segment * modes + np.arange(modes))] += block
        offset += len(partitions) * int(skeleton.modes[face])
    return matrix


def _boundary(
    skeleton: TriangularSkeleton, dirichlet: Any, neumann: dict[int, Any], order: int
) -> tuple[FloatArray, dict[int, float]]:
    """Integrate weak pressure moments and project prescribed outward normal fluxes."""
    mesh = skeleton.mesh
    if any(face not in mesh.boundary_faces for face in neumann):
        raise ValueError("Neumann keys must identify external triangular faces")
    load = np.zeros(skeleton.size)
    fixed: dict[int, float] = {}
    bary, weights = triangle_quadrature(order)
    for face in mesh.boundary_faces:
        partitions = skeleton.face_partition(int(face))
        subvertices = partitions @ mesh.points[mesh.faces[face]]
        points = np.einsum("qi,sij->sqj", bary, subvertices)
        data = scalar_values_3d(neumann.get(int(face), dirichlet), points.reshape(-1, 3)).reshape(
            len(partitions), -1
        )
        if skeleton.degrees[face] == 0:
            moments = data @ weights
            if face in neumann:
                fixed.update(zip(skeleton.dofs(int(face)), moments, strict=True))
            else:
                load[skeleton.dofs(int(face))] = (
                    mesh.areas[face] * skeleton.face_weights(int(face)) * moments
                )
            continue
        basis = skeleton.basis(int(face), bary)
        moments = np.einsum("sq,q,qi->si", data, weights, basis)
        if face in neumann:
            mass = np.einsum("q,qi,qj->ij", weights, basis, basis)
            projection = np.linalg.solve(mass, moments.T).T
            fixed.update(zip(skeleton.dofs(int(face)), projection.ravel(), strict=True))
        else:
            load[skeleton.dofs(int(face))] = (
                mesh.areas[face] * skeleton.face_weights(int(face))[:, None] * moments
            ).ravel()
    return load, fixed


@dataclass(frozen=True)
class _DarcyFactory:
    """Picklable complete local Pk assembly for the serial/thread/spawn factory map."""

    mesh: TetraMesh
    skeleton: TriangularSkeleton
    refinement: int
    degree: int
    permeability: Any
    source: Any
    order: int
    local_meshes: tuple[TetraMesh, ...] | None = None

    def __call__(self, cell: int) -> LocalAssembly:
        """Build one local Neumann operator, physical mean and oriented trace form."""
        fine = (
            self.mesh.submesh(cell, self.refinement)
            if self.local_meshes is None
            else self.local_meshes[cell]
        )
        matrix, mass, load = tetra_operators(
            fine, self.degree, diffusion=self.permeability, source=self.source, order=self.order
        )
        coupling = tetra_trace_coupling(self.mesh, cell, fine, self.skeleton, self.degree)
        kernel = np.ones((len(load), 1))
        mean = np.asarray(mass @ kernel).ravel()
        problem = LocalProblem(
            matrix,
            coupling,
            load,
            self.skeleton.cell_dofs(cell),
            kernel=kernel,
            constraints=mean[:, None],
        )
        return LocalAssembly(problem, (fine, mean))


@dataclass(frozen=True)
class Darcy3DSolution:
    """Broken tetrahedral pressure and physical gradient flux with conservative trace.

    ``flux`` evaluates ``-K grad(p)`` and is generally not H(div)-conforming.
    Macro conservation refers to ``hybrid.trace``, not this raw gradient field.
    """

    skeleton: TriangularSkeleton
    local_meshes: tuple[TetraMesh, ...]
    pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    permeability: Any
    source: Any
    quadrature_order: int

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate pressure and physical flux at common fine-cell barycentric points."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.local_meshes):
            raise ValueError("cell outside local meshes")
        fine = self.local_meshes[cell]
        dofs, _, basis, gradient = tetra_tabulate(fine, self.degree, bary)
        pressure = self.pressure[cell][dofs] @ basis.T
        grad = np.einsum("ti,tqij->tqj", self.pressure[cell][dofs], gradient)
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        diffusion = tensor_values_3d(self.permeability, points.reshape(-1, 3)).reshape(
            *points.shape[:2], 3, 3
        )
        return pressure, -np.einsum("tqij,tqj->tqi", diffusion, grad)

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the broken pressure error with independently selected positive quadrature."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            values = self.evaluate(cell, bary)[0]
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            expected = scalar_values_3d(exact, points.reshape(-1, 3)).reshape(values.shape)
            total += float(fine.volumes @ ((values - expected) ** 2 @ weights))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the full physical flux, including P2 gradient and variable diffusion."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            values = self.evaluate(cell, bary)[1]
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            expected = _real(
                exact(points.reshape(-1, 3)) if callable(exact) else exact, "exact flux"
            )
            try:
                expected = np.broadcast_to(expected, (len(points.reshape(-1, 3)), 3)).reshape(
                    values.shape
                )
            except ValueError as exc:
                raise ValueError("exact flux must return three components per point") from exc
            total += float(fine.volumes @ (np.sum((values - expected) ** 2, axis=2) @ weights))
        return float(np.sqrt(total))

    def conservation_residuals(self, order: int | None = None) -> FloatArray:
        """Return outward skeletal flux minus integrated source in every macrocell."""
        mesh = self.skeleton.mesh
        bary, weights = tetrahedron_quadrature(self.quadrature_order if order is None else order)
        residuals = []
        for cell, fine in enumerate(self.local_meshes):
            total = sum(
                mesh.signs[cell, side]
                * mesh.areas[face]
                * (
                    self.skeleton.face_weights(int(face))
                    @ self.hybrid.trace[self.skeleton.dofs(int(face))]
                    .reshape(-1, self.skeleton.modes[face])
                    .mean(axis=1)
                )
                for side, face in enumerate(mesh.cell_faces[cell])
            )
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            load = scalar_values_3d(self.source, points.reshape(-1, 3)).reshape(len(fine.cells), -1)
            residuals.append(total - float(fine.volumes @ (load @ weights)))
        return np.asarray(residuals)


def solve_darcy_3d(
    mesh: TetraMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: TriangularSkeleton | None = None,
    local_refinement: int = 2,
    local_meshes: tuple[TetraMesh, ...] | None = None,
    degree: int = 2,
    quadrature_order: int = 5,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> Darcy3DSolution:
    """Solve div(q)=f, q=-K grad(p), using weak pressure and outward normal flux data.

    Local refinements and face subdivisions are powers of two with aligned
    partitions unless ``local_meshes`` supplies a conforming tetrahedral partition
    of each macrocell. Explicit meshes must cover each macrocell exactly, and each
    fine exterior triangle must lie in one active skeletal subtriangle. Every
    unspecified exterior face receives ``dirichlet`` pressure.
    Pure Neumann data impose one physical pressure mean and must be compatible.
    Local Pk spaces accept positive polynomial degrees; skeletal modes have an
    independently selected polynomial degree on each triangular subdivision.
    Nonrepresentable trace enrichment is rejected by
    the condensed rank check, without diagonal regularization.
    """
    refinement = _dyadic(local_refinement, "local_refinement")
    tetra_nodal_space(mesh, degree)
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    skeleton = TriangularSkeleton(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to the supplied tetrahedral mesh")
    if local_meshes is not None:
        if len(local_meshes) != len(mesh.cells):
            raise ValueError("local_meshes must contain one mesh per macrocell")
        for cell, fine in enumerate(local_meshes):
            validate_tetra_submesh(mesh, cell, fine)
    elif np.any(skeleton.subdivisions > refinement):
        raise ValueError("local refinement must resolve every skeleton subdivision")
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    neumann = {} if neumann is None else neumann
    boundary, fixed = _boundary(skeleton, dirichlet, neumann, order)
    factory = _DarcyFactory(
        mesh, skeleton, refinement, degree, permeability, source, order, local_meshes
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    constraints = []
    if len(neumann) == len(mesh.boundary_faces):
        constraints = [
            system.mean_constraint(
                [metadata[1] for metadata in system.local_metadata],
                float(mean_pressure * mesh.volumes.sum()),
            )
        ]
    solution = system.solve(fixed=fixed, constraints=constraints, solver=solver)
    return Darcy3DSolution(
        skeleton,
        tuple(metadata[0] for metadata in system.local_metadata),
        solution.fields,
        solution,
        degree,
        permeability,
        source,
        order,
    )
