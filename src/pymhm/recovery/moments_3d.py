"""Three-dimensional canonical RT reconstruction of primal MHM Darcy flux.

Boundary moments are inherited from the physical multiplier, interior face
moments from the arithmetic one-sided raw flux average, and volume moments
from the raw flux. The resulting equilibrium is against continuous local Pm
tests; it does not impose separate fine-cell source balances.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.darcy.primal_3d import Darcy3DSolution
from pymhm.core.validation import FloatArray, positive_int
from pymhm.execution.cpu import map_local
from pymhm.fem.hdiv.family_3d import cell_quadrature, face_polynomials, face_quadrature, face_shape
from pymhm.fem.hdiv.rt_3d import RTTetraFamily, rt3d_interior_tests
from pymhm.fem.scalar.tetrahedron import tetra_basis, tetra_nodal_space
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.materials.evaluation import scalar_values_3d, tensor_values_3d, vector_values_3d
from pymhm.materials.planar import PlanarMaterial
from pymhm.meshes.mixed import AffineMixedMesh, hdiv3d_basis, hdiv3d_dofs, hdiv3d_face_offsets


def _boundary_moments(
    skeleton: TriangularSkeleton,
    cell: int,
    fine: AffineMixedMesh,
    family: RTTetraFamily,
    trace: FloatArray,
    order: int,
) -> FloatArray:
    """Integrate signed multiplier moments, requiring aligned fine boundary faces."""
    coarse = skeleton.mesh
    uv, weights = face_quadrature(3, order)
    tests = face_polynomials(uv, 3, family.normal_degree)
    count = tests.shape[1]
    result = np.zeros(len(fine.boundary_faces) * count)
    for row, face in enumerate(fine.boundary_faces):
        nodes = fine.points[fine.faces[face]]
        found = False
        for side, parent in enumerate(coarse.cell_faces[cell]):
            vertices = coarse.points[coarse.faces[parent]]
            scale = np.max(np.linalg.norm(vertices[1:] - vertices[0], axis=1))
            if np.max(abs((nodes - vertices[0]) @ coarse.normals[parent])) > 1e-12 * scale:
                continue
            xy = (nodes - vertices[0]) @ np.linalg.pinv((vertices[1:] - vertices[0]).T).T
            bary = np.column_stack((1 - xy.sum(axis=1), xy))
            partitions = skeleton.face_partition(int(parent))
            containing = [
                i for i, sub in enumerate(partitions) if np.min(bary @ np.linalg.inv(sub)) >= -1e-12
            ]
            if len(containing) != 1:
                raise ValueError("RT reconstruction requires aligned skeleton subfaces")
            segment = containing[0]
            coordinates = face_shape(uv, 3) @ bary @ np.linalg.inv(partitions[segment])
            modes = int(skeleton.modes[parent])
            # The skeleton owns its density basis and partition convention.
            basis = skeleton.basis(int(parent), coordinates)
            coefficients = trace[skeleton.dofs(int(parent))].reshape(-1, modes)[segment]
            value = coarse.signs[cell, side] * (basis @ coefficients)
            result[row * count : (row + 1) * count] = (
                fine.measures[face] * tests.T @ (weights * value)
            )
            found = True
            break
        if not found:
            raise ValueError("a fine boundary face has no containing macroface")
    return result


@dataclass(frozen=True)
class _Factory:
    """Compute one macrocell's independent canonical moments inside its worker."""

    solution: Darcy3DSolution
    family: RTTetraFamily
    order: int

    def __call__(self, cell: int) -> tuple[AffineMixedMesh, FloatArray]:
        """Return the conforming local topology and RT moment vector."""
        solution, family = self.solution, self.family
        primal = solution.local_meshes[cell]
        mesh = AffineMixedMesh(primal.points, primal.cells)
        dofs, _ = tetra_nodal_space(primal, solution.degree)
        coefficients = solution.pressure[cell][dofs]

        def raw(points: FloatArray, owners: np.ndarray) -> FloatArray:
            """Evaluate the physically owned raw gradient without interface averaging."""
            xi = np.einsum(
                "qij,qj->qi", mesh.inverse[owners], points - mesh.points[mesh.cells[owners, 0]]
            )
            bary = np.column_stack((1 - xi.sum(axis=1), xi))
            derivative = tetra_basis(solution.degree, bary)[1]
            reference = np.einsum("qi,qij->qj", coefficients[owners], derivative)
            gradient = np.einsum(
                "qi,qij->qj", reference[:, 1:] - reference[:, :1], mesh.inverse[owners]
            )
            material = solution.permeability
            if isinstance(material, PlanarMaterial):
                material = material.trace_values(
                    points, mesh.points[mesh.cells[owners]].mean(axis=1)
                )
            return -np.einsum("qij,qj->qi", tensor_values_3d(material, points), gradient)

        offsets = hdiv3d_face_offsets(mesh, family)
        field = np.zeros(int(hdiv3d_dofs(mesh, family).max()) + 1)
        boundary = np.concatenate(
            [np.arange(offsets[f], offsets[f + 1]) for f in mesh.boundary_faces]
        )
        field[boundary] = _boundary_moments(
            solution.skeleton, cell, mesh, family, solution.hybrid.trace, self.order
        )
        uv, weights = face_quadrature(3, self.order)
        tests = face_polynomials(uv, 3, family.normal_degree)
        for face, owners in enumerate(mesh.incidence):
            if len(owners) != 2:
                continue
            points = face_shape(uv, 3) @ mesh.points[mesh.faces[face]]
            average = (
                sum(raw(points, np.full(len(points), owner, dtype=int)) for owner, _ in owners) / 2
            )
            field[offsets[face] : offsets[face + 1]] = (
                mesh.measures[face] * tests.T @ (weights * (average @ mesh.normals[face]))
            )
        if family.interior_size:
            points, weights = cell_quadrature("tetrahedron", self.order)
            physical = mesh.geometry(points)
            values = raw(
                physical.reshape(-1, 3), np.repeat(np.arange(len(mesh.cells)), len(points))
            ).reshape(physical.shape)
            pulled = np.einsum("tba,tqa,t->tqb", mesh.inverse, values, mesh.determinants)
            tests = rt3d_interior_tests(points, family.pressure_degree)
            field[offsets[-1] :] = np.einsum("q,qia,tqa->ti", weights, tests, pulled).ravel()
        return mesh, field


@dataclass(frozen=True)
class MomentFlux3DSolution:
    """Canonical tetrahedral RT flux with continuous-test equilibrium diagnostics."""

    solution: Darcy3DSolution
    family: RTTetraFamily
    local_meshes: tuple[AffineMixedMesh, ...]
    flux: tuple[FloatArray, ...]
    quadrature_order: int

    def evaluate(self, cell: int, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return physical flux and divergence at common reference Cartesian points."""
        mesh = self.local_meshes[cell]
        basis, div, _ = hdiv3d_basis(mesh, self.family, points)
        coefficients = self.flux[cell][hdiv3d_dofs(mesh, self.family)]
        return np.einsum("tqia,ti->tqa", basis, coefficients), np.einsum(
            "tqi,ti->tq", div, coefficients
        )

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the physical vector error with independent positive volume quadrature."""
        points, weights = cell_quadrature("tetrahedron", order)
        total = 0.0
        for cell, mesh in enumerate(self.local_meshes):
            values = self.evaluate(cell, points)[0]
            expected = vector_values_3d(exact, mesh.geometry(points).reshape(-1, 3)).reshape(
                values.shape
            )
            total += np.einsum(
                "t,q,tq->", mesh.determinants, weights, np.sum((values - expected) ** 2, axis=2)
            )
        return float(np.sqrt(total))

    def normal_flux_residuals(self) -> tuple[FloatArray, ...]:
        """Compare every boundary RT moment with its original physical skeletal density."""
        result = []
        for cell, mesh in enumerate(self.local_meshes):
            offsets = hdiv3d_face_offsets(mesh, self.family)
            ids = np.concatenate(
                [np.arange(offsets[f], offsets[f + 1]) for f in mesh.boundary_faces]
            )
            expected = _boundary_moments(
                self.solution.skeleton,
                cell,
                mesh,
                self.family,
                self.solution.hybrid.trace,
                self.quadrature_order,
            )
            result.append(self.flux[cell][ids] - expected)
        return tuple(result)

    def continuous_moment_residuals(self) -> tuple[FloatArray, ...]:
        """Return moments of div(q)-f against continuous Pm, or one macroconstant at m=0."""
        points, weights = cell_quadrature("tetrahedron", self.quadrature_order)
        bary = np.column_stack((1 - points.sum(axis=1), points))
        result = []
        degree = self.family.pressure_degree
        for cell, mesh in enumerate(self.local_meshes):
            if degree:
                dofs, nodes = tetra_nodal_space(self.solution.local_meshes[cell], degree)
                basis = tetra_basis(degree, bary)[0]
                size = len(nodes)
            else:
                dofs, basis, size = (
                    np.zeros((len(mesh.cells), 1), dtype=int),
                    np.ones((len(points), 1)),
                    1,
                )
            divergence = self.evaluate(cell, points)[1]
            source = scalar_values_3d(
                self.solution.source, mesh.geometry(points).reshape(-1, 3)
            ).reshape(divergence.shape)
            local = np.einsum(
                "t,q,qi,tq->ti", mesh.determinants, weights, basis, divergence - source
            )
            result.append(np.bincount(dofs.ravel(), weights=local.ravel(), minlength=size))
        return tuple(result)

    def fine_conservation_residuals(self) -> tuple[FloatArray, ...]:
        """Measure fine-cell integral defects, which canonical reconstruction does not impose."""
        points, weights = cell_quadrature("tetrahedron", self.quadrature_order)
        result = []
        for cell, mesh in enumerate(self.local_meshes):
            divergence = self.evaluate(cell, points)[1]
            source = scalar_values_3d(
                self.solution.source, mesh.geometry(points).reshape(-1, 3)
            ).reshape(divergence.shape)
            result.append(mesh.determinants * ((divergence - source) @ weights))
        return tuple(result)


def reconstruct_darcy_moments_3d(
    solution: Darcy3DSolution,
    *,
    degree: int | None = None,
    quadrature_order: int | None = None,
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> MomentFlux3DSolution:
    """Construct canonical RT_m face/volume moments of a primal tetrahedral MHM field.

    The scalar local test space must contain continuous Pm and the multiplier
    must belong to its RT normal space. Coefficient quadrature must resolve
    material interfaces; pointwise callbacks do not supply geometric cuts.
    """
    m = solution.degree if degree is None else positive_int(degree, "RT degree", 0)
    if m > solution.degree or np.any(solution.skeleton.degrees > m):
        raise ValueError("RT degree must contain the trace and fit the primal local test space")
    order = (
        max(solution.quadrature_order, m + 3)
        if quadrature_order is None
        else positive_int(quadrature_order, "quadrature order")
    )
    if order < m + 2:
        raise ValueError("quadrature must integrate the canonical RT polynomial moments")
    family = RTTetraFamily(m)
    data = map_local(
        _Factory(solution, family, order),
        range(len(solution.local_meshes)),
        backend=backend,
        workers=workers,
    )
    return MomentFlux3DSolution(
        solution, family, tuple(row[0] for row in data), tuple(row[1] for row in data), order
    )
