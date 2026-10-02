"""Central discontinuous Galerkin operators for TM and vector Maxwell MHM."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.linalg import solve_triangular

from pymhm.cut_cells import material_triangle_quadrature
from pymhm.darcy3d import TriangularSkeleton
from pymhm.elements import tensor_values, triangle_quadrature
from pymhm.helmholtz_forms import acoustic_quadrature, positive_values
from pymhm.lagrange import reference_basis
from pymhm.mesh import FloatArray, IntArray, SkeletonSpace, TriangleMesh
from pymhm.planar_material import PlanarMaterial
from pymhm.planar_quadrature import planar_simplex_quadrature
from pymhm.quadrilateral import CartesianMacroMesh, qk_basis
from pymhm.tetra_lagrange import tetra_polynomials
from pymhm.tetrahedral import tensor_values_3d, tetrahedron_quadrature


def scalar_basis(degree: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Use cardinal simplex functions without merging DG element nodes."""
    basis, gradient, _ = (
        reference_basis(degree, bary) if bary.shape[1] == 3 else tetra_polynomials(degree, bary)
    )
    return basis, gradient


def quadrature(mesh: Any, material: Any, order: int) -> tuple[FloatArray, FloatArray, Any]:
    """Return physical-weight simplex/rectangle rules split at material interfaces."""
    if isinstance(mesh, CartesianMacroMesh):
        _, points, weights, _, _, data = acoustic_quadrature(mesh, 1, material, order)
        origins = mesh.points[mesh.cells[:, 0], None, :]
        return (points - origins) / mesh.spacing, weights, data
    if isinstance(mesh, TriangleMesh):
        bary, weights, data = material_triangle_quadrature(mesh, material, order)
        measure = mesh.areas
    else:
        measure = mesh.volumes
        if isinstance(material, PlanarMaterial):
            bary, weights, data = planar_simplex_quadrature(mesh, material, order)
        else:
            bary, weights = tetrahedron_quadrature(order)
            bary = np.broadcast_to(bary, (len(mesh.cells), *bary.shape))
            weights = np.broadcast_to(weights, bary.shape[:2])
            data = material
    return bary, weights * measure[:, None], data


def physical_points(mesh: Any, reference: FloatArray) -> FloatArray:
    """Map common or elementwise reference coordinates to physical DG points."""
    if isinstance(mesh, CartesianMacroMesh):
        return mesh.points[mesh.cells[:, 0], None, :] + reference * mesh.spacing
    subscripts = "qi,tia->tqa" if reference.ndim == 2 else "tqi,tia->tqa"
    return np.einsum(subscripts, reference, mesh.points[mesh.cells])


def physical_basis(mesh: Any, degree: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Tabulate common or per-cell simplex/rectangle rules with physical gradients."""
    if bary.ndim == 2:
        bary = np.broadcast_to(bary, (len(mesh.cells), *bary.shape))
    if isinstance(mesh, CartesianMacroMesh):
        basis, gradient = qk_basis(degree, bary.reshape(-1, 2))
        basis = basis.reshape(*bary.shape[:2], -1)
        return basis, gradient.reshape(*basis.shape, 2) / mesh.spacing
    basis, gradient = scalar_basis(degree, bary.reshape(-1, bary.shape[-1]))
    basis = basis.reshape(*bary.shape[:2], -1)
    gradient = gradient.reshape(*basis.shape, bary.shape[-1])
    vertices = mesh.points[mesh.cells]
    inverse = np.linalg.inv(np.swapaxes(vertices[:, 1:] - vertices[:, :1], 1, 2))
    gradients = np.concatenate((-inverse.sum(axis=1)[:, None], inverse), axis=1)
    return basis, np.einsum("tqia,taj->tqij", gradient, gradients)


def cell_basis(mesh: Any, cell: int, degree: int, points: FloatArray) -> FloatArray:
    """Evaluate one incident DG polynomial at physical facet quadrature points."""
    vertices = mesh.points[mesh.cells[cell]]
    if isinstance(mesh, CartesianMacroMesh):
        return qk_basis(degree, (points - vertices[0]) / mesh.spacing)[0]
    coordinate = (points - vertices[0]) @ np.linalg.inv((vertices[1:] - vertices[0]).T).T
    return scalar_basis(degree, np.column_stack((1 - coordinate.sum(axis=1), coordinate)))[0]


def facet_rule(mesh: Any, face: int, order: int) -> tuple[FloatArray, FloatArray]:
    """Integrate a fine edge or triangular face with positive physical weights."""
    vertices = mesh.points[mesh.faces[face]]
    if mesh.points.shape[1] == 2:
        t, w = leggauss(order)
        points = vertices[0] + (t[:, None] + 1) * (vertices[1] - vertices[0]) / 2
        return points, w * mesh.lengths[face] / 2
    bary, weights = triangle_quadrature(order)
    return bary @ vertices, weights * mesh.areas[face]


def mass_blocks(mesh: Any, degree: int, material: Any, components: int, order: int) -> FloatArray:
    """Integrate SPD mass blocks with component-interleaved element-local nodes."""
    bary, weights, data = quadrature(mesh, material, order)
    basis, _ = physical_basis(mesh, degree, bary)
    points = physical_points(mesh, bary)
    points = points.reshape(-1, mesh.points.shape[1])
    if components == 1:
        values = positive_values(data, points)[:, None, None]
    else:
        evaluate = tensor_values if components == 2 else tensor_values_3d
        values = evaluate(data, points)
    values = values.reshape(*weights.shape, components, components)
    blocks = np.einsum("tq,tqi,tqj,tqab->tiajb", weights, basis, basis, values)
    return blocks.reshape(len(mesh.cells), basis.shape[-1] * components, -1)


def derivatives(mesh: Any, degree: int, order: int) -> tuple[Any, ...]:
    """Assemble central DG derivatives with incident traces at macro boundaries."""
    bary, weights, _ = quadrature(mesh, 1, order)
    basis, gradient = physical_basis(mesh, degree, bary)
    width = basis.shape[-1]
    matrices = [
        sparse.block_diag(
            [
                sparse.csr_matrix(block)
                for block in np.einsum("tq,tqi,tqj->tij", weights, basis, gradient[..., axis])
            ],
            format="lil",
        )
        for axis in range(mesh.points.shape[1])
    ]
    for face, (first, second) in enumerate(mesh.face_cells):
        if second < 0:
            continue
        points, face_weights = facet_rule(mesh, face, order)
        values = [cell_basis(mesh, int(cell), degree, points) for cell in (first, second)]
        ids = [int(cell) * width + np.arange(width) for cell in (first, second)]
        for row in range(2):
            for column in range(2):
                block = values[row].T @ (face_weights[:, None] * values[column])
                sign = (1 - 2 * row) * (0.5 if row != column else -0.5)
                for axis, matrix in enumerate(matrices):
                    matrix[np.ix_(ids[row], ids[column])] += sign * mesh.normals[face, axis] * block
    return tuple(matrix.tocsc() for matrix in matrices)


class MaxwellSkeleton:
    """Scalar TM or two-component tangential density on actual macrofaces.

    In 3D canonical faces use deterministic orthonormal tangent frames.
    Coefficients represent lambda=H cross n; the opposite incident cell
    receives the opposite sign. No normal multiplier is stored.
    """

    def __init__(self, base: SkeletonSpace | TriangularSkeleton) -> None:
        """Wrap scalar face spaces retaining their independent degree and partition."""
        if not isinstance(base, (SkeletonSpace, TriangularSkeleton)):
            raise TypeError("Maxwell skeleton requires scalar edge or triangular face spaces")
        if isinstance(base, SkeletonSpace) and base.components != 1:
            raise ValueError("TM face spaces must be scalar")
        self.base: Any = base
        self.mesh: Any = base.mesh
        self.components = 1 if isinstance(base, SkeletonSpace) else 2
        self.size = self.components * base.size
        self.frames: list[FloatArray] = []
        if self.components == 2:
            for normal in self.mesh.normals:
                axis = np.eye(3)[np.argmin(abs(normal))]
                first = np.cross(normal, axis)
                first /= np.linalg.norm(first)
                self.frames.append(np.column_stack((first, np.cross(normal, first))))

    def dofs(self, face: int) -> IntArray:
        """Return component-interleaved global tangent coefficients of one face."""
        return (
            self.base.dofs(face)[:, None] * self.components + np.arange(self.components)
        ).ravel()

    def cell_dofs(self, cell: int) -> IntArray:
        """Concatenate face blocks in actual macrocell incidence order."""
        return np.concatenate([self.dofs(int(face)) for face in self.mesh.cell_faces[cell]])


def trace_coupling(skeleton: MaxwellSkeleton, cell: int, fine: Any, degree: int, order: int) -> Any:
    """Integrate signed scalar/tangential moments against incident DG electric fields."""
    macro = skeleton.mesh
    components = 1 if skeleton.components == 1 else 3
    width = (
        (degree + 1) ** 2
        if isinstance(fine, CartesianMacroMesh)
        else scalar_basis(degree, np.full((1, fine.cells.shape[1]), 1 / fine.cells.shape[1]))[
            0
        ].shape[1]
    )
    matrix = np.zeros((len(fine.cells) * width * components, len(skeleton.cell_dofs(cell))))
    if skeleton.components == 2:
        vertices = macro.points[macro.cells[cell]]
        inverse = np.linalg.inv((vertices[1:] - vertices[0]).T)
        coord = (fine.points - vertices[0]) @ inverse.T
        macro_bary = np.column_stack((1 - coord.sum(axis=1), coord))
    offset = 0
    for side, face in enumerate(macro.cell_faces[cell]):
        face = int(face)
        for fine_face in fine.boundary_faces:
            nodes = fine.faces[fine_face]
            if skeleton.components == 1:
                start = macro.points[macro.faces[face, 0]]
                distance = (fine.points[nodes] - start) @ macro.normals[face]
                if np.max(abs(distance)) > 1e-11 * macro.lengths[face]:
                    continue
            else:
                excluded = np.flatnonzero(~np.isin(macro.cells[cell], macro.faces[face]))[0]
                if np.max(abs(macro_bary[nodes, excluded])) > 1e-11:
                    continue
            owner = int(fine.face_cells[fine_face, 0])
            if skeleton.components == 1:
                start, end = macro.points[macro.faces[face]]
                tangent = end - start
                position = (fine.points[nodes] - start) @ tangent / (tangent @ tangent)
                low, high = sorted(position)
                space = skeleton.base.faces[face]
                breaks = np.asarray(space.breaks)
                cuts = np.unique(np.r_[low, breaks[(breaks > low) & (breaks < high)], high])
                gauss, weight = leggauss(order)
                parameter = np.concatenate(
                    [
                        left + (gauss + 1) * (right - left) / 2
                        for left, right in zip(cuts[:-1], cuts[1:], strict=True)
                    ]
                )
                weights = np.concatenate(
                    [
                        weight * (right - left) / 2 * macro.lengths[face]
                        for left, right in zip(cuts[:-1], cuts[1:], strict=True)
                    ]
                )
                points = start + parameter[:, None] * tangent
                trace = space.evaluate(parameter)
                local_columns = np.arange(space.size)
                frame = np.ones((1, 1))
            else:
                positions = [
                    int(np.flatnonzero(macro.cells[cell] == node)[0]) for node in macro.faces[face]
                ]
                face_bary = macro_bary[nodes][:, positions]
                partitions = skeleton.base.face_partition(face)
                coordinates = np.einsum(
                    "i,sij->sj", face_bary.mean(axis=0), np.linalg.inv(partitions)
                )
                candidates = np.flatnonzero(np.min(coordinates, axis=1) >= -1e-11)
                if len(candidates) != 1:
                    raise ValueError("Maxwell fine boundary must align with skeleton subtriangles")
                segment = int(candidates[0])
                if np.min(face_bary @ np.linalg.inv(partitions[segment])) < -1e-11:
                    raise ValueError("Maxwell fine boundary must align with skeleton subtriangles")
                face_rule, weight = triangle_quadrature(order)
                points = face_rule @ fine.points[nodes]
                weights = weight * fine.areas[fine_face]
                trace = skeleton.base.basis(
                    face, (face_rule @ face_bary) @ np.linalg.inv(partitions[segment])
                )
                local_columns = segment * trace.shape[1] + np.arange(trace.shape[1])
                frame = skeleton.frames[face]
            values = cell_basis(fine, owner, degree, points)
            block = macro.signs[cell, side] * values.T @ (weights[:, None] * trace)
            rows = owner * width * components + np.arange(width * components)
            columns = (
                offset
                + (
                    local_columns[:, None] * skeleton.components + np.arange(skeleton.components)
                ).ravel()
            )
            matrix[np.ix_(rows, columns)] += np.kron(block, frame)
        offset += len(skeleton.dofs(face))
    return sparse.csc_matrix(matrix)


@dataclass(frozen=True)
class MaxwellLocal:
    """DG mass blocks and magnetic evolution C=-curl, paired with electric -C transpose."""

    mesh: Any
    degree: int
    electric_blocks: FloatArray
    magnetic_blocks: FloatArray
    curl: Any
    coupling: Any
    trace_dofs: IntArray

    @cached_property
    def electric_mass(self) -> Any:
        """Return the block-diagonal SPD electric mass matrix."""
        return sparse.block_diag([sparse.csc_matrix(b) for b in self.electric_blocks], format="csc")

    @cached_property
    def magnetic_mass(self) -> Any:
        """Return the block-diagonal SPD magnetic mass matrix."""
        return sparse.block_diag([sparse.csc_matrix(b) for b in self.magnetic_blocks], format="csc")

    def frequency_bound(self) -> float:
        """Bound the mass-scaled curl using positive weighted Gram row sums.

        For S=L_mu^-1 C L_epsilon^-T, any strictly positive x gives
        rho(S.T S) <= max_i (abs(S.T S) x)_i / x_i. A few positive
        power iterations tighten this Gershgorin bound; no Ritz estimate is
        accepted as an upper bound. The elementary 1/infinity-norm bound is
        retained if it is smaller. Roundoff padding covers sparse dot products.
        """
        factors = []
        for blocks in (self.magnetic_blocks, self.electric_blocks):
            lower = np.linalg.cholesky(blocks)
            inverse = [solve_triangular(item, np.eye(len(item)), lower=True) for item in lower]
            factors.append(sparse.block_diag([sparse.csc_matrix(b) for b in inverse], format="csc"))
        scaled = factors[0] @ self.curl @ factors[1].T
        absolute = abs(scaled)
        squared = float(np.max(absolute.sum(axis=0)) * np.max(absolute.sum(axis=1)))
        gamma = scaled.shape[0] * np.finfo(float).eps
        gram = abs((scaled.T @ scaled).tocsr()) + gamma / (1 - gamma) * (absolute.T @ absolute)
        vector = np.ones(gram.shape[0])
        for _ in range(16):
            image = gram @ vector
            squared = min(squared, float(np.max(image / vector)))
            # The retained positive part also covers disconnected zero blocks.
            vector = (vector + image / max(float(np.max(image)), 1.0)) / 2
        padding = 1 + 256 * np.finfo(float).eps * max(1, int(np.max(gram.getnnz(axis=1))))
        return float(np.sqrt(squared * padding))


def assemble_local(
    mesh: Any,
    skeleton: MaxwellSkeleton,
    cell: int,
    degree: int,
    permittivity: Any,
    permeability: Any,
    order: int,
) -> MaxwellLocal:
    """Assemble scalar TM in 2D or the full vector Maxwell operator in 3D."""
    dimension = mesh.points.shape[1]
    electric_components = 1 if dimension == 2 else 3
    electric = mass_blocks(mesh, degree, permittivity, electric_components, order)
    magnetic = mass_blocks(mesh, degree, permeability, dimension, order)
    derivative = derivatives(mesh, degree, order)
    if dimension == 2:
        curl = sparse.kron(-derivative[1], [[1], [0]]) + sparse.kron(derivative[0], [[0], [1]])
    else:
        curl = sparse.csc_matrix((len(mesh.cells) * electric.shape[1],) * 2)
        for axis, matrix in enumerate(derivative):
            cross = np.cross(np.eye(3)[axis], np.eye(3)).T
            curl -= sparse.kron(matrix, cross)
    return MaxwellLocal(
        mesh,
        degree,
        electric,
        magnetic,
        curl.tocsc(),
        trace_coupling(skeleton, cell, mesh, degree, order),
        skeleton.cell_dofs(cell),
    )
