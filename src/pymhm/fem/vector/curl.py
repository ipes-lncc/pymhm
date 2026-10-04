"""DG mass, curl and oriented scalar/tangential trace operators."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.linalg import solve_triangular

from pymhm.core.validation import FloatArray, IntArray
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.quadrature.planar import planar_simplex_quadrature
from pymhm.fem.scalar.helmholtz import acoustic_quadrature, positive_values
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.quadrilateral import qk_basis
from pymhm.fem.scalar.tetrahedron import tetrahedron_quadrature
from pymhm.fem.scalar.tetrahedron_topology import tetra_polynomials
from pymhm.fem.scalar.triangle import reference_basis
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.materials.evaluation import tensor_values, tensor_values_3d
from pymhm.materials.planar import PlanarMaterial
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.triangle import TriangleMesh


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


class TangentialTraceSpace:
    """Scalar TM or two-component tangential density on actual macrofaces.

    In 3D canonical faces use deterministic orthonormal tangent frames.
    Coefficients represent a density in the declared canonical tangent frame;
    the opposite incident cell receives the opposite sign. In scalar 2D
    pairings there is one coordinate per face basis. No normal coordinate is
    stored by the vector 3D pairing.
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


def trace_coupling(
    skeleton: TangentialTraceSpace, cell: int, fine: Any, degree: int, order: int
) -> Any:
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
class CurlOperators:
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
    skeleton: TangentialTraceSpace,
    cell: int,
    degree: int,
    permittivity: Any,
    permeability: Any,
    order: int,
) -> CurlOperators:
    """Assemble paired scalar/vector 2D or vector/vector 3D DG curl operators.

    Electric and magnetic labels identify the two mass-weighted coordinate
    groups; their differential pairing is C=-curl and -C.T. Coefficients are
    interleaved by component within each independent element-local node.
    """
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
    return CurlOperators(
        mesh,
        degree,
        electric,
        magnetic,
        curl.tocsc(),
        trace_coupling(skeleton, cell, mesh, degree, order),
        skeleton.cell_dofs(cell),
    )


def field_values(value: Any, points: FloatArray, components: int) -> FloatArray:
    """Evaluate a real scalar/vector Maxwell field with explicit component count."""
    data = value(points) if callable(value) else value
    if np.iscomplexobj(data):
        raise ValueError("time-domain Maxwell data must be real")
    array = np.asarray(data, dtype=float)
    if components == 1 and array.shape == (len(points),):
        array = array[:, None]
    try:
        result = np.broadcast_to(array, (len(points), components))
    except ValueError as error:
        raise ValueError("Maxwell data have incompatible component shape") from error
    if not np.isfinite(result).all():
        raise ValueError("Maxwell data must be finite")
    return result


def volume_load(
    local: CurlOperators, value: Any, components: int, material: Any, order: int
) -> FloatArray:
    """Integrate a physical field against DG tests with its optional mass material."""
    mesh = local.mesh
    bary, weights, data = quadrature(mesh, material, order)
    basis, _ = physical_basis(mesh, local.degree, bary)
    points = physical_points(mesh, bary)
    points = points.reshape(-1, mesh.points.shape[1])
    field = field_values(value, points, components)
    if components == 1:
        weighted = positive_values(data, points)[:, None] * field
    else:
        evaluate = tensor_values if components == 2 else tensor_values_3d
        weighted = np.einsum("qab,qb->qa", evaluate(data, points), field)
    return np.einsum(
        "tq,tqi,tqa->tia", weights, basis, weighted.reshape(*weights.shape, components)
    ).ravel()


def tangential_rules(
    skeleton: TangentialTraceSpace, faces: Iterable[int], order: int
) -> Iterator[tuple[int, FloatArray, FloatArray, FloatArray, FloatArray, IntArray]]:
    """Yield physical face rules, canonical frames and independent trace coordinates.

    The basis and weights retain every explicit segment or triangular partition.
    Scalar edge pairings use the unit scalar frame; vector face pairings use
    the declared canonical orthonormal tangent frame.
    """
    mesh, base = skeleton.mesh, skeleton.base
    for face in faces:
        if skeleton.components == 1:
            parameter, weights = base.faces[face].quadrature(order)
            start, end = mesh.points[mesh.faces[face]]
            points = start + parameter[:, None] * (end - start)
            yield (
                face,
                points,
                weights * mesh.lengths[face],
                base.faces[face].evaluate(parameter),
                np.ones((1, 1)),
                skeleton.dofs(face),
            )
        else:
            bary, weights = triangle_quadrature(order)
            for segment, partition in enumerate(base.face_partition(face)):
                points = bary @ partition @ mesh.points[mesh.faces[face]]
                basis = base.basis(face, bary)
                ids = skeleton.dofs(face)[
                    segment * basis.shape[1] * 2 : (segment + 1) * basis.shape[1] * 2
                ]
                yield (
                    face,
                    points,
                    weights * mesh.areas[face] * base.face_weights(face)[segment],
                    basis,
                    skeleton.frames[face],
                    ids,
                )


def tangential_mass(
    skeleton: TangentialTraceSpace, face_weights: Mapping[int, Any], order: int
) -> Any:
    """Assemble a positive scalar-weighted mass in canonical tangent coordinates."""
    matrix = sparse.lil_matrix((skeleton.size,) * 2)
    for face, points, weights, basis, frame, ids in tangential_rules(skeleton, face_weights, order):
        alpha = positive_values(face_weights[face], points)
        block = basis.T @ ((weights * alpha)[:, None] * basis)
        matrix[np.ix_(ids, ids)] += np.kron(block, frame.T @ frame)
    return matrix.tocsc()


def tangential_load(
    skeleton: TangentialTraceSpace, value: Any, faces: Iterable[int], order: int
) -> FloatArray:
    """Integrate a scalar/vector field's canonical tangential moments on chosen faces.

    A callable receives ``(physical_points, canonical_normals)``. Only the
    tangential projection contributes in 3D. Each face is added once in the
    caller's explicit order; this routine does not infer boundary conditions.
    """
    result = np.zeros(skeleton.size)
    components = 1 if skeleton.components == 1 else 3
    for face, points, weights, basis, frame, ids in tangential_rules(skeleton, faces, order):
        normals = np.broadcast_to(skeleton.mesh.normals[face], points.shape)
        data = value(points, normals) if callable(value) else value
        values = field_values(data, points, components) @ frame
        result[ids] += np.einsum("q,qi,qa->ia", weights, basis, values).ravel()
    return result


def tangential_moments(
    operators: Iterable[CurlOperators], fields: Iterable[FloatArray], trace_size: int
) -> FloatArray:
    """Accumulate signed incident tangential field moments in explicit cell order."""
    result = np.zeros(trace_size)
    for local, field in zip(operators, fields, strict=True):
        np.add.at(result, local.trace_dofs, local.coupling.T @ field)
    return result
