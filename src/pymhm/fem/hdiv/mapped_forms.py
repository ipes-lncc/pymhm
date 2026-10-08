"""Mapped H(div) volume integration and declared normal-density trace moments."""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.hdiv.mapped import mapped_rt_basis, mapped_rt_dofs, tensor_legendre_values
from pymhm.materials.evaluation import scalar_values_3d, tensor_values_3d
from pymhm.meshes.hexahedron import _UV, HexMesh, QuadratureOrder, _geometry, cube_quadrature

_QUADRATURE_BATCH_ENTRIES = 2_000_000


def mapped_rt_operators(
    mesh: HexMesh, degree: int, permeability: Any, source: Any, order: QuadratureOrder
) -> tuple:
    """Assemble mixed operators with bounded quadrature batches and physical material values."""
    points, w = cube_quadrature(order)
    dofs = mapped_rt_dofs(mesh, degree)
    cells, width = dofs.shape
    pressure_width = (degree + 1) ** 3
    mass = np.zeros((cells, width, width))
    divergence = np.zeros((cells, pressure_width, width))
    load = np.zeros((cells, pressure_width))
    mean = np.zeros_like(load)
    # Bound the largest physical-basis arrays independently of the quadrature order.
    for part in quadrature_slices(mesh, degree, len(points)):
        subset = points[part]
        physical, _, det = mesh.geometry(subset)
        basis, div, pressure = mapped_rt_basis(mesh, degree, subset)
        inverse = np.linalg.inv(tensor_values_3d(permeability, physical.reshape(-1, 3))).reshape(
            *det.shape, 3, 3
        )
        weights = det * w[part]
        weighted = np.einsum("tq,tqab,tqjb->tqja", weights, inverse, basis, optimize=True)
        mass += np.einsum("tqia,tqja->tij", basis, weighted, optimize=True)
        divergence += np.einsum("tq,qi,tqj->tij", weights, pressure, div, optimize=True)
        f = scalar_values_3d(source, physical.reshape(-1, 3)).reshape(det.shape)
        load += np.einsum("tq,qi,tq->ti", weights, pressure, f, optimize=True)
        mean += np.einsum("tq,qi->ti", weights, pressure, optimize=True)
    nq, npres = int(dofs.max()) + 1, cells * pressure_width
    pids = np.arange(npres).reshape(len(mesh.cells), -1)
    return (
        assemble_element_blocks(mass, dofs, dofs, (nq, nq)),
        assemble_element_blocks(divergence, pids, dofs, (npres, nq)),
        load.ravel(),
        mean.ravel(),
    )


def quadrature_slices(mesh: HexMesh, degree: int, count: int) -> Iterator[slice]:
    """Partition one unchanged rule to bound simultaneous physical-basis storage."""
    width = 3 * (degree + 2) * (degree + 1) ** 2
    batch = max(1, _QUADRATURE_BATCH_ENTRIES // (len(mesh.cells) * width * 3))
    for start in range(0, count, batch):
        yield slice(start, start + batch)


@dataclass(frozen=True)
class HexSkeleton:
    """Tensor-Qk reference flux-density traces with aligned uniform face subdivisions."""

    mesh: HexMesh
    degree: int = 1
    subdivisions: int = 1

    def __post_init__(self) -> None:
        """Validate scalar polynomial degree and positive subdivision count."""
        positive_int(self.degree, "trace degree", 0)
        positive_int(self.subdivisions, "trace subdivisions")

    @property
    def face_size(self) -> int:
        """Return the number of reference flux-density modes on each macroface."""
        return self.subdivisions**2 * (self.degree + 1) ** 2

    @property
    def size(self) -> int:
        """Return the global number of scalar normal-flux trace modes."""
        return len(self.mesh.faces) * self.face_size

    def cell_dofs(self, cell: int) -> IntArray:
        """Return the six oriented macroface blocks in local side order."""
        return (
            self.face_size * self.mesh.cell_faces[cell, :, None] + np.arange(self.face_size)
        ).ravel()

    def evaluate(self, points: FloatArray) -> FloatArray:
        """Evaluate reference flux densities, normalized by segment moments."""
        n = self.subdivisions
        indices = np.minimum(np.floor(points * n).astype(int), n - 1)
        local = points * n - indices
        values = tensor_legendre_values(self.degree, local)
        factors = np.array(
            [(2 * i + 1) * (2 * j + 1) for i, j in product(range(self.degree + 1), repeat=2)]
        )
        result = np.zeros((len(points), self.face_size))
        start = (indices[:, 0] * n + indices[:, 1]) * (self.degree + 1) ** 2
        result[np.arange(len(points))[:, None], start[:, None] + np.arange(values.shape[1])] = (
            n * n * values * factors
        )
        return result


def mapped_rt_trace_mapping(
    mesh: HexMesh,
    cell: int,
    fine: HexMesh,
    reference: FloatArray,
    skeleton: HexSkeleton,
    degree: int,
) -> FloatArray:
    """Integrate fine normal moments of each oriented mapped macroface trace."""
    uv, weights = cube_quadrature(degree + skeleton.degree + 2, 2)
    count = (degree + 1) ** 2
    tests = tensor_legendre_values(degree, uv)
    shape = np.prod(np.where(_UV[None], uv[:, None], 1 - uv[:, None]), axis=2)
    result = np.zeros((count * len(fine.boundary_faces), 6 * skeleton.face_size))
    for row, face in enumerate(fine.boundary_faces):
        nodes = reference[fine.faces[face]]
        axis = int(np.flatnonzero(np.ptp(nodes, axis=0) < 1e-13)[0])
        end = int(round(nodes[0, axis]))
        side = 2 * axis + end
        parent_uv = (shape @ nodes)[:, np.arange(3) != axis]
        canonical = (
            np.column_stack((np.ones(len(uv)), parent_uv)) @ mesh.face_transforms[cell, side]
        )
        area = np.prod(np.ptp(nodes[:, np.arange(3) != axis], axis=0))
        result[
            row * count : (row + 1) * count,
            side * skeleton.face_size : (side + 1) * skeleton.face_size,
        ] = (
            mesh.signs[cell, side]
            * area
            * tests.T
            @ (weights[:, None] * skeleton.evaluate(canonical))
        )
    return result


def mapped_boundary_data(
    skeleton: HexSkeleton, dirichlet: Any, neumann: dict[int, Any], order: int
) -> tuple[FloatArray, dict[int, float]]:
    """Integrate weak pressure data and project outward physical normal fluxes."""
    mesh = skeleton.mesh
    if not set(neumann).issubset(set(mesh.boundary_faces)):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    load, fixed = np.zeros(skeleton.size), {}
    uv, weights = cube_quadrature(order, 2)
    n = skeleton.subdivisions
    for face in mesh.boundary_faces:
        cell, side = mesh.incidence[face][0]
        axis, end = divmod(side, 2)
        transform = mesh.face_transforms[cell, side]
        for i, j in product(range(n), repeat=2):
            canonical = (uv + np.array([i, j])) / n
            local = (canonical - transform[0]) @ np.linalg.inv(transform[1:])
            points = np.empty((len(uv), 3))
            points[:, axis], points[:, np.arange(3) != axis] = end, local
            physical, jac, det = _geometry(mesh.points[mesh.cells[cell]][None], points)
            normal = det[0, :, None] * np.linalg.inv(jac[0]).transpose(0, 2, 1)[:, :, axis]
            measure = np.linalg.norm(normal, axis=1)
            trace = skeleton.evaluate(canonical)
            indices = face * skeleton.face_size + np.arange(skeleton.face_size)
            if face not in neumann:
                p = scalar_values_3d(dirichlet, physical[0])
                load[indices] -= trace.T @ (weights * p / n**2)
            else:
                flux = scalar_values_3d(neumann[face], physical[0])
                value = tensor_legendre_values(skeleton.degree, uv).T @ (
                    weights * flux * measure / n**2
                )
                start = (i * n + j) * (skeleton.degree + 1) ** 2
                for a, v in enumerate(value):
                    fixed[int(indices[start + a])] = float(v)
    return load, fixed
