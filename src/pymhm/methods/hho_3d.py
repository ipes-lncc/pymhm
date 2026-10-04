"""Three-dimensional MsHHO on tetrahedral and convex polyhedral macro partitions."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.transport.polyhedral import (
    PolygonalSkeleton3D,
    _face_quadrature,
    polygonal_trace_coupling,
)
from pymhm.core.moments import energy_reconstruction
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.reference import legendre_values
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_operators, tetra_tabulate, tetrahedron_quadrature
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.materials.evaluation import scalar_values_3d, tensor_values_3d, vector_values_3d
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.methods.hho import MsHHOLocal, MsHHOSolution, _condense_moments

# Preserve the previous module attribute without duplicating the implementation.


@dataclass(frozen=True)
class MsHHO3DSolution(MsHHOSolution):
    """Broken tetrahedral pressure reconstructions with original-macroface moments."""

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate pressure error in physical volume using independent quadrature."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for local, field in zip(self.local, self.pressure, strict=True):
            fine = local.mesh
            dofs, _, basis, _ = tetra_tabulate(fine, self.degree, bary)
            physical = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
            target = scalar_values_3d(exact, physical.reshape(-1, 3)).reshape(physical.shape[:2])
            difference = field[dofs] @ basis.T - target
            total += float(fine.volumes @ (difference**2 @ weights))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the physical raw flux -A grad(p), without an H(div) assertion."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for local, field in zip(self.local, self.pressure, strict=True):
            fine = local.mesh
            dofs, _, _, gradient = tetra_tabulate(fine, self.degree, bary)
            physical = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
            material = tensor_values_3d(self.permeability, physical.reshape(-1, 3)).reshape(
                *physical.shape[:2], 3, 3
            )
            flux = -np.einsum("tqab,tqib,ti->tqa", material, gradient, field[dofs])
            exact_flux = vector_values_3d(exact, physical.reshape(-1, 3)).reshape(flux.shape)
            total += float(fine.volumes @ (np.sum((flux - exact_flux) ** 2, axis=-1) @ weights))
        return float(np.sqrt(total))


def _local(
    mesh: Any,
    cell: int,
    skeleton: Any,
    material: Any,
    source: Any,
    cell_degree: int,
    degree: int,
    refinement: int,
    order: int,
    variant: str,
    refinement_precision: Literal["double", "extended"],
) -> MsHHOLocal:
    """Construct constrained-energy bases with volume and original-face moments."""
    fine = mesh.submesh(cell, refinement)
    matrix, mass, force = tetra_operators(
        fine, degree, diffusion=material, source=source, order=order
    )
    bary, weights = tetrahedron_quadrature(max(order, degree + 2, cell_degree + 2))
    dofs, nodes, basis, _ = tetra_tabulate(fine, degree, bary)
    physical = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
    coordinates = 2 * (physical - fine.points.min(axis=0)) / np.ptp(fine.points, axis=0) - 1
    if cell_degree < 0:
        cell_basis = np.empty((*physical.shape[:2], 0))
    else:
        factors = [legendre_values(coordinates[..., a], cell_degree) for a in range(3)]
        cell_basis = np.stack(
            [
                factors[0][..., i] * factors[1][..., j] * factors[2][..., k]
                for k in range(cell_degree + 1)
                for j in range(cell_degree + 1 - k)
                for i in range(cell_degree + 1 - k - j)
            ],
            axis=-1,
        )
    count = cell_basis.shape[-1]
    volume = np.zeros((len(nodes), count))
    if count:
        entries = np.einsum("t,q,qi,tqj->tij", fine.volumes, weights, basis, cell_basis)
        np.add.at(volume, dofs.ravel(), entries.reshape(-1, count))
    coupling = (
        polygonal_trace_coupling if isinstance(mesh, PolyhedralMesh) else tetra_trace_coupling
    )
    faces = coupling(mesh, cell, fine, skeleton, degree)
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        width = len(skeleton.dofs(int(face)))
        faces[:, offset : offset + width] *= mesh.signs[cell][side]
        offset += width
    moments = np.column_stack((volume, faces))
    reconstruction, energy = energy_reconstruction(matrix, moments, refinement_precision)
    if variant == "reconstructed":
        load = reconstruction.T @ force
    else:
        gram = np.einsum("t,q,tqi,tqj->ij", fine.volumes, weights, cell_basis, cell_basis)
        sampled = scalar_values_3d(source, physical.reshape(-1, 3)).reshape(physical.shape[:2])
        projected = np.linalg.solve(
            gram, np.einsum("t,q,tqi,tq->i", fine.volumes, weights, cell_basis, sampled)
        )
        load = np.r_[projected, np.zeros(faces.shape[1])]
    integral = np.asarray(mass.sum(axis=1)).ravel() @ reconstruction
    return MsHHOLocal(fine, reconstruction, moments, energy, load, count, integral)


def _face_rule(
    mesh: Any, skeleton: Any, face: int, order: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return physical points, area weights and moment basis for either face topology."""
    if isinstance(mesh, PolyhedralMesh):
        points, weights = _face_quadrature(mesh, face, order)
        return points, weights, np.asarray(skeleton.basis(face, points), dtype=np.float64)
    bary, weights = triangle_quadrature(order)
    partition = skeleton.face_partition(int(face))
    points = np.einsum("qi,sij->sqj", bary, partition @ mesh.points[mesh.faces[face]])
    values = skeleton.basis(face, bary)
    basis = np.kron(np.eye(len(partition)), values)
    return (
        points.reshape(-1, 3),
        (skeleton.face_weights(int(face))[:, None] * weights).ravel() * mesh.areas[face],
        np.asarray(basis, dtype=np.float64),
    )


def solve_mshho_3d(
    mesh: TetraMesh | PolyhedralMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: TriangularSkeleton | PolygonalSkeleton3D | None = None,
    cell_degree: int = 0,
    degree: int = 2,
    local_refinement: int = 2,
    source_variant: Literal["projected", "reconstructed"] = "projected",
    quadrature_order: int = 5,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_refinement_precision: Literal["double", "extended"] = "double",
) -> MsHHO3DSolution:
    """Solve MsHHO with tetrahedral Pk energy lifts and polynomial face moments.

    Polyhedral faces retain one polynomial on each original polygon. Tetrahedral
    faces additionally permit independent aligned subdivisions. ``neumann``
    prescribes physical outward Darcy flux; other exterior faces prescribe
    pressure. Pure Neumann data require compatibility and fix the volume mean.
    The face-only case m=-1 requires the reconstructed-source variant. Local
    numerical spaces must represent all independent volume and face moments.
    ``local_refinement_precision='extended'`` uses the same preserved moment
    arithmetic as the two-dimensional and polygonal solvers, without changing
    local spaces, represented operators or the original 1e-10 residual check.
    """
    if not isinstance(mesh, (TetraMesh, PolyhedralMesh)):
        raise TypeError("MsHHO 3D requires TetraMesh or PolyhedralMesh")
    refinement = _dyadic(local_refinement, "local_refinement")
    degree = positive_int(degree, "degree")
    m = positive_int(cell_degree, "cell_degree", -1)
    order = positive_int(quadrature_order, "quadrature_order")
    if local_refinement_precision not in {"double", "extended"}:
        raise ValueError("local_refinement_precision must be double or extended")
    if source_variant not in ("projected", "reconstructed") or (
        m == -1 and source_variant != "reconstructed"
    ):
        raise ValueError(
            "source variant must be projected or reconstructed; m=-1 needs reconstructed"
        )
    if skeleton is None:
        skeleton = (
            PolygonalSkeleton3D(mesh, 0)
            if isinstance(mesh, PolyhedralMesh)
            else TriangularSkeleton(mesh)
        )
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to this macro mesh")
    local = tuple(
        _local(
            mesh,
            c,
            skeleton,
            permeability,
            source,
            m,
            degree,
            refinement,
            order,
            source_variant,
            local_refinement_precision,
        )
        for c in range(len(mesh.cells))
    )
    prescribed = {} if neumann is None else neumann
    if not set(prescribed).issubset(set(mesh.boundary_faces)):
        raise ValueError("Neumann data require exterior face indices")
    pure = set(prescribed) == set(mesh.boundary_faces)
    if not np.isfinite(mean_pressure) or (not pure and mean_pressure != 0):
        raise ValueError("mean_pressure must be finite and is a gauge only for pure Neumann data")
    constants, natural = np.zeros(skeleton.size), np.zeros(skeleton.size)
    fixed: dict[int, float] = {}
    for face in range(len(mesh.faces)):
        points, weights, basis = _face_rule(mesh, skeleton, face, max(order, 3))
        ids = skeleton.dofs(face)
        constants[ids] = basis.T @ weights
        if face in prescribed:
            mass = basis.T @ (weights[:, None] * basis)
            natural[ids] = -np.linalg.solve(
                mass, basis.T @ (weights * scalar_values_3d(prescribed[face], points))
            )
        elif face in mesh.boundary_faces:
            fixed.update(
                zip(ids, basis.T @ (weights * scalar_values_3d(dirichlet, points)), strict=True)
            )
    volume = float(sum(data.mesh.volumes.sum() for data in local))
    matrix, values, coarse, fields, residual = _condense_moments(
        local,
        skeleton,
        fixed,
        natural,
        constants,
        mean_pressure * volume if pure else None,
        solver,
        local_refinement_precision,
    )
    return MsHHO3DSolution(
        skeleton,
        local,
        fields,
        values,
        coarse,
        matrix,
        residual,
        degree,
        permeability,
        source_variant,
        local_refinement_precision,
    )
