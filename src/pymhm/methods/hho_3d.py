"""Three-dimensional MsHHO on tetrahedral and convex polyhedral macro partitions."""

from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.transport.polyhedral import (
    PolygonalSkeleton3D,
    polygonal_trace_coupling,
)
from pymhm.core.moments import energy_reconstruction
from pymhm.core.validation import positive_int
from pymhm.fem.reference import legendre_values
from pymhm.fem.scalar.tetrahedron import tetra_operators, tetra_tabulate, tetrahedron_quadrature
from pymhm.fem.traces.moments_3d import face_moment_rule_3d as _face_rule
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.materials.evaluation import scalar_values_3d
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.methods.hho import MsHHOLocal, _condense_moments
from pymhm.postprocessing.solutions import MsHHO3DSolution as MsHHO3DSolution

# Preserve the previous module attribute without duplicating the implementation.


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
