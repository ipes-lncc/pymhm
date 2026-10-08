"""Application MsHHO forms in integral-moment coordinates.

The reconstruction is the shared constrained energy minimum. The application
declares its cell/face Galerkin blocks, source functional and boundary moments;
the generic core owns elimination and global assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np

from pymhm.core.contracts import HybridSolution
from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.moments import energy_reconstruction
from pymhm.core.multiscale import MultiscaleProblem, MultiscaleSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.reference import legendre_values
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import scalar_operators, tabulate, trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.materials.evaluation import scalar_values
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.solutions import MsHHOLocal, MsHHOSolution


@dataclass(frozen=True)
class MomentDefinition:
    """Declared moment equations and data needed for physical pressure reconstruction."""

    problem: MultiscaleProblem[int]
    skeleton: SkeletonSpace
    degree: int
    permeability: Any
    source_variant: str
    pressure_integral: float | None
    reconstruction_precision: Literal["double", "extended"]


def local_equations(
    cell: int,
    *,
    mesh: TriangleMesh,
    skeleton: SkeletonSpace,
    permeability: Any,
    source: Any,
    cell_degree: int,
    degree: int,
    refinement: int,
    order: int,
    variant: str,
    reconstruction_precision: Literal["double", "extended"],
) -> LocalEquations:
    """Declare all four blocks of R.T A R and the selected source functional.

    C contains volume and unsigned macroface integral functionals. R satisfies
    C.T R=I and A R+C mu=0. Integral moments are independent of pressure/flux
    coefficients. The operator retains its represented small antisymmetry.
    Reconstructed sources use R.T f; projected sources test the cell polynomial
    projection. The face-only case is admissible only for reconstructed sources.
    """
    fine = mesh.submesh(cell, refinement)
    matrix, _, force = scalar_operators(
        fine, degree, diffusion=permeability, source=source, order=order
    )
    bary, weights = triangle_quadrature(max(order, degree + 2, cell_degree + 2))
    dofs, nodes, basis, _, _ = tabulate(fine, degree, bary)
    points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
    vertices = mesh.points[mesh.cells[cell]]
    if cell_degree < 0:
        cell_basis = np.empty((*points.shape[:2], 0))
    else:
        coordinates = 2 * (points - vertices.min(axis=0)) / np.ptp(vertices, axis=0) - 1
        px, py = (legendre_values(coordinates[..., axis], cell_degree) for axis in range(2))
        cell_basis = np.stack(
            [
                px[..., i] * py[..., j]
                for j in range(cell_degree + 1)
                for i in range(cell_degree + 1 - j)
            ],
            axis=-1,
        )
    count = cell_basis.shape[-1]
    volume = np.zeros((len(nodes), count))
    local = np.einsum("t,q,qi,tqj->tij", fine.areas, weights, basis, cell_basis)
    if count:
        np.add.at(volume, dofs.ravel(), local.reshape(-1, count))
    face = trace_coupling(mesh, cell, fine, skeleton, degree)
    offset = 0
    for side, face_id in enumerate(mesh.cell_faces[cell]):
        width = skeleton.faces[face_id].size
        face[:, offset : offset + width] *= mesh.signs[cell, side]
        offset += width
    moments = np.column_stack((volume, face))
    reconstruction, energy = energy_reconstruction(matrix, moments, reconstruction_precision)
    if variant == "reconstructed":
        load = reconstruction.T @ force
    else:
        gram = np.einsum("t,q,tqi,tqj->ij", fine.areas, weights, cell_basis, cell_basis)
        sampled = scalar_values(source, points.reshape(-1, 2)).reshape(points.shape[:2])
        projected = np.linalg.solve(
            gram, np.einsum("t,q,tqi,tq->i", fine.areas, weights, cell_basis, sampled)
        )
        load = np.r_[projected, np.zeros(face.shape[1])]
    integral = (
        np.bincount(
            dofs.ravel(),
            weights=np.einsum("t,q,qi->ti", fine.areas, weights, basis).ravel(),
            minlength=len(nodes),
        )
        @ reconstruction
    )
    data = MsHHOLocal(fine, reconstruction, moments, energy, load, count, integral)
    return LocalEquations(
        energy[:count, :count],
        load[:count],
        energy[:count, count:],
        energy[count:, :count],
        skeleton.cell_dofs(cell),
        d=energy[count:, count:],
        g=load[count:],
        metadata=(data, integral[:count]),
        field_data=(
            nodal_field(
                "pressure",
                fine,
                degree,
                reconstruction=reconstruction[:, :count],
                trace_reconstruction=reconstruction[:, count:],
                trace_dofs=skeleton.cell_dofs(cell),
            ),
        ),
    )


def define_moment_diffusion(
    mesh: TriangleMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    skeleton: SkeletonSpace | None = None,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    cell_degree: int = 0,
    degree: int = 2,
    local_refinement: int = 4,
    source_variant: Literal["projected", "reconstructed"] = "projected",
    quadrature_order: int = 6,
    reconstruction_precision: Literal["double", "extended"] = "double",
) -> MomentDefinition:
    """Declare MsHHO energy/source blocks and pressure-moment boundary conditions.

    Physical outward Neumann flux produces a negative dual coefficient load in
    the face-moment equation. Dirichlet data fix actual face integrals. Pure
    Neumann data impose the volume mean of the complete reconstructed pressure.
    Extended reconstruction keeps its existing numerical basis precision;
    execution and linear solve choices remain explicit at generic assembly.
    """
    cell_degree = positive_int(cell_degree, "cell_degree", -1)
    degree = positive_int(degree, "degree")
    refinement = positive_int(local_refinement, "local_refinement")
    order = positive_int(quadrature_order, "quadrature_order")
    if source_variant not in {"projected", "reconstructed"}:
        raise ValueError("source_variant must be projected or reconstructed")
    if reconstruction_precision not in {"double", "extended"}:
        raise ValueError("reconstruction_precision must be double or extended")
    if cell_degree == -1 and source_variant != "reconstructed":
        raise ValueError("face-only moments require the reconstructed source")
    skeleton = (
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(0) for _ in mesh.faces))
        if skeleton is None
        else skeleton
    )
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("moment diffusion requires its own scalar skeleton")
    prescribed = {} if neumann is None else dict(neumann)
    if not set(prescribed).issubset(set(mesh.boundary_faces)):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    pure_neumann = set(prescribed) == set(mesh.boundary_faces)
    if not pure_neumann and mean_pressure != 0:
        raise ValueError("mean_pressure applies only to pure Neumann data")
    fixed: dict[int, float] = {}
    natural = np.zeros(skeleton.size)
    for face_id in range(len(mesh.faces)):
        face = skeleton.faces[face_id]
        t, w = face.quadrature(max(order, max(face.degrees) + 2))
        start, end = mesh.points[mesh.faces[face_id]]
        points = start + t[:, None] * (end - start)
        ids, basis = skeleton.dofs(face_id), face.evaluate(t)
        measure = mesh.lengths[face_id] * w
        if face_id in prescribed:
            mass = np.einsum("q,qi,qj->ij", measure, basis, basis)
            natural[ids] = -np.linalg.solve(
                mass, basis.T @ (measure * scalar_values(prescribed[face_id], points))
            )
        elif face_id in mesh.boundary_faces:
            values = basis.T @ (measure * scalar_values(dirichlet, points))
            fixed.update(zip(ids, values, strict=True))
    provider = partial(
        local_equations,
        mesh=mesh,
        skeleton=skeleton,
        permeability=permeability,
        source=source,
        cell_degree=cell_degree,
        degree=degree,
        refinement=refinement,
        order=order,
        variant=source_variant,
        reconstruction_precision=reconstruction_precision,
    )
    problem = MultiscaleProblem(
        Equation(0, natural),
        provider,
        range(len(mesh.cells)),
        skeleton.size,
        (0,) * len(mesh.cells),
        fixed=fixed,
    )
    return MomentDefinition(
        problem,
        skeleton,
        degree,
        permeability,
        source_variant,
        mean_pressure * float(mesh.areas.sum()) if pure_neumann else None,
        reconstruction_precision,
    )


def pressure_constraints(
    definition: MomentDefinition, system: MultiscaleSystem
) -> list[tuple[FloatArray, float]]:
    """Declare the full reconstructed physical pressure integral, including face moments."""
    if definition.pressure_integral is None:
        return []
    row, target = system.mean_constraint(
        [item[1] for item in system.local_metadata], definition.pressure_integral
    )
    for cell, metadata in enumerate(system.local_metadata):
        local = metadata[0]
        np.add.at(row, definition.skeleton.cell_dofs(cell), local.integral[local.cell_count :])
    return [(row, target)]


def recover_moment_diffusion(
    definition: MomentDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> MsHHOSolution:
    """Recover pressure through the exact executed constrained reconstruction matrix."""
    local = tuple(item[0] for item in system.local_metadata)
    fields = tuple(
        data.reconstruction @ np.r_[interior, solution.trace[definition.skeleton.cell_dofs(cell)]]
        for cell, (data, interior) in enumerate(zip(local, solution.fields, strict=True))
    )
    return MsHHOSolution(
        definition.skeleton,
        local,
        fields,
        solution.trace,
        solution.fields,
        system.matrix,
        solution.residual,
        definition.degree,
        definition.permeability,
        definition.source_variant,
        definition.reconstruction_precision,
    )
