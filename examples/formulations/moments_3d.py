"""User-defined tetrahedral moment-energy forms with shared algebraic recovery."""

from __future__ import annotations

from functools import partial
from typing import Any, Literal, cast

import numpy as np

from examples.formulations.moments import MomentDefinition, recover_moment_diffusion
from pymhm.core.contracts import HybridSolution
from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.moments import energy_reconstruction
from pymhm.core.multiscale import MultiscaleProblem, MultiscaleSystem
from pymhm.core.validation import positive_int
from pymhm.fem.reference import legendre_values
from pymhm.fem.scalar.tetrahedron import tetra_operators, tetra_tabulate, tetrahedron_quadrature
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.materials.evaluation import scalar_values_3d
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.methods.hho import MsHHOLocal
from pymhm.methods.hho_3d import MsHHO3DSolution, _face_rule


def local_equations(
    cell: int,
    *,
    mesh: TetraMesh,
    skeleton: TriangularSkeleton,
    permeability: Any,
    source: Any,
    cell_degree: int,
    degree: int,
    refinement: int,
    order: int,
    variant: str,
    reconstruction_precision: Literal["double", "extended"],
) -> LocalEquations:
    """Declare cell/face blocks of the actual constrained-energy reconstruction.

    Volume and unsigned face integrals define C. The shared operation returns
    R with C.T R=I and A R+C mu=0, retaining the executed numerical basis and
    represented small antisymmetry. The selected source is R.T f or the
    projected volume-polynomial functional. Global coordinates are physical
    pressure integrals, rather than flux coefficients.
    """
    fine = mesh.submesh(cell, refinement)
    matrix, mass, force = tetra_operators(
        fine, degree, diffusion=permeability, source=source, order=order
    )
    bary, weights = tetrahedron_quadrature(max(order, degree + 2, cell_degree + 2))
    dofs, nodes, basis, _ = tetra_tabulate(fine, degree, bary)
    physical = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
    coordinates = 2 * (physical - fine.points.min(axis=0)) / np.ptp(fine.points, axis=0) - 1
    if cell_degree < 0:
        cell_basis = np.empty((*physical.shape[:2], 0))
    else:
        factors = [legendre_values(coordinates[..., axis], cell_degree) for axis in range(3)]
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
    faces = tetra_trace_coupling(mesh, cell, fine, skeleton, degree)
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        width = len(skeleton.dofs(int(face)))
        faces[:, offset : offset + width] *= mesh.signs[cell][side]
        offset += width
    moments = np.column_stack((volume, faces))
    reconstruction, energy = energy_reconstruction(matrix, moments, reconstruction_precision)
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
    )


def define_moment_diffusion_3d(
    mesh: TetraMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: TriangularSkeleton | None = None,
    cell_degree: int = 0,
    degree: int = 2,
    local_refinement: int = 2,
    source_variant: Literal["projected", "reconstructed"] = "projected",
    quadrature_order: int = 5,
    mean_pressure: float = 0.0,
    reconstruction_precision: Literal["double", "extended"] = "double",
) -> MomentDefinition:
    """Declare tetrahedral moment forms and actual physical boundary integrals.

    Physical outward flux gives the negative dual face load. Dirichlet data
    prescribe face integrals, and full Neumann data additionally prescribe
    the integral of reconstructed pressure. Only tetrahedral geometry is
    declared here; the separate polyhedral studies keep their original spaces.
    The common moment constraint and reconstruction helpers apply unchanged.
    """
    if not isinstance(mesh, TetraMesh):
        raise TypeError("this definition uses tetrahedral moment geometry")
    refinement = _dyadic(local_refinement, "local_refinement")
    degree = positive_int(degree, "degree")
    m = positive_int(cell_degree, "cell_degree", -1)
    order = positive_int(quadrature_order, "quadrature_order")
    if reconstruction_precision not in {"double", "extended"}:
        raise ValueError("reconstruction_precision must be double or extended")
    if source_variant not in {"projected", "reconstructed"} or (
        m == -1 and source_variant != "reconstructed"
    ):
        raise ValueError(
            "projected or reconstructed source required; face-only needs reconstructed"
        )
    skeleton = TriangularSkeleton(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to this macro mesh")
    prescribed = {} if neumann is None else dict(neumann)
    if not set(prescribed).issubset(set(mesh.boundary_faces)):
        raise ValueError("Neumann data require exterior face indices")
    pure = set(prescribed) == set(mesh.boundary_faces)
    if not np.isfinite(mean_pressure) or (not pure and mean_pressure != 0):
        raise ValueError("mean_pressure must be finite and requires pure Neumann data")
    fixed: dict[int, float] = {}
    natural = np.zeros(skeleton.size)
    for face in range(len(mesh.faces)):
        points, weights, basis = _face_rule(mesh, skeleton, face, max(order, 3))
        ids = skeleton.dofs(face)
        if face in prescribed:
            gram = basis.T @ (weights[:, None] * basis)
            natural[ids] = -np.linalg.solve(
                gram, basis.T @ (weights * scalar_values_3d(prescribed[face], points))
            )
        elif face in mesh.boundary_faces:
            values = basis.T @ (weights * scalar_values_3d(dirichlet, points))
            fixed.update(zip(ids, values, strict=True))
    provider = partial(
        local_equations,
        mesh=mesh,
        skeleton=skeleton,
        permeability=permeability,
        source=source,
        cell_degree=m,
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
        cast(Any, skeleton),
        degree,
        permeability,
        source_variant,
        mean_pressure * float(mesh.volumes.sum()) if pure else None,
        reconstruction_precision,
    )


def recover_moment_diffusion_3d(
    definition: MomentDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> MsHHO3DSolution:
    """Reuse the executed algebraic reconstruction and provide physical 3D norms."""
    stored = recover_moment_diffusion(definition, system, solution)
    return MsHHO3DSolution(**vars(stored))
