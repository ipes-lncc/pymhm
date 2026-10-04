"""User-written three-field and moment equations for an affine diffusion patch.

The example declares local and global blocks instead of selecting a physical
solver in PyMHM. Scalar element integration and constrained energy reconstruction
are shared library operations. Pressure is 1+x+2y, diffusion is the identity,
and the independently derived volume source is zero.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm import (
    Equation,
    FaceSpace,
    LocalEquations,
    MultiscaleProblem,
    SkeletonSpace,
    TriangleMesh,
    assemble,
    energy_reconstruction,
)
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.core.validation import FloatArray
from pymhm.fem.scalar.triangle import scalar_operators, tabulate, trace_coupling


def pressure(points: FloatArray) -> FloatArray:
    """Evaluate the analytical pressure, with physical flux (-1,-2)."""
    return 1.0 + points[:, 0] + 2 * points[:, 1]


def _operators(
    cell: int, mesh: TriangleMesh, skeleton: SkeletonSpace
) -> tuple[Any, Any, FloatArray, FloatArray, TriangleMesh, FloatArray]:
    """Assemble A, volume moments and unsigned local boundary pairings for P2."""
    fine = mesh.submesh(cell, 2)
    a, mass, f = scalar_operators(fine, 2, diffusion=1.0, source=0.0, order=6)
    coupling = trace_coupling(mesh, cell, fine, skeleton, 2)
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        width = skeleton.faces[face].size
        coupling[:, offset : offset + width] *= mesh.signs[cell, side]
        offset += width
    _, points, _, _, _ = tabulate(fine, 2, [[1 / 3, 1 / 3, 1 / 3]])
    return a, mass, f, coupling, fine, points


def three_field_cell(cell: int, *, mesh: TriangleMesh, skeleton: SkeletonSpace) -> LocalEquations:
    """Declare Ap-Beta=f, Beta.T p=P rho and P.T eta=0.

    Here eta is the outward conormal A grad(p).n, minus physical Darcy flux.
    rho is a continuous P1 scalar trace with one coordinate per macro vertex.
    The local field contains pressure followed by private conormal coordinates.
    """
    a, _, f, beta, _, points = _operators(cell, mesh, skeleton)
    ids = np.sort(mesh.cells[cell])
    pairing = np.zeros((len(mesh.cell_faces[cell]), len(ids)))
    for side, face in enumerate(mesh.cell_faces[cell]):
        pairing[side, np.searchsorted(ids, mesh.faces[face])] = mesh.lengths[face] / 2
    local = sparse.bmat([[a, -beta], [-beta.T, None]], format="csc")
    b = np.vstack((np.zeros((len(points), len(ids))), pairing))
    return LocalEquations(
        local,
        np.r_[f, np.zeros(len(pairing))],
        b,
        b.T,
        ids,
        metadata={"points": points, "pressure_size": len(points)},
    )


def moment_cell(cell: int, *, mesh: TriangleMesh, skeleton: SkeletonSpace) -> LocalEquations:
    """Declare the Galerkin moment operator R.T A R and eliminate its cell moment.

    C contains one volume integral and one unsigned integral per macroface.
    R satisfies C.T R=I and A R+C mu=0. The affine zero-source patch uses the
    projected-source MsHHO equation with cell moment degree zero. Face unknowns
    are integrals of pressure, rather than pressure coefficients or fluxes.
    """
    a, mass, _, beta, _, points = _operators(cell, mesh, skeleton)
    moments = np.column_stack((np.asarray(mass.sum(axis=1)).ravel(), beta))
    reconstruction, energy = energy_reconstruction(a, moments)
    return LocalEquations(
        energy[:1, :1],
        [0.0],
        energy[:1, 1:],
        energy[1:, :1],
        skeleton.cell_dofs(cell),
        d=energy[1:, 1:],
        metadata={"points": points, "reconstruction": reconstruction},
    )


def build_problem(
    method: Literal["three-field", "moments"], subdivisions: int = 2
) -> MultiscaleProblem[int]:
    """Define a complete user-written variational patch with exact boundary data."""
    mesh = TriangleMesh.unit_square(subdivisions)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0) for _ in mesh.faces))
    if method == "three-field":
        boundary_vertices = np.unique(mesh.faces[mesh.boundary_faces])
        exact = pressure(mesh.points)
        fixed = {int(i): float(exact[i]) for i in boundary_vertices}
        provider = partial(three_field_cell, mesh=mesh, skeleton=skeleton)
        size = len(mesh.points)
    elif method == "moments":
        exact = pressure(mesh.points[mesh.faces].mean(axis=1)) * mesh.lengths
        fixed = {int(i): float(exact[i]) for i in mesh.boundary_faces}
        provider = partial(moment_cell, mesh=mesh, skeleton=skeleton)
        size = skeleton.size
    else:
        raise ValueError("method must be three-field or moments")
    return MultiscaleProblem(
        Equation(0, 0),
        provider,
        range(len(mesh.cells)),
        size,
        (0,) * len(mesh.cells),
        fixed=fixed,
    )


def recover_pressure(system: MultiscaleSystem, solution: Any) -> tuple[FloatArray, ...]:
    """Interpret each method's explicitly declared local coefficient layout."""
    values = []
    for response, data, field in zip(
        system.responses, system.local_metadata, solution.fields, strict=True
    ):
        if "pressure_size" in data:
            values.append(field[: data["pressure_size"]])
        else:
            moments = np.r_[field, solution.trace[response.problem.trace_dofs]]
            values.append(data["reconstruction"] @ moments)
    return tuple(values)


def run_patch(method: Literal["three-field", "moments"]) -> dict[str, Any]:
    """Solve and independently check the affine physical field of the declared blocks."""
    system = assemble(build_problem(method))
    solution = system.solve()
    fields = recover_pressure(system, solution)
    error = max(
        float(np.max(abs(field - pressure(data["points"]))))
        for field, data in zip(fields, system.local_metadata, strict=True)
    )
    if error > 1e-10:
        raise RuntimeError("the declared hybrid equations did not recover the affine field")
    return {
        "method": method,
        "pressure_coefficient_error": error,
        "original_equation_residual": solution.raw_residual,
        "scope": "affine analytical patch; no literature convergence claim",
    }
