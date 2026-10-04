"""User-defined complex acoustic forms with outgoing data on every exterior face.

This focused tutorial reuses reference bases, material quadrature, edge maps
and the exact real/imaginary embedding. It declares its own volume, impedance
and global continuity forms. Its affine manufactured patch is separate from
the archived wave and PML convergence studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.multiscale import MultiscaleProblem, MultiscaleSystem, assemble
from pymhm.fem.scalar.helmholtz import (
    acoustic_quadrature,
    acoustic_space,
    complex_values,
    complex_vector,
    edge_rule,
    material_trace,
    real_matrix,
    real_vector,
    volume_forms,
)
from pymhm.fem.traces.helmholtz import helmholtz_skeleton
from pymhm.fem.traces.interval import SkeletonSpace


@dataclass(frozen=True)
class AcousticCell:
    """Physical mesh/material data for a declared local acoustic form."""

    macro: Any
    skeleton: SkeletonSpace
    cell: int
    omega: float
    degree: int
    refinement: int
    order: int
    density: Any
    modulus: Any
    source: Any
    boundary: Any


def acoustic_equations(item: AcousticCell) -> LocalEquations:
    """Declare diffusion minus frequency mass with exterior outgoing impedance.

    The local equation is A p+B lambda=f. Interior B maps canonical physical
    normal-flux moments, and C=-B.T declares pressure continuity globally.
    Absorbing exterior coordinates are zero placeholders; the actual physical
    flux there follows from pressure and the prescribed impedance data.
    Complex coordinates are interleaved Re/Im through the common exact maps.
    """
    macro, skeleton, cell = item.macro, item.skeleton, item.cell
    fine = macro.submesh(cell, item.refinement)
    stiffness, mass, load = volume_forms(
        fine, item.degree, item.density, item.modulus, item.source, item.order, None
    )
    matrix = (stiffness - item.omega**2 * mass).astype(complex).tolil()
    _, nodes = acoustic_space(fine, item.degree)
    width = sum(skeleton.faces[face].size for face in macro.cell_faces[cell])
    pairing = np.zeros((len(nodes), width))
    offset = 0
    exterior = set(macro.boundary_faces)
    for side, face in enumerate(macro.cell_faces[cell]):
        space = skeleton.faces[face]
        normal = macro.normals[face] * macro.signs[cell, side]
        for ids, parameter, points, weights, basis in edge_rule(
            macro,
            fine,
            cell,
            face,
            item.degree,
            space,
            item.order,
            (item.density, item.modulus),
        ):
            if face in exterior:
                interior = macro.points[macro.cells[cell]].mean(axis=0)
                density = material_trace(item.density, points, interior)
                modulus = material_trace(item.modulus, points, interior)
                matrix[np.ix_(ids, ids)] -= (
                    1j
                    * item.omega
                    * (basis.T @ ((weights / np.sqrt(density * modulus))[:, None] * basis))
                )
                data = (
                    item.boundary(points, np.broadcast_to(normal, points.shape))
                    if callable(item.boundary)
                    else item.boundary
                )
                values = complex_values(data, points)
                np.add.at(load, ids, basis.T @ (weights * values))
            else:
                pairing[ids, offset : offset + space.size] += (
                    macro.signs[cell, side]
                    * basis.T
                    @ (weights[:, None] * space.evaluate(parameter))
                )
        offset += space.size
    coupling = sparse.kron(pairing, sparse.eye(2)).toarray()
    return LocalEquations(
        a=real_matrix(matrix.tocsc()),
        L=real_vector(load),
        b=coupling,
        c=-coupling.T,
        dofs=skeleton.cell_dofs(cell),
        metadata=fine,
    )


def solve_patch(
    mesh: Any,
    *,
    omega: float,
    source: Any,
    boundary: Any,
    degree: int = 2,
    local_refinement: int = 2,
    quadrature_order: int = 8,
    density: Any = 1.0,
    modulus: Any = 1.0,
) -> tuple[MultiscaleSystem, Any]:
    """Solve the explicitly declared all-impedance patch through the generic core."""
    skeleton = helmholtz_skeleton(mesh, omega)
    items = tuple(
        AcousticCell(
            mesh,
            skeleton,
            cell,
            omega,
            degree,
            local_refinement,
            quadrature_order,
            density,
            modulus,
            source,
            boundary,
        )
        for cell in range(len(mesh.cells))
    )
    fixed = {int(index): 0.0 for face in mesh.boundary_faces for index in skeleton.dofs(face)}
    problem = MultiscaleProblem(
        Equation(0, 0), acoustic_equations, items, skeleton.size, (0,) * len(items), fixed=fixed
    )
    system = assemble(problem)
    return system, system.solve()


def pressure_error(
    system: MultiscaleSystem, solution: Any, exact: Any, degree: int = 2, order: int = 8
) -> float:
    """Integrate the complex physical pressure error on independent local meshes."""
    total = 0.0
    for fine, values in zip(system.local_metadata, solution.fields, strict=True):
        dofs, points, weights, basis, _, _ = acoustic_quadrature(fine, degree, 1.0, order)
        field = np.einsum("tqi,ti->tq", basis, complex_vector(values)[dofs])
        truth = complex_values(exact, points.reshape(-1, 2)).reshape(weights.shape)
        total += float(np.sum(weights * abs(field - truth) ** 2))
    return float(np.sqrt(total))


def macro_balances(system: MultiscaleSystem, solution: Any) -> Any:
    """Evaluate the executed constant-test equations separately in each macrocell."""
    result = []
    for response, values in zip(system.responses, solution.fields, strict=True):
        problem = response.problem
        defect = (
            problem.matrix @ values
            + problem.coupling @ solution.trace[problem.trace_dofs]
            - problem.load
        )
        result.append(np.sum(complex_vector(defect)))
    return np.asarray(result)
