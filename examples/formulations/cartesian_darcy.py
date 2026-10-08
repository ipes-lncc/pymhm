"""Cartesian Qk Darcy forms declared by the user application."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import numpy as np

from pymhm.core.contracts import HybridSolution
from pymhm.core.equations import Equation, LocalEquations, columns, rows
from pymhm.core.multiscale import MultiscaleProblem, MultiscaleSystem
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.quadrilateral import quadrilateral_operators, quadrilateral_trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.cartesian import CartesianMacroMesh, cartesian_refinement
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.solutions import QuadrilateralDarcySolution


@dataclass(frozen=True)
class CartesianDarcyDefinition:
    """Declared Qk equations, physical material data and pressure integral gauge."""

    problem: MultiscaleProblem[int]
    skeleton: SkeletonSpace
    permeability: Any
    source: Any
    degree: int
    quadrature_order: int
    pressure_integral: float | None


def local_equations(
    cell: int,
    *,
    mesh: CartesianMacroMesh,
    skeleton: SkeletonSpace,
    degree: int,
    refinement: tuple[int, int],
    permeability: Any,
    source: Any,
    order: int,
) -> LocalEquations:
    """Declare Qk energy/source, oriented Neumann coupling and pressure continuity.

    The constant pressure kernel and its physical integral moment use the exact
    executed Qk nodal order. C=-B.T supplies global continuity; the trace is the
    oriented physical normal flux and is independent of the raw gradient field.
    """
    fine = mesh.submesh(cell, refinement)
    matrix, mass, load = quadrilateral_operators(
        fine, degree, permeability=permeability, source=source, order=order
    )
    coupling = quadrilateral_trace_coupling(mesh, cell, fine, skeleton, degree)
    kernel = np.ones((matrix.shape[0], 1))
    moments = mass @ kernel
    return LocalEquations(
        matrix,
        load,
        columns(*coupling.T),
        rows(*(-coupling.T)),
        skeleton.cell_dofs(cell),
        kernel=kernel,
        moments=moments,
        metadata=(fine, moments[:, 0]),
        field_data=(nodal_field("pressure", fine, degree),),
    )


def define_cartesian_darcy(
    mesh: CartesianMacroMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int | tuple[int, int] = 4,
    degree: int = 1,
    quadrature_order: int = 4,
    mean_pressure: float = 0.0,
) -> CartesianDarcyDefinition:
    """Declare affine Cartesian Qk locals and their global boundary equation.

    Only axis-aligned rectangles are supported. Independent x/y refinements and
    scalar hp face spaces reuse the geometry/FEM owners. The global load is the
    negative pressure boundary functional. Pure Neumann data additionally impose
    the physical pressure integral; execution is configured at generic assembly.
    """
    if not isinstance(mesh, CartesianMacroMesh):
        raise TypeError("Cartesian Darcy requires CartesianMacroMesh")
    degree = positive_int(degree, "degree")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 1)
    refinement = cartesian_refinement(local_refinement)
    skeleton = SkeletonSpace(cast(TriangleMesh, mesh)) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("Cartesian Darcy requires its own scalar skeleton")
    boundary, fixed = boundary_data(skeleton, dirichlet, neumann, order=order)
    provider = partial(
        local_equations,
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        refinement=refinement,
        permeability=permeability,
        source=source,
        order=order,
    )
    problem = MultiscaleProblem(
        Equation(0, np.r_[-boundary, np.zeros(len(mesh.cells))]),
        provider,
        range(len(mesh.cells)),
        skeleton.size,
        (1,) * len(mesh.cells),
        fixed=fixed,
    )
    pure_neumann = neumann is not None and set(neumann) == set(mesh.boundary_faces)
    return CartesianDarcyDefinition(
        problem,
        skeleton,
        permeability,
        source,
        degree,
        order,
        mean_pressure * float(mesh.areas.sum()) if pure_neumann else None,
    )


def recover_cartesian_darcy(
    definition: CartesianDarcyDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> QuadrilateralDarcySolution:
    """Interpret Qk pressure and sample its raw physical flux at fine-cell centers."""
    meshes = tuple(item[0] for item in system.local_metadata)
    record = QuadrilateralDarcySolution(
        definition.skeleton,
        meshes,
        solution.fields,
        (),
        solution,
        definition.permeability,
        definition.source,
        definition.degree,
        definition.quadrature_order,
    )
    flux = tuple(
        record.evaluate(cell, fine.points[fine.cells].mean(axis=1))[1]
        for cell, fine in enumerate(meshes)
    )
    return QuadrilateralDarcySolution(
        definition.skeleton,
        meshes,
        solution.fields,
        flux,
        solution,
        definition.permeability,
        definition.source,
        definition.degree,
        definition.quadrature_order,
    )
