"""Tetrahedral Pk Darcy forms with explicit trace, boundary and mean conventions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from pymhm._legacy.models.darcy.primal_3d import Darcy3DSolution
from pymhm.core.contracts import HybridSolution
from pymhm.core.equations import Equation, LocalEquations, columns, rows
from pymhm.core.multiscale import MultiscaleProblem, MultiscaleSystem
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space, tetra_operators
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, _boundary, tetra_trace_coupling
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.meshes.validation import validate_tetra_submesh


@dataclass(frozen=True)
class TetrahedralDarcyDefinition:
    """Declared tetrahedral equations and physical field interpretation data."""

    problem: MultiscaleProblem[int]
    skeleton: TriangularSkeleton
    permeability: Any
    source: Any
    degree: int
    quadrature_order: int
    pressure_integral: float | None


def local_equations(
    cell: int,
    *,
    mesh: TetraMesh,
    skeleton: TriangularSkeleton,
    degree: int,
    refinement: int,
    local_meshes: tuple[TetraMesh, ...] | None,
    permeability: Any,
    source: Any,
    order: int,
) -> LocalEquations:
    """Declare tetrahedral energy, source and two oriented trace incidence forms.

    Local coordinates retain the Pk nodal ordering. The constant pressure kernel
    uses physical integral moments without volume normalization. C=-B.T imposes
    global weak pressure continuity; the shared physical flux has fixed global
    normal orientation independently of raw reconstructed pressure gradients.
    """
    fine = mesh.submesh(cell, refinement) if local_meshes is None else local_meshes[cell]
    matrix, mass, load = tetra_operators(
        fine, degree, diffusion=permeability, source=source, order=order
    )
    coupling = tetra_trace_coupling(mesh, cell, fine, skeleton, degree)
    kernel = np.ones((len(load), 1))
    mean = np.asarray(mass @ kernel).ravel()
    return LocalEquations(
        matrix,
        load,
        columns(*coupling.T),
        rows(*(-coupling.T)),
        skeleton.cell_dofs(cell),
        kernel=kernel,
        moments=mean[:, None],
        metadata=(fine, mean),
    )


def define_tetrahedral_darcy(
    mesh: TetraMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: TriangularSkeleton | None = None,
    local_refinement: int = 2,
    local_meshes: tuple[TetraMesh, ...] | None = None,
    degree: int = 2,
    quadrature_order: int = 5,
    mean_pressure: float = 0.0,
) -> TetrahedralDarcyDefinition:
    """Declare Pk tetrahedral locals and weak pressure boundary data globally.

    Built-in refinements are dyadic and resolve every skeleton subdivision.
    Explicit fitted meshes are validated as conforming macrocell partitions.
    Neumann data are outward physical flux; all other exterior faces impose
    pressure weakly. The physical mean gauge applies only to pure Neumann data.
    """
    refinement = _dyadic(local_refinement, "local_refinement")
    tetra_nodal_space(mesh, degree)
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    skeleton = TriangularSkeleton(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh:
        raise ValueError("tetrahedral Darcy requires its own skeleton")
    if local_meshes is not None:
        if len(local_meshes) != len(mesh.cells):
            raise ValueError("provide one local tetrahedral mesh per macrocell")
        for cell, fine in enumerate(local_meshes):
            validate_tetra_submesh(mesh, cell, fine)
    elif np.any(skeleton.subdivisions > refinement):
        raise ValueError("local refinement must resolve skeleton subdivisions")
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    neumann = {} if neumann is None else neumann
    boundary, fixed = _boundary(skeleton, dirichlet, neumann, order)
    provider = partial(
        local_equations,
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        refinement=refinement,
        local_meshes=local_meshes,
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
    return TetrahedralDarcyDefinition(
        problem,
        skeleton,
        permeability,
        source,
        degree,
        order,
        mean_pressure * float(mesh.volumes.sum())
        if len(neumann) == len(mesh.boundary_faces)
        else None,
    )


def recover_tetrahedral_darcy(
    definition: TetrahedralDarcyDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> Darcy3DSolution:
    """Interpret the executed Pk coefficients through the shared 3D field record."""
    return Darcy3DSolution(
        definition.skeleton,
        tuple(item[0] for item in system.local_metadata),
        solution.fields,
        solution,
        definition.degree,
        definition.permeability,
        definition.source,
        definition.quadrature_order,
    )
