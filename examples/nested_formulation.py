"""Declare spatially nested diffusion equations using the public variational API.

Continuous local Q2 fields use signed P1 normal-flux moments. A parent face
has two polynomial segments, exactly representable on the child boundary.
The local matrix at the second scale is another declared MultiscaleProblem.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from pymhm import Equation, LocalEquations, MultiscaleProblem, NestedEquations
from pymhm.core.nested import nested_trace_map
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.quadrilateral import (
    quadrilateral_operators,
    quadrilateral_trace_coupling,
)
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.nodal import nodal_field


@dataclass(frozen=True)
class DiffusionDiscretization:
    """Picklable geometry, integration and physical data for continuous Qk leaves."""

    mesh: CartesianMacroMesh
    skeleton: SkeletonSpace
    source: Any = 0.0
    permeability: Any = 1.0
    degree: int = 2
    refinement: int = 2
    order: int = 6


def leaf_equations(cell: int, *, data: DiffusionDiscretization) -> LocalEquations:
    """Write ``(K grad p,grad v)+<lambda,v>=(f,v)`` and ``-<p,mu>=g``.

    The declared constant kernel remains a global amplitude. Its integral
    moment selects the local complement; the signed coupling uses the fixed
    macroface normal, including the incident cell's outward orientation.
    """
    fine = data.mesh.submesh(cell, data.refinement)
    a, mass, load = quadrilateral_operators(
        fine,
        data.degree,
        permeability=data.permeability,
        source=data.source,
        order=data.order,
    )
    b = quadrilateral_trace_coupling(data.mesh, cell, fine, data.skeleton, data.degree)
    kernel = np.ones((a.shape[0], 1))
    return LocalEquations(
        a,
        load,
        b,
        -b.T,
        data.skeleton.cell_dofs(cell),
        kernel=kernel,
        moments=mass @ kernel,
        metadata=(fine, np.asarray(mass @ kernel)[:, 0]),
        field_data=(nodal_field("pressure", fine, data.degree),),
    )


def child_problem(data: DiffusionDiscretization) -> MultiscaleProblem[int]:
    """Declare a child with no exterior pressure or gauge imposed by that scale."""
    return MultiscaleProblem(
        Equation(0, 0),
        partial(leaf_equations, data=data),
        range(len(data.mesh.cells)),
        data.skeleton.size,
        (1,) * len(data.mesh.cells),
    )


def parent_equations(cell: int, *, data: DiffusionDiscretization) -> NestedEquations:
    """Restrict a parent normal trace and retain the child's physical constant.

    The child boundary uses two straight segments of each parent edge. The
    restriction is exact in these declared P1 face spaces. The physical moment
    row is the cell-volume weighted sum of the child constant amplitudes:
    local source and trace complements have zero declared integral moments.
    """
    mesh = data.mesh.submesh(cell, 2)
    faces = tuple(FaceSpace.uniform(1) for _ in mesh.faces)
    skeleton = SkeletonSpace(mesh, faces)
    child = child_problem(
        DiffusionDiscretization(
            mesh,
            skeleton,
            data.source,
            data.permeability,
            data.degree,
            data.refinement,
            data.order,
        )
    )
    selected, transform = nested_trace_map(data.skeleton, cell, skeleton)
    kernel = np.r_[np.zeros(skeleton.size), np.ones(len(mesh.cells))][:, None]
    moments = np.r_[np.zeros(skeleton.size), mesh.areas][:, None]
    return NestedEquations(
        child,
        selected,
        transform,
        data.skeleton.cell_dofs(cell),
        kernel=kernel,
        moments=moments,
        metadata=mesh,
    )


def nested_problem(
    subdivisions: int = 1,
    *,
    source: Any = 0.0,
    permeability: Any = 1.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
) -> tuple[MultiscaleProblem[int], DiffusionDiscretization]:
    """Declare a two-scale spatial problem with weak pressure or physical outflow.

    Neumann values represent the physical Darcy outflow ``-K grad(p).n``.
    For a fully Neumann problem the caller supplies its physical mean through
    the assembled system's moment functional before solving.
    """
    mesh = CartesianMacroMesh(subdivisions)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces))
    data = DiffusionDiscretization(mesh, skeleton, source, permeability)
    boundary, fixed = boundary_data(skeleton, dirichlet, neumann, order=8)
    problem = MultiscaleProblem(
        Equation(0, -np.r_[boundary, np.zeros(len(mesh.cells))]),
        partial(parent_equations, data=data),
        range(len(mesh.cells)),
        skeleton.size,
        (1,) * len(mesh.cells),
        fixed=fixed,
    )
    return problem, data
