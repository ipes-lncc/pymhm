"""Cartesian quadrilateral MHM with continuous tensor-product Qk local pressures.

The skeleton uses the same signed physical normal flux as triangular Darcy.
Only axis-aligned Cartesian rectangles are supported; curved and bilinear
non-affine quadrilaterals require a different geometric transformation.
"""

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.quadrilateral import (
    _cardinal_values as _cardinal_values,
)
from pymhm.fem.scalar.quadrilateral import (
    _cardinals as _cardinals,
)
from pymhm.fem.scalar.quadrilateral import (
    _cartesian_rectangle_quadrature as _cartesian_rectangle_quadrature,
)
from pymhm.fem.scalar.quadrilateral import (
    _grid_resolves_material as _grid_resolves_material,
)
from pymhm.fem.scalar.quadrilateral import (
    qk_basis as qk_basis,
)
from pymhm.fem.scalar.quadrilateral import (
    qk_space as qk_space,
)
from pymhm.fem.scalar.quadrilateral import (
    quadrilateral_operators as quadrilateral_operators,
)
from pymhm.fem.scalar.quadrilateral import (
    quadrilateral_quadrature as quadrilateral_quadrature,
)
from pymhm.fem.scalar.quadrilateral import (
    quadrilateral_trace_coupling as quadrilateral_trace_coupling,
)
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.cartesian import (
    CartesianMacroMesh as CartesianMacroMesh,
)
from pymhm.meshes.cartesian import (
    _refinement as _refinement,
)
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import QuadrilateralDarcySolution as QuadrilateralDarcySolution


@dataclass(frozen=True)
class _QuadTask:
    """Picklable geometry and coefficients for one complete worker-local solve."""

    mesh: CartesianMacroMesh
    cell: int
    refinement: tuple[int, int]
    skeleton: SkeletonSpace
    degree: int
    permeability: Any
    source: Any
    order: int


def _assemble_quad(task: _QuadTask) -> LocalAssembly:
    """Build one original Qk Neumann problem and return reconstruction metadata."""
    fine = task.mesh.submesh(task.cell, task.refinement)
    matrix, mass, load = quadrilateral_operators(
        fine, task.degree, permeability=task.permeability, source=task.source, order=task.order
    )
    coupling = quadrilateral_trace_coupling(task.mesh, task.cell, fine, task.skeleton, task.degree)
    kernel = np.ones((matrix.shape[0], 1))
    constraints = mass @ kernel
    return LocalAssembly(
        LocalProblem(
            matrix, coupling, load, task.skeleton.cell_dofs(task.cell), kernel, constraints
        ),
        (fine, constraints[:, 0]),
    )


def solve_darcy_quadrilateral(
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
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> QuadrilateralDarcySolution:
    """Solve Darcy MHM on Cartesian macrorectangles with continuous Qk locals.

    Dirichlet pressure is imposed weakly on faces not listed in ``neumann``;
    Neumann values are outward physical normal fluxes. Pure Neumann problems
    require compatibility and impose the physical mean pressure. The local
    refinement may differ between x and y, independently of trace segments.
    Assembly and condensation both execute inside the selected CPU workers.
    Process execution requires picklable coefficient callbacks and the usual
    guarded script entry point. Sparse local operators reuse one factorization
    for every source and trace right-hand side.
    """
    if not isinstance(mesh, CartesianMacroMesh):
        raise TypeError("quadrilateral Darcy requires CartesianMacroMesh")
    degree = positive_int(degree, "degree")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 1)
    refinement = _refinement(local_refinement)
    skeleton = SkeletonSpace(cast(TriangleMesh, mesh)) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("quadrilateral Darcy requires a scalar skeleton on the supplied mesh")
    boundary, fixed = boundary_data(skeleton, dirichlet, neumann, order=order)
    tasks = [
        _QuadTask(mesh, cell, refinement, skeleton, degree, permeability, source, order)
        for cell in range(len(mesh.cells))
    ]
    system = HybridSystem.from_local_factory(
        _assemble_quad,
        tasks,
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    local_meshes, means = zip(*system.local_metadata, strict=True)
    pure_neumann = neumann is not None and set(neumann) == set(mesh.boundary_faces)
    constraints = (
        [system.mean_constraint(means, mean_pressure * float(np.sum(mesh.areas)))]
        if pure_neumann
        else None
    )
    hybrid = system.solve(solver=solver, fixed=fixed, constraints=constraints)
    solution = QuadrilateralDarcySolution(
        skeleton,
        tuple(local_meshes),
        hybrid.fields,
        (),
        hybrid,
        permeability,
        source,
        degree,
        order,
    )
    flux = tuple(
        solution.evaluate(cell, fine.points[fine.cells].mean(axis=1))[1]
        for cell, fine in enumerate(local_meshes)
    )
    return QuadrilateralDarcySolution(
        skeleton,
        tuple(local_meshes),
        hybrid.fields,
        flux,
        hybrid,
        permeability,
        source,
        degree,
        order,
    )
