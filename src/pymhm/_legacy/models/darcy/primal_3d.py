"""Three-dimensional primal MHM Darcy with positive-degree Pk tetrahedral local spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.tetrahedron import (
    tetra_nodal_space,
    tetra_operators,
)
from pymhm.fem.traces.triangle_3d import (
    TriangularSkeleton as TriangularSkeleton,
)
from pymhm.fem.traces.triangle_3d import (
    _boundary as _boundary,
)
from pymhm.fem.traces.triangle_3d import (
    tetra_trace_coupling as tetra_trace_coupling,
)
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.meshes.validation import validate_tetra_submesh
from pymhm.postprocessing.solutions import Darcy3DSolution as Darcy3DSolution


@dataclass(frozen=True)
class _DarcyFactory:
    """Picklable complete local Pk assembly for the serial/thread/spawn factory map."""

    mesh: TetraMesh
    skeleton: TriangularSkeleton
    refinement: int
    degree: int
    permeability: Any
    source: Any
    order: int
    local_meshes: tuple[TetraMesh, ...] | None = None

    def __call__(self, cell: int) -> LocalAssembly:
        """Build one local Neumann operator, physical mean and oriented trace form."""
        fine = (
            self.mesh.submesh(cell, self.refinement)
            if self.local_meshes is None
            else self.local_meshes[cell]
        )
        matrix, mass, load = tetra_operators(
            fine, self.degree, diffusion=self.permeability, source=self.source, order=self.order
        )
        coupling = tetra_trace_coupling(self.mesh, cell, fine, self.skeleton, self.degree)
        kernel = np.ones((len(load), 1))
        mean = np.asarray(mass @ kernel).ravel()
        problem = LocalProblem(
            matrix,
            coupling,
            load,
            self.skeleton.cell_dofs(cell),
            kernel=kernel,
            constraints=mean[:, None],
        )
        return LocalAssembly(problem, (fine, mean))


def solve_darcy_3d(
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
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> Darcy3DSolution:
    """Solve div(q)=f, q=-K grad(p), using weak pressure and outward normal flux data.

    Local refinements and face subdivisions are powers of two with aligned
    partitions unless ``local_meshes`` supplies a conforming tetrahedral partition
    of each macrocell. Explicit meshes must cover each macrocell exactly, and each
    fine exterior triangle must lie in one active skeletal subtriangle. Every
    unspecified exterior face receives ``dirichlet`` pressure.
    Pure Neumann data impose one physical pressure mean and must be compatible.
    Local Pk spaces accept positive polynomial degrees; skeletal modes have an
    independently selected polynomial degree on each macroface. Skeletal
    Bernstein coefficients may be discontinuous or C0 between its subtriangles;
    different macrofaces retain independent coordinates, including shared edges.
    Nonrepresentable trace enrichment is rejected by
    the condensed rank check, without diagonal regularization.
    """
    refinement = _dyadic(local_refinement, "local_refinement")
    tetra_nodal_space(mesh, degree)
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    skeleton = TriangularSkeleton(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to the supplied tetrahedral mesh")
    if local_meshes is not None:
        if len(local_meshes) != len(mesh.cells):
            raise ValueError("local_meshes must contain one mesh per macrocell")
        for cell, fine in enumerate(local_meshes):
            validate_tetra_submesh(mesh, cell, fine)
    elif np.any(skeleton.subdivisions > refinement):
        raise ValueError("local refinement must resolve every skeleton subdivision")
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    neumann = {} if neumann is None else neumann
    boundary, fixed = _boundary(skeleton, dirichlet, neumann, order)
    factory = _DarcyFactory(
        mesh, skeleton, refinement, degree, permeability, source, order, local_meshes
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    constraints = []
    if len(neumann) == len(mesh.boundary_faces):
        constraints = [
            system.mean_constraint(
                [metadata[1] for metadata in system.local_metadata],
                float(mean_pressure * mesh.volumes.sum()),
            )
        ]
    solution = system.solve(fixed=fixed, constraints=constraints, solver=solver)
    return Darcy3DSolution(
        skeleton,
        tuple(metadata[0] for metadata in system.local_metadata),
        solution.fields,
        solution,
        degree,
        permeability,
        source,
        order,
    )
