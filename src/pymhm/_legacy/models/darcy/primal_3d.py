"""Three-dimensional primal MHM Darcy with positive-degree Pk tetrahedral local spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm.core.contracts import HybridSolution, LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.core.validation import real_array as _real
from pymhm.fem.scalar.tetrahedron import (
    tetra_nodal_space,
    tetra_operators,
    tetra_tabulate,
    tetrahedron_quadrature,
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
from pymhm.materials.evaluation import scalar_values_3d, tensor_values_3d
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.meshes.validation import validate_tetra_submesh


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


@dataclass(frozen=True)
class Darcy3DSolution:
    """Broken tetrahedral pressure and physical gradient flux with conservative trace.

    ``flux`` evaluates ``-K grad(p)`` and is generally not H(div)-conforming.
    Macro conservation refers to ``hybrid.trace``, not this raw gradient field.
    """

    skeleton: TriangularSkeleton
    local_meshes: tuple[TetraMesh, ...]
    pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    permeability: Any
    source: Any
    quadrature_order: int

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate pressure and physical flux at common fine-cell barycentric points."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.local_meshes):
            raise ValueError("cell outside local meshes")
        fine = self.local_meshes[cell]
        dofs, _, basis, gradient = tetra_tabulate(fine, self.degree, bary)
        pressure = self.pressure[cell][dofs] @ basis.T
        grad = np.einsum("ti,tqij->tqj", self.pressure[cell][dofs], gradient)
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        diffusion = tensor_values_3d(self.permeability, points.reshape(-1, 3)).reshape(
            *points.shape[:2], 3, 3
        )
        return pressure, -np.einsum("tqij,tqj->tqi", diffusion, grad)

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the broken pressure error with independently selected positive quadrature."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            values = self.evaluate(cell, bary)[0]
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            expected = scalar_values_3d(exact, points.reshape(-1, 3)).reshape(values.shape)
            total += float(fine.volumes @ ((values - expected) ** 2 @ weights))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the full physical flux, including P2 gradient and variable diffusion."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            values = self.evaluate(cell, bary)[1]
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            expected = _real(
                exact(points.reshape(-1, 3)) if callable(exact) else exact, "exact flux"
            )
            try:
                expected = np.broadcast_to(expected, (len(points.reshape(-1, 3)), 3)).reshape(
                    values.shape
                )
            except ValueError as exc:
                raise ValueError("exact flux must return three components per point") from exc
            total += float(fine.volumes @ (np.sum((values - expected) ** 2, axis=2) @ weights))
        return float(np.sqrt(total))

    def conservation_residuals(self, order: int | None = None) -> FloatArray:
        """Return outward skeletal flux minus integrated source in every macrocell."""
        mesh = self.skeleton.mesh
        bary, weights = tetrahedron_quadrature(self.quadrature_order if order is None else order)
        residuals = []
        for cell, fine in enumerate(self.local_meshes):
            total = sum(
                mesh.signs[cell, side]
                * mesh.areas[face]
                * (
                    self.skeleton.face_weights(int(face))
                    @ np.array(
                        [
                            self.hybrid.trace[
                                self.skeleton.subtriangle_dofs(int(face), segment)
                            ].mean()
                            for segment in range(len(self.skeleton.face_partition(int(face))))
                        ]
                    )
                )
                for side, face in enumerate(mesh.cell_faces[cell])
            )
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            load = scalar_values_3d(self.source, points.reshape(-1, 3)).reshape(len(fine.cells), -1)
            residuals.append(total - float(fine.volumes @ (load @ weights)))
        return np.asarray(residuals)


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
