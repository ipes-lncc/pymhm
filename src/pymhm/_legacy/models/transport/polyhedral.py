"""Native conservative RAD on star-shaped polyhedra with one polynomial per original face."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.transport.rad_3d import (
    RAD3DSolution,
    _coefficient_derivatives,
    tetra_rad_operators,
)
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space, tetrahedron_quadrature
from pymhm.fem.traces.normal import boundary_tangent_3d
from pymhm.fem.traces.polygon_3d import PolygonalSkeleton3D as PolygonalSkeleton3D
from pymhm.fem.traces.polygon_3d import polygonal_boundary_data as _boundary
from pymhm.fem.traces.polygon_3d import polygonal_face_quadrature as polygonal_face_quadrature
from pymhm.fem.traces.polygon_3d import polygonal_trace_coupling as polygonal_trace_coupling
from pymhm.materials.evaluation import scalar_values_3d
from pymhm.materials.evaluation import vector_values_3d as vector_values_3d
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import _dyadic
from pymhm.postprocessing.scalar_3d import PolyhedralRADSolution as PolyhedralRADSolution

_face_quadrature = polygonal_face_quadrature


def _tangent(mesh: PolyhedralMesh, velocity: Any, order: int, faces: Any) -> bool:
    """Check the shared literal polygonal normal-component compatibility rule."""
    return boundary_tangent_3d(PolygonalSkeleton3D(mesh), velocity, order, faces=faces)


@dataclass(frozen=True)
class _PolyhedralFactory:
    """Picklable complete tetrahedral assembly inside one star-shaped polyhedral macrocell."""

    mesh: PolyhedralMesh
    skeleton: PolygonalSkeleton3D
    refinement: int
    degree: int
    coefficients: dict[str, Any]
    coarse_space: str

    def __call__(self, cell: int) -> LocalAssembly:
        """Retain constants through physical means without changing original macroface traces."""
        fine = self.mesh.submesh(cell, self.refinement)
        matrix, load, moments, pure = tetra_rad_operators(fine, self.degree, **self.coefficients)
        coupling = polygonal_trace_coupling(self.mesh, cell, fine, self.skeleton, self.degree)
        constant = np.ones((len(load), 1))
        retained = {"kernel": constant} if pure else {"coarse_basis": constant}
        constraints = moments[:, None]
        if self.coarse_space == "kernel" and not pure:
            bary, _ = tetrahedron_quadrature(self.coefficients["order"])
            points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(-1, 3)
            kernel = (
                not np.any(scalar_values_3d(self.coefficients["reaction"], points))
                and not np.any(scalar_values_3d(self.coefficients["velocity_divergence"], points))
                and _tangent(
                    self.mesh,
                    self.coefficients["velocity"],
                    self.coefficients["order"],
                    self.mesh.cell_faces[cell],
                )
            )
            retained = {"kernel": constant} if kernel else {}
            if not kernel:
                constraints = np.empty((len(load), 0))
        problem = LocalProblem(
            matrix,
            coupling,
            load,
            self.skeleton.cell_dofs(cell),
            constraints=constraints,
            **retained,
        )
        return LocalAssembly(problem, (fine, moments))


def solve_polyhedral_rad(
    mesh: PolyhedralMesh,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: PolygonalSkeleton3D | None = None,
    local_refinement: int = 1,
    degree: int = 4,
    quadrature_order: int = 6,
    velocity_divergence: Any = None,
    diffusion_divergence: Any = None,
    stabilization: str = "galerkin",
    mean_value: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    coarse_space: Literal["constants", "kernel"] = "constants",
    global_refinement_precision: Literal["double", "extended"] = "double",
) -> PolyhedralRADSolution:
    """Solve conservative RAD with Pk local fields and original polygonal P0/P1 traces.

    Natural data prescribe outward ``(-K grad(u)+beta*u/2).n``; other exterior
    faces carry weak Dirichlet moments. Every macrocell is star-shaped and triangulated
    internally without changing its skeleton. The constant is retained by default
    for stability near singular local operators. ``coarse_space='kernel'`` retains
    only exact constant null modes. Pure Robin problems admit a mean only for zero
    reaction, divergence-free velocity tangent to the entire exterior.
    """
    if not isinstance(mesh, PolyhedralMesh):
        raise TypeError("polyhedral RAD requires PolyhedralMesh")
    if coarse_space not in ("constants", "kernel"):
        raise ValueError("coarse_space must be constants or kernel")
    refinement = _dyadic(local_refinement, "local_refinement")
    tetra_nodal_space(mesh.submesh(0), degree)
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    if not np.isfinite(mean_value):
        raise ValueError("mean_value must be finite")
    div_beta, div_tensor = _coefficient_derivatives(
        mesh.points, diffusion, velocity, velocity_divergence, diffusion_divergence, stabilization
    )
    skeleton = PolygonalSkeleton3D(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to the supplied polyhedral mesh")
    natural = {} if neumann is None else dict(neumann)
    boundary, fixed = _boundary(skeleton, dirichlet, natural, order)
    coefficients = dict(
        diffusion=diffusion,
        velocity=velocity,
        reaction=reaction,
        source=source,
        velocity_divergence=div_beta,
        diffusion_divergence=div_tensor,
        stabilization=stabilization,
        order=order,
    )
    factory = _PolyhedralFactory(mesh, skeleton, refinement, degree, coefficients, coarse_space)
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    constraints = []
    gauge = False
    if set(natural) == set(mesh.boundary_faces):
        bary, _ = tetrahedron_quadrature(order)
        pure = True
        for fine, _ in system.local_metadata:
            points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(-1, 3)
            pure = (
                pure
                and not np.any(scalar_values_3d(reaction, points))
                and not np.any(scalar_values_3d(div_beta, points))
            )
        gauge = pure and _tangent(mesh, velocity, order, mesh.boundary_faces)
    if gauge:
        constraints.append(
            system.mean_constraint(
                [data[1] for data in system.local_metadata], mean_value * float(mesh.volumes.sum())
            )
        )
    elif mean_value != 0:
        raise ValueError(
            "mean_value requires all-Robin zero-reaction divergence-free tangential data"
        )
    result = system.solve(
        solver=solver,
        fixed=fixed,
        constraints=constraints,
        refinement_precision=global_refinement_precision,
    )
    field = RAD3DSolution(
        tuple(data[0] for data in system.local_metadata),
        result.fields,
        degree,
        diffusion,
        velocity,
        result,
        residual=result.residual,
    )
    return PolyhedralRADSolution(field, skeleton, result)
