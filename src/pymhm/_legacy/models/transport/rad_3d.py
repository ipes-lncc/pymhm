"""Conservative reaction-advection-diffusion on tetrahedra with polynomial face traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.tetrahedron import (
    tetra_boundary_nodes,
    tetra_nodal_space,
    tetrahedron_quadrature,
)
from pymhm.fem.scalar.transport_3d import coefficient_derivatives_3d as _coefficient_derivatives
from pymhm.fem.scalar.transport_3d import tetra_transport_operators as tetra_rad_operators
from pymhm.fem.traces.normal import boundary_tangent_3d as _boundary_tangent_3d
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, _boundary, tetra_trace_coupling
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import scalar_values_3d
from pymhm.materials.evaluation import (
    vector_values_3d as vector_values_3d,
)
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.postprocessing.solutions import RAD3DSolution as RAD3DSolution


@dataclass(frozen=True)
class _RADFactory:
    """Picklable local assembly for independent three-dimensional macro problems."""

    mesh: TetraMesh
    skeleton: TriangularSkeleton
    refinement: int
    degree: int
    coefficients: dict[str, Any]
    coarse_space: Literal["constants", "kernel"] = "constants"

    def __call__(self, cell: int) -> LocalAssembly:
        """Retain the physical constant, including nonzero or arbitrarily small transport."""
        fine = self.mesh.submesh(cell, self.refinement)
        matrix, load, moments, pure = tetra_rad_operators(fine, self.degree, **self.coefficients)
        coupling = tetra_trace_coupling(self.mesh, cell, fine, self.skeleton, self.degree)
        constant = np.ones((len(load), 1))
        retained = {"kernel": constant} if pure else {"coarse_basis": constant}
        constraints = moments[:, None]
        if self.coarse_space == "kernel" and not pure:
            bary, _ = tetrahedron_quadrature(self.coefficients["order"])
            points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(-1, 3)
            kernel = (
                not np.any(scalar_values_3d(self.coefficients["reaction"], points))
                and not np.any(scalar_values_3d(self.coefficients["velocity_divergence"], points))
                and _boundary_tangent_3d(
                    TriangularSkeleton(fine),
                    self.coefficients["velocity"],
                    self.coefficients["order"],
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


def solve_rad_3d(
    mesh: TetraMesh,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: TriangularSkeleton | None = None,
    local_refinement: int = 2,
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
) -> RAD3DSolution:
    """Solve conservative RAD using continuous P1–P4 local tetrahedral fields.

    The default skeleton has an independent P1 polynomial on each macroface.
    ``neumann`` prescribes outward ``(-K grad(u)+beta*u/2).n``. Other exterior
    faces receive weak Dirichlet moments. Variable velocity requires its
    divergence; SUPG with variable diffusion also requires ``div(K)``. A mean
    is permitted only with all-Robin data, zero reaction, divergence-free
    exterior-tangent velocity and a representable global constant mode.
    ``coarse_space='kernel'`` retains only genuine constant null modes, giving
    the selective mixed/primal formulation; the default retains every constant
    to handle near-singular local operators without a numerical rank threshold.
    """
    refinement = _dyadic(local_refinement, "local_refinement")
    if coarse_space not in ("constants", "kernel"):
        raise ValueError("coarse_space must be constants or kernel")
    tetra_nodal_space(mesh, degree)
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    div_beta, div_tensor = _coefficient_derivatives(
        mesh.points,
        diffusion,
        velocity,
        velocity_divergence,
        diffusion_divergence,
        stabilization,
    )
    if not np.isfinite(mean_value):
        raise ValueError("mean_value must be finite")
    skeleton = TriangularSkeleton(mesh, degree=1) if skeleton is None else skeleton
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to the supplied tetrahedral mesh")
    if np.any(skeleton.subdivisions > refinement):
        raise ValueError("local refinement must resolve every skeleton subdivision")
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
    factory = _RADFactory(mesh, skeleton, refinement, degree, coefficients, coarse_space)
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    constraints = []
    bary, _ = tetrahedron_quadrature(order)
    all_points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 3)
    gauge = (
        set(natural) == set(mesh.boundary_faces)
        and not np.any(scalar_values_3d(reaction, all_points))
        and not np.any(scalar_values_3d(div_beta, all_points))
        and _boundary_tangent_3d(skeleton, velocity, order)
    )
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
    return RAD3DSolution(
        tuple(data[0] for data in system.local_metadata),
        result.fields,
        degree,
        diffusion,
        velocity,
        result,
        skeleton,
        result.residual,
    )


def solve_rad_3d_conforming(
    mesh: TetraMesh,
    *,
    degree: int = 2,
    dirichlet: Any = 0.0,
    solver: str = "scipy",
    **coefficients: Any,
) -> RAD3DSolution:
    """Solve the classical continuous-Pk Galerkin/SUPG problem with strong exterior Dirichlet.

    This separate global assembly has no macroface restriction or local
    condensation. Boundary elimination lifts all prescribed nodal values into
    the original equations. Its discretization error must be checked by refinement.
    """
    matrix, load, _, _ = tetra_rad_operators(mesh, degree, **coefficients)
    dofs, nodes = tetra_nodal_space(mesh, degree)
    fixed = tetra_boundary_nodes(mesh, degree)
    free = np.setdiff1d(np.arange(len(nodes)), fixed)
    values = np.zeros(len(nodes))
    values[fixed] = scalar_values_3d(dirichlet, nodes[fixed])
    rhs = (load - matrix @ values)[free]
    if len(free):
        values[free] = solve_linear(matrix[free][:, free], rhs, solver=solver)
    residual = np.linalg.norm((matrix @ values - load)[free]) / max(
        np.linalg.norm(rhs) + np.linalg.norm(abs(matrix[free][:, free]) @ abs(values[free])),
        np.finfo(float).tiny,
    )
    return RAD3DSolution(
        (mesh,),
        (values,),
        degree,
        coefficients.get("diffusion", 1.0),
        coefficients.get("velocity", (0.0, 0.0, 0.0)),
        residual=float(residual),
    )
