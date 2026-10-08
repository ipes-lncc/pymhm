"""Executable 3D transport definitions using only public mathematical operations.

The adjacent scalar_3d module declares volume, trace, retained-mode and physical
mean equations. These application compositions choose geometry-specific pairings,
then use generic assembly/solve. No prepared physical solver is called.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from pymhm import ExecutionConfig, SolverConfig, assemble
from pymhm.fem.scalar.tetrahedron import tetra_boundary_nodes, tetra_nodal_space
from pymhm.fem.scalar.transport_3d import tetra_transport_operators
from pymhm.fem.traces.polygon_3d import (
    PolygonalSkeleton3D,
    polygonal_boundary_data,
    polygonal_trace_coupling,
)
from pymhm.fem.traces.triangle_3d import (
    TriangularSkeleton,
    tetra_boundary_data,
    tetra_trace_coupling,
)
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import scalar_values_3d
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.postprocessing.scalar_3d import PolyhedralRADSolution
from pymhm.postprocessing.solutions import RAD3DSolution

from .scalar_3d import (
    Scalar3DDefinition,
    Scalar3DDiscretization,
    define_scalar_3d,
    recover_transport_3d,
    scalar_constraints_3d,
)


def define_transport_3d(
    mesh: TetraMesh | PolyhedralMesh,
    *,
    skeleton: Any = None,
    local_refinement: int | None = None,
    degree: int = 4,
    quadrature_order: int = 6,
    **coefficients: Any,
) -> Scalar3DDefinition:
    """Declare the same conservative four-block form on either original macro geometry.

    Tetrahedral defaults use local refinement 2 and independent triangular P1
    traces; polyhedral defaults use refinement 1 and original polygonal P1 traces.
    All coefficient derivatives, stabilization, natural data and physical means
    are explicit options of define_scalar_3d. Geometry creates no method dispatch.
    """
    pairing: Callable[..., Any]
    boundary: Callable[..., Any]
    if isinstance(mesh, PolyhedralMesh):
        space = PolygonalSkeleton3D(mesh) if skeleton is None else skeleton
        refinement = 1 if local_refinement is None else local_refinement
        pairing, boundary = polygonal_trace_coupling, polygonal_boundary_data
    elif isinstance(mesh, TetraMesh):
        space = TriangularSkeleton(mesh, degree=1) if skeleton is None else skeleton
        refinement = 2 if local_refinement is None else local_refinement
        pairing, boundary = tetra_trace_coupling, tetra_boundary_data
    else:
        raise TypeError("3D transport requires a tetrahedral or polyhedral macro mesh")
    data = Scalar3DDiscretization(
        mesh, space, pairing, boundary, degree, refinement, quadrature_order
    )
    return define_scalar_3d(data, **coefficients)


def transport_3d(
    mesh: TetraMesh | PolyhedralMesh,
    *,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    local_refinement_precision: Literal["double", "extended"] = "double",
    global_refinement_precision: Literal["double", "extended"] = "double",
    **coefficients: Any,
) -> RAD3DSolution | PolyhedralRADSolution:
    """Execute user-defined volume/trace equations with declared scheduling and precision.

    The returned field exposes -K grad(u)+beta*u. The fixed natural multiplier
    differs by beta*u/2. Local constants use literal physical moments, including
    arbitrarily small nonzero transport; a gauge uses the complete domain integral.
    """
    definition = define_transport_3d(mesh, **coefficients)
    system = assemble(
        definition.problem,
        execution=ExecutionConfig(backend=backend, workers=workers),
        solvers=SolverConfig(
            local_solver=local_solver,
            global_solver=solver,
            local_refinement_precision=local_refinement_precision,
            global_refinement_precision=global_refinement_precision,
        ),
    )
    result = system.solve(constraints=scalar_constraints_3d(definition, system))
    return recover_transport_3d(definition, system, result)


def conforming_transport_3d(
    mesh: TetraMesh,
    *,
    degree: int = 2,
    dirichlet: Any = 0.0,
    solver: str = "scipy",
    **coefficients: Any,
) -> RAD3DSolution:
    """Assemble an independent conforming continuous-Pk Galerkin/SUPG reference.

    This original global matrix has strong exterior nodal values and no skeleton
    restriction or condensation. It uses the same declared operator, physical
    material, source and quadrature. Refinement must establish its own accuracy;
    sharing those inputs does not imply equality of different discrete spaces.
    """
    matrix, load, _, _ = tetra_transport_operators(mesh, degree, **coefficients)
    _, nodes = tetra_nodal_space(mesh, degree)
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
