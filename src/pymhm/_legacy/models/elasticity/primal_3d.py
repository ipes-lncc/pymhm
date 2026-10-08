"""Primal tetrahedral MHM elasticity with general stiffness and six rigid modes.

Kelvin ordering is (xx, yy, zz, sqrt(2) yz, sqrt(2) xz, sqrt(2) xy).
This displacement formulation is not claimed to be uniformly locking-free.
"""

from functools import partial
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.fem.vector.elasticity_3d import KELVIN_BASIS_3D
from pymhm.fem.vector.elasticity_3d import _component as _component
from pymhm.fem.vector.elasticity_3d import rigid_modes_3d as rigid_modes_3d
from pymhm.fem.vector.elasticity_3d import vector_boundary_data_3d as _boundary_vector
from pymhm.fem.vector.primal_3d import tetra_strain_operators
from pymhm.materials.elasticity import constitutive_values_3d as constitutive_values_3d
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.postprocessing.primal_elasticity_3d import Elasticity3DSolution as Elasticity3DSolution


def _local(
    cell: int,
    *,
    mesh: TetraMesh,
    skeleton: TriangularSkeleton,
    degree: int,
    refinement: int,
    constitutive: Any,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
) -> LocalAssembly:
    """Assemble Kelvin energy and force with six physical L2 rigid-motion constraints."""
    fine = mesh.submesh(cell, refinement)
    forms = tetra_strain_operators(
        fine,
        degree,
        constitutive=constitutive,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=source,
        order=order,
    )
    matrix, mass, load, nodes = forms.matrix, forms.mass, forms.load, forms.nodes
    size = 3 * len(nodes)
    coupling = np.kron(tetra_trace_coupling(mesh, cell, fine, skeleton, degree), np.eye(3))
    center = mesh.points[mesh.cells[cell]].mean(axis=0)
    rigid = rigid_modes_3d(nodes, center).reshape(size, 6)
    moments = sparse.kron(mass, sparse.eye(3)) @ rigid
    global_center = mesh.volumes @ mesh.points[mesh.cells].mean(axis=1) / mesh.volumes.sum()
    global_moments = sparse.kron(mass, sparse.eye(3)) @ rigid_modes_3d(
        nodes, global_center
    ).reshape(size, 6)
    indices = (3 * skeleton.cell_dofs(cell)[:, None] + np.arange(3)).ravel()
    return LocalAssembly(
        LocalProblem(matrix, coupling, load, indices, rigid, moments), (fine, global_moments)
    )


def solve_elasticity_3d(
    mesh: TetraMesh,
    *,
    constitutive: Any = None,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    source: Any = (0.0, 0.0, 0.0),
    dirichlet: Any = (0.0, 0.0, 0.0),
    traction: dict[int, Any] | None = None,
    skeleton: TriangularSkeleton | None = None,
    degree: int = 2,
    local_refinement: int = 2,
    quadrature_order: int = 6,
    rigid_moments: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    solver: str = "scipy",
    local_solver: str = "scipy",
    local_refinement_precision: Literal["double", "extended"] = "double",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> Elasticity3DSolution:
    """Solve primal 3D elasticity with P1–P4 locals, P1 face traces and six rigid modes.

    Tensor inputs use Kelvin (6,6) or Cartesian (3,3,3,3) coordinates with
    major/minor symmetry and sampled positive definiteness. Pure traction fixes
    six integrated displacement moments: translations then rotations e_i cross
    (x-centroid). Face data are physical outward Cauchy traction. Dyadic face
    subdivisions must align with local tetrahedral refinements. The P1 trace
    contract contains rigid-motion traces, but arbitrary local choices still
    face a numerical rank check. Material interfaces require adequate fitted
    discretization/integration; no cut-tetrahedron rule is implied. This primal
    method has no uniform incompressible-limit guarantee and requires finite lambda.
    ``local_refinement_precision="extended"`` explicitly accumulates local defect
    corrections in extended precision while retaining the original residual criterion.
    """
    refinement = _dyadic(local_refinement, "local_refinement")
    tetra_nodal_space(mesh, degree)
    order = positive_int(quadrature_order, "quadrature_order", degree + 2)
    skeleton = TriangularSkeleton(mesh, degree=1) if skeleton is None else skeleton
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to the tetrahedral mesh")
    if np.any(skeleton.degrees != 1):
        raise ValueError("3D elasticity requires P1 traces containing all rigid-motion traces")
    if np.any(skeleton.subdivisions > refinement):
        raise ValueError("local refinement must resolve every face subdivision")
    data = {} if traction is None else traction
    pure_traction = set(data) == set(mesh.boundary_faces)
    targets = np.asarray(rigid_moments)
    if pure_traction and (
        targets.shape != (6,) or np.iscomplexobj(targets) or not np.isfinite(targets).all()
    ):
        raise ValueError("rigid_moments must contain six finite real integrated moments")
    boundary, fixed = _boundary_vector(skeleton, dirichlet, data, order)
    factory = partial(
        _local,
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        refinement=refinement,
        constitutive=constitutive,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=source,
        order=order,
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        local_refinement_precision=local_refinement_precision,
        backend=backend,
        workers=workers,
    )
    gauges = []
    if pure_traction:
        gauges = [
            system.mean_constraint([m[1][:, i] for m in system.local_metadata], targets[i])
            for i in range(6)
        ]
    hybrid = system.solve(fixed=fixed, constraints=gauges, solver=solver)
    return Elasticity3DSolution(
        skeleton,
        tuple(m[0] for m in system.local_metadata),
        tuple(field.reshape(-1, 3) for field in hybrid.fields),
        hybrid,
        degree,
        constitutive,
        lame_lambda,
        lame_mu,
        source,
        order,
    )


_KELVIN3 = KELVIN_BASIS_3D
