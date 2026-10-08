"""Primal MHM elasticity with general symmetric constitutive tensors and Pk locals.

Constitutive matrices use orthonormal Kelvin coordinates
(epsilon_xx, epsilon_yy, sqrt(2)*epsilon_xy). A supplied fourth-order tensor
instead uses ordinary Cartesian indices and must have major/minor symmetries.
This displacement formulation has no uniform incompressibility guarantee.
"""

from functools import partial
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm._legacy.models.elasticity.mixed_pressure import _rigid
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import (
    trace_coupling,
)
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.vector.primal import trace_detecting_enrichment as _minimal_embedding
from pymhm.fem.vector.primal import triangle_strain_operators
from pymhm.materials.elasticity import KELVIN_BASIS_2D
from pymhm.materials.elasticity import constitutive_values as constitutive_values
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.primal_elasticity import (
    PrimalElasticitySolution as PrimalElasticitySolution,
)


def _local_primal(
    cell: int,
    *,
    mesh: TriangleMesh,
    skeleton: SkeletonSpace,
    degree: int,
    minimal_enrichment: bool,
    refinement: int,
    constitutive: Any,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
) -> LocalAssembly:
    """Assemble symmetric-strain energy, skeletal loads and the three rigid constraints."""
    fine = mesh.submesh(cell, refinement)
    polynomial_degree = degree + int(minimal_enrichment)
    forms = triangle_strain_operators(
        fine,
        polynomial_degree,
        constitutive=constitutive,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=source,
        order=order,
    )
    matrix, mass, load, nodes = forms.matrix, forms.mass, forms.load, forms.nodes
    size = 2 * len(nodes)
    scalar_coupling = trace_coupling(mesh, cell, fine, skeleton, polynomial_degree)
    coupling = np.kron(scalar_coupling, np.eye(2))
    center = mesh.points[mesh.cells[cell]].mean(axis=0)
    rigid = _rigid(nodes, center).reshape(size, 3)
    moments = sparse.kron(mass, sparse.eye(2)) @ rigid
    domain_center = mesh.areas @ mesh.points[mesh.cells].mean(axis=1) / mesh.areas.sum()
    global_moments = sparse.kron(mass, sparse.eye(2)) @ _rigid(nodes, domain_center).reshape(
        size, 3
    )
    embedding = None
    if minimal_enrichment:
        embedding = _minimal_embedding(fine, degree, scalar_coupling)
        matrix = sparse.csc_matrix(embedding.T @ matrix @ embedding)
        coupling, load = embedding.T @ coupling, embedding.T @ load
        moments, global_moments = embedding.T @ moments, embedding.T @ global_moments
        rigid = np.linalg.lstsq(embedding, rigid, rcond=None)[0]
    problem = LocalProblem(matrix, coupling, load, skeleton.cell_dofs(cell), rigid, moments)
    return LocalAssembly(problem, (fine, global_moments, embedding))


def solve_primal_elasticity(
    mesh: TriangleMesh,
    *,
    constitutive: Any = None,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    source: Any = (0.0, 0.0),
    dirichlet: Any = (0.0, 0.0),
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    degree: int = 1,
    minimal_enrichment: bool = False,
    local_refinement: int = 4,
    quadrature_order: int = 6,
    rigid_moments: Any = (0.0, 0.0, 0.0),
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> PrimalElasticitySolution:
    """Solve primal MHM elasticity with general Pk displacement and SPD stiffness.

    Neumann values are physical outward Cauchy tractions; other faces prescribe displacement weakly.
    Pure traction requires the three integrated rigid moments about the domain centroid. The default
    pure-traction skeleton is P1, so it detects rigid rotations across macrofaces; the
    full-displacement default remains P0. Local symmetric strain has exactly the three rigid null
    modes. Skeletal-to-local compatibility is checked by the global rank diagnostic; a single Pk
    triangle and trace Pell satisfy the sufficient local degree rule of
    [Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046): k>=ell+1 (ell
    even), k>=ell+2 (ell odd). The global theorem of
    [Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046) additionally
    requires rigid-motion traces in the skeleton (at least P1); the legacy P0 default is outside
    that theorem. ``minimal_enrichment=True`` implements Pk plus one P(k+1) polynomial per
    displacement component for even k and unsplit P(k-1) traces on one local triangle (Lemma 6.8 of
    [Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046)). The returned
    nodal representation has degree k+1 but the local system uses only dim(Pk)+1 scalar coordinates.
    Full polynomial p-enrichment is also supported. This is a primal displacement method and is not
    asserted to avoid locking near incompressibility.
    """
    if not isinstance(minimal_enrichment, bool):
        raise ValueError("minimal_enrichment must be boolean")
    degree = positive_int(degree, "degree")
    refinement = positive_int(local_refinement, "local_refinement")
    polynomial_degree = degree + int(minimal_enrichment)
    order = positive_int(quadrature_order, "quadrature_order", polynomial_degree + 1)
    data = {} if neumann is None else neumann
    skeleton = skeleton or (
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
        if set(data) == set(mesh.boundary_faces)
        else SkeletonSpace(mesh, components=2)
    )
    if skeleton.mesh is not mesh or skeleton.components != 2:
        raise ValueError("elasticity requires a two-component skeleton on this mesh")
    if minimal_enrichment and (
        refinement != 1
        or degree % 2 != 0
        or any(
            face.degrees != (degree - 1,) or face.breaks != (0.0, 1.0) for face in skeleton.faces
        )
    ):
        raise ValueError(
            "minimal enrichment requires one local triangle, even degree k and unsplit trace P(k-1)"
        )
    boundary, physical_fixed = boundary_data(
        skeleton, dirichlet, data, order=max(order, degree + 2)
    )
    factory = partial(
        _local_primal,
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        minimal_enrichment=minimal_enrichment,
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
        backend=backend,
        workers=workers,
    )
    constraints = []
    if set(data) == set(mesh.boundary_faces):
        if np.iscomplexobj(rigid_moments):
            raise ValueError("rigid_moments must contain three finite real moments")
        targets = np.asarray(rigid_moments, dtype=float)
        if targets.shape != (3,) or not np.isfinite(targets).all():
            raise ValueError("rigid_moments must contain three finite real moments")
        constraints = [
            system.mean_constraint([entry[1][:, i] for entry in system.local_metadata], targets[i])
            for i in range(3)
        ]
    hybrid = system.solve(
        solver=solver, fixed={i: -v for i, v in physical_fixed.items()}, constraints=constraints
    )
    return PrimalElasticitySolution(
        skeleton,
        tuple(entry[0] for entry in system.local_metadata),
        tuple(
            (field if entry[2] is None else entry[2] @ field).reshape(-1, 2)
            for field, entry in zip(hybrid.fields, system.local_metadata, strict=True)
        ),
        (),
        hybrid,
        polynomial_degree,
        1,
        constitutive,
        lame_lambda,
        lame_mu,
        source,
        order,
    )


_KELVIN = KELVIN_BASIS_2D
