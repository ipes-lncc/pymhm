"""Primal MHM elasticity with general symmetric constitutive tensors and Pk locals.

Constitutive matrices use orthonormal Kelvin coordinates
(epsilon_xx, epsilon_yy, sqrt(2)*epsilon_xy). A supplied fourth-order tensor
instead uses ordinary Cartesian indices and must have major/minor symmetries.
This displacement formulation has no uniform incompressibility guarantee.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np
from scipy import linalg, sparse

from pymhm._legacy.models.elasticity.mixed_pressure import _rigid, _strain_and_divergence
from pymhm._legacy.models.vector import VectorSolution
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import (
    element_tabulate,
    multiindices,
    nodal_space,
    reference_basis,
    trace_coupling,
)
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.vector.operators import _assemble_blocks
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.meshes.triangle import TriangleMesh

_KELVIN = np.array(
    [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
        [[0.0, 1 / np.sqrt(2)], [1 / np.sqrt(2), 0.0]],
    ]
)


def constitutive_values(
    material: Any, points: FloatArray, *, lame_lambda: Any = 1.0, lame_mu: Any = 1.0
) -> FloatArray:
    """Evaluate and validate stiffness in orthonormal Kelvin coordinates.

    Constant or callable matrices may have shape (3,3) or (2,2,2,2), optionally
    preceded by the point axis. Positivity is checked at supplied points;
    quadrature sampling does not certify uniform ellipticity everywhere.
    With material=None, finite nonnegative lambda and positive mu define the
    isotropic plane-strain tensor. A primal tensor with infinite bulk modulus
    is not supported; use the mixed incompressible formulations instead.
    """
    if material is None:
        lam, mu = scalar_values(lame_lambda, points), scalar_values(lame_mu, points)
        if np.any(lam < 0) or np.any(mu <= 0):
            raise ValueError("Lame mu must be positive and lambda nonnegative")
        result = np.zeros((len(points), 3, 3))
        result[:, 0, 0] = result[:, 1, 1] = lam + 2 * mu
        result[:, 0, 1] = result[:, 1, 0] = lam
        result[:, 2, 2] = 2 * mu
        return result
    raw = material(points) if callable(material) else material
    if np.iscomplexobj(raw):
        raise ValueError("constitutive tensor must be real")
    values = np.asarray(raw, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("constitutive tensor must be finite")
    if values.shape[-4:] == (2, 2, 2, 2):
        values = np.broadcast_to(values, (len(points), 2, 2, 2, 2))
        scale = np.max(np.abs(values), axis=(1, 2, 3, 4))
        for transpose in (
            values.swapaxes(1, 2),
            values.swapaxes(3, 4),
            values.transpose(0, 3, 4, 1, 2),
        ):
            if np.any(
                np.max(np.abs(values - transpose), axis=(1, 2, 3, 4))
                > 64 * np.finfo(float).eps * scale
            ):
                raise ValueError("constitutive tensor requires major and minor symmetries")
        values = np.einsum("aij,nijkl,bkl->nab", _KELVIN, values, _KELVIN)
    else:
        if values.shape[-2:] != (3, 3):
            raise ValueError(
                "constitutive tensor requires Kelvin (3,3) or Cartesian (2,2,2,2) shape"
            )
        values = np.broadcast_to(values, (len(points), 3, 3))
    if np.any(
        np.max(np.abs(values - values.swapaxes(1, 2)), axis=(1, 2))
        > 64 * np.finfo(float).eps * np.max(np.abs(values), axis=(1, 2))
    ):
        raise ValueError("Kelvin constitutive matrix must be symmetric")
    if np.any(np.linalg.eigvalsh(values)[:, 0] <= 0):
        raise ValueError("constitutive tensor must be positive definite on symmetric strains")
    return values


@dataclass(frozen=True)
class PrimalElasticitySolution(VectorSolution):
    """Broken Pk displacement with raw symmetric stress from a general stiffness.

    Raw stress is pointwise symmetric. A finite-dimensional local primal solve
    does not generally make it H(div)-conforming or equilibrated on every fine
    cell; skeletal rigid-motion equations impose macro force/moment balance.
    """

    constitutive: Any = None
    lame_lambda: Any = 1.0
    lame_mu: Any = 1.0
    source: Any = (0.0, 0.0)
    quadrature_order: int = 5

    def gradient(self, cell: int, bary: FloatArray) -> FloatArray:
        """Evaluate the full displacement gradient in each local fine cell."""
        mesh = self.local_meshes[cell]
        bary = np.broadcast_to(bary, (len(mesh.cells), *np.asarray(bary).shape[-2:]))
        dofs, _, _, gradient, _ = element_tabulate(mesh, self.degree, bary)
        return np.einsum("tqib,tia->tqab", gradient, self.values[cell][dofs])

    def _stress_values(self, cell: int, bary: FloatArray, material: Any) -> FloatArray:
        """Evaluate stress using a resolved material at cellwise physical points."""
        mesh = self.local_meshes[cell]
        points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
        stiffness = constitutive_values(
            material,
            points.reshape(-1, 2),
            lame_lambda=self.lame_lambda,
            lame_mu=self.lame_mu,
        )
        strain = np.einsum("aij,tqij->tqa", _KELVIN, self.gradient(cell, bary))
        sigma = np.einsum("nab,nb->na", stiffness, strain.reshape(-1, 3))
        return np.einsum("na,aij->nij", sigma, _KELVIN).reshape(*points.shape[:2], 2, 2)

    def stress(self, cell: int, bary: FloatArray) -> FloatArray:
        """Evaluate physical stress at common or cellwise barycentric points.

        At a material interface the coefficient's pointwise convention applies.
        Error integration instead uses exact pixel-intersection quadrature and
        material indices, preserving the two distinct traces without averaging.
        """
        mesh = self.local_meshes[cell]
        bary = np.broadcast_to(bary, (len(mesh.cells), *np.asarray(bary).shape[-2:]))
        return self._stress_values(cell, bary, self.constitutive)

    def stress_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate Frobenius stress error, splitting Cartesian material interfaces."""
        total = 0.0
        for cell, mesh in enumerate(self.local_meshes):
            bary, weights, material = material_triangle_quadrature(mesh, self.constitutive, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            target = exact(points.reshape(-1, 2)) if callable(exact) else exact
            target = np.asarray(target)
            if np.iscomplexobj(target) or not np.isfinite(target).all():
                raise ValueError("exact stress must be real and finite")
            error = self._stress_values(cell, bary, material) - np.broadcast_to(
                target, (points.shape[0] * points.shape[1], 2, 2)
            ).reshape(*points.shape[:2], 2, 2)
            total += float(
                np.einsum("t,tq,tq->", mesh.areas, weights, np.sum(error**2, axis=(-1, -2)))
            )
        return float(np.sqrt(total))


def _minimal_embedding(fine: TriangleMesh, degree: int, coupling: FloatArray) -> FloatArray:
    """Enrich Pk by one P(k+1) mode that detects its missing odd-degree trace.

    This realizes the polynomial enrichment hypothesis of Lemma 6.8 in
    [Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046) on one triangular
    element. The added mode is selected from the null trace's nonzero moment functional and is not a
    zero-boundary bubble. Coordinate projection removes its Pk part without changing its
    missing-trace moment.
    """
    if len(fine.cells) != 1:
        raise ValueError("minimal enrichment requires exactly one local triangle")
    low_dofs, low_nodes = nodal_space(fine, degree)
    high_dofs, high_nodes = nodal_space(fine, degree + 1)
    bary = multiindices(degree + 1) / (degree + 1)
    embedding = np.zeros((len(high_nodes), len(low_nodes)))
    embedding[np.ix_(high_dofs[0], low_dofs[0])] = reference_basis(degree, bary)[0]
    # Resolve the analytically null trace at the scale of its assembled moments;
    # physical edge integration also contributes floating-point roundoff.
    moments = embedding.T @ coupling
    null = linalg.null_space(moments, rcond=64 * np.finfo(float).eps * max(moments.shape))
    if null.shape[1] != 1:
        raise ValueError("minimal enrichment requires exactly one invisible scalar trace mode")
    mode = coupling @ null[:, 0]
    mode -= embedding @ np.linalg.lstsq(embedding, mode, rcond=None)[0]
    mode /= np.linalg.norm(mode)
    return np.kron(np.column_stack((embedding, mode)), np.eye(2))


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
    bary, weights, material = material_triangle_quadrature(fine, constitutive, order)
    dofs, nodes, basis, gradients, hessian = element_tabulate(fine, polynomial_degree, bary)
    ns, size = basis.shape[-1], 2 * len(nodes)
    udofs = (2 * dofs[:, :, None] + np.arange(2)).reshape(len(fine.cells), 2 * ns)
    strain, _, _ = _strain_and_divergence(gradients, hessian)
    strain[:, :, 2] /= np.sqrt(2)
    physical = np.einsum("tqi,tij->tqj", bary, fine.points[fine.cells])
    stiffness = constitutive_values(
        material, physical.reshape(-1, 2), lame_lambda=lame_lambda, lame_mu=lame_mu
    ).reshape(*weights.shape, 3, 3)
    blocks = np.einsum("tq,tqai,tqab,tqbj,t->tij", weights, strain, stiffness, strain, fine.areas)
    matrix = _assemble_blocks(blocks, udofs, size)
    force = vector_values(source, physical.reshape(-1, 2)).reshape(*weights.shape, 2)
    loads = np.einsum("tq,tqi,tqa,t->tia", weights, basis, force, fine.areas)
    load = np.bincount(udofs.ravel(), weights=loads.ravel(), minlength=size)
    scalar_coupling = trace_coupling(mesh, cell, fine, skeleton, polynomial_degree)
    coupling = np.kron(scalar_coupling, np.eye(2))
    center = mesh.points[mesh.cells[cell]].mean(axis=0)
    rigid = _rigid(nodes, center).reshape(size, 3)
    mass = _assemble_blocks(
        np.einsum("tq,tqi,tqj,t->tij", weights, basis, basis, fine.areas), dofs, len(nodes)
    )
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
