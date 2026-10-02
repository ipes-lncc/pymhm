"""Weakly symmetric mixed MHM elasticity on Cartesian rectangles.

Stress rows use RT normal degree k and interior order s=k+n. Displacement is
Q_s squared, whereas independent rotation is total-degree P_s. This distinction
is essential to the quadrilateral family in the 2021 weak-symmetry analysis.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, cast

import numpy as np
from numpy.polynomial.legendre import legvander
from scipy import sparse

from pymhm.elasticity import _boundary_volume_flux
from pymhm.elasticity_compatibility import require_compatible_displacement_flux
from pymhm.elasticity_compliance import compliance_products
from pymhm.elasticity_mixed import _bulk_compliance, _rigid_values, _scatter
from pymhm.elements import boundary_data, scalar_values, vector_values
from pymhm.hybrid import HybridSolution, HybridSystem, LocalAssembly, LocalProblem
from pymhm.mesh import FaceSpace, FloatArray, SkeletonSpace, positive_int
from pymhm.quadrilateral import (
    CartesianMacroMesh,
    _grid_resolves_material,
    quadrilateral_quadrature,
)
from pymhm.reservoir import CartesianCellField
from pymhm.tensor_rt import _trace_map, tensor_rt_basis, tensor_rt_dofs


def _rotation_basis(degree: int, points: FloatArray) -> FloatArray:
    """Tabulate total-degree Legendre products P_s, excluding the Q_s cross corners."""
    x, y = np.moveaxis(points, -1, 0)
    lx, ly = legvander(2 * x - 1, degree), legvander(2 * y - 1, degree)
    return np.stack(
        [lx[..., a] * ly[..., b] for b in range(degree + 1) for a in range(degree + 1 - b)], axis=-1
    )


def _modal_rigid(
    mesh: CartesianMacroMesh, degree: int, center: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Project rigid displacement into Q_s and return its physical moment matrix."""
    points, weights = quadrilateral_quadrature(degree + 2)
    _, _, basis = tensor_rt_basis(mesh, 1, degree - 1, points)
    physical = mesh.points[mesh.cells[:, 0], None] + points * mesh.spacing
    moments = np.einsum(
        "q,qi,tqak,t->tiak", weights, basis, _rigid_values(physical, center), mesh.areas
    )
    mass = mesh.areas[:, None] * np.einsum("q,qi,qi->i", weights, basis, basis)
    return (moments / mass[:, :, None, None]).reshape(-1, 3), moments.reshape(-1, 3)


def _operators(
    mesh: CartesianMacroMesh,
    degree: int,
    enrichment: int,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    compliance: Any = None,
) -> tuple:
    """Assemble compliance, Q_s divergence, P_s asymmetry and physical force moments."""
    points, weights = quadrilateral_quadrature(order)
    values, divergence, displacement = tensor_rt_basis(mesh, degree, enrichment, points)
    rotation = _rotation_basis(degree + enrichment, points)
    physical = mesh.points[mesh.cells[:, 0], None] + points * mesh.spacing
    width = values.shape[2]
    tensors = np.zeros((*values.shape[:2], 2 * width, 2, 2))
    tensors[:, :, 0::2, 0], tensors[:, :, 1::2, 1] = values, values
    trace = np.trace(tensors, axis1=-2, axis2=-1)
    if compliance is None:
        mu = scalar_values(lame_mu, physical.reshape(-1, 2)).reshape(physical.shape[:2])
        bulk = _bulk_compliance(lame_lambda, mu.ravel(), physical.reshape(-1, 2)).reshape(mu.shape)
        deviator = tensors - trace[..., None, None] * np.eye(2) / 2
        mass = np.einsum(
            "q,tq,tqiab,tqjab,t->tij", weights, 1 / (2 * mu), deviator, deviator, mesh.areas
        )
        mass += np.einsum("q,tq,tqi,tqj,t->tij", weights, bulk / 2, trace, trace, mesh.areas)
        weighted_trace = bulk[:, :, None] * trace
        compliance_scale = float(bulk.max())
    else:
        products, weighted_trace, compliance_scale = compliance_products(
            compliance, physical, tensors
        )
        mass = np.einsum("q,tqij,t->tij", weights, products, mesh.areas)
    scalar_div = np.einsum("q,qi,tqj,t->tij", weights, displacement, divergence, mesh.areas)
    nd, nr = displacement.shape[1], rotation.shape[1]
    div = np.zeros((len(mesh.cells), 2 * nd, 2 * width))
    div[:, 0::2, 0::2], div[:, 1::2, 1::2] = scalar_div, scalar_div
    asym = np.einsum(
        "q,qi,tqj,t->tij", weights, rotation, tensors[..., 0, 1] - tensors[..., 1, 0], mesh.areas
    )
    force = np.einsum(
        "q,qi,tqa,t->tia",
        weights,
        displacement,
        vector_values(source, physical.reshape(-1, 2)).reshape(*physical.shape[:2], 2),
        mesh.areas,
    )
    dofs = (2 * tensor_rt_dofs(mesh, degree, enrichment)[:, :, None] + np.arange(2)).reshape(
        -1, 2 * width
    )
    ns, nu, nrot = int(dofs.max()) + 1, 2 * nd * len(mesh.cells), nr * len(mesh.cells)
    plain = np.bincount(
        dofs.ravel(),
        weights=np.einsum("q,tqi,t->ti", weights, trace, mesh.areas).ravel(),
        minlength=ns,
    )
    weighted = np.bincount(
        dofs.ravel(),
        weights=np.einsum("q,tqi,t->ti", weights, weighted_trace, mesh.areas).ravel(),
        minlength=ns,
    )
    return (
        _scatter(mass, dofs, dofs, (ns, ns)),
        _scatter(div, np.arange(nu).reshape(len(mesh.cells), -1), dofs, (nu, ns)),
        _scatter(asym, np.arange(nrot).reshape(len(mesh.cells), -1), dofs, (nrot, ns)),
        force.ravel(),
        plain,
        weighted,
        compliance_scale,
    )


def _local(
    cell: int,
    *,
    mesh: CartesianMacroMesh,
    skeleton: SkeletonSpace,
    degree: int,
    enrichment: int,
    refinement: int,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    compliance: Any = None,
) -> LocalAssembly:
    """Augment a mixed Neumann problem with boundary displacement and three rigid modes."""
    fine = mesh.submesh(cell, refinement)
    for coefficient in (lame_lambda, lame_mu):
        if isinstance(coefficient, CartesianCellField) and not _grid_resolves_material(
            fine, coefficient
        ):
            raise ValueError("Cartesian Lame material interfaces must align with fine rectangles")
    mass, div, asym, force, trace, bulk_trace, bulk_max = _operators(
        fine, degree, enrichment, lame_lambda, lame_mu, source, order, compliance
    )
    ns, nu, nr = mass.shape[0], div.shape[0], asym.shape[0]
    nb = 2 * (degree + 1) * len(fine.boundary_faces)
    rows = (2 * (degree + 1) * fine.boundary_faces[:, None] + np.arange(2 * (degree + 1))).ravel()
    selector = sparse.coo_matrix((np.ones(nb), (rows, np.arange(nb))), shape=(ns, nb)).tocsc()
    matrix = sparse.bmat(
        [
            [mass, div.T, asym.T, -selector],
            [
                div,
                sparse.csc_matrix((nu, nu)),
                sparse.csc_matrix((nu, nr)),
                sparse.csc_matrix((nu, nb)),
            ],
            [
                asym,
                sparse.csc_matrix((nr, nu)),
                sparse.csc_matrix((nr, nr)),
                sparse.csc_matrix((nr, nb)),
            ],
            [
                -selector.T,
                sparse.csc_matrix((nb, nu)),
                sparse.csc_matrix((nb, nr)),
                sparse.csc_matrix((nb, nb)),
            ],
        ],
        format="csc",
    )
    size = ns + nu + nr + nb
    center = mesh.points[mesh.cells[cell]].mean(axis=0)
    rigid, moments = _modal_rigid(fine, degree + enrichment, center)
    kernel = np.zeros((size, 3))
    kernel[ns : ns + nu] = rigid
    nrot = nr // len(fine.cells)
    kernel[ns + nu : ns + nu + nr : nrot, 2] = -1
    boundary = _rigid_values(fine.points[fine.faces[fine.boundary_faces]], center)
    coefficients = np.zeros((len(boundary), degree + 1, 2, 3))
    coefficients[:, 0], coefficients[:, 1] = (
        boundary.mean(axis=1),
        (boundary[:, 1] - boundary[:, 0]) / 2,
    )
    kernel[-nb:] = coefficients.reshape(-1, 3)
    constraints = np.zeros_like(kernel)
    constraints[ns : ns + nu] = moments
    mapping = np.kron(_trace_map(mesh, cell, fine, skeleton, degree), np.eye(2))
    coupling = np.zeros((size, mapping.shape[1]))
    coupling[-nb:] = -mapping
    problem = LocalProblem(
        matrix,
        coupling,
        np.r_[np.zeros(ns), -force, np.zeros(nr + nb)],
        skeleton.cell_dofs(cell),
        kernel,
        constraints,
    )
    global_center = mesh.areas @ mesh.points[mesh.cells].mean(axis=1) / mesh.areas.sum()
    global_moments = np.zeros_like(kernel)
    global_moments[ns : ns + nu] = _modal_rigid(fine, degree + enrichment, global_center)[1]
    return LocalAssembly(
        problem,
        (
            fine,
            ns,
            nu,
            nr,
            np.pad(trace, (0, size - ns)),
            np.pad(bulk_trace, (0, size - ns)),
            bulk_max,
            global_moments,
        ),
    )


@dataclass(frozen=True)
class TensorRTElasticitySolution:
    """H(div) Cauchy stress with Q_s displacement and independent P_s weak rotation."""

    skeleton: SkeletonSpace
    local_meshes: tuple[CartesianMacroMesh, ...]
    stress: tuple[FloatArray, ...]
    displacement: tuple[FloatArray, ...]
    rotation: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    enrichment: int
    source: Any
    quadrature_order: int

    def evaluate(
        self, cell: int, points: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return displacement, full stress, stress divergence and rotation at reference points."""
        mesh = self.local_meshes[cell]
        basis, div, scalar = tensor_rt_basis(mesh, self.degree, self.enrichment, points)
        values = self.stress[cell][tensor_rt_dofs(mesh, self.degree, self.enrichment)]
        scalar = np.broadcast_to(scalar, (*basis.shape[:2], scalar.shape[-1]))
        rotation = _rotation_basis(self.degree + self.enrichment, points)
        rotation = np.broadcast_to(rotation, (*basis.shape[:2], rotation.shape[-1]))
        return (
            np.einsum("tqi,tia->tqa", scalar, self.displacement[cell]),
            np.einsum("tqib,tia->tqab", basis, values),
            np.einsum("tqi,tia->tqa", div, values),
            np.einsum(
                "tqi,ti->tq",
                rotation,
                self.rotation[cell],
            ),
        )

    def errors(
        self, displacement: Any, stress: Any, divergence: Any, rotation: Any, order: int = 6
    ) -> dict[str, float]:
        """Integrate four physical L2 errors with full Frobenius stress, without symmetrization."""
        points, weights = quadrilateral_quadrature(order)
        errors = np.zeros(4)
        for cell, mesh in enumerate(self.local_meshes):
            physical = mesh.points[mesh.cells[:, 0], None] + points * mesh.spacing
            flat = physical.reshape(-1, 2)
            target = stress(flat) if callable(stress) else stress
            if np.iscomplexobj(target) or not np.isfinite(target).all():
                raise ValueError("exact stress must be real and finite")
            u, sigma, div, rot = self.evaluate(cell, points)
            targets = (
                vector_values(displacement, flat).reshape(u.shape),
                np.broadcast_to(target, (len(flat), 2, 2)).reshape(sigma.shape),
                vector_values(divergence, flat).reshape(div.shape),
                scalar_values(rotation, flat).reshape(rot.shape),
            )
            for index, (value, exact) in enumerate(zip((u, sigma, div, rot), targets, strict=True)):
                error = (value - exact).reshape(len(mesh.cells), len(points), -1)
                errors[index] += mesh.areas @ (np.sum(error**2, axis=-1) @ weights)
        return dict(
            zip(
                ("displacement_l2", "stress_l2", "divergence_l2", "rotation_l2"),
                np.sqrt(errors),
                strict=True,
            )
        )

    def fine_force_residuals(self) -> tuple[FloatArray, ...]:
        """Return every fine-cell Q_s vector moment of div(sigma)+f."""
        points, weights = quadrilateral_quadrature(self.quadrature_order)
        result = []
        for cell, mesh in enumerate(self.local_meshes):
            physical = mesh.points[mesh.cells[:, 0], None] + points * mesh.spacing
            force = vector_values(self.source, physical.reshape(-1, 2)).reshape(physical.shape)
            _, _, basis = tensor_rt_basis(mesh, self.degree, self.enrichment, points)
            result.append(
                np.einsum(
                    "t,q,qi,tqa->tia",
                    mesh.areas,
                    weights,
                    basis,
                    self.evaluate(cell, points)[2] + force,
                )
            )
        return tuple(result)

    def weak_symmetry_residuals(self) -> tuple[FloatArray, ...]:
        """Return P_s moments of stress asymmetry, which need not vanish pointwise."""
        points, weights = quadrilateral_quadrature(self.degree + self.enrichment + 2)
        result = []
        for cell, mesh in enumerate(self.local_meshes):
            sigma = self.evaluate(cell, points)[1]
            result.append(
                np.einsum(
                    "t,q,qi,tq->ti",
                    mesh.areas,
                    weights,
                    _rotation_basis(self.degree + self.enrichment, points),
                    sigma[..., 0, 1] - sigma[..., 1, 0],
                )
            )
        return tuple(result)

    def normal_traction_residuals(self) -> tuple[FloatArray, ...]:
        """Return moments of sigma n plus the signed macro traction on each fine boundary."""
        result = []
        for cell, mesh in enumerate(self.local_meshes):
            ids = (
                (self.degree + 1) * mesh.boundary_faces[:, None] + np.arange(self.degree + 1)
            ).ravel()
            mapping = _trace_map(
                cast(CartesianMacroMesh, self.skeleton.mesh), cell, mesh, self.skeleton, self.degree
            )
            result.append(
                self.stress[cell][ids]
                + mapping @ self.hybrid.trace[self.skeleton.cell_dofs(cell)].reshape(-1, 2)
            )
        return tuple(result)

    def equilibrium_residuals(self) -> FloatArray:
        """Return macro force and moment balance using physical signed skeletal traction."""
        points, weights = quadrilateral_quadrature(self.quadrature_order)
        result = np.zeros((len(self.local_meshes), 3))
        coarse = self.skeleton.mesh
        for cell, mesh in enumerate(self.local_meshes):
            center = coarse.points[coarse.cells[cell]].mean(axis=0)
            physical = mesh.points[mesh.cells[:, 0], None] + points * mesh.spacing
            force = vector_values(self.source, physical.reshape(-1, 2)).reshape(physical.shape)
            result[cell] -= np.einsum(
                "t,q,tqa,tqak->k", mesh.areas, weights, force, _rigid_values(physical, center)
            )
            for side, face in enumerate(coarse.cell_faces[cell]):
                space = self.skeleton.faces[face]
                parameter, w = space.quadrature(self.quadrature_order)
                start, end = coarse.points[coarse.faces[face]]
                face_points = start + parameter[:, None] * (end - start)
                traction = space.evaluate(parameter) @ self.hybrid.trace[
                    self.skeleton.dofs(int(face))
                ].reshape(-1, 2)
                result[cell] += (
                    coarse.signs[cell, side]
                    * coarse.lengths[face]
                    * np.einsum("q,qa,qak->k", w, traction, _rigid_values(face_points, center))
                )
        return result


def solve_elasticity_tensor_rt(
    mesh: CartesianMacroMesh,
    *,
    degree: int = 1,
    enrichment: int = 0,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    compliance: Any = None,
    source: Any = (0.0, 0.0),
    dirichlet: Any = (0.0, 0.0),
    traction: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 2,
    quadrature_order: int = 6,
    rigid_moments: Any = (0.0, 0.0, 0.0),
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> TensorRTElasticitySolution:
    """Solve the Cartesian RT_k/RT-enriched weak-symmetry family with physical gauges.

    k>=1 is required. Fine stress normal degree is k; interior RT order s=k+n
    yields displacement Q_s squared and rotation P_s. The skeleton has P1 on
    interior macrofaces and full fine Pk on exterior faces by default. Custom
    normal traces must align with fine edges and have degree at most k.
    Traction means physical outward sigma n; the multiplier is its negative.
    Fully prescribed displacement uses the exact bulk-compliance identity at
    finite lambda and mean(-trace(sigma)/2) at infinite lambda. Pure traction
    prescribes three integrated displacement moments against global rigid modes.
    ``compliance`` optionally supplies a positive full Cartesian anisotropic
    compliance in (xx,xy,yx,yy) coordinates, replacing the Lame material. Its
    explicit skew extension must preserve symmetric tensors; the physical
    boundary identity then uses tr(A sigma).
    Coefficients may be smooth callbacks; material discontinuities must align
    with the fine grid for the ordinary per-rectangle Gaussian rule used here.
    """
    k = positive_int(degree, "stress degree", 1)
    n = positive_int(enrichment, "enrichment", 0)
    s = k + n
    refinement = positive_int(local_refinement, "local_refinement")
    order = positive_int(quadrature_order, "quadrature_order", s + 2)
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    boundary_faces = set(mesh.boundary_faces)
    skeleton = skeleton or SkeletonSpace(
        cast(Any, mesh),
        tuple(
            FaceSpace.uniform(k, refinement) if f in boundary_faces else FaceSpace.uniform(1)
            for f in range(len(mesh.faces))
        ),
        2,
    )
    if skeleton.mesh is not mesh or skeleton.components != 2:
        raise ValueError("mixed elasticity requires a two-component skeleton on this mesh")
    for face in skeleton.faces:
        if max(face.degrees) > k or not np.allclose(
            np.array(face.breaks) * refinement,
            np.round(np.array(face.breaks) * refinement),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("RT trace degrees exceed k or segments do not align with fine edges")
    data = {} if traction is None else traction
    if data and mean_pressure != 0:
        raise ValueError("mean_pressure requires full displacement boundaries")
    negative = {
        face: (lambda points, value=value: -vector_values(value, points))
        for face, value in data.items()
    }
    boundary, fixed = boundary_data(skeleton, dirichlet, negative, order=order)
    factory = partial(
        _local,
        mesh=mesh,
        skeleton=skeleton,
        degree=k,
        enrichment=n,
        refinement=refinement,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        compliance=compliance,
        source=source,
        order=order,
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=-boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    metadata = system.local_metadata
    gauges = []
    if not data:
        scale = max(entry[6] for entry in metadata)
        flux = _boundary_volume_flux(skeleton, boundary)
        if scale == 0:
            require_compatible_displacement_flux(
                flux, _boundary_volume_flux(skeleton, boundary, absolute=True)
            )
            gauges.append(
                system.mean_constraint(
                    [entry[4] for entry in metadata], -2 * mean_pressure * sum(mesh.areas)
                )
            )
        else:
            if mean_pressure != 0:
                raise ValueError("mean_pressure is only a gauge in the incompressible limit")
            gauges.append(
                system.mean_constraint([entry[5] / scale for entry in metadata], flux / scale)
            )
    if set(data) == boundary_faces:
        targets = np.asarray(rigid_moments)
        if targets.shape != (3,) or np.iscomplexobj(targets) or not np.isfinite(targets).all():
            raise ValueError("rigid_moments must contain three finite real moments")
        gauges.extend(
            system.mean_constraint([entry[7][:, i] for entry in metadata], targets[i])
            for i in range(3)
        )
    hybrid = system.solve(solver=solver, fixed=fixed, constraints=gauges)
    stress, displacement, rotation = [], [], []
    for values, entry in zip(hybrid.fields, metadata, strict=True):
        fine, ns, nu, nr = entry[:4]
        stress.append(values[:ns].reshape(-1, 2))
        displacement.append(values[ns : ns + nu].reshape(len(fine.cells), (s + 1) ** 2, 2))
        rotation.append(
            values[ns + nu : ns + nu + nr].reshape(len(fine.cells), (s + 1) * (s + 2) // 2)
        )
    return TensorRTElasticitySolution(
        skeleton,
        tuple(entry[0] for entry in metadata),
        tuple(stress),
        tuple(displacement),
        tuple(rotation),
        hybrid,
        k,
        n,
        source,
        order,
    )
