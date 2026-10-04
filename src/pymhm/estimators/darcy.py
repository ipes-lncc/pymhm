"""Conforming potential recovery and the unit-diffusion MHM energy estimator."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from pymhm._legacy.models.darcy.primal import DarcySolution
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.conditions import validate_estimator_spaces
from pymhm.fem.hdiv.rt import rt_evaluate
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import nodal_space, tabulate, trace_coupling
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.meshes.triangle import TriangleMesh
from pymhm.recovery.moments import (
    MomentFluxSolution,
    _continuous_basis,
    reconstruct_darcy_moments,
)


def _conforming_mesh(solution: DarcySolution) -> TriangleMesh:
    """Join coincident fine vertices and reject unmatched interior boundary edges."""
    meshes = solution.local_meshes
    points = np.concatenate([mesh.points for mesh in meshes])
    scale = max(float(np.max(np.abs(points))), float(np.ptp(points, axis=0).max()))
    tolerance = 128 * np.finfo(float).eps * scale
    pairs = cKDTree(points).query_pairs(tolerance, output_type="ndarray")
    graph = sparse.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points))
    )
    _, groups = connected_components(graph, directed=False)
    _, first = np.unique(groups, return_index=True)
    cells, offset = [], 0
    for mesh in meshes:
        indices = groups[offset : offset + len(mesh.points)]
        if len(np.unique(indices)) != len(indices):
            raise ValueError("fine vertices are too close to distinguish safely")
        cells.append(indices[mesh.cells])
        offset += len(mesh.points)
    result = TriangleMesh(points[first], np.concatenate(cells))
    coarse = solution.skeleton.mesh
    exterior = coarse.points[coarse.faces[coarse.boundary_faces]]
    vectors = exterior[:, 1] - exterior[:, 0]
    lengths2 = np.sum(vectors**2, axis=1)
    for face in result.boundary_faces:
        vertices = result.points[result.faces[face]]
        delta = vertices[None] - exterior[:, :1]
        parameter = np.einsum("fqa,fa->fq", delta, vectors) / lengths2[:, None]
        distance = np.linalg.norm(delta - parameter[..., None] * vectors[:, None], axis=2)
        inside = (parameter >= -tolerance / np.sqrt(lengths2)[:, None]) & (
            parameter <= 1 + tolerance / np.sqrt(lengths2)[:, None]
        )
        if not np.any(np.all(inside & (distance <= tolerance), axis=1)):
            raise ValueError("fine meshes must form a globally conforming triangulation")
    return result


@dataclass(frozen=True)
class ConformingPotential:
    """Oswald nodal average in a global continuous Pk space with zero boundary."""

    mesh: TriangleMesh
    degree: int
    values: FloatArray
    local_values: tuple[FloatArray, ...]

    def l2_error(self, exact: Any, order: int = 8) -> float:
        """Integrate the recovered potential error on the global fine mesh."""
        bary, weights = triangle_quadrature(order)
        dofs, _, basis, _, _ = tabulate(self.mesh, self.degree, bary)
        points = np.einsum("qi,tij->tqj", bary, self.mesh.points[self.mesh.cells])
        values = self.values[dofs] @ basis.T
        error = values - scalar_values(exact, points.reshape(-1, 2)).reshape(values.shape)
        return float(np.sqrt(self.mesh.areas @ (error**2 @ weights)))


def _average_potential(solution: DarcySolution) -> tuple[TriangleMesh, Any, Any, FloatArray]:
    """Average by incident triangles before prescribing any exterior nodal data."""
    if solution.formulation != "primal":
        raise ValueError("potential recovery requires a primal Darcy solution")
    mesh = _conforming_mesh(solution)
    dofs, points = nodal_space(mesh, solution.degree)
    local_dofs = [nodal_space(fine, solution.degree)[0] for fine in solution.local_meshes]
    broken = np.concatenate(
        [values[indices] for values, indices in zip(solution.pressure, local_dofs, strict=True)]
    )
    sums = np.zeros(len(points), dtype=broken.dtype)
    np.add.at(sums, dofs.ravel(), broken.ravel())
    counts = np.bincount(dofs.ravel(), minlength=len(points))
    values = sums / counts
    return mesh, dofs, local_dofs, values


def _restrict_potential(
    solution: DarcySolution, mesh: TriangleMesh, dofs: Any, local_dofs: Any, values: FloatArray
) -> ConformingPotential:
    """Restrict one conforming nodal function back onto all broken local coefficient arrays."""
    local_values, offset = [], 0
    for fine, indices, original in zip(
        solution.local_meshes, local_dofs, solution.pressure, strict=True
    ):
        recovered = np.empty_like(original)
        recovered[indices] = values[dofs[offset : offset + len(fine.cells)]]
        local_values.append(recovered)
        offset += len(fine.cells)
    return ConformingPotential(mesh, solution.degree, values, tuple(local_values))


def recover_potential(
    solution: DarcySolution, *, homogeneous_dirichlet: bool
) -> ConformingPotential:
    """Average each nodal trace over incident fine triangles, then impose zero BC.

    This is the Oswald operator in equation (5.2) of Barrenechea et al.
    (2026), for globally conforming triangular submeshes and uniform local
    polynomial degree. Repeated values inside one macrocell count once per
    incident fine triangle, not once per macrocell. No smoothing is applied
    to the original solution. The caller must declare homogeneous Dirichlet
    data because DarcySolution does not retain its boundary specification.
    """
    if homogeneous_dirichlet is not True:
        raise ValueError("potential recovery requires homogeneous_dirichlet=True")
    mesh, dofs, local_dofs, values = _average_potential(solution)
    boundary = np.unique(mesh.faces[mesh.boundary_faces])
    edge_nodes = (
        len(mesh.points)
        + (solution.degree - 1) * mesh.boundary_faces[:, None]
        + np.arange(solution.degree - 1)
    ).ravel()
    values[np.r_[boundary, edge_nodes]] = 0.0
    return _restrict_potential(solution, mesh, dofs, local_dofs, values)


def _project(
    mesh: TriangleMesh, degree: int, data: FloatArray, bary: FloatArray, weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Project scalar samples onto continuous macro-local P_m, using one quadrature."""
    dofs, basis, size = _continuous_basis(mesh, degree, bary)
    width = basis.shape[1]
    blocks = np.einsum("q,qi,qj,t->tij", weights, basis, basis, mesh.areas)
    matrix = sparse.coo_matrix(
        (
            blocks.ravel(),
            (np.repeat(dofs, width, axis=1).ravel(), np.tile(dofs, (1, width)).ravel()),
        ),
        shape=(size, size),
    ).tocsc()
    moments = np.einsum("q,qi,tq,t->ti", weights, basis, data, mesh.areas)
    load = np.bincount(dofs.ravel(), weights=moments.ravel(), minlength=size)
    coefficients = solve_linear(matrix, load)
    return coefficients[dofs] @ basis.T, load


@dataclass(frozen=True)
class DarcyEstimator:
    """Four macro-local estimator contributions and their recovered fields.

    ``flux_defect``, ``nonconformity``, ``divergence_defect`` and ``oscillation``
    are eta_1, eta_2, eta_3 and eta_osc in equations (5.3)--(5.7), specialized
    to unit diffusion. Their quadrature evaluation is not interval-certified.
    """

    solution: DarcySolution
    potential: ConformingPotential
    reconstructed_flux: MomentFluxSolution
    flux_defect: FloatArray
    nonconformity: FloatArray
    divergence_defect: FloatArray
    oscillation: FloatArray
    equilibrium_defect: FloatArray
    quadrature_order: int

    @property
    def local_squared(self) -> FloatArray:
        """Return additive squared macro indicators in the reliability bound."""
        return (
            self.flux_defect + self.divergence_defect + self.oscillation
        ) ** 2 + self.nonconformity**2

    @property
    def total(self) -> float:
        """Return the square root of the sum of local squared indicators."""
        return float(np.sqrt(self.local_squared.sum()))

    def energy_error(self, exact_gradient: Any, order: int = 10) -> float:
        """Integrate the genuine broken energy error for unit diffusion."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, coefficients in zip(
            self.solution.local_meshes, self.solution.pressure, strict=True
        ):
            dofs, _, _, gradient, _ = tabulate(mesh, self.solution.degree, bary)
            values = np.einsum("ti,tqia->tqa", coefficients[dofs], gradient)
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            error = values - vector_values(exact_gradient, points.reshape(-1, 2)).reshape(
                values.shape
            )
            total += float(mesh.areas @ (np.sum(error**2, axis=2) @ weights))
        return float(np.sqrt(total))


def estimate_darcy_error(
    solution: DarcySolution,
    *,
    homogeneous_dirichlet: bool,
    degree: int = 1,
    quadrature_order: int = 8,
) -> DarcyEstimator:
    """Evaluate the L09 energy estimator for identity diffusion and zero Dirichlet data.

    Require a primal solution, globally conforming fine triangles, local
    degree k>=ell+2 and ell<=m<=k. Here ell is the largest skeletal degree
    and m is the nonnegative RT reconstruction degree. Macrocells are convex
    triangles, hence their Poincare constant is bounded by diameter/pi.

    Coefficients must be the literal scalar 1 or identity matrix; arbitrary
    SPD tensors and callbacks are rejected rather than inheriting an
    unsupported coefficient-independent bound. Boundary moments and the
    continuous-test conservation identity are checked using numerical
    tolerances. The true source is integrated for oscillation; it is not
    replaced by a projected source. The theorem assumes exact integration;
    this implementation reports quadrature-based indicators and a separate
    equilibrium defect, not an interval-certified upper bound.
    """
    potential = recover_potential(solution, homogeneous_dirichlet=homogeneous_dirichlet)
    material = solution.permeability
    if callable(material) or np.iscomplexobj(material):
        raise ValueError("the estimator currently requires literal identity diffusion")
    array = np.asarray(material, dtype=float)
    if not (
        (array.ndim == 0 and array == 1)
        or (array.shape == (2, 2) and np.array_equal(array, np.eye(2)))
    ):
        raise ValueError("the estimator currently requires literal identity diffusion")
    ell = max(max(face.degrees) for face in solution.skeleton.faces)
    validate_estimator_spaces(solution.degree, ell, degree, 2)
    order = positive_int(quadrature_order, "quadrature_order", max(solution.degree + 2, degree + 2))
    reconstruction = reconstruct_darcy_moments(solution, degree=degree, quadrature_order=order)
    coarse, skeleton = solution.skeleton.mesh, solution.skeleton
    boundary_moments = np.zeros(skeleton.size)
    boundary_scale = np.zeros(skeleton.size)
    for cell, (fine, coefficients) in enumerate(
        zip(solution.local_meshes, solution.pressure, strict=True)
    ):
        coupling = trace_coupling(coarse, cell, fine, skeleton, solution.degree)
        dofs = skeleton.cell_dofs(cell)
        boundary_moments[dofs] += coupling.T @ coefficients
        boundary_scale[dofs] += np.sum(abs(coupling), axis=0) * np.max(abs(coefficients))
    exterior_dofs = np.concatenate([skeleton.dofs(int(face)) for face in coarse.boundary_faces])
    if np.linalg.norm(boundary_moments[exterior_dofs]) > 1e-9 * max(
        np.linalg.norm(boundary_scale), np.finfo(float).tiny
    ):
        raise ValueError("solution does not satisfy homogeneous Dirichlet boundary moments")
    bary, weights = triangle_quadrature(order)
    terms = np.zeros((5, len(coarse.cells)))
    for cell, (fine, coefficients, recovered, flux) in enumerate(
        zip(
            solution.local_meshes,
            solution.pressure,
            potential.local_values,
            reconstruction.flux,
            strict=True,
        )
    ):
        dofs, _, _, gradient, _ = tabulate(fine, solution.degree, bary)
        grad = np.einsum("ti,tqia->tqa", coefficients[dofs], gradient)
        delta_grad = np.einsum("ti,tqia->tqa", (coefficients - recovered)[dofs], gradient)
        physical_flux, divergence = rt_evaluate(fine, flux, degree, bary)
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        source = scalar_values(solution.source, points.reshape(-1, 2)).reshape(divergence.shape)
        projected_source, source_moments = _project(fine, degree, source, bary, weights)
        projected_div, div_moments = _project(fine, degree, divergence, bary, weights)
        diameter = np.max(coarse.lengths[coarse.cell_faces[cell]])
        terms[0, cell] = np.sqrt(
            fine.areas @ (np.sum((grad + physical_flux) ** 2, axis=2) @ weights)
        )
        terms[1, cell] = np.sqrt(fine.areas @ (np.sum(delta_grad**2, axis=2) @ weights))
        terms[2, cell] = (
            diameter / np.pi * np.sqrt(fine.areas @ ((projected_div - divergence) ** 2 @ weights))
        )
        terms[3, cell] = (
            diameter / np.pi * np.sqrt(fine.areas @ ((source - projected_source) ** 2 @ weights))
        )
        terms[4, cell] = np.linalg.norm(source_moments - div_moments)
        scale = max(
            np.linalg.norm(source_moments),
            np.linalg.norm(div_moments),
            np.linalg.norm(grad) * fine.areas.sum(),
        )
        if terms[4, cell] > 1e-8 * max(scale, np.finfo(float).tiny):
            raise ValueError(
                "continuous-test equilibrium failed; increase source assembly quadrature"
            )
    return DarcyEstimator(
        solution, potential, reconstruction, terms[0], terms[1], terms[2], terms[3], terms[4], order
    )
