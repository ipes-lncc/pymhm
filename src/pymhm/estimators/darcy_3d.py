"""Tetrahedral Oswald recovery and the MHM flux-based energy decomposition.

The four terms use canonical RT moments, continuous local source/divergence
projections and diameter/pi Poincare bounds on convex macrotetrahedra. Energy
normalization includes certified lower ellipticity bounds; literal published
normalization is an indicator for general materials, without that bound claim.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from pymhm._legacy.models.darcy.primal_3d import Darcy3DSolution
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.conditions import validate_estimator_spaces
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import (
    tetra_basis,
    tetra_face_basis,
    tetra_nodal_space,
    tetra_tabulate,
    tetrahedron_quadrature,
)
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import scalar_values_3d, tensor_values_3d, vector_values_3d
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.recovery.moments_3d import MomentFlux3DSolution, reconstruct_darcy_moments_3d


def _on_face(points: FloatArray, vertices: FloatArray, tolerance: float) -> np.ndarray:
    """Identify points in a closed physical triangle using distance and barycentric tests."""
    tangent = (vertices[1:] - vertices[0]).T
    coordinates = (points - vertices[0]) @ np.linalg.pinv(tangent).T
    residual = points - vertices[0] - coordinates @ tangent.T
    scaled = tolerance / np.linalg.norm(tangent, axis=0).min()
    return (
        (np.linalg.norm(residual, axis=1) <= tolerance)
        & (coordinates.min(axis=1) >= -scaled)
        & (coordinates.sum(axis=1) <= 1 + scaled)
    )


@dataclass(frozen=True)
class ConformingPotential3D:
    """Oswald potential and its restrictions to each original local nodal space."""

    mesh: TetraMesh
    degree: int
    values: FloatArray
    local_values: tuple[FloatArray, ...]

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the global conforming potential error with positive quadrature."""
        bary, weights = tetrahedron_quadrature(order)
        dofs, _, basis, _ = tetra_tabulate(self.mesh, self.degree, bary)
        actual = self.values[dofs] @ basis.T
        points = np.einsum("qi,tij->tqj", bary, self.mesh.points[self.mesh.cells])
        expected = scalar_values_3d(exact, points.reshape(-1, 3)).reshape(actual.shape)
        return float(np.sqrt(self.mesh.volumes @ ((actual - expected) ** 2 @ weights)))


def recover_potential_3d(
    solution: Darcy3DSolution,
    *,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
) -> ConformingPotential3D:
    """Average incident fine tetrahedra, then impose exactly represented Dirichlet data.

    Local fine partitions must form a globally conforming tetrahedral mesh.
    Neumann faces remain free except at shared Dirichlet nodes. Nonpolynomial
    boundary data need a separate lifting estimator and are rejected here.
    """
    coarse = solution.skeleton.mesh
    natural = {} if neumann is None else neumann
    if not set(natural).issubset(coarse.boundary_faces):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    points = np.concatenate([fine.points for fine in solution.local_meshes])
    tolerance = 128 * np.finfo(float).eps * max(np.max(abs(points)), np.ptp(points, axis=0).max())
    pairs = cKDTree(points).query_pairs(tolerance, output_type="ndarray")
    graph = sparse.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points))
    )
    _, groups = connected_components(graph, directed=False)
    _, first = np.unique(groups, return_index=True)
    cells, offset = [], 0
    for fine in solution.local_meshes:
        indices = groups[offset : offset + len(fine.points)]
        if len(np.unique(indices)) != len(indices):
            raise ValueError("fine vertices are too close to distinguish safely")
        cells.append(indices[fine.cells])
        offset += len(fine.points)
    mesh = TetraMesh(points[first], np.concatenate(cells))
    parent_faces = {}
    for face in mesh.boundary_faces:
        vertices = mesh.points[mesh.faces[face]]
        parents = [
            int(f)
            for f in coarse.boundary_faces
            if np.all(_on_face(vertices, coarse.points[coarse.faces[f]], tolerance))
        ]
        if len(parents) != 1:
            raise ValueError("fine meshes must form a globally conforming tetrahedral partition")
        parent_faces[int(face)] = parents[0]
    dofs, nodes = tetra_nodal_space(mesh, solution.degree)
    local_dofs = [tetra_nodal_space(fine, solution.degree)[0] for fine in solution.local_meshes]
    broken = np.concatenate(
        [values[ids] for values, ids in zip(solution.pressure, local_dofs, strict=True)]
    )
    sums = np.bincount(dofs.ravel(), weights=broken.ravel(), minlength=len(nodes))
    counts = np.bincount(dofs.ravel(), minlength=len(nodes))
    values = sums / counts
    for face in set(coarse.boundary_faces) - set(natural):
        selected = _on_face(nodes, coarse.points[coarse.faces[face]], tolerance)
        values[selected] = scalar_values_3d(dirichlet, nodes[selected])
    face_bary, _ = triangle_quadrature(solution.degree + 3)
    for fine_face, parent in parent_faces.items():
        if parent in natural:
            continue
        owner = int(mesh.face_cells[fine_face, 0])
        bary = np.zeros((len(face_bary), 4))
        for j, vertex in enumerate(mesh.faces[fine_face]):
            bary[:, int(np.flatnonzero(mesh.cells[owner] == vertex)[0])] = face_bary[:, j]
        samples = face_bary @ mesh.points[mesh.faces[fine_face]]
        expected = scalar_values_3d(dirichlet, samples)
        opposite = int(np.flatnonzero(mesh.cell_faces[owner] == fine_face)[0])
        actual = (
            tetra_face_basis(solution.degree, bary, opposite_vertex=opposite) @ values[dofs[owner]]
        )
        scale = max(np.max(abs(expected)), np.max(abs(actual)), np.finfo(float).tiny)
        if np.max(abs(actual - expected)) > 1e-10 * scale:
            raise ValueError("Dirichlet data must be represented by the fine polynomial trace")
    local_values, offset = [], 0
    for fine, ids, original in zip(
        solution.local_meshes, local_dofs, solution.pressure, strict=True
    ):
        recovered = np.empty_like(original)
        recovered[ids] = values[dofs[offset : offset + len(fine.cells)]]
        local_values.append(recovered)
        offset += len(fine.cells)
    return ConformingPotential3D(mesh, solution.degree, values, tuple(local_values))


def _project(
    fine: TetraMesh, degree: int, data: FloatArray, bary: FloatArray, weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Project scalar samples onto the continuous macro-local Pm space."""
    if degree:
        dofs, nodes = tetra_nodal_space(fine, degree)
        basis, _ = tetra_basis(degree, bary)
        size = len(nodes)
    else:
        dofs, basis, size = np.zeros((len(fine.cells), 1), dtype=int), np.ones((len(bary), 1)), 1
    width = basis.shape[1]
    blocks = np.einsum("t,q,qi,qj->tij", fine.volumes, weights, basis, basis)
    matrix = sparse.coo_matrix(
        (
            blocks.ravel(),
            (np.repeat(dofs, width, axis=1).ravel(), np.tile(dofs, (1, width)).ravel()),
        ),
        shape=(size, size),
    ).tocsc()
    local = np.einsum("t,q,qi,tq->ti", fine.volumes, weights, basis, data)
    load = np.bincount(dofs.ravel(), weights=local.ravel(), minlength=size)
    coefficients = solve_linear(matrix, load)
    return coefficients[dofs] @ basis.T, load


@dataclass(frozen=True)
class Darcy3DEstimator:
    """Four physical terms with explicit equilibrium and material-bound diagnostics."""

    solution: Darcy3DSolution
    potential: ConformingPotential3D
    reconstruction: MomentFlux3DSolution
    flux_defect: FloatArray
    nonconformity: FloatArray
    divergence_defect: FloatArray
    oscillation: FloatArray
    equilibrium_defect: FloatArray
    ellipticity_lower_bounds: FloatArray
    convention: str
    quadrature_order: int

    @property
    def local_squared(self) -> FloatArray:
        """Return additive local squared indicators with the prescribed cross terms."""
        return (
            self.flux_defect + self.divergence_defect + self.oscillation
        ) ** 2 + self.nonconformity**2

    @property
    def total(self) -> float:
        """Return the root of the total squared indicator."""
        return float(np.sqrt(self.local_squared.sum()))

    def energy_error(self, exact_gradient: Any, order: int = 8) -> float:
        """Integrate the actual broken energy error using the physical SPD tensor."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for fine, coefficients in zip(
            self.solution.local_meshes, self.solution.pressure, strict=True
        ):
            dofs, _, _, gradient = tetra_tabulate(fine, self.solution.degree, bary)
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            actual = np.einsum("ti,tqia->tqa", coefficients[dofs], gradient)
            exact = vector_values_3d(exact_gradient, points.reshape(-1, 3)).reshape(actual.shape)
            tensor = tensor_values_3d(self.solution.permeability, points.reshape(-1, 3)).reshape(
                *points.shape[:2], 3, 3
            )
            total += np.einsum(
                "t,q,tqa,tqab,tqb->", fine.volumes, weights, actual - exact, tensor, actual - exact
            )
        return float(np.sqrt(total))


def estimate_darcy_error_3d(
    solution: Darcy3DSolution,
    *,
    degree: int = 1,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    ellipticity_lower_bound: Any = None,
    convention: Literal["energy", "published"] = "energy",
    quadrature_order: int = 7,
) -> Darcy3DEstimator:
    """Evaluate the dimension-independent energy decomposition on tetrahedra.

    Use the estimator of [Barrenechea et al. (2026)](https://doi.org/10.1137/24M1673073).

    Require local k>=ell+3 and ell<=m<=k for skeletal degree ell and RT order m. The degree offset
    is the spatial dimension in Theorem 5.2, not a dimension-independent value of two. The energy
    bound assumes exact integration, continuous-test equilibrium, represented boundary data, and
    certified ellipticity. Numerical quadrature supplies measured indicators, not interval-certified
    upper bounds.
    """
    ell = int(solution.skeleton.degrees.max())
    m = positive_int(degree, "RT degree", 0)
    validate_estimator_spaces(solution.degree, ell, m, 3)
    if convention not in ("energy", "published"):
        raise ValueError("convention must be energy or published")
    order = positive_int(quadrature_order, "quadrature order", max(solution.degree + 2, m + 2))
    coarse = solution.skeleton.mesh
    if convention == "published":
        alpha = np.ones(len(coarse.cells))
    else:
        if ellipticity_lower_bound is None:
            if callable(solution.permeability):
                raise ValueError("variable callbacks require a certified ellipticity_lower_bound")
            tensor = tensor_values_3d(solution.permeability, np.zeros((1, 3)))[0]
            ellipticity_lower_bound = np.linalg.eigvalsh(tensor).min()
        raw = np.asarray(ellipticity_lower_bound)
        if np.iscomplexobj(raw) or raw.shape not in ((), (len(coarse.cells),)):
            raise ValueError("ellipticity bound must be a scalar or one value per macrocell")
        alpha = np.broadcast_to(raw, (len(coarse.cells),)).astype(float)
        if not np.all(np.isfinite(alpha) & (alpha > 0)):
            raise ValueError("ellipticity bounds must be finite and positive")
    potential = recover_potential_3d(solution, dirichlet=dirichlet, neumann=neumann)
    face_bary, _ = triangle_quadrature(order)
    for face, datum in ({} if neumann is None else neumann).items():
        partition = solution.skeleton.face_partition(int(face))
        points = np.einsum("qi,sij->sqj", face_bary, partition @ coarse.points[coarse.faces[face]])
        expected = scalar_values_3d(datum, points.reshape(-1, 3)).reshape(points.shape[:2])
        coefficients = solution.hybrid.trace[solution.skeleton.dofs(face)].reshape(
            len(partition), -1
        )
        represented = coefficients @ solution.skeleton.basis(face, face_bary).T
        scale = max(np.max(abs(expected)), np.max(abs(represented)), np.finfo(float).tiny)
        if np.max(abs(expected - represented)) > 1e-10 * scale:
            raise ValueError("Neumann data must be represented by the skeletal trace")
    reconstruction = reconstruct_darcy_moments_3d(solution, degree=m, quadrature_order=order)
    bary, weights = tetrahedron_quadrature(order)
    terms = np.zeros((5, len(coarse.cells)))
    for cell, fine in enumerate(solution.local_meshes):
        dofs, _, _, gradient = tetra_tabulate(fine, solution.degree, bary)
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        tensor = tensor_values_3d(solution.permeability, points.reshape(-1, 3)).reshape(
            *points.shape[:2], 3, 3
        )
        if convention == "energy" and np.linalg.eigvalsh(tensor).min() < alpha[cell] * (1 - 1e-12):
            raise ValueError("ellipticity bound exceeds a sampled material eigenvalue")
        pressure = solution.pressure[cell]
        grad = np.einsum("ti,tqia->tqa", pressure[dofs], gradient)
        delta = np.einsum("ti,tqia->tqa", (pressure - potential.local_values[cell])[dofs], gradient)
        q, divergence = reconstruction.evaluate(cell, bary[:, 1:])
        source = scalar_values_3d(solution.source, points.reshape(-1, 3)).reshape(divergence.shape)
        projected_source, source_moments = _project(fine, m, source, bary, weights)
        projected_div, div_moments = _project(fine, m, divergence, bary, weights)
        defect = q + np.einsum("tqab,tqb->tqa", tensor, grad)
        inverse = (
            np.linalg.inv(tensor)
            if convention == "energy"
            else np.broadcast_to(np.eye(3), tensor.shape)
        )
        terms[0, cell] = np.sqrt(
            np.einsum("t,q,tqa,tqab,tqb->", fine.volumes, weights, defect, inverse, defect)
        )
        terms[1, cell] = np.sqrt(
            np.einsum("t,q,tqa,tqab,tqb->", fine.volumes, weights, delta, tensor, delta)
        )
        vertices = coarse.points[coarse.cells[cell]]
        diameter = np.linalg.norm(vertices[:, None] - vertices[None], axis=2).max()
        poincare = diameter / (np.pi * np.sqrt(alpha[cell]))
        terms[2, cell] = poincare * np.sqrt(
            np.einsum("t,q,tq->", fine.volumes, weights, (projected_div - divergence) ** 2)
        )
        terms[3, cell] = poincare * np.sqrt(
            np.einsum("t,q,tq->", fine.volumes, weights, (source - projected_source) ** 2)
        )
        terms[4, cell] = np.linalg.norm(source_moments - div_moments)
        scale = max(
            np.linalg.norm(source_moments),
            np.linalg.norm(div_moments),
            np.linalg.norm(q) * fine.volumes.sum(),
            np.finfo(float).tiny,
        )
        if terms[4, cell] > 1e-8 * scale:
            raise ValueError(
                "continuous-test equilibrium failed; resolve volume and face integration"
            )
    return Darcy3DEstimator(
        solution,
        potential,
        reconstruction,
        terms[0],
        terms[1],
        terms[2],
        terms[3],
        terms[4],
        alpha if convention == "energy" else np.empty(0),
        convention,
        order,
    )
