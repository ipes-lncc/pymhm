"""Locally equilibrated H(div) fluxes from a primal Darcy solution."""

from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse

from pymhm.darcy import DarcySolution
from pymhm.elements import (
    face_integration,
    rt0_evaluate,
    rt0_operators,
    triangle_quadrature,
    vector_values,
)
from pymhm.hybrid import LocalProblem
from pymhm.lagrange import tabulate
from pymhm.mesh import FloatArray, TriangleMesh


@dataclass(frozen=True)
class EquilibratedFlux:
    """RT0 fluxes, preserving macro traces and balancing each fine source moment."""

    meshes: tuple[TriangleMesh, ...]
    coefficients: tuple[FloatArray, ...]
    source_moments: tuple[FloatArray, ...]

    def conservation_residuals(self) -> tuple[FloatArray, ...]:
        """Return cellwise integrated divergence minus the assembled source."""
        return tuple(
            np.sum(q[mesh.cell_faces] * mesh.signs, axis=1) - f
            for mesh, q, f in zip(self.meshes, self.coefficients, self.source_moments, strict=True)
        )

    def l2_error(self, exact: object, order: int = 5) -> float:
        """Integrate the vector flux error with independent quadrature."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, flux in zip(self.meshes, self.coefficients, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            difference = rt0_evaluate(mesh, flux, bary) - vector_values(
                exact, points.reshape(-1, 2)
            ).reshape(len(mesh.cells), len(weights), 2)
            total += float(mesh.areas @ (np.sum(difference**2, axis=2) @ weights))
        return float(np.sqrt(total))


def equilibrate_flux(solution: DarcySolution, *, solver: str = "scipy") -> EquilibratedFlux:
    """Minimize weighted RT0 distance to the raw flux subject to exact balances.

    On each macrocell, minimize ``||q_h + K grad(p_h)||_(K^-1)`` subject to
    ``div(q_h)|_t = average_t(f)`` and the prescribed skeleton normal flux.
    This constrained minimization is distinct from the published face-moment
    reconstruction; no equivalence to that particular operator is claimed.
    Degree-zero skeleton segments must coincide with fine boundary faces.
    """
    if solution.formulation != "primal":
        raise ValueError("equilibration expects a primal Darcy solution")
    skeleton = solution.skeleton
    bary, weights = triangle_quadrature(max(4, solution.degree + 2))
    coefficients, moments = [], []
    for cell, (mesh, pressure) in enumerate(
        zip(solution.local_meshes, solution.pressure, strict=True)
    ):
        for face in skeleton.mesh.cell_faces[cell]:
            space = skeleton.faces[face]
            start, end = skeleton.mesh.points[skeleton.mesh.faces[face]]
            parameter = (mesh.points - start) @ (end - start) / np.sum((end - start) ** 2)
            projected = start + parameter[:, None] * (end - start)
            boundary = np.all(np.isclose(mesh.points, projected, atol=1e-12), axis=1)
            if any(space.degrees) or any(
                not np.any(np.isclose(parameter[boundary], cut, atol=1e-12)) for cut in space.breaks
            ):
                raise ValueError("RT0 equilibration requires degree-zero aligned trace segments")
        mass, divergence, force = rt0_operators(
            mesh, solution.permeability, solution.source, solution.quadrature_order
        )
        nq, npres, nb = len(mesh.faces), len(mesh.cells), len(mesh.boundary_faces)
        selector = sparse.coo_matrix(
            (np.ones(nb), (mesh.boundary_faces, np.arange(nb))), shape=(nq, nb)
        ).tocsc()
        matrix = sparse.bmat(
            [
                [mass, -divergence.T, selector],
                [-divergence, None, sparse.csc_matrix((npres, nb))],
                [selector.T, sparse.csc_matrix((nb, npres)), None],
            ],
            format="csc",
        )
        dofs, _, _, gradients, _ = tabulate(mesh, solution.degree, bary)
        grad = np.einsum("ti,tqia->tqa", pressure[dofs], gradients)
        vertices = mesh.points[mesh.cells]
        points = np.einsum("qi,tij->tqj", bary, vertices)
        rt_basis = (points[:, :, None, :] - vertices[:, None, [2, 0, 1], :]) / (
            2 * mesh.areas[:, None, None, None]
        )
        local_rhs = -np.einsum(
            "q,tqia,tqa,t,ti->ti", weights, rt_basis, grad, mesh.areas, mesh.signs
        )
        rhs_flux = np.bincount(mesh.cell_faces.ravel(), weights=local_rhs.ravel(), minlength=nq)
        flux_map = face_integration(skeleton.mesh, cell, mesh, skeleton)[1]
        boundary_flux = flux_map @ solution.hybrid.trace[skeleton.cell_dofs(cell)]
        load = np.r_[rhs_flux, -force, boundary_flux]
        kernel = np.r_[np.zeros(nq), np.ones(npres + nb)][:, None]
        constraint = np.r_[np.zeros(nq), mesh.areas, np.zeros(nb)][:, None]
        local = LocalProblem(
            matrix, np.empty((len(load), 0)), load, np.array([], dtype=int), kernel, constraint
        )
        result = local.condense(solver).source
        defect = linalg.norm(matrix @ result - load)
        scale = max(
            linalg.norm(load),
            linalg.norm(abs(matrix) @ np.abs(result)),
            np.finfo(float).tiny,
        )
        if defect > 1e-9 * scale:
            raise ValueError("incompatible skeleton flux and source for equilibration")
        coefficients.append(result[:nq])
        moments.append(force)
    return EquilibratedFlux(solution.local_meshes, tuple(coefficients), tuple(moments))
