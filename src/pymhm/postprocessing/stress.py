"""Physical field records in the executed H(div) stress and broken kinematic spaces."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.contracts import HybridSolution
from pymhm.core.validation import FloatArray
from pymhm.fem.hdiv.bdm_family import BDMFamily
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import reference_basis
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.vector.stress import rigid_values as _rigid_values
from pymhm.fem.vector.stress import traction_mapping as _traction_map
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class MixedElasticitySolution:
    """H(div) Cauchy stress, broken displacement and independent weak rotation.

    Stress coefficients have shape ``(family.size(mesh), 2)`` on each local mesh;
    displacement and rotation have shapes ``(cells,d,2)`` and ``(cells,d)``,
    where d is the scalar dimension of P(k+n-1).
    Rotation approximates half the asymmetry of the exact displacement gradient. The skeletal
    traction is ``-sigma n`` and stress symmetry holds through P(k+n-1) moments.
    """

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    stress: tuple[FloatArray, ...]
    displacement: tuple[FloatArray, ...]
    rotation: tuple[FloatArray, ...]
    hybrid: HybridSolution
    source: Any
    quadrature_order: int
    family: BDMFamily = BDMFamily()

    @property
    def displacement_degree(self) -> int:
        """Return the discontinuous displacement and rotation polynomial degree."""
        return self.family.polynomial_degree - 1

    def l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate the broken displacement error against an analytical field."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, field in zip(self.local_meshes, self.displacement, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            values = np.einsum(
                "qi,tia->tqa", reference_basis(self.displacement_degree, bary)[0], field
            )
            error = values - vector_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ (np.sum(error**2, axis=2) @ weights))
        return float(np.sqrt(total))

    def stress_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate the full Cauchy stress Frobenius error, including its skew part."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, field in zip(self.local_meshes, self.stress, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            value = exact(points.reshape(-1, 2)) if callable(exact) else exact
            target = np.broadcast_to(np.asarray(value), (points.shape[0] * points.shape[1], 2, 2))
            if np.iscomplexobj(target) or not np.isfinite(target).all():
                raise ValueError("exact stress must be real and finite")
            values, _ = self.family.evaluate(mesh, field, bary)
            error = values - target.reshape(values.shape)
            total += float(mesh.areas @ (np.sum(error**2, axis=(2, 3)) @ weights))
        return float(np.sqrt(total))

    def divergence_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate stress-divergence error, distinct from integrated force moments."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, field in zip(self.local_meshes, self.stress, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            _, divergence = self.family.evaluate(mesh, field, bary)
            target = vector_values(exact, points.reshape(-1, 2)).reshape(divergence.shape)
            total += float(mesh.areas @ (np.sum((divergence - target) ** 2, axis=2) @ weights))
        return float(np.sqrt(total))

    def rotation_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate rotation error with the convention q=(du_x/dy-du_y/dx)/2."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, field in zip(self.local_meshes, self.rotation, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            values = field @ reference_basis(self.displacement_degree, bary)[0].T
            error = values - scalar_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ (error**2 @ weights))
        return float(np.sqrt(total))

    def fine_force_residuals(self) -> tuple[FloatArray, ...]:
        """Return displacement-space moments of div(sigma)+f in every fine cell and component."""
        bary, weights = triangle_quadrature(self.quadrature_order)
        residuals = []
        for mesh, field in zip(self.local_meshes, self.stress, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            _, divergence = self.family.evaluate(mesh, field, bary)
            force = vector_values(self.source, points.reshape(-1, 2)).reshape(divergence.shape)
            residuals.append(
                np.einsum(
                    "q,qi,tqa,t->tia",
                    weights,
                    reference_basis(self.displacement_degree, bary)[0],
                    divergence + force,
                    mesh.areas,
                )
            )
        return tuple(residuals)

    def weak_symmetry_residuals(self) -> tuple[FloatArray, ...]:
        """Return rotation-space moments of sigma_xy-sigma_yx, without symmetrizing the field."""
        bary, weights = triangle_quadrature(self.family.polynomial_degree + 2)
        result = []
        for mesh, field in zip(self.local_meshes, self.stress, strict=True):
            values, _ = self.family.evaluate(mesh, field, bary)
            result.append(
                np.einsum(
                    "q,qi,tq,t->ti",
                    weights,
                    reference_basis(self.displacement_degree, bary)[0],
                    values[..., 0, 1] - values[..., 1, 0],
                    mesh.areas,
                )
            )
        return tuple(result)

    def normal_traction_residuals(self) -> tuple[FloatArray, ...]:
        """Return moments of sigma n plus the signed macro traction on fine faces."""
        residuals = []
        for cell, (mesh, stress) in enumerate(zip(self.local_meshes, self.stress, strict=True)):
            count = self.family.degree + 1
            rows = (count * mesh.boundary_faces[:, None] + np.arange(count)).ravel()
            expected = (
                _traction_map(self.skeleton.mesh, cell, mesh, self.skeleton, self.family)
                @ (self.hybrid.trace[self.skeleton.cell_dofs(cell)])
            )
            residuals.append(stress[rows] + expected.reshape(-1, 2))
        return tuple(residuals)

    def equilibrium_residuals(self) -> FloatArray:
        """Return macro force and moment defects for the negative-traction skeleton."""
        bary, weights = triangle_quadrature(self.quadrature_order)
        result = np.zeros((len(self.local_meshes), 3))
        coarse = self.skeleton.mesh
        for cell, fine in enumerate(self.local_meshes):
            center = coarse.points[coarse.cells[cell]].mean(axis=0)
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            force = vector_values(self.source, points.reshape(-1, 2)).reshape(points.shape)
            result[cell] -= np.einsum(
                "t,q,tqa,tqak->k", fine.areas, weights, force, _rigid_values(points, center)
            )
            for side, face in enumerate(coarse.cell_faces[cell]):
                space = self.skeleton.faces[face]
                parameter, w = space.quadrature(max(4, self.family.degree + 2))
                start, end = coarse.points[coarse.faces[face]]
                points_face = start + parameter[:, None] * (end - start)
                traction = space.evaluate(parameter) @ self.hybrid.trace[
                    self.skeleton.dofs(int(face))
                ].reshape(-1, 2)
                result[cell] += (
                    coarse.signs[cell, side]
                    * coarse.lengths[face]
                    * np.einsum("q,qa,qak->k", w, traction, _rigid_values(points_face, center))
                )
        return result
