"""Physical field records in the executed H(div) stress and broken kinematic spaces."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pymhm.core.contracts import HybridSolution
from pymhm.core.validation import FloatArray
from pymhm.fem.hdiv.tensor_rt import _trace_map, tensor_rt_basis, tensor_rt_dofs
from pymhm.fem.scalar.quadrilateral import quadrilateral_quadrature
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.vector.stress import rigid_values as _rigid_values
from pymhm.fem.vector.stress_tensor import complete_rotation_basis as _rotation_basis
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.meshes.cartesian import CartesianMacroMesh


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
