"""Physical displacement and raw Cauchy stress in the executed nodal coordinates."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.contracts import HybridSolution
from pymhm.core.validation import FloatArray
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_tabulate, tetrahedron_quadrature
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.fem.vector.elasticity_3d import KELVIN_BASIS_3D as _KELVIN3
from pymhm.fem.vector.elasticity_3d import _component as _component
from pymhm.fem.vector.elasticity_3d import rigid_modes_3d as rigid_modes_3d
from pymhm.materials.elasticity import constitutive_values_3d as constitutive_values_3d
from pymhm.materials.evaluation import vector_values_3d
from pymhm.meshes.tetrahedron import TetraMesh


@dataclass(frozen=True)
class Elasticity3DSolution:
    """Broken tetrahedral displacement and raw symmetric three-dimensional Cauchy stress.

    The scalar triangular skeleton defines the geometry and P1 face modes;
    ``hybrid.trace`` interleaves three negative-traction components per scalar
    skeleton coefficient. Raw stress is not automatically H(div)-conforming.
    """

    skeleton: TriangularSkeleton
    local_meshes: tuple[TetraMesh, ...]
    values: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    constitutive: Any
    lame_lambda: Any
    lame_mu: Any
    source: Any
    quadrature_order: int

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return displacement, full gradient and symmetric stress at barycentric points."""
        fine = self.local_meshes[cell]
        dofs, _, basis, gradient = tetra_tabulate(fine, self.degree, bary)
        values = self.values[cell][dofs]
        displacement = np.einsum("qi,tia->tqa", basis, values)
        grad = np.einsum("tqib,tia->tqab", gradient, values)
        physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        stiffness = constitutive_values_3d(
            self.constitutive,
            physical.reshape(-1, 3),
            lame_lambda=self.lame_lambda,
            lame_mu=self.lame_mu,
        )
        strain = np.einsum("aij,tqij->tqa", _KELVIN3, grad).reshape(-1, 6)
        sigma = np.einsum("nab,nb,aij->nij", stiffness, strain, _KELVIN3).reshape(
            *physical.shape[:2], 3, 3
        )
        return displacement, grad, sigma

    def errors(self, displacement: Any, stress: Any, order: int = 7) -> dict[str, float]:
        """Integrate displacement L2 and raw-stress Frobenius errors over the physical volume."""
        bary, weights = tetrahedron_quadrature(order)
        totals = np.zeros(2)
        for cell, fine in enumerate(self.local_meshes):
            physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells]).reshape(-1, 3)
            u, _, sigma = self.evaluate(cell, bary)
            exact_u = vector_values_3d(displacement, physical).reshape(u.shape)
            raw = stress(physical) if callable(stress) else stress
            if np.iscomplexobj(raw) or not np.isfinite(raw).all():
                raise ValueError("exact stress must be finite and real")
            exact_sigma = np.broadcast_to(raw, (len(physical), 3, 3)).reshape(sigma.shape)
            totals[0] += fine.volumes @ (np.sum((u - exact_u) ** 2, axis=-1) @ weights)
            totals[1] += fine.volumes @ (
                np.sum((sigma - exact_sigma) ** 2, axis=(-1, -2)) @ weights
            )
        return dict(zip(("displacement_l2", "stress_l2"), np.sqrt(totals), strict=True))

    def equilibrium_residuals(self) -> FloatArray:
        """Return three force and three moment defects from body force and skeletal traction."""
        mesh = self.skeleton.mesh
        bary, weights = tetrahedron_quadrature(self.quadrature_order)
        fbary, fweights = triangle_quadrature(self.quadrature_order)
        result = np.zeros((len(mesh.cells), 6))
        for cell, fine in enumerate(self.local_meshes):
            center = mesh.points[mesh.cells[cell]].mean(axis=0)
            physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            force = vector_values_3d(self.source, physical.reshape(-1, 3)).reshape(physical.shape)
            result[cell] -= np.einsum(
                "t,q,tqa,tqak->k", fine.volumes, weights, force, rigid_modes_3d(physical, center)
            )
            for side, face in enumerate(mesh.cell_faces[cell]):
                partitions = self.skeleton.face_partition(int(face))
                subvertices = partitions @ mesh.points[mesh.faces[face]]
                points = np.einsum("qi,sij->sqj", fbary, subvertices)
                coefficients = np.array(
                    [
                        self.hybrid.trace.reshape(-1, 3)[
                            self.skeleton.subtriangle_dofs(int(face), segment)
                        ]
                        for segment in range(len(partitions))
                    ]
                )
                traction = np.einsum(
                    "qi,sia->sqa", self.skeleton.basis(int(face), fbary), coefficients
                )
                result[cell] += (
                    mesh.signs[cell, side]
                    * mesh.areas[face]
                    * np.einsum(
                        "s,q,sqa,sqak->k",
                        self.skeleton.face_weights(int(face)),
                        fweights,
                        traction,
                        rigid_modes_3d(points, center),
                    )
                )
        return result
