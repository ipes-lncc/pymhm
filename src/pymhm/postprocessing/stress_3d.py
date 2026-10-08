"""Physical field records in the executed H(div) stress and broken kinematic spaces."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.contracts import HybridSolution
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.hdiv.family_3d import (
    HDiv3DFamily,
    cell_quadrature,
)
from pymhm.fem.hdiv.mixed_3d import hdiv3d_trace_mapping as _trace_mapping
from pymhm.fem.traces.traction_3d import TractionSkeleton3D
from pymhm.fem.vector.stress_3d import (
    boundary_selector,
)
from pymhm.materials.evaluation import vector_values_3d
from pymhm.meshes.mixed import AffineMixedMesh, hdiv3d_basis, hdiv3d_dofs


@dataclass(frozen=True)
class MixedElasticity3DSolution:
    """Row-wise H(div) stress, DG displacement and axial weak rotation on local tetrahedra.

    Rotation is ((du_y/dz-du_z/dy), (du_z/dx-du_x/dz),
    (du_x/dy-du_y/dx))/2. Symmetry holds in DG P_(k-1) moments, not pointwise.
    Stress arrays have shape (Hdiv DOFs,3), with columns denoting stress rows;
    displacement and rotation have shape (fine cells,scalar modes,3).
    """

    skeleton: TractionSkeleton3D
    family: HDiv3DFamily
    local_meshes: tuple[AffineMixedMesh, ...]
    stress: tuple[FloatArray, ...]
    displacement: tuple[FloatArray, ...]
    rotation: tuple[FloatArray, ...]
    hybrid: HybridSolution
    source: Any
    quadrature_order: int

    def evaluate(
        self, cell: int, points: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Evaluate displacement, full stress, axial rotation and divergence at reference points."""
        fine = self.local_meshes[cell]
        vectors, divergence, scalar = hdiv3d_basis(fine, self.family, points)
        stress = self.stress[cell][hdiv3d_dofs(fine, self.family)]
        return (
            np.einsum("qi,tia->tqa", scalar, self.displacement[cell]),
            np.einsum("tqib,tia->tqab", vectors, stress),
            np.einsum("qi,tia->tqa", scalar, self.rotation[cell]),
            np.einsum("tqi,tia->tqa", divergence, stress),
        )

    def errors(
        self, displacement: Any, stress: Any, rotation: Any, *, order: int = 6
    ) -> dict[str, float]:
        """Integrate displacement, Frobenius stress and axial-rotation physical L2 errors."""
        xi, weights = cell_quadrature("tetrahedron", positive_int(order, "error quadrature order"))
        total = np.zeros(3)
        for cell, fine in enumerate(self.local_meshes):
            points = fine.geometry(xi).reshape(-1, 3)
            values = self.evaluate(cell, xi)
            target_stress = stress(points) if callable(stress) else stress
            if np.iscomplexobj(target_stress) or not np.isfinite(target_stress).all():
                raise ValueError("exact stress must be real and finite")
            targets = (
                vector_values_3d(displacement, points).reshape(values[0].shape),
                np.broadcast_to(target_stress, (len(points), 3, 3)).reshape(values[1].shape),
                vector_values_3d(rotation, points).reshape(values[2].shape),
            )
            for i, target in enumerate(targets):
                delta = (values[i] - target).reshape(len(fine.cells), len(xi), -1)
                total[i] += float(fine.determinants @ (np.sum(delta**2, axis=2) @ weights))
        return dict(
            zip(
                ("displacement_l2", "stress_l2", "rotation_l2"),
                np.sqrt(total).tolist(),
                strict=True,
            )
        )

    def fine_force_residuals(self) -> tuple[FloatArray, ...]:
        """Return every DG displacement moment of div(sigma)+f on every fine tetrahedron."""
        xi, weights = cell_quadrature("tetrahedron", self.quadrature_order)
        scalar = self.family.tabulate(xi)[2]
        result = []
        for cell, fine in enumerate(self.local_meshes):
            divergence = self.evaluate(cell, xi)[3]
            force = vector_values_3d(self.source, fine.geometry(xi).reshape(-1, 3)).reshape(
                divergence.shape
            )
            result.append(
                np.einsum("t,q,qi,tqa->tia", fine.determinants, weights, scalar, divergence + force)
            )
        return tuple(result)

    def weak_symmetry_residuals(self) -> tuple[FloatArray, ...]:
        """Return the three independent skew-stress DG moments without symmetrizing the field."""
        xi, weights = cell_quadrature("tetrahedron", self.family.pressure_degree + 3)
        scalar = self.family.tabulate(xi)[2]
        result = []
        for cell, fine in enumerate(self.local_meshes):
            stress = self.evaluate(cell, xi)[1]
            asym = np.stack(
                [stress[..., i, j] - stress[..., j, i] for i, j in ((1, 2), (2, 0), (0, 1))],
                axis=-1,
            )
            result.append(np.einsum("t,q,qi,tqa->tia", fine.determinants, weights, scalar, asym))
        return tuple(result)

    def normal_traction_residuals(self) -> tuple[FloatArray, ...]:
        """Return canonical normal moments of sigma n plus the physical skeletal traction."""
        result = []
        for cell, fine in enumerate(self.local_meshes):
            selector = boundary_selector(fine, self.family, self.stress[cell].size)
            mapping = np.kron(
                _trace_mapping(self.skeleton.scalar, cell, fine, self.family.normal_degree),
                np.eye(3),
            )
            expected = mapping @ self.hybrid.trace[self.skeleton.cell_dofs(cell)]
            result.append(selector.T @ self.stress[cell].ravel() + expected)
        return tuple(result)
