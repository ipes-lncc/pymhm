"""Physical displacement and raw Cauchy stress in the executed nodal coordinates."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.triangle import (
    element_tabulate,
)
from pymhm.materials.elasticity import KELVIN_BASIS_2D as _KELVIN
from pymhm.materials.elasticity import constitutive_values as constitutive_values
from pymhm.postprocessing.solutions import VectorSolution


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
