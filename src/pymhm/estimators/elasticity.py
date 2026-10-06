"""Face residuals of the one-level primal elasticity estimator.

The estimator follows [Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046).

The reliability theorem assumes exact local Neumann lifts. With computed local finite elements this
is a face indicator, not a certified estimator of the full two-level error. In particular, an
unresolved continuous local field can have a zero jump indicator and nonzero interior error.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm._legacy.models.elasticity.primal import PrimalElasticitySolution, constitutive_values
from pymhm.core.validation import FloatArray, positive_int
from pymhm.estimators.flow import _boundary_intervals, _edge_points
from pymhm.fem.scalar.operators import p1_geometry
from pymhm.fem.scalar.triangle import nodal_space, reference_basis
from pymhm.materials.evaluation import vector_values


@dataclass(frozen=True)
class PrimalElasticityIndicator:
    """Per-segment displacement jumps scaled by the declared ellipticity constant.

    ``face_squared`` contains c_min²/H_F times the squared displacement residual.
    Internal faces have the factor one half in the displacement jump and count
    twice in ``eta``. The denominator is the original macroface length. The
    object records no assertion that finite local lifts satisfy the one-level
    reliability theorem; that additional hypothesis is not numerically inferred.
    """

    solution: PrimalElasticitySolution
    face_squared: tuple[FloatArray, ...]
    c_min: float
    quadrature_order: int

    @property
    def eta(self) -> float:
        """Return the face indicator with the macrocell-boundary multiplicity.

        The indicator follows
        [Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046).
        """
        return float(
            np.sqrt(
                sum(
                    value.sum() * (1 + (neighbors[1] >= 0))
                    for value, neighbors in zip(
                        self.face_squared, self.solution.skeleton.mesh.face_cells, strict=True
                    )
                )
            )
        )


def _displacement(
    solution: PrimalElasticitySolution, macro: int, points: FloatArray, owners: np.ndarray
) -> FloatArray:
    """Evaluate one-sided displacement without averaging across local or macro faces."""
    fine = solution.local_meshes[macro]
    gradients, _ = p1_geometry(fine)
    bary = np.einsum("nva,na->nv", gradients[owners], points - fine.points[fine.cells[owners, 0]])
    bary[:, 0] += 1
    basis = reference_basis(solution.degree, bary)[0]
    dofs = nodal_space(fine, solution.degree)[0][owners]
    return np.einsum("qi,qia->qa", basis, solution.values[macro][dofs])


def estimate_primal_elasticity_error(
    solution: PrimalElasticitySolution,
    *,
    dirichlet: Any = (0.0, 0.0),
    c_min: float,
    full_dirichlet: bool,
    quadrature_order: int = 8,
) -> PrimalElasticityIndicator:
    """Evaluate the face estimator under an explicit full-Dirichlet contract.

    Use equations (5.4)--(5.6) of
    [Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046).

    Supply c_min such that A(x) epsilon:epsilon >= c_min² |epsilon|² everywhere. The supplied bound
    is checked at face integration points, but only the user can establish its global validity for a
    generic callable. The theorem's constants depend on the material; no incompressibility-uniform
    or unit reliability constant is claimed. For finite local approximations, refine the local space
    separately to assess the unmeasured local consistency error.
    """
    if full_dirichlet is not True:
        raise ValueError("the L17 face estimator requires full Dirichlet boundaries")
    if not np.isfinite(c_min) or c_min <= 0:
        raise ValueError("c_min must be finite and positive")
    order = positive_int(quadrature_order, "quadrature_order", solution.degree + 1)
    skeleton = solution.skeleton
    if any(min(face.degrees) < 1 for face in skeleton.faces):
        raise ValueError("L17 requires trace spaces containing rigid-motion traces (at least P1)")
    coarse = skeleton.mesh
    lengths = coarse.lengths
    lengths.setflags(write=False)
    boundaries = [
        _boundary_intervals(coarse, cell, fine, lengths)
        for cell, fine in enumerate(solution.local_meshes)
    ]
    result = []
    for face, neighbors in enumerate(coarse.face_cells):
        space = skeleton.faces[face]
        entries = [boundaries[int(cell)][face] for cell in neighbors if cell >= 0]
        values = []
        for lo, hi in zip(space.breaks[:-1], space.breaks[1:], strict=True):
            cuts = np.unique(
                np.r_[
                    lo,
                    hi,
                    [
                        end
                        for sides in entries
                        for entry in sides
                        for end in entry[:2]
                        if lo < end < hi
                    ],
                ]
            )
            parameter, points, weights = _edge_points(
                coarse.points[coarse.faces[face]], cuts, order
            )
            fields = []
            for cell, sides in zip(neighbors, entries, strict=False):
                owners = np.array(
                    [
                        next(
                            entry[2] for entry in sides if entry[0] - 1e-13 <= t <= entry[1] + 1e-13
                        )
                        for t in parameter
                    ]
                )
                fields.append(_displacement(solution, int(cell), points, owners))
            residual = (
                (fields[0] - fields[1]) / 2
                if len(fields) == 2
                else vector_values(dirichlet, points) - fields[0]
            )
            stiffness = constitutive_values(
                solution.constitutive,
                points,
                lame_lambda=solution.lame_lambda,
                lame_mu=solution.lame_mu,
            )
            if np.any(
                np.linalg.eigvalsh(stiffness)[:, 0] < c_min**2 * (1 - 64 * np.finfo(float).eps)
            ):
                raise ValueError("c_min exceeds a sampled constitutive ellipticity bound")
            # Physical edge integration contributes H_F, canceling the original
            # H_F denominator even on subdivided segments; weights retain hi-lo.
            values.append(float(c_min**2 * (weights @ np.sum(residual**2, axis=1))))
        result.append(np.asarray(values))
    return PrimalElasticityIndicator(solution, tuple(result), c_min, order)
