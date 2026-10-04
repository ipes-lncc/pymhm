"""Polynomial H(div) families with independently enriched interior triangle modes.

BDM(k,n) retains normal degree k and every interior mode of BDM(k+n).
The choices n=0,1,2 realize the BDM, BDM-plus and BDM-double-plus families.
All degrees of freedom are physical normal moments or reference cell moments;
contravariant Piola mapping preserves their duality.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.hdiv.reference import vector_tabulation
from pymhm.fem.reference import legendre_values, monomial_tabulation
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


def _powers(degree: int) -> tuple[tuple[int, int], ...]:
    """Enumerate complete scalar monomials in total-degree order."""
    return tuple((total - j, j) for total in range(degree + 1) for j in range(total + 1))


def _polynomials(points: FloatArray, degree: int) -> tuple[FloatArray, FloatArray]:
    """Tabulate native Basix BDM candidates before the declared moment transform."""
    return vector_tabulation("BDM", "triangle", degree, points)


@cache
def _full_dual(degree: int) -> FloatArray:
    """Invert the complete BDM moment matrix using a Nedelec cell test space."""
    count = (degree + 1) * (degree + 2)
    face_count = 3 * (degree + 1)
    dual = np.zeros((count, count))
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x, weights = leggauss(degree + 2)
    for side in range(3):
        start, end = vertices[side], vertices[(side + 1) % 3]
        tangent = end - start
        normal_measure = np.array([tangent[1], -tangent[0]])
        values = _polynomials(start + (x[:, None] + 1) * tangent / 2, degree)[0]
        dual[side * (degree + 1) : (side + 1) * (degree + 1)] = legendre_values(x, degree).T @ (
            weights[:, None] / 2 * (values @ normal_measure)
        )
    if degree > 1:
        bary, weights = triangle_quadrature(degree + 2)
        points = bary[:, 1:]
        scalar = monomial_tabulation(points, _powers(degree - 2), nderiv=0)[0]
        tests = [
            scalar[:, i, None] * np.eye(2)[axis]
            for axis in range(2)
            for i in range(scalar.shape[1])
        ]
        # Homogeneous degree k-2 times x-perp completes Nedelec(k-2).
        for i in range(degree - 1):
            homogeneous = scalar[:, len(_powers(degree - 3)) + i]
            tests.append(homogeneous[:, None] * np.column_stack((-points[:, 1], points[:, 0])))
        dual[face_count:] = np.einsum(
            "q,iqa,qja->ij", weights / 2, np.asarray(tests), _polynomials(points, degree)[0]
        )
    return np.linalg.solve(dual, np.eye(count))


@dataclass(frozen=True)
class BDMFamily:
    """Triangle vector polynomial family with normal degree k and interior degree k+n.

    ``degree`` is at least one; ``enrichment`` is a nonnegative integer. The complete
    polynomial degree is their sum, divergence has degree one less, and the
    normal trace retains degree k. Each fine face has k+1 moments and each cell
    has (k+n)^2-1 interior moments. Removing higher face modes leaves the full
    zero-normal bubble subspace, rather than truncating component monomials.
    """

    degree: int = 2
    enrichment: int = 0

    def __post_init__(self) -> None:
        """Validate the finite polynomial family without imposing a tested-degree cap."""
        positive_int(self.degree, "degree")
        positive_int(self.enrichment, "enrichment", 0)

    @property
    def polynomial_degree(self) -> int:
        """Return the complete interior vector polynomial degree."""
        return self.degree + self.enrichment

    @property
    def interior_size(self) -> int:
        """Return the number of cell moments per vector."""
        return self.polynomial_degree**2 - 1

    @property
    def local_size(self) -> int:
        """Return the total number of vector basis functions in one triangle."""
        return 3 * (self.degree + 1) + self.interior_size

    def size(self, mesh: TriangleMesh) -> int:
        """Return the global conforming vector dimension on a local mesh."""
        return (self.degree + 1) * len(mesh.faces) + self.interior_size * len(mesh.cells)

    def dofs(self, mesh: TriangleMesh) -> IntArray:
        """Return global-oriented face moments followed by private cell moments."""
        count = self.degree + 1
        edges = (count * mesh.cell_faces[:, :, None] + np.arange(count)).reshape(
            len(mesh.cells), 3 * count
        )
        interior = (
            count * len(mesh.faces)
            + self.interior_size * np.arange(len(mesh.cells))[:, None]
            + np.arange(self.interior_size)
        )
        return np.column_stack((edges, interior))

    def basis(self, mesh: TriangleMesh, barycentric: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate Piola vector bases and divergence at common or cellwise points.

        Barycentric coordinates may have shape (q,3) or (cells,q,3), allowing
        separate material-intersection quadrature without changing the basis.
        Returned arrays have shape (cells,q,dofs,2) and (cells,q,dofs).
        """
        if np.iscomplexobj(barycentric):
            raise ValueError("barycentric coordinates must be real")
        bary = np.asarray(barycentric, dtype=float)
        if (
            bary.ndim not in (2, 3)
            or bary.shape[-1] != 3
            or (bary.ndim == 3 and len(bary) != len(mesh.cells))
            or not np.isfinite(bary).all()
            or not np.allclose(bary.sum(axis=-1), 1, atol=1e-13, rtol=0)
        ):
            raise ValueError(
                "barycentric coordinates require finite shape (q,3) or (cells,q,3) and sum one"
            )
        degree = self.polynomial_degree
        selected = np.r_[
            np.concatenate([np.arange(self.degree + 1) + side * (degree + 1) for side in range(3)]),
            np.arange(3 * (degree + 1), (degree + 1) * (degree + 2)),
        ]
        coefficients = _full_dual(degree)[:, selected]
        cellwise = np.broadcast_to(bary, (len(mesh.cells), *bary.shape)) if bary.ndim == 2 else bary
        polynomial, derivative = _polynomials(cellwise.reshape(-1, 3)[:, 1:], degree)
        reference = np.einsum("qja,ji->qia", polynomial, coefficients).reshape(
            len(mesh.cells), -1, self.local_size, 2
        )
        reference_div = (derivative @ coefficients).reshape(len(mesh.cells), -1, self.local_size)
        vertices = mesh.points[mesh.cells]
        jacobian = (vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2)
        orientation = np.ones((len(mesh.cells), self.local_size))
        orientation[:, : 3 * (self.degree + 1)] = (
            mesh.signs[:, :, None] ** np.arange(1, self.degree + 2)
        ).reshape(len(mesh.cells), -1)
        inverse_det = 1 / (2 * mesh.areas)
        return (
            np.einsum("tab,tqib,ti,t->tqia", jacobian, reference, orientation, inverse_det),
            np.einsum("tqi,ti,t->tqi", reference_div, orientation, inverse_det),
        )

    def evaluate(
        self, mesh: TriangleMesh, coefficients: FloatArray, barycentric: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Evaluate a conforming vector or row-wise tensor and its divergence."""
        coefficients = np.asarray(coefficients)
        if (
            coefficients.ndim not in (1, 2)
            or coefficients.shape[0] != self.size(mesh)
            or np.iscomplexobj(coefficients)
            or not np.isfinite(coefficients).all()
        ):
            raise ValueError("BDM coefficients require a finite real vector or row-wise tensor")
        basis, divergence = self.basis(mesh, barycentric)
        local = coefficients[self.dofs(mesh)]
        return (
            np.einsum("tqia,ti...->tq...a", basis, local),
            np.einsum("tqi,ti...->tq...", divergence, local),
        )

    def validate_trace(self, skeleton: SkeletonSpace, refinement: int) -> None:
        """Require normal degrees no greater than k and fine-edge-aligned segments."""
        for face in skeleton.faces:
            if max(face.degrees) > self.degree or not np.allclose(
                np.array(face.breaks) * refinement,
                np.round(np.array(face.breaks) * refinement),
                atol=1e-12,
                rtol=0,
            ):
                raise ValueError(
                    "BDM trace degrees exceed k or segments are not aligned with fine edges"
                )

    def trace_map(
        self, mesh: TriangleMesh, cell: int, fine: TriangleMesh, skeleton: SkeletonSpace
    ) -> FloatArray:
        """Map signed scalar macro tractions to the oriented fine normal moments."""
        count = self.degree + 1
        result = np.zeros(
            (
                count * len(fine.boundary_faces),
                sum(skeleton.faces[f].size for f in mesh.cell_faces[cell]),
            )
        )
        x, weights = leggauss(self.degree + 2)
        parameter, weights = (x + 1) / 2, weights / 2
        fine_basis = legendre_values(x, self.degree)
        offset = 0
        for side, face in enumerate(mesh.cell_faces[cell]):
            space = skeleton.faces[face]
            start, end = mesh.points[mesh.faces[face]]
            tangent = end - start
            for row, edge in enumerate(fine.boundary_faces):
                coordinates = fine.points[fine.faces[edge]]
                t = (coordinates - start) @ tangent / (tangent @ tangent)
                if (
                    not np.allclose(coordinates, start + t[:, None] * tangent, atol=1e-12, rtol=0)
                    or min(t) < -1e-12
                    or max(t) > 1 + 1e-12
                ):
                    continue
                macro_parameter = np.clip(t[0] + parameter * (t[1] - t[0]), 0, 1)
                result[count * row : count * (row + 1), offset : offset + space.size] = (
                    fine.lengths[edge]
                    * mesh.signs[cell, side]
                    * fine_basis.T
                    @ (weights[:, None] * space.evaluate(macro_parameter))
                )
            offset += space.size
        return result
