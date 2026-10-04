"""H(div) families with independent face order and complete interior enrichment.

Tetrahedra retain all zero-normal BDM_(p+1) bubbles with normal Pk, k<=p+1.
Prisms use horizontal [P_(p+1)(triangle) tensor P_p(z)]^2 and vertical
P_p(triangle) tensor P_(p+1)(z), restricted to triangular Pk and rectangular
Qk normals, k<=p. Divergence is the complete Pp or W_pp pressure space.
"""

from dataclasses import dataclass
from functools import cache
from itertools import product
from typing import Literal

import numpy as np
from scipy.linalg import null_space

from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.reference import (
    monomial_tabulation,
    orthogonal_polynomial_tabulation,
    simplex_lagrange_tabulation,
    tensor_lagrange_tabulation,
)
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetrahedron_quadrature
from pymhm.meshes.hexahedron import cube_quadrature

CellKind = Literal["tetrahedron", "prism"]


def reference_vertices(kind: CellKind) -> FloatArray:
    """Return the reference vertices in the declared tetrahedral or prism order."""
    if kind == "tetrahedron":
        return np.vstack((np.zeros(3), np.eye(3)))
    if kind == "prism":
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
    raise ValueError("cell kind must be tetrahedron or prism")


def reference_faces(kind: CellKind) -> tuple[tuple[int, ...], ...]:
    """Return triangle corners or tensor-ordered quadrilateral corners of each face."""
    reference_vertices(kind)
    if kind == "tetrahedron":
        return tuple(tuple(j for j in range(4) if j != i) for i in range(4))
    return ((0, 1, 2), (3, 4, 5), (0, 1, 3, 4), (1, 2, 4, 5), (2, 0, 5, 3))


def cell_quadrature(kind: CellKind, order: int) -> tuple[FloatArray, FloatArray]:
    """Return positive reference volume quadrature, with physical reference weights."""
    reference_vertices(kind)
    if kind == "tetrahedron":
        bary, w = tetrahedron_quadrature(order)
        return bary[:, 1:], w / 6
    bary, w = triangle_quadrature(order)
    z, wz = cube_quadrature(order, 1)
    return (
        np.column_stack((np.repeat(bary[:, 1:], len(z), axis=0), np.tile(z[:, 0], len(bary)))),
        np.repeat(w / 2, len(z)) * np.tile(wz, len(bary)),
    )


def face_quadrature(corners: int, order: int) -> tuple[FloatArray, FloatArray]:
    """Return positive unit-triangle or unit-square face quadrature."""
    if corners == 3:
        bary, w = triangle_quadrature(order)
        return bary[:, 1:], w / 2
    if corners == 4:
        return cube_quadrature(order, 2)
    raise ValueError("a face must have three or four corners")


def face_shape(uv: FloatArray, corners: int) -> FloatArray:
    """Evaluate affine triangle or bilinear tensor-square geometric shape functions."""
    if corners == 3:
        bary = np.column_stack((1 - np.asarray(uv).sum(axis=1), uv))
        return simplex_lagrange_tabulation("triangle", 1, bary, nodes=np.eye(3), nderiv=0)[0]
    if corners == 4:
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        return tensor_lagrange_tabulation("quadrilateral", 1, uv, nodes=nodes, nderiv=0)[0]
    raise ValueError("a face must have three or four corners")


def face_polynomials(uv: FloatArray, corners: int, degree: int = 1) -> FloatArray:
    """Evaluate complete Pk triangle or Qk square normal moment tests.

    Degrees zero and one preserve the monomial convention 1, u, v, uv.
    Higher orders use orthogonal polynomials normalized in the mean-square
    face norm (Dubiner on triangles, tensor Legendre on quadrilaterals).
    Their first function is one. Ordering follows total degree and descending
    first index, with no singular evaluation at triangle vertices.
    """
    degree = positive_int(degree, "face degree", 0)
    if corners not in (3, 4):
        raise ValueError("a face must have three or four corners")
    powers = sorted(
        (
            (a, b)
            for a, b in product(range(degree + 1), repeat=2)
            if corners == 4 or a + b <= degree
        ),
        key=lambda e: (sum(e), -e[0]),
    )
    if degree < 2:
        return monomial_tabulation(uv, tuple(powers), nderiv=0)[0]
    if corners == 3:
        # Native triangle Legendre functions are normalized in area measure;
        # declared Dubiner moments use the face mean-square measure instead.
        order = [total * (total + 1) // 2 + a for a, b in powers for total in (a + b,)]
        return orthogonal_polynomial_tabulation("triangle", degree, uv)[0][:, order] / np.sqrt(2)
    order = [a * (degree + 1) + b for a, b in powers]
    return orthogonal_polynomial_tabulation("quadrilateral", degree, uv)[0][:, order]


def face_size(corners: int, degree: int) -> int:
    """Return the dimension of a complete Pk or tensor Qk face space."""
    degree = positive_int(degree, "face degree", 0)
    if corners not in (3, 4):
        raise ValueError("a face must have three or four corners")
    return (degree + 1) * (degree + 2) // 2 if corners == 3 else (degree + 1) ** 2


def _powers(kind: CellKind, degree: int) -> tuple[tuple[int, int, int], ...]:
    """Enumerate complete tetrahedral or tensor-prismatic scalar monomials."""
    return tuple(
        (e[0], e[1], e[2])
        for e in product(range(degree + 1), repeat=3)
        if (sum(e) <= degree if kind == "tetrahedron" else e[0] + e[1] <= degree)
    )


def _scalar(points: FloatArray, powers: tuple) -> FloatArray:
    """Tabulate an explicitly ordered monomial basis through Basix."""
    return monomial_tabulation(points, powers, nderiv=0)[0]


def _vectors(points: FloatArray, powers: tuple) -> tuple[FloatArray, FloatArray]:
    """Tabulate component monomials and Cartesian divergence through Basix."""
    table = monomial_tabulation(points, powers, nderiv=1)
    scalar = table[0]
    size = len(powers)
    values = np.zeros((len(points), 3 * size, 3))
    div = np.zeros((len(points), 3 * size))
    for axis in range(3):
        values[:, axis * size : (axis + 1) * size, axis] = scalar
        div[:, axis * size : (axis + 1) * size] = table[axis + 1]
    return values, div


def _interior_seeds(
    kind: CellKind, pressure_degree: int
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return fixed monomial tests unisolvent on the zero-normal bubble subspace.

    A pair ``(a, e)`` denotes the reference integral of ``q[a] * x**e``.
    The list follows total degree, exponent and component order, omitting
    functionals dependent on preceding ones in the declared polynomial space.
    These lists define coordinates; they are never selected by a numerical
    pivot or singular-vector orientation at runtime.
    """
    constant = tuple((axis, (0, 0, 0)) for axis in range(3))
    if kind == "prism":
        return constant + (
            (0, (0, 0, 1)),
            (1, (0, 0, 1)),
            (0, (0, 1, 0)),
            (2, (0, 1, 0)),
            (2, (1, 0, 0)),
            (0, (0, 1, 1)),
        )
    if pressure_degree == 1:
        return constant + ((0, (0, 0, 1)), (1, (0, 0, 1)), (0, (0, 1, 0)))
    linear = tuple(
        (axis, exponent) for exponent in ((0, 0, 1), (0, 1, 0), (1, 0, 0)) for axis in range(3)
    )
    return (
        constant
        + linear
        + (
            (0, (0, 0, 2)),
            (1, (0, 0, 2)),
            (0, (0, 1, 1)),
            (1, (0, 1, 1)),
            (0, (0, 2, 0)),
            (0, (1, 0, 1)),
            (1, (1, 0, 1)),
            (0, (1, 1, 0)),
        )
    )


@cache
def _coefficients(kind: CellKind, pressure_degree: int) -> FloatArray:
    """Construct face-dual lifts and uniquely oriented orthonormal interior bubbles.

    Nullspaces identify polynomial subspaces only. Fixed monomial moments
    remove their arbitrary singular-vector rotations before positive-diagonal
    Cholesky orthonormalization. The resulting reference basis is determined
    mathematically, independently of the SVD implementation or thread count.
    Persisted fields should additionally store this matrix to preserve their
    exact floating-point basis across numerical-library versions.
    """
    vertices = reference_vertices(kind)
    powers = _powers(kind, pressure_degree + 1)
    xyz, w = cell_quadrature(kind, pressure_degree + 4)
    values, div = _vectors(xyz, powers)
    constraints, faces = [], []
    for indices in reference_faces(kind):
        nodes = vertices[list(indices)]
        uv, weights = face_quadrature(len(indices), pressure_degree + 4)
        normal = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
        if normal @ (nodes.mean(axis=0) - vertices.mean(axis=0)) < 0:
            normal *= -1
        normal_values = _vectors(face_shape(uv, len(indices)) @ nodes, powers)[0] @ normal
        tests = face_polynomials(uv, len(indices))
        constraints.append(
            normal_values - tests @ np.linalg.lstsq(tests, normal_values, rcond=None)[0]
        )
        faces.append(tests.T @ (weights[:, None] * normal_values))
    pressure = _scalar(xyz, _powers(kind, pressure_degree))
    constraints.append(div - pressure @ np.linalg.lstsq(pressure, div, rcond=None)[0])
    if kind == "prism":
        # The maximal trace/divergence-constrained W22 space has an additional
        # solenoidal bubble. The tensor family excludes this extra enrichment.
        selected = [
            axis * len(powers) + i
            for axis in range(3)
            for i, e in enumerate(powers)
            if (e[2] > 1 if axis < 2 else e[0] + e[1] > 1)
        ]
        constraints.append(np.eye(3 * len(powers))[selected])
    # Constraints are polynomial identities evaluated at an unisolvent positive rule.
    constraint = np.vstack(constraints)
    span = null_space(constraint, rcond=2e-12)
    expected = (
        18
        if kind == "tetrahedron" and pressure_degree == 1
        else 32
        if kind == "tetrahedron"
        else 27
    )
    if span.shape[1] != expected:
        raise ArithmeticError("polynomial constraints do not have the declared family dimension")
    sampled = np.einsum("qia,ij->qja", values, span)
    mass = np.einsum("q,qia,qja->ij", w, sampled, sampled)
    span = span @ np.linalg.inv(np.linalg.cholesky(mass).T)
    sampled = np.einsum("qia,ij->qja", values, span)
    moments = np.vstack(faces) @ span
    bubbles = null_space(moments)
    bubble_values = np.einsum("qia,ij->qja", sampled, bubbles)
    seeds = _interior_seeds(kind, pressure_degree)
    tests = _scalar(xyz, tuple(exponent for _, exponent in seeds))
    seed_moments = np.stack(
        [
            np.einsum("q,qi,q->i", w, bubble_values[..., axis], tests[:, i])
            for i, (axis, _) in enumerate(seeds)
        ]
    )
    ordered = bubbles @ np.linalg.inv(seed_moments)
    ordered_values = np.einsum("qia,ij->qja", sampled, ordered)
    gram = np.einsum("q,qia,qja->ij", w, ordered_values, ordered_values)
    interior = ordered @ np.linalg.inv(np.linalg.cholesky(gram).T)
    dual = np.column_stack((moments.T @ np.linalg.inv(moments @ moments.T), interior))
    coefficients = span @ dual
    coefficients.setflags(write=False)
    return coefficients


@dataclass(frozen=True)
class HDiv3DFamily:
    """Mixed polynomial family with separate normal and pressure degrees.

    Tetrahedral pressure degree p uses all zero-normal bubbles of [P_(p+1)]^3;
    normal degree k may range from zero to p+1. Prisms use the tensor family
    declared in this module, with zero<=k<=p. Default k=p=1 reproduces the
    18-mode tetrahedron and 27-mode prism; p=2,k=1 gives 32 tetrahedral modes.
    Fixed normal order and increasing pressure degree enrich only interiors.
    """

    kind: CellKind = "tetrahedron"
    pressure_degree: int = 1
    normal_degree: int = 1

    def __post_init__(self) -> None:
        """Validate the supported mathematical spaces."""
        reference_vertices(self.kind)
        positive_int(self.pressure_degree, "pressure degree", 0)
        positive_int(self.normal_degree, "normal degree", 0)
        maximum = self.pressure_degree + (self.kind == "tetrahedron")
        if self.normal_degree > maximum:
            raise ValueError("normal degree exceeds the polynomial trace of the mixed family")

    @property
    def face_sizes(self) -> tuple[int, ...]:
        """Return complete triangular Pk or rectangular Qk normal moment counts."""
        return tuple(
            face_size(len(face), self.normal_degree) for face in reference_faces(self.kind)
        )

    @property
    def local_size(self) -> int:
        """Return the number of local flux modes, including zero-normal interior bubbles."""
        return self.coefficients.shape[1]

    @property
    def pressure_size(self) -> int:
        """Return the complete divergence-space dimension."""
        return len(_powers(self.kind, self.pressure_degree))

    @property
    def interior_size(self) -> int:
        """Return the number of cell-private flux modes."""
        return self.local_size - sum(self.face_sizes)

    @property
    def coefficients(self) -> FloatArray:
        """Return the read-only component-monomial coefficient matrix of this basis.

        Rows are grouped by vector component. The original normal-P1 tetrahedral
        p=1,2 and prismatic p=1 families retain their lexicographic monomial
        candidates. Other orders use the stable Bernstein candidates declared
        in ``pymhm.fem.hdiv.moments_3d.candidates``. Columns are face moments followed
        by uniquely oriented interior bubbles. The family degrees and candidate
        convention are part of the coefficient-vector contract.
        Save this matrix alongside persisted flux DOFs and pass it to
        :meth:`tabulate` when replaying those coordinates.
        """
        if self.normal_degree == 1 and self.pressure_degree in (
            (1, 2) if self.kind == "tetrahedron" else (1,)
        ):
            return _coefficients(self.kind, self.pressure_degree)
        from pymhm.fem.hdiv.moments_3d import coefficients

        return coefficients(self.kind, self.pressure_degree, self.normal_degree)

    @property
    def interior_moment_seeds(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        """Return the ordered ``(component, monomial exponent)`` interior seed tests.

        Bubble functions are dual to these tests before positive-diagonal
        Cholesky orthonormalization. The final interior DOFs are their L2
        moments against the resulting orthonormal bubble functions; face
        lifts are L2 orthogonal to every bubble.
        """
        if self.normal_degree != 1 or self.pressure_degree not in (
            (1, 2) if self.kind == "tetrahedron" else (1,)
        ):
            raise ValueError("general families use vector Nedelec moment tests")
        return _interior_seeds(self.kind, self.pressure_degree)

    def tabulate(
        self, points: FloatArray, *, coefficients: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate reference flux, analytic divergence and pressure monomials.

        ``coefficients`` is an optional archived reference basis matrix with
        exactly the same row/column convention as :attr:`coefficients`.
        Supplying it preserves the executed basis when replaying stored DOFs.
        """
        raw = np.asarray(points)
        if np.iscomplexobj(raw) or raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise ValueError("reference points must be finite real triples")
        if self.normal_degree == 1 and self.pressure_degree in (
            (1, 2) if self.kind == "tetrahedron" else (1,)
        ):
            values, div = _vectors(raw, _powers(self.kind, self.pressure_degree + 1))
        else:
            from pymhm.fem.hdiv.moments_3d import candidates

            values, div = candidates(self.kind, self.pressure_degree, raw)
        if coefficients is None:
            coefficients = self.coefficients
        else:
            coefficients = np.asarray(coefficients)
            if (
                np.iscomplexobj(coefficients)
                or coefficients.shape != self.coefficients.shape
                or not np.isfinite(coefficients).all()
            ):
                raise ValueError(
                    "archived basis coefficients must be a finite real matrix of the declared shape"
                )
        return (
            np.einsum("qia,ij->qja", values, coefficients),
            div @ coefficients,
            _scalar(raw, _powers(self.kind, self.pressure_degree)),
        )
