"""Canonical arbitrary-order Raviart--Thomas moments on affine tetrahedra.

RT_m = [P_m]^3 + x times homogeneous P_m. Face moments use the declared
Pk normal tests; interior moments test [P_(m-1)]^3. The interior test basis is
Bernstein, and its degrees of freedom are moments, not expansion coefficients.
"""

from dataclasses import dataclass
from functools import cache
from itertools import product

import numpy as np

from pymhm.element_backends import (
    ReferenceElementSpec,
    create_reference_element,
    interpolate_reference,
    reference_interpolation_points,
)
from pymhm.hdiv3d_family import (
    HDiv3DFamily,
    _powers,
    _scalar,
    cell_quadrature,
    face_polynomials,
    face_quadrature,
    face_shape,
    reference_faces,
    reference_vertices,
)
from pymhm.hdiv3d_general import _bernstein
from pymhm.hdiv_reference import vector_tabulation
from pymhm.mesh import FloatArray, positive_int


def rt3d_interior_tests(points: FloatArray, degree: int) -> FloatArray:
    """Tabulate the componentwise P_(m-1) canonical volume moment tests."""
    degree = positive_int(degree, "RT degree", 0)
    if not degree:
        return np.empty((len(points), 0, 3))
    scalar = _bernstein(points, degree - 1)[0]
    return np.concatenate([scalar[..., None] * np.eye(3)[axis] for axis in range(3)], axis=1)


@cache
def _candidate_map(degree: int) -> FloatArray:
    """Express the archived RT candidate rows in the native Basix basis."""
    element = create_reference_element(
        ReferenceElementSpec("RT", "tetrahedron", degree + 1, lagrange_variant="legendre")
    )
    points = reference_interpolation_points(element)
    scalar = _bernstein(points, degree)[0]
    homogeneous = tuple(e for e in product(range(degree + 1), repeat=3) if sum(e) == degree)
    radial = _scalar(points, homogeneous)
    fields = [scalar[..., None] * np.eye(3)[axis] for axis in range(3)]
    fields.append(radial[..., None] * points[:, None])
    result = interpolate_reference(element, np.concatenate(fields, axis=1))
    result.setflags(write=False)
    return result


def _candidates(points: FloatArray, degree: int) -> tuple[FloatArray, FloatArray]:
    """Tabulate native RT in the unchanged archived Bernstein/radial coordinates."""
    values, divergence = vector_tabulation("RT", "tetrahedron", degree + 1, points)
    transform = _candidate_map(degree)
    return np.einsum("qia,ij->qja", values, transform), divergence @ transform


@cache
def _coefficients(degree: int) -> FloatArray:
    """Invert the unisolvent normal/volume moment matrix of RT_m."""
    vertices = reference_vertices("tetrahedron")
    rows = []
    for face in reference_faces("tetrahedron"):
        nodes = vertices[list(face)]
        uv, weights = face_quadrature(3, degree + 3)
        normal = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
        normal *= np.sign(normal @ (nodes.mean(axis=0) - vertices.mean(axis=0)))
        values = _candidates(face_shape(uv, 3) @ nodes, degree)[0] @ normal
        rows.append(face_polynomials(uv, 3, degree).T @ (weights[:, None] * values))
    points, weights = cell_quadrature("tetrahedron", degree + 3)
    interior = rt3d_interior_tests(points, degree)
    rows.append(np.einsum("q,qia,qja->ij", weights, interior, _candidates(points, degree)[0]))
    moments = np.vstack(rows)
    result = np.linalg.solve(moments, np.eye(len(moments)))
    result.setflags(write=False)
    return result


@dataclass(frozen=True, init=False)
class RTTetraFamily(HDiv3DFamily):
    """RT_m tetrahedral flux with complete P_m divergence and normal trace.

    This family shares canonical physical face orientation and affine Piola
    mapping with :class:`HDiv3DFamily`. Its interior coordinates are the
    canonical volume moments, without bubble orthonormalization.
    """

    def __init__(self, degree: int = 0) -> None:
        """Select a nonnegative RT polynomial order."""
        super().__init__("tetrahedron", degree, degree)

    @property
    def coefficients(self) -> FloatArray:
        """Return the executed RT candidate-to-moment-dual basis for archival replay."""
        return _coefficients(self.pressure_degree)

    def tabulate(
        self, points: FloatArray, *, coefficients: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate flux, divergence and complete pressure monomials."""
        raw = np.asarray(points)
        if np.iscomplexobj(raw) or raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise ValueError("reference points must be finite real triples")
        values, div = _candidates(raw, self.pressure_degree)
        basis = self.coefficients if coefficients is None else np.asarray(coefficients)
        if (
            np.iscomplexobj(basis)
            or basis.shape != self.coefficients.shape
            or not np.isfinite(basis).all()
        ):
            raise ValueError("archived RT basis must be a finite real matrix of the declared shape")
        return (
            np.einsum("qia,ij->qja", values, basis),
            div @ basis,
            _scalar(raw, _powers("tetrahedron", self.pressure_degree)),
        )
