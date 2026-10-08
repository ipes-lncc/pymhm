"""Scalar polynomial spaces and oriented skeleton coordinates on planar macroedges."""

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class FaceSpace:
    """Discontinuous Legendre or continuous nodal polynomials on one macroface.

    Breaks parameterize the oriented face on [0, 1]. Each segment has its own
    degree. The default is one constant multiplier on the entire face.
    ``continuous=True`` shares endpoints between segments on this macroface,
    with no continuity imposed across different macrofaces. Degrees must then
    be positive; interior interpolation nodes are equally spaced.
    """

    breaks: tuple[float, ...] = (0.0, 1.0)
    degrees: tuple[int, ...] = (0,)
    continuous: bool = False

    def __post_init__(self) -> None:
        """Reject gaps, reversed intervals and inconsistent polynomial degrees."""
        breaks = tuple(float(x) for x in self.breaks)
        degrees = tuple(positive_int(p, "degree", 0) for p in self.degrees)
        if len(breaks) != len(degrees) + 1 or not degrees:
            raise ValueError("one degree is required per face segment")
        if breaks[0] != 0 or breaks[-1] != 1 or not np.all(np.diff(breaks) > 0):
            raise ValueError("face breaks must increase strictly from 0 to 1")
        if not isinstance(self.continuous, bool) or (self.continuous and min(degrees) < 1):
            raise ValueError("continuous faces require positive degrees and a boolean flag")
        object.__setattr__(self, "breaks", breaks)
        object.__setattr__(self, "degrees", degrees)

    @classmethod
    def uniform(
        cls, degree: int = 0, subdivisions: int = 1, *, continuous: bool = False
    ) -> "FaceSpace":
        """Construct an equally partitioned face with the same degree per segment."""
        n = positive_int(subdivisions, "subdivisions")
        return cls(tuple(np.linspace(0, 1, n + 1)), (degree,) * n, continuous)

    @property
    def size(self) -> int:
        """Return the number of scalar face unknowns."""
        return 1 + sum(self.degrees) if self.continuous else sum(p + 1 for p in self.degrees)

    def evaluate(self, parameter: Any) -> FloatArray:
        """Evaluate Basix scalar bases at oriented coordinates in [0, 1].

        Continuous segment nodes are ordered left endpoint, right endpoint,
        then increasing interior nodes. Discontinuous modes use conventional
        unnormalized Legendre polynomials, whose degree-j squared mass on a
        segment of length L is L/(2j+1). Internal breaks belong to the segment
        on their right; a face endpoint belongs to its adjacent segment.
        """
        from pymhm.fem.reference import legendre_values, simplex_lagrange_tabulation

        t = np.atleast_1d(np.asarray(parameter, dtype=float))
        if t.ndim != 1 or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
            raise ValueError("face coordinates must be a finite vector in [0, 1]")
        values = np.zeros((len(t), self.size))
        offset = 0
        for i, degree in enumerate(self.degrees):
            lo, hi = self.breaks[i : i + 2]
            mask = (t >= lo) & ((t < hi) | ((i == len(self.degrees) - 1) & (t <= hi)))
            local = (t[mask] - lo) / (hi - lo)
            if self.continuous:
                nodes = np.r_[0.0, 1.0, np.arange(1, degree) / degree]
                basis, _, _ = simplex_lagrange_tabulation(
                    "interval",
                    degree,
                    np.column_stack((1 - local, local)),
                    nodes=np.column_stack((1 - nodes, nodes)),
                    nderiv=0,
                )
                ids = np.r_[i, i + 1, len(self.breaks) + offset + np.arange(degree - 1)]
                values[np.ix_(mask, ids)] = basis
                offset += degree - 1
            else:
                values[mask, offset : offset + degree + 1] = legendre_values(2 * local - 1, degree)
                offset += degree + 1
        return values

    def constant_coefficients(self) -> FloatArray:
        """Represent the constant one exactly in this face's declared basis."""
        if self.continuous:
            return np.ones(self.size)
        coefficients = np.zeros(self.size)
        coefficients[np.r_[0, np.cumsum(np.asarray(self.degrees[:-1]) + 1)].astype(int)] = 1
        return coefficients

    def quadrature(self, order: int = 4) -> tuple[FloatArray, FloatArray]:
        """Return Gauss points and weights, separately on every segment.

        Normalize the reference rule to its exact mass two before mapping
        each subinterval. Extended accumulation prevents a rounded weight-sum
        defect from becoming an artificial volume change in nearly
        incompressible displacement boundary data. No field value is clipped.
        Returned coordinates and weights retain binary64 storage.
        """
        x, w = leggauss(positive_int(order, "quadrature order"))
        w = np.asarray(w, dtype=np.longdouble)
        w *= np.longdouble(2) / np.sum(w)
        points: list[float] = []
        weights: list[float] = []
        for lo, hi in zip(self.breaks[:-1], self.breaks[1:], strict=True):
            points.extend(lo + (x + 1) * (hi - lo) / 2)
            weights.extend(w * (hi - lo) / 2)
        return np.asarray(points), np.asarray(weights, dtype=float)


@dataclass(frozen=True, init=False)
class SkeletonSpace:
    """Independent face partitions and degrees with component-interleaved DOFs."""

    mesh: TriangleMesh
    faces: tuple[FaceSpace, ...]
    components: int = 1
    offsets: IntArray = field(init=False, repr=False)

    def __init__(
        self, mesh: TriangleMesh, faces: tuple[FaceSpace, ...] | None = None, components: int = 1
    ) -> None:
        """Assign unique face numbers and validate the geometry-to-space association."""
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "components", positive_int(components, "components"))
        faces = tuple(FaceSpace() for _ in mesh.faces) if faces is None else tuple(faces)
        if len(faces) != len(self.mesh.faces) or not all(
            isinstance(face, FaceSpace) for face in faces
        ):
            raise ValueError("provide exactly one FaceSpace per mesh face")
        offsets = np.r_[0, np.cumsum([face.size * components for face in faces])].astype(np.int64)
        offsets.setflags(write=False)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "offsets", offsets)

    @property
    def size(self) -> int:
        """Return the total number of trace coefficients."""
        return int(self.offsets[-1])

    def dofs(self, face: int) -> IntArray:
        """Return the global DOFs belonging to a face."""
        positive_int(face, "face", 0)
        if face >= len(self.mesh.faces):
            raise ValueError("face index outside mesh")
        return np.arange(self.offsets[face], self.offsets[face + 1])

    def cell_dofs(self, cell: int) -> IntArray:
        """Concatenate trace DOFs in the cell's counterclockwise side order."""
        positive_int(cell, "cell", 0)
        if cell >= len(self.mesh.cells):
            raise ValueError("cell index outside mesh")
        return np.concatenate([self.dofs(int(face)) for face in self.mesh.cell_faces[cell]])


def _pairing_indices(value: Any, size: int) -> IntArray:
    """Check distinct declared coordinates without changing their local ordering."""
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or not np.issubdtype(raw.dtype, np.integer)
        or np.any(raw < 0)
        or np.any(raw >= size)
        or len(np.unique(raw)) != len(raw)
    ):
        raise ValueError("interface coordinates must be distinct valid integer indices")
    return np.asarray(raw, dtype=np.int64)


def _pairing_layout(space: Any, mesh: Any, cell: int) -> tuple[IntArray, list[IntArray], int]:
    """Associate each face basis with its declared, possibly shared local coordinates."""
    faces = getattr(space, "faces", ())
    cell_dofs = getattr(space, "cell_dofs", None)
    face_dofs = getattr(space, "face_dofs", ())
    dofs = getattr(space, "dofs", None)
    if (
        len(faces) != len(mesh.faces)
        or not all(isinstance(face, FaceSpace) for face in faces)
        or not callable(cell_dofs)
        or (not callable(dofs) and len(face_dofs) != len(mesh.faces))
    ):
        raise TypeError("interface pairing requires declared FaceSpace bases and coordinate maps")
    size = positive_int(space.size, "interface size", 0)
    components = positive_int(getattr(space, "components", 1), "interface components")
    local = _pairing_indices(cell_dofs(cell), size)
    lookup = {int(coordinate): position for position, coordinate in enumerate(local)}
    positions: list[IntArray] = []
    support: set[int] = set()
    for face in mesh.cell_faces[cell]:
        indices = _pairing_indices(dofs(int(face)) if callable(dofs) else face_dofs[face], size)
        if len(indices) != faces[face].size * components or not set(indices).issubset(lookup):
            raise ValueError(
                "face basis width and coordinates must match the declared cell support"
            )
        positions.append(np.asarray([lookup[int(index)] for index in indices], dtype=np.int64))
        support.update(indices)
    if support != set(local):
        raise ValueError("face coordinates must cover the declared local interface basis")
    return local, positions, components


def interface_pairing(
    test_space: Any, trial_space: Any, cell: int, *, order: int = 6
) -> FloatArray:
    """Compatibility entry point for the dimension-independent physical trace pairing."""
    from pymhm.fem.traces.pairing import interface_pairing as physical_pairing

    mesh: Any = getattr(test_space, "mesh", None)
    if mesh is None:
        raise TypeError("interface pairing requires mesh-associated polynomial spaces")
    if not all(hasattr(mesh, name) for name in ("points", "faces", "cell_faces", "cells")):
        raise TypeError("interface pairing requires mesh-associated polynomial spaces")
    if getattr(trial_space, "mesh", None) is not mesh:
        raise ValueError("paired interfaces must share the same macro mesh")
    mesh = cast(Any, mesh)
    points, faces = np.asarray(mesh.points), np.asarray(mesh.faces)
    if points.ndim != 2 or points.shape[1] != 2 or faces.ndim != 2 or faces.shape[1] != 2:
        raise TypeError("automatic interface pairing requires planar edge geometry")
    if not hasattr(mesh, "lengths"):
        raise TypeError("interface pairing requires declared physical face lengths")
    cell = positive_int(cell, "cell", 0)
    order = positive_int(order, "quadrature order")
    if cell >= len(mesh.cells):
        raise ValueError("cell index outside mesh")
    test, test_positions, components = _pairing_layout(test_space, mesh, cell)
    trial, trial_positions, trial_components = _pairing_layout(trial_space, mesh, cell)
    if components != trial_components:
        raise ValueError("paired interface spaces must have the same component count")
    for face in mesh.cell_faces[cell]:
        length = float(mesh.lengths[face])
        if not np.isfinite(length) or length <= 0:
            raise ValueError("paired interfaces require finite positive face lengths")

    return physical_pairing(test_space, trial_space, cell, order=order)
