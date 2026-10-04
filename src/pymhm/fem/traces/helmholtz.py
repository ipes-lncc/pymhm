"""Polynomial and oscillatory skeleton spaces for frequency-domain Helmholtz."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.reference import legendre_values
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace


@dataclass(frozen=True)
class PolynomialNeumannTrace:
    """Prescribe physical outward flux q.n by declared Legendre coefficients.

    Coefficients multiply L_j(2*t-1), with t increasing from the first to the
    second vertex in ``mesh.faces[face]``. They are complex scalar data on one
    complete straight macroface, independent of its subdivisions. No normal
    sign is inferred from the coefficient vector: the values already denote
    physical outward q.n on that exterior face.

    This declaration preserves its polynomial degree under trace refinement
    and exact nested restriction. A generic callback has no declared degree
    and is instead projected using the executed boundary quadrature. Nonzero
    coefficients, however small, are never removed when determining degree.
    """

    coefficients: tuple[complex, ...]

    def __post_init__(self) -> None:
        """Require finite one-dimensional data and trim only exact trailing zeros."""
        values = np.asarray(self.coefficients, dtype=complex)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Neumann polynomial coefficients must be a finite nonempty vector")
        stop = len(values)
        while stop > 1 and values[stop - 1] == 0:
            stop -= 1
        object.__setattr__(self, "coefficients", tuple(complex(v) for v in values[:stop]))

    def evaluate(self, parameter: Any) -> np.ndarray:
        """Evaluate the declared polynomial at oriented macroface coordinates."""
        if np.iscomplexobj(parameter):
            raise ValueError("face coordinates must be a finite vector in [0, 1]")
        points = np.atleast_1d(np.asarray(parameter, dtype=float))
        if points.ndim != 1 or not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
            raise ValueError("face coordinates must be a finite vector in [0, 1]")
        return legendre_values(2 * points - 1, len(self.coefficients) - 1) @ self.coefficients

    def coefficients_on(self, space: FaceSpace) -> np.ndarray:
        """Transfer declared data exactly in degree to every target face segment.

        Affine pullbacks are computed as polynomial compositions. Coordinates
        above the declared degree are structurally zero, including when lower
        coefficients have rounding error. No tolerance discards a mode.
        Nonconstant data require ordinary polynomial trace spaces of adequate
        degree; constants use the actual basis's constant representation.
        """
        degree = len(self.coefficients) - 1
        if degree == 0:
            return self.coefficients[0] * space.constant_coefficients()
        if type(space) is not FaceSpace or min(space.degrees) < degree:
            raise ValueError("declared Neumann polynomial is not represented by the trace space")
        if space.continuous:
            nodes = [*space.breaks]
            for left, right, local_degree in zip(
                space.breaks[:-1], space.breaks[1:], space.degrees, strict=True
            ):
                nodes.extend(left + (right - left) * np.arange(1, local_degree) / local_degree)
            return self.evaluate(nodes)
        result = np.zeros(space.size, dtype=complex)
        polynomial = np.polynomial.Legendre(self.coefficients)
        offset = 0
        for left, right, local_degree in zip(
            space.breaks[:-1], space.breaks[1:], space.degrees, strict=True
        ):
            affine = np.polynomial.Legendre([left + right - 1, right - left])
            local = polynomial(affine).coef
            result[offset : offset + len(local)] = local
            offset += local_degree + 1
        return result


@dataclass(frozen=True)
class OscillatoryFaceSpace(FaceSpace):
    """Real coordinates for the complex exponential space of Section 6 (2020).

    On each segment, an even degree 2k uses a constant and k cosine/sine
    pairs; an odd degree 2k+1 additionally contains the linear polynomial.
    Frequencies are omega*length*cos(n*pi/(2k+2)). A deterministic weighted
    QR changes coordinates within this exact span. Its executed matrix is
    available as ``transforms`` for persisted coefficient-vector contracts.
    ``continuous`` is unavailable because the published space is discontinuous.
    """

    wave_number_length: float = 1.0
    transforms: tuple[FloatArray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the frequency and compute independent segment basis transforms."""
        self._validate_definition()
        transforms = []
        for segment, degree in enumerate(self.degrees):
            frequency = self.wave_number_length * np.diff(self.breaks)[segment]
            x, weights = leggauss(max(24, degree + 2, int(np.ceil(frequency)) + 4))
            raw = self._raw(degree, frequency, (x + 1) / 2)
            _, factor = np.linalg.qr(np.sqrt(weights[:, None] / 2) * raw)
            singular = np.linalg.svd(factor, compute_uv=False)
            if singular[-1] <= 128 * np.finfo(float).eps * singular[0]:
                raise ValueError(
                    "oscillatory basis is numerically unresolved at this frequency/degree"
                )
            signs = np.where(np.diag(factor) < 0, -1.0, 1.0)
            transform = np.linalg.solve(factor, np.diag(signs))
            transform.setflags(write=False)
            transforms.append(transform)
        object.__setattr__(self, "transforms", tuple(transforms))

    def _validate_definition(self) -> None:
        """Share the declared partition/frequency conditions between acquisition and replay."""
        super().__post_init__()
        if (
            self.continuous
            or np.iscomplexobj(self.wave_number_length)
            or not np.isfinite(self.wave_number_length)
            or self.wave_number_length <= 0
        ):
            raise ValueError(
                "oscillatory faces require a positive frequency and discontinuous segments"
            )

    @classmethod
    def from_executed_transforms(
        cls,
        breaks: tuple[float, ...],
        degrees: tuple[int, ...],
        transforms: Sequence[Any],
        *,
        wave_number_length: float,
        continuous: bool = False,
    ) -> "OscillatoryFaceSpace":
        """Restore owned binary64 segment coordinate matrices without recomputing QR.

        Matrices multiply the declared raw columns on the right. They must
        be finite, square, nonsingular real binary64 arrays of size degree+1;
        their input bytes are copied and protected from later mutation. Face
        orientation is the direction of increasing parameter, including
        reversed evaluation at 1-t. Evaluation and the representation of one
        use these exact executed matrices. Callers verify their persisted
        digest and associate the same partition, degree, frequency and mesh
        orientation with the archived coefficient vector before restoration.
        This reconstruction does not reassert the current QR's conditioning
        test or the approximation hypotheses for an arbitrary supplied basis.
        """
        space = cls.__new__(cls)
        object.__setattr__(space, "breaks", breaks)
        object.__setattr__(space, "degrees", degrees)
        object.__setattr__(space, "continuous", continuous)
        object.__setattr__(space, "wave_number_length", wave_number_length)
        space._validate_definition()
        if len(transforms) != len(space.degrees):
            raise ValueError("provide exactly one executed transform per face segment")
        restored = []
        for degree, values in zip(space.degrees, transforms, strict=True):
            matrix = np.array(values, copy=True)
            if (
                matrix.shape != (degree + 1, degree + 1)
                or matrix.dtype.kind != "f"
                or matrix.dtype.itemsize != 8
                or not np.isfinite(matrix).all()
            ):
                raise ValueError("executed transforms require finite square real binary64 matrices")
            sign, _ = np.linalg.slogdet(matrix)
            if sign == 0:
                raise ValueError("executed transforms must be nonsingular")
            matrix.setflags(write=False)
            restored.append(matrix)
        object.__setattr__(space, "transforms", tuple(restored))
        return space

    @staticmethod
    def _raw(degree: int, frequency: float, parameter: FloatArray) -> FloatArray:
        """Evaluate the declared span, subtracting low-order polynomials stably."""
        t = parameter - 0.5
        columns = [np.ones(len(t))]
        if degree % 2:
            columns.append(t)
        count = degree // 2
        for index in range(1, count + 1):
            delta = frequency * np.cos(index * np.pi / (2 * count + 2))
            z = delta * t
            columns.append(-2 * np.sin(z / 2) ** 2 / delta**2)
            if degree % 2:
                # sin(z)-z has avoidable cancellation near the polynomial limit.
                remainder = np.where(
                    np.abs(z) < 0.1,
                    z**3 * (-1 / 6 + z**2 * (1 / 120 + z**2 * (-1 / 5040 + z**2 / 362880))),
                    np.sin(z) - z,
                )
                columns.append(remainder / delta**3)
            else:
                columns.append(np.sin(z) / delta)
        return np.column_stack(columns)

    def evaluate(self, parameter: Any) -> FloatArray:
        """Evaluate the executed real basis at oriented face parameters."""
        t = np.atleast_1d(np.asarray(parameter, dtype=float))
        if t.ndim != 1 or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
            raise ValueError("face coordinates must be a finite vector in [0, 1]")
        values = np.zeros((len(t), self.size))
        offset = 0
        for i, degree in enumerate(self.degrees):
            low, high = self.breaks[i : i + 2]
            selected = (t >= low) & ((t < high) | ((i == len(self.degrees) - 1) & (t <= high)))
            local = (t[selected] - low) / (high - low)
            raw = self._raw(degree, self.wave_number_length * (high - low), local)
            values[selected, offset : offset + degree + 1] = raw @ self.transforms[i]
            offset += degree + 1
        return values

    def constant_coefficients(self) -> FloatArray:
        """Represent one in the executed segment bases without a fit."""
        return np.concatenate(
            [np.linalg.solve(transform, np.eye(len(transform))[0]) for transform in self.transforms]
        )


def helmholtz_skeleton(
    mesh: Any, omega: float, *, degree: int = 1, subdivisions: int = 1, oscillatory: bool = False
) -> SkeletonSpace:
    """Build two-component real/imaginary coordinates for a complex scalar trace.

    The oscillatory frequencies use the physical edge length and the supplied
    angular frequency. They correspond to the homogeneous unit-wave-speed
    experiments in the article; no effective heterogeneous speed is inferred.
    """
    degree = positive_int(degree, "degree", 0)
    subdivisions = positive_int(subdivisions, "subdivisions")
    if np.iscomplexobj(omega) or not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be positive and finite")
    breaks = tuple(np.linspace(0, 1, subdivisions + 1))
    faces = tuple(
        OscillatoryFaceSpace(breaks, (degree,) * subdivisions, False, omega * length)
        if oscillatory
        else FaceSpace(breaks, (degree,) * subdivisions)
        for length in mesh.lengths
    )
    return SkeletonSpace(mesh, faces, components=2)
