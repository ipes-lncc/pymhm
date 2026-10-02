"""Positive piecewise-constant tensors on explicitly described planar material regions."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.mesh import FloatArray, IntArray


def _tensor(value: Any, dimension: int) -> FloatArray:
    """Validate one real symmetric positive-definite physical material tensor."""
    if np.iscomplexobj(value):
        raise ValueError("material tensors must be real")
    raw = np.asarray(value, dtype=float)
    tensor = np.eye(dimension) * raw if raw.ndim == 0 else raw.copy()
    if tensor.shape != (dimension, dimension) or not np.isfinite(tensor).all():
        raise ValueError("material tensor has an invalid shape or nonfinite entries")
    if not np.allclose(
        tensor, tensor.T, rtol=0, atol=32 * np.finfo(float).eps * np.max(abs(tensor))
    ):
        raise ValueError("material tensor must be symmetric")
    tensor = (tensor + tensor.T) / 2
    if np.linalg.eigvalsh(tensor).min() <= 0:
        raise ValueError("material tensor must be positive definite")
    tensor.setflags(write=False)
    return tensor


@dataclass(frozen=True)
class PlanarRegion:
    """Intersection of halfspaces n_i dot x <= b_i carrying a constant SPD tensor.

    Normals may be unnormalized. Halfspaces can describe bounded inclusions or
    unbounded layers. The region dimension must be two or three; geometry is
    expressed in the same physical coordinates as the mesh.
    """

    normals: Any
    offsets: Any
    permeability: Any

    def __post_init__(self) -> None:
        """Normalize the physical plane equations without changing their halfspaces."""
        if np.iscomplexobj(self.normals) or np.iscomplexobj(self.offsets):
            raise ValueError("plane equations must be real")
        normals, offsets = (
            np.asarray(self.normals, dtype=float),
            np.asarray(self.offsets, dtype=float),
        )
        if (
            normals.ndim != 2
            or normals.shape[1] not in (2, 3)
            or len(normals) == 0
            or offsets.shape != (len(normals),)
            or not np.isfinite(normals).all()
            or not np.isfinite(offsets).all()
        ):
            raise ValueError("plane equations need finite normals and one offset per plane")
        norm = np.linalg.norm(normals, axis=1)
        if np.any(norm == 0):
            raise ValueError("plane normals must be nonzero")
        normals, offsets = normals / norm[:, None], offsets / norm
        normals.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "permeability", _tensor(self.permeability, normals.shape[1]))

    @classmethod
    def box(cls, lower: Any, upper: Any, permeability: Any) -> "PlanarRegion":
        """Describe an axis-aligned inclusion using physical lower and upper coordinates."""
        if np.iscomplexobj(lower) or np.iscomplexobj(upper):
            raise ValueError("box bounds must be real")
        a, b = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if a.shape not in ((2,), (3,)) or b.shape != a.shape or np.any(b <= a):
            raise ValueError("box bounds must have matching dimension and positive extent")
        return cls(np.vstack((np.eye(len(a)), -np.eye(len(a)))), np.r_[b, -a], permeability)


@dataclass(frozen=True)
class PlanarMaterial:
    """Background tensor with ordered planar inclusions; the last containing region wins.

    Region zero denotes the background. Positive region IDs are one-based in
    ``regions``. Values exactly on a plane belong to its closed halfspace;
    ``trace_values`` instead selects a physical one-sided value using the
    incident-cell interior, without displacing evaluation coordinates.
    """

    background: Any
    regions: tuple[PlanarRegion, ...]

    def __post_init__(self) -> None:
        """Validate common dimension and retain immutable physical material data."""
        regions = tuple(self.regions)
        if not regions or not all(isinstance(r, PlanarRegion) for r in regions):
            raise ValueError("material requires at least one PlanarRegion")
        dimension = regions[0].normals.shape[1]
        if any(r.normals.shape[1] != dimension for r in regions):
            raise ValueError("all material regions must have the same dimension")
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "background", _tensor(self.background, dimension))

    @property
    def dimension(self) -> int:
        """Physical dimension of every tensor and plane."""
        return int(self.background.shape[0])

    @property
    def planes(self) -> tuple[FloatArray, FloatArray]:
        """All interface planes, including repeated planes with identical geometry."""
        return np.vstack([r.normals for r in self.regions]), np.concatenate(
            [r.offsets for r in self.regions]
        )

    @property
    def tensors(self) -> FloatArray:
        """Tensor table indexed by background/region identifiers."""
        return np.array([self.background, *(r.permeability for r in self.regions)])

    def _points(self, points: Any) -> FloatArray:
        """Validate arrays of finite physical coordinates."""
        raw = np.asarray(points)
        if (
            np.iscomplexobj(raw)
            or raw.ndim != 2
            or raw.shape[1] != self.dimension
            or not np.isfinite(raw).all()
        ):
            raise ValueError("material points must be finite real coordinate arrays")
        return np.asarray(raw, dtype=float)

    def region_ids(self, points: Any) -> IntArray:
        """Identify the last closed material region containing each point."""
        locations = self._points(points)
        result = np.zeros(len(locations), dtype=np.int64)
        for index, region in enumerate(self.regions, 1):
            result[np.all(locations @ region.normals.T <= region.offsets, axis=1)] = index
        return result

    def __call__(self, points: Any) -> FloatArray:
        """Evaluate the physical permeability tensor at each point."""
        return self.tensors[self.region_ids(points)]

    def trace_values(self, points: Any, interior_points: Any) -> FloatArray:
        """Select exact incident-side material traces on planar interfaces.

        The zero-distance envelope bounds floating-point plane evaluation.
        An interior point tangent to a relevant plane cannot define a side
        and is rejected. No shifted material sampling or averaging is used.
        """
        locations = self._points(points)
        raw = np.asarray(interior_points)
        if raw.shape == (self.dimension,):
            raw = np.broadcast_to(raw, locations.shape)
        interior = self._points(raw)
        if interior.shape != locations.shape:
            raise ValueError("interior points must match evaluation points")
        result = np.zeros(len(locations), dtype=np.int64)
        for index, region in enumerate(self.regions, 1):
            distances = locations @ region.normals.T - region.offsets
            tolerance = (
                16
                * np.finfo(float).eps
                * (abs(locations) @ abs(region.normals).T + abs(region.offsets))
            )
            on_plane = abs(distances) <= tolerance
            side = (interior - locations) @ region.normals.T
            relevant = np.all((distances <= 0) | on_plane, axis=1)
            if np.any(on_plane & (side == 0) & relevant[:, None]):
                raise ValueError("interior point must select a strict side of the material plane")
            inside = np.where(on_plane, side < 0, distances < 0)
            result[np.all(inside, axis=1)] = index
        return self.tensors[result]
