"""Coordinate perturbation bounds shared by physical mesh validators."""

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.meshes.triangle import TriangleMesh


def area_coordinate_uncertainty(mesh: TriangleMesh, displacement: FloatArray) -> float:
    """Bound area changes under a componentwise coordinate perturbation envelope.

    For edges ``a,b`` and coordinate perturbations bounded by ``d``, each edge
    perturbation is bounded by ``2d``. Expanding their determinant supplies the
    linear and quadratic terms. Eight roundoff units account for represented
    physical coordinates; the absolute-product term bounds arithmetic evaluation
    and summation. This area bound is distinct from a point-membership test.
    """
    vertices = mesh.points[mesh.cells]
    epsilon = np.finfo(float).eps
    uncertainty = displacement + 8 * epsilon * np.max(abs(vertices), axis=1)
    first, second = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    edge_error = 2 * uncertainty
    linear = np.sum(edge_error * (abs(first[:, ::-1]) + abs(second[:, ::-1])), axis=1)
    quadratic = 2 * edge_error[:, 0] * edge_error[:, 1]
    arithmetic = 32 * epsilon * np.sum(abs(first * second[:, ::-1]), axis=1)
    return float(np.sum((linear + quadratic + arithmetic) / 2))


def coordinate_difference(points: FloatArray, origin: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Subtract represented coordinates and bound input quantization and arithmetic.

    This bound is expressed in physical units, independently of whether the
    difference represents a coordinate, a cell width or a material spacing.
    Inputs are finite float64 arrays with broadcastable shapes.
    """
    unit = np.finfo(float).eps / 2
    difference = points - origin
    represented = (abs(np.spacing(points)) + abs(np.spacing(origin))) / 2
    return difference, (represented + unit * abs(difference)) / (1 - unit)


def cartesian_coordinates(
    points: FloatArray, origin: FloatArray, spacing: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Transform represented physical points and bound their grid-coordinate error.

    The envelope accounts for half-ulp input quantization, the subtraction and
    division, and half-ulp uncertainty in each grid spacing. For ``z=(x-o)/h``
    the coordinate and scale contributions are propagated separately; no
    distance-to-origin relative tolerance is applied to physical mesh areas.
    Inputs are finite float64 arrays with positive, broadcastable spacing.
    Callers retain their arithmetic-operation envelope in addition to this
    bound. Two representations may differ by the sum of their envelopes.
    Their comparison neighborhoods must not overlap adjacent integer grid
    lines; unresolved coordinate data are rejected rather than merged.
    """
    unit = np.finfo(float).eps / 2
    difference, difference_error = coordinate_difference(points, origin)
    coordinates = difference / spacing
    error = difference_error / spacing
    error += unit * abs(coordinates) / (1 - unit)
    error += abs(coordinates) * abs(np.spacing(spacing)) / (2 * spacing * (1 - unit))
    if not np.isfinite(coordinates).all() or np.any(2 * error >= 0.5):
        raise ValueError("Cartesian grid lines are not distinguishable in physical coordinates")
    return coordinates, error
