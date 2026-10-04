"""Geometry shared by the published two-dimensional unfitted scalar cases."""

from pymhm.meshes.triangle import TriangleMesh


def macro_mesh(delta: float) -> TriangleMesh:
    """Construct the 16 crisscross triangles with the middle row shifted upward.

    The cell order and arithmetic coincide with the original campaign geometry.
    This module has no plotting dependencies and is safe to import in workers.
    """
    xs, ys = [0, 0.5, 1], [0, 0.5 + delta, 1]
    points = [[x, y] for y in ys for x in xs]
    cells = []
    for j in range(2):
        for i in range(2):
            corners = [3 * j + i, 3 * j + i + 1, 3 * (j + 1) + i + 1, 3 * (j + 1) + i]
            center = len(points)
            points.append([(xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2])
            for k in range(4):
                cells.append([corners[k], corners[(k + 1) % 4], center])
    return TriangleMesh(points, cells)
