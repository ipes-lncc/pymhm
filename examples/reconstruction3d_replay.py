"""One-sided physical evaluation of archived tetrahedral MHM and RT coefficients."""

from typing import Any

import numpy as np

from pymhm import AffineMixedMesh, RTTetraFamily, TetraMesh
from pymhm.hdiv3d_mesh import hdiv3d_basis, hdiv3d_dofs
from pymhm.tetrahedral import tetra_basis, tetra_nodal_space


def evaluate(archive: Any, macro: int, owners: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """Return pressure and three flux components using each sample's explicit fine-cell owner."""
    fine = TetraMesh(archive[f"points_{macro}"], archive[f"cells_{macro}"])
    degree = int(archive["local_degree"]) if "local_degree" in archive else 2
    rt_degree = int(archive["reconstruction_degree"]) if "reconstruction_degree" in archive else 1
    dofs, _ = tetra_nodal_space(fine, degree)
    pressure = np.einsum(
        "qi,qi->q", tetra_basis(degree, bary)[0], archive[f"pressure_{macro}"][dofs[owners]]
    )
    mixed = AffineMixedMesh(fine.points, fine.cells)
    if not np.array_equal(mixed.cells, fine.cells):
        raise ValueError("replay requires the archived oriented tetrahedral vertex convention")
    family = RTTetraFamily(rt_degree)
    basis = hdiv3d_basis(mixed, family, bary[:, 1:], coefficients=archive["rt_basis"])[0]
    basis = basis[owners, np.arange(len(owners))]
    coefficients = archive[f"rt_flux_{macro}"][hdiv3d_dofs(mixed, family)[owners]]
    return np.column_stack((pressure, np.einsum("qia,qi->qa", basis, coefficients)))


def line_intervals(mesh: TetraMesh, first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Clip a physical segment against each closed tetrahedron using affine barycentric bounds.

    Rows contain the owning cell and entry/exit parameters in [0,1]. Point
    intersections are omitted. Overlapping intervals indicate a line lying
    along a mesh interface and require an explicitly one-sided profile.
    """
    result = []
    for owner, vertices in enumerate(mesh.points[mesh.cells]):
        inverse = np.linalg.inv((vertices[1:] - vertices[0]).T)
        xi = inverse @ (first - vertices[0])
        delta = inverse @ (last - first)
        start, slope = np.r_[1 - xi.sum(), xi], np.r_[-delta.sum(), delta]
        lower, upper = 0.0, 1.0
        for value, derivative in zip(start, slope, strict=True):
            if abs(derivative) < 1e-14:
                if value < -1e-13:
                    upper = -1.0
                    break
            elif derivative > 0:
                lower = max(lower, -value / derivative)
            else:
                upper = min(upper, -value / derivative)
        if upper - lower > 1e-13:
            result.append((owner, lower, upper))
    return np.asarray(result).reshape(-1, 3)


def profile(archive: Any, first: np.ndarray, last: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate each intersected fine-cell polynomial on an independent closed segment."""
    macro_mesh = TetraMesh(archive["macro_points"], archive["macro_cells"])
    macro_intervals = line_intervals(macro_mesh, first, last)
    parameters, points, values = [], [], []
    for macro in macro_intervals[:, 0].astype(int):
        fine = TetraMesh(archive[f"points_{macro}"], archive[f"cells_{macro}"])
        intervals = line_intervals(fine, first, last)
        for owner, lower, upper in intervals:
            parameter = np.linspace(lower, upper, 25)
            locations = first + parameter[:, None] * (last - first)
            vertices = fine.points[fine.cells[int(owner)]]
            xi = (locations - vertices[0]) @ np.linalg.inv((vertices[1:] - vertices[0]).T).T
            bary = np.column_stack((1 - xi.sum(axis=1), xi))
            parameters.append(parameter)
            points.append(locations)
            values.append(evaluate(archive, macro, np.full(len(bary), int(owner)), bary))
    order = np.argsort(np.array(parameters)[:, 0])
    ordered = np.array(parameters)[order]
    if not np.isclose(np.diff(ordered[:, [0, -1]]).sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("profile must cross one complete nonoverlapping fine-cell partition")
    return dict(
        parameter=ordered,
        points=np.array(points)[order],
        actual=np.array(values)[order],
        macro_breaks=np.unique(macro_intervals[:, 1:]),
    )
