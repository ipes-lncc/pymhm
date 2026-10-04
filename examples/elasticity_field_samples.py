"""One-sided polynomial field sampling and exact local displacement projection diagnostics."""

from typing import Any

import numpy as np

from examples.plot_mesh import macro_profile_breaks
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import reference_basis
from pymhm.meshes.triangle import TriangleMesh


def sample_elasticity_fields(solution: Any, refinement: int = 6) -> dict[str, np.ndarray]:
    """Sample physical fields on disconnected plotting meshes, retaining every fine-cell side.

    A separate display triangulation is mapped into each fine triangle or
    rectangle. No vertices or fields are averaged across an element boundary.
    Stress includes both independently computed off-diagonal components.
    """
    tensor = hasattr(solution, "degree")
    if tensor:
        display = TriangleMesh.unit_square(refinement)
        reference = display.points
    else:
        display = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]]).submesh(
            0, refinement
        )
        reference = np.column_stack((1 - display.points.sum(axis=1), display.points))
        basis = reference_basis(solution.displacement_degree, reference)[0]
    coordinates, fields, cells = [], [], []
    offset = 0
    for macro, fine in enumerate(solution.local_meshes):
        if tensor:
            physical = fine.points[fine.cells[:, 0], None] + reference * fine.spacing
            u, stress, _, _ = solution.evaluate(macro, reference)
        else:
            physical = np.einsum("qi,tia->tqa", reference, fine.points[fine.cells])
            u = np.einsum("qi,tia->tqa", basis, solution.displacement[macro])
            stress = solution.family.evaluate(fine, solution.stress[macro], reference)[0]
        coordinates.append(physical.reshape(-1, 2))
        fields.append(
            np.concatenate((u, stress.reshape(*stress.shape[:2], 4)), axis=2).reshape(-1, 6)
        )
        for cell in range(len(fine.cells)):
            cells.append(display.cells + offset + cell * len(reference))
        offset += len(fine.cells) * len(reference)
    macro = solution.skeleton.mesh
    return dict(
        points=np.concatenate(coordinates),
        cells=np.concatenate(cells),
        actual=np.concatenate(fields),
        macro_edges=macro.points[macro.faces],
    )


def projection_decomposition(
    solution: Any, exact: Any, order: int = 8
) -> tuple[dict[str, Any], tuple[np.ndarray, ...]]:
    """Separate best local L2 approximation and solved-coefficient error by orthogonal projection.

    The identity ||u_h-u||²=||u_h-Pu||²+||Pu-u||² is evaluated independently
    for both displacement components. It is a quantitative diagnostic, not a
    claim that the MHM displacement equals its local L2 projection.
    """
    bary, weights = triangle_quadrature(order)
    basis = reference_basis(solution.displacement_degree, bary)[0]
    gram = basis.T @ (weights[:, None] * basis)
    error_norms = np.zeros((3, 2))
    cross = np.zeros(2)
    projected = []
    for fine, values in zip(solution.local_meshes, solution.displacement, strict=True):
        points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
        target = exact(points.reshape(-1, 2)).reshape(*points.shape[:2], 2)
        moments = np.einsum("q,qi,tqa->ita", weights, basis, target)
        coefficients = (
            np.linalg.solve(gram, moments.reshape(len(gram), -1))
            .reshape(moments.shape)
            .transpose(1, 0, 2)
        )
        projected.append(coefficients)
        projection_error = np.einsum("qi,tia->tqa", basis, coefficients) - target
        discrete_error = np.einsum("qi,tia->tqa", basis, values - coefficients)
        for i, error in enumerate(
            (projection_error, discrete_error, projection_error + discrete_error)
        ):
            error_norms[i] += np.einsum("t,q,tqa->a", fine.areas, weights, error**2)
        cross += np.einsum("t,q,tqa,tqa->a", fine.areas, weights, projection_error, discrete_error)
    return dict(
        projection_l2=np.sqrt(error_norms[0]).tolist(),
        discrete_l2=np.sqrt(error_norms[1]).tolist(),
        total_l2=np.sqrt(error_norms[2]).tolist(),
        orthogonality=cross.tolist(),
        pythagorean_defect=(error_norms[2] - error_norms[0] - error_norms[1]).tolist(),
    ), tuple(projected)


def sample_elasticity_profile(
    solution: Any, start: Any, end: Any, samples: int = 9
) -> dict[str, np.ndarray]:
    """Evaluate independent polynomial segments between every intersected fine-cell boundary.

    Every segment retains both endpoints from its own cell. Thus plotting each
    segment separately preserves jumps and does not draw an artificial line
    joining two one-sided values. Macroface crossings are stored separately.
    Profiles coincident with a fine interface require a specified side and are
    rejected by this single-side sampler.
    """
    first, last = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    macro_breaks = macro_profile_breaks(solution.skeleton.mesh, first, last)
    tensor = hasattr(solution, "degree")
    parameters, values, locations = [], [], []
    for macro, fine in enumerate(solution.local_meshes):
        vertices = fine.points[fine.cells]
        inverse = (
            None if tensor else np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        )
        breaks = macro_profile_breaks(fine, first, last)
        for a, b in zip(breaks[:-1], breaks[1:], strict=True):
            midpoint = first + (a + b) / 2 * (last - first)
            if tensor:
                xi = (midpoint - vertices[:, 0]) / fine.spacing
                contained = (xi.min(axis=1) >= -1e-12) & (xi.max(axis=1) <= 1 + 1e-12)
            else:
                xi = np.einsum("tij,tj->ti", inverse, midpoint - vertices[:, 0])
                contained = (xi.min(axis=1) >= -1e-12) & (xi.sum(axis=1) <= 1 + 1e-12)
            ids = np.flatnonzero(contained)
            if not len(ids):
                continue
            if len(ids) != 1:
                raise ValueError("profile coincides with a fine interface; select a one-sided line")
            cell = int(ids[0])
            parameter = np.linspace(a, b, samples)
            points = first + parameter[:, None] * (last - first)
            if tensor:
                reference = (points - vertices[cell, 0]) / fine.spacing
                u, stress, _, _ = solution.evaluate(macro, reference)
                u, stress = u[cell], stress[cell]
            else:
                coordinates = (points - vertices[cell, 0]) @ inverse[cell].T
                bary = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
                u = (
                    reference_basis(solution.displacement_degree, bary)[0]
                    @ solution.displacement[macro][cell]
                )
                stress = solution.family.evaluate(fine, solution.stress[macro], bary)[0][cell]
            parameters.append(parameter)
            values.append(np.column_stack((u, stress.reshape(-1, 4))))
            locations.append(points)
    ordering = np.argsort(np.array(parameters)[:, 0])
    parameter = np.array(parameters)[ordering]
    if not np.isclose(np.diff(parameter[:, [0, -1]], axis=1).sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("profile must cross a complete nonoverlapping mesh partition")
    return dict(
        parameter=parameter,
        actual=np.array(values)[ordering],
        points=np.array(locations)[ordering],
        macro_breaks=macro_breaks,
    )
