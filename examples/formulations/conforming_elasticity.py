"""Independent continuous-displacement reference expressed by public strain forms."""

from typing import Any

import numpy as np

from pymhm import Equation, MultiscaleProblem, SolverConfig, assemble
from pymhm.fem.traces.conforming import quadrilateral_boundary_data
from pymhm.fem.vector.quadrilateral import quadrilateral_strain_operators
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import vector_values
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.fields import DiscreteField
from pymhm.postprocessing.nodal import nodal_field


def conforming_elasticity(
    mesh: CartesianMacroMesh,
    *,
    degree: int = 1,
    constitutive: Any = None,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    source: Any = (0.0, 0.0),
    integration_field: CartesianCellField | None = None,
    dirichlet: Any = (0.0, 0.0),
    free_faces: tuple[int, ...] = (),
    order: int = 4,
    solver: str = "scipy",
) -> tuple[DiscreteField, float]:
    """Declare energy/load, strong displacement and homogeneous natural traction.

    This reference has finite positive strain stiffness and a nonempty strongly
    fixed boundary. It makes no claim of equality with a mixed stress space.
    All material cuts are supplied explicitly through integration_field.
    Returned displacement owns its executed nodal basis, and gradients are raw
    displacement derivatives rather than an equilibrated H(div) stress field.
    """
    forms = quadrilateral_strain_operators(
        mesh,
        degree,
        constitutive=constitutive,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=source,
        integration_field=integration_field,
        order=order,
    )
    _, scalar_fixed = quadrilateral_boundary_data(
        mesh, degree, neumann={face: 0.0 for face in free_faces}, order=order
    )
    if not scalar_fixed:
        raise ValueError("this reference requires a strongly fixed displacement boundary")
    ids = np.array(list(scalar_fixed), dtype=int)
    values = vector_values(dirichlet, forms.nodes[ids])
    fixed = dict(zip((2 * ids[:, None] + np.arange(2)).ravel(), values.ravel(), strict=True))
    problem = MultiscaleProblem.from_global(
        Equation(forms.matrix, forms.load), len(forms.load), fixed=fixed
    )
    solution = assemble(problem, solvers=SolverConfig(global_solver=solver)).solve()
    displacement = DiscreteField(
        nodal_field("displacement", mesh, degree, components=2), solution.trace
    )
    return displacement, solution.residual


def displacement_difference(
    fine: DiscreteField, coarse: DiscreteField, *, order: int = 3
) -> dict[str, float]:
    """Integrate displacement and raw-gradient L2 increments on the finer grid.

    Nested Cartesian meshes are required so the finer grid resolves both fields.
    Differences are finite-reference increments, not exact-solution errors.
    """
    from pymhm.fem.scalar.quadrilateral import quadrilateral_quadrature

    ratio = coarse.mesh.spacing / fine.mesh.spacing
    if (
        not np.array_equal(np.rint(ratio), ratio)
        or np.any(ratio < 1)
        or fine.mesh.bounds != coarse.mesh.bounds
    ):
        raise ValueError("reference increments require nested Cartesian grids on the same bounds")
    reference, weights = quadrilateral_quadrature(order)
    total, scale = np.zeros(2), np.zeros(2)
    for begin in range(0, len(fine.mesh.cells), 64):
        cells = np.arange(begin, min(begin + 64, len(fine.mesh.cells)))
        origin = fine.mesh.points[fine.mesh.cells[cells, 0]]
        points = (origin[:, None] + reference * fine.mesh.spacing).reshape(-1, 2)
        cell_ids = np.repeat(cells, len(reference))
        actual = fine.values_and_gradient(points, cells=cell_ids)
        comparison = coarse.values_and_gradient(points)
        measure = np.repeat(fine.mesh.areas[cells], len(reference)) * np.tile(weights, len(cells))
        for axis, (value, other) in enumerate(zip(actual, comparison, strict=True)):
            total[axis] += np.sum(
                measure * np.sum((value - other).reshape(len(points), -1) ** 2, axis=1)
            )
            scale[axis] += np.sum(measure * np.sum(value.reshape(len(points), -1) ** 2, axis=1))
    return {
        "displacement_l2": float(np.sqrt(total[0])),
        "gradient_l2": float(np.sqrt(total[1])),
        "relative_displacement_l2": float(np.sqrt(total[0] / scale[0])),
        "relative_gradient_l2": float(np.sqrt(total[1] / scale[1])),
    }
