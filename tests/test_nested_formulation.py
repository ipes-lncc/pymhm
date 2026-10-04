"""Spatial user-written hierarchy preserves flat Q2 fields and physical mean gauges."""

from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.nested_formulation import nested_problem
from pymhm import ExecutionConfig, assemble, leaf_moment
from pymhm._legacy.models.darcy.cartesian import solve_darcy_quadrilateral
from pymhm.fem.scalar.quadrilateral import qk_space
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.cartesian import CartesianMacroMesh


def pressure(points, *, quadratic):
    """Represent the declared affine or quadratic physical pressure independently."""
    return 1 + points[:, 0] + (np.sum(points**2, axis=1) if quadratic else 0.0)


def outflow(mesh, *, quadratic):
    """Differentiate pressure and take physical -grad(p).n on each exterior face."""
    return {
        int(face): lambda points, normal=mesh.normals[face]: (
            -(np.array([1.0, 0.0]) + (2 * points if quadratic else 0.0)) @ normal
        )
        for face in mesh.boundary_faces
    }


@pytest.mark.parametrize("backend", ["serial", "process"])
@pytest.mark.parametrize("natural", [False, True])
@pytest.mark.parametrize("quadratic", [False, True])
def test_spatial_hierarchy_matches_declared_leaf_discretization(backend, natural, quadratic):
    outer_mesh = CartesianMacroMesh(2)
    exact = partial(pressure, quadratic=quadratic)
    mean = 1.5 + (2 / 3 if quadratic else 0.0)
    problem, _ = nested_problem(
        2,
        source=-4.0 if quadratic else 0.0,
        dirichlet=exact,
        neumann=outflow(outer_mesh, quadratic=quadratic) if natural else None,
    )
    system = assemble(problem, execution=ExecutionConfig(backend, workers=2, batch_size=2))
    constraints = []
    if natural:
        row, offset = leaf_moment(system, lambda metadata: metadata[1])
        constraints.append((row, mean - offset))
    result = system.solve(constraints=constraints)
    flat_mesh = CartesianMacroMesh(4)
    flat = solve_darcy_quadrilateral(
        flat_mesh,
        skeleton=SkeletonSpace(flat_mesh, tuple(FaceSpace.uniform(1) for _ in flat_mesh.faces)),
        degree=2,
        local_refinement=2,
        quadrature_order=6,
        source=-4.0 if quadratic else 0.0,
        dirichlet=exact,
        neumann=outflow(flat_mesh, quadratic=quadratic) if natural else None,
        mean_pressure=mean,
    )
    for cell, child in enumerate(result.children):
        for subcell, field in enumerate(child.fields):
            i, j = 2 * (cell % 2) + subcell % 2, 2 * (cell // 2) + subcell // 2
            index = j * 4 + i
            assert_allclose(field, flat.pressure[index], atol=2e-11, rtol=2e-11)
            assert_allclose(field, exact(qk_space(flat.local_meshes[index], 2)[1]), atol=2e-11)
        assert child.raw_residual < 1e-10
    assert result.raw_residual < 1e-10
    row, offset = leaf_moment(system, lambda metadata: metadata[1])
    assert_allclose(row @ np.r_[result.trace, *result.coarse] + offset, mean, atol=2e-12)
