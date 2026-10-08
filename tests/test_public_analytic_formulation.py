"""Public metric operators reproduce all executed analytical harmonic/source stages."""

import pickle

import numpy as np
import pytest

from examples.formulations.analytic_darcy import (
    analytic_constraints,
    analytic_darcy,
    define_analytic_darcy,
    recover_analytic_darcy,
)
from pymhm import ExecutionConfig, TetraMesh, TriangleMesh, assemble
from pymhm._legacy.models.darcy.analytic import solve_darcy_analytic
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetrahedron_quadrature


def variable_source(points):
    return 1 + points[:, 0] + points[:, 1] ** 2


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("source", [0.5, variable_source])
def test_public_analytic_equations_preserve_harmonic_and_full_source_fields(dimension, source):
    mesh = TriangleMesh.unit_square() if dimension == 2 else TetraMesh.unit_cube()
    tensor = 2 * np.eye(dimension) + 0.25
    options = dict(source=source, permeability=tensor, dirichlet=1.0)
    actual, expected = analytic_darcy(mesh, **options), solve_darcy_analytic(mesh, **options)
    np.testing.assert_allclose(actual.hybrid.trace, expected.hybrid.trace, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(actual.hybrid.coarse, expected.hybrid.coarse, atol=1e-12, rtol=1e-10)
    bary, _ = triangle_quadrature(4) if dimension == 2 else tetrahedron_quadrature(4)
    for cell in range(len(mesh.cells)):
        for value, reference in zip(
            actual.evaluate(cell, bary), expected.evaluate(cell, bary), strict=True
        ):
            np.testing.assert_allclose(value, reference, atol=1e-12, rtol=1e-10)
    replay = pickle.loads(pickle.dumps(actual))
    for value, reference in zip(replay.evaluate(0, bary), actual.evaluate(0, bary), strict=True):
        np.testing.assert_array_equal(value, reference)
    assert pickle.loads(b"cpymhm._legacy.models.darcy.analytic\nAnalyticDarcySolution\n.") is type(
        actual
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_public_analytic_natural_data_and_physical_mean(dimension):
    mesh = TriangleMesh.unit_square() if dimension == 2 else TetraMesh.unit_cube()
    natural = {int(face): 0.0 for face in mesh.boundary_faces}
    actual = analytic_darcy(mesh, neumann=natural, mean_pressure=2.5)
    bary, _ = triangle_quadrature(3) if dimension == 2 else tetrahedron_quadrature(3)
    for cell in range(len(mesh.cells)):
        p, q = actual.evaluate(cell, bary)
        np.testing.assert_allclose(p, 2.5, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(q, 0, atol=1e-12, rtol=1e-10)


def test_analytic_source_stages_spawn_and_input_contracts():
    mesh = TriangleMesh.unit_square()
    definition = define_analytic_darcy(mesh, source=variable_source, dirichlet=1)
    system = assemble(definition.problem, execution=ExecutionConfig("process", workers=2))
    result = recover_analytic_darcy(
        definition, system, system.solve(constraints=analytic_constraints(definition, system))
    )
    serial = analytic_darcy(mesh, source=variable_source, dirichlet=1)
    np.testing.assert_allclose(result.hybrid.trace, serial.hybrid.trace, atol=1e-12, rtol=1e-10)
    with pytest.raises(TypeError, match="triangle or tetrahedron"):
        define_analytic_darcy(object())
    with pytest.raises(ValueError, match="finite"):
        define_analytic_darcy(mesh, mean_pressure=np.inf)
