"""Storage-only condensation changes preserve original physical equations."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from examples.helmholtz_compact_family import CompactFamily, CompactLocal
from pymhm.helmholtz import _HelmholtzFactory, solve_helmholtz
from pymhm.helmholtz_spaces import helmholtz_skeleton
from pymhm.loads import split_point_sources
from pymhm.quadrilateral import CartesianMacroMesh
from pymhm.reservoir import CartesianCellField


@pytest.mark.parametrize("boundary", ["absorbing", "dirichlet", "mixed"])
def test_compact_all_degrees_match_original_equations(boundary):
    mesh = CartesianMacroMesh(2, 1)
    skeleton = helmholtz_skeleton(mesh, 1.2, degree=4)
    exterior = list(mesh.boundary_faces)
    absorbing = (
        dict.fromkeys(exterior, 0.2 + 0.1j)
        if boundary == "absorbing"
        else {int(exterior[0]): -0.2j}
        if boundary == "mixed"
        else {}
    )
    neumann = {int(exterior[-1]): 0.0} if boundary == "mixed" else {}
    material = CartesianCellField(np.array([[1.0], [2.0], [4.0]]), (1 / 3, 1))
    points = ((0.5, 0.5, 0.1),)
    options = dict(
        omega=1.2,
        density=material,
        bulk_modulus=3.0,
        source=1 + 0.3j,
        dirichlet=0.2 + 0.4j,
        absorbing=absorbing,
        neumann=neumann,
        point_sources=points,
        degree=3,
        local_refinement=2,
        quadrature_order=9,
    )
    original = solve_helmholtz(mesh, skeleton=skeleton, **options)
    factory = _HelmholtzFactory(
        mesh,
        skeleton,
        1.2,
        3,
        2,
        9,
        material,
        3.0,
        1 + 0.3j,
        split_point_sources(mesh, points),
        0.2 + 0.4j,
        absorbing,
        neumann,
        None,
    )
    compact = CompactFamily.prepare(factory)
    assert_array_equal(compact.matrix.toarray(), original.system.matrix.toarray())
    assert_array_equal(compact.rhs, original.system.rhs)
    for item, response in zip(compact.local, original.system.responses, strict=True):
        assert_array_equal(item.coupling.toarray(), response.problem.coupling)
        assert_array_equal(item.source, response.source)
        assert_array_equal(item.lifts, response.lifts)
        assert item.storage_bytes > 0
    for degree in range(5):
        value = compact.solve(degree)
        direct = solve_helmholtz(
            mesh, skeleton=helmholtz_skeleton(mesh, 1.2, degree=degree), **options
        )
        assert_allclose(value.trace, direct.trace, rtol=2e-10, atol=2e-11)
        assert_allclose(value.pressure, direct.pressure, rtol=2e-10, atol=2e-11)
        assert value.residual < 1e-12
        assert value.local_residual_max < 1e-12
        assert_allclose(value.balance, 0, atol=2e-12)


def test_compact_rejects_retained_modes_and_corrupted_responses():
    from pymhm.hybrid import LocalProblem

    problem = LocalProblem(
        np.eye(2), np.eye(2), np.ones(2), np.arange(2), coarse_basis=np.ones((2, 1))
    )
    with pytest.raises(ValueError, match="retained"):
        CompactLocal.from_response(problem.condense(), (None, None, np.zeros(2), {}))
    problem = LocalProblem(np.eye(2), np.eye(2), np.ones(2), np.arange(2))
    item = CompactLocal.from_response(problem.condense(), (None, None, np.zeros(2), {}))
    bad = replace(item, source=item.source + 0.1)
    with pytest.raises(ValueError, match="original local"):
        bad.reconstruct(np.zeros(2))


def test_compact_spawn_preserves_arrays_and_fields():
    mesh = CartesianMacroMesh(2, 1)
    skeleton = helmholtz_skeleton(mesh, 1.2, degree=2)
    factory = _HelmholtzFactory(
        mesh,
        skeleton,
        1.2,
        3,
        2,
        8,
        1.0,
        2.0,
        1.0 + 0.4j,
        split_point_sources(mesh, ((0.5, 0.5, 0.1),)),
        0j,
        dict.fromkeys(mesh.boundary_faces, 0.0),
        {},
        None,
    )
    serial = CompactFamily.prepare(factory)
    parallel = CompactFamily.prepare(factory, backend="process", workers=2)
    assert_array_equal(serial.matrix.toarray(), parallel.matrix.toarray())
    assert_array_equal(serial.rhs, parallel.rhs)
    first, second = serial.solve(1), parallel.solve(1)
    assert_array_equal(first.trace, second.trace)
    assert_array_equal(first.pressure, second.pressure)
