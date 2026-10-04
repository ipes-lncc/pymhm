"""Nested real/complex trace solves reuse the exact original local operators."""

from copy import copy
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.helmholtz_trace_family import (
    polynomial_injection,
    restrict_helmholtz_trace,
    solve_restricted_coordinates,
)
from pymhm._legacy.models.waves.helmholtz import solve_helmholtz
from pymhm.fem.traces.helmholtz import helmholtz_skeleton
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.cartesian import CartesianMacroMesh


@pytest.mark.parametrize("boundary", ["absorbing", "dirichlet", "mixed"])
def test_all_nested_degrees_match_direct_physical_fields(boundary):
    mesh = CartesianMacroMesh(2, 2)
    exterior = list(mesh.boundary_faces)
    absorbing = (
        {int(face): 0.2 + 0.1j for face in exterior}
        if boundary == "absorbing"
        else {int(exterior[0]): -0.1 + 0.3j}
        if boundary == "mixed"
        else None
    )
    options = dict(
        omega=1.7,
        density=CartesianCellField(np.array([[1.0, 2.0], [3.0, 1.5]]), (0.5, 0.5)),
        source=lambda x: 1 + x[:, 0] + 0.4j * x[:, 1],
        point_sources=((0.5, 0.25, 0.3), (0.5, 0.5, -0.1)),
        dirichlet=lambda x: 0.1 + 0.2 * x[:, 0] + 0.3j * x[:, 1],
        absorbing=absorbing,
        degree=3,
        local_refinement=2,
        quadrature_order=9,
    )
    prepared = solve_helmholtz(mesh, skeleton=helmholtz_skeleton(mesh, 1.7, degree=4), **options)
    local_ids = [id(response.problem.matrix) for response in prepared.system.responses]
    for degree in range(5):
        result = restrict_helmholtz_trace(prepared, degree)
        direct = solve_helmholtz(mesh, skeleton=result.skeleton, **options)
        assert result.solution.system is prepared.system
        assert [id(r.problem.matrix) for r in result.solution.system.responses] == local_ids
        assert_allclose(result.trace, direct.trace, rtol=3e-10, atol=3e-11)
        for first, second in zip(result.solution.pressure, direct.pressure, strict=True):
            assert_allclose(first, second, rtol=3e-10, atol=3e-11)
        assert result.restricted_residual < 1e-12
        assert result.original_trace_residual < 1e-12
        assert_allclose(result.solution.conservation_residuals(), 0, atol=5e-12)


def test_segmentwise_interleaving_and_rejected_nonpolynomial_spaces():
    mesh = CartesianMacroMesh(1, 1)
    face = FaceSpace((0, 0.3, 1), (2, 4))
    high = SkeletonSpace(mesh, tuple(face for _ in mesh.faces), components=2)
    low, injection, selected = polynomial_injection(high, 1)
    assert np.array_equal(selected[:8], [0, 1, 2, 3, 6, 7, 8, 9])
    coefficients = np.arange(low.size, dtype=float) + 0.2
    embedded = np.asarray(injection @ coefficients)
    points = np.array([0.1, 0.2, 0.4, 0.8])
    for index in range(len(mesh.faces)):
        expected = low.faces[index].evaluate(points) @ coefficients[low.dofs(index)].reshape(-1, 2)
        actual = high.faces[index].evaluate(points) @ embedded[high.dofs(index)].reshape(-1, 2)
        assert_allclose(actual, expected, atol=2e-14)
    with pytest.raises(ValueError, match="polynomial"):
        polynomial_injection(helmholtz_skeleton(mesh, 1.7, degree=2, oscillatory=True), 0)
    with pytest.raises(ValueError, match="polynomial"):
        polynomial_injection(high, 3)


def test_nonrepresentable_neumann_data_are_not_silently_projected():
    mesh = CartesianMacroMesh(1, 1)
    exterior = list(mesh.boundary_faces)
    result = solve_helmholtz(
        mesh,
        omega=1.2,
        skeleton=helmholtz_skeleton(mesh, 1.2, degree=2),
        degree=3,
        local_refinement=2,
        absorbing=None,
        neumann={int(exterior[0]): lambda x, n: x[:, 0] + x[:, 1]},
    )
    with pytest.raises(ValueError, match="Neumann trace"):
        restrict_helmholtz_trace(result, 0)


def test_quadrature_eight_and_nine_give_the_same_pixel_operator():
    mesh = CartesianMacroMesh(2, 1)
    material = CartesianCellField(np.array([[1.0], [4.0], [2.0]]), (1 / 3, 1))
    options = dict(
        omega=2.1,
        density=material,
        degree=3,
        local_refinement=2,
        source=1 + 0.4j,
        point_sources=((0.5, 0.25, 1.0),),
    )
    first = solve_helmholtz(mesh, quadrature_order=8, **options)
    second = solve_helmholtz(mesh, quadrature_order=9, **options)
    for a, b in zip(first.system.responses, second.system.responses, strict=True):
        assert_allclose(a.problem.matrix.toarray(), b.problem.matrix.toarray(), atol=2e-14)
        assert_allclose(a.problem.load, b.problem.load, atol=5e-16)
    assert_allclose(first.trace, second.trace, atol=2e-12, rtol=2e-12)
    for a, b in zip(first.pressure, second.pressure, strict=True):
        assert_allclose(a, b, atol=3e-12, rtol=3e-12)


def test_original_global_guard_and_fully_prescribed_trace(monkeypatch):
    from scipy import sparse

    skeleton = helmholtz_skeleton(CartesianMacroMesh(1, 1), 1.0, degree=0)
    matrix = sparse.eye(skeleton.size, format="csc")
    _, _, free, trace, residual = solve_restricted_coordinates(
        skeleton, matrix, np.zeros(skeleton.size), dict.fromkeys(range(skeleton.size), 0.0), 0
    )
    assert len(free) == 0
    assert np.array_equal(trace, np.zeros(skeleton.size))
    assert residual == 0
    monkeypatch.setattr(
        "examples.helmholtz_trace_family.solve_linear", lambda matrix, rhs, **kwargs: 0 * rhs
    )
    with pytest.raises(ValueError, match="original Helmholtz trace equations"):
        solve_restricted_coordinates(skeleton, matrix, np.ones(skeleton.size), {}, 0)


def test_restriction_rejects_a_response_with_retained_modes():
    from types import SimpleNamespace

    problem = SimpleNamespace(coarse_basis=np.ones((1, 1)))
    prepared = SimpleNamespace(system=SimpleNamespace(responses=[SimpleNamespace(problem=problem)]))
    with pytest.raises(ValueError, match="no retained"):
        restrict_helmholtz_trace(prepared, 0)


def test_original_field_gate_rejects_corrupted_lift_with_unchanged_schur():
    mesh = CartesianMacroMesh(2)
    prepared = solve_helmholtz(
        mesh, omega=1.2, degree=3, local_refinement=2, source=1 + 0.4j, dirichlet=0.2j
    )
    original = prepared.system.responses[0]
    corrupted = replace(original, source=original.source + 0.5)
    system = copy(prepared.system)
    system.responses = (corrupted, *prepared.system.responses[1:])
    prepared = replace(prepared, system=system)
    with pytest.raises(ValueError, match="original Helmholtz local equations"):
        restrict_helmholtz_trace(prepared, 0)
