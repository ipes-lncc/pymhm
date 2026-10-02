"""Verify abstract Petrov condensation against independent complete systems."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.hybrid import HybridSystem, LocalProblem


@pytest.mark.parametrize("retained", ["none", "kernel", "general"])
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_distinct_test_trial_spaces_match_full_system(retained, scale):
    rng = np.random.default_rng(217)
    left, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    right, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    eigenvalues = np.arange(1.0, 8.0)
    if retained == "kernel":
        eigenvalues[-2:] = 0
    matrix = scale * (left @ np.diag(eigenvalues) @ right.T)
    coupling = scale * rng.normal(size=(7, 3))
    test_coupling = rng.normal(size=(7, 3))
    force, boundary = scale * rng.normal(size=7), rng.normal(size=3)
    options = {}
    if retained != "none":
        z, w = right[:, -2:], left[:, -2:]
        options = {
            "constraints": np.diag(np.arange(1, 8)) @ z,
            "test_constraints": np.diag(np.arange(2, 9)) @ w,
        }
        if retained == "kernel":
            options.update(kernel=z, left_kernel=w)
        else:
            options.update(coarse_basis=z, test_basis=w)
    problem = LocalProblem(
        matrix, coupling, force, np.arange(3), test_coupling=test_coupling, **options
    )
    full = np.block([[matrix, coupling], [test_coupling.T, np.zeros((3, 3))]])
    expected = np.linalg.solve(full, np.r_[force, boundary])
    for candidate in (problem, problem.with_load(force)):
        response = candidate.condense()
        result = HybridSystem.from_responses([response], boundary_load=boundary).solve()
        assert_allclose(result.fields[0], expected[:7], atol=3e-11, rtol=3e-11)
        assert_allclose(result.trace, expected[7:], atol=3e-11, rtol=3e-11)
        assert_allclose(
            (matrix @ result.fields[0] + coupling @ result.trace) / scale, force / scale, atol=2e-11
        )
        assert_allclose(test_coupling.T @ result.fields[0], boundary, atol=2e-11)


def test_different_nullspaces_get_independent_default_constraints():
    problem = LocalProblem(
        [[0.0, 1.0], [0.0, 0.0]],
        [[1.0], [1.0]],
        [2.0, 3.0],
        np.array([0]),
        kernel=[[1.0], [0.0]],
        left_kernel=[[0.0], [1.0]],
        test_coupling=[[2.0], [1.0]],
    )
    result = HybridSystem([problem], boundary_load=[4.0]).solve()
    assert_allclose(result.trace, [3.0])
    assert_allclose(result.fields[0], [2.5, -1.0])
    with pytest.raises(ValueError, match="matching test and trial"):
        problem.condense("pyamg")


@pytest.mark.parametrize(
    "options,match",
    [
        ({"left_kernel": [[1.0], [0.0]], "coarse_basis": [[1.0], [0.0]]}, "exclusive"),
        ({"left_kernel": [[1.0]]}, "shape"),
        ({"kernel": [[1.0], [0.0]], "test_basis": [[1.0], [0.0]]}, "exclusive"),
        ({"coarse_basis": [[1.0], [0.0]], "test_constraints": [[0.0], [1.0]]}, "nonsingular"),
        ({"test_coupling": [[1.0]]}, "shape"),
        ({"test_basis": [[1.0], [0.0]]}, "shape"),
        ({"kernel": [[1.0], [0.0]], "left_kernel": [[1.0], [0.0]]}, "nullspace"),
    ],
)
def test_petrov_contract_rejects_inconsistent_spaces(options, match):
    with pytest.raises(ValueError, match=match):
        LocalProblem([[0.0, 1.0], [0.0, 0.0]], np.eye(2), [0, 0], np.arange(2), **options)


def test_response_contract_checks_backend_shapes():
    problem = LocalProblem(np.eye(2), np.eye(2), [1, 2], np.arange(2))
    with pytest.raises(ValueError, match="condensation solution"):
        problem.response_from_solution(np.eye(2))
    response = problem.condense()
    for indices in ([0], [1.0]):
        with pytest.raises(ValueError, match="coarse_dofs"):
            response.global_contribution(indices)
    with pytest.raises(TypeError, match="LocalResponse"):
        HybridSystem.from_responses([None])
    with pytest.raises(ValueError, match="metadata"):
        HybridSystem.from_responses([response], metadata=[])
    system = HybridSystem.from_responses([response], metadata=["physical mesh"])
    assert system.local_metadata == ("physical mesh",)
