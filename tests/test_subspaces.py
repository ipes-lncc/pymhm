"""Prepared trace restriction agrees with fresh local Petrov solves."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import LocalProblem
from pymhm.core.subspaces import restrict_response


@pytest.mark.parametrize("retained", ["none", "kernel", "general"])
def test_trial_and_test_subspaces_match_new_condensation(retained):
    """All source, harmonic, coarse and global blocks preserve the restricted method."""
    rng = np.random.default_rng(91)
    left, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    right, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    diagonal = np.arange(1.0, 7.0)
    options = {}
    if retained == "kernel":
        diagonal[-1] = 0
        options.update(kernel=right[:, -1:], left_kernel=left[:, -1:])
    elif retained == "general":
        options.update(coarse_basis=right[:, -1:], test_basis=left[:, -1:])
    problem = LocalProblem(
        left @ np.diag(diagonal) @ right.T,
        rng.normal(size=(6, 4)),
        rng.normal(size=6),
        np.arange(4),
        test_coupling=rng.normal(size=(6, 4)),
        **options,
    )
    prepared = problem.condense()
    injection, tests = rng.normal(size=(4, 2)), rng.normal(size=(4, 2))
    for test in (None, tests):
        reduced = restrict_response(prepared, injection, [0, 1], test_injection=test)
        fresh = reduced.problem.condense()
        assert_allclose(reduced.source, fresh.source, atol=1e-13)
        assert_allclose(reduced.lifts, fresh.lifts, atol=1e-13)
        assert_allclose(reduced.retained_basis, fresh.retained_basis, atol=1e-13)
        a = reduced.global_contribution(np.arange(2, 2 + problem.coarse_basis.shape[1]))
        b = fresh.global_contribution(np.arange(2, 2 + problem.coarse_basis.shape[1]))
        for actual, expected in zip(a, b, strict=True):
            assert_allclose(actual, expected, atol=2e-13)


def test_invalid_subspace_maps():
    """Reject rank loss and incompatible injection dimensions before multiplication."""
    response = LocalProblem(np.eye(2), np.eye(2), [0, 0], [0, 1]).condense()
    with pytest.raises(ValueError, match="vector"):
        restrict_response(response, np.eye(2), [[0, 1]])
    with pytest.raises(ValueError, match="full column rank"):
        restrict_response(response, np.ones((2, 2)), [0, 1])
    with pytest.raises(ValueError, match="full column rank"):
        restrict_response(response, np.eye(2), [0, 1], test_injection=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="injection"):
        restrict_response(response, np.eye(3), [0, 1])
