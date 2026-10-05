"""Exact cache identity includes original moments, dtype and per-cell injection maps."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from examples.local_response_cache import ExactResponseCache, array_identity, operator_identity
from pymhm.core.contracts import LocalProblem
from pymhm.linalg.linear import LinearSolveError


def problem(kind="plain", load=None, dofs=(0, 1), **kwargs):
    """Create independently specified invertible, kernel and retained Petrov contracts."""
    matrix = np.diag([2.0, 3.0, 4.0])
    coupling = np.array([[0.0, 0.0], [1.0, 0.2], [0.3, 1.0]])
    options = {}
    if kind == "kernel":
        matrix[0, 0] = 0.0
        options["kernel"] = np.eye(3)[:, :1]
    if kind == "coarse":
        options.update(coarse_basis=np.eye(3)[:, :1], test_coupling=coupling + 0.1)
    values = np.array([0.0, 1.0, 0.2]) if load is None else np.asarray(load)
    return LocalProblem(matrix, coupling, values, np.asarray(dofs), **(options | kwargs))


@pytest.mark.parametrize("kind", ["plain", "kernel", "coarse"])
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("precision", ["double", "extended"])
def test_exact_responses_maps_and_changed_rhs_match_original_owner(kind, homogeneous, precision):
    if precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("extended correction requires a wider native mantissa")
    first = problem(kind, load=[0.0, 0.0, 0.0] if homogeneous else [0.0, 1.0, 0.2])
    original = first.condense(refinement_precision=precision)
    with ExactResponseCache(refinement_precision=precision) as cache:
        value = cache.condense(first)
        shifted = problem(kind, load=first.load, dofs=(3, 7))
        repeated = cache.condense(shifted)
        assert repeated.problem is shifted
        assert_array_equal(repeated.problem.trace_dofs, [3, 7])
        for name in ("source", "lifts"):
            assert_array_equal(getattr(value, name), getattr(original, name), strict=True)
            assert getattr(value, name) is getattr(repeated, name)
            assert not getattr(value, name).flags.writeable
        assert cache.factorizations == 1 and cache.response_hits == 1
        changed = problem(kind, load=[0.0, -0.4, 0.8], dofs=(3, 7))
        acquired = cache.condense(changed)
        direct = changed.condense(refinement_precision=precision)
        assert acquired.lifts is value.lifts
        assert_allclose(acquired.source, direct.source, rtol=3e-14, atol=3e-15)
        if direct.coarse_vectors is not None:
            assert_array_equal(acquired.coarse_vectors, direct.coarse_vectors, strict=True)
        assert cache.source_solves == 2 and cache.factorizations == 1
        trace = np.zeros(8)
        trace[[3, 7]] = [0.15, -0.3]
        coarse = np.full(changed.coarse_basis.shape[1], 0.25)
        assert_allclose(
            acquired.reconstruct(trace[changed.trace_dofs], coarse),
            direct.reconstruct(trace[changed.trace_dofs], coarse),
            rtol=3e-14,
            atol=3e-15,
        )
        factor = next(iter(cache._operators.values())).factor
    with pytest.raises(RuntimeError, match="closed"):
        factor.solve(np.zeros(factor._matrix.shape[0]))


def test_operator_moments_test_basis_and_native_dtype_are_identity_inputs():
    base = problem("kernel")
    altered = problem("kernel", constraints=2 * np.eye(3)[:, :1])
    rotated = problem("kernel", kernel=-np.eye(3)[:, :1])
    assert len({operator_identity(p) for p in (base, altered, rotated)}) == 3
    for first, second in (
        (np.array([0.0]), np.array([-0.0])),
        (np.ones(1), np.ones(1, dtype=np.float32)),
        (np.ones(1), np.ones((1, 1))),
    ):
        assert array_identity(first) != array_identity(second)
    native = np.ones(1, dtype=np.longdouble)
    assert (array_identity(np.ones(1)) == array_identity(native)) == (
        native.dtype == np.dtype(float)
    )
    with ExactResponseCache(max_operators=1, max_sources_per_operator=1) as cache:
        cache.condense(base)
        old = next(iter(cache._operators.values())).factor
        cache.condense(altered)
        assert len(cache._operators) == 1
        with pytest.raises(RuntimeError, match="closed"):
            old.solve(np.zeros(old._matrix.shape[0]))
        cache.condense(altered.with_load([0.0, -0.2, 0.6]))
        assert len(next(iter(cache._operators.values())).sources) == 1


def test_cache_requires_explicit_lifetime():
    cache = ExactResponseCache()
    with pytest.raises(RuntimeError, match="must be open"):
        cache.condense(problem())
    with pytest.raises(ValueError):
        ExactResponseCache(refinement_precision="wrong")
    with pytest.raises(ValueError):
        ExactResponseCache(solver="pyamg")
    with pytest.raises(ValueError):
        ExactResponseCache(max_operators=0)
    with cache, pytest.raises(RuntimeError, match="already open"):
        cache.__enter__()
    assert cache._operators == {}


def test_failed_native_response_releases_factor_before_cache_exit(monkeypatch):
    from examples import local_response_cache as owner

    factors = []
    original = owner.factorize

    def broken_factor(matrix, **kwargs):
        factor = original(matrix, **kwargs)
        factors.append(factor)
        factor._solve = lambda rhs: np.full_like(rhs, np.nan)
        return factor

    monkeypatch.setattr(owner, "factorize", broken_factor)
    with ExactResponseCache() as cache:
        with pytest.raises(LinearSolveError, match="nonfinite"):
            cache.condense(problem())
        assert not cache._operators
        assert len(factors) == 1
        with pytest.raises(RuntimeError, match="closed"):
            factors[0].solve(np.zeros(3))


def test_changed_extended_source_is_a_distinct_original_load_contract():
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("extended correction requires a wider native mantissa")
    base = problem(load=[0.0, 3.0, 0.0])
    wider = np.array([0.0, 3.0, 0.0], dtype=np.longdouble)
    wider[1] += np.longdouble(2) ** -58
    changed = base.with_load(wider, preserve_precision=True)
    with ExactResponseCache(refinement_precision="extended") as cache:
        cache.condense(base)
        response = cache.condense(changed)
        direct = changed.condense(refinement_precision="extended")
        assert response.source.dtype == np.dtype(np.longdouble)
        assert response.problem.load[1] > np.longdouble(3)
        assert response.source[1] == direct.source[1]
        assert cache.factorizations == 1 and cache.source_solves == 2
