"""Compare offline/online source queries with independently rebuilt hybrid systems."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.hybrid import HybridSystem, LocalProblem
from pymhm.offline import (
    LocalFactorCache,
    OfflineHybridSystem,
    OfflineLocalProblem,
    condense_cached,
)
from pymhm.solvers import LinearSolveError, factorize


def problems(kind: str) -> list[LocalProblem]:
    """Build two coupled cells with symmetric, retained or Petrov local operators."""
    result = []
    for cell in range(2):
        matrix = np.array([[1.0, -1.0], [-1.0, 1.0]])
        kwargs: dict[str, Any] = {"kernel": np.ones((2, 1)), "constraints": np.ones((2, 1)) / 2}
        if kind == "retained":
            matrix += np.eye(2) * 0.3
            kwargs["coarse_basis"] = kwargs.pop("kernel")
        if kind == "petrov":
            left = np.diag([1.0, 2.0])
            matrix = left @ matrix
            kwargs["left_kernel"] = np.array([[1.0], [0.5]])
            kwargs["test_constraints"] = np.array([[0.2], [0.5]])
            kwargs["test_coupling"] = np.array([[0.7, 0], [0, -1.4]])
        result.append(
            LocalProblem(
                matrix, np.diag([1.0, -1.0]), [0.0, 0.0], np.array([cell, cell + 1]), **kwargs
            )
        )
    return result


@pytest.mark.parametrize("kind", ["kernel", "retained", "petrov"])
def test_online_sources_and_boundary_values_match_reassembly(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse local/global factors for changing sources and inhomogeneous boundary moments."""
    from pymhm import solvers

    cells = problems(kind)
    count = []
    original = solvers._equilibrated_lu

    def recorded(matrix: Any) -> Any:
        """Count genuine LU factorizations without replacing their numerical work."""
        count.append(matrix.shape)
        return original(matrix)

    monkeypatch.setattr(solvers, "_equilibrated_lu", recorded)
    with OfflineHybridSystem(cells) as online:
        assert len(count) == 3
        assert online.system.matrix.shape == (5, 5)
        for scalar in (0.0, 1.0, -2.0):
            loads = [scalar * np.array([1.0, 2.0]), scalar * np.array([-1.0, 0.5])]
            boundary = np.array([scalar + 1, 0, 2.0])
            result = online.solve(loads, boundary_load=boundary)
            before = len(count)
            expected = HybridSystem(
                [p.with_load(f) for p, f in zip(cells, loads, strict=True)], boundary_load=boundary
            ).solve()
            assert len(count) == before + 3
            assert_allclose(result.trace, expected.trace, atol=1e-12)
            assert_allclose(result.fields, expected.fields, atol=1e-12)
        actual = online.solve_many([[np.zeros(2), np.zeros(2)]] * 2)
        assert_allclose(actual[0].fields, actual[1].fields)


def test_online_physical_moments_and_prescribed_flux_values() -> None:
    """Account for changing source liftings in global physical mean constraints."""
    cells = problems("kernel")
    weights = [np.ones(2) / 2] * 2
    with OfflineHybridSystem(cells, fixed={0: 0.0, 2: 0.0}, moments=[(weights, 0.0)]) as online:
        for value in (0.0, 1.0, -3.0):
            loads = [np.array([value, 0.0]), np.array([0.0, -value])]
            result = online.solve(loads, fixed={0: 0.2, 2: 0.2}, targets=[2.0])
            rebuilt = HybridSystem([p.with_load(f) for p, f in zip(cells, loads, strict=True)])
            expected = rebuilt.solve(
                fixed={0: 0.2, 2: 0.2}, constraints=[rebuilt.mean_constraint(weights, 2.0)]
            )
            assert_allclose(result.fields, expected.fields, atol=1e-12)
            assert_allclose(sum(w @ u for w, u in zip(weights, result.fields, strict=True)), 2.0)
        with pytest.raises(ValueError, match="gauge changed"):
            online.solve([np.ones(2), np.ones(2)])


def test_offline_local_lifecycle_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Close local resources on exceptional preparation and context exit."""
    p = problems("kernel")[0]
    with OfflineLocalProblem(p) as prepared:
        actual = prepared.response([1.0, -1.0])
        assert_allclose(actual.source, p.with_load([1.0, -1.0]).condense().source)
        assert actual.lifts is prepared.template.lifts
    prepared.close()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.__enter__()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.response([0.0, 0.0])
    with pytest.raises(TypeError, match="LocalProblem"):
        OfflineLocalProblem(None)
    from pymhm import offline

    class BrokenFactor:
        """Expose failed-solve cleanup without simulating numerical correctness."""

        closed = False

        def solve(self, rhs: Any) -> Any:
            """Raise after allocation to test deterministic release."""
            raise LinearSolveError("deliberate")

        def close(self) -> None:
            """Record native release after failed preparation."""
            self.closed = True

    broken = BrokenFactor()
    monkeypatch.setattr(offline, "factorize", lambda *a, **kw: broken)
    with pytest.raises(LinearSolveError, match="deliberate"):
        OfflineLocalProblem(p)
    assert broken.closed


@pytest.mark.parametrize("fixed", [{-1: 0}, {3: 0}, {True: 0}, {0: np.nan}])
def test_offline_fixed_validation(fixed: dict[int, float]) -> None:
    """Reject invalid prepared trace conditions before global factorization."""
    with pytest.raises(ValueError, match="fixed DOFs"):
        OfflineHybridSystem(problems("kernel"), fixed=fixed)


@pytest.mark.parametrize("boundary", [[0.0], [0.0, 0.0, np.nan], np.zeros(3, complex)])
def test_online_boundary_validation(boundary: Any) -> None:
    """Reject invalid boundary arrays without changing the prepared system."""
    with (
        OfflineHybridSystem(problems("kernel")) as online,
        pytest.raises(ValueError, match="boundary_load"),
    ):
        online.solve([np.zeros(2)] * 2, boundary_load=boundary)


def test_offline_empty_fixed_all_and_online_contracts() -> None:
    """Cover empty systems, all prescribed traces and immutable query topology."""
    with pytest.raises(ValueError, match="at least one"):
        OfflineHybridSystem([])
    p = LocalProblem([[2.0]], [[1.0]], [1.0], np.array([0]))
    with OfflineHybridSystem([p], fixed={0: 2.0}) as online:
        assert_allclose(online.solve([[4.0]]).fields, [[1.0]])
        with pytest.raises(ValueError, match="one load"):
            online.solve([])
        with pytest.raises(ValueError, match="locations"):
            online.solve([[4.0]], fixed={})
        with pytest.raises(ValueError, match="one target"):
            online.solve([[4.0]], targets=[0.0])
    online.close()
    with pytest.raises(RuntimeError, match="closed"):
        online.solve([[4.0]])
    with pytest.raises(RuntimeError, match="closed"):
        online.__enter__()


def test_prepared_global_factor_must_match_exact_operator() -> None:
    """Prevent accidental reuse after operator, constraint or boundary-topology changes."""
    system = HybridSystem([LocalProblem([[2.0]], [[1.0]], [1.0], np.array([0]))])
    with factorize([[2.0]]) as wrong:
        assert not wrong.matches(np.eye(2))
        with pytest.raises(ValueError, match="does not match"):
            system.solve(factorization=wrong)
        with pytest.raises(ValueError, match="does not match"):
            system.solve(fixed={0: 0.0}, factorization=wrong)


def test_exact_matrix_factor_cache_has_no_tolerance_matching() -> None:
    """Share factors for identical operators while retaining independent loads and couplings."""
    cells = [
        LocalProblem([[2.0]], [[1.0]], [1.0], np.array([0])),
        LocalProblem([[2.0]], [[-3.0]], [4.0], np.array([1])),
        LocalProblem([[2.0 + 1e-14]], [[1.0]], [2.0], np.array([2])),
    ]
    with LocalFactorCache() as cache:
        results = [cache.condense(p) for p in cells]
        assert cache.size == 2
        assert cache.hits == 1
        for p, result in zip(cells, results, strict=True):
            assert_allclose(result.source, p.condense().source)
            assert_allclose(result.lifts, p.condense().lifts)
    assert cache.size == 0
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        cache.condense(cells[0])
    with pytest.raises(RuntimeError, match="closed"):
        cache.__enter__()
    assert len(condense_cached(cells)) == 3
    assert condense_cached([]) == ()
