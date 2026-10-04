"""Verify that factory assembly, condensation and portable metadata remain in one worker."""

from __future__ import annotations

import multiprocessing
import os
import threading
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import HybridSystem, LocalAssembly, LocalProblem
from pymhm.linalg.linear import LinearSolveError


def make_problem(item: tuple[int, float]) -> LocalProblem:
    """Build a distinct coercive local problem with one globally numbered trace."""
    index, value = item
    return LocalProblem(
        [[4 + value, 1], [1, 2 + value]],
        [[1], [0.5]],
        [value + 1, 2],
        np.array([index]),
    )


def factory(item: tuple[int, float]) -> LocalAssembly:
    """Return numerical assembly and observable metadata from the actual execution worker."""
    return LocalAssembly(
        make_problem(item),
        {
            "index": item[0],
            "value": item[1],
            "pid": os.getpid(),
            "thread": threading.get_ident(),
            "start_method": multiprocessing.get_start_method(),
            "coordinates": np.array([[0.0, item[1]], [1.0, item[1]]]),
        },
    )


def fail_factory(item: int) -> LocalProblem:
    """Expose an assembly failure without creating any local matrix."""
    raise RuntimeError(f"assembly failed for cell {item}")


def invalid_factory(item: int) -> Any:
    """Return invalid user data to exercise the explicit factory contract."""
    return item


def singular_factory(item: int) -> LocalProblem:
    """Construct a singular local operator whose numerical failure must propagate."""
    return LocalProblem([[0.0]], [[1.0]], [1.0], np.array([item]))


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
@pytest.mark.parametrize("with_metadata", [False, True])
def test_factory_matches_reference_and_preserves_worker_order(
    backend: Any, with_metadata: bool
) -> None:
    """Compare fields and skeleton equations while proving where each factory executed."""
    items = [(2, 4.0), (0, 1.0), (1, 3.0)]
    boundary = [0.3, -0.2, 1.0]
    reference = HybridSystem([make_problem(item) for item in items], boundary_load=boundary)
    actual = HybridSystem.from_local_factory(
        factory if with_metadata else make_problem,
        (item for item in items),
        boundary_load=boundary,
        backend=backend,
        workers=2,
    )
    assert_allclose(actual.matrix.toarray(), reference.matrix.toarray())
    assert_allclose(actual.rhs, reference.rhs)
    result, expected = actual.solve(), reference.solve()
    assert_allclose(result.trace, expected.trace)
    assert_allclose(result.fields, expected.fields)
    assert reference.local_metadata == (None, None, None)
    if with_metadata:
        for item, metadata in zip(items, actual.local_metadata, strict=True):
            assert (metadata["index"], metadata["value"]) == item
            assert_allclose(metadata["coordinates"], [[0, item[1]], [1, item[1]]])
            if backend == "process":
                assert metadata["pid"] != os.getpid()
                assert metadata["start_method"] == "spawn"
            elif backend == "thread":
                assert metadata["pid"] == os.getpid()
                assert metadata["thread"] != threading.get_ident()
            else:
                assert metadata["pid"] == os.getpid()
    else:
        assert actual.local_metadata == (None, None, None)


@pytest.mark.parametrize("backend", ["serial", "thread"])
def test_factory_and_condensation_execute_once(
    backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A closure factory builds once and subsequent global solves reuse its harmonic lifts."""
    assembled, condensed = [], []
    original = LocalProblem.condense

    def build(item: tuple[int, float]) -> LocalAssembly:
        """Record closure execution without transporting the closure to a process."""
        assembled.append(item)
        return factory(item)

    def record(self: LocalProblem, solver: str = "scipy", **kwargs: Any) -> Any:
        """Record each actual local numerical factorization request."""
        condensed.append(int(self.trace_dofs[0]))
        return original(self, solver, **kwargs)

    monkeypatch.setattr(LocalProblem, "condense", record)
    items = [(0, 1.0), (1, 3.0)]
    system = HybridSystem.from_local_factory(build, items, backend=backend, workers=2)
    first, second = system.solve(), system.solve()
    assert_allclose(first.fields, second.fields)
    assert sorted(assembled) == items
    assert sorted(condensed) == [0, 1]


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_factory_and_numerical_exceptions_propagate(backend: Any) -> None:
    """Assembly errors and local singularities abort the call without backend fallback."""
    with pytest.raises(RuntimeError, match="assembly failed for cell 3"):
        HybridSystem.from_local_factory(fail_factory, [3], backend=backend, workers=1)
    with pytest.raises(LinearSolveError, match="zero row"):
        HybridSystem.from_local_factory(singular_factory, [0], backend=backend, workers=1)


def test_factory_payload_and_empty_input_validation() -> None:
    """Reject malformed local payloads and an empty generator before global assembly."""
    with pytest.raises(TypeError, match="LocalAssembly.problem"):
        LocalAssembly(None)
    with pytest.raises(TypeError, match="local factory must return"):
        HybridSystem.from_local_factory(invalid_factory, [1])
    with pytest.raises(ValueError, match="at least one local problem"):
        HybridSystem.from_local_factory(make_problem, iter(()))


def test_factory_keeps_subclass_type() -> None:
    """Class construction retains custom hybrid-system subclasses."""

    class ApplicationSystem(HybridSystem):
        """Represent a user extension without an altered local-condensation policy."""

    result = ApplicationSystem.from_local_factory(make_problem, [(0, 1.0)])
    assert isinstance(result, ApplicationSystem)
    assert result.local_metadata == (None,)


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_full_darcy_factory_reconstructs_physical_fields(
    backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the benchmark's complete P1 factory path and verify the exact affine Darcy field."""
    import importlib
    from pathlib import Path

    from threadpoolctl import threadpool_limits

    from pymhm import TriangleMesh
    from pymhm._legacy.models.darcy.primal import solve_darcy

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    benchmark = importlib.import_module("large_local")
    mesh = TriangleMesh.unit_square(2)
    with threadpool_limits(limits=1):
        expected = solve_darcy(mesh, local_refinement=3, dirichlet=benchmark.exact_pressure)
        result, durations, prepared = benchmark.profiled_solve(
            mesh, 3, backend, 2, factory_path=True
        )
        errors = benchmark.verify_solution(result, expected)
    assert max(errors.values()) < 1e-11
    assert len(prepared) == len(mesh.cells)
    assert durations["local_assembly_and_condensation"] > 0
    assert_allclose(
        sum(value for key, value in durations.items() if key != "total"), durations["total"]
    )
