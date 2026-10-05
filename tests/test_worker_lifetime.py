"""Resident provider resources have an explicit ordered-executor lifetime."""

from __future__ import annotations

import json
import os
from contextlib import closing
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from pymhm.core.assembly import HybridProblem, SolverConfig, assemble_hybrid
from pymhm.core.contracts import LocalProblem
from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.multiscale import MultiscaleProblem, assemble
from pymhm.core.variational import GlobalForm
from pymhm.execution import cpu
from pymhm.execution.cpu import ExecutionConfig, iter_local, map_local

Backend = Literal["serial", "thread", "process"]


class _ResidentCallable:
    """Keep a reusable process-local counter and persist its final release."""

    def __init__(self, directory: Path, fail: bool = False) -> None:
        """Declare a release directory without allocating unpicklable resources."""
        self.directory, self.fail = directory, fail
        self.calls, self.releases = 0, 0

    def __call__(self, item: int) -> tuple[int, int]:
        """Reuse the resident state while exposing a controlled numerical failure."""
        self.calls += 1
        if self.fail and item == 2:
            raise ArithmeticError("local item two")
        return os.getpid(), item * item

    def close(self) -> None:
        """Record one resource release in the process that actually owned it."""
        self.releases += 1
        (self.directory / f"{os.getpid()}.json").write_text(
            json.dumps({"calls": self.calls, "releases": self.releases})
        )


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_resident_factory_reused_and_closed_at_exhaustion(
    tmp_path: Path, backend: Backend, pipeline: bool
) -> None:
    """Each spawned copy is released once; parent resources remain caller-owned."""
    factory = _ResidentCallable(tmp_path)
    values = map_local(factory, range(7), backend=backend, workers=1, pipeline=pipeline)
    assert [result for _, result in values] == [i * i for i in range(7)]
    releases = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert releases == [{"calls": 7, "releases": 1}]
    assert factory.calls == (0 if backend == "process" else 7)
    assert factory.releases == (0 if backend == "process" else 1)


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_resident_factory_closed_on_failure(
    tmp_path: Path, backend: Backend, pipeline: bool
) -> None:
    """An item failure releases the provider after pending native calls finish."""
    factory = _ResidentCallable(tmp_path, fail=True)
    with pytest.raises(ArithmeticError, match="local item two"):
        map_local(factory, range(7), backend=backend, workers=1, batch_size=1, pipeline=pipeline)
    releases = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert releases == [{"calls": 3, "releases": 1}]


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_resident_factory_closed_when_consumer_stops(tmp_path: Path, backend: Backend) -> None:
    """Explicit early generator closure preserves the input bound and cleans state."""
    factory = _ResidentCallable(tmp_path)
    with closing(iter_local(factory, range(7), backend=backend, workers=1, batch_size=1)) as values:
        assert next(values)[1] == 0
    releases = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert releases == [{"calls": 1, "releases": 1}]


def test_process_release_is_idempotent_and_close_errors_are_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear the process owner before cleanup so a repeated finalizer cannot repeat it."""

    class Broken:
        """Expose a failing explicit resource cleanup hook."""

        def __call__(self, item: int) -> int:
            """Return the supplied item without allocating another resource."""
            return item

        def close(self) -> None:
            """Raise an observable cleanup error."""
            raise RuntimeError("cleanup failed")

    factory = Broken()
    monkeypatch.setattr(cpu, "_PROCESS_FUNCTION", factory)
    monkeypatch.setattr(cpu, "_PROCESS_THREADS", 2)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        cpu._finalize_process_function()
    assert cpu._PROCESS_FUNCTION is None and cpu._PROCESS_THREADS is None
    cpu._finalize_process_function()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        map_local(factory, [])


class _LocalProvider(_ResidentCallable):
    """Provide a numerical local operator with worker-lifetime resources."""

    def __call__(self, cell: int) -> LocalProblem:
        """Construct distinct diagonal operators with explicitly oriented traces."""
        super().__call__(cell)
        return LocalProblem([[2.0 + cell]], [[1.0]], [1.0], [cell])


class _EquationProvider(_ResidentCallable):
    """Provide mathematical DSL blocks while owning persistent backend state."""

    def __call__(self, cell: int) -> LocalEquations:
        """Declare one local and one global equation without choosing a PDE."""
        super().__call__(cell)
        return LocalEquations(a=[[2.0 + cell]], L=[1.0], b=[[1.0]], c=[[-1.0]], dofs=[cell])


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
@pytest.mark.parametrize("dsl", [False, True])
def test_hybrid_and_dsl_forward_provider_cleanup(
    tmp_path: Path, backend: Backend, dsl: bool
) -> None:
    """Both generic equation APIs retain numeric results after native workspace release."""
    config = ExecutionConfig(backend, workers=1, batch_size=1, pipeline=True)
    if dsl:
        provider = _EquationProvider(tmp_path)
        system = assemble(
            MultiscaleProblem(Equation(0, 0), provider, range(2), 2, (0, 0)), execution=config
        )
    else:
        provider = _LocalProvider(tmp_path)
        system = assemble_hybrid(
            HybridProblem(GlobalForm(2, (0, 0)), provider, range(2)), execution=config
        )
    releases = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert releases == [{"calls": 2, "releases": 1}]
    solution = system.solve()
    np.testing.assert_allclose(solution.trace, [1.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(solution.fields, [[0.0], [0.0]], atol=1e-14)


class _ResidentSolver(_ResidentCallable):
    """Own an external solver workspace independently of the finite-element provider."""

    def __call__(self, matrix: object, rhs: np.ndarray) -> np.ndarray:
        """Solve original columns exactly while retaining only a process-local counter."""
        self.calls += 1
        return np.linalg.solve(matrix.toarray(), rhs)


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_external_solver_resident_resources_are_released(tmp_path: Path, backend: Backend) -> None:
    """Custom numerical providers can retain solver state without leaking native resources."""
    provider_path, solver_path = tmp_path / "provider", tmp_path / "solver"
    provider_path.mkdir()
    solver_path.mkdir()
    provider, solver = _LocalProvider(provider_path), _ResidentSolver(solver_path)
    system = assemble_hybrid(
        HybridProblem(GlobalForm(2, (0, 0)), provider, range(2)),
        execution=ExecutionConfig(backend, workers=1),
        solvers=SolverConfig(local_solver=solver),
    )
    np.testing.assert_allclose(system.solve().trace, [1.0, 1.0], atol=1e-14)
    for directory in (provider_path, solver_path):
        records = [json.loads(path.read_text()) for path in directory.glob("*.json")]
        assert records == [{"calls": 2, "releases": 1}]
