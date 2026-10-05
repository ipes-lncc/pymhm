"""Selected native libraries are initialized before limits and numerical assembly."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_info

from pymhm.core import assembly, multiscale, system
from pymhm.core.assembly import HybridProblem, SolverConfig, assemble_hybrid
from pymhm.core.contracts import LocalProblem
from pymhm.core.equations import Equation, LocalEquations, compile_form
from pymhm.core.multiscale import MultiscaleProblem, assemble
from pymhm.core.system import HybridSystem
from pymhm.core.variational import GlobalForm
from pymhm.execution import cpu
from pymhm.execution.cpu import map_local
from pymhm.linalg import linear


@pytest.mark.parametrize(
    "solver,modules",
    [
        ("scipy", ()),
        ("cg", ()),
        ("minres", ()),
        ("gmres", ()),
        ("pypardiso", ("pypardiso",)),
        ("pypardiso-symmetric", ("pypardiso",)),
        ("pypardiso-symmetric-matching", ("pypardiso",)),
        ("petsc", ("petsc4py",)),
        ("petsc-symmetric", ("petsc4py",)),
        ("pyamg", ("pyamg",)),
        ("amgx", ("cupy", "pyamgx")),
        ("cupy", ("cupy", "cupyx.scipy.sparse", "cupyx.scipy.sparse.linalg")),
        ("cudss", ("cupy", "cupyx.scipy.sparse", "nvmath.sparse.advanced")),
    ],
)
def test_preparation_imports_only_selected_modules_without_native_sessions(
    monkeypatch: pytest.MonkeyPatch, solver: str, modules: tuple[str, ...]
) -> None:
    """Preparation selects libraries without constructing factors or accelerator sessions."""
    imported: list[str] = []

    def optional(name: str, installation: str) -> Any:
        """Expose the requested import while rejecting any native attribute access."""
        imported.append(name)
        assert solver in installation
        return object()

    monkeypatch.setattr(linear, "_optional", optional)
    linear.preload_solver_backend(solver)
    assert tuple(imported) == modules
    linear.preload_solver_backend(solver)
    assert tuple(imported) == modules + modules
    with pytest.raises(ValueError, match="Unsupported"):
        linear.preload_solver_backend("unavailable-name")


def test_preparation_preserves_native_import_diagnosis(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable selected dependency propagates without a numerical fallback."""

    def unavailable(name: str) -> Any:
        """Model a native shared-library loading failure."""
        raise OSError("native library absent")

    monkeypatch.setattr(linear, "import_module", unavailable)
    with pytest.raises(linear.SolverUnavailableError, match="native library absent"):
        linear.preload_solver_backend("pypardiso")


@pytest.mark.parametrize("solver", ["pyamg", "amgx"])
@pytest.mark.parametrize("api", ["preassembled", "factory", "hybrid", "dsl"])
def test_operator_capability_is_checked_before_unavailable_native_backend(
    monkeypatch: pytest.MonkeyPatch, solver: str, api: str
) -> None:
    """Native availability cannot mask unsupported retained-space operator contracts."""
    imports: list[str] = []

    def unavailable(module: str, installation: str) -> Any:
        """Model optional packages absent in a portable installation."""
        imports.append(module)
        raise linear.SolverUnavailableError("optional runtime unavailable")

    def provider(item: int) -> LocalProblem:
        """Declare an invertible reaction operator with a general retained mode."""
        return LocalProblem([[2.0]], [[1.0]], [1.0], [0], coarse_basis=[[1.0]])

    monkeypatch.setattr(linear, "_optional", unavailable)
    with pytest.raises(ValueError, match="general coarse_basis"):
        if api == "preassembled":
            HybridSystem([provider(0)], local_solver=solver)
        elif api == "factory":
            HybridSystem.from_local_factory(provider, [0], local_solver=solver)
        elif api == "hybrid":
            assemble_hybrid(
                HybridProblem(GlobalForm(1, (1,)), provider, [0]),
                solvers=SolverConfig(local_solver=solver),
            )
        else:

            def equations(item: int) -> LocalEquations:
                """Express the same inadmissible AMG retained space mathematically."""
                return LocalEquations(
                    a=[[2.0]],
                    b=[[1.0]],
                    c=[[-1.0]],
                    L=[1.0],
                    dofs=[0],
                    coarse_basis=[[1.0]],
                )

            assemble(
                MultiscaleProblem(Equation(0, 0), equations, [0], 1, (1,)),
                solvers=SolverConfig(local_solver=solver),
            )
    assert imports


@pytest.mark.parametrize("solver", ["pyamg", "amgx"])
def test_unused_and_invalid_local_operators_keep_their_native_free_contract(
    monkeypatch: pytest.MonkeyPatch, solver: str
) -> None:
    """Unused spaces and malformed input require no optional numerical session."""

    def unavailable(module: str, installation: str) -> Any:
        """Reject every attempted optional numerical backend import."""
        raise linear.SolverUnavailableError("optional runtime unavailable")

    def unused(item: int) -> LocalProblem:
        """Expose accidental input consumption for an empty iterable."""
        raise AssertionError("unused factory called")

    def invalid(item: int) -> LocalProblem:
        """Delegate malformed local data to its unchanged validation owner."""
        return LocalProblem([[np.nan]], [[1.0]], [1.0], [0])

    monkeypatch.setattr(linear, "_optional", unavailable)
    empty = LocalProblem(np.empty((0, 0)), np.empty((0, 0)), np.empty(0), np.empty(0, dtype=int))
    assert HybridSystem([empty], local_solver=solver).responses[0].source.size == 0
    with pytest.raises(ValueError, match="at least one local problem"):
        HybridSystem.from_local_factory(unused, [], local_solver=solver)
    with pytest.raises(ValueError, match="finite and square"):
        HybridSystem.from_local_factory(invalid, [0], local_solver=solver)
    valid = LocalProblem([[2.0]], [[1.0]], [1.0], [0])
    with pytest.raises(linear.SolverUnavailableError, match="optional runtime unavailable"):
        HybridSystem([valid], local_solver=solver)


def test_custom_preparation_failure_is_never_deferred() -> None:
    """Only named optional import failure is deferred, never an explicit user hook."""

    class Custom:
        """Declare a selected custom runtime that cannot prepare its own libraries."""

        def prepare_runtime(self) -> None:
            """Report an explicit callback preparation failure."""
            raise linear.SolverUnavailableError("custom runtime unavailable")

    with pytest.raises(linear.SolverUnavailableError, match="custom runtime unavailable"):
        system._prepare_solver(Custom())


class _PreparingCallable:
    """Load explicit resident state once and observe its worker-owned lifetime."""

    def __init__(self, directory: Path, *, fail: bool = False) -> None:
        """Declare plain picklable inputs without constructing native resources."""
        self.directory, self.fail = directory, fail
        self.preparations, self.calls, self.releases = 0, 0, 0

    def prepare_runtime(self) -> None:
        """Install resident libraries before any item enters its limit context."""
        self.preparations += 1
        if self.fail:
            raise ArithmeticError("prepare failed")

    def __call__(self, item: int) -> tuple[int, tuple[int, ...]]:
        """Require successful preparation while observing actual native thread limits."""
        assert self.preparations == 1 and self.releases == 0
        self.calls += 1
        np.dot(np.eye(2), np.ones(2))
        return item, tuple(int(pool["num_threads"]) for pool in threadpool_info())

    def close(self) -> None:
        """Persist exact worker preparation, call and release counts."""
        self.releases += 1
        (self.directory / f"{os.getpid()}.json").write_text(
            json.dumps(
                {
                    "preparations": self.preparations,
                    "calls": self.calls,
                    "releases": self.releases,
                }
            )
        )


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_preparation_is_once_per_worker_before_every_limited_item(
    tmp_path: Path, backend: Any, pipeline: bool
) -> None:
    """Resident preparation precedes actual BLAS work under serial/thread/spawn limits."""
    function = _PreparingCallable(tmp_path)
    values = map_local(function, range(3), backend=backend, workers=1, pipeline=pipeline)
    assert [item for item, _ in values] == [0, 1, 2]
    assert all(set(pools) <= {1} for _, pools in values)
    assert [json.loads(path.read_text()) for path in tmp_path.glob("*.json")] == [
        {"preparations": 1, "calls": 3, "releases": 1}
    ]


@pytest.mark.parametrize("backend", ["serial", "thread"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_new_libraries_are_visible_when_native_limit_context_is_constructed(
    monkeypatch: pytest.MonkeyPatch, backend: Any, pipeline: bool
) -> None:
    """The thread controller's library snapshot must include the selected runtime."""
    events: list[str] = []

    class ExplicitLibrary:
        """Model a library that is absent until a selected callback prepares it."""

        def prepare_runtime(self) -> None:
            """Mark library initialization before the controller snapshots pools."""
            events.append("library")

        def __call__(self, item: int) -> int:
            """Require that the controller includes this newly loaded library."""
            assert events[-1] == "limit"
            events.append("assembly")
            return item

    @contextmanager
    def limits(*, limits: int | None) -> Any:
        """Reject limit contexts constructed before their native library is loaded."""
        assert events == ["library"]
        events.append("limit")
        yield

    monkeypatch.setattr(cpu, "threadpool_limits", limits)
    assert map_local(ExplicitLibrary(), [1], backend=backend, workers=1, pipeline=pipeline) == [1]
    assert events == ["library", "limit", "assembly"]


@pytest.mark.parametrize("backend", ["serial", "thread"])
def test_failed_preparation_releases_resources_before_any_item(
    tmp_path: Path, backend: Any
) -> None:
    """Preparation failures release a resident callback without consuming input."""
    function = _PreparingCallable(tmp_path, fail=True)
    with pytest.raises(ArithmeticError, match="prepare failed"):
        map_local(function, [], backend=backend)
    assert json.loads((tmp_path / f"{os.getpid()}.json").read_text()) == {
        "preparations": 1,
        "calls": 0,
        "releases": 1,
    }


def test_failed_process_initialization_releases_and_clears_owner(tmp_path: Path) -> None:
    """A failed spawn initializer does not retain or finalize the callback twice."""
    function = _PreparingCallable(tmp_path, fail=True)
    with pytest.raises(ArithmeticError, match="prepare failed"):
        cpu._initialize_process(function, 1)
    assert cpu._PROCESS_FUNCTION is None and cpu._PROCESS_THREADS is None
    cpu._finalize_process_function()
    assert function.releases == 1 and function.calls == 0


def test_plain_callables_and_noncallable_hooks_keep_lazy_execution() -> None:
    """Existing portable callbacks require no preparation protocol."""

    class Plain:
        """Expose a noncallable optional attribute without selecting a runtime."""

        prepare_runtime = None

        def __call__(self, item: int) -> int:
            """Return an unchanged value."""
            return item

    assert map_local(Plain(), [2]) == [2]
    cpu._prepare_callable(lambda value: value)
    system._prepare_solver(lambda matrix, rhs: rhs)


@pytest.mark.parametrize("api", ["factory", "hybrid", "dsl"])
def test_generic_providers_forward_preparation_before_assembly(
    monkeypatch: pytest.MonkeyPatch, api: str
) -> None:
    """Every equation API loads its selected runtime before constructing operators."""
    events: list[str] = []

    class Provider:
        """Return a local equation after explicitly loading its own native libraries."""

        def prepare_runtime(self) -> None:
            """Observe library-only initialization without creating a matrix."""
            events.append("provider")

        def __call__(self, item: int) -> Any:
            """Verify preparation precedes construction of the local operator."""
            assert events[:2] == ["provider", "scipy"]
            events.append("assembly")
            if api == "dsl":
                return LocalEquations(a=[[2.0]], b=[[1.0]], c=[[-1.0]], L=[1.0], dofs=[0])
            return LocalProblem([[2.0]], [[1.0]], [1.0], [0])

    monkeypatch.setattr(system, "preload_solver_backend", events.append)
    if api == "factory":
        result = HybridSystem.from_local_factory(Provider(), [0])
    elif api == "hybrid":
        result = assemble_hybrid(HybridProblem(GlobalForm(1, (0,)), Provider(), [0]))
    else:
        result = assemble(MultiscaleProblem(Equation(0, 0), Provider(), [0], 1, (0,)))
    np.testing.assert_allclose(result.solve().fields, [[0.0]], atol=1e-14)
    assert events == ["provider", "scipy", "assembly"]


def test_external_solver_and_compiler_hooks_are_forwarded_deliberately() -> None:
    """Only explicitly supplied hooks run; they create no numerical solver session."""
    events: list[str] = []

    class Hook:
        """Record one explicitly declared resource initialization."""

        def prepare_runtime(self) -> None:
            """Load libraries without taking numerical arguments."""
            events.append("hook")

    custom = Hook()
    system._prepare_solver(custom)
    system._CondenseWorker(custom, "double").prepare_runtime()
    system._FactoryWorker(custom, custom, "double").prepare_runtime()
    assembly._CellWorker(
        custom, SolverConfig(), 1, (0,), np.array([1, 1]), None, False
    ).prepare_runtime()
    multiscale._Provider(custom, custom, SolverConfig()).prepare_runtime()
    assert events == ["hook"] * 7
    multiscale._Provider(lambda item: item, compile_form, SolverConfig()).prepare_runtime()


@pytest.mark.parametrize(
    "profile", ["pypardiso", "pypardiso-symmetric", "pypardiso-symmetric-matching"]
)
def test_pardiso_independent_factor_entries_are_serialized(
    monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    """Independent native engines never enter construction/factor/solve/free concurrently."""
    events: list[str] = []
    guard = threading.Lock()
    active = maximum = 0

    @contextmanager
    def native(phase: str) -> Any:
        """Model a non-reentrant host entry and detect overlapping native phases."""
        nonlocal active, maximum
        assert linear._PARDISO_HOST_LOCK.locked()
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.001)
        events.append(phase)
        try:
            yield
        finally:
            with guard:
                active -= 1

    class Engine:
        """Keep distinct factor data with an upstream-like non-reentrant host API."""

        def __init__(self, mtype: int = 11) -> None:
            """Validate the selected numerical profile without changing defaults."""
            with native("create"):
                assert mtype == (11 if profile == "pypardiso" else -2)

        def set_iparm(self, index: int, value: int) -> None:
            """Require serialization of explicit profile configuration too."""
            with native("configure"):
                assert profile.endswith("matching")

        def factorize(self, matrix: Any) -> None:
            """Retain only this engine's fixed diagonal operator."""
            with native("factor"):
                self.matrix = matrix.toarray()

        def solve(self, matrix: Any, rhs: np.ndarray) -> np.ndarray:
            """Solve only this engine's fixed operator while the host entry is held."""
            with native("solve"):
                return np.linalg.solve(self.matrix, rhs)

        def free_memory(self, *, everything: bool) -> None:
            """Release exactly this factor's own resource lifetime."""
            with native("free"):
                assert everything

    monkeypatch.setattr(linear, "_optional", lambda *args: SimpleNamespace(PyPardisoSolver=Engine))

    def solve(value: int) -> np.ndarray:
        """Exercise multiple complete independent native lifetimes in concurrent threads."""
        with linear.factorize(np.diag([2.0, 3.0]), solver=profile) as factor:
            return factor.solve(np.array([value, 2 * value]))

    with linear.factorize(np.diag([2.0, 3.0]), solver=profile) as factor:
        assert not linear._PARDISO_HOST_LOCK.locked()
        with linear.factorize(np.diag([4.0, 5.0]), solver=profile) as second:
            np.testing.assert_allclose(factor.solve([2.0, 3.0]), [1.0, 1.0])
            np.testing.assert_allclose(second.solve([4.0, 5.0]), [1.0, 1.0])
    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(solve, range(1, 5)))
    np.testing.assert_allclose(values, np.array([[0.5 * i, 2 * i / 3] for i in range(1, 5)]))
    assert maximum == 1 and active == 0
    assert events.count("create") == events.count("factor") == events.count("free") == 6


def test_pardiso_failed_factorization_releases_locked_native_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analysis failure releases the native factor under the entry lock and unlocks it."""
    releases: list[bool] = []

    class BrokenEngine:
        """Fail native analysis after allocating a resource that must be freed."""

        def factorize(self, matrix: Any) -> None:
            """Report an unchanged native factorization failure."""
            assert linear._PARDISO_HOST_LOCK.locked()
            raise RuntimeError("analysis failed")

        def free_memory(self, *, everything: bool) -> None:
            """Observe release while serialization is held."""
            assert linear._PARDISO_HOST_LOCK.locked()
            releases.append(everything)

    monkeypatch.setattr(
        linear, "_optional", lambda *args: SimpleNamespace(PyPardisoSolver=BrokenEngine)
    )
    with pytest.raises(RuntimeError, match="analysis failed"):
        linear.factorize([[2.0]], solver="pypardiso")
    assert releases == [True] and not linear._PARDISO_HOST_LOCK.locked()
