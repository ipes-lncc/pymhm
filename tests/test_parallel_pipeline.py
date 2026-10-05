"""Bounded ordered pipelines, native limits and physical worker-reduction parity."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import threading
from collections.abc import Generator
from contextlib import closing
from dataclasses import FrozenInstanceError
from itertools import count
from typing import Any, Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_info, threadpool_limits

from pymhm.core.assembly import HybridProblem, assemble_hybrid
from pymhm.core.contracts import LocalAssembly, LocalProblem, LocalResponse
from pymhm.core.contributions import local_global_contribution
from pymhm.core.system import HybridSystem
from pymhm.core.variational import GlobalForm
from pymhm.execution import cpu as parallel
from pymhm.execution.cpu import ExecutionConfig, iter_local, map_local

Backend = Literal["serial", "thread", "process"]


def _square(value: int) -> int:
    """Return a small, importable spawn workload."""
    return value * value


def _fail_one(value: int) -> int:
    """Expose the distinction between atomic batches and ordered item delivery."""
    if value == 1:
        raise ArithmeticError("original item 1")
    return value


def _native_state(value: int) -> tuple[int, str | None, tuple[int, ...]]:
    """Observe actual worker pools after performing a native arithmetic operation."""
    np.dot(np.eye(2), np.full(2, value))
    return (
        os.getpid(),
        multiprocessing.get_start_method(allow_none=True),
        tuple(int(pool["num_threads"]) for pool in threadpool_info()),
    )


def _wide_array(value: int) -> np.ndarray:
    """Return extended coordinates that must survive worker transport unchanged."""
    return np.array([value, 1 + np.longdouble(2) ** -60], dtype=np.longdouble)


def _resources() -> tuple[set[int | None], set[int | None]]:
    """Snapshot live children and threads, including pre-existing caller resources."""
    return (
        {child.pid for child in multiprocessing.active_children()},
        {thread.ident for thread in threading.enumerate()},
    )


def _pool_sizes() -> tuple[tuple[str, int], ...]:
    """Observe process-wide native thread limits without changing them."""
    return tuple((pool["prefix"], int(pool["num_threads"])) for pool in threadpool_info())


def _neumann_cell(cell: int) -> LocalAssembly:
    """Declare two oriented unit diffusion cells with physical half-volume moments."""
    problem = LocalProblem(
        [[1.0, -1], [-1, 1]],
        np.eye(2) if cell == 0 else np.diag([-1.0, 1]),
        [0.0, 0.0],
        [cell, cell + 1],
        kernel=[[1.0], [1.0]],
        constraints=[[0.5], [0.5]],
    )
    return LocalAssembly(
        problem, {"cell": cell, "provider_owner": (os.getpid(), threading.get_ident())}
    )


def _owned_contribution(
    response: LocalResponse, metadata: dict[str, Any], coarse_dofs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Record declared callback ownership using only this response's owned metadata."""
    metadata["contribution_owner"] = (os.getpid(), threading.get_ident())
    return local_global_contribution(response, coarse_dofs)


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
@pytest.mark.parametrize("bound", [1, 3])
def test_pipeline_consumption_is_lazy_ordered_and_bounded(backend: Backend, bound: int) -> None:
    """A rolling window never consumes more than its unyielded-input bound."""
    consumed: list[int] = []

    def items() -> Generator[int, None, None]:
        """Expose each input request to the coordinator."""
        for item in range(7):
            consumed.append(item)
            yield item

    with closing(
        iter_local(_square, items(), backend=backend, workers=2, batch_size=bound, pipeline=True)
    ) as results:
        assert consumed == []
        effective = 1 if backend == "serial" else bound
        for yielded in range(7):
            assert next(results) == yielded * yielded
            assert len(consumed) == min(7, effective + yielded)
            assert len(consumed) - (yielded + 1) < effective
        assert list(results) == []
    assert consumed == list(range(7))


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_infinite_input_is_bounded_and_caller_owned_on_close(backend: Backend) -> None:
    """Closing after one result joins workers without exhausting or closing the source."""
    consumed: list[int] = []

    def items() -> Generator[int, None, None]:
        """Keep yielding until the test's caller explicitly closes this generator."""
        for item in count():
            consumed.append(item)
            yield item

    source = items()
    resources = _resources()
    with closing(
        iter_local(_square, source, backend=backend, workers=2, batch_size=4, pipeline=True)
    ) as results:
        assert next(results) == 0
    bound = 1 if backend == "serial" else 4
    assert consumed == list(range(bound))
    assert next(source) == bound
    source.close()
    assert _resources() == resources


def test_pipeline_delivers_before_the_rest_of_its_window_finishes() -> None:
    """An Event-blocked second job cannot impose a whole-window delivery barrier."""
    second_started, release_second = threading.Event(), threading.Event()

    def solve(item: int) -> int:
        """Require the consumer of result zero to release the later worker."""
        if item == 0:
            assert second_started.wait(timeout=5)
        else:
            second_started.set()
            assert release_second.wait(timeout=5)
        return item

    resources = _resources()
    try:
        with closing(
            iter_local(solve, [0, 1], backend="thread", workers=2, batch_size=2, pipeline=True)
        ) as results:
            assert next(results) == 0
            assert second_started.is_set() and not release_second.is_set()
            release_second.set()
            assert list(results) == [1]
    finally:
        release_second.set()
    assert _resources() == resources


@pytest.mark.parametrize("backend", ["thread", "process"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_atomic_batch_default_and_pipeline_failure_delivery(
    backend: Backend, pipeline: bool
) -> None:
    """The opt-in changes error delivery while leaving the default batch contract intact."""
    resources, pools = _resources(), _pool_sizes()
    results = iter_local(
        _fail_one, [0, 1, 2], backend=backend, workers=2, batch_size=2, pipeline=pipeline
    )
    if pipeline:
        assert next(results) == 0
    with pytest.raises(ArithmeticError, match="original item 1"):
        next(results)
    assert list(results) == []
    assert _resources() == resources
    assert _pool_sizes() == pools


@pytest.mark.parametrize("backend", ["thread", "process"])
@pytest.mark.parametrize("failure_after", [1, 2])
def test_source_failure_during_prefill_or_refill_closes_workers(
    backend: Backend, failure_after: int
) -> None:
    """Source exceptions stop bounded scheduling and restore worker/native resources."""

    def items() -> Generator[int, None, None]:
        """Fail either during the initial window or after a successful handoff."""
        yield from range(failure_after)
        raise LookupError("source failure")

    resources, pools = _resources(), _pool_sizes()
    results = iter_local(_square, items(), backend=backend, workers=2, batch_size=2, pipeline=True)
    if failure_after == 2:
        assert next(results) == 0
    with pytest.raises(LookupError, match="source failure"):
        next(results)
    assert _resources() == resources
    assert _pool_sizes() == pools


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_pipeline_real_native_limits_and_resource_cleanup(backend: Backend) -> None:
    """Spawn is real; process limits stay local while thread limits cover its consumer."""
    resources, pools = _resources(), _pool_sizes()
    with closing(
        iter_local(
            _native_state,
            range(8),
            backend=backend,
            workers=2,
            native_threads=1,
            batch_size=2,
            pipeline=True,
        )
    ) as results:
        pid, method, counts = next(results)
        assert counts and all(value == 1 for value in counts)
        if backend == "thread":
            assert _pool_sizes() and all(value == 1 for _, value in _pool_sizes())
        else:
            assert _pool_sizes() == pools
        if backend == "process":
            assert pid != os.getpid() and method == "spawn"
        else:
            assert pid == os.getpid()
    assert _resources() == resources
    assert _pool_sizes() == pools


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_unstarted_empty_and_inherited_limits_preserve_dtype(backend: Backend) -> None:
    """Empty/unstarted generators consume nothing and real values retain their dtype."""
    resources, pools = _resources(), _pool_sizes()
    source = iter([1, 2])
    iterator = iter_local(_square, source, backend=backend, workers=2, pipeline=True)
    iterator.close()
    assert list(source) == [1, 2]
    assert map_local(_square, [], backend=backend, workers=2, pipeline=True) == []
    for value, actual in enumerate(
        map_local(
            _wide_array, range(3), backend=backend, workers=2, native_threads=None, pipeline=True
        )
    ):
        assert actual.dtype == np.dtype(np.longdouble)
        assert_array_equal(actual, _wide_array(value))
    assert _resources() == resources
    assert _pool_sizes() == pools


@pytest.mark.parametrize("value", [0, 1, None, "true", 1.0, np.bool_(True)])
def test_pipeline_requires_an_actual_bool_before_consuming_inputs(value: Any) -> None:
    """Reject ambiguous opt-in values immediately, independently of backend selection."""
    source = iter([1, 2])
    with pytest.raises(ValueError, match="pipeline must be a bool"):
        iter_local(_square, source, pipeline=value)
    with pytest.raises(ValueError, match="pipeline must be a bool"):
        ExecutionConfig(pipeline=value)
    assert list(source) == [1, 2]


def test_pipeline_config_is_frozen_picklable_and_default_is_unchanged() -> None:
    """Preserve four-position construction and serialize the explicit fifth policy."""
    assert ExecutionConfig("thread", 2, 1, 3).pipeline is False
    config = ExecutionConfig("process", 2, 1, 3, True)
    assert pickle.loads(pickle.dumps(config)) == config
    with pytest.raises(FrozenInstanceError):
        config.pipeline = False  # type: ignore[misc]


def test_executor_construction_failure_restores_pipeline_thread_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process-wide scope restores even when no executor could be constructed."""

    def fail(**options: Any) -> Any:
        """Reject executor creation before any item is requested."""
        raise RuntimeError("executor construction failed")

    pools, resources = _pool_sizes(), _resources()
    monkeypatch.setattr(parallel, "ThreadPoolExecutor", fail)
    with pytest.raises(RuntimeError, match="executor construction failed"):
        next(iter_local(_square, [0], backend="thread", pipeline=True))
    assert _pool_sizes() == pools
    assert _resources() == resources


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
@pytest.mark.parametrize("bound", [1, 3])
@pytest.mark.parametrize("placement", ["coordinator", "worker"])
def test_pipeline_physical_operator_fields_and_callback_ownership(
    backend: Backend, bound: int, placement: Literal["coordinator", "worker"]
) -> None:
    """Ordered worker blocks preserve diffusion, boundary data and declared callback owners."""
    coordinator = (os.getpid(), threading.get_ident())
    form = GlobalForm(3, (1, 1), boundary_load=[0.0, 0.0, 2.0])
    definition = HybridProblem(
        form, _neumann_cell, range(2), _owned_contribution, contribution_execution=placement
    )
    expected = HybridSystem(
        [_neumann_cell(cell).problem for cell in range(2)], boundary_load=form.boundary_load
    )
    with threadpool_limits(limits=1):
        actual = assemble_hybrid(
            definition,
            execution=ExecutionConfig(backend=backend, workers=2, batch_size=bound, pipeline=True),
        )
        solution, reference = actual.solve(), expected.solve()
    for name in ("matrix", "rhs", "load_scale", "kernel_offsets"):
        a, b = getattr(actual, name), getattr(expected, name)
        assert_array_equal(
            a.toarray() if name == "matrix" else a, b.toarray() if name == "matrix" else b
        )
    assert_array_equal(solution.trace, reference.trace)
    for cell, (field, baseline) in enumerate(zip(solution.fields, reference.fields, strict=True)):
        assert_array_equal(field, baseline)
        assert_allclose(field, [cell, cell + 1], atol=1e-14)
        local = actual.responses[cell].problem
        assert_allclose(
            local.matrix @ field + local.coupling @ solution.trace[local.trace_dofs],
            local.load,
            atol=1e-14,
        )
    for cell, metadata in enumerate(actual.local_metadata):
        assert metadata["cell"] == cell
        assert metadata["contribution_owner"] == (
            coordinator if placement == "coordinator" else metadata["provider_owner"]
        )
        if backend != "serial":
            assert metadata["provider_owner"] != coordinator


def test_coordinator_callback_can_release_a_pending_local_worker() -> None:
    """The declared consumer runs in input order before later work has completed."""
    blocked, release = threading.Event(), threading.Event()
    callback_events: list[tuple[int, int]] = []
    coordinator = threading.get_ident()

    def provider(cell: int) -> LocalAssembly:
        """Make delivery of the first cell necessary for the second to complete."""
        if cell == 0:
            assert blocked.wait(timeout=5)
        else:
            blocked.set()
            assert release.wait(timeout=5)
        return _neumann_cell(cell)

    def contribution(response: LocalResponse, record: dict[str, Any], slots: np.ndarray) -> Any:
        """Release the other worker from an explicitly coordinator-owned callback."""
        callback_events.append((record["cell"], threading.get_ident()))
        if record["cell"] == 0:
            assert blocked.is_set() and not release.is_set()
            release.set()
        return local_global_contribution(response, slots)

    form = GlobalForm(3, (1, 1), boundary_load=[0.0, 0.0, 2.0])
    resources, pools = _resources(), _pool_sizes()
    try:
        actual = assemble_hybrid(
            HybridProblem(form, provider, range(2), contribution),
            execution=ExecutionConfig("thread", 2, 1, 2, True),
        )
    finally:
        release.set()
    assert callback_events == [(0, coordinator), (1, coordinator)]
    assert_allclose(actual.solve().fields, [[0.0, 1.0], [1.0, 2.0]], atol=1e-14)
    assert _resources() == resources
    assert _pool_sizes() == pools
