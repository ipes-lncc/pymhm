"""Verify bounded streaming, ordered batches and real worker cleanup."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import threading
from collections.abc import Generator
from contextlib import closing
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
from threadpoolctl import threadpool_info

from pymhm.execution import cpu as parallel
from pymhm.execution.cpu import ExecutionConfig, iter_local, map_local

Backend = Literal["serial", "thread", "process"]


def _square(value: int) -> int:
    """Provide a spawn-compatible independent calculation."""
    return value * value


def _fail_two(value: int) -> int:
    """Fail in a later batch so earlier yielded results can be retained."""
    if value == 2:
        raise ArithmeticError("original item 2")
    return value


def _native_state(value: int) -> tuple[int, str | None, tuple[int, ...]]:
    """Observe actual native pools while executing a numerical workload."""
    np.dot(np.eye(2), np.full(2, value))
    return (
        os.getpid(),
        multiprocessing.get_start_method(allow_none=True),
        tuple(int(pool["num_threads"]) for pool in threadpool_info()),
    )


def _wide_array(value: int) -> np.ndarray:
    """Transfer extended coordinates without a dtype conversion."""
    return np.array([value, 1 + np.longdouble(2) ** -60], dtype=np.longdouble)


def _pool_sizes() -> tuple[tuple[str, int], ...]:
    """Snapshot loaded native pools without changing their limits."""
    return tuple((pool["prefix"], int(pool["num_threads"])) for pool in threadpool_info())


def _resources() -> tuple[set[int], set[int | None]]:
    """Record pre-existing children and threads to distinguish owned workers."""
    return (
        {child.pid for child in multiprocessing.active_children() if child.pid is not None},
        {thread.ident for thread in threading.enumerate()},
    )


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_consumption_is_lazy_and_strictly_bounded(backend: Backend) -> None:
    consumed: list[int] = []

    def items() -> Generator[int, None, None]:
        for item in range(7):
            consumed.append(item)
            yield item

    with closing(iter_local(_square, items(), backend=backend, workers=2, batch_size=3)) as results:
        assert consumed == []
        assert next(results) == 0
        bound = 1 if backend == "serial" else 3
        assert consumed == list(range(bound))
        if backend != "serial":
            assert next(results) == 1
            assert next(results) == 4
            assert consumed == [0, 1, 2]
            assert next(results) == 9
            assert consumed == list(range(6))
            assert list(results) == [16, 25, 36]
        else:
            assert next(results) == 1
            assert consumed == [0, 1]
            assert list(results) == [4, 9, 16, 25, 36]
        assert consumed == list(range(7))


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_default_batch_bound_is_effective_worker_count(backend: Backend) -> None:
    consumed: list[int] = []

    def items() -> Generator[int, None, None]:
        for item in range(8):
            consumed.append(item)
            yield item

    resources = _resources()
    with closing(iter_local(_square, items(), backend=backend, workers=2)) as results:
        assert next(results) == 0
        assert consumed == [0, 1]
    assert consumed == [0, 1]
    assert _resources() == resources


def test_serial_yields_before_solving_or_consuming_next_item() -> None:
    events: list[tuple[str, int]] = []

    def items() -> Generator[int, None, None]:
        for item in range(3):
            events.append(("consume", item))
            yield item

    def solve(item: int) -> int:
        events.append(("solve", item))
        return item

    with closing(iter_local(solve, items(), batch_size=20)) as results:
        assert next(results) == 0
        assert events == [("consume", 0), ("solve", 0)]
    assert events == [("consume", 0), ("solve", 0)]


@pytest.mark.parametrize("backend", ["serial", "thread"])
def test_closures_remain_supported(backend: Backend) -> None:
    offset = 7
    assert map_local(lambda item: item + offset, range(5), backend=backend, workers=2) == [
        7,
        8,
        9,
        10,
        11,
    ]


def test_thread_results_follow_inputs_despite_reversed_completion() -> None:
    second_finished = threading.Event()
    completed: list[int] = []

    def solve(item: int) -> int:
        if item == 0:
            assert second_finished.wait(timeout=5)
        completed.append(item)
        if item == 1:
            second_finished.set()
        return item

    assert map_local(solve, [0, 1], backend="thread", workers=2, batch_size=2) == [0, 1]
    assert completed == [1, 0]


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_later_failure_preserves_prior_batches_and_closes_workers(backend: Backend) -> None:
    consumed: list[int] = []

    def items() -> Generator[int, None, None]:
        for item in range(10):
            consumed.append(item)
            yield item

    resources, pools = _resources(), _pool_sizes()
    results = iter_local(_fail_two, items(), backend=backend, workers=1, batch_size=2)
    assert next(results) == 0
    assert next(results) == 1
    with pytest.raises(ArithmeticError, match="original item 2"):
        next(results)
    assert consumed == list(range(3 if backend == "serial" else 4))
    assert list(results) == []
    assert _resources() == resources
    assert _pool_sizes() == pools


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_input_failure_also_closes_executor(backend: Backend) -> None:
    def items() -> Generator[int, None, None]:
        yield 0
        raise LookupError("source failed")

    resources = _resources()
    results = iter_local(_square, items(), backend=backend, workers=1, batch_size=2)
    with pytest.raises(LookupError, match="source failed"):
        next(results)
    assert _resources() == resources


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_native_limit_restored_before_yield_and_after_early_close(backend: Backend) -> None:
    resources, pools = _resources(), _pool_sizes()
    with closing(
        iter_local(_native_state, range(20), backend=backend, workers=2, native_threads=1)
    ) as results:
        pid, method, native_counts = next(results)
        assert native_counts and all(count == 1 for count in native_counts)
        if backend == "process":
            assert pid != os.getpid() and method == "spawn"
        else:
            assert pid == os.getpid()
        assert _pool_sizes() == pools
    assert _pool_sizes() == pools
    assert _resources() == resources


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_closing_unstarted_iterator_consumes_nothing(backend: Backend) -> None:
    items = iter([3, 4])
    resources = _resources()
    results = iter_local(_square, items, backend=backend, workers=1)
    results.close()
    assert list(items) == [3, 4]
    assert _resources() == resources


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_runtime_passes_numerical_values_and_dtypes_unchanged(backend: Backend) -> None:
    actual = map_local(_wide_array, range(3), backend=backend, workers=1, batch_size=2)
    for item, value in enumerate(actual):
        assert value.dtype == np.dtype(np.longdouble)
        np.testing.assert_array_equal(value, _wide_array(item))


@pytest.mark.parametrize(
    "name,value",
    [
        ("workers", 0),
        ("workers", -1),
        ("workers", True),
        ("workers", False),
        ("workers", 1.5),
        ("workers", "2"),
        ("native_threads", 0),
        ("batch_size", 0),
    ],
)
def test_invalid_configuration_is_immediate(name: str, value: Any) -> None:
    """Shared limit validation rejects each type and field before consuming inputs."""
    items = iter([1, 2])
    with pytest.raises(ValueError, match=name):
        iter_local(_square, items, **{name: value})
    with pytest.raises(ValueError, match=name):
        ExecutionConfig(**{name: value})
    with pytest.raises(ValueError, match=name):
        map_local(_square, items, **{name: value})
    assert list(items) == [1, 2]


def test_invalid_backend_is_immediate_and_config_is_frozen_picklable() -> None:
    with pytest.raises(ValueError, match="backend"):
        iter_local(_square, [], backend="invalid")  # type: ignore[arg-type]
    config = ExecutionConfig("process", workers=2, native_threads=None, batch_size=3)
    assert pickle.loads(pickle.dumps(config)) == config
    with pytest.raises(FrozenInstanceError):
        config.workers = 4  # type: ignore[misc]
    assert config.effective_workers == 2
    assert config.effective_batch_size == 3
    assert ExecutionConfig("serial", workers=2, batch_size=8).effective_workers == 1
    assert ExecutionConfig("serial", batch_size=8).effective_batch_size == 1


@pytest.mark.parametrize(
    ("backend", "platform", "cpu", "expected"),
    [
        ("thread", "posix", 4, 8),
        ("thread", "posix", 80, 32),
        ("process", "posix", 80, 80),
        ("process", "nt", 80, 61),
    ],
)
def test_worker_defaults_follow_executor_platform_conventions(
    monkeypatch: pytest.MonkeyPatch, backend: Backend, platform: str, cpu: int, expected: int
) -> None:
    monkeypatch.setattr(
        parallel,
        "os",
        SimpleNamespace(cpu_count=lambda: 99, process_cpu_count=lambda: cpu, name=platform),
    )
    config = ExecutionConfig(backend)
    assert config.effective_workers == config.effective_batch_size == expected


def test_worker_default_handles_absent_process_cpu_count_and_unknown_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallel, "os", SimpleNamespace(cpu_count=lambda: None, name="posix"))
    assert ExecutionConfig("process").effective_workers == 1
    assert ExecutionConfig("thread").effective_workers == 5


def test_explicit_windows_process_worker_limit_is_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallel, "os", SimpleNamespace(name="nt"))
    with pytest.raises(ValueError, match="workers.*61"):
        ExecutionConfig("process", workers=62)
    assert ExecutionConfig("process", workers=61).effective_workers == 61
    assert ExecutionConfig("thread", workers=62).effective_workers == 62


def test_installed_process_dispatch_restores_actual_native_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallel, "_PROCESS_FUNCTION", None)
    monkeypatch.setattr(parallel, "_PROCESS_THREADS", None)
    pools = _pool_sizes()
    parallel._initialize_process(_native_state, 1)
    pid, _, counts = parallel._process_call(2)
    assert pid == os.getpid() and counts and all(count == 1 for count in counts)
    assert _pool_sizes() == pools


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_none_native_limit_and_empty_batches(backend: Backend) -> None:
    assert list(iter_local(_square, [], backend=backend, workers=1, native_threads=None)) == []
    assert map_local(_square, [2], backend=backend, workers=1, native_threads=None) == [4]
