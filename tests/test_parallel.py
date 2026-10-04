"""Exercise real serial, thread and spawn-process execution, including errors."""

from __future__ import annotations

import multiprocessing
import os
from typing import Any

import numpy as np
import pytest

from pymhm.execution.cpu import _limited_call, map_local


def square(value: int) -> int:
    """Return a deterministic picklable workload."""
    return value * value


def fail(value: int) -> int:
    """Expose an exception from inside any executor."""
    raise RuntimeError(f"failed for {value}")


def identify(value: int) -> tuple[int, str]:
    """Identify the actual worker and its multiprocessing start method."""
    return os.getpid(), str(multiprocessing.get_start_method())


def solve_local(value: int) -> float:
    """Run a native solve within a worker to check ordered numerical parity."""
    from pymhm.linalg.linear import solve_linear

    return float(solve_linear(np.diag([2.0, 4.0]), [value, value])[0])


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_execution_order_empty_and_generator(backend: Any) -> None:
    assert map_local(square, (i for i in range(8)), backend=backend, workers=2) == [
        i * i for i in range(8)
    ]
    assert map_local(square, [], backend=backend) == []
    assert map_local(solve_local, [3, 1, 4], backend=backend, workers=2) == [1.5, 0.5, 2]


def test_process_uses_real_spawn_workers() -> None:
    results = map_local(identify, [0, 1, 2], backend="process", workers=2)
    assert all(pid != os.getpid() and method == "spawn" for pid, method in results)


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_worker_exception_propagates(backend: Any) -> None:
    with pytest.raises(RuntimeError, match="failed for 3"):
        map_local(fail, [3], backend=backend, workers=1)


@pytest.mark.parametrize("option", ["workers", "native_threads"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_limits(option: str, value: Any) -> None:
    with pytest.raises(ValueError, match=option):
        map_local(square, [1], **{option: value})


def test_invalid_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        map_local(square, [1], backend="other")


def test_no_native_limit_and_worker_wrapper() -> None:
    assert map_local(square, [2], native_threads=None) == [4]
    assert _limited_call((square, 3, 1)) == 9


class CountedFactory:
    """Record parent-side serialization, without sharing mutable state with workers."""

    def __init__(self) -> None:
        """Initialize the number of complete factory payloads sent to workers."""
        self.transfers = 0

    def __getstate__(self) -> dict[str, int]:
        """Count each factory transfer and serialize only an ordinary instance state."""
        self.transfers += 1
        return {"transfers": 0}

    def __call__(self, value: int) -> int:
        """Return ordered deterministic values after the spawned initialization."""
        return value + 1


def test_process_factory_is_transferred_once_per_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mesh-bearing factories must not be serialized again for every macrocell."""
    from pymhm.execution import cpu as parallel

    factory = CountedFactory()
    assert map_local(factory, range(9), backend="process", workers=1) == list(range(1, 10))
    assert factory.transfers == 1
    monkeypatch.setattr(parallel, "_PROCESS_FUNCTION", None)
    with pytest.raises(RuntimeError, match="not been initialized"):
        parallel._process_call(1)
