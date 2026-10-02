"""Ordered execution of independent local problems on portable CPU workers."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Literal, TypeVar, cast

from threadpoolctl import threadpool_limits

Input = TypeVar("Input")
Output = TypeVar("Output")
_PROCESS_FUNCTION: Callable[[Any], Any] | None = None
_PROCESS_THREADS: int | None = None


def _initialize_process(function: Callable[[Any], Any], native_threads: int | None) -> None:
    """Install one factory copy per spawned worker, independently of its item count."""
    global _PROCESS_FUNCTION, _PROCESS_THREADS
    _PROCESS_FUNCTION, _PROCESS_THREADS = function, native_threads


def _process_call(item: Input) -> Any:
    """Execute a small task payload using the factory already owned by this worker."""
    if _PROCESS_FUNCTION is None:
        raise RuntimeError("process factory has not been initialized")
    return _limited_call((_PROCESS_FUNCTION, item, _PROCESS_THREADS))


def _limited_call(task: tuple[Callable[[Input], Output], Input, int | None]) -> Output:
    """Run one spawned task under its process-local native thread limit."""
    function, item, native_threads = task
    with threadpool_limits(limits=native_threads):
        return function(item)


def map_local(
    function: Callable[[Input], Output],
    items: Iterable[Input],
    *,
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    native_threads: int | None = 1,
) -> list[Output]:
    """Apply independent local solves in input order, propagating exceptions.

    Threads avoid serialization and suit native kernels that release the GIL.
    Processes use ``spawn`` on every platform: functions and inputs must be
    picklable and callers must protect executable scripts with
    ``if __name__ == '__main__':``. No live PETSc, CUDA, or factorization object
    should be passed between processes; construct these inside the worker.
    The factory is serialized once per worker; individual tasks transfer only
    their item. Large read-only mesh metadata in a factory is therefore not
    repeatedly sent with every macrocell.

    ``native_threads`` limits supported BLAS/OpenMP pools (default one) to avoid
    worker/native-thread oversubscription. ``None`` preserves native settings.
    In thread mode that limit is process-wide during the call; do not overlap
    calls that request different limits. Serial execution also honors the limit.
    """
    if backend not in {"serial", "thread", "process"}:
        raise ValueError("backend must be 'serial', 'thread', or 'process'")
    for name, value in (("workers", workers), ("native_threads", native_threads)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer or None")
    if backend == "process":
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_process,
            initargs=(function, native_threads),
        ) as executor:
            return cast(list[Output], list(executor.map(_process_call, items)))
    with threadpool_limits(limits=native_threads):
        if backend == "serial":
            return [function(item) for item in items]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(function, items))
