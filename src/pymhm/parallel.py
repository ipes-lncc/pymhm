"""Ordered execution of independent local problems on portable CPU workers."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from typing import Any, Literal, TypeVar, cast

from threadpoolctl import threadpool_limits

Input = TypeVar("Input")
Output = TypeVar("Output")
_PROCESS_FUNCTION: Callable[[Any], Any] | None = None
_PROCESS_THREADS: int | None = None


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime policy for ordered independent local computations.

    This configuration changes scheduling and supported BLAS/OpenMP thread
    counts only; it does not select numerical dtypes, operators or tolerances.
    ``None`` workers use the executor's platform-appropriate CPU default.
    ``None`` batch size uses that worker count in parallel and one in serial.
    An explicit batch size bounds the number of items submitted together;
    serial execution always consumes and solves one item at a time.
    ``native_threads=None`` preserves the existing native thread settings.
    """

    backend: Literal["serial", "thread", "process"] = "serial"
    workers: int | None = None
    native_threads: int | None = 1
    batch_size: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid scheduling limits before any input is consumed."""
        if self.backend not in ("serial", "thread", "process"):
            raise ValueError("backend must be 'serial', 'thread', or 'process'")
        for name in ("workers", "native_threads", "batch_size"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if (
            self.backend == "process"
            and os.name == "nt"
            and self.workers is not None
            and self.workers > 61
        ):
            raise ValueError("workers must be at most 61 for Windows process execution")

    @property
    def effective_workers(self) -> int:
        """Return the actual pool size, including the Windows process limit."""
        if self.backend == "serial":
            return 1
        if self.workers is not None:
            return self.workers
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        if self.backend == "thread":
            return min(32, cpu_count + 4)
        return min(61, cpu_count) if os.name == "nt" else cpu_count

    @property
    def effective_batch_size(self) -> int:
        """Return the input bound; serial execution always has bound one."""
        if self.backend == "serial":
            return 1
        return self.batch_size if self.batch_size is not None else self.effective_workers


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


def _iter_local(
    function: Callable[[Input], Output],
    items: Iterable[Input],
    config: ExecutionConfig,
) -> Generator[Output, None, None]:
    """Own the executor until exhaustion or explicit generator closure."""
    if config.backend == "serial":
        for item in items:
            yield _limited_call((function, item, config.native_threads))
        return
    executor: Executor
    if config.backend == "process":
        executor = ProcessPoolExecutor(
            max_workers=config.effective_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_process,
            initargs=(function, config.native_threads),
        )
        call = cast(Callable[[Input], Output], _process_call)
    else:
        executor = ThreadPoolExecutor(max_workers=config.effective_workers)
        call = function
    try:
        iterator = iter(items)
        while batch := list(islice(iterator, config.effective_batch_size)):
            futures: list[Future[Output]] = []
            limits = (
                threadpool_limits(limits=config.native_threads)
                if config.backend == "thread"
                else nullcontext()
            )
            with limits:
                try:
                    for item in batch:
                        futures.append(executor.submit(call, item))
                    results = [future.result() for future in futures]
                except BaseException:
                    for future in futures:
                        future.cancel()
                    wait(futures)
                    raise
            yield from results
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def iter_local(
    function: Callable[[Input], Output],
    items: Iterable[Input],
    *,
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    native_threads: int | None = 1,
    batch_size: int | None = None,
) -> Generator[Output, None, None]:
    """Yield independent local results in input order with bounded consumption.

    Validation is immediate; computation starts on the first iteration.
    Serial execution solves and yields one item before consuming the next.
    Parallel execution consumes at most one batch, completes it, and yields its
    results before consuming another batch. ``batch_size=None`` uses the
    effective worker count. Parallel failures propagate before that batch is
    yielded; already yielded earlier batches remain available to the caller.
    Call ``close()`` or use ``contextlib.closing`` when ending iteration early
    to join workers and release executor resources. The input iterable remains
    caller-owned. Running tasks finish before closure returns; pending tasks
    are cancelled after a failure.

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
    In thread mode that limit is process-wide while a batch executes; do not
    overlap calls that request different limits. The limit is restored before
    yielding results. Serial execution limits each individual call.
    """
    config = ExecutionConfig(backend, workers, native_threads, batch_size)
    return _iter_local(function, items, config)


def map_local(
    function: Callable[[Input], Output],
    items: Iterable[Input],
    *,
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    native_threads: int | None = 1,
    batch_size: int | None = None,
) -> list[Output]:
    """Collect :func:`iter_local` in order, preserving its execution contract.

    The returned list holds all results; only input consumption and submitted
    tasks are bounded by ``batch_size``. Use :func:`iter_local` to release
    results incrementally. Numerical values and dtypes are passed unchanged.
    """
    return list(
        iter_local(
            function,
            items,
            backend=backend,
            workers=workers,
            native_threads=native_threads,
            batch_size=batch_size,
        )
    )
