"""Batched CUDA contracts with explicit CPU simulation and separate native checks."""

import ctypes
import threading
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg, sparse

from pymhm.core.contracts import LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.execution import cuda as gpu
from pymhm.linalg.linear import LinearSolveError


class HostPointerArray(np.ndarray):
    """Expose host pointers for an explicitly simulated cuBLAS ABI contract."""

    @property
    def data(self) -> Any:
        """Expose a pointer shape matching CuPy without pretending to use CUDA."""
        return SimpleNamespace(ptr=self.ctypes.data)

    def __array_finalize__(self, array: Any) -> None:
        """Preserve the explicitly simulated device tag through views and arithmetic."""
        self._device_id = getattr(array, "_device_id", 0)

    @property
    def device(self) -> Any:
        """Expose a simulated CUDA-device identity, distinct from ordinary host arrays."""
        return SimpleNamespace(id=self._device_id)


class SimulatedDevice:
    """Enter a thread-local device and restore its predecessor on every exit."""

    def __init__(self, backend: Any, device_id: int) -> None:
        """Retain the explicit simulator and requested device identity."""
        self.backend, self.device_id = backend, device_id

    def __enter__(self) -> Any:
        """Record entry on this worker without sharing its active device."""
        self.previous = self.backend.active_device
        self.backend.active_device = self.device_id
        self.backend.contexts.append(("enter", self.device_id, threading.get_ident()))
        return self

    def __exit__(self, *args: Any) -> None:
        """Restore the worker's active device after normal and exceptional operations."""
        self.backend.contexts.append(("exit", self.device_id, threading.get_ident()))
        self.backend.active_device = self.previous


def pointer_array(pointer: int, length: int, ctype: Any) -> np.ndarray:
    """View test-owned host storage passed through the simulated native ABI."""
    return np.ctypeslib.as_array((ctype * length).from_address(int(pointer)))


class SimulatedCuPy:
    """Execute array algebra on NumPy and LU on SciPy for API contract coverage."""

    def __init__(self) -> None:
        """Create isolated counters and failure controls for each test."""
        self.factor_calls = self.solve_calls = 0
        self.argument_error = False
        self.corrupt = 0
        self._thread_state = threading.local()
        self.contexts: list[tuple[str, int, int]] = []
        self.synchronized: list[int] = []
        self.handles: list[int] = []
        self.cuda = SimpleNamespace(
            cublas=SimpleNamespace(
                dgetrfBatched=self.getrf, dgetrsBatched=self.getrs, CUBLAS_OP_N=0
            ),
            device=SimpleNamespace(get_cublas_handle=self.current_handle),
            runtime=SimpleNamespace(getDevice=lambda: self.active_device, getDeviceCount=lambda: 2),
            Device=lambda device_id: SimulatedDevice(self, device_id),
            get_current_stream=lambda: SimpleNamespace(synchronize=self.synchronize),
        )

    @property
    def active_device(self) -> int:
        """Return the current simulated device for this particular worker."""
        return getattr(self._thread_state, "device_id", 0)

    @active_device.setter
    def active_device(self, device_id: int) -> None:
        """Set only this worker's simulated active device."""
        self._thread_state.device_id = device_id

    def current_handle(self) -> int:
        """Record which device owns the requested simulated cuBLAS handle."""
        self.handles.append(self.active_device)
        return self.active_device

    def synchronize(self) -> None:
        """Record synchronization of the active owner without claiming GPU execution."""
        self.synchronized.append(self.active_device)

    def resident(self, value: Any) -> HostPointerArray:
        """Tag one pointer-bearing array with the active simulated device."""
        result = np.asarray(value).view(HostPointerArray)
        result._device_id = self.active_device
        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate mathematical operations to NumPy, retaining native-like pointer arrays."""
        if name == "asnumpy":
            return np.asarray
        if name == "ndarray":
            return HostPointerArray
        value = getattr(np, name)
        if name in {"asarray", "ascontiguousarray", "arange", "empty"}:
            return lambda *args, **kwargs: self.resident(value(*args, **kwargs))
        return value

    def getrf(
        self, handle: int, n: int, pointers: int, lda: int, pivots: int, info: int, count: int
    ) -> None:
        """Apply actual SciPy LU to each simulated column-major matrix pointer."""
        self.factor_calls += 1
        matrices = pointer_array(pointers, count, ctypes.c_size_t)
        pivot_array = pointer_array(pivots, count * n, ctypes.c_int).reshape(count, n)
        information = pointer_array(info, count, ctypes.c_int)
        for index, pointer in enumerate(matrices):
            matrix = pointer_array(pointer, n * n, ctypes.c_double).reshape((n, n), order="F")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", linalg.LinAlgWarning)
                lu, p = linalg.lu_factor(matrix)
            matrix[:] = lu
            pivot_array[index] = p + 1
            zeros = np.flatnonzero(np.diag(lu) == 0)
            information[index] = int(zeros[0] + 1) if len(zeros) else 0

    def getrs(
        self,
        handle: int,
        trans: int,
        n: int,
        nrhs: int,
        pointers: int,
        lda: int,
        pivots: int,
        outputs: int,
        ldb: int,
        info: int,
        count: int,
    ) -> None:
        """Apply actual triangular solves while exposing cleanup/error branches."""
        self.solve_calls += 1
        matrices = pointer_array(pointers, count, ctypes.c_size_t)
        right_sides = pointer_array(outputs, count, ctypes.c_size_t)
        pivot_array = pointer_array(pivots, count * n, ctypes.c_int).reshape(count, n)
        pointer_array(info, 1, ctypes.c_int)[0] = int(self.argument_error)
        for index, (ap, bp) in enumerate(zip(matrices, right_sides, strict=True)):
            lu = pointer_array(ap, n * n, ctypes.c_double).reshape((n, n), order="F")
            rhs = pointer_array(bp, n * nrhs, ctypes.c_double).reshape((n, nrhs), order="F")
            rhs[:] = linalg.lu_solve((lu, pivot_array[index] - 1), rhs)
            if self.corrupt == 1 and self.solve_calls == 1:
                rhs[:] += 0.1
            if self.corrupt == 2:
                rhs[:] = 0
            if self.corrupt == 3:
                rhs[:] = np.nan


@pytest.fixture
def simulated_cuda(monkeypatch: pytest.MonkeyPatch) -> SimulatedCuPy:
    """Install an explicit CPU-only simulation of the optional CUDA ABI."""
    backend = SimulatedCuPy()
    monkeypatch.setattr(gpu, "_cuda", lambda: backend)
    return backend


def test_batched_lu_reuses_factors_and_preserves_matrix(simulated_cuda: SimulatedCuPy) -> None:
    """Check SPD and zero-diagonal saddle matrices with multiple column RHS."""
    matrices = np.array([[[2.0, 1.0], [1.0, 3.0]], [[0.0, 1.0], [1.0, 0.0]]])
    original = matrices.copy()
    exact = np.arange(1.0, 13.0).reshape(2, 2, 3)
    with gpu.BatchedFactorization(matrices) as prepared:
        matrices[:] = 0
        for factor in (1.0, -3.0):
            result = prepared.solve(original @ (factor * exact), host=bool(factor == 1))
            assert_allclose(result, factor * exact, atol=1e-12)
        assert simulated_cuda.factor_calls == 1
        assert simulated_cuda.solve_calls == 2
        assert_allclose(prepared.solve(np.zeros((2, 2, 1))), 0)
    prepared.close()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.solve(np.zeros((2, 2, 1)))
    with pytest.raises(RuntimeError, match="closed"):
        prepared.__enter__()


def test_factorization_owns_device_and_restores_callers_context(
    simulated_cuda: SimulatedCuPy,
) -> None:
    """Device switches cannot move factors, solves, synchronization or array release."""
    simulated_cuda.active_device = 1
    prepared = gpu.BatchedFactorization([[[2.0]]])
    assert prepared.device_id == 1
    simulated_cuda.active_device = 0
    assert_allclose(prepared.solve([[[4.0]]], host=True), 2.0)
    assert simulated_cuda.active_device == 0
    assert set(simulated_cuda.handles) == {1}
    foreign = simulated_cuda.resident([[[4.0]]])
    with pytest.raises(ValueError, match="CUDA device"):
        prepared.solve(foreign)
    assert simulated_cuda.active_device == 0
    prepared.close()
    assert simulated_cuda.synchronized == [1]
    assert simulated_cuda.active_device == 0
    assert prepared.factors is prepared.matrix is prepared.row_scale is prepared.col_scale is None
    prepared.close()
    assert simulated_cuda.synchronized == [1]


def test_foreign_matrix_and_constructor_failure_release_owner(
    simulated_cuda: SimulatedCuPy,
) -> None:
    """Reject another device's matrix and release partially initialized factors on errors."""
    foreign = simulated_cuda.resident([[[2.0]]])
    simulated_cuda.active_device = 1
    with pytest.raises(ValueError, match="CUDA device"):
        gpu.BatchedFactorization(foreign)
    assert simulated_cuda.active_device == 1
    assert simulated_cuda.synchronized == [1]
    with pytest.raises(LinearSolveError, match="zero row"):
        gpu.BatchedFactorization([[[0.0]]])
    assert simulated_cuda.active_device == 1
    assert simulated_cuda.synchronized == [1, 1]


def test_context_failure_releases_arrays_on_owning_device(simulated_cuda: SimulatedCuPy) -> None:
    """Exceptional exit synchronizes and releases the original device's factor arrays."""
    with (
        pytest.raises(RuntimeError, match="user failure"),
        gpu.BatchedFactorization([[[2.0]]]) as prepared,
    ):
        simulated_cuda.active_device = 1
        raise RuntimeError("user failure")
    assert prepared._closed and prepared.matrix is None
    assert simulated_cuda.active_device == 1
    assert simulated_cuda.synchronized == [0]


def test_close_releases_arrays_even_when_stream_reports_failure(
    simulated_cuda: SimulatedCuPy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An asynchronous error cannot retain this object's device allocations."""
    prepared = gpu.BatchedFactorization([[[2.0]]])

    def failure() -> None:
        raise RuntimeError("stream failure")

    monkeypatch.setattr(
        simulated_cuda.cuda, "get_current_stream", lambda: SimpleNamespace(synchronize=failure)
    )
    simulated_cuda.active_device = 1
    with pytest.raises(RuntimeError, match="stream failure"):
        prepared.close()
    assert prepared._closed and prepared.matrix is None and prepared.factors is None
    assert simulated_cuda.active_device == 1


def heterogeneous_problems() -> list[LocalProblem]:
    """Declare unequal sizes, a physical kernel, Petrov retained modes and an empty local space."""
    return [
        LocalProblem([[2.0]], [[1.0]], [3.0], np.array([0])),
        LocalProblem(
            [[1.0, -1.0], [-1.0, 1.0]],
            [[1.0], [0.0]],
            [1.0, 2.0],
            np.array([1]),
            np.ones((2, 1)),
            constraints=[[1.0], [2.0]],
        ),
        LocalProblem(
            [[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 4.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [1.0, 2.0, 3.0],
            np.array([2, 3]),
            coarse_basis=np.ones((3, 1)),
            test_basis=[[1.0], [0.0], [1.0]],
            constraints=[[1.0], [2.0], [3.0]],
            test_coupling=[[1.0, 2.0], [2.0, 0.0], [1.0, -1.0]],
        ),
        LocalProblem(sparse.csc_matrix((0, 0)), np.empty((0, 1)), [], np.array([4])),
    ]


@pytest.mark.parametrize("devices", [(0,), (0, 1)])
@pytest.mark.parametrize("solver", ["auto", "batched", "cudss"])
def test_multi_gpu_preserves_generic_responses_and_device_ownership(
    simulated_cuda: SimulatedCuPy,
    monkeypatch: pytest.MonkeyPatch,
    devices: tuple[int, ...],
    solver: str,
) -> None:
    """Dedicated devices return heterogeneous responses in input order, including zero spaces."""
    from pymhm.linalg.linear import factorize

    sparse_devices = []

    def simulated_sparse(matrix, *, solver, rhs_columns):
        assert solver == "cudss"
        assert rhs_columns > 1
        sparse_devices.append(simulated_cuda.active_device)
        return factorize(matrix, solver="scipy")

    monkeypatch.setattr(gpu, "factorize", simulated_sparse)
    cells = heterogeneous_problems() * 3
    actual = gpu.condense_multi_gpu(
        iter(cells), devices=devices, batch_size=2, solver=solver, dense_size_limit=2
    )
    for problem, response in zip(cells, actual, strict=True):
        expected = problem.condense()
        assert response.problem is problem
        for attribute in ("source", "lifts", "retained_basis"):
            assert_allclose(getattr(response, attribute), getattr(expected, attribute), atol=1e-12)
    assert simulated_cuda.active_device == 0
    assert set(simulated_cuda.synchronized) <= set(devices)
    if solver != "batched":
        assert sparse_devices and set(sparse_devices) <= set(devices)
    assert gpu.condense_multi_gpu([], devices=devices) == ()


@pytest.mark.parametrize(
    "options,match",
    [
        ({"devices": []}, "distinct"),
        ({"devices": [0, 0]}, "distinct"),
        ({"devices": [-1]}, "device"),
        ({"devices": [True]}, "device"),
        ({"devices": [0.5]}, "device"),
        ({"devices": [2]}, "visible"),
        ({"devices": [0], "batch_size": 0}, "batch_size"),
        ({"devices": [0], "dense_size_limit": 0}, "dense_size_limit"),
        ({"devices": [0], "solver": "invalid"}, "solver"),
    ],
)
def test_multi_gpu_rejects_ambiguous_dispatch_inputs(simulated_cuda, options, match) -> None:
    """Validate device ownership and numerical-size choices before submitting work."""
    with pytest.raises(ValueError, match=match):
        gpu.condense_multi_gpu([], **options)


def test_multi_gpu_input_failure_drains_bounded_work_and_restores_owner(simulated_cuda) -> None:
    """A late bad host entry cannot leave previously submitted device resources open."""
    cells = heterogeneous_problems()
    with pytest.raises(TypeError, match="LocalProblem"):
        gpu.condense_multi_gpu([cells[0], None], devices=[0, 1], batch_size=1)
    assert simulated_cuda.active_device == 0
    assert simulated_cuda.synchronized


def test_multi_gpu_worker_failure_is_propagated_after_cleanup(simulated_cuda, monkeypatch) -> None:
    """Factorization failure terminates bounded work while every worker releases its resources."""

    def failure(*args, **kwargs):
        raise RuntimeError("factor failure")

    monkeypatch.setattr(gpu, "factorize", failure)
    with pytest.raises(RuntimeError, match="factor failure"):
        gpu.condense_multi_gpu(
            heterogeneous_problems() * 3, devices=[0, 1], solver="cudss", batch_size=1
        )
    assert simulated_cuda.active_device == 0
    assert simulated_cuda.synchronized


def test_multi_gpu_bounds_input_and_preserves_order_under_unequal_completion(
    simulated_cuda, monkeypatch
) -> None:
    """No device owns more than one pending batch, even when another device completes first."""
    first_started, second_started, release = (threading.Event() for _ in range(3))
    consumed = []
    problem = heterogeneous_problems()[0]

    def items():
        for index in range(8):
            consumed.append(index)
            yield problem.with_load([float(index + 1)])

    def delayed(cells, device, solver, threshold):
        (first_started if device == 0 else second_started).set()
        assert release.wait(timeout=5)
        return tuple(cell.condense() for cell in cells)

    monkeypatch.setattr(gpu, "_condense_device", delayed)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(gpu.condense_multi_gpu, items(), devices=[0, 1], batch_size=2)
        try:
            assert first_started.wait(timeout=5) and second_started.wait(timeout=5)
            assert consumed == [0, 1, 2, 3]
        finally:
            release.set()
        responses = future.result(timeout=5)
    assert_allclose([response.source[0] for response in responses], np.arange(1, 9) / 2)


@pytest.mark.parametrize(
    "matrix",
    [
        [],
        [[1.0]],
        np.empty((0, 1, 1)),
        np.empty((1, 0, 0)),
        np.ones((1, 2, 3)),
        [[[np.nan]]],
        [[[1j]]],
    ],
)
def test_batch_matrix_validation(simulated_cuda: SimulatedCuPy, matrix: Any) -> None:
    """Reject invalid batch geometry, complex values and nonfinite matrices."""
    with pytest.raises(ValueError, match="matrices"):
        gpu.BatchedFactorization(matrix)


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        ([[[0.0, 0.0], [1.0, 2.0]]], "zero row"),
        ([[[0.0, 1.0], [0.0, 2.0]]], "zero column"),
        ([[[1.0, 1.0], [1.0, 1.0]]], "singular pivot"),
        ([[[1.0, 1.0], [1.0, 1.0 + 2e-16]]], "rank deficient"),
    ],
)
def test_batch_singularity_checks(simulated_cuda: SimulatedCuPy, matrix: Any, match: str) -> None:
    """Reject structural and numerical singularity rather than accepting compatible RHS."""
    with pytest.raises(LinearSolveError, match=match):
        gpu.BatchedFactorization(matrix)


@pytest.mark.parametrize(
    "rhs", [[], [[1.0]], np.zeros((1, 2, 1)), np.zeros((1, 1, 0)), [[[np.inf]]], [[[1j]]]]
)
def test_batch_rhs_validation(simulated_cuda: SimulatedCuPy, rhs: Any) -> None:
    """Enforce explicit batch and RHS axes with finite real values."""
    with gpu.BatchedFactorization([[[1.0]]]) as prepared, pytest.raises(ValueError, match="rhs"):
        prepared.solve(rhs)


def test_batch_residual_refinement_and_failures(simulated_cuda: SimulatedCuPy) -> None:
    """Refine with unchanged tolerances and reject persistent/invalid backend output."""
    with gpu.BatchedFactorization([[[2.0]]]) as prepared:
        simulated_cuda.corrupt = 1
        assert_allclose(prepared.solve([[[2.0]]]), 1.0)
        assert simulated_cuda.solve_calls == 2
        simulated_cuda.corrupt = 2
        with pytest.raises(LinearSolveError, match="residual"):
            prepared.solve([[[2.0]]])
        simulated_cuda.corrupt = 3
        with pytest.raises(LinearSolveError, match="nonfinite"):
            prepared.solve([[[2.0]]])
        simulated_cuda.corrupt = 0
        simulated_cuda.argument_error = True
        with pytest.raises(LinearSolveError, match="arguments"):
            prepared.solve([[[2.0]]])


def test_grouped_condensation_matches_shared_algebra(simulated_cuda: SimulatedCuPy) -> None:
    """Group heterogeneous sizes while preserving source, trace and retained-mode order."""
    cells = [
        LocalProblem([[2.0]], [[1.0]], [3.0], np.array([0])),
        LocalProblem(
            [[1.0, -1.0], [-1.0, 1.0]], [[1.0], [0.0]], [1.0, 2.0], np.array([1]), np.ones((2, 1))
        ),
        LocalProblem([[2.0]], [[1.0]], [4.0], np.array([2]), coarse_basis=np.ones((1, 1))),
    ]
    actual = gpu.condense_batched(cells)
    for p, response in zip(cells, actual, strict=True):
        expected = p.condense()
        assert_allclose(response.source, expected.source, atol=1e-12)
        assert_allclose(response.lifts, expected.lifts, atol=1e-12)
        assert_allclose(response.retained_basis, expected.retained_basis, atol=1e-12)
    assert gpu.condense_batched([]) == ()
    with pytest.raises(TypeError, match="LocalProblem"):
        gpu.condense_batched([None])


@pytest.mark.parametrize("dimension", [2, 3])
def test_resident_p1_assembly_moments(simulated_cuda: SimulatedCuPy, dimension: int) -> None:
    """Integrate affine-simplex moments independently in two and three dimensions."""
    from math import factorial

    points = np.vstack((np.zeros(dimension), np.eye(dimension)))
    coordinates = np.stack((points, points * 2 + 0.3))
    cells = np.arange(dimension + 1)[None]
    tensor = np.diag(np.arange(1.0, dimension + 1))
    stiffness, mass, load = gpu.assemble_p1_batch(coordinates, cells, diffusion=tensor, source=3.0)
    for index, length in enumerate((1.0, 2.0)):
        volume = length**dimension / factorial(dimension)
        gradient = np.vstack((-np.ones(dimension), np.eye(dimension))) / length
        assert_allclose(stiffness[index], volume * gradient @ tensor @ gradient.T, atol=1e-14)
        assert_allclose(mass[index].sum(), volume)
        assert_allclose(load[index], volume * 3 / (dimension + 1))
    a, _, _ = gpu.assemble_p1_batch(coordinates, cells)
    assert_allclose(a.sum(axis=-1), 0, atol=1e-14)


@pytest.mark.parametrize(
    ("points", "cells", "kwargs", "message"),
    [
        ([[0.0, 0.0]], [[0, 1, 2]], {}, "points"),
        ([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], [[0, 1]], {}, "cells"),
        ([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], [[0.0, 1.0, 2.0]], {}, "cells"),
        ([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], [[0, 1, 3]], {}, "cells"),
        ([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], [[0, 1, 2]], {}, "degenerate"),
        ([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], [[0, 1, 2]], {"diffusion": -1.0}, "diffusion"),
        (
            [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
            [[0, 1, 2]],
            {"diffusion": [[1.0, 2.0], [0.0, 1.0]]},
            "diffusion",
        ),
        ([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], [[0, 1, 2]], {"source": np.nan}, "source"),
    ],
)
def test_resident_assembly_invalid_contracts(
    simulated_cuda: SimulatedCuPy, points: Any, cells: Any, kwargs: Any, message: str
) -> None:
    """Reject malformed geometry and coefficients before device assembly."""
    with pytest.raises(ValueError, match=message):
        gpu.assemble_p1_batch(points, cells, **kwargs)


@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.parametrize("device_id", [0, 1])
def test_native_factorization_device_switch_and_resource_release(device_id: int) -> None:
    """Native factors solve and close on their owner after an external device switch."""
    cp = pytest.importorskip("cupy")
    count = cp.cuda.runtime.getDeviceCount()
    if device_id >= count:
        pytest.skip("requested CUDA device unavailable")
    matrix = np.array([[[2.0, 1.0], [1.0, 3.0]]])
    expected = np.array([[[1.0, -2.0], [3.0, 0.5]]])
    with cp.cuda.Device(device_id):
        prepared = gpu.BatchedFactorization(matrix)
        sentinel = cp.ones(4)
    caller = (1 - device_id) if count > 1 else device_id
    with cp.cuda.Device(caller):
        result = prepared.solve(matrix @ expected)
        assert result.device.id == device_id
        assert cp.cuda.runtime.getDevice() == caller
        assert_allclose(cp.asnumpy(result), expected, atol=1e-12)
        if caller != device_id:
            foreign = cp.asarray(matrix @ expected)
            with pytest.raises(ValueError, match="CUDA device"):
                prepared.solve(foreign)
            with cp.cuda.Device(device_id):
                owner_matrix = cp.asarray(matrix)
            with pytest.raises(ValueError, match="CUDA device"):
                gpu.BatchedFactorization(owner_matrix)
        with pytest.raises(RuntimeError, match="user failure"), prepared:
            raise RuntimeError("user failure")
        assert cp.cuda.runtime.getDevice() == caller
        assert prepared._closed and prepared.matrix is None and prepared.factors is None
    with cp.cuda.Device(device_id):
        assert_allclose(cp.asnumpy(sentinel), 1.0)
    prepared.close()


@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.parametrize("devices", [(0,), (0, 1)])
def test_native_multi_gpu_mixed_sizes_original_operator_and_physical_gauge(devices) -> None:
    """One/two-device condensation preserves local equations and a nonzero physical mean."""
    cp = pytest.importorskip("cupy")
    pytest.importorskip("nvmath.sparse.advanced")
    if max(devices) >= cp.cuda.runtime.getDeviceCount():
        pytest.skip("two visible CUDA devices are required for this configuration")
    cells = [
        LocalProblem(
            [[1.0, -1.0], [-1.0, 1.0]],
            np.diag([1.0, -1.0]),
            [0.0, 0.0],
            np.array([i, i + 1]),
            np.ones((2, 1)),
            constraints=np.ones((2, 1)) / 2,
        )
        for i in range(4)
    ]
    # An invertible local field shares trace zero and exercises the smaller dense group.
    cells.insert(1, LocalProblem([[2.0]], [[1.0]], [3.0], np.array([0])))
    before = [
        (problem.matrix.copy(), problem.coupling.copy(), problem.load.copy()) for problem in cells
    ]
    actual_responses = gpu.condense_multi_gpu(
        iter(cells), devices=devices, batch_size=2, dense_size_limit=2
    )
    boundary = [1.0, 0.0, 0.0, 0.0, -2.0]
    actual = HybridSystem.from_responses(actual_responses, boundary_load=boundary)
    expected = HybridSystem(cells, boundary_load=boundary)
    reference = expected.solve()
    weights = [np.ones(len(problem.load)) / len(problem.load) for problem in cells]
    target = sum(
        float(weight @ field) for weight, field in zip(weights, reference.fields, strict=True)
    )
    row, value = actual.mean_constraint(weights, target)
    solution = actual.solve(constraints=[(row, value)])
    assert abs(target) > 0.1
    assert_allclose(solution.trace, reference.trace, atol=2e-12, rtol=1e-12)
    assert_allclose(solution.gauge_multipliers, 0.0, atol=2e-12)
    for index, (response, field, original) in enumerate(
        zip(actual_responses, solution.fields, before, strict=True)
    ):
        assert_allclose(field, reference.fields[index], atol=2e-12, rtol=1e-12)
        assert_allclose(
            response.problem.matrix @ field
            + response.problem.coupling @ solution.trace[response.problem.trace_dofs],
            response.problem.load,
            atol=2e-12,
            rtol=1e-12,
        )
        assert_allclose(response.problem.matrix.toarray(), original[0].toarray(), atol=0.0)
        assert_allclose(response.problem.coupling, original[1], atol=0.0)
        assert_allclose(response.problem.load, original[2], atol=0.0)


@pytest.mark.serial
@pytest.mark.gpu
def test_native_multi_gpu_large_sparse_grid_repeated_condensation(monkeypatch) -> None:
    """Repeat two-device sparse Q1 saddles with 21 RHS and a nonzero physical mean.

    Each local mesh has 50 by 50 quadrilaterals and 2,601 nodal values. These
    graphs exercise native sparse analysis beyond tiny dense ordering controls.
    The declared constant kernel and mass-weighted mean are fixed explicitly;
    source data satisfy the original local equations for prescribed trace data.
    This is a correctness and resource test, with no timing or speedup claim.
    """
    cp = pytest.importorskip("cupy")
    native = pytest.importorskip("nvmath.sparse.advanced")
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("two visible CUDA devices are required")
    from pymhm.fem.scalar.quadrilateral import quadrilateral_operators
    from pymhm.meshes.cartesian import CartesianMacroMesh

    mesh = CartesianMacroMesh(50, 50)
    stiffness, mass, _ = quadrilateral_operators(mesh, 1, order=4)
    size, width = len(mesh.points), 20
    mean = np.asarray(mass.sum(axis=1)).ravel()
    mean /= mean.sum()
    kernel = np.ones((size, 1))
    x, y = mesh.points.T
    boundary = np.any((mesh.points == 0) | (mesh.points == 1), axis=1)
    rng = np.random.default_rng(87521)
    coupling = np.zeros((size, width))
    coupling[boundary] = rng.normal(size=(np.count_nonzero(boundary), width)) / 50
    trace = np.linspace(-0.7, 0.9, width)
    problems, targets = [], []
    for index in range(4):
        matrix = stiffness * (1 + 0.25 * index)
        target = 1.2 + 0.1 * index + 0.2 * np.sin(np.pi * x) * np.sin(2 * np.pi * y) + 0.1 * x
        local = LocalProblem(
            matrix,
            coupling,
            matrix @ target + coupling @ trace,
            np.arange(width),
            kernel,
            constraints=mean[:, None],
        )
        augmented, rhs = local.condensation_system()
        assert augmented.shape == (2602, 2602) and rhs.shape[1] >= 21
        problems.append(local)
        targets.append(target)
    reference = tuple(problem.condense() for problem in problems)
    original_plan, original_free = native.DirectSolver.plan, native.DirectSolver.free
    counters = {"active": 0, "maximum": 0}
    plans, freed = [], []
    audit_lock = threading.Lock()

    def checked_plan(engine: Any, *args: Any, **kwargs: Any) -> Any:
        """Observe native ANALYSIS entry without adding any serialization to it."""
        owner = engine.device_id
        assert cp.cuda.runtime.getDevice() == owner
        with audit_lock:
            counters["active"] += 1
            counters["maximum"] = max(counters["maximum"], counters["active"])
            plans.append((owner, threading.get_ident()))
        try:
            return original_plan(engine, *args, **kwargs)
        finally:
            with audit_lock:
                counters["active"] -= 1

    def checked_free(engine: Any) -> None:
        """Require native resource destruction on the construction device."""
        assert cp.cuda.runtime.getDevice() == engine.device_id
        owner = engine.device_id
        original_free(engine)
        assert not engine.valid_state
        with audit_lock:
            freed.append(owner)

    monkeypatch.setattr(native.DirectSolver, "plan", checked_plan)
    monkeypatch.setattr(native.DirectSolver, "free", checked_free)
    caller = cp.cuda.runtime.getDevice()
    sentinels = []
    for device in (0, 1):
        with cp.cuda.Device(device):
            sentinels.append(cp.full(8, device + 2.0))
    for _ in range(3):
        responses = gpu.condense_multi_gpu(problems, devices=(0, 1), batch_size=1, solver="cudss")
        assert cp.cuda.runtime.getDevice() == caller
        for actual, expected, target in zip(responses, reference, targets, strict=True):
            assert actual.problem is expected.problem
            assert_allclose(actual.source, expected.source, atol=2e-12, rtol=1e-10)
            assert_allclose(actual.lifts, expected.lifts, atol=2e-12, rtol=1e-10)
            assert_allclose(actual.retained_basis, expected.retained_basis, atol=2e-12, rtol=1e-10)
            field = actual.reconstruct(trace, np.array([mean @ target]))
            assert_allclose(field, target, atol=2e-12, rtol=1e-10)
            assert_allclose(mean @ field, mean @ target, atol=2e-12, rtol=1e-12)
            defect = (
                actual.problem.matrix @ field
                + actual.problem.coupling @ trace
                - actual.problem.load
            )
            assert np.linalg.norm(defect) <= 1e-10 * np.linalg.norm(actual.problem.load)
    assert counters == {"active": 0, "maximum": 1}
    assert len(plans) == len(freed) == 12 and set(freed) == {0, 1}
    assert {owner for owner, _ in plans} == {0, 1}
    assert len({thread for _, thread in plans}) >= 2
    for device, sentinel in enumerate(sentinels):
        with cp.cuda.Device(device):
            cp.cuda.get_current_stream().synchronize()
            assert_allclose(cp.asnumpy(sentinel), device + 2.0, atol=0.0)


@pytest.mark.serial
@pytest.mark.gpu
def test_native_gpu_batched_condensation_and_resident_repeated_rhs() -> None:
    """Exercise actual cuBLAS batches and MHM condensation on a visible CUDA device."""
    cupy = pytest.importorskip("cupy")
    if not cupy.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA device unavailable")
    rng = np.random.default_rng(271)
    matrices = rng.normal(size=(16, 12, 12))
    matrices = matrices @ matrices.swapaxes(-1, -2) + np.eye(12)
    expected = cupy.asarray(rng.normal(size=(16, 12, 4)))
    device_matrices = cupy.asarray(matrices)
    with gpu.BatchedFactorization(device_matrices) as prepared:
        for scale in (1.0, -2.0):
            actual = prepared.solve(device_matrices @ (expected * scale))
            assert isinstance(actual, cupy.ndarray)
            assert_allclose(cupy.asnumpy(actual), cupy.asnumpy(expected * scale), atol=1e-12)
    cells = [
        LocalProblem(
            [[1.0, -1.0], [-1.0, 1.0]],
            np.diag([1.0, -1.0]),
            [0.0, 0.0],
            np.array([i, i + 1]),
            np.ones((2, 1)),
        )
        for i in range(4)
    ]
    actual_system = HybridSystem.from_responses(
        gpu.condense_batched(cells), boundary_load=[1.0, 0.0, 0.0, 0.0, -2.0]
    )
    expected_system = HybridSystem(cells, boundary_load=[1.0, 0.0, 0.0, 0.0, -2.0])
    assert_allclose(actual_system.solve().fields, expected_system.solve().fields, atol=1e-12)
    from pymhm.fem.scalar.operators import p1_operators
    from pymhm.meshes.triangle import TriangleMesh

    mesh = TriangleMesh.unit_square(3)
    a, m, f = gpu.assemble_p1_batch(
        cupy.asarray(np.stack([mesh.points] * 3)),
        cupy.asarray(mesh.cells),
        diffusion=[[2.0, 0.2], [0.2, 1.0]],
        source=3.0,
    )
    expected_a, expected_m, expected_f = p1_operators(
        mesh, diffusion=[[2.0, 0.2], [0.2, 1.0]], source=3.0
    )
    for index in range(3):
        assert_allclose(cupy.asnumpy(a[index]), expected_a.toarray(), atol=1e-13)
        assert_allclose(cupy.asnumpy(m[index]), expected_m.toarray(), atol=1e-14)
        assert_allclose(cupy.asnumpy(f[index]), expected_f, atol=1e-14)
    from pymhm.fem.scalar.tetrahedron import tetra_operators
    from pymhm.meshes.tetrahedron import TetraMesh

    tetra = TetraMesh.unit_cube(1).submesh(0, 2)
    tensor = [[2.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 3.0]]
    a, m, f = gpu.assemble_p1_batch(
        cupy.asarray(np.stack([tetra.points] * 2)),
        cupy.asarray(tetra.cells),
        diffusion=tensor,
        source=2.5,
    )
    expected_a, expected_m, expected_f = tetra_operators(
        tetra, degree=1, diffusion=tensor, source=2.5
    )
    for index in range(2):
        assert_allclose(cupy.asnumpy(a[index]), expected_a.toarray(), atol=1e-13)
        assert_allclose(cupy.asnumpy(m[index]), expected_m.toarray(), atol=1e-14)
        assert_allclose(cupy.asnumpy(f[index]), expected_f, atol=1e-14)
