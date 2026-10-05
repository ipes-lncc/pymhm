"""Resident, batched double-precision local algebra for CUDA devices.

Dense batching is intended for many small local systems of equal size. Sparse
finite-element assembly remains a separate operation; this module accepts both
host arrays and matrices assembled directly as CuPy arrays. Factorizations,
right-hand sides, residual checks and returned solutions stay on the device
unless the caller explicitly requests host arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from itertools import islice
from math import factorial
from types import TracebackType
from typing import Any, Literal

import numpy as np

from pymhm.core.contracts import LocalProblem, LocalResponse
from pymhm.core.validation import positive_int
from pymhm.linalg.linear import LinearSolveError, _cuda, _tolerances, factorize


def assemble_p1_batch(
    points: Any, cells: Any, *, diffusion: Any = 1.0, source: Any = 0.0
) -> tuple[Any, Any, Any]:
    """Assemble dense P1 simplex operators entirely on the current CUDA device.

    ``points`` has shape ``(batch,npoints,d)``, d=2 or 3; all local meshes share
    the integer connectivity ``cells[ncells,d+1]``. Diffusion is a positive scalar
    or SPD tensors broadcastable to ``(batch,ncells,d,d)``. Source values are
    constant per fine cell, broadcastable to ``(batch,ncells)``. The returned
    stiffness, consistent mass and source vectors are resident CuPy arrays.
    Geometry, Jacobians, physical gradients, material contractions and assembly
    scatter-add all run on the GPU. No coefficient callback or curved geometry
    is inferred. Dense global-in-each-local-mesh storage limits useful sizes.
    """
    cp = _cuda()
    coordinates = cp.asarray(points)
    connectivity = cp.asarray(cells)
    if (
        coordinates.ndim != 3
        or coordinates.shape[-1] not in (2, 3)
        or not coordinates.shape[0]
        or not coordinates.shape[1]
        or coordinates.dtype.kind == "c"
        or not bool(cp.all(cp.isfinite(coordinates)))
    ):
        raise ValueError("points must be finite real arrays shaped (batch,npoints,2 or 3)")
    count, size, dimension = coordinates.shape
    if (
        connectivity.ndim != 2
        or connectivity.shape[1] != dimension + 1
        or not len(connectivity)
        or connectivity.dtype.kind not in "iu"
        or bool(cp.any(connectivity < 0))
        or bool(cp.any(connectivity >= size))
    ):
        raise ValueError("cells must contain valid simplex node indices")
    coordinates = coordinates.astype(cp.float64, copy=False)
    vertices = coordinates[:, connectivity]
    jacobian = (vertices[:, :, 1:] - vertices[:, :, :1]).swapaxes(-1, -2)
    determinant = cp.linalg.det(jacobian)
    bound = cp.prod(cp.linalg.norm(jacobian, axis=-2), axis=-1)
    if bool(cp.any(cp.abs(determinant) <= np.finfo(float).eps * bound)):
        raise ValueError("degenerate simplex in resident geometry")
    volume = cp.abs(determinant) / factorial(dimension)
    reference = cp.concatenate((-cp.ones((1, dimension)), cp.eye(dimension)))
    gradients = cp.einsum("ij,btjk->btik", reference, cp.linalg.inv(jacobian))
    raw = cp.asarray(diffusion)
    if raw.ndim == 0:
        raw = raw * cp.eye(dimension)
    tensor = cp.broadcast_to(raw, (count, len(connectivity), dimension, dimension))
    if (
        tensor.dtype.kind == "c"
        or not bool(cp.all(cp.isfinite(tensor)))
        or not bool(cp.allclose(tensor, tensor.swapaxes(-1, -2), rtol=1e-12, atol=0.0))
        or bool(cp.any(cp.linalg.eigvalsh(tensor) <= 0))
    ):
        raise ValueError("diffusion must be finite real symmetric positive definite")
    forcing = cp.broadcast_to(cp.asarray(source), (count, len(connectivity)))
    if forcing.dtype.kind == "c" or not bool(cp.all(cp.isfinite(forcing))):
        raise ValueError("source must be real finite cell values")
    local_stiffness = cp.einsum("et,etia,etab,etjb->etij", volume, gradients, tensor, gradients)
    local_mass = (
        volume[:, :, None, None]
        * (cp.ones((dimension + 1, dimension + 1)) + cp.eye(dimension + 1))
        / ((dimension + 1) * (dimension + 2))
    )
    stiffness, mass = cp.zeros((count, size, size)), cp.zeros((count, size, size))
    load = cp.zeros((count, size))
    batch_indices = cp.arange(count)[:, None, None, None]
    rows = connectivity[None, :, :, None]
    columns = connectivity[None, :, None, :]
    cp.add.at(stiffness, (batch_indices, rows, columns), local_stiffness)
    cp.add.at(mass, (batch_indices, rows, columns), local_mass)
    cp.add.at(
        load,
        (cp.arange(count)[:, None, None], connectivity[None]),
        (volume * forcing)[:, :, None] / (dimension + 1),
    )
    return stiffness, mass, load


def _pointers(array: Any, cupy: Any) -> Any:
    """Build a device array of pointers to contiguous matrices in one batch."""
    stride = array.shape[-2] * array.shape[-1] * array.itemsize
    return cupy.arange(array.shape[0], dtype=cupy.uintp) * stride + array.data.ptr


def _factor_batch(matrices: Any, cupy: Any) -> tuple[Any, Any, Any]:
    """Factor independent column-major matrices using cuBLAS batched pivoted LU."""
    count, size, _ = matrices.shape
    storage = cupy.ascontiguousarray(matrices.swapaxes(-1, -2))
    pointers = _pointers(storage, cupy)
    pivots = cupy.empty((count, size), dtype=cupy.int32)
    info = cupy.empty(count, dtype=cupy.int32)
    cupy.cuda.cublas.dgetrfBatched(
        cupy.cuda.device.get_cublas_handle(),
        size,
        pointers.data.ptr,
        size,
        pivots.data.ptr,
        info.data.ptr,
        count,
    )
    if bool(cupy.any(info != 0)):
        raise LinearSolveError("batched LU failed: a local matrix has a singular pivot")
    diagonal = cupy.abs(cupy.diagonal(storage, axis1=-2, axis2=-1))
    scale = cupy.max(cupy.abs(storage), axis=(-2, -1))
    if bool(cupy.any(cupy.min(diagonal, axis=1) <= np.finfo(float).eps * size * scale)):
        raise LinearSolveError("batched local matrix is numerically rank deficient")
    return storage, pointers, pivots


def _solve_batch(factors: tuple[Any, Any, Any], rhs: Any, cupy: Any) -> Any:
    """Apply resident batched LU factors to all source/trace columns together."""
    storage, pointers, pivots = factors
    count, size, _ = storage.shape
    result = cupy.ascontiguousarray(rhs.swapaxes(-1, -2))
    output_pointers = _pointers(result, cupy)
    # cuBLAS getrsBatched reports argument status through a host integer.
    info = np.zeros(1, dtype=np.int32)
    cupy.cuda.cublas.dgetrsBatched(
        cupy.cuda.device.get_cublas_handle(),
        cupy.cuda.cublas.CUBLAS_OP_N,
        size,
        rhs.shape[-1],
        pointers.data.ptr,
        size,
        pivots.data.ptr,
        output_pointers.data.ptr,
        size,
        info.ctypes.data,
        count,
    )
    if info[0] != 0:
        raise LinearSolveError("batched triangular solve rejected its arguments")
    return result.swapaxes(-1, -2)


class BatchedFactorization:
    """Reusable resident LU for real arrays of shape ``(batch, n, n)``.

    The matrices are copied and equilibrated by rows and columns before LU.
    Residual acceptance is checked in original units, separately for every
    matrix and RHS column. Refinement reuses the same factors. This is dense
    algebra; quadratic storage and cubic local factorization limit useful sizes.
    A single object belongs to the CUDA device active at construction. Operations
    enter that device and restore the caller's context. Host inputs are copied to
    the owning device; resident arrays from another device are rejected. The
    object must not be shared concurrently between threads or processes.
    """

    def __init__(self, matrices: Any, *, rtol: float = 1e-10, atol: float = 0.0) -> None:
        """Upload or copy one equal-size matrix batch and factor it once."""
        _tolerances(rtol, atol)
        self.cupy = _cuda()
        cp = self.cupy
        self.device_id = int(cp.cuda.runtime.getDevice())
        self.matrix: Any = None
        self.row_scale: Any = None
        self.col_scale: Any = None
        self.factors: Any = None
        self._closed = False
        try:
            with cp.cuda.Device(self.device_id):
                raw = self._owned_array(matrices)
                if (
                    raw.ndim != 3
                    or raw.shape[0] == 0
                    or raw.shape[1] == 0
                    or raw.shape[1] != raw.shape[2]
                    or raw.dtype.kind == "c"
                    or not bool(cp.all(cp.isfinite(raw)))
                ):
                    raise ValueError("matrices must be finite real arrays of shape (batch,n,n)")
                self.matrix = raw.astype(cp.float64, copy=True)
                row_max = cp.max(cp.abs(self.matrix), axis=2)
                if bool(cp.any(row_max == 0)):
                    raise LinearSolveError("batched matrix contains a zero row")
                self.row_scale = 1.0 / row_max
                scaled = self.matrix * self.row_scale[:, :, None]
                col_max = cp.max(cp.abs(scaled), axis=1)
                if bool(cp.any(col_max == 0)):
                    raise LinearSolveError("batched matrix contains a zero column")
                self.col_scale = 1.0 / col_max
                self.factors = _factor_batch(scaled * self.col_scale[:, None, :], cp)
                self.rtol, self.atol = rtol, atol
        except BaseException:
            self.close()
            raise

    def _owned_array(self, value: Any) -> Any:
        """Copy host operands to the owner and reject resident cross-device operands."""
        if isinstance(value, self.cupy.ndarray) and value.device.id != self.device_id:
            raise ValueError("resident arrays must belong to the factorization's CUDA device")
        return self.cupy.asarray(value)

    def solve(self, rhs: Any, *, host: bool = False) -> Any:
        """Solve ``(batch,n,nrhs)`` loads; return resident arrays unless ``host=True``."""
        if self._closed:
            raise RuntimeError("batched factorization is closed")
        cp = self.cupy
        with cp.cuda.Device(self.device_id):
            raw = self._owned_array(rhs)
            if (
                raw.ndim != 3
                or raw.shape[:2] != self.matrix.shape[:2]
                or raw.shape[2] == 0
                or raw.dtype.kind == "c"
                or not bool(cp.all(cp.isfinite(raw)))
            ):
                raise ValueError("rhs must be finite real arrays of shape (batch,n,nrhs)")
            forcing = raw.astype(cp.float64, copy=False)
            result = self._apply(forcing)
            for correction in range(3):
                if not bool(cp.all(cp.isfinite(result))):
                    raise LinearSolveError("batched solver returned nonfinite coefficients")
                residual = forcing - self.matrix @ result
                scale = cp.maximum(self.atol, cp.max(cp.abs(forcing), axis=1))
                scale = cp.where(scale > 0, scale, 1.0)
                norms = cp.linalg.norm(residual / scale[:, None, :], axis=1)
                limits = cp.maximum(
                    self.atol / scale,
                    self.rtol * cp.linalg.norm(forcing / scale[:, None, :], axis=1),
                )
                if bool(cp.all(norms <= limits)):
                    return cp.asnumpy(result) if host else result
                if correction < 2:
                    result += self._apply(residual)
            raise LinearSolveError("batched residual criterion failed")

    def _apply(self, rhs: Any) -> Any:
        """Apply row scaling, batched triangular solves and column scaling on device."""
        with self.cupy.cuda.Device(self.device_id):
            return (
                _solve_batch(self.factors, rhs * self.row_scale[:, :, None], self.cupy)
                * self.col_scale[:, :, None]
            )

    def close(self) -> None:
        """Synchronize the owner and release arrays without clearing its application pool."""
        if not self._closed:
            with self.cupy.cuda.Device(self.device_id):
                try:
                    self.cupy.cuda.get_current_stream().synchronize()
                finally:
                    self._closed = True
                    self.factors = self.matrix = self.row_scale = self.col_scale = None

    def __enter__(self) -> BatchedFactorization:
        """Enter an open device-factorization context."""
        if self._closed:
            raise RuntimeError("batched factorization is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release this object's device arrays on normal and exceptional exit."""
        self.close()


def condense_batched(problems: Sequence[LocalProblem]) -> tuple[LocalResponse, ...]:
    """Condense homogeneous-size groups on the GPU, preserving original order.

    Matrices originate from each problem's shared condensation contract, including
    nonsymmetric test/trial spaces. Grouping uses augmented size and RHS count.
    Host conversion occurs once per group after all lifts have been computed;
    the returned responses interoperate with the ordinary global assembler.
    For repeated resident algebra use ``BatchedFactorization`` directly.
    """
    groups: dict[tuple[int, int], list[tuple[int, LocalProblem, Any, Any]]] = {}
    for index, problem in enumerate(problems):
        if not isinstance(problem, LocalProblem):
            raise TypeError("every batch entry must be a LocalProblem")
        matrix, rhs = problem.condensation_system()
        groups.setdefault((matrix.shape[0], rhs.shape[1]), []).append((index, problem, matrix, rhs))
    responses: dict[int, LocalResponse] = {}
    for group in groups.values():
        with BatchedFactorization(np.stack([entry[2].toarray() for entry in group])) as batch:
            solved = batch.solve(np.stack([entry[3] for entry in group]), host=True)
        for (index, problem, _, _), result in zip(group, solved, strict=True):
            responses[index] = problem.response_from_solution(result)
    return tuple(responses[index] for index in range(len(problems)))


def _condense_device(
    problems: Sequence[LocalProblem],
    device_id: int,
    solver: Literal["auto", "batched", "cudss"],
    dense_size_limit: int,
) -> tuple[LocalResponse, ...]:
    """Own one device batch, its transfers, factors and synchronization in one worker."""
    cp = _cuda()
    responses: dict[int, LocalResponse] = {}
    small: list[tuple[int, LocalProblem]] = []
    with cp.cuda.Device(device_id):
        try:
            for index, problem in enumerate(problems):
                size = problem.matrix.shape[0] + problem.coarse_basis.shape[1]
                if not problem.matrix.shape[0]:
                    responses[index] = problem.condense()
                elif solver == "batched" or (solver == "auto" and size <= dense_size_limit):
                    small.append((index, problem))
                else:
                    matrix, rhs = problem.condensation_system()
                    with factorize(matrix, solver="cudss", rhs_columns=rhs.shape[1]) as prepared:
                        responses[index] = problem.response_from_solution(prepared.solve(rhs))
            for (index, _), response in zip(
                small, condense_batched([problem for _, problem in small]), strict=True
            ):
                responses[index] = response
        finally:
            cp.cuda.get_current_stream().synchronize()
    return tuple(responses[index] for index in range(len(problems)))


def condense_multi_gpu(
    problems: Iterable[LocalProblem],
    *,
    devices: Sequence[int],
    batch_size: int = 8,
    solver: Literal["auto", "batched", "cudss"] = "auto",
    dense_size_limit: int = 512,
) -> tuple[LocalResponse, ...]:
    """Condense generic host local problems on explicitly assigned CUDA devices.

    One dedicated thread owns each unique device. At most one batch per device
    is submitted; input is consumed incrementally and responses retain original
    order. Returned responses remain host data suitable for the ordinary global
    assembler. Local operators, moments, test/trial spaces and residual tolerances
    come from the shared condensation contract; no PDE or trace is inferred.

    ``auto`` uses dense batched LU up to ``dense_size_limit`` augmented rows and
    sparse cuDSS above that limit. Explicit ``batched`` and ``cudss`` select one
    path for all nonempty systems. Dense groups reuse :func:`condense_batched`;
    sparse systems reuse the shared checked factorization owner. Host/device
    transfers, native setup, stream synchronization and executor shutdown finish
    before return. Factors are released within the owning device context on
    normal and exceptional exit. Resident cross-device arrays are not accepted
    by the batched factorization; this interface receives host ``LocalProblem``
    objects. The bound controls submitted work, not total stored responses.
    """
    batch_size = positive_int(batch_size, "batch_size")
    dense_size_limit = positive_int(dense_size_limit, "dense_size_limit")
    device_ids = tuple(positive_int(value, "device", 0) for value in devices)
    if not device_ids or len(set(device_ids)) != len(device_ids):
        raise ValueError("devices must contain distinct explicit CUDA device IDs")
    if solver not in {"auto", "batched", "cudss"}:
        raise ValueError("solver must be auto, batched or cudss")
    cp = _cuda()
    if any(device >= cp.cuda.runtime.getDeviceCount() for device in device_ids):
        raise ValueError("device ID is outside the visible CUDA devices")
    iterator = enumerate(problems)
    responses: dict[int, LocalResponse] = {}
    pending: dict[
        int, tuple[list[tuple[int, LocalProblem]], Future[tuple[LocalResponse, ...]]]
    ] = {}

    def submit(executor: ThreadPoolExecutor, device_id: int) -> bool:
        """Consume only this device's next bounded batch and validate its host contract."""
        batch = list(islice(iterator, batch_size))
        if not batch:
            return False
        if not all(isinstance(problem, LocalProblem) for _, problem in batch):
            raise TypeError("every multi-GPU batch entry must be a LocalProblem")
        future = executor.submit(
            _condense_device,
            [problem for _, problem in batch],
            device_id,
            solver,
            dense_size_limit,
        )
        pending[device_id] = batch, future
        return True

    with ExitStack() as resources:
        executors = {
            device: resources.enter_context(
                ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"pymhm-cuda-{device}")
            )
            for device in device_ids
        }
        try:
            for device in device_ids:
                submit(executors[device], device)
            while pending:
                completed, _ = wait(
                    [future for _, future in pending.values()], return_when=FIRST_COMPLETED
                )
                for device in tuple(pending):
                    batch, future = pending[device]
                    if future not in completed:
                        continue
                    for (index, _), response in zip(batch, future.result(), strict=True):
                        responses[index] = response
                    del pending[device]
                    submit(executors[device], device)
        except BaseException:
            for _, future in pending.values():
                future.cancel()
            raise
    return tuple(responses[index] for index in range(len(responses)))
