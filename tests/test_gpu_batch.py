"""Batched CUDA contracts with explicit CPU simulation and separate native checks."""

import ctypes
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg

from pymhm import gpu
from pymhm.hybrid import HybridSystem, LocalProblem
from pymhm.solvers import LinearSolveError


class HostPointerArray(np.ndarray):
    """Expose host pointers for an explicitly simulated cuBLAS ABI contract."""

    @property
    def data(self) -> Any:
        """Expose a pointer shape matching CuPy without pretending to use CUDA."""
        return SimpleNamespace(ptr=self.ctypes.data)


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
        self.cuda = SimpleNamespace(
            cublas=SimpleNamespace(
                dgetrfBatched=self.getrf, dgetrsBatched=self.getrs, CUBLAS_OP_N=0
            ),
            device=SimpleNamespace(get_cublas_handle=lambda: 0),
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate mathematical operations to NumPy, retaining native-like pointer arrays."""
        if name == "asnumpy":
            return np.asarray
        value = getattr(np, name)
        if name in {"asarray", "ascontiguousarray", "arange", "empty"}:
            return lambda *args, **kwargs: value(*args, **kwargs).view(HostPointerArray)
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
    from pymhm.elements import p1_operators
    from pymhm.mesh import TriangleMesh

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
    from pymhm.tetrahedral import TetraMesh, tetra_operators

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
