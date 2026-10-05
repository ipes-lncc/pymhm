"""Numerical core checks and explicitly simulated optional-backend API contracts."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm._legacy.models.vector import solve_brinkman
from pymhm.linalg import linear as solvers
from pymhm.linalg.linear import (
    LinearSolveError,
    SolverUnavailableError,
    factorize,
    prepare_amgx,
    solve_linear,
)


@pytest.mark.parametrize("solver", ["scipy", "cg", "minres", "gmres"])
@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("multiple", [False, True])
def test_solvers_manufactured_system(solver: str, as_sparse: bool, multiple: bool) -> None:
    matrix = np.diag(np.arange(3, 9)) + 0.25 * np.ones((6, 6))
    expected = np.arange(6.0)[:, None] * np.array([1.0, -0.3]) if multiple else np.arange(6.0)
    rhs = matrix @ expected
    operator = sparse.coo_array(matrix) if as_sparse else matrix
    assert_allclose(solve_linear(operator, rhs, solver=solver), expected, atol=1e-12)


def test_external_solution_checks_independently_scaled_original_columns() -> None:
    """An accurate large column cannot mask an inaccurate small external response."""
    matrix = np.array([[2.0, -1], [-1, 3]])
    expected = np.array([[1.0, 1e-15], [-2.0, -2e-15]])
    rhs = matrix @ expected
    assert_allclose(solvers.check_linear_solution(matrix, rhs, expected), expected, rtol=0, atol=0)
    altered = expected.copy()
    altered[:, 1] *= 0.999
    with pytest.raises(LinearSolveError, match="residual criterion"):
        solvers.check_linear_solution(matrix, rhs, altered)


@pytest.mark.parametrize("solver", ["scipy", "cg", "gmres"])
def test_complex_system(solver: str) -> None:
    matrix = np.array([[3, 1j], [-1j, 2]])
    expected = np.array([1 + 2j, -2j])
    assert_allclose(solve_linear(matrix, matrix @ expected, solver=solver), expected)


@pytest.mark.parametrize("solver", ["scipy", "cg", "minres", "gmres"])
@pytest.mark.parametrize("multiple", [False, True])
def test_extended_original_rhs_digits_define_residual(solver: str, multiple: bool) -> None:
    """The requested criterion belongs to the supplied RHS, before backend rounding."""
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        return
    epsilon = np.longdouble(2) ** -60
    # A coordinate axis avoids introducing double-precision Krylov normalization
    # roundoff into this check of the supplied RHS digits.
    rhs = np.array([1 + epsilon, 0], dtype=np.longdouble)
    if multiple:
        rhs = np.column_stack((rhs, rhs * 2))
    actual = solve_linear(
        np.eye(2), rhs, solver=solver, rtol=1e-22, refinement_precision="extended"
    )
    assert_allclose(actual, rhs, rtol=0, atol=0)
    assert not np.array_equal(np.asarray(actual, dtype=float).astype(np.longdouble), rhs)
    with factorize(np.eye(2), rtol=1e-22) as factor:
        assert_allclose(factor.solve(rhs, refinement_precision="extended"), rhs, rtol=0, atol=0)
        assert_allclose(factor.solve(rhs), np.asarray(rhs, dtype=float), rtol=0, atol=0)


@pytest.mark.parametrize("solver", ["scipy", "cg"])
def test_extended_complex_rhs_preserves_original_components(solver: str) -> None:
    """Complex backends receive ordinary components while refinement retains both tails."""
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        return
    epsilon = np.longdouble(2) ** -60
    rhs = np.array([1 + epsilon + 1j * (2 + 2 * epsilon)], dtype=np.clongdouble)
    actual = solve_linear(
        [[1 + 0j]], rhs, solver=solver, rtol=1e-22, refinement_precision="extended"
    )
    assert_allclose(actual, rhs, rtol=0, atol=0)


def test_extended_complex_krylov_coordinate_rhs() -> None:
    """A complex coordinate-axis source retains its imaginary correction digits."""
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        return
    rhs = np.array([1j * (1 + np.longdouble(2) ** -60)], dtype=np.clongdouble)
    actual = solve_linear(
        [[1 + 0j]], rhs, solver="gmres", rtol=1e-22, refinement_precision="extended"
    )
    assert_allclose(actual, rhs, rtol=0, atol=0)


def test_factorization_reused_and_matrix_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    matrix = sparse.csr_matrix([[3.0, 1], [1, 4]])
    original = matrix.copy()
    count = 0
    real_splu = solvers.splinalg.splu

    def counting_splu(operator: Any) -> Any:
        nonlocal count
        count += 1
        return real_splu(operator)

    monkeypatch.setattr(solvers.splinalg, "splu", counting_splu)
    with factorize(matrix) as factor:
        matrix.data[:] = 0
        for rhs in (np.ones(2), np.eye(2), np.array([1j, 2 + 1j])):
            assert_allclose(original @ factor.solve(rhs), rhs, atol=1e-14)
    assert count == 1
    factor.close()
    with pytest.raises(RuntimeError, match="closed"):
        factor.solve(np.ones(2))
    with pytest.raises(RuntimeError, match="closed"):
        factor.__enter__()


@pytest.mark.parametrize(
    ("matrix", "rhs", "match"),
    [
        (np.zeros((0, 0)), [], "square"),
        (np.ones((2, 3)), [1, 2], "square"),
        ([[np.nan]], [1], "matrix entries"),
        ([[np.inf]], [1], "matrix entries"),
        ([[1]], [np.nan], "rhs entries"),
        ([[1]], [], "rhs must have shape"),
        ([[1]], 1, "rhs must have shape"),
        ([[1]], [[[1]]], "rhs must have shape"),
        ([[1]], np.zeros((1, 0)), "at least one column"),
    ],
)
def test_input_validation(matrix: Any, rhs: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_linear(matrix, rhs)


@pytest.mark.parametrize(("rtol", "atol"), [(-1, 0), (0, -1), (np.nan, 0), (1, np.inf), (0, 0)])
def test_bad_tolerances(rtol: float, atol: float) -> None:
    with pytest.raises(ValueError, match="tolerance|rtol"):
        solve_linear([[1]], [1], rtol=rtol, atol=atol)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_bad_maxiter(value: Any) -> None:
    with pytest.raises(ValueError, match="maxiter"):
        solve_linear([[1]], [1], maxiter=value)


def test_singular_and_unknown_solver() -> None:
    with pytest.raises(LinearSolveError, match="factorization"):
        solve_linear([[1, 1], [1, 1]], [1, 2])
    with pytest.raises(ValueError, match="Unsupported"):
        factorize([[1]], solver="unknown")
    with pytest.raises(ValueError, match="Unsupported"):
        factorize([[1]], solver="cupy")


@pytest.mark.parametrize("solver", ["cg", "minres"])
def test_symmetry_required(solver: str) -> None:
    with pytest.raises(ValueError, match="Hermitian"):
        solve_linear([[1, 1], [0, 1]], [1, 1], solver=solver)


@pytest.mark.parametrize(("matrix", "rhs"), [([[1j]], [1]), ([[1]], [1j])])
def test_minres_rejects_complex(matrix: Any, rhs: Any) -> None:
    with pytest.raises(ValueError, match="real"):
        solve_linear(matrix, rhs, solver="minres")


def test_minres_indefinite_zero_and_absolute_tolerance() -> None:
    matrix = [[1.0, 0], [0, -2]]
    assert_allclose(solve_linear(matrix, [1, 2], solver="minres"), [1, -1])
    assert_allclose(solve_linear(matrix, [0, 0], solver="minres"), [0, 0])
    assert_allclose(solve_linear(matrix, [1, 2], solver="minres", rtol=0, atol=1e-12), [1, -1])


def test_iteration_failure_and_bad_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(LinearSolveError, match="converge"):
        solve_linear(np.diag([1.0, 2.0, 4.0]), np.ones(3), solver="cg", maxiter=1)
    monkeypatch.setattr(solvers.splinalg, "gmres", lambda *args, **kwargs: (np.zeros(2), 0))
    with pytest.raises(LinearSolveError, match="residual"):
        solve_linear(np.eye(2), np.ones(2), solver="gmres")


@pytest.mark.parametrize("bad", [np.array([np.inf]), np.array([1, 2])])
def test_nonfinite_or_wrong_shape_backend_result(bad: np.ndarray) -> None:
    with factorize([[1]]) as factor:
        factor._solve = lambda rhs: bad
        with pytest.raises(LinearSolveError, match="nonfinite solution|shape"):
            factor.solve([1])


def test_residual_is_checked_per_column() -> None:
    with factorize(np.eye(2)) as factor:
        factor._solve = lambda rhs: np.array([[1e10, 0.0], [1e10, 0.0]])
        with pytest.raises(LinearSolveError, match="residual"):
            factor.solve(np.array([[1e10, 1.0], [1e10, 1.0]]))


@pytest.mark.parametrize("exception", [ImportError("missing"), OSError("native library missing")])
def test_optional_import_actionable(monkeypatch: pytest.MonkeyPatch, exception: Exception) -> None:
    def unavailable(name: str) -> Any:
        raise exception

    monkeypatch.setattr(solvers, "import_module", unavailable)
    with pytest.raises(SolverUnavailableError, match="Install pypardiso"):
        factorize([[1]], solver="pypardiso")


def test_cuda_initialization_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0))
    )
    monkeypatch.setattr(solvers, "import_module", lambda name: fake_cupy)
    with pytest.raises(SolverUnavailableError, match="visible CUDA"):
        solve_linear([[1]], [1], solver="cupy")

    def failed_count() -> int:
        raise RuntimeError("driver mismatch")

    fake_cupy.cuda.runtime.getDeviceCount = failed_count
    with pytest.raises(SolverUnavailableError, match="initialization failed"):
        solve_linear([[1]], [1], solver="cupy")


@pytest.fixture
def simulated_backends(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Simulate documented APIs only; these tests do not validate native solvers."""
    state = SimpleNamespace(
        factors=0,
        plans=0,
        solves=0,
        frees=0,
        reason=1,
        scalar=np.float64,
        mumps=True,
        pc_type=None,
        factor_solver=None,
        setup_error=None,
        matrix_options={},
        current_device=0,
        reset_shapes=[],
        solve_devices=[],
        free_devices=[],
    )

    class DirectEngine:
        """Contract fake shared by the PARDISO and nvmath interfaces."""

        def __init__(self, matrix: Any = None, rhs: Any = None, *, mtype: int = 11) -> None:
            self.matrix = matrix
            self.rhs = rhs
            self.mtype = mtype
            self.iparm = {}
            self.plan_config = SimpleNamespace(matching_algorithm=None)
            self.solution_config = SimpleNamespace(ir_num_steps=0)
            state.engine = self

        def set_iparm(self, index: int, value: int) -> None:
            """Record the documented one-based PARDISO parameter interface."""
            self.iparm[index] = value

        def factorize(self, matrix: Any = None) -> None:
            state.factors += 1
            if matrix is not None:
                self.matrix = matrix

        def plan(self) -> None:
            state.plans += 1

        def solve(self, matrix: Any = None, rhs: Any = None) -> Any:
            state.solves += 1
            state.solve_devices.append(state.current_device)
            if matrix is not None:
                self.matrix, self.rhs = matrix, rhs
            operator = self.matrix.toarray()
            if self.mtype == -2:
                operator = operator + np.triu(operator, 1).T
            return np.linalg.solve(operator, self.rhs)

        def reset_operands(self, *, b: Any) -> None:
            self.rhs = b
            state.reset_shapes.append((b.shape, b.flags.f_contiguous))

        def free(self) -> None:
            state.frees += 1
            state.free_devices.append(state.current_device)

        def free_memory(self, *, everything: bool) -> None:
            assert everything
            self.free()

    class Matrix:
        """Minimal PETSc matrix API contract fake."""

        Option = SimpleNamespace(SYMMETRIC="symmetric", SPD="spd")

        def setOption(self, option: str, value: bool) -> None:
            """Record the distinction between symmetry and positive definiteness."""
            state.matrix_options[option] = value

        def createAIJ(self, *, size: Any, csr: Any, comm: Any) -> Matrix:
            self.array = sparse.csr_matrix((csr[2], csr[1], csr[0]), shape=size).toarray()
            state.petsc_operator = self.array
            return self

        def assemble(self) -> None:
            pass

        def destroy(self) -> None:
            state.frees += 1

    class Vector:
        """Minimal PETSc vector API contract fake."""

        def createWithArray(self, array: Any, *, comm: Any) -> Vector:
            self.array = array
            return self

        def duplicate(self) -> Vector:
            return Vector().createWithArray(np.zeros_like(self.array), comm=None)

        def getArray(self, *, readonly: bool) -> np.ndarray:
            assert readonly
            return self.array

        def destroy(self) -> None:
            state.frees += 1

    class KSP:
        """Minimal sequential PETSc LU API contract fake."""

        def create(self, *, comm: Any) -> KSP:
            return self

        def setOperators(self, matrix: Matrix) -> None:
            self.matrix = matrix

        def setType(self, kind: str) -> None:
            assert kind == "preonly"

        def getPC(self) -> Any:
            return SimpleNamespace(
                setType=lambda kind: setattr(state, "pc_type", kind),
                setFactorSolverType=lambda kind: setattr(state, "factor_solver", kind),
            )

        def setUp(self) -> None:
            state.factors += 1
            if state.setup_error is not None:
                raise state.setup_error

        def solve(self, rhs: Vector, result: Vector) -> None:
            state.solves += 1
            result.array[:] = np.linalg.solve(self.matrix.array, rhs.array)

        def getConvergedReason(self) -> int:
            return state.reason

        def destroy(self) -> None:
            state.frees += 1

    @contextmanager
    def device_context(device: int):
        """Simulate CUDA's thread-local current-device restoration."""
        previous = state.current_device
        state.current_device = device
        try:
            yield
        finally:
            state.current_device = previous

    cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                getDeviceCount=lambda: 2, getDevice=lambda: state.current_device
            ),
            Device=device_context,
        ),
        zeros=np.zeros,
        asarray=np.asarray,
        asnumpy=np.asarray,
    )
    petsc = SimpleNamespace(
        Mat=Matrix,
        Vec=Vector,
        KSP=KSP,
        IntType=np.int32,
        ScalarType=state.scalar,
        COMM_SELF="self",
        Sys=SimpleNamespace(hasExternalPackage=lambda name: name == "mumps" and state.mumps),
    )
    modules = {
        "cupy": cupy,
        "cupyx.scipy.sparse": SimpleNamespace(csr_matrix=sparse.csr_matrix),
        "cupyx.scipy.sparse.linalg": SimpleNamespace(
            spsolve=lambda a, b: np.linalg.solve(a.toarray(), b)
        ),
        "nvmath.sparse.advanced": SimpleNamespace(
            DirectSolver=DirectEngine,
            DirectSolverMatchingAlg=SimpleNamespace(MAX_DIAG_PRODUCT=5),
        ),
        "pypardiso": SimpleNamespace(PyPardisoSolver=DirectEngine),
        "petsc4py.PETSc": petsc,
    }
    state.modules = modules
    monkeypatch.setattr(solvers, "import_module", modules.__getitem__)
    return state


@pytest.mark.parametrize("backend", ["pypardiso", "petsc", "petsc-symmetric", "cudss", "cupy"])
@pytest.mark.parametrize("multiple", [False, True])
def test_simulated_optional_backend_contract(
    backend: str, multiple: bool, simulated_backends: Any
) -> None:
    matrix = np.array([[3.0, 1], [1, 2]])
    expected = np.eye(2) if multiple else np.ones(2)
    assert_allclose(solve_linear(matrix, matrix @ expected, solver=backend), expected, atol=1e-14)
    if backend == "cudss":
        assert simulated_backends.engine.plan_config.matching_algorithm == 5
        assert simulated_backends.engine.solution_config.ir_num_steps == 5
        assert simulated_backends.solves == 1
    if backend.startswith("petsc"):
        assert simulated_backends.pc_type == ("cholesky" if backend.endswith("symmetric") else "lu")
        assert simulated_backends.matrix_options == (
            {"symmetric": True, "spd": False} if backend.endswith("symmetric") else {}
        )
        assert simulated_backends.factor_solver == "mumps"
    if backend != "cupy":
        assert simulated_backends.factors == 1
        assert simulated_backends.frees > 0


@pytest.mark.parametrize("backend", ["pypardiso", "petsc", "petsc-symmetric", "cudss"])
def test_simulated_repeated_factorization_contract(backend: str, simulated_backends: Any) -> None:
    with factorize(np.eye(2), solver=backend) as factor:
        for rhs in (np.ones(2), np.eye(2)):
            assert_allclose(factor.solve(rhs), rhs)
    assert simulated_backends.factors == 1


def test_cudss_response_columns_share_one_solve_and_owning_device(simulated_backends: Any) -> None:
    """Batch all basis RHS, reuse factors for other widths, and release on owner."""
    state = simulated_backends
    matrix = np.array([[2.0, -1.0, 1.0], [-1.0, 2.0, 1.0], [1.0, 1.0, 0.0]])
    expected = np.arange(15.0).reshape(3, 5)
    with factorize(matrix, solver="cudss", rhs_columns=5) as factor:
        state.current_device = 1
        assert_allclose(factor.solve(matrix @ expected), expected, atol=1e-13)
        assert state.solves == 1
        assert state.current_device == 1
        for width in (1, 3, 7):
            exact = np.arange(3.0 * width).reshape(3, width)
            assert_allclose(factor.solve(matrix @ exact), exact, atol=1e-13)
        assert_allclose(factor.solve(matrix @ np.ones(3)), 1.0, atol=1e-13)
    assert state.factors == 1
    assert state.solves == 6
    assert state.reset_shapes == [((3, 5), True)] * 6
    assert state.solve_devices == [0] * 6
    assert state.free_devices == [0]
    assert state.current_device == 1


@pytest.fixture
def concurrent_cudss_contract(
    simulated_backends: SimpleNamespace,
) -> SimpleNamespace:
    """Instrument native host entries with thread-local devices and separate transfers."""
    state = simulated_backends
    current = threading.local()
    native_entry_lock = threading.Lock()
    state.native_calls = []
    state.failure_phase = None
    state.transfer_barrier = None
    base_engine = state.modules["nvmath.sparse.advanced"].DirectSolver

    @contextmanager
    def native_entry(phase: str) -> Iterator[None]:
        """Reject overlapping native entrypoints and inject a one-time phase failure."""
        assert native_entry_lock.acquire(blocking=False), "cuDSS native host entries overlap"
        try:
            state.native_calls.append((phase, getattr(current, "device", 0)))
            # Release the GIL while a simulated native call remains active. Both
            # workers start together; an unprotected owner cannot rely on the GIL.
            time.sleep(0.002)
            yield
            if state.failure_phase == phase:
                state.failure_phase = None
                raise RuntimeError(f"injected cuDSS {phase} failure")
        finally:
            native_entry_lock.release()

    class CheckedConfig(SimpleNamespace):
        """Include native configuration writes in the host-entry contract."""

        def __setattr__(self, name: str, value: Any) -> None:
            """Record configuration mutation independently of engine construction."""
            with native_entry("configuration"):
                super().__setattr__(name, value)

    class CheckedEngine(base_engine):
        """Reuse the optional-backend fake while checking every native entrypoint."""

        def __init__(self, matrix: Any, rhs: Any) -> None:
            """Record the device that owns this engine's independent operands."""
            with native_entry("create"):
                super().__init__(matrix, rhs)
                self.owner = getattr(current, "device", 0)
                self.plan_config = CheckedConfig()
                self.solution_config = CheckedConfig()

        def plan(self) -> None:
            """Record synchronous native analysis without sharing an engine."""
            with native_entry("plan"):
                super().plan()

        def factorize(self) -> None:
            """Record the numerical-factorization host launch."""
            with native_entry("factorize"):
                super().factorize()

        def reset_operands(self, *, b: Any) -> None:
            """Record native pointer updates for each response block."""
            with native_entry("reset"):
                assert getattr(current, "device", 0) == self.owner
                super().reset_operands(b=b)

        def solve(self) -> Any:
            """Record an asynchronous solve launch on the owning device."""
            with native_entry("solve"):
                assert getattr(current, "device", 0) == self.owner
                return super().solve()

        def free(self) -> None:
            """Record native cleanup after transfer or a failed phase."""
            with native_entry("free"):
                assert getattr(current, "device", 0) == self.owner
                super().free()

    @contextmanager
    def device_context(device: int) -> Iterator[None]:
        """Restore each worker's independent current-device state."""
        previous = getattr(current, "device", 0)
        current.device = device
        try:
            yield
        finally:
            current.device = previous

    def host_transfer(result: Any) -> np.ndarray:
        """Require both devices to enqueue a solve before either transfer finishes."""
        if state.transfer_barrier is not None:
            state.transfer_barrier.wait(timeout=5)
        return np.asarray(result)

    cupy = state.modules["cupy"]
    cupy.cuda.Device = device_context
    cupy.cuda.runtime.getDevice = lambda: getattr(current, "device", 0)
    cupy.asnumpy = host_transfer
    state.modules["nvmath.sparse.advanced"].DirectSolver = CheckedEngine
    return state


def test_cudss_serializes_host_entries_without_serializing_transfers(
    concurrent_cudss_contract: SimpleNamespace,
) -> None:
    """Native entrypoints are exclusive while separate device workflows can progress."""
    state = concurrent_cudss_contract
    state.transfer_barrier = threading.Barrier(2)
    ready = threading.Barrier(2)
    cupy = state.modules["cupy"]
    matrix = np.array([[2.0, -1.0, 1.0], [-1.0, 2.0, 1.0], [1.0, 1.0, 0.0]])
    expected = np.arange(15.0).reshape(3, 5)

    def worker(device: int) -> np.ndarray:
        """Construct on one device and restore the caller's device after each solve."""
        ready.wait(timeout=5)
        with cupy.cuda.Device(device):
            prepared = factorize(matrix, solver="cudss", rhs_columns=5)
        with cupy.cuda.Device(1 - device), prepared:
            actual = prepared.solve(matrix @ expected)
            assert cupy.cuda.runtime.getDevice() == 1 - device
        assert cupy.cuda.runtime.getDevice() == 0
        return actual

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, device) for device in (0, 1)]
        for future in futures:
            assert_allclose(future.result(timeout=10), expected, atol=1e-12)
    assert state.factors == state.plans == state.solves == state.frees == 2
    for phase in ("create", "configuration", "plan", "factorize", "reset", "solve", "free"):
        assert {device for name, device in state.native_calls if name == phase} == {0, 1}


@pytest.mark.parametrize("phase", ["configuration", "plan", "factorize", "reset", "solve", "free"])
def test_cudss_host_entry_failure_releases_lock_and_owns_cleanup(
    concurrent_cudss_contract: SimpleNamespace, phase: str
) -> None:
    """A failed native phase cannot retain the host-entry lock or another device's resources."""
    state = concurrent_cudss_contract
    state.failure_phase = phase
    cupy = state.modules["cupy"]
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    expected = np.arange(4.0).reshape(2, 2)
    with (
        cupy.cuda.Device(1),
        pytest.raises(RuntimeError, match=f"cuDSS {phase} failure"),
        factorize(matrix, solver="cudss", rhs_columns=2) as prepared,
    ):
        prepared.solve(matrix @ expected)
    assert state.frees == 1
    assert ("free", 1) in state.native_calls
    assert cupy.cuda.runtime.getDevice() == 0
    recovered = []
    errors = []

    def recover() -> None:
        """Use a different thread and device after the failed host call."""
        try:
            with factorize(matrix, solver="cudss", rhs_columns=2) as prepared:
                recovered.append(prepared.solve(matrix @ expected))
        except BaseException as exc:
            errors.append(exc)

    recovery = threading.Thread(target=recover, daemon=True)
    recovery.start()
    recovery.join(timeout=5)
    assert not recovery.is_alive(), "failed native phase retained the host-entry lock"
    assert not errors
    assert_allclose(recovered[0], expected, atol=1e-12)
    assert state.frees == 2
    assert ("free", 0) in state.native_calls


@pytest.mark.parametrize("width", [0, -1, True, 1.5, None])
def test_factorization_rejects_invalid_rhs_width(width: Any) -> None:
    """Reject invalid solve-width hints before optional resources are allocated."""
    with pytest.raises(ValueError, match="rhs_columns"):
        factorize([[1.0]], rhs_columns=width)


@pytest.mark.parametrize("backend", ["pypardiso", "petsc", "petsc-symmetric"])
def test_simulated_backend_rejects_complex_matrix(backend: str, simulated_backends: Any) -> None:
    with pytest.raises(ValueError, match="complex|real"):
        factorize([[1j]], solver=backend)


@pytest.mark.parametrize("backend", ["pypardiso", "petsc", "petsc-symmetric", "cudss"])
def test_simulated_backend_rejects_unsupported_complex_rhs(
    backend: str, simulated_backends: Any
) -> None:
    with (
        factorize([[1]], solver=backend) as factor,
        pytest.raises(ValueError, match="complex|real"),
    ):
        factor.solve([1j])


@pytest.mark.parametrize("backend", ["petsc", "cudss", "cupy"])
def test_simulated_complex_backend_contract(backend: str, simulated_backends: Any) -> None:
    simulated_backends.modules["petsc4py.PETSc"].ScalarType = np.complex128
    assert_allclose(solve_linear([[1j]], [1j], solver=backend), [1])


def test_simulated_petsc_failure_releases_resources(simulated_backends: Any) -> None:
    simulated_backends.reason = -1
    with pytest.raises(LinearSolveError, match="PETSc"):
        solve_linear([[1]], [1], solver="petsc")
    assert simulated_backends.frees == 4


def test_simulated_petsc_requires_mumps_without_fallback(simulated_backends: Any) -> None:
    simulated_backends.mumps = False
    with pytest.raises(SolverUnavailableError, match="requires MUMPS.*--download-mumps"):
        factorize([[1.0]], solver="petsc")
    assert simulated_backends.factors == simulated_backends.frees == 0


def test_simulated_petsc_setup_failure_releases_resources(simulated_backends: Any) -> None:
    simulated_backends.setup_error = RuntimeError("factor setup failed")
    with pytest.raises(RuntimeError, match="factor setup failed"):
        factorize([[0.0, 1.0], [1.0, 0.0]], solver="petsc")
    assert simulated_backends.frees == 2


def test_simulated_petsc_preserves_zero_diagonal_saddle(simulated_backends: Any) -> None:
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert_allclose(solve_linear(matrix, matrix @ np.ones(2), solver="petsc"), 1)
    assert_allclose(simulated_backends.petsc_operator, matrix, rtol=0, atol=0)
    assert simulated_backends.factor_solver == "mumps"
    assert simulated_backends.frees == 4


@pytest.mark.parametrize(
    ("backend", "module"), [("petsc", "petsc4py.PETSc"), ("pypardiso", "pypardiso")]
)
def test_native_optional_cpu_integration(backend: str, module: str) -> None:
    """Validate an actual native backend when it is installed in this environment."""
    pytest.importorskip(module)
    matrix = sparse.csr_matrix([[3.0, 1.0], [1.0, 2.0]])
    with factorize(matrix, solver=backend) as decomposition:
        for rhs in (np.array([1.0, 2.0]), np.eye(2)):
            assert_allclose(matrix @ decomposition.solve(rhs), rhs, atol=1e-12)


@pytest.mark.fem
@pytest.mark.parametrize("backend", ["petsc", "petsc-symmetric"])
@pytest.mark.parametrize("equilibration", ["none", "symmetric"])
def test_native_petsc_zero_diagonal_saddle_integration(backend: str, equilibration: str) -> None:
    """Exercise actual MUMPS pivoting with absent diagonal entries and reused factors."""
    pytest.importorskip("petsc4py.PETSc")
    matrix = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    original = matrix.copy()
    with factorize(matrix, solver=backend, equilibration=equilibration) as decomposition:
        for rhs in (np.array([1.0, 2.0]), np.eye(2)):
            result = decomposition.solve(rhs)
            assert_allclose(result, solve_linear(matrix, rhs), atol=1e-13)
            assert_allclose(matrix @ result, rhs, atol=1e-13)
    assert_allclose(matrix.toarray(), original.toarray(), rtol=0, atol=0)


@pytest.mark.fem
@pytest.mark.parametrize("formulation", ["primal", "mixed"])
@pytest.mark.parametrize(
    ("local_solver", "global_solver"),
    [
        ("petsc", "scipy"),
        ("scipy", "petsc"),
        ("petsc", "petsc"),
        ("petsc-symmetric", "petsc-symmetric"),
    ],
)
def test_native_petsc_darcy_saddle_integration(
    formulation: str, local_solver: str, global_solver: str
) -> None:
    """Compare source-driven local/global PETSc saddles with SciPy and macro balances."""
    pytest.importorskip("petsc4py.PETSc")
    mesh = TriangleMesh.unit_square(2)
    options = {
        "dirichlet": lambda points: (
            1 + points[:, 0] + 2 * points[:, 1] + points[:, 0] * (1 - points[:, 0])
        ),
        "source": 2.0,
        "formulation": formulation,
        "local_refinement": 4,
    }
    reference = solve_darcy(mesh, **options)
    result = solve_darcy(mesh, local_solver=local_solver, solver=global_solver, **options)
    for actual, expected in zip(result.hybrid.fields, reference.hybrid.fields, strict=True):
        assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    assert_allclose(result.hybrid.trace, reference.hybrid.trace, rtol=1e-10, atol=1e-11)
    assert_allclose(result.conservation_residuals(), 0, atol=1e-11)
    assert result.hybrid.residual < 1e-12


@pytest.mark.fem
@pytest.mark.parametrize("formulation", ["taylor-hood", "usfem"])
@pytest.mark.parametrize("drag", [0.0, 3.0])
def test_native_petsc_brinkman_saddle_and_pressure_gauge(formulation: str, drag: float) -> None:
    """Check mixed local matrices, Stokes kernels, and the global pressure gauge."""
    pytest.importorskip("petsc4py.PETSc")
    mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)

    def velocity(points: np.ndarray) -> np.ndarray:
        """Return a divergence-free rigid rotation."""
        return np.column_stack((points[:, 1], -points[:, 0]))

    options = {
        "dirichlet": velocity,
        "source": lambda points: drag * velocity(points) + [1.0, 2.0],
        "drag": drag,
        "formulation": formulation,
        "skeleton": skeleton,
        "local_refinement": 4,
    }
    reference = solve_brinkman(mesh, **options)
    result = solve_brinkman(mesh, solver="petsc", local_solver="petsc", **options)
    for actual, expected in zip(result.hybrid.fields, reference.hybrid.fields, strict=True):
        assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert_allclose(result.hybrid.trace, reference.hybrid.trace, rtol=1e-10, atol=1e-10)
    assert result.l2_error(velocity) < 1e-11
    assert result.pressure_l2_error(lambda points: points[:, 0] + 2 * points[:, 1] - 1.5) < 1e-10
    assert result.divergence_l2() < 1e-10
    assert result.hybrid.residual < 1e-12


@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cupy", "cudss"])
def test_native_optional_gpu_integration(backend: str) -> None:
    """Validate actual CUDA execution separately from simulated API contracts."""
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("No visible CUDA device")
    if backend == "cudss":
        pytest.importorskip("nvmath.sparse.advanced")
    matrix = sparse.csr_matrix([[3.0, 1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert_allclose(matrix @ solve_linear(matrix, rhs, solver=backend), rhs, atol=1e-12)


@pytest.mark.parametrize("complex_values", [False, True])
def test_native_pyamg_elliptic_solve(complex_values: bool) -> None:
    """Check the installed PyAMG hierarchy against the manufactured solution."""
    pyamg = pytest.importorskip("pyamg")
    matrix = pyamg.gallery.poisson((12, 12), format="csr")
    expected = np.column_stack((np.ones(matrix.shape[0]), np.linspace(0, 1, matrix.shape[0])))
    if complex_values:
        matrix = matrix.astype(complex)
        expected = expected * (1 + 1j)
    solution = solve_linear(
        matrix, matrix @ expected, solver="pyamg", near_nullspace=np.ones((matrix.shape[0], 1))
    )
    assert_allclose(solution, expected, atol=2e-9)
    assert_allclose(solve_linear([[2.0]], [2.0], solver="pyamg"), [1.0])


@pytest.mark.parametrize("backend", ["pyamg", "amgx"])
def test_amg_rejects_obvious_nonelliptic_operators(backend: str) -> None:
    with pytest.raises(ValueError, match="positive diagonals"):
        solve_linear([[1, 1], [1, 0]], [1, 2], solver=backend)
    with pytest.raises(ValueError, match="Hermitian"):
        solve_linear([[1, 1], [0, 2]], [1, 2], solver=backend)


def test_pyamg_candidate_validation() -> None:
    with pytest.raises(ValueError, match="only supported"):
        solve_linear([[1]], [1], near_nullspace=[[1]])
    with pytest.raises(ValueError, match="near_nullspace must have shape"):
        solve_linear([[1]], [1], solver="pyamg", near_nullspace=[1])
    with pytest.raises(ValueError, match="rhs must have shape"):
        solve_linear([[1]], [1], solver="pyamg", near_nullspace=[[1], [1]])


@pytest.fixture
def simulated_amgx(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Simulate AmgX API/resource ownership, without claiming GPU execution."""
    state = SimpleNamespace(
        created=[], destroyed=[], configuration=None, status="success", setups=0, solves=0, gain=1.0
    )

    class Resource:
        """Track a native handle and its destruction order."""

        def __init__(self, kind: str) -> None:
            self.kind = kind

        def create(self, *args: Any, mode: str) -> Resource:
            assert mode == "dDDI"
            state.created.append(self.kind)
            return self

        def create_from_dict(self, options: Any) -> Resource:
            state.configuration = options
            state.created.append(self.kind)
            return self

        def create_simple(self, config: Any) -> Resource:
            state.created.append(self.kind)
            return self

        def destroy(self) -> None:
            state.destroyed.append(self.kind)

        def upload_CSR(self, matrix: Any) -> None:
            self.matrix = matrix

        def setup(self, operator: Resource) -> None:
            self.matrix = operator.matrix
            state.setups += 1

        def upload(self, array: Any) -> None:
            self.array = array.copy()

        def download(self) -> np.ndarray:
            return self.array.copy()

        def solve(self, rhs: Resource, result: Resource, *, zero_initial_guess: bool) -> None:
            assert zero_initial_guess
            state.solves += 1
            # A native recursive stopping floor must not erase a tiny physical load.
            result.array = (
                state.gain * np.linalg.solve(self.matrix.toarray(), rhs.array)
                if np.linalg.norm(rhs.array) > 1e-20
                else np.zeros_like(rhs.array)
            )

        @property
        def status(self) -> str:
            return state.status

    module = SimpleNamespace(
        initialize=lambda: state.created.append("initialize"),
        finalize=lambda: state.destroyed.append("finalize"),
        Config=lambda: Resource("config"),
        Resources=lambda: Resource("resources"),
        Matrix=lambda: Resource("matrix"),
        Vector=lambda: Resource("vector"),
        Solver=lambda: Resource("solver"),
    )
    state.module = module
    monkeypatch.setattr(solvers, "import_module", lambda name: module)
    return state


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("absolute_only", [False, True])
def test_simulated_amgx_solve_contract(
    multiple: bool, absolute_only: bool, simulated_amgx: Any
) -> None:
    rhs = np.array([[1.0, 0.0], [2.0, 0.0]]) if multiple else np.array([1.0, 2.0])
    rtol, atol = (0.0, 1e-12) if absolute_only else (1e-10, 0.0)
    result = solve_linear(
        np.eye(2), rhs, solver="amgx", rtol=rtol, atol=atol, maxiter=30 if multiple else None
    )
    assert_allclose(result, rhs)
    assert simulated_amgx.setups == 1
    assert simulated_amgx.destroyed == [
        "vector",
        "vector",
        "solver",
        "matrix",
        "resources",
        "config",
        "finalize",
    ]
    configuration = simulated_amgx.configuration["solver"]
    assert configuration["solver"] == "FGMRES"
    assert configuration["preconditioner"]["solver"] == "AMG"
    assert configuration["convergence"] == ("ABSOLUTE" if absolute_only else "RELATIVE_INI")


def test_simulated_amgx_failure_cleanup(simulated_amgx: Any) -> None:
    simulated_amgx.status = "not_converged"
    with pytest.raises(LinearSolveError, match="AmgX did not converge"):
        solve_linear([[1]], [1], solver="amgx")
    assert simulated_amgx.destroyed[-1] == "finalize"


def test_simulated_amgx_small_loads_and_prepared_reuse(simulated_amgx: Any) -> None:
    """One fixed hierarchy preserves independently scaled and homogeneous loads."""
    matrix = np.array([[3.0, -1], [-1, 2]])
    vector = np.array([1.0, -2])
    rhs = (matrix @ vector)[:, None] * np.array([1.0, 1e-15, 1e-120, 0.0])
    with prepare_amgx(matrix) as prepared:
        assert_allclose(
            prepared.solve(rhs), vector[:, None] * [1, 1e-15, 1e-120, 0], rtol=1e-12, atol=0
        )
        assert_allclose(prepared.solve(rhs[:, 1]), vector * 1e-15, rtol=1e-12, atol=0)
        assert prepared.matches(matrix)
    assert simulated_amgx.setups == 1
    assert simulated_amgx.solves == 4
    assert simulated_amgx.destroyed[-1] == "finalize"


def test_simulated_amgx_reuses_hierarchy_for_true_residual_corrections(
    simulated_amgx: Any,
) -> None:
    """Native convergence cannot replace the unchanged original-equation criterion."""
    simulated_amgx.gain = 0.999
    matrix = np.array([[3.0, -1], [-1, 2]])
    rhs = matrix @ np.array([1.0, -2])
    with prepare_amgx(matrix, rtol=1e-13) as prepared:
        with pytest.raises(LinearSolveError, match="residual criterion"):
            prepared.solve(rhs * 1e-15, refinement_steps=3)
        result = prepared.solve(rhs * 1e-15, refinement_steps=4)
        assert_allclose(result, np.array([1.0, -2]) * 1e-15, rtol=2e-14, atol=0)
    assert simulated_amgx.setups == 1
    assert simulated_amgx.solves == 9


@pytest.mark.parametrize("maxiter", [False, 0, -1, 1.5])
def test_prepared_amgx_rejects_invalid_iteration_limit(maxiter: Any) -> None:
    """Invalid limits fail before loading a native accelerator."""
    with pytest.raises(ValueError, match="maxiter must"):
        prepare_amgx(np.eye(2), maxiter=maxiter)


@pytest.mark.parametrize("matrix", [[[1, 2], [0, 1]], [[-1, 0], [0, 1]]])
def test_prepared_amgx_requires_elliptic_operator(matrix: Any) -> None:
    """Nonsymmetric and saddle-like diagonals fail before native setup."""
    with pytest.raises(ValueError, match="symmetric|positive diagonals"):
        prepare_amgx(matrix)


def test_simulated_amgx_initialization_error(simulated_amgx: Any) -> None:
    def fail_initialize() -> None:
        raise RuntimeError("missing device")

    simulated_amgx.module.initialize = fail_initialize
    with pytest.raises(SolverUnavailableError, match="initialization failed"):
        solve_linear([[1]], [1], solver="amgx")
    assert simulated_amgx.destroyed == []


@pytest.mark.parametrize(("matrix", "rhs"), [([[1 + 0j]], [1]), ([[1]], [1j])])
def test_amgx_rejects_complex(matrix: Any, rhs: Any) -> None:
    with pytest.raises(ValueError, match="real float64"):
        solve_linear(matrix, rhs, solver="amgx")


def test_simulated_prepared_amgx_rejects_complex_rhs(simulated_amgx: Any) -> None:
    """Prepared real operators retain the same scalar-type contract on later loads."""
    with prepare_amgx(np.eye(2)) as prepared, pytest.raises(ValueError, match="real float64"):
        prepared.solve([1j, 0])


@pytest.mark.serial
@pytest.mark.gpu
def test_native_optional_amgx_integration() -> None:
    """Exercise native GPU AMG when an AmgX/pyamgx installation is provided."""
    pytest.importorskip("pyamgx")
    matrix = sparse.diags([-np.ones(99), 2 * np.ones(100), -np.ones(99)], [-1, 0, 1], format="csr")
    rhs = np.ones(100)
    assert_allclose(matrix @ solve_linear(matrix, rhs, solver="amgx"), rhs, atol=1e-9)


@pytest.mark.serial
@pytest.mark.gpu
def test_native_amgx_scaled_rhs_original_residual_and_reuse() -> None:
    """Native AMG preserves tiny loads, zero loads and fixed-operator true corrections."""
    pytest.importorskip("pyamgx")
    matrix = sparse.diags([-np.ones(99), 2 * np.ones(100), -np.ones(99)], [-1, 0, 1], format="csr")
    expected = np.linspace(-1, 1, 100)
    rhs = (matrix @ expected)[:, None] * np.array([1.0, 1e-15, 1e-120, 0.0])
    with prepare_amgx(matrix) as prepared:
        result = prepared.solve(rhs)
        assert_allclose(result, expected[:, None] * [1, 1e-15, 1e-120, 0], rtol=1e-8, atol=0)
        assert_allclose(prepared.solve(rhs[:, 1]), result[:, 1], rtol=1e-8, atol=0)


@pytest.mark.serial
@pytest.mark.gpu
def test_native_cudss_indefinite_multiple_rhs() -> None:
    """Verify a zero-diagonal saddle block with cuDSS matching and refinement."""
    pytest.importorskip("cupy")
    pytest.importorskip("nvmath.sparse.advanced")
    matrix = sparse.csr_matrix([[2.0, -1, 1], [-1, 2, 1], [1, 1, 0]])
    expected = np.column_stack((np.arange(1.0, 4.0), [-1.0, 0.5, 2.0]))
    assert_allclose(solve_linear(matrix, matrix @ expected, solver="cudss"), expected, atol=1e-12)


@pytest.mark.serial
@pytest.mark.gpu
def test_native_cudss_matrix_rhs_and_device_restoration() -> None:
    """Exercise one full matrix RHS, padded reuse and cleanup across CUDA devices."""
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("nvmath.sparse.advanced")
    if cupy.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("two CUDA devices required")
    matrix = sparse.csr_matrix([[2.0, -1.0, 1.0], [-1.0, 2.0, 1.0], [1.0, 1.0, 0.0]])
    with cupy.cuda.Device(0):
        factor = factorize(matrix, solver="cudss", rhs_columns=5)
    with cupy.cuda.Device(1):
        with factor:
            for width in (5, 1, 7):
                exact = np.arange(3.0 * width).reshape(3, width)
                assert_allclose(factor.solve(matrix @ exact), exact, atol=1e-12)
                assert cupy.cuda.runtime.getDevice() == 1
        assert cupy.cuda.runtime.getDevice() == 1


@pytest.mark.parametrize("format", ["csr", "csc", "coo", "lil", "dok", "dia", "bsr"])
def test_all_sparse_formats_preserve_complex_values(format: str) -> None:
    matrix = sparse.csr_matrix([[3, 1j], [-1j, 2]])
    expected = np.array([1 + 2j, -2j])
    assert_allclose(solve_linear(matrix.asformat(format), matrix @ expected), expected)


@pytest.mark.parametrize("scale", [1e200, 1e-250])
def test_residual_check_avoids_norm_overflow_and_underflow(scale: float) -> None:
    with factorize(np.eye(2)) as factors:
        factors._solve = lambda rhs: np.zeros(2)
        with pytest.raises(LinearSolveError, match="residual criterion"):
            factors.solve([scale, scale])


def test_nonfinite_residual_is_rejected() -> None:
    with factorize([[1e200]]) as factors:
        factors._solve = lambda rhs: np.array([1e200])
        with pytest.raises(LinearSolveError, match="nonfinite values"):
            factors.solve([1e200])


@pytest.mark.parametrize("matrix", [[[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [2.0, 0.0]]])
def test_zero_rows_or_columns_rejected(matrix: Any) -> None:
    with pytest.raises(LinearSolveError, match="zero row|zero column"):
        solvers.validate_invertible(matrix)


def test_numerical_rank_rejected_even_with_compatible_rhs() -> None:
    matrix = np.array([[1.0, 1.0], [1.0, 1.0 + np.finfo(float).eps]])
    with pytest.raises(LinearSolveError, match="numerical rank"):
        solve_linear(matrix, matrix @ np.ones(2))


def test_row_column_equilibration_preserves_scaled_invertible_system() -> None:
    matrix = np.diag([1e-200, 1e200])
    solvers.validate_invertible(matrix)
    assert_allclose(solve_linear(matrix, [1e-200, 1e200]), [1, 1])
    base = np.array([[3.0, 1.0], [1.0, 2.0]])
    matrix = np.diag([1e-20, 1e20]) @ base @ np.diag([1e-10, 1e10])
    expected = np.array([1e10, 1e-10])
    assert_allclose(solve_linear(matrix, matrix @ expected), expected, rtol=1e-13)


@pytest.mark.parametrize("complex_rhs", [False, True])
def test_refinement_reuses_backend_and_preserves_columnwise_tolerance(complex_rhs: bool) -> None:
    """A correction fixes a finite inaccurate first solve without changing its tolerance."""
    matrix = sparse.csr_matrix([[3.0, 1.0], [1.0, 2.0]])
    expected = np.array([[1.0, 2.0], [-3.0, 4.0]]) * (1 + 2j if complex_rhs else 1)
    rhs = matrix @ expected
    with factorize(matrix, rtol=1e-13) as factors:
        backend = factors._solve
        calls = []

        def inaccurate_first(forcing: Any) -> Any:
            """Perturb only the first call; later calls use the existing native factor."""
            calls.append(np.array(forcing, copy=True))
            result = backend(forcing)
            return result + 1e-5 if len(calls) == 1 else result

        factors._solve = inaccurate_first
        assert_allclose(factors.solve(rhs), expected, rtol=0, atol=2e-15)
        assert len(calls) == 2
        assert np.max(np.abs(calls[1])) < 1e-3


@pytest.mark.parametrize("method", ["cg", "gmres", "minres", "pyamg"])
def test_krylov_true_residual_correction(monkeypatch: Any, method: str) -> None:
    """Correct a premature recursive-residual stop using the original physical residual."""
    if method == "pyamg":
        pytest.importorskip("pyamg")
    matrix = sparse.csr_matrix([[3.0, 1.0], [1.0, 2.0]])
    exact = np.array([1.0, -3.0])
    calls = []

    def premature(operator: Any, rhs: Any, **kwargs: Any) -> tuple[np.ndarray, int]:
        """Simulate recurrence drift on the first call and exact residual correction."""
        calls.append((np.array(rhs, copy=True), kwargs.get("M")))
        result = np.linalg.solve(operator.toarray(), rhs)
        return result + (1e-5 if len(calls) == 1 else 0), 0

    monkeypatch.setattr(solvers.splinalg, "cg" if method == "pyamg" else method, premature)
    assert_allclose(solve_linear(matrix, matrix @ exact, solver=method), exact, atol=2e-15)
    assert len(calls) == 2
    assert np.max(np.abs(calls[1][0])) < 1e-3
    if method == "pyamg":
        assert calls[0][1] is calls[1][1]


def test_krylov_failed_correction_is_rejected(monkeypatch: Any) -> None:
    """Repeated inaccurate accepted backend iterates never bypass the residual gate."""
    calls = []

    def stalled(operator: Any, rhs: Any, **kwargs: Any) -> tuple[np.ndarray, int]:
        """Return a finite but uncorrected iterate claiming backend convergence."""
        calls.append(1)
        return np.zeros_like(rhs), 0

    monkeypatch.setattr(solvers.splinalg, "cg", stalled)
    with pytest.raises(LinearSolveError, match="residual criterion failed"):
        solve_linear([[1.0]], [1.0], solver="cg")
    assert len(calls) == 3


def test_extended_krylov_refinement_preserves_solution_precision(monkeypatch: Any) -> None:
    """Mixed-precision corrections retain digits required by the original residual gate."""
    if not solvers._EXTENDED_PRECISION:
        pytest.skip("NumPy longdouble is not wider than double on this platform")
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0 + 1e-8]])
    rhs = np.array([0.3, -0.3 + 1e-8])

    def correction(operator: Any, column: Any, **kwargs: Any) -> tuple[np.ndarray, int]:
        """Use double-precision solves for both original and residual systems."""
        assert column.dtype == np.float64
        return np.linalg.solve(operator.toarray(), column), 0

    monkeypatch.setattr(solvers.splinalg, "cg", correction)
    result = solve_linear(matrix, rhs, solver="cg", rtol=1e-17, refinement_precision="extended")
    assert result.dtype == np.longdouble
    residual = matrix.astype(np.longdouble) @ result - rhs.astype(np.longdouble)
    assert np.linalg.norm(residual) <= 1e-17 * np.linalg.norm(rhs)
    with pytest.raises(LinearSolveError, match="residual criterion"):
        solve_linear(matrix, rhs, solver="cg", rtol=1e-17)


def test_extended_refinement_validates_backend_and_platform(monkeypatch: Any) -> None:
    """The explicit precision contract rejects unsupported modes without changing tolerance."""
    with pytest.raises(ValueError, match="double or extended"):
        solve_linear([[1]], [1], refinement_precision="quadruple")
    with pytest.raises(ValueError, match="Krylov or reusable direct backend"):
        solve_linear([[1]], [1], solver="cupy", refinement_precision="extended")
    monkeypatch.setattr(solvers, "_EXTENDED_PRECISION", False)
    with pytest.raises(solvers.SolverUnavailableError, match="wider long-double"):
        solve_linear([[1]], [1], solver="cg", refinement_precision="extended")


@pytest.mark.parametrize("bad", [np.array([np.inf]), np.array([1.0, 2.0])])
def test_refinement_rejects_invalid_correction(bad: np.ndarray) -> None:
    """Correction outputs must obey the same finite shape contract as initial solves."""
    with factorize([[1.0]]) as factors:
        calls = 0

        def inaccurate_then_invalid(rhs: Any) -> Any:
            """Trigger one refinement and return malformed correction data."""
            nonlocal calls
            calls += 1
            return np.array([0.9]) if calls == 1 else bad

        factors._solve = inaccurate_then_invalid
        with pytest.raises(LinearSolveError, match="nonfinite solution|shape"):
            factors.solve([1.0])
        assert calls == 2


@pytest.mark.parametrize(
    ("backend", "module"),
    [
        ("scipy", "scipy"),
        pytest.param("petsc", "petsc4py.PETSc", marks=pytest.mark.fem),
        pytest.param("petsc-symmetric", "petsc4py.PETSc", marks=pytest.mark.fem),
        ("pypardiso", "pypardiso"),
        ("pypardiso-symmetric", "pypardiso"),
        ("pypardiso-symmetric-matching", "pypardiso"),
        pytest.param(
            "cudss", "nvmath.sparse.advanced", marks=[pytest.mark.gpu, pytest.mark.serial]
        ),
    ],
)
def test_high_contrast_rt0_neumann_lifts_preserve_strict_residuals(
    backend: str, module: str
) -> None:
    """A real mixed saddle with a fitted 10000-fold jump needs iterative refinement."""
    pytest.importorskip(module)
    grid = TriangleMesh.unit_square(10)
    mesh = TriangleMesh(grid.points, grid.cells[[143]])

    def permeability(points: Any) -> Any:
        """The horizontal interface cuts the macro but aligns with its fine edges."""
        return np.where(points[:, 1] < 0.75, 1e-4, 1.0)

    solution = solve_darcy(
        mesh,
        permeability=permeability,
        dirichlet=lambda points: points[:, 0],
        formulation="mixed",
        local_refinement=4,
        quadrature_order=6,
        local_solver=backend,
        solver=backend,
    )
    assert np.isfinite(solution.pressure[0]).all()
    assert_allclose(solution.conservation_residuals(), 0, atol=1e-13)
    assert_allclose(solution.fine_conservation_residuals()[0], 0, atol=1e-13)


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("backend", ["pypardiso-symmetric", "pypardiso-symmetric-matching"])
def test_symmetric_pardiso_triangle_and_original_residual(multiple, backend, simulated_backends):
    """Zero saddle diagonals are stored; lower-triangle roundoff stays in the residual."""
    matrix = np.array([[2.0, 0, 1], [0, 3, 1], [1 + 1e-13, 1, 0]])
    expected = np.column_stack(([1.0, 2, 3], [-1.0, 1, 2])) if multiple else np.array([1.0, 2, 3])
    with factorize(matrix, solver=backend, rtol=1e-15) as factor:
        actual = factor.solve(matrix @ expected)
        assert_allclose(actual, expected, atol=1e-13)
        assert factor.matches(matrix)
        engine = simulated_backends.engine
        assert engine.mtype == -2
        assert engine.iparm == (
            {1: 1, 2: 2, 8: 5, 10: 13, 11: 1, 13: 1, 21: 1, 27: 1}
            if backend.endswith("-matching")
            else {}
        )
        assert np.all(np.diff(engine.matrix.indptr) >= 1)
        assert engine.matrix[-1, -1] == 0
        assert sparse.tril(engine.matrix, -1).nnz == 0
        assert simulated_backends.solves >= 2
    assert simulated_backends.frees == 1


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
@pytest.mark.parametrize(
    "solver", ["cg", "minres", "pyamg", "pypardiso-symmetric", "petsc-symmetric"]
)
def test_symmetry_requirement_is_invariant_under_physical_scaling(
    scale, solver, simulated_backends
):
    """Small absolute entries cannot make a materially nonsymmetric operator admissible."""
    matrix = scale * np.array([[2.0, 1], [0.5, 2]])
    with pytest.raises(ValueError, match="Hermitian|symmetric"):
        solve_linear(matrix, scale * np.ones(2), solver=solver)


def test_symmetric_pardiso_extended_accumulation(simulated_backends):
    """The explicit saddle backend participates in the common refinement policy."""
    if np.finfo(np.longdouble).eps == np.finfo(float).eps:
        pytest.skip("platform has no wider accumulator")
    matrix = np.array([[2.0, 1], [1, 0]])
    result = solve_linear(
        matrix, [3.0, 1.0], solver="pypardiso-symmetric", refinement_precision="extended"
    )
    assert_allclose(result, [1, 1], atol=1e-15)


@pytest.mark.parametrize("complex_values", [False, True])
def test_residual_accumulation_distinguishes_cancellation_from_solver_error(
    complex_values: bool,
) -> None:
    """Recheck a cancellation-dominated dot product at higher accumulation precision."""
    dtype = np.complex64 if complex_values else np.float32
    matrix = sparse.csr_matrix([[1e8, 1, -1e8], [1, 0, 0], [0, 0, 1]], dtype=dtype)
    solution = np.ones(3, dtype=dtype) * (1 + 1j if complex_values else 1)
    rhs = solution.copy()
    assert np.linalg.norm(matrix @ solution - rhs) > 0.5
    assert_allclose(solvers._checked(matrix, rhs, solution, 1e-10, 0), solution, atol=0)
    with pytest.raises(LinearSolveError, match="residual criterion"):
        solvers._checked(matrix, rhs, solution + 1e-4, 1e-10, 0)


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("multiple_rhs", [False, True])
def test_compensated_residual_on_platform_without_extended_precision(
    monkeypatch: pytest.MonkeyPatch, complex_values: bool, multiple_rhs: bool
) -> None:
    """Simulate Windows long-double precision without weakening the residual criterion."""
    monkeypatch.setattr(solvers, "_EXTENDED_PRECISION", False)
    matrix = sparse.csr_matrix([[1e16, 1, -1e16], [1, 0, 0], [0, 0, 1]])
    solution = np.ones((3, 2) if multiple_rhs else 3) * (1 + 2j if complex_values else 1)
    rhs = solution.copy()
    assert np.linalg.norm(matrix @ solution - rhs) > 0.5
    assert_allclose(solvers._checked(matrix, rhs, solution, 1e-10, 0), solution, atol=0)
    with pytest.raises(LinearSolveError, match="residual criterion"):
        solvers._checked(matrix, rhs, solution + 1e-4, 1e-10, 0)


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("complex_values", [False, True])
def test_symmetric_equilibration_preserves_equations_and_factor_reuse(
    multiple: bool, complex_values: bool
) -> None:
    """A Hermitian congruence preserves saddle solutions and the original operator."""
    core = np.array([[3, 1, 2], [1, 2, -1], [2, -1, 0]], dtype=complex if complex_values else float)
    if complex_values:
        core[0, 1], core[1, 0] = 1j, -1j
    units = np.array([1e-4, 1.0, 1e4])
    matrix = units[:, None] * core * units[None, :]
    original = matrix.copy()
    expected = np.array([0.7, -2.0, 1.5]) / units
    if complex_values:
        expected = expected * (1 + 0.2j)
    if multiple:
        expected = expected[:, None] * np.array([1.0, -0.25])
    rhs = matrix @ expected
    with factorize(matrix, equilibration="symmetric", rtol=1e-12) as factors:
        assert factors.matches(original)
        matrix[:] = 0
        assert_allclose(factors.solve(rhs), expected, rtol=2e-13)
        assert_allclose(factors.solve(2 * rhs), 2 * expected, rtol=2e-13)
    assert_allclose(
        solve_linear(original, rhs, equilibration="symmetric", rtol=1e-12),
        expected,
        rtol=2e-13,
    )


def test_equilibration_checks_original_not_balanced_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small residual after unit scaling cannot mask an unsatisfied physical row."""
    matrix = sparse.diags([1e-12, 1e12], format="csr")
    actual_splu = solvers.splinalg.splu

    def inaccurate_factor(operator: Any) -> Any:
        """Return a factor that systematically loses the second equation."""
        engine = actual_splu(operator)

        def solve(rhs: np.ndarray) -> np.ndarray:
            """Solve the first row only, even during correction solves."""
            result = engine.solve(rhs)
            result[1] = 0
            return result

        return SimpleNamespace(U=engine.U, solve=solve)

    monkeypatch.setattr(solvers.splinalg, "splu", inaccurate_factor)
    with pytest.raises(LinearSolveError, match="residual criterion"):
        solve_linear(matrix, np.ones(2), equilibration="symmetric")


@pytest.mark.parametrize("api", [factorize, solve_linear])
def test_equilibration_rejects_invalid_mode(api: Any) -> None:
    """Unknown modes are rejected before a backend is constructed."""
    arguments = ([[1.0]], [1.0]) if api is solve_linear else ([[1.0]],)
    with pytest.raises(ValueError, match="equilibration must"):
        api(*arguments, equilibration="automatic")


@pytest.mark.parametrize("backend", ["cg", "minres", "gmres", "pyamg", "amgx", "cupy"])
def test_equilibration_requires_direct_backend(backend: str) -> None:
    """An explicit direct-factor option does not silently change a Krylov method."""
    with pytest.raises(ValueError, match="reusable direct"):
        solve_linear([[1.0]], [1.0], solver=backend, equilibration="symmetric")


def test_equilibration_requires_hermitian_invertible_operator() -> None:
    """Congruence rejects asymmetric input and a structurally absent equation."""
    with pytest.raises(ValueError, match="Hermitian"):
        factorize([[1, 1], [0, 2]], equilibration="symmetric")
    with pytest.raises(LinearSolveError, match="zero row"):
        factorize([[1, 0], [0, 0]], equilibration="symmetric")


@pytest.mark.parametrize("backend", ["pypardiso-symmetric", "pypardiso-symmetric-matching"])
def test_native_pardiso_symmetric_equilibration(backend: str) -> None:
    """Native MKL factors solve a scaled mixed saddle block in its physical units."""
    pytest.importorskip("pypardiso")
    core = np.array([[2.0, 0.3, 1], [0.3, 3, -1], [1, -1, 0]])
    units = np.array([1e-4, 1e4, 1.0])
    matrix = units[:, None] * core * units[None, :]
    expected = np.array([1.0, -2.0, 3.0]) / units
    expected = np.column_stack((expected, -0.3 * expected))
    rhs = matrix @ expected
    with factorize(matrix, solver=backend, equilibration="symmetric", rtol=1e-12) as factor:
        result = factor.solve(rhs)
    assert_allclose(result, expected, rtol=2e-13)
    assert_allclose(matrix @ result, rhs, rtol=2e-13)


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("shape", ["wide", "tall", "empty_rows", "empty_columns"])
def test_rectangular_original_residual_compensated_fallback(
    monkeypatch: pytest.MonkeyPatch, multiple: bool, complex_values: bool, shape: str
) -> None:
    """Original Darcy volume/coupling rows and projections work without native LD."""
    monkeypatch.setattr(solvers, "_EXTENDED_PRECISION", False)
    matrix = np.array([[1e16, 1, -1e16, 0], [0, 2, 0, 1]])
    if shape == "tall":
        matrix = matrix.T
    elif shape == "empty_rows":
        matrix = matrix[:0]
    elif shape == "empty_columns":
        matrix = matrix[:, :0]
    x = np.ones(matrix.shape[1]) * (1 + 2j if complex_values else 1)
    b = np.arange(matrix.shape[0], dtype=float) * (1 + 2j if complex_values else 1)
    expected = np.array([sum(float(value) for value in row) for row in matrix])
    if shape == "wide":
        expected[0] = 1  # The exact sum is masked by ordinary float64 accumulation.
    expected = b - expected * (1 + 2j if complex_values else 1)
    if multiple:
        x = x[:, None] * np.array([1, -0.25])
        b = b[:, None] * np.array([1, -0.25])
        expected = expected[:, None] * np.array([1, -0.25])
    actual = solvers._accurate_residual(sparse.csr_matrix(matrix), b, x)
    assert_allclose(actual, expected, atol=0)
