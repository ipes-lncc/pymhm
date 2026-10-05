"""Checked sparse solvers with explicit, lazily loaded optional backends.

Every accepted solution satisfies the requested columnwise Euclidean residual
criterion. Optional backends never silently fall back to a different solver.
Factorizations use double precision (real or complex). Difficult residuals are
accumulated with extended precision or compensated sums before acceptance.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from importlib import import_module
from math import fsum
from threading import Lock
from types import ModuleType, TracebackType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import linalg as splinalg

Array = npt.NDArray[Any]
_AMGX_LOCK = Lock()
_CUDSS_HOST_LOCK = Lock()
_PARDISO_HOST_LOCK = Lock()
_EXTENDED_PRECISION = np.finfo(np.longdouble).eps < np.finfo(np.float64).eps


class LinearSolveError(RuntimeError):
    """A factorization, iteration, or independently checked residual failed."""


class SolverUnavailableError(ImportError):
    """An explicitly requested optional solver cannot be loaded or executed."""


class _ResidualFailure(LinearSolveError):
    """A finite backend result needs correction to satisfy the requested residual."""


def _matrix(matrix: Any) -> sparse.csr_matrix:
    """Copy a finite square operator to canonical double-precision CSR form."""
    is_complex = matrix.dtype.kind == "c" if sparse.issparse(matrix) else np.iscomplexobj(matrix)
    dtype = np.complex128 if is_complex else np.float64
    result = sparse.csr_matrix(matrix, dtype=dtype, copy=True)
    if result.shape[0] == 0 or result.shape[0] != result.shape[1]:
        raise ValueError("matrix must be nonempty and square")
    result.sum_duplicates()
    if not np.all(np.isfinite(result.data)):
        raise ValueError("matrix entries must be finite")
    result.sort_indices()
    return result


def _rhs(rhs: Any, size: int, *, preserve_extended: bool = False) -> Array:
    """Validate finite columns, preserving original forcing in explicit extended mode."""
    raw = np.asarray(rhs)
    if raw.ndim not in (1, 2) or raw.shape[0] != size:
        raise ValueError("rhs must have shape (n,) or (n, nrhs) matching matrix")
    if raw.ndim == 2 and raw.shape[1] == 0:
        raise ValueError("rhs must contain at least one column")
    dtype = (
        (np.clongdouble if np.iscomplexobj(raw) else np.longdouble)
        if preserve_extended
        else (np.complex128 if np.iscomplexobj(raw) else np.float64)
    )
    result = np.asarray(raw, dtype=dtype)
    if not np.all(np.isfinite(result)):
        raise ValueError("rhs entries must be finite")
    return result


def _tolerances(rtol: float, atol: float) -> None:
    """Reject nonfinite, negative, or simultaneously zero tolerances."""
    if not np.isfinite(rtol) or not np.isfinite(atol) or min(rtol, atol) < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    if rtol == 0 and atol == 0:
        raise ValueError("at least one tolerance must be positive")


def _refinement_steps(steps: int) -> None:
    """Reject invalid explicit correction limits without changing the residual criterion."""
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("refinement_steps must be a nonnegative integer")


def _valid_solution(rhs: Array, result: Any) -> Array:
    """Validate backend output before residual evaluation or correction arithmetic."""
    solution = np.asarray(result)
    if solution.shape != rhs.shape or not np.all(np.isfinite(solution)):
        raise LinearSolveError("solver returned a nonfinite solution or an incompatible shape")
    return solution


def _accurate_residual(matrix: sparse.csr_matrix, rhs: Array, solution: Array) -> Array:
    """Accumulate b-Ax with extended precision or portable compensated row sums."""
    if _EXTENDED_PRECISION:
        precision = np.result_type(matrix.dtype, rhs.dtype, solution.dtype, np.longdouble)
        return np.asarray(rhs, dtype=precision) - matrix.astype(precision) @ np.asarray(
            solution, dtype=precision
        )
    width = int(np.prod(solution.shape[1:]))
    columns = solution.reshape(matrix.shape[1], width)
    forcing = rhs.reshape(matrix.shape[0], width)
    result = np.empty_like(forcing, dtype=np.result_type(matrix.dtype, rhs.dtype, solution.dtype))
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row : row + 2]
        products = -matrix.data[start:stop, None] * columns[matrix.indices[start:stop]]
        terms = np.vstack((forcing[row], products))
        for column in range(columns.shape[1]):
            real = fsum(terms[:, column].real)
            result[row, column] = (
                real + 1j * fsum(terms[:, column].imag) if np.iscomplexobj(terms) else real
            )
    return result.reshape(rhs.shape)


def _checked(matrix: sparse.csr_matrix, rhs: Array, result: Any, rtol: float, atol: float) -> Array:
    """Check each right-hand side independently, including zero forcing."""
    solution = _valid_solution(rhs, result)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = matrix @ solution - rhs
    if not np.all(np.isfinite(difference)):
        raise LinearSolveError("residual evaluation produced nonfinite values")
    scale = np.maximum(atol, np.max(np.abs(rhs), axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    with np.errstate(over="ignore"):
        residual = np.linalg.norm(difference / scale, axis=0)
        threshold = np.maximum(atol / scale, rtol * np.linalg.norm(rhs / scale, axis=0))
    if np.any(residual > threshold):
        # Cancellation in the sparse dot product can mask an accurate solve.
        # Reevaluate the same criterion, without changing its physical scale.
        residual = np.linalg.norm(_accurate_residual(matrix, rhs, solution) / scale, axis=0)
    if np.any(residual > threshold):
        raise _ResidualFailure(
            f"residual criterion failed: scaled ||Ax-b||={residual}, allowed={threshold}"
        )
    return solution


def check_linear_solution(
    matrix: Any,
    rhs: Any,
    solution: Any,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> Array:
    """Check an external solver result against each original right-hand side.

    Require ``||A x-b|| <= max(atol, rtol*||b||)`` separately for every column,
    using the same residual evaluation as the native solver adapters. This
    function does not correct a result or select another solver. The operator
    is represented in real or complex binary64; explicitly wider right-hand
    sides and solution digits remain available to residual evaluation.
    Passing this algebraic check does not establish discretization stability
    or accuracy of physical fields.
    """
    _tolerances(rtol, atol)
    operator = _matrix(matrix)
    raw = np.asarray(rhs)
    forcing = _rhs(
        rhs,
        operator.shape[0],
        preserve_extended=raw.dtype in (np.dtype(np.longdouble), np.dtype(np.clongdouble)),
    )
    return _checked(operator, forcing, solution, rtol, atol)


def _optional(module: str, installation: str) -> ModuleType:
    """Import one optional module while preserving the underlying diagnosis."""
    try:
        return import_module(module)
    except (ImportError, OSError) as exc:
        raise SolverUnavailableError(
            f"Cannot load {module!r}. {installation} Original error: {exc}"
        ) from exc


def preload_solver_backend(solver: str) -> None:
    """Load only the selected solver's native libraries before thread limits.

    This idempotent preparation constructs no explicit factors, AMG hierarchy,
    communicator or GPU session. Call before a new ``threadpool_limits`` context so newly
    loaded BLAS/OpenMP libraries receive the same policy during assembly and
    solution. Portable SciPy solvers require no additional imports. Optional
    dependencies remain lazy until explicitly selected; unavailable backends
    raise the same diagnostic as the numerical adapters. Native parameter
    profiles, precision and residual tolerances are unchanged. PETSc preparation
    imports only its Python package; PETSc/MPI initialization remains owned by
    the existing numerical adapter.
    """
    modules = {
        "scipy": (),
        "cg": (),
        "minres": (),
        "gmres": (),
        "pypardiso": ("pypardiso",),
        "pypardiso-symmetric": ("pypardiso",),
        "pypardiso-symmetric-matching": ("pypardiso",),
        "petsc": ("petsc4py",),
        "petsc-symmetric": ("petsc4py",),
        "pyamg": ("pyamg",),
        "amgx": ("cupy", "pyamgx"),
        "cupy": ("cupy", "cupyx.scipy.sparse", "cupyx.scipy.sparse.linalg"),
        "cudss": ("cupy", "cupyx.scipy.sparse", "nvmath.sparse.advanced"),
    }
    if solver not in modules:
        raise ValueError(f"Unsupported solver: {solver!r}")
    for module in modules[solver]:
        _optional(module, f"Install the native dependencies for the {solver!r} backend.")


def _cuda() -> ModuleType:
    """Require a CuPy runtime and at least one usable CUDA device."""
    cupy = _optional("cupy", "Install CuPy matching your CUDA runtime.")
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:
        raise SolverUnavailableError(f"CUDA initialization failed: {exc}") from exc
    if count == 0:
        raise SolverUnavailableError("The selected GPU solver requires a visible CUDA device")
    return cupy


@dataclass
class LinearFactorization:
    """Reusable fixed-matrix factorization with checked solves and explicit cleanup.

    Create with :func:`factorize` or :func:`prepare_amgx` and use as a context
    manager, or call ``close`` explicitly. The latter prepares an iterative AMG
    hierarchy rather than direct factors. Prepared backend state must not be
    shared concurrently between threads. The matrix is copied when constructed.
    """

    _matrix: sparse.csr_matrix
    _solve: Callable[[Array], Array] | None
    _release: Callable[[], None]
    solver: str
    rtol: float = 1e-10
    atol: float = 0.0

    def matches(self, matrix: Any) -> bool:
        """Check exact operator identity before reusing factors for an assembled system."""
        other = _matrix(matrix)
        return bool(self._matrix.shape == other.shape and (self._matrix != other).nnz == 0)

    def solve(
        self,
        rhs: Any,
        *,
        refinement_precision: Literal["double", "extended"] = "double",
        refinement_steps: int = 2,
    ) -> Array:
        """Solve and refine up to ``refinement_steps`` times using the same factorization.

        Corrections use an extended-precision residual where available and
        compensated row sums elsewhere. Acceptance always uses the original
        columnwise criterion; invalid output and failed refinement are rejected.
        ``refinement_precision="extended"`` retains correction digits in a
        wider solution array, while the factorization and correction solves use
        their original precision. Original wider RHS digits are retained for
        residuals; only backend operands are rounded to double precision.
        This explicit mode requires a wider NumPy
        long-double type; no backend or tolerance is substituted.
        The nonnegative correction limit defaults to two; zero checks only the
        initial backend solve. Increasing it permits additional defect corrections
        but does not change acceptance or guarantee convergence.
        """
        if self._solve is None:
            raise RuntimeError("factorization is closed")
        _refinement_steps(refinement_steps)
        if refinement_precision not in {"double", "extended"}:
            raise ValueError("refinement_precision must be double or extended")
        if refinement_precision == "extended" and not _EXTENDED_PRECISION:
            raise SolverUnavailableError("extended refinement requires a wider long-double type")
        forcing = _rhs(
            rhs, self._matrix.shape[0], preserve_extended=refinement_precision == "extended"
        )
        backend_forcing = np.asarray(
            forcing, dtype=np.complex128 if np.iscomplexobj(forcing) else np.float64
        )
        result = np.asarray(self._solve(backend_forcing))
        correction_dtype = result.dtype
        if refinement_precision == "extended":
            result = result.astype(np.result_type(result.dtype, np.longdouble))
        for _ in range(refinement_steps):
            try:
                return _checked(self._matrix, forcing, result, self.rtol, self.atol)
            except _ResidualFailure:
                residual = np.asarray(
                    _accurate_residual(self._matrix, forcing, result), dtype=correction_dtype
                )
                correction = _valid_solution(forcing, self._solve(residual))
                result = result + correction
        return _checked(self._matrix, forcing, result, self.rtol, self.atol)

    def close(self) -> None:
        """Release backend resources once; further solves are rejected."""
        if self._solve is not None:
            self._solve = None
            self._release()

    def __enter__(self) -> LinearFactorization:
        """Enter an open factorization context."""
        if self._solve is None:
            raise RuntimeError("factorization is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release resources even when a solve raises an exception."""
        self.close()


def _petsc_factor(
    matrix: sparse.csr_matrix, stack: ExitStack, *, symmetric: bool = False
) -> Callable[[Array], Array]:
    """Build pivoted MUMPS LU or real symmetric-indefinite LDLt on COMM_SELF."""
    if symmetric:
        if np.iscomplexobj(matrix.data):
            raise ValueError("petsc-symmetric requires a real symmetric matrix")
        _hermitian(matrix, "petsc-symmetric")
    petsc = _optional("petsc4py.PETSc", "Install petsc4py with a compatible PETSc build.")
    scalar = np.dtype(petsc.ScalarType)
    if np.iscomplexobj(matrix.data) and scalar.kind != "c":
        raise ValueError("complex matrix requires a complex-scalar PETSc build")
    if not petsc.Sys.hasExternalPackage("mumps"):
        raise SolverUnavailableError(
            "The PETSc backend requires MUMPS for pivoted LU of mixed saddle systems. "
            "Install PETSc with MUMPS support or rebuild PETSc with --download-mumps."
        )
    operator = petsc.Mat().createAIJ(
        size=matrix.shape,
        csr=(
            matrix.indptr.astype(petsc.IntType),
            matrix.indices.astype(petsc.IntType),
            matrix.data.astype(scalar),
        ),
        comm=petsc.COMM_SELF,
    )
    stack.callback(operator.destroy)
    operator.assemble()
    if symmetric:
        operator.setOption(petsc.Mat.Option.SYMMETRIC, True)
        # MUMPS selects sym=2 (pivoted LDLt) when PETSc's Cholesky interface
        # receives an explicitly non-SPD matrix. No positive-definiteness
        # assumption is valid for the retained-mode saddle equations.
        operator.setOption(petsc.Mat.Option.SPD, False)
    ksp = petsc.KSP().create(comm=petsc.COMM_SELF)
    stack.callback(ksp.destroy)
    ksp.setOperators(operator)
    ksp.setType("preonly")
    preconditioner = ksp.getPC()
    preconditioner.setType("cholesky" if symmetric else "lu")
    preconditioner.setFactorSolverType("mumps")
    ksp.setUp()

    def solve(rhs: Array) -> Array:
        """Solve columnwise using the same PETSc preconditioner factorization."""
        if np.iscomplexobj(rhs) and scalar.kind != "c":
            raise ValueError("complex rhs requires a complex-scalar PETSc build")
        columns = rhs[:, None] if rhs.ndim == 1 else rhs
        solution = np.empty(columns.shape, dtype=scalar)
        for index in range(columns.shape[1]):
            with ExitStack() as vectors:
                forcing = petsc.Vec().createWithArray(
                    np.ascontiguousarray(columns[:, index], dtype=scalar), comm=petsc.COMM_SELF
                )
                vectors.callback(forcing.destroy)
                result = forcing.duplicate()
                vectors.callback(result.destroy)
                ksp.solve(forcing, result)
                if ksp.getConvergedReason() <= 0:
                    raise LinearSolveError("PETSc direct solve failed to converge")
                solution[:, index] = result.getArray(readonly=True)
        return solution[:, 0] if rhs.ndim == 1 else solution

    return solve


def _equilibrated_lu(matrix: sparse.csr_matrix) -> tuple[Any, Array, Array]:
    """Factor row/column-equilibrated CSR and reject numerically deficient pivots."""
    row_scale = np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
    if np.any(row_scale == 0):
        raise LinearSolveError("SuperLU factorization failed: matrix has a zero row")
    balanced = matrix.copy()
    balanced.data /= np.repeat(row_scale, np.diff(balanced.indptr))
    column_scale = np.asarray(abs(balanced).max(axis=0).toarray()).ravel()
    if np.any(column_scale == 0):
        raise LinearSolveError("SuperLU factorization failed: matrix has a zero column")
    balanced.data /= column_scale[balanced.indices]
    try:
        decomposition = splinalg.splu(balanced.tocsc())
    except RuntimeError as exc:
        raise LinearSolveError(f"SuperLU factorization failed: {exc}") from exc
    pivots = np.abs(decomposition.U.diagonal())
    threshold = np.finfo(np.float64).eps * matrix.shape[0] * np.max(np.abs(decomposition.U.data))
    if np.min(pivots) <= threshold:
        raise LinearSolveError(
            "SuperLU factorization has insufficient numerical rank after equilibration; "
            "check discrete stability, missing constraints or severe ill-conditioning"
        )
    return decomposition, row_scale, column_scale


def validate_invertible(matrix: Any) -> None:
    """Check finite square input and numerical rank using equilibrated sparse LU.

    This diagnostic rejects zero rows/columns, failed LU, or a pivot no larger
    than ``eps * n * max(abs(U))`` after row/column max-norm equilibration. It is
    a floating-point rank diagnostic, not proof of exact singularity or a bound
    on the condition number. In MHM it detects unsupported trace enrichment or
    missing constraints even when a compatible RHS has a small residual.
    The factorization is discarded; ordinary SciPy solves perform the same check
    while reusing their LU. Use this separately for other global backends when
    a backend-independent stability check is needed.
    """
    _equilibrated_lu(_matrix(matrix))


def _symmetric_equilibration(matrix: sparse.csr_matrix) -> tuple[sparse.csr_matrix, Array]:
    """Return five infinity-row Ruiz congruences and their positive diagonal.

    The relation is ``balanced = D @ matrix @ D``. Symmetry is preserved;
    neither rows nor equations are discarded, and the diagonal is not a
    regularization. A zero row is incompatible with invertible direct solves.
    """
    _hermitian(matrix, "symmetric equilibration")
    balanced = matrix.copy()
    diagonal = np.ones(matrix.shape[0])
    for _ in range(5):
        row_norm = np.asarray(abs(balanced).max(axis=1).toarray()).ravel()
        if np.any(row_norm == 0):
            raise LinearSolveError("symmetric equilibration failed: matrix has a zero row")
        scaling = 1.0 / np.sqrt(row_norm)
        diagonal *= scaling
        balanced = (sparse.diags(scaling) @ balanced @ sparse.diags(scaling)).tocsr()
    return balanced, diagonal


def factorize(
    matrix: Any,
    *,
    solver: str = "scipy",
    rtol: float = 1e-10,
    atol: float = 0.0,
    equilibration: Literal["none", "symmetric"] = "none",
    rhs_columns: int = 1,
) -> LinearFactorization:
    """Factor a fixed matrix once for repeated, independently checked solves.

    ``scipy`` uses SuperLU; ``pypardiso`` uses MKL PARDISO (real matrices only).
    ``pypardiso-symmetric`` explicitly selects its symmetric-indefinite factor
    (mtype=-2), with an upper-triangle representation and stored zero diagonals.
    ``pypardiso-symmetric-matching`` additionally selects the documented explicit
    weighted-matching/pivot profile; neither profile substitutes for the other.
    Symmetry is checked; residuals and corrections always use the full original
    matrix, including any assembly roundoff in its lower triangle.
    ``petsc`` requires MUMPS and uses its pivoted LU on PETSc COMM_SELF;
    ``petsc-symmetric`` uses its real symmetric-indefinite pivoted LDLt through
    PETSc's Cholesky interface with the SPD flag explicitly false.
    ``cudss`` uses NVIDIA cuDSS through
    nvmath and CuPy. Native resources are released by ``close`` or a context
    manager. CuPy's QR and Krylov methods are only available in ``solve_linear``.
    No backend is substituted when the requested one is unavailable. SciPy LU
    equilibrates rows/columns and checks numerical pivots before solving.
    ``equilibration="symmetric"`` applies five infinity-row Ruiz congruences
    before the selected factorization: ``A' = D A D``, ``b' = D b``, ``x = D y``.
    It requires a Hermitian operator, preserves the original residual criterion,
    and is independent of the backend's own scaling. The default is ``"none"``.
    ``rhs_columns`` is the positive cuDSS solve width. It permits one native
    solve/transfer for all local response columns, instead of one per basis.
    Other backends accept the hint without changing their multi-RHS behavior.
    Subsequent cuDSS solves of another width reuse the same factors in chunks,
    padding a final short chunk with zero loads. CUDA factors and cleanup stay
    on their construction device and restore the caller's current device.
    cuDSS host API entries are serialized within the process because its
    analysis and factorization phases do not guarantee host thread safety.
    Device transfers and asynchronous GPU work do not hold that entry lock.
    PARDISO construction, factorization, solve and cleanup entries are also
    serialized per process; each factor retains its independent lifetime.
    """
    _tolerances(rtol, atol)
    if isinstance(rhs_columns, bool) or not isinstance(rhs_columns, int) or rhs_columns < 1:
        raise ValueError("rhs_columns must be a positive integer")
    operator = _matrix(matrix)
    if equilibration not in {"none", "symmetric"}:
        raise ValueError("equilibration must be none or symmetric")
    diagonal = None
    backend_operator = operator
    if equilibration == "symmetric":
        backend_operator, diagonal = _symmetric_equilibration(operator)
    solve_function: Callable[[Array], Array]
    with ExitStack() as stack:
        if solver == "scipy":
            decomposition, row_scale, column_scale = _equilibrated_lu(backend_operator)

            def solve_scipy(rhs: Array) -> Array:
                """Reuse equilibrated SuperLU without losing complex RHS components."""
                rows = row_scale[:, None] if rhs.ndim == 2 else row_scale
                columns = column_scale[:, None] if rhs.ndim == 2 else column_scale
                scaled_rhs = rhs / rows
                if np.iscomplexobj(rhs) and not np.iscomplexobj(backend_operator.data):
                    result = decomposition.solve(scaled_rhs.real) + 1j * decomposition.solve(
                        scaled_rhs.imag
                    )
                else:
                    result = decomposition.solve(scaled_rhs)
                return np.asarray(result / columns)

            solve_function = solve_scipy
        elif solver in {"pypardiso", "pypardiso-symmetric", "pypardiso-symmetric-matching"}:
            if np.iscomplexobj(backend_operator.data):
                raise ValueError(
                    "pypardiso supports real matrices only; use scipy for complex data"
                )
            pardiso = _optional("pypardiso", "Install pypardiso and Intel MKL on Linux or Windows.")
            if solver != "pypardiso":
                _hermitian(backend_operator, solver)
                factored_operator = sparse.triu(backend_operator, format="csr")
                # PARDISO requires every diagonal entry, including structural
                # zeros in multiplier rows of a symmetric saddle system.
                factored_operator.setdiag(backend_operator.diagonal())
                factored_operator.sort_indices()
            else:
                factored_operator = backend_operator
            with _PARDISO_HOST_LOCK:
                engine = (
                    pardiso.PyPardisoSolver(mtype=-2)
                    if solver != "pypardiso"
                    else pardiso.PyPardisoSolver()
                )

            def release_pardiso() -> None:
                """Release this factor's native memory under the shared entry lock."""
                with _PARDISO_HOST_LOCK:
                    engine.free_memory(everything=True)

            stack.callback(release_pardiso)
            with _PARDISO_HOST_LOCK:
                if solver == "pypardiso-symmetric-matching":
                    # PyPardiso uses one-based IPARM indices. Explicit mode is
                    # necessary: automatic defaults otherwise replace these values.
                    # Matching changes the pivot strategy; neither profile is a
                    # fallback for the other, and both retain the physical gate.
                    for parameter, value in (
                        (1, 1),
                        (2, 2),
                        (8, 5),
                        (10, 13),
                        (11, 1),
                        (13, 1),
                        (21, 1),
                        (27, 1),
                    ):
                        engine.set_iparm(parameter, value)
                engine.factorize(factored_operator)

            def solve_pardiso(rhs: Array) -> Array:
                """Reuse the PARDISO factorization without complex truncation."""
                if np.iscomplexobj(rhs):
                    raise ValueError("pypardiso supports real right-hand sides only")
                with _PARDISO_HOST_LOCK:
                    return np.asarray(engine.solve(factored_operator, rhs))

            solve_function = solve_pardiso
        elif solver in {"petsc", "petsc-symmetric"}:
            solve_function = _petsc_factor(
                backend_operator, stack, symmetric=solver == "petsc-symmetric"
            )
        elif solver == "cudss":
            cupy = _cuda()
            device = int(cupy.cuda.runtime.getDevice())
            gpu_sparse = _optional("cupyx.scipy.sparse", "Install a compatible CuPy package.")
            nvmath = _optional(
                "nvmath.sparse.advanced", "Install nvmath-python with cuDSS and CUDA support."
            )
            device_matrix = gpu_sparse.csr_matrix(backend_operator)
            device_rhs = cupy.zeros(
                (backend_operator.shape[0], rhs_columns),
                dtype=backend_operator.dtype,
                order="F",
            )
            with _CUDSS_HOST_LOCK:
                engine = nvmath.DirectSolver(device_matrix, device_rhs)

                def free_cudss() -> None:
                    """Release native factors on their device with serialized host entry."""
                    with cupy.cuda.Device(device), _CUDSS_HOST_LOCK:
                        engine.free()

                stack.callback(free_cudss)
                engine.plan_config.matching_algorithm = (
                    nvmath.DirectSolverMatchingAlg.MAX_DIAG_PRODUCT
                )
                engine.solution_config.ir_num_steps = 5
                engine.plan()
                engine.factorize()

            def solve_cudss(rhs: Array) -> Array:
                """Reuse factors for column-major blocks on the construction device."""
                if np.iscomplexobj(rhs) and not np.iscomplexobj(backend_operator.data):
                    raise ValueError("complex rhs requires a complex matrix for cudss")
                columns = rhs[:, None] if rhs.ndim == 1 else rhs
                result = np.empty(columns.shape, dtype=backend_operator.dtype)
                with cupy.cuda.Device(device):
                    for first in range(0, columns.shape[1], rhs_columns):
                        last = min(first + rhs_columns, columns.shape[1])
                        block = np.zeros(
                            (columns.shape[0], rhs_columns),
                            dtype=backend_operator.dtype,
                            order="F",
                        )
                        block[:, : last - first] = columns[:, first:last]
                        device_block = cupy.asarray(block, dtype=backend_operator.dtype, order="F")
                        with _CUDSS_HOST_LOCK:
                            engine.reset_operands(b=device_block)
                            device_result = engine.solve()
                        result[:, first:last] = cupy.asnumpy(device_result)[:, : last - first]
                return result[:, 0] if rhs.ndim == 1 else result

            solve_function = solve_cudss
        else:
            raise ValueError(f"Unsupported factorization solver: {solver!r}")
        resources = stack.pop_all()
    if diagonal is not None:
        direct_solve = solve_function

        def solve_equilibrated(rhs: Array) -> Array:
            """Map RHS and solutions through the same symmetric congruence."""
            scaling = diagonal[:, None] if rhs.ndim == 2 else diagonal
            return scaling * direct_solve(scaling * rhs)

        solve_function = solve_equilibrated
    return LinearFactorization(operator, solve_function, resources.close, solver, rtol, atol)


def _hermitian(matrix: sparse.csr_matrix, solver: str) -> None:
    """Reject operators violating the symmetry required by the selected method."""
    difference = matrix - matrix.conjugate().T
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-13 * float(
        np.max(np.abs(matrix.data), initial=0)
    ):
        raise ValueError(f"{solver} requires a Hermitian/symmetric matrix")


def prepare_amgx(
    matrix: Any,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> LinearFactorization:
    """Prepare one GPU FGMRES/AMG hierarchy for repeated checked elliptic solves.

    The real float64 operator is copied. It must be symmetric with positive
    diagonal entries; these checks do not establish positive definiteness.
    Each nonzero relative-tolerance RHS is divided by its largest absolute
    entry, and the returned solution is multiplied by the same value. This
    linear change of scale keeps tiny loads above AmgX's native absolute
    stopping floor. Original RHS columns and tolerances define acceptance.

    The returned :class:`LinearFactorization` reuses this hierarchy, native
    buffers and transfers for all RHS columns and true-residual corrections.
    It owns process-global AmgX initialize/finalize and holds the package's
    lifecycle lock until ``close``; use a context manager and never overlap an
    external pyamgx session in the same process. Separate spawned processes
    can select separate GPUs before importing CUDA libraries. No numerical
    matrix, response or hierarchy is reused for a different operator.
    """
    _tolerances(rtol, atol)
    if maxiter is not None and (
        isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1
    ):
        raise ValueError("maxiter must be a positive integer or None")
    matrix = _matrix(matrix)
    if np.iscomplexobj(matrix.data):
        raise ValueError("the amgx adapter currently supports real float64 systems only")
    _hermitian(matrix, "amgx")
    if np.any(matrix.diagonal() <= 0):
        raise ValueError(
            "AMG elliptic presets require positive diagonals; do not pass saddle systems"
        )
    amgx = _optional(
        "pyamgx", "Build/install pyamgx against NVIDIA AmgX and a compatible CUDA runtime."
    )
    # AmgX initialization is process-global; serialize this package's lifecycles.
    with ExitStack() as stack:
        _AMGX_LOCK.acquire()
        stack.callback(_AMGX_LOCK.release)
        try:
            amgx.initialize()
        except Exception as exc:
            raise SolverUnavailableError(f"AmgX/CUDA initialization failed: {exc}") from exc
        stack.callback(amgx.finalize)
        configuration = amgx.Config().create_from_dict(
            {
                "config_version": 2,
                "determinism_flag": 1,
                "exception_handling": 1,
                "solver": {
                    "solver": "FGMRES",
                    "max_iters": 1000 if maxiter is None else maxiter,
                    "gmres_n_restart": 30,
                    "monitor_residual": 1,
                    "convergence": "RELATIVE_INI" if rtol > 0 else "ABSOLUTE",
                    "tolerance": rtol if rtol > 0 else atol,
                    "norm": "L2",
                    "preconditioner": {
                        "solver": "AMG",
                        "algorithm": "AGGREGATION",
                        "selector": "SIZE_2",
                        "smoother": "BLOCK_JACOBI",
                        "presweeps": 1,
                        "postsweeps": 1,
                        "max_iters": 1,
                        "max_levels": 30,
                        "cycle": "V",
                        "coarse_solver": "DENSE_LU_SOLVER",
                    },
                },
            }
        )
        stack.callback(configuration.destroy)
        resources = amgx.Resources().create_simple(configuration)
        stack.callback(resources.destroy)
        operator = amgx.Matrix().create(resources, mode="dDDI")
        stack.callback(operator.destroy)
        operator.upload_CSR(matrix)
        engine = amgx.Solver().create(resources, configuration, mode="dDDI")
        stack.callback(engine.destroy)
        engine.setup(operator)
        forcing = amgx.Vector().create(resources, mode="dDDI")
        stack.callback(forcing.destroy)
        result = amgx.Vector().create(resources, mode="dDDI")
        stack.callback(result.destroy)

        def solve(rhs: Array) -> Array:
            """Reuse native resources and map each RHS through its own scalar scale."""
            if np.iscomplexobj(rhs):
                raise ValueError("the amgx adapter currently supports real float64 systems only")
            columns = rhs[:, None] if rhs.ndim == 1 else rhs
            solution = np.zeros(columns.shape)
            for index in range(columns.shape[1]):
                column = columns[:, index]
                scale = float(np.max(np.abs(column)))
                if scale == 0:
                    continue
                if rtol == 0:
                    scale = 1.0
                forcing.upload(np.ascontiguousarray(column / scale))
                result.upload(np.zeros(matrix.shape[0]))
                engine.solve(forcing, result, zero_initial_guess=True)
                if engine.status != "success":
                    raise LinearSolveError(f"AmgX did not converge: {engine.status}")
                solution[:, index] = result.download() * scale
            return solution[:, 0] if rhs.ndim == 1 else solution

        resources = stack.pop_all()
    return LinearFactorization(matrix, solve, resources.close, "amgx", rtol, atol)


def solve_linear(
    matrix: Any,
    rhs: Any,
    *,
    solver: str = "scipy",
    rtol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
    near_nullspace: Any = None,
    refinement_precision: Literal["double", "extended"] = "double",
    refinement_steps: int = 2,
    equilibration: Literal["none", "symmetric"] = "none",
) -> Array:
    """Solve ``matrix @ x = rhs`` and require ``||r|| <= max(atol, rtol*||b||)``.

    The criterion is checked separately for every column. Inputs must be finite
    and the square matrix must be invertible for direct solvers. Available
    methods are ``scipy``, ``cg``, ``minres``, ``gmres``, ``pypardiso``,
    ``pypardiso-symmetric`` and ``pypardiso-symmetric-matching`` (real symmetric
    indefinite), ``petsc`` (MUMPS LU), ``petsc-symmetric`` (real MUMPS LDLt),
    ``cupy`` (GPU sparse QR), ``cudss``, ``pyamg`` (SA-preconditioned CPU CG),
    and ``amgx`` (GPU FGMRES with aggregation AMG). CG requires a Hermitian positive
    definite matrix; MINRES requires real symmetry, permits indefiniteness, and
    may solve compatible singular systems. GMRES accepts nonsymmetric systems.
    Symmetry is checked for CG/MINRES; positive definiteness is the caller's
    responsibility. Optional backends require a working native runtime.

    The two AMG presets target positive-definite elliptic operators, not the
    complete indefinite MHM saddle system. Symmetry and positive diagonals are
    checked; these checks do not establish positive definiteness. PyAMG accepts
    optional ``near_nullspace`` candidate columns (n,k), such as rigid modes for
    constrained elasticity. Other backends reject this argument. The AmgX
    adapter owns initialize/finalize and must not overlap an external pyamgx
    session in the same process. Its native resource lifecycles are serialized.

    GPU transfers are included in this call. PETSc requires MUMPS and uses
    COMM_SELF, not a distributed matrix. Use :func:`factorize` to amortize
    repeated direct solves. Direct backends accept ``equilibration="symmetric"``
    with the congruence and original-residual contract of :func:`factorize`.

    Krylov methods may correct up to ``refinement_steps`` prematurely accepted iterates using
    independently accumulated true residuals, reusing the preconditioner.
    Reusable direct factors use the same nonnegative correction limit, defaulting
    to two. Zero checks only the initial solve; extra corrections never relax the
    requested criterion. AmgX reuses one prepared hierarchy for all columns and
    corrections. Its relative-tolerance operands are normalized per column to
    avoid an absolute native stopping floor on tiny loads. CuPy QR does not
    expose this correction loop and rejects nondefault values.
    ``refinement_precision="extended"`` retains the corrected solution in
    extended precision while matrix storage and correction solves remain in
    double precision. This explicit mixed-precision mode supports Krylov
    methods and reusable direct factors (SciPy, PARDISO, PETSc and cuDSS); it
    requires a wider NumPy long-double type. It is useful
    when the requested residual lies below the double-precision solution's
    rounding floor. No tolerance is changed, and unsupported platforms reject it.
    """
    _tolerances(rtol, atol)
    _refinement_steps(refinement_steps)
    if refinement_steps != 2 and solver == "cupy":
        raise ValueError("refinement_steps requires a Krylov or reusable direct backend")
    if equilibration not in {"none", "symmetric"}:
        raise ValueError("equilibration must be none or symmetric")
    if equilibration != "none" and solver in {"cg", "minres", "gmres", "pyamg", "amgx", "cupy"}:
        raise ValueError("symmetric equilibration requires a reusable direct backend")
    if refinement_precision not in {"double", "extended"}:
        raise ValueError("refinement_precision must be double or extended")
    if refinement_precision == "extended":
        if solver not in {
            "cg",
            "minres",
            "gmres",
            "pyamg",
            "scipy",
            "pypardiso",
            "pypardiso-symmetric",
            "pypardiso-symmetric-matching",
            "petsc",
            "petsc-symmetric",
            "cudss",
        }:
            raise ValueError("extended refinement requires a Krylov or reusable direct backend")
        if not _EXTENDED_PRECISION:
            raise SolverUnavailableError("extended refinement requires a wider long-double type")
    if maxiter is not None and (
        isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1
    ):
        raise ValueError("maxiter must be a positive integer or None")
    operator = _matrix(matrix)
    forcing = _rhs(rhs, operator.shape[0], preserve_extended=refinement_precision == "extended")
    if near_nullspace is not None and solver != "pyamg":
        raise ValueError("near_nullspace is only supported by the pyamg backend")
    if solver in {"pyamg", "amgx"}:
        _hermitian(operator, solver)
        if np.any(operator.diagonal().real <= 0):
            raise ValueError(
                "AMG elliptic presets require positive diagonals; do not pass saddle systems"
            )
    if solver == "amgx":
        if np.iscomplexobj(forcing):
            raise ValueError("the amgx adapter currently supports real float64 systems only")
        with prepare_amgx(operator, rtol=rtol, atol=atol, maxiter=maxiter) as prepared:
            return prepared.solve(forcing, refinement_steps=refinement_steps)
    preconditioner = None
    method = solver
    if solver == "pyamg":
        candidates = None
        if near_nullspace is not None:
            candidates = _rhs(near_nullspace, operator.shape[0])
            if candidates.ndim != 2:
                raise ValueError("near_nullspace must have shape (n, number_of_candidates)")
        pyamg = _optional("pyamg", "Install the pyamg package to use CPU algebraic multigrid.")
        hierarchy = pyamg.smoothed_aggregation_solver(
            operator,
            B=candidates,
            symmetry="hermitian",
            smooth=("jacobi", {"weighting": "local"}),
        )
        preconditioner = hierarchy.aspreconditioner(cycle="V")
        method = "cg"
    if method in {"cg", "minres", "gmres"}:
        if solver == "minres" and (np.iscomplexobj(operator.data) or np.iscomplexobj(forcing)):
            raise ValueError("SciPy minres requires real matrix and rhs")
        if solver in {"cg", "minres"}:
            _hermitian(operator, solver)
        columns = forcing[:, None] if forcing.ndim == 1 else forcing
        dtype = np.result_type(
            operator.dtype, np.complex128 if np.iscomplexobj(forcing) else np.float64
        )
        output_dtype = (
            np.result_type(dtype, np.longdouble) if refinement_precision == "extended" else dtype
        )
        solution = np.empty(columns.shape, dtype=output_dtype)

        def iterate(column: Array, correction: bool = False) -> Array:
            """Run one Krylov solve, reusing any existing AMG hierarchy."""
            column = np.asarray(column, dtype=dtype)
            relative = max(rtol, np.finfo(float).eps) if correction else rtol
            absolute = 0.0 if correction else atol
            if solver == "minres":
                norm = float(np.linalg.norm(column))
                effective_rtol = max(relative, absolute / norm) if norm else relative
                result, status = splinalg.minres(
                    operator, column, rtol=effective_rtol, maxiter=maxiter
                )
            else:
                result, status = getattr(splinalg, method)(
                    operator,
                    column,
                    rtol=relative,
                    atol=absolute,
                    maxiter=maxiter,
                    M=preconditioner,
                )
            if status != 0:
                raise LinearSolveError(f"{solver} failed to converge (info={status})")
            return _valid_solution(column, result)

        for index in range(columns.shape[1]):
            column = columns[:, index]
            result = np.asarray(iterate(column), dtype=output_dtype)
            for _ in range(refinement_steps):
                try:
                    _checked(operator, column, result, rtol, atol)
                    break
                except _ResidualFailure:
                    residual = np.asarray(_accurate_residual(operator, column, result), dtype=dtype)
                    result = result + iterate(residual, correction=True)
            _checked(operator, column, result, rtol, atol)
            solution[:, index] = result
        result = solution[:, 0] if forcing.ndim == 1 else solution
    elif solver == "cupy":
        cupy = _cuda()
        gpu_sparse = _optional("cupyx.scipy.sparse", "Install a compatible CuPy package.")
        gpu_linalg = _optional("cupyx.scipy.sparse.linalg", "Install a compatible CuPy package.")
        dtype = np.result_type(operator.dtype, forcing.dtype)
        result = cupy.asnumpy(
            gpu_linalg.spsolve(
                gpu_sparse.csr_matrix(operator.astype(dtype)), cupy.asarray(forcing, dtype=dtype)
            )
        )
    else:
        with factorize(
            operator,
            solver=solver,
            rtol=rtol,
            atol=atol,
            equilibration=equilibration,
            rhs_columns=forcing.shape[1] if forcing.ndim == 2 else 1,
        ) as decomposition:
            if refinement_steps != 2:
                return decomposition.solve(
                    forcing,
                    refinement_precision=refinement_precision,
                    refinement_steps=refinement_steps,
                )
            if refinement_precision == "extended":
                return decomposition.solve(forcing, refinement_precision="extended")
            return decomposition.solve(forcing)
    return _checked(operator, forcing, result, rtol, atol)
