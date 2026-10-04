"""Backend-independent local elimination and sparse MHM saddle assembly."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np
from scipy import linalg, sparse

from pymhm.mesh import FloatArray, IntArray
from pymhm.parallel import map_local
from pymhm.solvers import (
    LinearFactorization,
    SolverUnavailableError,
    _accurate_residual,
    factorize,
    solve_linear,
    validate_invertible,
)


def _array(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
    """Copy a real finite array and verify its expected dimensions."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _preserved_array(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
    """Validate real arrays while retaining explicitly supplied wider floating digits."""
    checked = _array(value, shape, name)
    raw = np.asarray(value)
    return (
        np.array(raw, dtype=np.result_type(raw.dtype, float), copy=True)
        if raw.dtype.kind == "f"
        else checked
    )


def _matrix_action(matrix: Any, values: FloatArray) -> FloatArray:
    """Accumulate cancelled mode actions, then preserve the operands' real dtype.

    The shared residual helper uses wider precision where available and
    compensated row sums otherwise; float64 accelerator inputs remain float64.
    """
    dtype = np.result_type(matrix.dtype, values.dtype)
    return np.asarray(
        -_accurate_residual(matrix.tocsr(), np.zeros_like(values), values), dtype=dtype
    )


@dataclass(frozen=True, init=False)
class LocalProblem:
    """A local variational problem ``A u + B lambda = f``.

    ``kernel`` contains a basis Z of the right nullspace of A; ``left_kernel``
    defaults to the same basis. Distinct test and trial kernels are allowed.
    Alternatively, ``coarse_basis`` retains an arbitrary basis Z, including
    nearly null modes, in the global system. It is mutually exclusive with
    ``kernel``; no nullspace claim is made for this general basis. ``constraints``
    is C in C.T w = 0 and must pair nonsingularly with the retained Z; by default
    C=Z. Physical L2 orthogonality uses C=M Z. Signed normal orientations must already
    be included in B. ``trace_dofs`` maps its columns to the global skeleton.
    ``test_coupling`` is C in the global equation C.T u = g and defaults to B.
    ``test_basis`` and ``test_constraints`` specify the corresponding test-side
    retained modes and compatibility lifting. This permits Petrov formulations
    with different left/right kernels and independent B/C couplings.
    A declared kernel is tested at the matrix scale. Its nonzero floating-point
    actions A Z and A.T W are retained in the original equations through
    E=Z-R A Z; only exactly vanishing represented actions use the kernel shortcut.
    No boundary pressure constraints are applied locally.
    """

    matrix: Any
    coupling: FloatArray
    load: FloatArray
    trace_dofs: IntArray
    kernel: FloatArray
    coarse_basis: FloatArray
    constraints: FloatArray
    test_coupling: FloatArray
    left_kernel: FloatArray
    test_basis: FloatArray
    test_constraints: FloatArray
    _retained_action: FloatArray
    _test_action: FloatArray
    _correct_kernel: bool

    def __init__(
        self,
        matrix: Any,
        coupling: Any,
        load: Any,
        trace_dofs: Any,
        kernel: Any = None,
        constraints: Any = None,
        *,
        coarse_basis: Any = None,
        test_coupling: Any = None,
        left_kernel: Any = None,
        test_basis: Any = None,
        test_constraints: Any = None,
    ) -> None:
        """Check dimensions, finite coefficients and the declared local kernel."""
        if np.iscomplexobj(matrix):
            raise ValueError("local matrix must be real")
        matrix = sparse.csc_matrix(matrix, dtype=float, copy=True)
        n = matrix.shape[0]
        if matrix.shape != (n, n) or n == 0 or not np.isfinite(matrix.data).all():
            raise ValueError("local matrix must be finite, nonempty and square")
        dofs = np.asarray(trace_dofs)
        if (
            dofs.ndim != 1
            or not np.issubdtype(dofs.dtype, np.integer)
            or np.any(dofs < 0)
            or len(np.unique(dofs)) != len(dofs)
        ):
            raise ValueError("trace_dofs must be distinct nonnegative integer indices")
        coupling = _array(coupling, (n, len(dofs)), "coupling")
        test_coupling = (
            coupling.copy()
            if test_coupling is None
            else _array(test_coupling, coupling.shape, "test_coupling")
        )
        load = _array(load, (n,), "load")
        if kernel is not None and coarse_basis is not None:
            raise ValueError("kernel and coarse_basis are mutually exclusive")
        kernel = np.empty((n, 0)) if kernel is None else np.asarray(kernel)
        if kernel.ndim != 2:
            raise ValueError("kernel must be a matrix")
        kernel = _array(kernel, (n, kernel.shape[1]), "kernel")
        basis = kernel if coarse_basis is None else np.asarray(coarse_basis)
        if basis.ndim != 2:
            raise ValueError("coarse_basis must be a matrix")
        basis = _array(basis, (n, basis.shape[1]), "coarse_basis")
        if left_kernel is not None and coarse_basis is not None:
            raise ValueError("left_kernel and coarse_basis are mutually exclusive")
        left_kernel = (
            kernel.copy()
            if left_kernel is None
            else _array(left_kernel, kernel.shape, "left_kernel")
        )
        if test_basis is not None and kernel.shape[1]:
            raise ValueError("test_basis and kernel are mutually exclusive; use left_kernel")
        test_basis = (
            (left_kernel if coarse_basis is None else basis).copy()
            if test_basis is None
            else _array(test_basis, basis.shape, "test_basis")
        )
        constraints = (
            basis.copy() if constraints is None else _array(constraints, basis.shape, "constraints")
        )
        if basis.shape[1] and np.linalg.matrix_rank(constraints.T @ basis) != basis.shape[1]:
            raise ValueError("constraints must pair nonsingularly with the retained basis")
        test_constraints = (
            (constraints if np.array_equal(test_basis, basis) else test_basis).copy()
            if test_constraints is None
            else _array(test_constraints, basis.shape, "test_constraints")
        )
        if (
            basis.shape[1]
            and np.linalg.matrix_rank(test_basis.T @ test_constraints) != basis.shape[1]
        ):
            raise ValueError("test_constraints must pair nonsingularly with the test basis")
        if kernel.shape[1]:
            matrix_scale = float(np.max(np.abs(matrix.data), initial=0.0))
            if matrix_scale:
                normalized = matrix.copy()
                normalized.data /= matrix_scale
                modes = kernel / np.max(np.abs(kernel), axis=0)
                left_modes = left_kernel / np.max(np.abs(left_kernel), axis=0)
                matrix_norm = float(sparse.linalg.norm(normalized))
                for operator, vectors in ((normalized, modes), (normalized.T, left_modes)):
                    scales = matrix_norm * np.linalg.norm(vectors, axis=0)
                    if np.any(np.linalg.norm(operator @ vectors, axis=0) > 1e-10 * scales):
                        raise ValueError("declared kernel is not a left and right nullspace")
        for name, value in (
            ("matrix", matrix),
            ("coupling", coupling),
            ("load", load),
            ("trace_dofs", dofs.astype(np.int64, copy=True)),
            ("kernel", kernel),
            ("coarse_basis", basis),
            ("constraints", constraints),
            ("test_coupling", test_coupling),
            ("left_kernel", left_kernel),
            ("test_basis", test_basis),
            ("test_constraints", test_constraints),
        ):
            object.__setattr__(self, name, value)
        action = _matrix_action(matrix, basis) if basis.shape[1] else np.empty_like(basis)
        left_action = (
            _matrix_action(matrix.T, test_basis) if basis.shape[1] else np.empty_like(basis)
        )
        object.__setattr__(self, "_retained_action", action)
        object.__setattr__(self, "_test_action", left_action)
        object.__setattr__(
            self,
            "_correct_kernel",
            bool(kernel.shape[1] and (np.any(action) or np.any(left_action))),
        )

    def condense(
        self,
        solver: str = "scipy",
        *,
        refinement_precision: Literal["double", "extended"] = "double",
    ) -> "LocalResponse":
        """Factor once for source, trace and retained responses, with explicit precision.

        Direct factors can retain corrected lifts in extended precision without
        changing their residual criterion. AMG local projection currently accepts
        only the default double-precision accumulation.
        """
        if refinement_precision not in {"double", "extended"}:
            raise ValueError("refinement_precision must be double or extended")
        n, k = self.matrix.shape[0], self.coarse_basis.shape[1]
        general = k != self.kernel.shape[1]
        if solver in {"pyamg", "amgx"}:
            if refinement_precision != "double":
                raise ValueError("AMG local solvers require double refinement precision")
            if general:
                raise ValueError("AMG local solvers do not support a general coarse_basis")
            if not np.array_equal(self.kernel, self.left_kernel) or not np.array_equal(
                self.constraints, self.test_constraints
            ):
                raise ValueError("AMG kernel projection requires matching test and trial spaces")
            if self._correct_kernel:
                rhs = np.column_stack((self.load, self.coupling, self._retained_action))
                response = np.zeros_like(rhs)
                _, _, pivots = linalg.qr(self.kernel.T, pivoting=True)
                free = np.setdiff1d(np.arange(n), pivots[:k])
                pairing = self.kernel.T @ self.constraints
                for step in range(4):
                    defect = rhs - _matrix_action(self.matrix, response)
                    defect -= self.constraints @ np.linalg.solve(pairing, self.kernel.T @ defect)
                    norms = np.linalg.norm(rhs, axis=0)
                    if step and np.all(np.linalg.norm(defect, axis=0) <= 1e-10 * norms):
                        break
                    correction = np.zeros_like(rhs)
                    if len(free):
                        correction[free] = solve_linear(
                            self.matrix[free][:, free], defect[free], solver=solver
                        )
                    correction -= self.kernel @ np.linalg.solve(
                        self.constraints.T @ self.kernel, self.constraints.T @ correction
                    )
                    response += correction
                else:
                    raise ValueError("AMG constrained residual refinement did not converge")
                width = self.coupling.shape[1] + 1
                return LocalResponse(
                    self,
                    response[:, 0],
                    response[:, 1:width],
                    self.coarse_basis - response[:, width:],
                )
            rhs = np.column_stack((self.load, self.coupling))
            if k:
                # Pin independent kernel coordinates only during the elliptic solve.
                # Afterwards restore the physical mean, with no dense rank update.
                rhs -= self.constraints @ np.linalg.solve(
                    self.kernel.T @ self.constraints, self.kernel.T @ rhs
                )
                _, _, pivots = linalg.qr(self.kernel.T, pivoting=True)
                free = np.setdiff1d(np.arange(n), pivots[:k])
                response = np.zeros_like(rhs)
                if len(free):
                    response[free] = solve_linear(
                        self.matrix[free][:, free], rhs[free], solver=solver
                    )
                response -= self.kernel @ np.linalg.solve(
                    self.constraints.T @ self.kernel, self.constraints.T @ response
                )
            else:
                response = solve_linear(self.matrix, rhs, solver=solver)
            return LocalResponse(self, response[:, 0], response[:, 1:])
        matrix, rhs = self.condensation_system()
        with factorize(matrix, solver=solver) as decomposition:
            return self.response_from_solution(
                decomposition.solve(rhs, refinement_precision=refinement_precision)
            )

    def condensation_system(self) -> tuple[Any, FloatArray]:
        """Return the constrained operator and source/trace/retained-mode right sides.

        The augmented operator is ``[[A, C_left], [C_right.T, 0]]`` with
        constraint columns scaled down only when they exceed the operator's
        largest entry. This change of auxiliary coordinates prevents constraints
        from dominating small-unit operators without magnifying constraints in
        mixed high-contrast saddles. It leaves the physical complement unchanged.
        Its inverse is never explicitly formed. Backends may factor this matrix
        once, retain the factorization, and solve new source columns online.
        """
        n, k = self.matrix.shape[0], self.coarse_basis.shape[1]
        general = k != self.kernel.shape[1] or self._correct_kernel
        matrix = self.condensation_matrix()
        width = self.coupling.shape[1] + 1
        rhs = np.zeros(
            (n + k, width + (k if general else 0)), dtype=np.result_type(self.load.dtype, float)
        )
        rhs[:n, 0] = self.load
        rhs[:n, 1:width] = self.coupling
        if general:
            rhs[:n, width:] = self._retained_action
        return matrix, rhs

    def condensation_matrix(self) -> Any:
        """Return the augmented operator without allocating response right-hand sides.

        This is the same scaled left/right constraint operator returned by
        ``condensation_system``. It permits one caller-owned factorization for
        several direct reconstructions, including nonsymmetric Petrov data.
        """
        k = self.coarse_basis.shape[1]
        matrix = self.matrix
        if k:
            scale = float(np.max(np.abs(matrix.data), initial=0.0)) or 1.0
            left = self.test_constraints * np.minimum(
                1.0, scale / np.max(np.abs(self.test_constraints), axis=0)
            )
            right = self.constraints * np.minimum(
                1.0, scale / np.max(np.abs(self.constraints), axis=0)
            )
            matrix = sparse.bmat(
                [
                    [matrix, sparse.csc_matrix(left)],
                    [sparse.csc_matrix(right.T), None],
                ],
                format="csc",
            )
        return matrix

    def reconstruct(
        self,
        trace: Any,
        coarse: Any,
        *,
        retained_basis: Any = None,
        solver: str = "scipy",
        factorization: LinearFactorization | None = None,
        refinement_precision: Literal["double", "extended"] = "double",
    ) -> FloatArray:
        """Reconstruct local fields without storing every trace response.

        ``trace`` uses local coupling-column order and ``coarse`` uses retained
        trial coordinates. Vectors produce one field; matching nonempty column
        arrays produce several fields with one factorization. With R the
        volumetric block of the augmented inverse,
        the field is R(f-B trace-A Z coarse)+Z coarse for an arbitrary retained Z.
        This includes nonsymmetric Petrov problems and an empty retained basis.
        ``retained_basis`` supplies an executed E=Z-R A Z for persisted coarse
        coordinates; then the field is R(f-B trace)+E coarse. Its numerical basis
        and orientation belong to the caller's archived coefficient contract.
        Only combined source columns are solved. Explicit extended precision preserves
        forcing and correction digits; backend factors stay double precision.
        A supplied factorization must match the augmented operator exactly and
        remains caller-owned. The linear residual uses the combined augmented
        RHS; full physical compatibility remains a global original-equation check.
        """
        n, k = self.matrix.shape[0], self.coarse_basis.shape[1]

        def coefficients(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
            """Validate coefficient dimensions without truncating archived wider values."""
            checked = _array(value, shape, name)
            raw = np.asarray(value)
            return raw.copy() if raw.dtype.kind == "f" else checked

        raw_trace = np.asarray(trace)
        if raw_trace.ndim not in (1, 2) or (raw_trace.ndim == 2 and not raw_trace.shape[1]):
            raise ValueError("trace must be a vector or a nonempty column array")
        suffix = raw_trace.shape[1:]
        local_trace = coefficients(trace, (self.coupling.shape[1], *suffix), "trace")
        local_coarse = coefficients(coarse, (k, *suffix), "coarse")
        basis = (
            self.coarse_basis
            if retained_basis is None
            else coefficients(retained_basis, (n, k), "retained_basis")
        )
        dtype = np.result_type(local_trace.dtype, local_coarse.dtype, basis.dtype)
        if refinement_precision == "extended":
            dtype = np.result_type(dtype, np.longdouble)
        load = self.load[:, None] if suffix else self.load
        forcing = load.astype(dtype) - np.einsum(
            "ij,j...->i...", self.coupling, local_trace, dtype=dtype
        )
        if retained_basis is None:
            forcing -= np.einsum("ij,j...->i...", self._retained_action, local_coarse, dtype=dtype)
        rhs = np.concatenate((forcing, np.zeros((k, *suffix), dtype=dtype)))
        matrix = self.condensation_matrix()
        if factorization is not None:
            if not factorization.matches(matrix):
                raise ValueError("prepared factorization does not match the local augmented matrix")
            solved = factorization.solve(rhs, refinement_precision=refinement_precision)
        else:
            solved = solve_linear(
                matrix, rhs, solver=solver, refinement_precision=refinement_precision
            )
        return solved[:n] + basis @ local_coarse

    def response_from_solution(self, solution: Any) -> "LocalResponse":
        """Decode finite real responses, preserving explicitly retained wider precision."""
        n, k = self.matrix.shape[0], self.coarse_basis.shape[1]
        width = self.coupling.shape[1] + 1
        general = k != self.kernel.shape[1] or self._correct_kernel
        shape = (n + k, width + (k if general else 0))
        if np.iscomplexobj(solution):
            raise ValueError("condensation solution must be real")
        dtype = np.longdouble if np.asarray(solution).dtype == np.dtype(np.longdouble) else float
        response = np.array(solution, dtype=dtype, copy=True)
        if response.shape != shape or not np.isfinite(response).all():
            raise ValueError(f"condensation solution must be finite with shape {shape}")
        response = response[:n]
        coarse_vectors = self.coarse_basis - response[:, width:] if general else None
        return LocalResponse(self, response[:, 0], response[:, 1:width], coarse_vectors)

    def condensed_load(
        self, source_response: Any, *, load: Any = None, corrected_retained: bool | None = None
    ) -> FloatArray:
        """Project executed source responses to the original reduced Petrov RHS.

        ``source_response`` is R times the supplied ``load`` (the stored source
        by default). A vector gives one reduced RHS; nonempty column arrays give
        several. Real wider source/load digits are preserved. The trace part is
        test_coupling.T R load. General or rounded retained modes use
        (A.T W).T R load-W.T load; exact represented kernels use -W.T load.
        ``corrected_retained`` can preserve an executed LocalResponse branch.
        It does not change the augmented operator or test/trial conventions.
        """
        raw = np.asarray(source_response)
        if raw.ndim not in (1, 2) or (raw.ndim == 2 and not raw.shape[1]):
            raise ValueError("source_response must be a vector or nonempty column array")
        n = len(self.load)
        suffix = raw.shape[1:]
        response = _preserved_array(source_response, (n, *suffix), "source_response")
        forcing = self.load if load is None else load
        if suffix and np.asarray(forcing).ndim == 1:
            forcing = np.broadcast_to(np.asarray(forcing)[:, None], (n, *suffix))
        forcing = _preserved_array(forcing, (n, *suffix), "load")
        if corrected_retained is not None and not isinstance(corrected_retained, (bool, np.bool_)):
            raise ValueError("corrected_retained must be a boolean or None")
        general = self.coarse_basis.shape[1] != self.kernel.shape[1] or self._correct_kernel
        corrected = general if corrected_retained is None else corrected_retained
        coarse_rhs = (
            self._test_action.T @ response - self.test_basis.T @ forcing
            if corrected
            else -self.left_kernel.T @ forcing
        )
        return np.concatenate((self.test_coupling.T @ response, coarse_rhs), axis=0)

    def with_load(self, load: Any, *, preserve_precision: bool = False) -> "LocalProblem":
        """Copy this contract with a different finite source vector.

        The default retains the canonical double input convention. Explicit
        ``preserve_precision=True`` keeps real wider defect-source digits for
        original-equation refinement, including augmented/reduced right sides.
        Matrices, kernels, moment constraints and trace orientation are unchanged.
        """
        result = LocalProblem(
            self.matrix,
            self.coupling,
            load,
            self.trace_dofs,
            kernel=self.kernel if self.kernel.shape[1] else None,
            constraints=self.constraints,
            coarse_basis=self.coarse_basis if not self.kernel.shape[1] else None,
            test_coupling=self.test_coupling,
            left_kernel=self.left_kernel if self.kernel.shape[1] else None,
            test_basis=self.test_basis if not self.kernel.shape[1] else None,
            test_constraints=self.test_constraints,
        )
        if preserve_precision:
            object.__setattr__(result, "load", _preserved_array(load, self.load.shape, "load"))
        return result


@dataclass(frozen=True)
class LocalAssembly:
    """A local variational problem and optional application reconstruction data.

    ``metadata`` can hold a NumPy mesh, DOF map or other application data that
    should return with the condensed response. Process workers require picklable
    inputs and metadata: do not capture or return live FEniCS, PETSc, MPI or CUDA
    objects. Construct native resources inside the worker, extract numerical
    arrays into ``LocalProblem``, and release resources inside that worker.
    """

    problem: LocalProblem
    metadata: Any = None

    def __post_init__(self) -> None:
        """Reject a factory payload that does not contain a validated local problem."""
        if not isinstance(self.problem, LocalProblem):
            raise TypeError("LocalAssembly.problem must be a LocalProblem")


@dataclass(frozen=True)
class LocalResponse:
    """Cached source/trace lifts and reconstructed retained modes of one cell."""

    problem: LocalProblem
    source: FloatArray
    lifts: FloatArray
    coarse_vectors: FloatArray | None = None

    @property
    def retained_basis(self) -> FloatArray:
        """Return Z for an exact represented kernel, otherwise E=Z-R A Z."""
        return self.problem.coarse_basis if self.coarse_vectors is None else self.coarse_vectors

    def reconstruct(self, trace: FloatArray, coarse: FloatArray) -> FloatArray:
        """Recover local coefficients from skeleton and retained coarse amplitudes."""
        return self.source - self.lifts @ trace + self.retained_basis @ coarse

    def kernel_roundoff_bound(self, field: FloatArray) -> FloatArray:
        """Bound floating-point evaluation of retained kernel equations.

        For declared left/right kernels only, sum the componentwise row bound
        ``gamma_(nnz_i+1) * (abs(A) @ abs(u))_i`` against ``abs(W)`` and
        include the final test projection's dot-product bound. Here gamma uses
        the stored double-precision operator, even for wider reconstruction.
        No operator entry, represented kernel action or physical residual is
        replaced by zero. General retained modes receive no kernel allowance.
        """
        p = self.problem
        if not p.kernel.shape[1] or not p._correct_kernel:
            return np.zeros(p.coarse_basis.shape[1])
        units = (p.matrix.getnnz(axis=1) + 1) * np.finfo(float).eps
        row_bound = (units / (1 - units)) * (abs(p.matrix) @ abs(field))
        projected = abs(p.test_basis).T @ row_bound
        units_test = (len(field) + 1) * np.finfo(float).eps
        action = _matrix_action(p.matrix, field)
        return projected + units_test / (1 - units_test) * (abs(p.test_basis).T @ abs(action))

    def global_load(self) -> FloatArray:
        """Return source-dependent reduced loads without rebuilding matrix blocks."""
        return self.problem.condensed_load(
            self.source, corrected_retained=self.coarse_vectors is not None
        )

    def global_contribution(self, coarse_dofs: Any) -> tuple[IntArray, FloatArray, FloatArray]:
        """Return indices, matrix and RHS of this cell's reduced Petrov equations.

        This is the common assembly contract for serial, MPI and accelerator
        backends. The trace rows enforce C.T u=g; retained test rows enforce
        W.T (A u+B lambda-f)=0. The negative signs preserve the symmetric
        saddle convention whenever C=B and test/trial data coincide.
        """
        p = self.problem
        coarse_dofs = np.asarray(coarse_dofs)
        if (
            coarse_dofs.shape != (p.coarse_basis.shape[1],)
            or not np.issubdtype(coarse_dofs.dtype, np.integer)
            or np.any(coarse_dofs < 0)
        ):
            raise ValueError("coarse_dofs must contain one nonnegative integer per retained mode")
        g = p.test_coupling.T @ self.retained_basis
        if self.coarse_vectors is None:
            coarse_trace = p.left_kernel.T @ p.coupling
            coarse_matrix = np.zeros((len(coarse_dofs), len(coarse_dofs)))
        else:
            coarse_trace = p.test_basis.T @ p.coupling - p._test_action.T @ self.lifts
            coarse_matrix = p._test_action.T @ self.retained_basis
        block = np.block([[p.test_coupling.T @ self.lifts, -g], [-coarse_trace, -coarse_matrix]])
        return (
            np.r_[p.trace_dofs, coarse_dofs].astype(np.int64),
            block,
            self.global_load(),
        )


@dataclass(frozen=True)
class HybridSolution:
    """Skeleton coefficients, retained coarse amplitudes and reconstructed local fields.

    ``residual`` is the verified relative compatibility defect after any
    componentwise declared-kernel evaluation allowance. Global solves also
    preserve its unadjusted value in ``raw_residual`` and its absolute norm in
    ``raw_residual_norm``. Manually constructed solutions may omit these records.
    """

    trace: FloatArray
    coarse: tuple[FloatArray, ...]
    fields: tuple[FloatArray, ...]
    residual: float
    gauge_multipliers: FloatArray
    raw_residual: float | None = None
    raw_residual_norm: float | None = None


class HybridSystem:
    """Assemble the global MHM equations independently of the local FEM backend.

    The unknown is (lambda, c). Dirichlet traces enter through ``boundary_load``
    as integrals against oriented skeleton basis functions. Prescribed Neumann
    flux coefficients are passed to ``solve`` as ``fixed`` DOFs. Arbitrary linear
    mean constraints can remove global pressure or rigid-motion nullspaces.
    """

    def __init__(
        self,
        problems: list[LocalProblem] | tuple[LocalProblem, ...],
        *,
        boundary_load: Any = None,
        local_solver: str = "scipy",
        local_refinement_precision: Literal["double", "extended"] = "double",
        backend: Literal["serial", "thread", "process"] = "serial",
        workers: int | None = None,
    ) -> None:
        """Condense independent subdomains and assemble their sparse contributions."""
        responses = tuple(
            map_local(
                partial(
                    _condense,
                    solver=local_solver,
                    refinement_precision=local_refinement_precision,
                ),
                problems,
                backend=backend,
                workers=workers,
            )
        )
        self._assemble_global(responses, (None,) * len(responses), boundary_load)

    @classmethod
    def from_local_factory(
        cls,
        factory: Callable[[Any], LocalProblem | LocalAssembly],
        items: Iterable[Any],
        *,
        boundary_load: Any = None,
        local_solver: str = "scipy",
        local_refinement_precision: Literal["double", "extended"] = "double",
        backend: Literal["serial", "thread", "process"] = "serial",
        workers: int | None = None,
    ) -> "HybridSystem":
        """Assemble and condense each independent local problem inside its worker.

        ``factory(item)`` returns ``LocalProblem`` or ``LocalAssembly``. The
        factory and one condensation execute together, exactly once per item;
        global assembly and solution remain in the parent. Input order defines
        response order and ``local_metadata``, including ``None`` for a bare
        problem. Exceptions propagate without a serial fallback.

        Serial and thread backends accept closures. The portable spawn backend
        requires picklable factories, items and metadata, and an executable
        script protected by ``if __name__ == '__main__':``. Native FEM, PETSc,
        MPI and CUDA resources must be created and released inside a worker,
        never sent to or returned from a worker. Native thread pools are limited
        to one thread; backend library thread safety is the factory's concern.
        """
        assembled = map_local(
            partial(
                _assemble_and_condense,
                factory=factory,
                solver=local_solver,
                refinement_precision=local_refinement_precision,
            ),
            items,
            backend=backend,
            workers=workers,
        )
        system = cls.__new__(cls)
        system._assemble_global(
            tuple(response for response, _ in assembled),
            tuple(metadata for _, metadata in assembled),
            boundary_load,
        )
        return system

    @classmethod
    def from_responses(
        cls,
        responses: Iterable[LocalResponse],
        *,
        boundary_load: Any = None,
        metadata: Iterable[Any] | None = None,
    ) -> "HybridSystem":
        """Assemble cached local responses without repeating any factorization."""
        values = tuple(responses)
        if any(not isinstance(value, LocalResponse) for value in values):
            raise TypeError("responses must contain LocalResponse objects")
        records = (None,) * len(values) if metadata is None else tuple(metadata)
        if len(records) != len(values):
            raise ValueError("one metadata record per local response is required")
        system = cls.__new__(cls)
        system._assemble_global(values, records, boundary_load)
        return system

    @classmethod
    def from_contributions(
        cls,
        contributions: Iterable[tuple[IntArray, FloatArray, FloatArray]],
        *,
        trace_size: int,
        coarse_sizes: Iterable[int],
        boundary_load: Any = None,
    ) -> "HybridSystem":
        """Assemble compact cell contributions without retaining local lifts.

        Each tuple is the output of ``LocalResponse.global_contribution`` with
        coarse indices numbered after the ``trace_size`` skeleton unknowns.
        ``coarse_sizes`` declares their ordered cell partition. ``solve`` returns
        trace/coarse coefficients and an empty field tuple; callers reconstruct
        one local response at a time using the executed, persisted local basis.
        Physical mean rows must be supplied explicitly to ``solve``: this compact
        representation cannot derive them or certify kernel evaluation roundoff.
        """
        sizes = tuple(coarse_sizes)
        if (
            isinstance(trace_size, (bool, np.bool_))
            or not isinstance(trace_size, (int, np.integer))
            or trace_size < 0
            or not sizes
            or any(
                isinstance(size, (bool, np.bool_))
                or not isinstance(size, (int, np.integer))
                or size < 0
                for size in sizes
            )
        ):
            raise ValueError("trace_size and each coarse size must be nonnegative integers")
        values = tuple(contributions)
        if len(values) != len(sizes):
            raise ValueError("one contribution per coarse cell partition is required")
        system = cls.__new__(cls)
        system.responses = ()
        system.local_metadata = ()
        system.trace_size = int(trace_size)
        system.kernel_offsets = np.r_[trace_size, trace_size + np.cumsum(sizes)].astype(np.int64)
        system._assemble_contributions(values, boundary_load)
        return system

    def _assemble_global(
        self,
        responses: tuple[LocalResponse, ...],
        metadata: tuple[Any, ...],
        boundary_load: Any,
    ) -> None:
        """Initialize one global skeleton system from already condensed local responses."""
        if not responses:
            raise ValueError("at least one local problem is required")
        self.responses = responses
        self.local_metadata = metadata
        problems = tuple(response.problem for response in responses)
        all_dofs = np.concatenate([p.trace_dofs for p in problems])
        self.trace_size = int(all_dofs.max()) + 1 if len(all_dofs) else 0
        if not np.array_equal(np.unique(all_dofs), np.arange(self.trace_size)):
            raise ValueError("global trace numbering must be contiguous")
        self.kernel_offsets = np.r_[
            self.trace_size,
            self.trace_size + np.cumsum([p.coarse_basis.shape[1] for p in problems]),
        ].astype(np.int64)
        contributions = (
            response.global_contribution(
                np.arange(self.kernel_offsets[cell], self.kernel_offsets[cell + 1])
            )
            for cell, response in enumerate(self.responses)
        )
        self._assemble_contributions(contributions, boundary_load)

    def _assemble_contributions(
        self,
        contributions: Iterable[tuple[IntArray, FloatArray, FloatArray]],
        boundary_load: Any,
    ) -> None:
        """Accumulate cell blocks and loads without truncating wider real arithmetic."""
        size = int(self.kernel_offsets[-1])
        rhs = np.zeros(size)
        load_scale = np.zeros(size)
        rows: list[int] = []
        columns: list[int] = []
        entries: list[float] = []
        all_indices = []
        for cell, (indices, block, local_rhs) in enumerate(contributions):
            indices = np.asarray(indices)
            if (
                indices.ndim != 1
                or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0)
                or np.any(indices >= size)
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError("contribution indices must be distinct valid global integers")
            if not np.array_equal(
                np.sort(indices[indices >= self.trace_size]),
                np.arange(self.kernel_offsets[cell], self.kernel_offsets[cell + 1]),
            ):
                raise ValueError(
                    "contribution coarse indices must match its ordered cell partition"
                )
            checked_block = _array(block, (len(indices), len(indices)), "contribution matrix")
            checked_rhs = _array(local_rhs, (len(indices),), "contribution rhs")
            # Real floating inputs may contain explicitly retained correction digits.
            # Validation remains identical for lists and other convertible inputs.
            block, local_rhs = np.asarray(block), np.asarray(local_rhs)
            block = block if block.dtype.kind == "f" else checked_block
            local_rhs = local_rhs if local_rhs.dtype.kind == "f" else checked_rhs
            dtype = np.result_type(rhs.dtype, local_rhs.dtype)
            rhs, load_scale = rhs.astype(dtype, copy=False), load_scale.astype(dtype, copy=False)
            all_indices.append(indices)
            rows.extend(np.repeat(indices, len(indices)))
            columns.extend(np.tile(indices, len(indices)))
            entries.extend(block.ravel())
            np.add.at(rhs, indices, local_rhs)
            np.add.at(load_scale, indices, np.abs(local_rhs))
        if not np.array_equal(np.unique(np.concatenate(all_indices)), np.arange(size)):
            raise ValueError("contributions must cover the contiguous global numbering")
        self.matrix = sparse.coo_matrix((entries, (rows, columns)), shape=(size, size)).tocsc()
        self.matrix.eliminate_zeros()
        if boundary_load is not None:
            boundary = _array(boundary_load, (self.trace_size,), "boundary_load")
            raw_boundary = np.asarray(boundary_load)
            if raw_boundary.dtype.kind == "f":
                boundary = raw_boundary
            dtype = np.result_type(rhs.dtype, boundary.dtype)
            rhs, load_scale = rhs.astype(dtype, copy=False), load_scale.astype(dtype, copy=False)
            rhs[: self.trace_size] -= boundary
            load_scale[: self.trace_size] += np.abs(boundary)
        self.rhs = rhs
        self.load_scale = load_scale

    def mean_constraint(
        self, local_weights: list[FloatArray] | tuple[FloatArray, ...], value: float = 0.0
    ) -> tuple[FloatArray, float]:
        """Express a prescribed integral of reconstructed fields as ``r.T x=b``.

        For incompressible flow, weights integrate only the pressure components;
        for scalar pure Neumann diffusion, they integrate the pressure field.
        """
        if not self.responses:
            raise ValueError("compact contributions require explicit physical mean rows")
        if len(local_weights) != len(self.responses) or not np.isfinite(value):
            raise ValueError("one weight vector per local problem and finite value required")
        weights = tuple(
            _preserved_array(w, r.source.shape, "local_weights")
            for w, r in zip(local_weights, self.responses, strict=True)
        )
        moment_value = _preserved_array([value], (1,), "value")[0]
        dtype = np.result_type(
            np.asarray(moment_value).dtype,
            float,
            *(response.source.dtype for response in self.responses),
            *(response.lifts.dtype for response in self.responses),
            *(response.retained_basis.dtype for response in self.responses),
            *(w.dtype for w in weights),
        )
        row = np.zeros(len(self.rhs), dtype=dtype)
        target = np.asarray(moment_value, dtype=dtype).item()
        for i, (response, local) in enumerate(zip(self.responses, weights, strict=True)):
            np.add.at(row, response.problem.trace_dofs, -response.lifts.T @ local)
            row[self.kernel_offsets[i] : self.kernel_offsets[i + 1]] = (
                response.retained_basis.T @ local
            )
            target -= local @ response.source
        return row, target

    def with_rhs(self, rhs: Any, *, load_scale: Any = None) -> "HybridSystem":
        """Reuse the executed condensed matrix with a new explicit reduced load.

        This compact online system retains the original trace numbering and
        ordered retained partitions, and returns no reconstructed fields.
        ``load_scale`` is the sum of absolute contributions before cancellation;
        omitting it uses ``abs(rhs)``. Wider real floating loads are preserved.
        Physical moment rows must be supplied explicitly to ``solve``.
        """
        shape = self.rhs.shape
        load = _preserved_array(rhs, shape, "rhs")
        scale = (
            np.abs(load)
            if load_scale is None
            else _preserved_array(load_scale, shape, "load_scale")
        )
        if np.any(scale < 0) or np.any(scale < np.abs(load)):
            raise ValueError("load_scale must bound the absolute reduced load")
        system = self.__class__.__new__(self.__class__)
        system.responses = ()
        system.local_metadata = ()
        system.trace_size = self.trace_size
        system.kernel_offsets = self.kernel_offsets.copy()
        system.matrix = self.matrix
        system.rhs = load
        system.load_scale = scale
        return system

    def solve(
        self,
        *,
        solver: str = "scipy",
        fixed: dict[int, float] | None = None,
        constraints: list[tuple[FloatArray, float]] | None = None,
        factorization: LinearFactorization | None = None,
        refinement_precision: Literal["double", "extended"] = "double",
        rtol: float | None = None,
    ) -> HybridSolution:
        """Solve the saddle system, enforce gauges and verify original equations.

        A singular skeleton typically indicates missing gauges or a trace space
        richer than the local response space. Such systems are never repaired by
        an undocumented diagonal perturbation or least-squares solution. For a
        nonsymmetric global operator, the symmetric gauge augmentation requires
        constraint rows pairing with both left and right nullspaces.
        A supplied ``factorization`` must match the complete constrained matrix
        exactly. It is reused without transferring ownership, allowing repeated
        sources and boundary values through an offline/online preparation.
        Explicit ``refinement_precision="extended"`` retains correction digits
        in trace, coarse coefficients and reconstructed fields. Backend factors
        remain double precision. ``rtol`` selects the global linear-solver
        relative residual tolerance; ``None`` uses 1e-10 for a fresh solve or
        the tolerance of a supplied factorization. An explicit tolerance must
        match that prepared factorization. Physical compatibility and local
        reconstruction criteria remain unchanged. Extended arithmetic requires
        a wider NumPy long-double type.
        The original-equation residual uses the absolute local load contributions
        before assembly, and the net prescribed action before elimination. This
        accounts for cancellation roundoff without applying a relative tolerance
        to cancelled large prescribed fluxes or to artificial gauge equations.
        Prescribed elimination also admits its componentwise floating-point
        bound ``gamma_(m+1) * abs(A_fixed) @ abs(x_fixed)`` per physical row,
        where ``m`` is its number of terms and gamma uses double precision.
        A gauged declared kernel can additionally use the componentwise
        evaluation bound of its retained physical rows. This bound is consulted
        only after the usual compatibility test fails; it changes neither the
        represented operator nor the raw residual or previously accepted result.
        General retained modes receive no kernel allowance.
        """
        if refinement_precision not in {"double", "extended"}:
            raise ValueError("refinement_precision must be double or extended")
        if rtol is not None and (np.iscomplexobj(rtol) or not np.isfinite(rtol) or rtol <= 0):
            raise ValueError("global rtol must be finite and positive")
        if factorization is not None and rtol is not None and rtol != factorization.rtol:
            raise ValueError("explicit rtol must match the prepared factorization tolerance")
        solve_rtol = 1e-10 if rtol is None else float(rtol)
        extended = refinement_precision == "extended"
        if extended and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
            raise SolverUnavailableError("extended refinement requires a wider long-double type")
        fixed = {} if fixed is None else fixed
        n = len(self.rhs)
        solution = np.zeros(n, dtype=np.longdouble if extended else float)
        indices = np.array(list(fixed), dtype=int)
        if (
            any(
                not isinstance(i, (int, np.integer))
                or isinstance(i, bool)
                or i < 0
                or i >= self.trace_size
                for i in fixed
            )
            or not np.isfinite(list(fixed.values())).all()
        ):
            raise ValueError("fixed DOFs must be valid trace indices with finite values")
        solution[indices] = list(fixed.values())
        free = np.setdiff1d(np.arange(n), indices)
        matrix = self.matrix[free][:, free]
        prescribed_action = (self.matrix @ solution)[free]
        prescribed_matrix = self.matrix[free][:, indices]
        rounding_units = (prescribed_matrix.getnnz(axis=1) + 1) * np.finfo(float).eps
        prescribed_roundoff = (rounding_units / (1 - rounding_units)) * (
            abs(prescribed_matrix) @ np.abs(solution[indices])
        )
        rhs = self.rhs[free] - prescribed_action
        physical_matrix, physical_rhs = matrix, rhs
        count = len(constraints) if constraints else 0
        if constraints:
            array = _preserved_array if extended else _array
            rows = np.array([array(row, (n,), "constraint") for row, _ in constraints])
            targets = (
                array([target for _, target in constraints], (count,), "constraint targets")
                - rows @ solution
            )
            matrix = sparse.bmat(
                [
                    [matrix, sparse.csc_matrix(rows[:, free].T)],
                    [sparse.csc_matrix(rows[:, free]), None],
                ],
                format="csc",
            )
            rhs = np.r_[rhs, targets]
        if factorization is not None and (not len(rhs) or not factorization.matches(matrix)):
            raise ValueError("prepared factorization does not match the constrained global matrix")
        if len(rhs) and solver != "scipy" and factorization is None:
            validate_invertible(matrix)
        if factorization is not None:
            solved = (
                factorization.solve(rhs, refinement_precision="extended")
                if extended
                else factorization.solve(rhs)
            )
        else:
            if extended:
                solved = (
                    solve_linear(
                        matrix, rhs, solver=solver, rtol=solve_rtol, refinement_precision="extended"
                    )
                    if len(rhs)
                    else np.empty(0, dtype=np.longdouble)
                )
            else:
                solved = (
                    solve_linear(matrix, rhs, solver=solver, rtol=solve_rtol)
                    if len(rhs)
                    else np.empty(0)
                )
        solution[free] = solved[: len(free)]
        multipliers = solved[len(free) :]
        action_scale = abs(physical_matrix) @ np.abs(solution[free])
        norm = (
            np.linalg.norm
            if extended or self.matrix.dtype.itemsize > np.dtype(float).itemsize
            else linalg.norm
        )
        physical_defect = physical_matrix @ solution[free] - physical_rhs
        raw_residual_norm = float(norm(physical_defect))
        physical_scale = max(
            norm(self.rhs[free]),
            norm(self.load_scale[free]),
            norm(prescribed_action),
            norm(physical_rhs),
            norm(action_scale),
            norm(prescribed_roundoff) / 1e-8,
            np.finfo(float).tiny,
        )
        raw_residual = float(norm(physical_defect) / physical_scale)
        residual = raw_residual
        if residual > 1e-8 and constraints:
            bounds = np.zeros(n)
            for cell, response in enumerate(self.responses):
                first, last = self.kernel_offsets[cell : cell + 2]
                field = response.reconstruct(
                    solution[response.problem.trace_dofs], solution[first:last]
                )
                bounds[first:last] = response.kernel_roundoff_bound(field)
            # This is an absolute evaluation-error certificate, not a relaxed
            # relative tolerance on the small, cancelled retained equation.
            certified_defect = np.maximum(abs(physical_defect) - bounds[free], 0)
            residual = float(norm(certified_defect) / physical_scale)
        if residual > 1e-8:
            raise ValueError("incompatible data: gauge changed physical equations")
        coarse = tuple(
            solution[self.kernel_offsets[i] : self.kernel_offsets[i + 1]].copy()
            for i in range(len(self.kernel_offsets) - 1)
        )
        fields = (
            tuple(
                response.reconstruct(solution[response.problem.trace_dofs], c)
                for response, c in zip(self.responses, coarse, strict=True)
            )
            if self.responses
            else ()
        )
        return HybridSolution(
            solution[: self.trace_size],
            coarse,
            fields,
            residual,
            multipliers,
            raw_residual,
            raw_residual_norm,
        )


def _condense(
    problem: LocalProblem,
    solver: str,
    refinement_precision: Literal["double", "extended"] = "double",
) -> LocalResponse:
    """Provide a spawn-pickleable worker for local elimination."""
    return problem.condense(solver, refinement_precision=refinement_precision)


def _assemble_and_condense(
    item: Any,
    *,
    factory: Callable[[Any], LocalProblem | LocalAssembly],
    solver: str,
    refinement_precision: Literal["double", "extended"] = "double",
) -> tuple[LocalResponse, Any]:
    """Keep factory construction and one numerical condensation in the same worker."""
    assembled = factory(item)
    if isinstance(assembled, LocalAssembly):
        problem, metadata = assembled.problem, assembled.metadata
    elif isinstance(assembled, LocalProblem):
        problem, metadata = assembled, None
    else:
        raise TypeError("local factory must return LocalProblem or LocalAssembly")
    return problem.condense(solver, refinement_precision=refinement_precision), metadata
