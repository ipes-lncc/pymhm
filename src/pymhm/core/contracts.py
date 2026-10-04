"""Validated local equations, cached responses and coefficient data contracts."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.core.validation import FloatArray, IntArray
from pymhm.linalg.linear import (
    LinearFactorization,
    _accurate_residual,
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
        """Check finite square blocks and the declared kernel, including a zero local space."""
        if np.iscomplexobj(matrix):
            raise ValueError("local matrix must be real")
        matrix = sparse.csc_matrix(matrix, dtype=float, copy=True)
        n = matrix.shape[0]
        if matrix.shape != (n, n) or not np.isfinite(matrix.data).all():
            raise ValueError("local matrix must be finite and square")
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
        """Delegate constrained source/trace elimination to :func:`condense_local`."""
        from pymhm.core.condensation import condense_local

        return condense_local(self, solver, refinement_precision=refinement_precision)

    def condensation_system(self) -> tuple[Any, FloatArray]:
        """Return the constrained operator and source/trace/retained right sides."""
        from pymhm.core.condensation import local_condensation_system

        return local_condensation_system(self)

    def condensation_matrix(self) -> Any:
        """Return the scaled constraint operator without allocating response columns."""
        from pymhm.core.condensation import local_condensation_matrix

        return local_condensation_matrix(self)

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
        """Reconstruct physical fields through :func:`reconstruct_local`."""
        from pymhm.core.reconstruction import reconstruct_local

        return reconstruct_local(
            self,
            trace,
            coarse,
            retained_basis=retained_basis,
            solver=solver,
            factorization=factorization,
            refinement_precision=refinement_precision,
        )

    def response_from_solution(self, solution: Any) -> "LocalResponse":
        """Decode finite real responses, preserving explicitly retained wider precision."""
        from pymhm.core.condensation import local_response_from_solution

        return local_response_from_solution(self, solution)

    def condensed_load(
        self, source_response: Any, *, load: Any = None, corrected_retained: bool | None = None
    ) -> FloatArray:
        """Project executed source responses into the original reduced Petrov right side."""
        from pymhm.core.reconstruction import local_condensed_load

        return local_condensed_load(
            self, source_response, load=load, corrected_retained=corrected_retained
        )

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
        """Evaluate cached lifts and retained modes through :func:`reconstruct_response`."""
        from pymhm.core.reconstruction import reconstruct_response

        return reconstruct_response(self, trace, coarse)

    def kernel_roundoff_bound(self, field: FloatArray) -> FloatArray:
        """Bound declared-kernel evaluation through the shared reconstruction owner."""
        from pymhm.core.reconstruction import kernel_roundoff_bound

        return kernel_roundoff_bound(self, field)

    def global_load(self) -> FloatArray:
        """Return source-dependent reduced loads without rebuilding matrix blocks."""
        return self.problem.condensed_load(
            self.source, corrected_retained=self.coarse_vectors is not None
        )

    def global_contribution(self, coarse_dofs: Any) -> tuple[IntArray, FloatArray, FloatArray]:
        """Return the oriented cell block and load; see `local_global_contribution`."""
        from pymhm.core.contributions import local_global_contribution

        return local_global_contribution(self, coarse_dofs)


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
