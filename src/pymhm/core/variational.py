"""Explicit local forms and the scoped algebraic global hybrid form.

Local expressions are opaque to the portable core. A supplied compiler owns
their finite-element interpretation; the global form uses the existing hybrid
trace/retained coordinates and does not compile arbitrary skeleton UFL forms.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar

import numpy as np

from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.validation import FloatArray, IntArray

Item = TypeVar("Item")
LocalProvider: TypeAlias = Callable[[Item], LocalProblem | LocalAssembly]


def _real_data(value: Any, name: str) -> FloatArray:
    """Copy finite real coefficients without discarding supplied wider digits."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain finite real coefficients")
    result = np.array(raw, dtype=np.result_type(raw.dtype, float), copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real coefficients")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LocalForm:
    """Describe ``a(u,v) + sum_j lambda_j b_j(v) = L(v)`` on one cell.

    ``a`` and ``L`` are compiler-owned expressions, for example UFL forms.
    ``trace_forms`` are the signed linear forms ``b_j`` in exactly the order of
    distinct nonnegative global ``trace_dofs``. Outward/global-normal signs,
    spaces, quadrature and essential boundary elimination are explicit inputs.
    No trace orientation or boundary convention is inferred.

    ``kernel`` declares literal nullspace coefficients; ``coarse_basis`` instead
    retains arbitrary modes without a nullspace claim. They are mutually
    exclusive. ``moment_forms`` has one linear physical moment per retained
    mode; omitting it requests the compiler's coefficient-orthogonality default.
    The assembled ``LocalProblem`` checks operator/nullspace compatibility and
    moment pairing. Those algebraic checks do not establish inf-sup stability.
    Forms are stored as tuples and numerical maps/bases are copied read-only.
    """

    a: Any
    L: Any
    trace_forms: tuple[Any, ...]
    trace_dofs: IntArray
    kernel: FloatArray | None = field(default=None, kw_only=True)
    moment_forms: tuple[Any, ...] | None = field(default=None, kw_only=True)
    coarse_basis: FloatArray | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Freeze the declared order and validate maps and literal mode arrays."""
        traces = tuple(self.trace_forms)
        dofs = np.asarray(self.trace_dofs)
        if dofs.size == 0:
            dofs = dofs.astype(np.int64)
        if (
            dofs.ndim != 1
            or not np.issubdtype(dofs.dtype, np.integer)
            or np.any(dofs < 0)
            or np.any(dofs > np.iinfo(np.int64).max)
            or len(np.unique(dofs)) != len(dofs)
            or len(dofs) != len(traces)
        ):
            raise ValueError("trace_dofs must be distinct nonnegative int64 indices, one per form")
        dofs = dofs.astype(np.int64, copy=True)
        dofs.setflags(write=False)
        object.__setattr__(self, "trace_forms", traces)
        object.__setattr__(self, "trace_dofs", dofs)
        if self.kernel is not None and self.coarse_basis is not None:
            raise ValueError("kernel and coarse_basis are mutually exclusive")
        width = 0
        for name in ("kernel", "coarse_basis"):
            value = getattr(self, name)
            if value is not None:
                basis = _real_data(value, name)
                if basis.ndim != 2 or basis.shape[0] == 0:
                    raise ValueError(f"{name} must be a matrix with nonempty local dimension")
                width = basis.shape[1]
                object.__setattr__(self, name, basis)
        if self.moment_forms is not None:
            moments = tuple(self.moment_forms)
            if len(moments) != width:
                raise ValueError("moment_forms must contain one form per retained mode")
            object.__setattr__(self, "moment_forms", moments)


def compile_local_forms(
    forms: LocalForm, compiler: Callable[[LocalForm], LocalProblem]
) -> LocalProblem:
    """Compile one record while preserving its trace map and literal basis.

    A compiler is an ordinary callable; no inheritance or optional backend is
    required. It must return a validated ``LocalProblem``. The caller owns any
    compiler-native resources and must release them before returning arrays.
    Numerical operator and integration accuracy remain the compiler's concern.
    A changed map, undeclared mode or equivalent rotated basis is rejected.
    """
    if not isinstance(forms, LocalForm):
        raise TypeError("forms must be a LocalForm")
    problem = compiler(forms)
    if not isinstance(problem, LocalProblem):
        raise TypeError("compiler must return a LocalProblem")
    if not np.array_equal(problem.trace_dofs, forms.trace_dofs):
        raise ValueError("compiler changed the declared trace_dofs")
    if forms.kernel is None:
        if problem.kernel.shape[1]:
            raise ValueError("compiler introduced an undeclared kernel")
    elif not np.array_equal(problem.kernel, forms.kernel):
        raise ValueError("compiler changed the declared kernel basis")
    declared = forms.kernel if forms.coarse_basis is None else forms.coarse_basis
    if declared is None:
        if problem.coarse_basis.shape[1]:
            raise ValueError("compiler introduced undeclared retained modes")
    elif not np.array_equal(problem.coarse_basis, declared):
        raise ValueError("compiler changed the declared retained basis")
    return problem


@dataclass(frozen=True)
class GlobalForm:
    """Declare the sum of local condensed hybrid forms in explicit coordinates.

    Trace coordinates occupy ``[0, trace_size)``. Retained coordinates follow
    in cell order with widths ``coarse_sizes``. For each cell, its response's
    ``global_contribution`` supplies the bilinear block and linear load; the
    shared hybrid assembler sums these records on their declared indices.
    This is an algebraic global form, not an arbitrary UFL skeleton compiler.

    ``boundary_load`` is a trace vector subtracted from the assembled RHS with
    the ``HybridSystem`` boundary convention. ``fixed_trace`` prescribes trace
    coefficients, for example oriented normal fluxes. Each ``constraints``
    entry is ``(row, target)`` imposing ``row.T @ (lambda,c) = target`` in the
    full reduced coordinates. Supply physical pressure/rigid-motion gauges;
    neither a gauge nor a stability condition is inferred. Local field moments
    can be converted to these rows by ``HybridSystem.mean_constraint``.
    Wider floating coefficients are preserved; records are copied on creation.
    """

    trace_size: int
    coarse_sizes: tuple[int, ...]
    boundary_load: FloatArray | None = None
    fixed_trace: Mapping[int, float] | None = None
    constraints: tuple[tuple[FloatArray, float], ...] = ()

    def __post_init__(self) -> None:
        """Validate the ordered coordinate partition, loads and physical rows."""
        sizes = tuple(self.coarse_sizes)
        for width in (self.trace_size, *sizes):
            if (
                isinstance(width, (bool, np.bool_))
                or not isinstance(width, (int, np.integer))
                or width < 0
            ):
                raise ValueError("trace_size and coarse_sizes must be nonnegative integers")
        if not sizes:
            raise ValueError("coarse_sizes must declare at least one local cell")
        object.__setattr__(self, "trace_size", int(self.trace_size))
        object.__setattr__(self, "coarse_sizes", tuple(int(width) for width in sizes))
        size = self.trace_size + sum(self.coarse_sizes)
        if self.boundary_load is not None:
            boundary = _real_data(self.boundary_load, "boundary_load")
            if boundary.shape != (self.trace_size,):
                raise ValueError("boundary_load must have shape (trace_size,)")
            object.__setattr__(self, "boundary_load", boundary)
        fixed = {} if self.fixed_trace is None else dict(self.fixed_trace)
        for index, value in fixed.items():
            if (
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or not 0 <= index < self.trace_size
            ):
                raise ValueError("fixed_trace keys must be valid trace indices")
            coefficient = _real_data(value, "fixed_trace value")
            if coefficient.ndim != 0:
                raise ValueError("fixed_trace values must be scalar")
            fixed[index] = coefficient.item()
        object.__setattr__(self, "fixed_trace", fixed)
        constraints = []
        for row, target in self.constraints:
            weights = _real_data(row, "constraint row")
            scalar = _real_data(target, "constraint target")
            if weights.shape != (size,) or scalar.ndim != 0:
                raise ValueError("constraints require a full reduced row and scalar target")
            constraints.append((weights, scalar.item()))
        object.__setattr__(self, "constraints", tuple(constraints))
