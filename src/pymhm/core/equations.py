"""User-defined local and global variational blocks, independent of a physical model.

Rows are test coordinates and columns are trial coordinates. The two equations
are ``A u + B lambda = f`` and ``C u + D lambda = g``. In particular, the sign
of C is declared by the user rather than inferred from a method or equation.
UFL forms are assembled by an optional native adapter; arrays are already
assembled forms. Neither representation selects a PDE or a stabilization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from scipy import sparse

from pymhm.core.contracts import LocalProblem
from pymhm.core.validation import FloatArray, IntArray
from pymhm.core.variational import _real_data


@dataclass(frozen=True)
class Equation:
    """A bilinear form ``a`` and linear right-hand side ``L`` in declared coordinates.

    A literal zero denotes a zero form with dimensions supplied by the problem.
    A UFL form keeps its trial/test spaces, mesh, coefficients and quadrature.
    An array or sparse matrix denotes an already assembled form. Global forms
    act on trace coordinates followed by the ordered retained cell coordinates.
    """

    a: Any
    L: Any


@dataclass(frozen=True)
class LinearForms:
    """Linear forms representing columns B or rows C in an explicit trace basis.

    Every entry represents one trace basis function, including its orientation
    and physical units. An empty sequence requires the problem's dimensions.
    This representation avoids requiring cross-mesh UFL assembly: each linear
    form lives on its local integration mesh. It also accepts numerical vectors.
    """

    forms: tuple[Any, ...]
    axis: Literal["columns", "rows"]

    def __post_init__(self) -> None:
        """Freeze the sequence and reject an ambiguous pairing direction."""
        if self.axis not in {"columns", "rows"}:
            raise ValueError("linear form axis must be columns or rows")
        object.__setattr__(self, "forms", tuple(self.forms))


def columns(*forms: Any) -> LinearForms:
    """Declare one local linear form for each global trial coordinate."""
    return LinearForms(forms, "columns")


def rows(*forms: Any) -> LinearForms:
    """Declare one local linear form for each global test coordinate."""
    return LinearForms(forms, "rows")


class FormCompiler(Protocol):
    """Assemble a real rank-one or rank-two form without changing its coordinates."""

    def __call__(self, form: Any, shape: tuple[int, ...] | None = None) -> Any:
        """Return an owned coefficient array or sparse operator of the declared shape."""
        ...


def compile_form(form: Any, shape: tuple[int, ...] | None = None) -> Any:
    """Assemble arrays, linear pairings or native UFL without importing unused FEM.

    Sparse matrices remain sparse. Literal bilinear zeros use CSC storage;
    literal linear zeros use dense vectors. Finite wider real array precision is
    retained. A nonzero scalar is not silently broadcast to a vector or operator. UFL
    assembly currently requires a real serial DOLFINx mesh. Other backends can
    supply a callable compiler with this same form/shape contract.
    """
    if isinstance(form, LinearForms):
        if shape is None or len(shape) != 2:
            raise ValueError("linear form pairings require a matrix shape")
        count, size = shape[::-1] if form.axis == "columns" else shape
        if len(form.forms) != count:
            raise ValueError("one linear form per pairing coordinate is required")
        values = [compile_form(value, (size,)) for value in form.forms]
        if not values:
            return np.zeros(shape)
        return np.column_stack(values) if form.axis == "columns" else np.vstack(values)
    if hasattr(form, "arguments") and hasattr(form, "integrals"):
        from pymhm.backends.forms import assemble_form

        return assemble_form(form, shape=shape)
    if sparse.issparse(form):
        result = sparse.csc_matrix(form, copy=True)
        if np.iscomplexobj(result.data) or not np.isfinite(result.data).all():
            raise ValueError("forms must have finite real coefficients")
        if shape is not None and result.shape != shape:
            raise ValueError("assembled form does not have the declared shape")
        return result
    value = np.asarray(form)
    if np.iscomplexobj(value):
        raise ValueError("forms must have real coefficients")
    if value.ndim == 0 and value == 0 and shape is not None:
        return sparse.csc_matrix(shape) if len(shape) == 2 else np.zeros(shape)
    if value.ndim not in {1, 2} or (shape is not None and value.shape != shape):
        raise ValueError("assembled form must be a vector or matrix of the declared shape")
    return _real_data(value, "form")


def _dofs(value: Any, name: str) -> IntArray:
    """Copy distinct nonnegative integer coordinates, preserving their declared order."""
    result = np.asarray(value)
    if result.size == 0 and result.ndim == 1:
        result = result.astype(np.int64)
    if (
        result.ndim != 1
        or result.dtype.kind not in "iu"
        or np.any(result < 0)
        or np.any(result > np.iinfo(np.int64).max)
        or len(np.unique(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain distinct nonnegative integer coordinates")
    result = np.array(result, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LocalEquations:
    """Define both local equations and their contribution to the global balance.

    ``a, L, b`` express ``a(u,v)+b(lambda,v)=L(v)``. ``c, d, g``
    express ``c(u,mu)+d(lambda,mu)=g(mu)``. ``dofs`` and ``test_dofs``
    map the local trial/test trace bases to the global coordinates. They can
    differ in order and length. Basis functions include normal orientation;
    their signs are not inferred. ``moments`` and ``test_moments`` are column
    pairings with explicit retained trial/test modes. No kernel is discovered
    automatically and no stability or consistency theorem is implied.

    Scalar, vector and mixed unknowns use this same contract. A method with
    several local fields can use a mixed UFL space or a block matrix. Native
    objects must remain worker-owned, with optional reuse between provider
    invocations and explicit ``close()`` cleanup at worker shutdown, when
    using spawn workers; ``metadata`` must then contain only picklable data.
    A 0-by-0 ``a`` declares no eliminated local coordinates: ``d`` and ``g``
    then contribute the complete face equation without a local factorization.
    """

    a: Any
    L: Any
    b: Any
    c: Any
    dofs: Any
    test_dofs: Any = None
    d: Any = 0
    g: Any = 0
    kernel: Any = None
    moments: Any = None
    coarse_basis: Any = None
    left_kernel: Any = None
    test_basis: Any = None
    test_moments: Any = None
    metadata: Any = None
    field_data: tuple[Any, ...] = ()
    trace_binding: Any = None

    def __post_init__(self) -> None:
        """Freeze explicit maps without assembling forms or inspecting native spaces."""
        trial = _dofs(self.dofs, "dofs")
        test = trial if self.test_dofs is None else _dofs(self.test_dofs, "test_dofs")
        object.__setattr__(self, "dofs", trial)
        object.__setattr__(self, "test_dofs", test)


@dataclass(frozen=True)
class CompiledLocalEquations:
    """Numerical local contract and its uncondensed direct global terms.

    The signed balance C is stored as ``test_coupling=-C.T`` in the shared
    condensation owner. ``matrix`` and ``load`` are D and g in the union trace
    numbering, and are added once during ordered global reduction.
    """

    problem: LocalProblem
    matrix: FloatArray
    load: FloatArray
    metadata: Any = None
    field_data: tuple[Any, ...] = ()
    trace_binding: Any = None


def compile_local_equations(
    equations: LocalEquations, compiler: FormCompiler = compile_form
) -> CompiledLocalEquations:
    """Compile four variational blocks and reuse the checked local condensation contract.

    Trial and test trace maps embed into their ordered union. This permits
    rectangular B/C/D pairings while the complete global problem remains square.
    Retained compatibility equations preserve their explicit test basis and
    physical moments. Already assembled matrices do not pass through UFL.
    """
    if not isinstance(equations, LocalEquations):
        raise TypeError("equations must be LocalEquations")
    a = compiler(equations.a)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("the local trial and test operator must be square")
    n = a.shape[0]
    trial, test = equations.dofs, equations.test_dofs
    union = np.array(list(dict.fromkeys([*trial, *test])), dtype=np.int64)
    indices = {int(value): i for i, value in enumerate(union)}
    trial_indices = np.array([indices[int(value)] for value in trial], dtype=np.int64)
    test_indices = np.array([indices[int(value)] for value in test], dtype=np.int64)
    b = compiler(equations.b, (n, len(trial)))
    c = compiler(equations.c, (len(test), n))
    b = b.toarray() if sparse.issparse(b) else b
    c = c.toarray() if sparse.issparse(c) else c
    coupling = np.zeros((n, len(union)), dtype=np.result_type(b, c))
    test_coupling = np.zeros_like(coupling)
    coupling[:, trial_indices] = b
    test_coupling[:, test_indices] = -c.T
    basis = equations.kernel if equations.coarse_basis is None else equations.coarse_basis
    width = 0 if basis is None else np.asarray(basis).shape[-1]
    moments = None if equations.moments is None else compiler(equations.moments, (n, width))
    test_moments = (
        None if equations.test_moments is None else compiler(equations.test_moments, (n, width))
    )
    if moments is not None and sparse.issparse(moments):
        moments = moments.toarray()
    if test_moments is not None and sparse.issparse(test_moments):
        test_moments = test_moments.toarray()
    problem = LocalProblem(
        a,
        coupling,
        compiler(equations.L, (n,)),
        union,
        kernel=equations.kernel,
        constraints=moments,
        coarse_basis=equations.coarse_basis,
        test_coupling=test_coupling,
        left_kernel=equations.left_kernel,
        test_basis=equations.test_basis,
        test_constraints=test_moments,
    )
    direct = compiler(equations.d, (len(test), len(trial)))
    direct = direct.toarray() if sparse.issparse(direct) else direct
    matrix = np.zeros((len(union), len(union)), dtype=np.result_type(direct, float))
    matrix[np.ix_(test_indices, trial_indices)] = direct
    load = compiler(equations.g, (len(test),))
    global_load = np.zeros(len(union), dtype=np.result_type(load, float))
    global_load[test_indices] = load
    return CompiledLocalEquations(
        problem,
        matrix,
        global_load,
        equations.metadata,
        equations.field_data,
        equations.trace_binding,
    )
