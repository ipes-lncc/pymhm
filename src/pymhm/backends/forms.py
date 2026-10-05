"""Generic optional DOLFINx assembly of user-declared UFL blocks and pairings.

Rows belong to the test arguments and columns to the trial arguments. Each
block is assembled independently; symmetry, boundary conditions, skeleton
orientation and trial/test compatibility are explicit caller responsibilities.
The module imports no optional FEM, UFL, MPI or PETSc package at import time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import import_module
from types import ModuleType
from typing import Any, Literal, cast

import numpy as np
from scipy import sparse

from pymhm.core.validation import FloatArray


def _require(module: str) -> ModuleType:
    """Import an optional native module only for an actual assembly invocation."""
    try:
        return import_module(module)
    except (ImportError, OSError) as error:
        raise ImportError(
            f"Cannot load {module}. Install compatible DOLFINx, Basix and UFL "
            "packages, or use the Pixi fem environment."
        ) from error


def _shape(shape: tuple[int, ...] | None) -> tuple[int, ...] | None:
    """Validate optional rank-one or rank-two coefficient dimensions."""
    if shape is None:
        return None
    result = tuple(shape)
    if len(result) not in {1, 2} or any(
        isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 0
        for size in result
    ):
        raise ValueError("shape must contain one or two nonnegative integer dimensions")
    return tuple(int(size) for size in result)


def _zero(form: Any) -> bool:
    """Identify UFL zero integrals and argument-preserving ZeroBaseForm objects."""
    integrals = getattr(form, "integrals", None)
    if callable(integrals):
        return all(integral.integrand() == 0 for integral in integrals())
    ufl = _require("ufl")
    return isinstance(form, ufl.ZeroBaseForm)


def _form_domains(form: Any) -> tuple[Any, ...]:
    """Read integral domains, or argument domains of an argument-preserving base form."""
    return tuple(
        form.ufl_domains()
        if callable(getattr(form, "integrals", None))
        else tuple(argument.ufl_function_space().ufl_domain() for argument in form.arguments())
    )


def _serial_domains(form: Any) -> None:
    """Check integral domains or base-form argument domains before collective JIT."""
    domains = _form_domains(form)
    for domain in domains:
        mesh = domain.ufl_cargo()
        if mesh is not None and mesh.comm.size != 1:
            raise ValueError("DOLFINx form domains must use a single-rank communicator (COMM_SELF)")


def _space_size(space: Any) -> int:
    """Count owned scalar coefficients in a native single-rank function space."""
    try:
        return int(space.dofmap.index_map.size_local * space.dofmap.index_map_bs)
    except AttributeError as error:
        raise ValueError(
            "shape is required when zero-form dimensions cannot be inferred"
        ) from error


def _real_coefficients(value: Any, *, rank: int) -> FloatArray:
    """Copy finite real native coefficients without silently discarding imaginary parts."""
    if np.iscomplexobj(value):
        raise ValueError("the generic DOLFINx form adapter requires real-valued forms")
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != rank or not np.isfinite(array).all():
        raise ValueError("assembled coefficients must be finite with the declared form rank")
    return array


def _assemble_compiled(
    compiled: Any,
    rank: int,
    expected: tuple[int, ...] | None,
    buffer: Any = None,
) -> tuple[FloatArray | sparse.csr_matrix, Any]:
    """Assemble into a reusable native buffer and return an independent coefficient copy.

    Native assembly accumulates entries. Reused buffers are therefore zeroed
    before every invocation, and coefficients/constants are freshly packed by
    DOLFINx. No numerical matrix, load or material-dependent response is cached.
    """
    fem = _require("dolfinx.fem")
    if compiled.mesh.comm.size != 1:
        raise ValueError("DOLFINx form domains must use a single-rank communicator (COMM_SELF)")
    if rank == 1:
        if buffer is None:
            buffer = fem.assemble_vector(compiled)
        else:
            buffer.array.fill(0)
            fem.assemble_vector(buffer.array, compiled)
        buffer.scatter_reverse(_require("dolfinx.la").InsertMode.add)
        result: FloatArray | sparse.csr_matrix = _real_coefficients(buffer.array, rank=1)
    else:
        if buffer is None:
            buffer = fem.assemble_matrix(compiled)
        else:
            buffer.data.fill(0)
            fem.assemble_matrix(buffer, compiled)
        buffer.scatter_reverse()
        native = buffer.to_scipy()
        if np.iscomplexobj(native.data):
            raise ValueError("the generic DOLFINx form adapter requires real-valued forms")
        result = sparse.csr_matrix(native, dtype=np.float64, copy=True)
        if not np.isfinite(result.data).all():
            raise ValueError("assembled coefficients must be finite with the declared form rank")
    if expected is not None and result.shape != expected:
        raise ValueError(
            f"assembled shape {result.shape} differs from the declared shape {expected}"
        )
    return result, buffer


def assemble_form(
    form: Any,
    *,
    shape: tuple[int, ...] | None = None,
    form_compiler_options: Mapping[str, Any] | None = None,
    jit_options: Mapping[str, Any] | None = None,
    entity_maps: Mapping[Any, Any] | None = None,
) -> FloatArray | sparse.csr_matrix:
    """Assemble one linear or bilinear UFL form into owned real coefficients.

    A linear form returns a binary64 vector. A bilinear form returns a binary64
    SciPy CSR matrix with shape ``(test_dofs, trial_dofs)``. Distinct spaces and
    rectangular matrices are accepted; a later condensation or solve decides
    whether its chosen pivot and full equations are algebraically admissible.
    Linear forms using either argument number zero or one are supported.

    ``shape`` checks the declared coefficient dimensions. Supply it for zero
    forms whose arguments were simplified away. Argument-preserving UFL zero
    forms can infer dimensions from their native function spaces. A zero form
    returns zeros without importing DOLFINx. Rank-zero nonzero functionals and
    higher-rank forms are outside this vector/matrix contract.

    All form domains must use a single-rank communicator, normally COMM_SELF.
    Compiler and JIT options are copied before delegation; ``entity_maps``
    explicitly describes cross-mesh integration to DOLFINx. This function
    infers no trace map, normal orientation, quadrature, boundary elimination,
    gauge or transposed coupling. Native matrices, vectors and compiled-form
    references are released before returning their independent array copies.
    """
    expected = _shape(shape)
    arguments = tuple(form.arguments())
    rank = len(arguments)
    zero = _zero(form)
    _serial_domains(form)
    if zero:
        if expected is None:
            if rank not in {1, 2}:
                raise ValueError(
                    "shape is required for zero forms without linear/bilinear arguments"
                )
            expected = tuple(_space_size(argument.ufl_function_space()) for argument in arguments)
        elif rank and rank != len(expected):
            raise ValueError("shape rank does not match the declared form arguments")
        return np.zeros(expected) if len(expected) == 1 else sparse.csr_matrix(expected)
    if rank not in {1, 2}:
        raise ValueError("form must be linear or bilinear")
    if expected is not None and len(expected) != rank:
        raise ValueError("shape rank does not match the declared form arguments")
    fem = _require("dolfinx.fem")
    compiled = None
    assembled = None
    result: FloatArray | sparse.csr_matrix
    try:
        compiled = fem.form(
            form,
            dtype=np.float64,
            form_compiler_options=dict(form_compiler_options or {}),
            jit_options=dict(jit_options or {}),
            entity_maps=None if entity_maps is None else dict(entity_maps),
        )
        result, assembled = _assemble_compiled(compiled, rank, expected)
        return result
    finally:
        del assembled
        del compiled


def assemble_pairing(
    forms: Iterable[Any],
    *,
    axis: Literal["columns", "rows"] = "columns",
    size: int | None = None,
    form_compiler_options: Mapping[str, Any] | None = None,
    jit_options: Mapping[str, Any] | None = None,
    entity_maps: Mapping[Any, Any] | None = None,
) -> FloatArray:
    """Assemble ordered linear basis pairings as independent columns or rows.

    Every argument-preserving form must use the same native function space.
    ``axis='columns'`` returns ``(size, number_of_forms)`` for a local test-side
    coupling. ``axis='rows'`` returns ``(number_of_forms, size)`` for a global
    equation paired with the local trial field. Argument number one is accepted
    natively, so a trial-side linear form needs no symbolic replacement.

    ``size`` provides the local dimension for an empty sequence or an initial
    zero form whose arguments were erased by UFL. Otherwise the first assembled
    vector determines it. Caller-supplied signs and form order are preserved;
    row and column pairings are never inferred from one another. Compilation
    options and cross-mesh maps have the same meaning as in ``assemble_form``.
    """
    if axis not in {"columns", "rows"}:
        raise ValueError("axis must be columns or rows")
    if size is not None:
        size = cast(tuple[int, ...], _shape((size,)))[0]
    declared = tuple(forms)
    space = None
    for form in declared:
        arguments = tuple(form.arguments())
        if len(arguments) > 1:
            raise ValueError("each pairing form must be linear")
        if arguments:
            current = arguments[0].ufl_function_space()
            if space is not None and current != space:
                raise ValueError("pairing forms must use the same function space")
            space = current
    vectors = []
    for form in declared:
        value = cast(
            FloatArray,
            assemble_form(
                form,
                shape=None if size is None else (size,),
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            ),
        )
        size = len(value)
        vectors.append(value)
    if size is None:
        raise ValueError("size is required for an empty pairing")
    result = np.column_stack(vectors) if vectors else np.empty((size, 0))
    return result if axis == "columns" else result.T.copy()
