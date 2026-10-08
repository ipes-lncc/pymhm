"""User-defined four-block forms preserve independent test/trial pairings and maps."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from pymhm import (
    Equation,
    LocalEquations,
    MultiscaleProblem,
    assemble,
    columns,
    compile_form,
    compile_local_equations,
    rows,
)
from pymhm.core.equations import LinearForms


@pytest.mark.parametrize("block", ["b", "c", "d", "L", "g", "moments", "test_moments"])
def test_form_shape_errors_identify_the_mathematical_block(block):
    """Incorrect pairings report their name and intended trial/test coordinate shape."""
    declaration = LocalEquations(
        np.eye(2),
        [0, 0],
        np.ones((2, 1)),
        np.ones((1, 2)),
        [0],
        coarse_basis=np.ones((2, 1)),
        moments=np.ones((2, 1)),
    )
    with pytest.raises(ValueError, match=f"local block '{block}'.*shape"):
        compile_local_equations(replace(declaration, **{block: np.ones((3, 3))}))


def test_custom_compiler_type_errors_keep_their_type_and_block_name():
    """Native/custom form failures retain the exception contract and useful form context."""

    def unsupported(form, shape=None):
        raise TypeError("unsupported declared form")

    with pytest.raises(TypeError, match="local block 'a'.*unsupported"):
        compile_local_equations(LocalEquations([[1]], [0], 0, 0, []), unsupported)


def test_four_blocks_use_explicit_balance_sign_and_independent_trace_maps():
    a = np.array([[2.0, 1.0], [0.0, 3.0]])
    b = np.array([[1.0, 2.0], [3.0, 4.0]])
    c = np.array([[5.0, 6.0]])
    d = np.array([[7.0, 8.0]])
    declaration = LocalEquations(a, [9.0, 10.0], b, c, [2, 0], test_dofs=[1], d=d, g=[11.0])
    compiled = compile_local_equations(declaration)
    p = compiled.problem
    assert_array_equal(p.trace_dofs, [2, 0, 1])
    assert_array_equal(p.coupling, np.column_stack((b, np.zeros(2))))
    assert_array_equal(p.test_coupling, np.column_stack((np.zeros((2, 2)), -c.T)))
    assert_array_equal(compiled.matrix, [[0, 0, 0], [0, 0, 0], [7, 8, 0]])
    assert_array_equal(compiled.load, [0, 0, 11])
    response = p.condense()
    assert_allclose(response.source, np.linalg.solve(a, [9, 10]))
    assert_allclose(response.lifts[:, :2], np.linalg.solve(a, b))


def test_array_pairings_and_zeros_preserve_declared_coordinate_directions():
    assert_array_equal(compile_form(columns([1.0, 2.0], [3.0, 4.0]), (2, 2)), [[1, 3], [2, 4]])
    assert_array_equal(compile_form(rows([1.0, 2.0], [3.0, 4.0]), (2, 2)), [[1, 2], [3, 4]])
    assert compile_form(columns(), (2, 0)).shape == (2, 0)
    assert compile_form(rows(), (0, 3)).shape == (0, 3)
    assert_array_equal(compile_form(0, (2,)), [0, 0])
    matrix = sparse.eye(2, format="csr")
    copied = compile_form(matrix)
    assert copied.format == "csc" and copied is not matrix
    wide = compile_form(np.array([1, 2], dtype=np.longdouble))
    assert wide.dtype == np.longdouble
    assert not wide.flags.writeable
    assert LinearForms([[1]], "columns").forms == ([1],)


def test_declared_retained_petrov_bases_and_moments_reach_shared_owner():
    a = [[0.0, 1.0], [0.0, 2.0]]
    right = np.array([[1.0], [0.0]])
    left = np.array([[-2.0], [1.0]])
    eq = LocalEquations(
        a,
        [0.0, 0.0],
        columns([1.0, 2.0]),
        rows([3.0, 4.0]),
        [0],
        kernel=right,
        left_kernel=left,
        moments=columns([2.0, 1.0]),
        test_moments=columns([-1.0, 2.0]),
        metadata={"basis": "declared"},
    )
    compiled = compile_local_equations(eq)
    assert_array_equal(compiled.problem.left_kernel, left)
    assert_array_equal(compiled.problem.constraints, [[2], [1]])
    assert_array_equal(compiled.problem.test_constraints, [[-1], [2]])
    assert compiled.metadata == {"basis": "declared"}
    ordinary = replace(
        eq, a=np.eye(2), kernel=None, left_kernel=None, coarse_basis=right, test_basis=left
    )
    assert_array_equal(compile_local_equations(ordinary).problem.test_basis, left)


def test_sparse_independent_pairings_match_dense_inputs():
    eq = LocalEquations(np.eye(2), [1, 2], sparse.eye(2), sparse.eye(2), [0, 1], d=sparse.eye(2))
    compiled = compile_local_equations(eq)
    assert_array_equal(compiled.problem.coupling, np.eye(2))
    assert_array_equal(compiled.problem.test_coupling, -np.eye(2))
    assert_array_equal(compiled.matrix, np.eye(2))


@pytest.mark.parametrize(
    "value", [[[-1]], [1.0, 2.0], [-1], [0, 0], [True], np.array([2**63], dtype=np.uint64)]
)
def test_trace_maps_reject_invalid_coordinates(value):
    with pytest.raises(ValueError, match="coordinates"):
        LocalEquations(np.eye(1), [0], [[1]], [[1]], value)


def test_empty_maps_and_readonly_owned_maps():
    dofs = np.array([1, 0])
    eq = LocalEquations(np.eye(2), [0, 0], np.eye(2), np.eye(2), dofs)
    dofs[:] = 3
    assert_array_equal(eq.dofs, [1, 0])
    assert eq.test_dofs is eq.dofs
    with pytest.raises(ValueError, match="read-only"):
        eq.dofs[0] = 2
    plain = LocalEquations(np.eye(2), [1, 2], columns(), rows(), [])
    assert compile_local_equations(plain).problem.coupling.shape == (2, 0)


@pytest.mark.parametrize(
    "form,shape",
    [
        (columns([1]), None),
        (rows([1]), (1,)),
        (columns([1]), (1, 2)),
        ([1], (2,)),
        (2, (2,)),
        ([[1]], (2, 2)),
        (sparse.eye(1), (2, 2)),
        (sparse.csc_matrix([[1j]]), None),
        (sparse.csc_matrix([[np.nan]]), None),
    ],
)
def test_form_dimensions_and_invalid_sparse_data_are_not_reinterpreted(form, shape):
    with pytest.raises(ValueError):
        compile_form(form, shape)


def test_invalid_pairing_axis_and_local_operator_are_rejected():
    with pytest.raises(ValueError, match="axis"):
        LinearForms((), "invalid")
    with pytest.raises(TypeError, match="LocalEquations"):
        compile_local_equations(None)
    for a in ([1, 2], np.zeros((1, 2))):
        with pytest.raises(ValueError, match="square"):
            compile_local_equations(LocalEquations(a, [], 0, 0, []))


def test_ufl_compilation_delegates_the_declared_form_and_shape(monkeypatch):
    import pymhm.backends.forms as native

    class Form:
        def arguments(self):
            return ()

        def integrals(self):
            return ()

    form = Form()
    seen = []
    monkeypatch.setattr(
        native, "assemble_form", lambda value, *, shape: seen.append((value, shape))
    )
    assert compile_form(form, (2, 2)) is None
    assert seen == [(form, (2, 2))]


# Scalar spellings are normalized before dimensions determine CSC allocation.
# Exercise every spelling and empty-axis case without their Cartesian product.
@pytest.mark.parametrize(
    "shape,zero",
    [
        ((3, 5), 0),
        ((3, 5), 0.0),
        ((3, 5), -0.0),
        ((3, 5), np.array(0.0)),
        ((0, 0), 0),
        ((0, 4), 0),
        ((7, 0), 0),
    ],
)
def test_literal_zero_bilinear_forms_are_owned_sparse_operators(shape, zero):
    operator = compile_form(zero, shape)
    assert sparse.isspmatrix_csc(operator)
    assert operator.shape == shape and operator.nnz == 0
    assert operator.dtype == np.float64
    assert operator is not compile_form(zero, shape)


def test_large_literal_zero_operator_never_allocates_a_dense_matrix(monkeypatch):
    """A zero global form scales with the CSC column index, not the matrix area."""
    original_zeros = np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2:
            pytest.fail("a literal bilinear zero must not allocate a dense matrix")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", guarded_zeros)
    operator = compile_form(0, (10_000_000, 75_000))
    assert sparse.isspmatrix_csc(operator)
    assert operator.shape == (10_000_000, 75_000)
    assert not operator.nnz and not np.count_nonzero(operator.indptr)
    assert operator.indptr.nbytes < 1_000_000


@pytest.mark.parametrize("shape", [(0,), (5,)])
def test_literal_zero_loads_keep_dense_vector_storage(shape):
    load = compile_form(0, shape)
    assert isinstance(load, np.ndarray) and load.shape == shape
    assert load.dtype == np.float64
    assert_array_equal(load, np.zeros(shape))


@pytest.mark.parametrize("shape,zero", [((3, 5), 0j), ((0, 0), np.array(0j)), ((5,), 0j)])
def test_complex_literal_zero_rejection_precedes_sparse_allocation(shape, zero):
    with pytest.raises(ValueError, match="real"):
        compile_form(zero, shape)


def test_explicit_dense_zero_form_keeps_its_declared_representation():
    form = np.zeros((2, 3), dtype=np.longdouble)
    compiled = compile_form(form, form.shape)
    assert isinstance(compiled, np.ndarray)
    assert compiled.dtype == form.dtype and not compiled.flags.writeable
    assert_array_equal(compiled, form)
    for form in (0, 1):
        with pytest.raises(ValueError, match="vector or matrix"):
            compile_form(form)


@pytest.mark.parametrize("moment", [0, sparse.csr_matrix((2, 0))])
def test_empty_sparse_moments_reach_local_array_contract(moment):
    equations = LocalEquations(np.eye(2), [1, 2], 0, 0, [], moments=moment, test_moments=moment)
    compiled = compile_local_equations(equations)
    assert compiled.problem.constraints.shape == (2, 0)
    assert compiled.problem.test_constraints.shape == (2, 0)
    assert isinstance(compiled.problem.constraints, np.ndarray)
    assert isinstance(compiled.problem.test_constraints, np.ndarray)


@pytest.mark.parametrize("test_moment", [False, True])
@pytest.mark.parametrize("zero", [0, sparse.csr_matrix((2, 1))])
def test_zero_retained_moments_keep_numerical_valueerror_contract(test_moment, zero):
    moment = np.ones((2, 1))
    equations = LocalEquations(
        [[1, -1], [-1, 1]],
        [0, 0],
        0,
        0,
        [],
        kernel=moment,
        moments=moment if test_moment else zero,
        test_moments=zero if test_moment else moment,
    )
    with pytest.raises(ValueError, match="test_constraints" if test_moment else "constraints"):
        compile_local_equations(equations)


def test_sparse_nonzero_moments_preserve_declared_physical_pairings():
    moment = sparse.csr_matrix([[1.0], [2.0]])
    equations = LocalEquations(
        [[1, -1], [-1, 1]],
        [0, 0],
        0,
        0,
        [],
        kernel=np.ones((2, 1)),
        moments=moment,
        test_moments=moment,
    )
    compiled = compile_local_equations(equations)
    assert_array_equal(compiled.problem.constraints, moment.toarray())
    assert_array_equal(compiled.problem.test_constraints, moment.toarray())


def _dense_zero_compiler(form, shape=None):
    """Represent literal real zero forms densely, as an external compiler may."""
    if not sparse.issparse(form):
        value = np.asarray(form)
        if not np.iscomplexobj(value) and value.ndim == 0 and value == 0 and shape is not None:
            return np.zeros(shape)
    return compile_form(form, shape)


def _zero_form_problem(compiler, *, nested, boundary_value):
    """Declare a coupled leaf or hierarchy with source and nonhomogeneous boundary data."""

    def leaf(_):
        return LocalEquations(
            [[2.0, -1.0], [-1.0, 2.0]],
            [1.0, 3.0],
            [[1.0], [0.0]],
            [[-1.0, 0.0]],
            [0],
            d=[[3.0]],
            g=[2.0],
            moments=0,
            test_moments=0,
        )

    child = MultiscaleProblem(Equation(0, 0), leaf, [0], 1, (0,), compiler=compiler)
    if nested:

        def provider(_):
            return LocalEquations(child, 0, [[2.0]], [[-3.0]], [0], d=[[4.0]], g=[1.0])

    else:
        provider = leaf
    return MultiscaleProblem(
        Equation(0, [0.75]),
        provider,
        [0],
        1,
        (0,),
        fixed={} if boundary_value is None else {0: boundary_value},
        compiler=compiler,
    )


def test_custom_compiler_still_receives_and_owns_literal_global_forms():
    """Only the default compiler chooses sparse zero storage; external assembly is delegated."""
    seen = []

    def compiler(form, shape=None):
        if not sparse.issparse(form):
            value = np.asarray(form)
            if value.ndim == 0 and value == 0 and shape == (1, 1):
                seen.append((form, shape))
                return np.array([[2.0]])
        return compile_form(form, shape)

    original = assemble(_zero_form_problem(compile_form, nested=False, boundary_value=None))
    custom = assemble(_zero_form_problem(compiler, nested=False, boundary_value=None))
    assert seen == [(0, (1, 1))]
    assert_array_equal(custom.matrix.toarray(), original.matrix.toarray() + 2.0)
    assert_array_equal(custom.rhs, original.rhs)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("boundary_value", [None, 0.0, 2.0])
def test_sparse_zero_and_external_dense_compilers_preserve_global_and_local_fields(
    nested, boundary_value
):
    default = assemble(
        _zero_form_problem(compile_form, nested=nested, boundary_value=boundary_value)
    )
    external = assemble(
        _zero_form_problem(_dense_zero_compiler, nested=nested, boundary_value=boundary_value)
    )
    for actual, expected in (
        (default.matrix.data, external.matrix.data),
        (default.matrix.indices, external.matrix.indices),
        (default.matrix.indptr, external.matrix.indptr),
        (default.rhs, external.rhs),
        (default.load_scale, external.load_scale),
    ):
        assert_array_equal(actual, expected)
    actual, expected = default.solve(), external.solve()
    assert_array_equal(actual.trace, expected.trace)
    assert_array_equal(actual.fields, expected.fields)
    if nested:
        assert_array_equal(actual.children[0].trace, expected.children[0].trace)
        assert_array_equal(actual.children[0].fields, expected.children[0].fields)
