"""User-defined four-block forms preserve independent test/trial pairings and maps."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from pymhm import LocalEquations, columns, compile_form, compile_local_equations, rows
from pymhm.core.equations import LinearForms


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


def test_complex_zero_does_not_silently_change_scalar_field_type():
    with pytest.raises(ValueError, match="real"):
        compile_form(0j, (2, 2))
