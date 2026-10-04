"""Generic Schur equations and three-level reconstruction agree with monolithic assembly."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import (
    Equation,
    ExecutionConfig,
    LocalEquations,
    MultiscaleProblem,
    NestedEquations,
    SolverConfig,
    assemble,
    compile_local_equations,
    leaf_moment,
    solve,
    with_global_equation,
)
from pymhm.core.contracts import LocalProblem


def nonsymmetric_cell(cell):
    """Supply unequal B/C and one common global coordinate for two subdomains."""
    a = np.array([[2.0, 1.0], [0.0, 3.0]]) + cell * np.eye(2)
    return LocalEquations(
        a,
        [1.0, 2.0],
        [[1.0, 2.0], [3.0, 1.0]],
        [[2.0, -1.0], [1.0, 3.0]],
        [cell, cell + 1],
        d=np.eye(2) * 5,
        g=[1.0, 2.0],
        metadata={"cell": cell},
    )


def leaf_cell(cell):
    """Return an invertible two-coordinate leaf with a one-coordinate interface."""
    return LocalEquations(
        [[2.0, -1.0], [-1.0, 2.0]],
        [1.0, 3.0],
        [[1.0], [0.0]],
        [[-1.0, 0.0]],
        [0],
        d=[[3.0]],
        g=[2.0],
    )


def leaf_problem():
    return MultiscaleProblem(Equation(0, 0), leaf_cell, [0], 1, (0,))


def middle_cell(cell):
    return LocalEquations(leaf_problem(), 0, [[2.0]], [[-3.0]], [0], d=[[4.0]], g=[1.0])


def middle_problem():
    return MultiscaleProblem(Equation(0, 0), middle_cell, [0], 1, (0,))


def top_cell(cell):
    return LocalEquations(middle_problem(), 0, [[1.0]], [[-2.0]], [0], d=[[6.0]], g=[3.0])


def test_independent_blocks_and_global_terms_match_uncondensed_matrix():
    global_a = np.array([[1.0, 0.0, 0.2], [0.0, 2.0, 0.0], [0.3, 0.0, 1.0]])
    global_f = np.array([2.0, 3.0, 4.0])
    problem = MultiscaleProblem(Equation(global_a, global_f), nonsymmetric_cell, [0, 1], 3, (0, 0))
    system = assemble(problem)
    result = system.solve()
    original = np.zeros((7, 7))
    rhs = np.zeros(7)
    original[4:, 4:] = global_a
    rhs[4:] = global_f
    for cell in range(2):
        eq = nonsymmetric_cell(cell)
        ii = np.arange(cell * 2, cell * 2 + 2)
        jj = 4 + np.array(eq.dofs)
        original[np.ix_(ii, ii)] = eq.a
        original[np.ix_(ii, jj)] = eq.b
        original[np.ix_(jj, ii)] = eq.c
        original[np.ix_(jj, jj)] += eq.d
        rhs[ii] = eq.L
        rhs[jj] += eq.g
    expected = np.linalg.solve(original, rhs)
    assert_allclose(
        np.concatenate((*result.fields, result.trace)), expected, rtol=2e-14, atol=2e-14
    )
    assert result.children == (None, None)
    assert system.local_metadata == ({"cell": 0}, {"cell": 1})
    assert result.raw_residual < 1e-13
    for cell, field in enumerate(result.fields):
        eq = nonsymmetric_cell(cell)
        assert_allclose(eq.a @ field + np.asarray(eq.b) @ result.trace[eq.dofs], eq.L, atol=1e-14)


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_three_levels_reconstruct_original_leaf_equations(backend):
    problem = MultiscaleProblem(Equation(0, 0), top_cell, [0], 1, (0,))
    result = solve(problem, execution=ExecutionConfig(backend, workers=2, batch_size=1))
    monolithic = np.array(
        [
            [2.0, -1.0, 1.0, 0.0, 0.0],
            [-1.0, 2.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 3.0, 2.0, 0.0],
            [0.0, 0.0, -3.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, -2.0, 6.0],
        ]
    )
    expected = np.linalg.solve(monolithic, [1.0, 3.0, 2.0, 1.0, 3.0])
    child = result.children[0]
    leaf = child.children[0]
    assert_allclose(
        np.r_[leaf.fields[0], leaf.trace, child.trace, result.trace],
        expected,
        atol=2e-14,
        rtol=2e-14,
    )
    assert_allclose(result.fields[0], child.trace, atol=1e-14)
    assert_allclose(child.fields[0], leaf.trace, atol=1e-14)
    assert max(result.residual, child.residual, leaf.residual) < 1e-13


def test_global_boundary_and_gauge_conventions_use_same_solver_owner():
    problem = MultiscaleProblem(Equation(0, 0), leaf_cell, [0], 1, (0,), fixed={0: 2.0})
    result = solve(problem)
    assert_array_equal(result.trace, [2])
    assert_allclose(result.fields[0], np.linalg.solve([[2, -1], [-1, 2]], [-1, 3]))
    system = assemble(replace(problem, fixed={}))
    row, target = system.mean_constraint(([1.0, 1.0],), sum(system.solve().fields[0]))
    gauged = system.solve(constraints=[(row, target)])
    assert_allclose(gauged.gauge_multipliers, 0, atol=1e-14)


def test_compiled_provider_is_an_external_backend_contract():
    compiled = compile_local_equations(leaf_cell(0))
    problem = MultiscaleProblem(Equation(0, 0), lambda _: compiled, [0], 1, (0,))
    assert_allclose(solve(problem).fields, solve(leaf_problem()).fields)
    external = SolverConfig(local_solver=lambda a, rhs: np.linalg.solve(a.toarray(), rhs))
    assert_allclose(solve(problem, solvers=external).fields, solve(leaf_problem()).fields)


def test_global_form_can_own_coordinates_without_local_trace_coupling():
    problem = MultiscaleProblem(
        Equation([[2.0]], [6.0]),
        lambda _: LocalEquations([[3.0]], [12.0], 0, 0, []),
        [0],
        1,
        (0,),
    )
    result = solve(problem)
    assert_array_equal(result.trace, [3.0])
    assert_array_equal(result.fields, [[4.0]])


def test_recursive_reconstruction_rejects_unrelated_coordinates():
    system = assemble(leaf_problem())
    with pytest.raises(ValueError, match="executed global"):
        system.reconstruct([100.0])
    assert_allclose(system.reconstruct(system.solve().trace).fields, system.solve().fields)


def test_global_load_update_reuses_basis_and_recovers_fields():
    system = assemble(leaf_problem())
    updated = system.with_rhs(system.rhs + 1.0)
    assert updated.matrix is system.matrix
    assert updated.responses is system.responses
    result = updated.solve()
    assert_allclose(updated.matrix @ result.trace, updated.rhs, atol=1e-14)
    eq = leaf_cell(0)
    assert_allclose(
        np.asarray(eq.a) @ result.fields[0] + np.asarray(eq.b) @ result.trace, eq.L, atol=1e-14
    )
    retained = assemble(
        MultiscaleProblem(Equation(np.diag([1.0, 1.0, 0.0]), 0), retained_child_cell, [0], 2, (1,))
    )
    with pytest.raises(ValueError, match="retained"):
        retained.with_rhs([0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "change,match",
    [
        ({"global_equation": None}, "Equation"),
        ({"local_provider": None}, "callable"),
        ({"compiler": None}, "callable"),
    ],
)
def test_invalid_problem_contracts_are_rejected(change, match):
    with pytest.raises(TypeError, match=match):
        replace(leaf_problem(), **change)


def test_provider_and_recursive_boundary_contracts_are_explicit():
    with pytest.raises(TypeError, match="MultiscaleProblem"):
        assemble(None)
    bad = replace(leaf_problem(), local_provider=lambda _: LocalProblem([[1]], [[1]], [0], [0]))
    with pytest.raises(TypeError, match="LocalEquations"):
        assemble(bad)
    fixed_child = replace(leaf_problem(), fixed={0: 0.0})
    parent = replace(
        leaf_problem(), local_provider=lambda _: LocalEquations(fixed_child, 0, [[1]], [[1]], [0])
    )
    with pytest.raises(ValueError, match="parent"):
        assemble(parent)


def retained_child_cell(cell):
    return LocalEquations(
        [[1.0, -1.0], [-1.0, 1.0]],
        [0.0, 0.0],
        np.eye(2),
        -np.eye(2),
        [0, 1],
        kernel=np.ones((2, 1)),
        moments=np.ones((2, 1)),
    )


@pytest.mark.parametrize("bad", ["coupling", "load", "none"])
def test_recursive_retained_rows_keep_leaf_compatibility(bad):
    inner = MultiscaleProblem(
        Equation(np.diag([1.0, 1.0, 0.0]), 0), retained_child_cell, [0], 2, (1,)
    )
    b = np.array([[1.0], [0.0], [1.0 if bad == "coupling" else 0.0]])
    load = np.array([0.0, 0.0, 1.0 if bad == "load" else 0.0])
    parent = MultiscaleProblem(
        Equation([[3.0]], [1.0]), lambda _: LocalEquations(inner, load, b, -b.T, [0]), [0], 1, (0,)
    )
    if bad != "none":
        with pytest.raises(ValueError, match="retained"):
            assemble(parent)
    else:
        result = solve(parent)
        child = result.children[0]
        assert child is not None and child.children == (None,)
        eq = retained_child_cell(0)
        assert_allclose(np.asarray(eq.a) @ child.fields[0] + child.trace, eq.L, atol=1e-14)


def nested_parent_cell(cell):
    child = MultiscaleProblem(Equation(0, 0), retained_child_cell, [0], 2, (1,))
    return NestedEquations(child, [0, 1], np.eye(2), [0, 1], kernel=[[0.0], [0.0], [1.0]])


@pytest.mark.parametrize("backend", ["serial", "process"])
def test_boundary_constraint_recursion_preserves_local_nullspace_and_leaf_fields(backend):
    problem = MultiscaleProblem(Equation(0, [0.0, -2.0, 0.0]), nested_parent_cell, [0], 2, (1,))
    result = solve(problem, execution=ExecutionConfig(backend, workers=2))
    child = result.children[0]
    assert_allclose(child.fields[0], [0.0, 2.0], atol=1e-14)
    assert_allclose(child.trace, result.trace, atol=1e-14)
    assert child.raw_residual < 1e-13
    assert result.raw_residual < 1e-13
    assert len(result.fields[0]) == 5


@pytest.mark.parametrize("bad", ["type", "boundary", "retained"])
def test_nested_constraint_rejects_ambiguous_child_contracts(bad):
    eq = nested_parent_cell(0)
    if bad == "type":
        eq = replace(eq, problem=None)
    elif bad == "boundary":
        eq = replace(eq, problem=replace(eq.problem, fixed={0: 0.0}))
    else:
        eq = replace(eq, boundary_dofs=[0, 2], kernel=None)
    problem = MultiscaleProblem(Equation(0, 0), lambda _: eq, [0], 2, (1,))
    with pytest.raises((ValueError, TypeError)):
        assemble(problem)


def test_recursive_physical_moment_includes_child_source_offsets():
    """A hierarchy's physical integral uses executed leaf coefficients and source terms."""
    problem = MultiscaleProblem(Equation(0, 0), top_cell, [0], 1, (0,))
    system = assemble(problem)
    result = system.solve()
    row, offset = leaf_moment(system, lambda _: [2.0, 3.0])
    expected = np.array([2.0, 3.0]) @ result.children[0].children[0].fields[0]
    assert abs(offset) > 1.0
    assert_allclose(row @ result.trace + offset, expected, atol=2e-14)
    gauged = system.solve(constraints=[(row, expected - offset)])
    assert np.linalg.norm(row * gauged.gauge_multipliers[0]) < 2e-14
    assert_allclose(gauged.trace, result.trace, atol=2e-14)
    with pytest.raises(ValueError, match="weights"):
        leaf_moment(system, lambda _: [1.0])


def test_nested_physical_moment_ignores_reaction_coordinates():
    problem = MultiscaleProblem(Equation(0, [0.0, -2.0, 0.0]), nested_parent_cell, [0], 2, (1,))
    system = assemble(problem)
    result = system.solve()
    row, offset = leaf_moment(system, lambda _: [1.0, 1.0])
    coordinates = np.r_[result.trace, *result.coarse]
    assert_allclose(row @ coordinates + offset, np.sum(result.children[0].fields[0]), atol=1e-14)


def test_additional_global_forms_preserve_local_bases_and_original_system():
    """A response-dependent global bilinear form reuses the exact executed coordinate maps."""
    system = assemble(leaf_problem())
    old_matrix, old_rhs = system.matrix.copy(), system.rhs.copy()
    additional = Equation([[2.0]], [3.0])
    updated = with_global_equation(system, additional)
    assert updated is not system
    assert updated.responses is system.responses and updated.cells is system.cells
    assert_array_equal(system.matrix.toarray(), old_matrix.toarray())
    assert_array_equal(system.rhs, old_rhs)
    result = updated.solve()
    eq = leaf_cell(0)
    original = np.block(
        [
            [np.asarray(eq.a), np.asarray(eq.b)],
            [np.asarray(eq.c), np.asarray(eq.d) + np.asarray(additional.a)],
        ]
    )
    expected = np.linalg.solve(original, np.r_[eq.L, np.asarray(eq.g) + additional.L])
    assert_allclose(np.r_[result.fields[0], result.trace], expected, atol=2e-14)
    with pytest.raises(TypeError, match="Equation"):
        with_global_equation(system, None)
    with pytest.raises(ValueError, match="declared shape"):
        with_global_equation(system, Equation(np.eye(2), 0))
