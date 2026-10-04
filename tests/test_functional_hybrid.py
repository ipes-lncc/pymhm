"""Functional hybrid operations retain physical rows and ordered reduction."""

import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import hybrid
from pymhm.solvers import factorize


def _petrov_problem():
    return hybrid.LocalProblem(
        [[2.0, 0.25], [-0.5, 3.0]],
        [[1.0], [-0.75]],
        [2.0, -1.0],
        np.array([0]),
        coarse_basis=[[1.0], [0.5]],
        constraints=[[0.75], [0.25]],
        test_coupling=[[0.5], [1.0]],
        test_basis=[[0.25], [1.0]],
        test_constraints=[[0.5], [0.75]],
    )


def test_functional_petrov_elimination_preserves_full_original_rows():
    problem = _petrov_problem()
    matrix, rhs = hybrid.local_condensation_system(problem)
    assert_array_equal(matrix.toarray(), hybrid.local_condensation_matrix(problem).toarray())
    assert_array_equal(matrix.toarray(), problem.condensation_matrix().toarray())
    assert_array_equal(rhs, problem.condensation_system()[1])
    with factorize(matrix) as prepared:
        response = hybrid.local_response_from_solution(problem, prepared.solve(rhs))
    condensed = hybrid.condense_local(problem)
    assert_array_equal(response.source, condensed.source)
    assert_array_equal(response.lifts, condensed.lifts)
    assert_array_equal(response.retained_basis, condensed.retained_basis)
    boundary = np.array([0.625])
    system = hybrid.HybridSystem.from_responses([response], boundary_load=boundary)
    result = hybrid.solve_hybrid_system(system)
    assert_array_equal(result.fields, system.solve().fields)
    complete = np.block(
        [[problem.matrix.toarray(), problem.coupling], [problem.test_coupling.T, np.zeros((1, 1))]]
    )
    expected = np.linalg.solve(complete, np.r_[problem.load, boundary])
    assert_allclose(result.fields[0], expected[:2], atol=2e-15, rtol=2e-15)
    assert_allclose(result.trace, expected[2:], atol=2e-15, rtol=2e-15)
    assert_allclose(
        problem.matrix @ result.fields[0] + problem.coupling @ result.trace,
        problem.load,
        atol=2e-15,
    )
    assert_allclose(problem.test_coupling.T @ result.fields[0], boundary, atol=2e-15)
    assert_array_equal(
        hybrid.local_condensed_load(problem, response.source), response.global_load()
    )
    for actual, original in zip(
        hybrid.local_global_contribution(response, np.array([1])),
        response.global_contribution(np.array([1])),
        strict=True,
    ):
        assert_array_equal(actual, original)


@pytest.mark.parametrize("precision", ["double", "extended"])
def test_functional_reconstruction_retains_executed_basis_and_caller_factor(precision):
    if precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("NumPy longdouble is not wider than double")
    problem = _petrov_problem()
    response = hybrid.condense_local(problem, refinement_precision=precision)
    dtype = np.longdouble if precision == "extended" else float
    trace = np.array([[0.25, -0.5]], dtype=dtype)
    coarse = np.array([[0.75, 1.25]], dtype=dtype)
    with factorize(hybrid.local_condensation_matrix(problem)) as prepared:
        actual = hybrid.reconstruct_local(
            problem,
            trace,
            coarse,
            retained_basis=response.retained_basis,
            factorization=prepared,
            refinement_precision=precision,
        )
        delegated = problem.reconstruct(
            trace,
            coarse,
            retained_basis=response.retained_basis,
            factorization=prepared,
            refinement_precision=precision,
        )
        assert_array_equal(actual, delegated)
        assert prepared.matches(hybrid.local_condensation_matrix(problem))
        assert_array_equal(prepared.solve(np.zeros(3)), np.zeros(3))
    replayed = np.column_stack(
        [hybrid.reconstruct_response(response, trace[:, i], coarse[:, i]) for i in range(2)]
    )
    delegated_replay = np.column_stack(
        [response.reconstruct(trace[:, i], coarse[:, i]) for i in range(2)]
    )
    assert_array_equal(replayed, delegated_replay)
    assert_allclose(actual, replayed, atol=2e-15, rtol=2e-15)
    assert actual.dtype == np.dtype(dtype)


def test_reduction_consumes_reused_buffers_before_requesting_next_cell():
    indices = np.array([0])
    block, rhs = np.array([[1.0]]), np.array([1.0])

    def contributions():
        yield indices, block, rhs
        block[0, 0], rhs[0] = 2.0, 3.0
        yield indices, block, rhs
        block[0, 0], rhs[0] = 1000.0, 1000.0

    matrix, load, scale = hybrid.assemble_hybrid_contributions(
        contributions(), trace_size=1, kernel_offsets=np.array([1, 1, 1]), boundary_load=[0.5]
    )
    assert_array_equal(matrix.toarray(), [[3.0]])
    assert_array_equal(load, [3.5])
    assert_array_equal(scale, [4.5])


def test_reduction_preserves_canonical_cancellation_and_wide_boundary_digits():
    contributions = (
        (np.array([0]), np.array([[1.0]]), np.array([value])) for value in (1e16, 1.0, -1e16)
    )
    boundary = np.array([0.5], dtype=np.longdouble) + np.finfo(np.longdouble).eps
    matrix, load, scale = hybrid.assemble_hybrid_contributions(
        contributions,
        trace_size=1,
        kernel_offsets=np.array([1, 1, 1, 1]),
        boundary_load=boundary,
    )
    assert_array_equal(matrix.toarray(), [[3.0]])
    assert load.dtype == scale.dtype == boundary.dtype
    assert load[0] == -boundary[0]
    assert scale[0] == np.longdouble(2e16) + boundary[0]


@pytest.mark.parametrize("trace_size", [True, -1, 0.5])
def test_reduction_requires_integer_trace_size(trace_size):
    with pytest.raises(ValueError, match="trace_size"):
        hybrid.assemble_hybrid_contributions(
            [], trace_size=trace_size, kernel_offsets=np.array([0, 0])
        )


@pytest.mark.parametrize(
    "offsets",
    [[], [[1, 1]], [1], [1.0, 1.0], [0, 1], [1, 3, 2], np.array([1, 3, 2], dtype=np.uint64)],
)
def test_reduction_requires_ordered_retained_partition(offsets):
    with pytest.raises(ValueError, match="kernel_offsets"):
        hybrid.assemble_hybrid_contributions([], trace_size=1, kernel_offsets=np.asarray(offsets))


@pytest.mark.parametrize("count", [0, 2])
def test_reduction_requires_one_contribution_per_partition(count):
    contributions = ((np.array([0]), np.eye(1), np.zeros(1)) for _ in range(count))
    with pytest.raises(ValueError, match="one contribution"):
        hybrid.assemble_hybrid_contributions(
            contributions, trace_size=1, kernel_offsets=np.array([1, 1])
        )


def test_historical_classes_and_public_functions_remain_pickleable():
    problem = _petrov_problem()
    response = hybrid.condense_local(problem)
    system = hybrid.HybridSystem.from_responses([response], metadata=[{"cell": 3}])
    restored = pickle.loads(pickle.dumps(system))
    assert type(restored) is hybrid.HybridSystem
    assert type(restored.responses[0]) is hybrid.LocalResponse
    assert type(restored.responses[0].problem) is hybrid.LocalProblem
    assert restored.local_metadata == ({"cell": 3},)
    assert_array_equal(restored.matrix.toarray(), system.matrix.toarray())
    assert_array_equal(restored.solve().fields, system.solve().fields)
    assert pickle.loads(pickle.dumps(hybrid.condense_local)) is hybrid.condense_local


def test_functional_physical_mean_keeps_original_neumann_equations():
    problem = hybrid.LocalProblem(
        [[1.0, -1.0], [-1.0, 1.0]],
        np.eye(2),
        [0.0, 0.0],
        np.arange(2),
        kernel=np.ones((2, 1)),
        constraints=np.ones((2, 1)) / 2,
    )
    system = hybrid.HybridSystem([problem])
    weights = [np.ones(2) / 2]
    actual = hybrid.hybrid_mean_constraint(system, weights, 3.0)
    expected = system.mean_constraint(weights, 3.0)
    assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]
    result = hybrid.solve_hybrid_system(system, fixed={0: -1.0, 1: 1.0}, constraints=[actual])
    assert_allclose(result.fields[0], [3.5, 2.5], atol=1e-14)
    with pytest.raises(ValueError, match="gauge changed physical equations"):
        hybrid.solve_hybrid_system(system, fixed={0: 0.0, 1: 1.0}, constraints=[actual])


@pytest.mark.parametrize(
    "method,function,args",
    [("solve", "solve_hybrid_system", ()), ("mean_constraint", "hybrid_mean_constraint", ([],))],
)
def test_global_methods_delegate_to_single_function_owner(monkeypatch, method, function, args):
    system = hybrid.HybridSystem([_petrov_problem()])
    sentinel = object()

    def operation(received, *unused_args, **unused_kwargs):
        assert received is system
        return sentinel

    monkeypatch.setattr(hybrid, function, operation)
    assert getattr(system, method)(*args) is sentinel


@pytest.mark.parametrize(
    "method,function,args,kwargs",
    [
        ("condense", "condense_local", (), {}),
        ("condensation_matrix", "local_condensation_matrix", (), {}),
        ("condensation_system", "local_condensation_system", (), {}),
        ("response_from_solution", "local_response_from_solution", (None,), {}),
        ("reconstruct", "reconstruct_local", (None, None), {}),
        ("condensed_load", "local_condensed_load", (None,), {}),
    ],
)
def test_local_problem_methods_delegate_to_single_function_owner(
    monkeypatch, method, function, args, kwargs
):
    problem, sentinel = _petrov_problem(), object()

    def operation(received, *unused_args, **unused_kwargs):
        assert received is problem
        return sentinel

    monkeypatch.setattr(hybrid, function, operation)
    assert getattr(problem, method)(*args, **kwargs) is sentinel
