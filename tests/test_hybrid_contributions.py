"""Compact hybrid phases preserve original operators and boundary conventions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm.core.contracts import LocalProblem
from pymhm.core.contributions import local_global_contribution
from pymhm.core.system import HybridSystem


@pytest.mark.parametrize("matrix", [None, [[0.25]]])
@pytest.mark.parametrize("load", [None, [0.75]])
def test_declared_direct_trace_terms_add_once(matrix, load):
    """Eliminate 2u+lambda=3 and -u+D lambda=g independently of shared algebra."""
    response = LocalProblem([[2.0]], [[1.0]], [3.0], [4]).condense()
    indices, block, rhs = local_global_contribution(
        response, np.array([], dtype=int), direct_matrix=matrix, direct_load=load
    )
    assert_array_equal(indices, [4])
    assert_array_equal(block, [[0.5 + (0.0 if matrix is None else 0.25)]])
    assert_array_equal(rhs, [1.5 + (0.0 if load is None else 0.75)])
    direct = 0.0 if matrix is None else 0.25
    forcing = 0.0 if load is None else 0.75
    full_solution = np.linalg.solve([[2.0, 1.0], [-1.0, direct]], [3.0, forcing])
    reduced_trace = np.linalg.solve(block, rhs)
    assert_allclose(reduced_trace, full_solution[1:], atol=1e-14)
    assert_allclose(response.reconstruct(reduced_trace, np.empty(0)), full_solution[:1], atol=1e-14)


def test_wide_direct_load_preserves_matrix_addition_digits():
    """A shared real dtype is chosen before adding either direct trace term."""
    response = LocalProblem([[2.0]], [[1.0]], [3.0], [0]).condense()
    digit = np.finfo(float).eps / 4
    _, matrix, load = local_global_contribution(
        response,
        np.array([], dtype=int),
        direct_matrix=[[digit]],
        direct_load=np.array([0.25], dtype=np.longdouble),
    )
    assert matrix.dtype == load.dtype == np.dtype(np.longdouble)
    assert matrix[0, 0] == np.longdouble(0.5) + digit


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"direct_matrix": [[1.0, 2.0]]}, "direct_matrix"),
        ({"direct_matrix": [[1j]]}, "real"),
        ({"direct_matrix": [[np.nan]]}, "direct_matrix"),
        ({"direct_load": []}, "direct_load"),
        ({"direct_load": [1j]}, "real"),
        ({"direct_load": [np.inf]}, "direct_load"),
    ],
)
def test_direct_trace_terms_require_finite_real_matching_shapes(kwargs, match):
    """Reject unsupported terms in the common owner before global scatter."""
    response = LocalProblem([[2.0]], [[1.0]], [3.0], [0]).condense()
    with pytest.raises(ValueError, match=match):
        local_global_contribution(response, np.array([], dtype=int), **kwargs)


def _systems(boundary=None):
    first = LocalProblem(
        [[1.0, -1], [-1, 1]], np.eye(2), [0.0, 0.0], [0, 1], [[1.0], [1.0]], [[0.5], [0.5]]
    )
    second = LocalProblem(
        first.matrix, np.diag([-1.0, 1]), first.load, [1, 2], first.kernel, first.constraints
    )
    full = HybridSystem([first, second], boundary_load=boundary)
    contributions = [
        response.global_contribution(np.arange(full.kernel_offsets[i], full.kernel_offsets[i + 1]))
        for i, response in enumerate(full.responses)
    ]
    compact = HybridSystem.from_contributions(
        contributions, trace_size=3, coarse_sizes=[1, 1], boundary_load=boundary
    )
    return full, compact


@pytest.mark.parametrize("boundary", [None, [0.0, 0.0, 2.0]])
def test_compact_operator_rhs_and_reconstruction(boundary):
    full, compact = _systems(boundary)
    assert_array_equal(compact.matrix.toarray(), full.matrix.toarray())
    assert_array_equal(compact.rhs, full.rhs)
    assert_array_equal(compact.load_scale, full.load_scale)
    expected, actual = full.solve(), compact.solve()
    assert actual.fields == ()
    assert_array_equal(actual.trace, expected.trace)
    assert_array_equal(actual.coarse, expected.coarse)
    rebuilt = tuple(
        response.reconstruct(actual.trace[response.problem.trace_dofs], coarse)
        for response, coarse in zip(full.responses, actual.coarse, strict=True)
    )
    assert_array_equal(rebuilt, expected.fields)


def test_compact_physical_mean_and_incompatible_neumann_data():
    full, compact = _systems()
    gauge = full.mean_constraint([np.array([0.5, 0.5])] * 2, 3.0)
    for system in [full, compact]:
        result = system.solve(fixed={0: -1.0, 2: 1.0}, constraints=[gauge])
        assert_allclose(result.coarse, [[2.0], [1.0]], atol=1e-14)
        with pytest.raises(ValueError, match="incompatible"):
            system.solve(fixed={0: 0.0, 2: 1.0}, constraints=[gauge])
    with pytest.raises(ValueError, match="explicit physical mean"):
        compact.mean_constraint([])


@pytest.mark.parametrize(
    "trace,sizes", [(True, [0]), (-1, [0]), (0.5, [0]), (0, []), (0, [True]), (0, [-1]), (0, [0.5])]
)
def test_compact_partition_validation(trace, sizes):
    with pytest.raises(ValueError, match="nonnegative"):
        HybridSystem.from_contributions([], trace_size=trace, coarse_sizes=sizes)


@pytest.mark.parametrize(
    "indices,block,rhs,match",
    [
        ([0, 0], np.eye(2), [0, 0], "indices"),
        ([0, 2], np.eye(2), [0, 0], "indices"),
        ([-1], [[1]], [0], "indices"),
        ([[0]], [[1]], [0], "indices"),
        ([0.5], [[1]], [0], "indices"),
        ([0, 1], [[1]], [0, 0], "contribution matrix"),
        ([0, 1], np.eye(2), [0], "contribution rhs"),
        ([1], [[1]], [0], "contiguous"),
    ],
)
def test_compact_contribution_validation(indices, block, rhs, match):
    with pytest.raises(ValueError, match=match):
        HybridSystem.from_contributions([(indices, block, rhs)], trace_size=1, coarse_sizes=[1])


def test_compact_requires_one_contribution_per_partition():
    with pytest.raises(ValueError, match="one contribution"):
        HybridSystem.from_contributions([], trace_size=1, coarse_sizes=[1])


@pytest.mark.parametrize("indices", [[0], [0, 2], [0, 1, 2]])
def test_compact_coarse_indices_must_match_their_cell(indices):
    size = len(indices)
    with pytest.raises(ValueError, match="ordered cell partition"):
        HybridSystem.from_contributions(
            [(indices, np.eye(size), np.zeros(size)), ([0, 1], np.eye(2), np.zeros(2))],
            trace_size=1,
            coarse_sizes=[1, 1],
        )


def test_compact_empty_algebraic_system_has_one_empty_coarse_partition():
    compact = HybridSystem.from_contributions(
        [(np.array([], dtype=int), np.empty((0, 0)), np.empty(0))], trace_size=0, coarse_sizes=[0]
    )
    result = compact.solve()
    assert result.trace.size == result.coarse[0].size == 0
    assert result.residual == 0.0


def test_compact_preserves_explicit_extended_operator_coefficients():
    block = np.array([[1.0]], dtype=np.longdouble)
    block[0, 0] += np.finfo(np.longdouble).eps
    compact = HybridSystem.from_contributions(
        [(np.array([0]), block, np.zeros(1))], trace_size=1, coarse_sizes=[0]
    )
    assert compact.matrix.dtype == block.dtype
    assert compact.matrix[0, 0] == block[0, 0]


@pytest.mark.parametrize("wide_boundary", [False, True])
def test_compact_preserves_wide_loads_and_nonhomogeneous_boundary_digits(wide_boundary):
    """Accumulate mixed real precisions without rounding source or boundary loads."""
    tiny = np.finfo(np.longdouble).eps
    wide = np.array([1.0], dtype=np.longdouble) + tiny
    boundary = wide if wide_boundary else np.array([0.25])
    compact = HybridSystem.from_contributions(
        [
            (np.array([0]), [[1.0]], np.array([0.5])),
            (np.array([0]), [[2.0]], wide),
        ],
        trace_size=1,
        coarse_sizes=[0, 0],
        boundary_load=boundary,
    )
    assert compact.rhs.dtype == compact.load_scale.dtype == np.dtype(np.longdouble)
    assert compact.rhs[0] == np.longdouble(0.5) + wide[0] - boundary[0]
    assert compact.load_scale[0] == np.longdouble(0.5) + wide[0] + abs(boundary[0])


def test_wide_boundary_promotes_double_loads_and_convertible_real_inputs():
    """Only the prescribed weak boundary datum can introduce retained wide digits."""
    boundary = np.array([1.0], dtype=np.longdouble) + np.finfo(np.longdouble).eps
    compact = HybridSystem.from_contributions(
        [(np.array([0]), [["1.0"]], ["2.0"])],
        trace_size=1,
        coarse_sizes=[0],
        boundary_load=boundary,
    )
    assert compact.rhs.dtype == compact.load_scale.dtype == np.dtype(np.longdouble)
    assert compact.rhs[0] == 2 - boundary[0]
    assert compact.load_scale[0] == 2 + boundary[0]


def test_compact_cannot_certify_unavailable_local_kernel_roundoff():
    matrix = np.array([[np.nextafter(1.0, np.inf), -1.0], [-1.0, 1.0]])
    problem = LocalProblem(
        matrix, np.empty((2, 0)), np.zeros(2), np.array([], dtype=int), kernel=np.ones((2, 1))
    )
    full = HybridSystem([problem])
    compact = HybridSystem.from_contributions(
        [full.responses[0].global_contribution(np.array([0]))], trace_size=0, coarse_sizes=[1]
    )
    gauge = full.mean_constraint([np.array([0.5, 0.5])], 2.0)
    assert full.solve(constraints=[gauge]).residual == 0.0
    with pytest.raises(ValueError, match="gauge changed physical equations"):
        compact.solve(constraints=[gauge])


def test_permuted_modes_reconstruct_in_the_declared_global_slot_order():
    """A caller maps global coarse slots back to the executed local basis columns."""
    problem = LocalProblem(
        np.diag([2.0, 3.0]), [[1.0], [0.5]], [4.0, 1.0], [0], coarse_basis=np.eye(2)
    )
    full = HybridSystem([problem])
    response = full.responses[0]
    coarse_dofs = np.array([2, 1])
    compact = HybridSystem.from_contributions(
        [response.global_contribution(coarse_dofs)], trace_size=1, coarse_sizes=[2]
    )
    expected, actual = full.solve(), compact.solve()
    amplitudes = actual.coarse[0][coarse_dofs - compact.kernel_offsets[0]]
    assert_allclose(amplitudes, expected.coarse[0], atol=1e-15)
    assert_allclose(actual.trace, expected.trace, atol=1e-15)
    assert_allclose(response.reconstruct(actual.trace, amplitudes), expected.fields[0], atol=1e-15)
