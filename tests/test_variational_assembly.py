"""Physical and execution invariants of the provider-based hybrid interface."""

import os
import threading
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm.core.assembly import HybridProblem, SolverConfig, assemble_hybrid, solve_hybrid
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.contributions import local_global_contribution
from pymhm.core.system import HybridSystem
from pymhm.core.variational import GlobalForm
from pymhm.execution.cpu import ExecutionConfig
from pymhm.linalg.linear import LinearSolveError, check_linear_solution


def local_cell(cell, *, metadata=False):
    """Return two oriented Neumann cells sharing their middle trace slot."""
    problem = LocalProblem(
        [[1.0, -1], [-1, 1]],
        np.eye(2) if cell == 0 else np.diag([-1.0, 1]),
        [0.0, 0.0],
        [cell, cell + 1],
        kernel=[[1.0], [1.0]],
        constraints=[[0.5], [0.5]],
    )
    return LocalAssembly(problem, {"cell": cell}) if metadata else problem


def external_solver(matrix, rhs):
    """Implement an independent dense linear solver through the portable hook."""
    return np.linalg.solve(matrix.toarray(), rhs)


def owned_cell(cell):
    """Declare coefficient-only metadata identifying the provider worker."""
    return LocalAssembly(local_cell(cell), {"provider_owner": (os.getpid(), threading.get_ident())})


def owned_contribution(response, record, slots):
    """Build a pure cell block, recording ownership in worker-owned metadata."""
    record["contribution_owner"] = (os.getpid(), threading.get_ident())
    return local_global_contribution(response, slots)


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
@pytest.mark.parametrize("placement", ["coordinator", "worker"])
def test_independent_contribution_runs_on_its_declared_owner(backend, placement):
    """Ownership changes execution only; shared-face operators and fields agree."""
    owner = (os.getpid(), threading.get_ident())
    form = GlobalForm(3, (1, 1), boundary_load=[0.0, 0.0, 2.0])
    system = assemble_hybrid(
        HybridProblem(
            form, owned_cell, range(2), owned_contribution, contribution_execution=placement
        ),
        execution=ExecutionConfig(backend=backend, workers=2, batch_size=2),
    )
    expected = HybridSystem([local_cell(0), local_cell(1)], boundary_load=form.boundary_load)
    assert_array_equal(system.matrix.toarray(), expected.matrix.toarray())
    assert_array_equal(system.rhs, expected.rhs)
    actual, reference = system.solve(), expected.solve()
    for field, baseline in zip(actual.fields, reference.fields, strict=True):
        assert_array_equal(field, baseline)
    for record in system.local_metadata:
        assert record["contribution_owner"] == (
            record["provider_owner"] if placement == "worker" else owner
        )
        if backend != "serial":
            assert record["provider_owner"] != owner


def test_default_schur_block_is_computed_on_local_thread(monkeypatch):
    """An ordinary provider delegates all dense local reduction to its worker."""
    import pymhm.core.assembly as assembly

    coordinator = threading.get_ident()
    owners = []

    def record_owner(response, slots):
        owners.append(threading.get_ident())
        return local_global_contribution(response, slots)

    monkeypatch.setattr(assembly, "local_global_contribution", record_owner)
    assemble_hybrid(
        HybridProblem(GlobalForm(3, (1, 1)), local_cell, range(2)),
        execution=ExecutionConfig(backend="thread", workers=2),
    )
    assert len(owners) == 2
    assert all(owner != coordinator for owner in owners)


def test_contribution_placement_validation_is_immediate():
    """Reject unknown placement before consuming local items."""
    with pytest.raises(ValueError, match="contribution_execution"):
        HybridProblem(GlobalForm(3, (1, 1)), local_cell, range(2), contribution_execution="gpu")


# Serial execution does not dispatch batches. With two cells, sizes two and
# three both dispatch one complete batch; sizes one/two test actual shared-face
# reduction across and within parallel batches.
@pytest.mark.parametrize(
    "backend,batch_size",
    [("serial", 1), ("thread", 1), ("thread", 2), ("process", 1), ("process", 2)],
)
@pytest.mark.parametrize("metadata", [False, True])
def test_shared_face_batches_preserve_operator_fields_and_boundary(backend, batch_size, metadata):
    """Every batch arrangement has the same orientation and cellwise reduction."""
    form = GlobalForm(trace_size=3, coarse_sizes=(1, 1), boundary_load=[0.0, 0.0, 2.0])
    definition = HybridProblem(form, partial(local_cell, metadata=metadata), range(2))
    system = assemble_hybrid(
        definition, execution=ExecutionConfig(backend=backend, workers=2, batch_size=batch_size)
    )
    expected = HybridSystem([local_cell(0), local_cell(1)], boundary_load=form.boundary_load)
    for name in ("matrix", "rhs", "load_scale", "kernel_offsets"):
        a, b = getattr(system, name), getattr(expected, name)
        assert_array_equal(
            a.toarray() if name == "matrix" else a, b.toarray() if name == "matrix" else b
        )
    actual, baseline = system.solve(), expected.solve()
    assert_array_equal(actual.trace, baseline.trace)
    for a, b in zip(actual.fields, baseline.fields, strict=True):
        assert_array_equal(a, b)
    for cell, (response, field) in enumerate(zip(system.responses, actual.fields, strict=True)):
        p = response.problem
        assert_allclose(
            p.matrix @ field + p.coupling @ actual.trace[p.trace_dofs], p.load, atol=1e-14
        )
        assert system.local_metadata[cell] == ({"cell": cell} if metadata else None)


def test_serial_provider_contributes_before_consuming_next_item(monkeypatch):
    """Serial assembly streams actual cell contributions, not just factory calls."""
    import pymhm.core.assembly as assembly

    events = []
    original = assembly.local_global_contribution

    def contribution(response, slots):
        events.append(("contribute", int(response.problem.trace_dofs[0])))
        return original(response, slots)

    def items():
        for cell in range(2):
            events.append(("request", cell))
            yield cell

    def provider(cell):
        events.append(("provide", cell))
        return local_cell(cell)

    monkeypatch.setattr(assembly, "local_global_contribution", contribution)
    assemble_hybrid(HybridProblem(GlobalForm(3, (1, 1)), provider, items()))
    assert events == [
        ("request", 0),
        ("provide", 0),
        ("contribute", 0),
        ("request", 1),
        ("provide", 1),
        ("contribute", 1),
    ]


def test_declared_neumann_gauge_and_custom_local_solver():
    """The custom solver preserves physical means and incompatible data rejection."""
    original = HybridSystem([local_cell(0), local_cell(1)])
    row, target = original.mean_constraint([np.array([0.5, 0.5])] * 2, 3.0)
    form = GlobalForm(3, (1, 1), fixed_trace={0: -1.0, 2: 1.0}, constraints=((row, target),))
    problem = HybridProblem(form, local_cell, range(2))
    for backend in ("serial", "process"):
        result = solve_hybrid(
            problem,
            solvers=SolverConfig(local_solver=external_solver),
            execution=ExecutionConfig(backend=backend, workers=2, batch_size=2),
        )
        assert_allclose(result.coarse, [[2.0], [1.0]], atol=1e-14)
        assert result.raw_residual < 1e-14
    bad = GlobalForm(3, (1, 1), fixed_trace={0: 0.0, 2: 1.0}, constraints=((row, target),))
    with pytest.raises(ValueError, match="incompatible"):
        solve_hybrid(HybridProblem(bad, local_cell, range(2)))


def test_custom_solver_retained_petrov_basis_and_every_rhs():
    """Retained nonkernel modes and distinct tests use the same augmented rows."""
    local = LocalProblem(
        [[2.0, 1.0], [0.0, 3.0]],
        [[1.0], [2.0]],
        [1e10, -1e10],
        [0],
        coarse_basis=[[1.0], [0.0]],
        constraints=[[1.0], [0.0]],
        test_basis=[[0.0], [1.0]],
        test_constraints=[[0.0], [1.0]],
        test_coupling=[[2.0], [1.0]],
    )
    problem = HybridProblem(GlobalForm(1, (1,), boundary_load=[0.3]), lambda _: local, [0])
    expected = HybridSystem([local], boundary_load=[0.3]).solve()
    actual = solve_hybrid(problem, solvers=SolverConfig(local_solver=external_solver))
    assert_allclose(actual.trace, expected.trace, rtol=2e-14)
    assert_allclose(actual.fields, expected.fields, rtol=2e-14)

    def corrupt_trace(matrix, rhs):
        result = external_solver(matrix, rhs)
        result[:, 1] += 0.01
        return result

    with pytest.raises(LinearSolveError, match="residual"):
        assemble_hybrid(problem, solvers=SolverConfig(local_solver=corrupt_trace))


def test_external_solver_cannot_change_the_verified_operator():
    """Acceptance checks copies of the original operator and source columns."""

    def change_input(matrix, rhs):
        matrix.data *= 2
        rhs[:] = 1
        return external_solver(matrix, rhs)

    problem = HybridProblem(GlobalForm(3, (1, 1)), local_cell, range(2))
    with pytest.raises(LinearSolveError, match="residual"):
        assemble_hybrid(problem, solvers=SolverConfig(local_solver=change_input))


def test_compatible_residual_does_not_admit_a_nonunique_local_response():
    """A zero load cannot make an undeclared local nullspace acceptable."""
    singular = LocalProblem([[0.0]], np.empty((1, 0)), [0.0], np.array([], dtype=int))
    problem = HybridProblem(GlobalForm(0, (0,)), lambda _: singular, [0])
    calls = []

    def compatible_result(matrix, rhs):
        calls.append(matrix.shape)
        return np.zeros_like(rhs)

    with pytest.raises(LinearSolveError, match="zero row|singular"):
        assemble_hybrid(problem, solvers=SolverConfig(local_solver=compatible_result))
    assert not calls


@pytest.mark.parametrize("result", [np.zeros((1, 1)), np.full((3, 3), np.nan), np.zeros((3, 3))])
def test_external_response_invalid_or_inaccurate_is_rejected(result):
    with pytest.raises(LinearSolveError):
        assemble_hybrid(
            HybridProblem(GlobalForm(3, (1, 1)), local_cell, range(2)),
            solvers=SolverConfig(local_solver=lambda _a, _b: result),
        )


@pytest.mark.parametrize(
    "items, provider, form, exception, match",
    [
        ([], local_cell, GlobalForm(3, (1, 1)), ValueError, "one contribution"),
        ([0, 1, 2], local_cell, GlobalForm(3, (1, 1)), ValueError, "one local item"),
        ([0, 1], local_cell, GlobalForm(3, (0, 1)), ValueError, "retained basis"),
        ([0, 1], local_cell, GlobalForm(1, (1, 1)), ValueError, "trace map"),
        ([0], lambda _: None, GlobalForm(3, (1,)), TypeError, "local_provider"),
    ],
)
def test_provider_and_declared_global_layout_must_match(items, provider, form, exception, match):
    with pytest.raises(exception, match=match):
        assemble_hybrid(HybridProblem(form, provider, items))


@pytest.mark.parametrize(
    "kwargs, exception, match",
    [
        ({"local_solver": 0}, TypeError, "local_solver"),
        ({"global_solver": None}, TypeError, "global_solver"),
        ({"local_refinement_precision": "half"}, ValueError, "local_refinement_precision"),
        ({"global_refinement_precision": "half"}, ValueError, "global_refinement_precision"),
        (
            {"local_solver": external_solver, "local_refinement_precision": "extended"},
            ValueError,
            "owns",
        ),
    ],
)
def test_solver_configuration_is_explicit(kwargs, exception, match):
    with pytest.raises(exception, match=match):
        SolverConfig(**kwargs)


def test_problem_requires_a_form_and_a_callable():
    with pytest.raises(TypeError, match="global_form"):
        HybridProblem(None, local_cell, [])
    with pytest.raises(TypeError, match="local_provider"):
        HybridProblem(GlobalForm(1, (0,)), None, [])


def test_external_residual_checker_preserves_wide_digits_and_zero_columns():
    rhs = np.array([[1.0, 0.0]], dtype=np.longdouble)
    rhs[0, 0] += np.finfo(np.longdouble).eps
    accepted = check_linear_solution([[1.0]], rhs, rhs)
    assert accepted.dtype == rhs.dtype
    assert_array_equal(accepted, rhs)
    wrong = rhs.copy()
    wrong[0, 1] = 1e-20
    with pytest.raises(LinearSolveError, match="residual"):
        check_linear_solution([[1.0]], rhs, wrong)
