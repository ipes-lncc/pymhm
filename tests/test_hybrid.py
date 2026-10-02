"""Algebraic verification independent of all finite-element assembly kernels."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_brinkman, solve_darcy
from pymhm.hybrid import HybridSystem, LocalProblem
from pymhm.solvers import LinearSolveError


def interval_problem(dofs, force=0.0):
    """Build an exact affine finite-element Neumann interval."""
    return LocalProblem(
        [[1.0, -1], [-1, 1]],
        np.eye(2),
        np.full(2, force / 2),
        np.array(dofs),
        np.ones((2, 1)),
        np.full((2, 1), 0.5),
    )


def test_condensation_matches_uncondensed_linear_system():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(5, 5))
    a = x.T @ x + np.eye(5)
    b = rng.normal(size=(5, 3))
    f = rng.normal(size=5)
    g = rng.normal(size=3)
    p = LocalProblem(a, b, f, np.arange(3))
    system = HybridSystem([p], boundary_load=g)
    result = system.solve()
    full = np.block([[a, b], [b.T, np.zeros((3, 3))]])
    expected = np.linalg.solve(full, np.r_[f, g])
    assert_allclose(result.fields[0], expected[:5], atol=1e-13)
    assert_allclose(result.trace, expected[5:], atol=1e-13)
    assert_allclose(p.condense().source, np.linalg.solve(a, f))


def test_mean_constraint_pure_neumann_and_incompatible_load():
    p = interval_problem([0, 1])
    system = HybridSystem([p])
    gauge = system.mean_constraint([np.array([0.5, 0.5])], 3.0)
    sol = system.solve(fixed={0: -1, 1: 1}, constraints=[gauge])
    assert_allclose(sol.fields[0], [3.5, 2.5])
    assert_allclose(sol.gauge_multipliers, 0)
    with pytest.raises(ValueError, match="incompatible"):
        system.solve(fixed={0: 0, 1: 1}, constraints=[gauge])


@pytest.mark.parametrize("force", [1.0, 1e-12, 1e-100])
def test_neumann_compatibility_is_independent_of_forcing_units(force):
    system = HybridSystem([interval_problem([0, 1], force=force)])
    gauge = system.mean_constraint([np.array([0.5, 0.5])], 3.0)
    with pytest.raises(ValueError, match="incompatible"):
        system.solve(fixed={0: 0.0, 1: 0.0}, constraints=[gauge])
    compatible = system.solve(fixed={0: 0.0, 1: force}, constraints=[gauge])
    assert compatible.residual < 1e-8


def test_multi_cell_signed_orientation():
    p = interval_problem([0, 1])
    q = LocalProblem(
        p.matrix, np.diag([-1.0, 1]), p.load, np.array([1, 2]), p.kernel, p.constraints
    )
    system = HybridSystem([p, q], boundary_load=[0, 0, 2])
    sol = system.solve()
    assert_allclose(sol.fields, [[0, 1], [1, 2]], atol=1e-14)
    assert_allclose(sol.trace, [1, -1, -1], atol=1e-14)
    assert sol.residual < 1e-14


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_parallel_condensation_agrees(backend):
    p = interval_problem([0, 1])
    serial = HybridSystem([p], boundary_load=[0, 1]).solve()
    parallel = HybridSystem([p], boundary_load=[0, 1], backend=backend, workers=2).solve()
    assert_allclose(serial.trace, parallel.trace)
    assert_allclose(serial.fields, parallel.fields)


def test_no_trace_and_no_free_equations():
    p = LocalProblem([[2.0]], np.empty((1, 0)), [4.0], np.array([], dtype=int))
    s = HybridSystem([p]).solve()
    assert_allclose(s.fields[0], [2.0])
    p = LocalProblem([[2.0]], [[1.0]], [4.0], np.array([0]))
    assert_allclose(HybridSystem([p]).solve(fixed={0: 2.0}).fields[0], [1.0])


@pytest.mark.parametrize(
    "change,match",
    [
        ({"matrix": [[1j]]}, "real"),
        ({"matrix": [[1, 2]]}, "square"),
        ({"matrix": sparse.csc_matrix((0, 0))}, "square"),
        ({"matrix": [[np.nan]]}, "finite"),
        ({"trace_dofs": np.array([0.5])}, "indices"),
        ({"trace_dofs": np.array([-1])}, "indices"),
        ({"trace_dofs": np.array([[0]])}, "indices"),
        ({"trace_dofs": np.array([0, 0])}, "indices"),
        ({"coupling": [[1j]]}, "real"),
        ({"load": [np.inf]}, "finite"),
        ({"kernel": [1]}, "matrix"),
        ({"kernel": [[1]], "constraints": [[0]]}, "nonsingularly"),
        ({"kernel": [[1]]}, "nullspace"),
    ],
)
def test_local_invalid_data(change, match):
    values = dict(matrix=[[2.0]], coupling=[[1.0]], load=[0.0], trace_dofs=np.array([0]))
    values.update(change)
    with pytest.raises(ValueError, match=match):
        LocalProblem(**values)


def test_global_validation():
    with pytest.raises(ValueError, match="at least"):
        HybridSystem([])
    with pytest.raises(ValueError, match="contiguous"):
        HybridSystem([interval_problem([0, 2])])
    system = HybridSystem([interval_problem([0, 1])])
    for fixed in ({-1: 0}, {2: 0}, {0.5: 0}, {True: 0}, {0: np.nan}):
        with pytest.raises(ValueError, match="fixed"):
            system.solve(fixed=fixed)
    for weights, target in (([], 0), ([np.ones(2)], np.nan)):
        with pytest.raises(ValueError, match="weight"):
            system.mean_constraint(weights, target)
    with pytest.raises(ValueError, match="shape"):
        system.mean_constraint([np.ones(3)])
    with pytest.raises(ValueError, match="constraint"):
        system.solve(constraints=[(np.ones(3), np.nan)])


@pytest.mark.parametrize("scale", [1e-300, 1e-20, 1.0, 1e200])
def test_kernel_validation_is_invariant_under_operator_units(scale):
    with pytest.raises(ValueError, match="nullspace"):
        LocalProblem([[scale]], [[1.0]], [0.0], np.array([0]), [[1.0]])
    base = interval_problem([0, 1])
    valid = LocalProblem(
        scale * base.matrix,
        base.coupling,
        base.load,
        base.trace_dofs,
        base.kernel,
        base.constraints,
    )
    assert_allclose(valid.kernel, base.kernel)


def test_kernel_validation_checks_each_mode_independently():
    with pytest.raises(ValueError, match="nullspace"):
        LocalProblem(
            np.diag([0.0, 1.0]),
            np.eye(2),
            np.zeros(2),
            np.arange(2),
            np.diag([1e8, 1e-8]),
            np.diag([1e-8, 1e8]),
        )


@pytest.mark.parametrize("solver", ["scipy", "gmres"])
@pytest.mark.parametrize("problem", ["darcy", "stokes"])
def test_unresolved_trace_modes_are_rejected_even_with_compatible_load(solver, problem):
    mesh = TriangleMesh.unit_square()
    degree, components = (1, 1) if problem == "darcy" else (2, 2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(degree) for _ in mesh.faces), components)
    with pytest.raises(LinearSolveError, match="singular|rank|ill-conditioned"):
        if problem == "darcy":
            solve_darcy(
                mesh,
                skeleton=skeleton,
                local_refinement=1,
                dirichlet=lambda points: points[:, 0] + points[:, 1],
                solver=solver,
            )
        else:
            solve_brinkman(
                mesh,
                skeleton=skeleton,
                local_refinement=1,
                dirichlet=lambda points: np.column_stack((points[:, 0], -points[:, 1])),
                solver=solver,
            )


def test_amg_kernel_projection_matches_constrained_lu():
    p = interval_problem([0, 1], force=3.0)
    expected = p.condense()
    result = p.condense("pyamg")
    assert_allclose(result.source, expected.source, atol=1e-12)
    assert_allclose(result.lifts, expected.lifts, atol=1e-12)
    scalar = LocalProblem([[0.0]], [[1.0]], [2.0], np.array([0]), [[1.0]])
    assert_allclose(scalar.condense("pyamg").source, 0)
    definite = LocalProblem([[2.0]], [[1.0]], [2.0], np.array([0]))
    assert_allclose(definite.condense("pyamg").source, [1.0])


def test_neumann_compatibility_uses_pre_elimination_physical_scale():
    """Roundoff after cancelling prescribed flux and source is not a relative unit defect."""
    kernel = np.array([[0.0], [1.0]])
    for source, compatible in ((0.1 + 0.2, True), (0.31, False)):
        problem = LocalProblem(np.diag([1.0, 0.0]), np.eye(2), [0.0, source], [0, 1], kernel)
        system = HybridSystem([problem])
        constraint = system.mean_constraint([np.array([0.0, 1.0])])
        if compatible:
            result = system.solve(fixed={0: 1.0, 1: 0.3}, constraints=[constraint])
            assert_allclose(result.fields[0], [-1.0, 0.0], atol=2e-15)
            assert result.residual < 3e-15
        else:
            with pytest.raises(ValueError, match="incompatible data"):
                system.solve(fixed={0: 1.0, 1: 0.3}, constraints=[constraint])


def test_neumann_incompatibility_cannot_be_hidden_by_cancelling_large_fluxes():
    """The roundoff bound must not become a relative tolerance on huge cancelled fluxes."""
    problem = LocalProblem(
        [[1.0, -1.0], [-1.0, 1.0]],
        np.diag([1.0, -1.0]),
        [1.0, 1.0],
        np.arange(2),
        np.ones((2, 1)),
    )
    system = HybridSystem([problem])
    constraint = system.mean_constraint([np.ones(2)])
    with pytest.raises(ValueError, match="incompatible data"):
        system.solve(fixed={0: 1e10, 1: 1e10}, constraints=[constraint])


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_zero_source_neumann_balance_accepts_only_prescribed_roundoff(scale):
    """A zero-load compatibility row has a componentwise, unit-independent roundoff bound."""
    coupling = np.array([[1.0, 0.0, 0.0], [0.1, 0.2, -0.3]])
    problem = LocalProblem(
        np.diag([1.0, 0.0]), coupling, [0.0, 0.0], np.arange(3), kernel=np.array([[0.0], [1.0]])
    )
    system = HybridSystem([problem])
    gauge = system.mean_constraint([np.array([0.0, 1.0])], 2.0)
    result = system.solve(fixed={0: scale, 1: scale, 2: scale}, constraints=[gauge])
    assert_allclose(result.fields[0], [-scale, 2.0], rtol=3e-15)
    assert result.residual < 1e-8
    with pytest.raises(ValueError, match="incompatible data"):
        system.solve(fixed={0: scale, 1: scale, 2: 1.001 * scale}, constraints=[gauge])


def test_global_tolerance_contract_and_prepared_factorization(monkeypatch):
    """A requested solve tolerance is forwarded and cannot contradict a reused factor."""
    import pymhm.hybrid as module

    problem = LocalProblem(np.eye(2), np.eye(2), [2.0, 3.0], np.arange(2))
    system = HybridSystem([problem])
    original = module.solve_linear
    requested = []

    def checked(matrix, rhs, **options):
        """Record the public numerical option while executing the real sparse solve."""
        requested.append(options["rtol"])
        return original(matrix, rhs, **options)

    monkeypatch.setattr(module, "solve_linear", checked)
    actual = system.solve(rtol=3e-13)
    assert requested == [3e-13]
    with module.factorize(system.matrix, rtol=2e-13) as prepared:
        reused = system.solve(factorization=prepared)
        explicit = system.solve(factorization=prepared, rtol=2e-13)
        assert_allclose(reused.trace, actual.trace)
        assert_allclose(explicit.trace, actual.trace)
        with pytest.raises(ValueError, match="prepared factorization tolerance"):
            system.solve(factorization=prepared, rtol=1e-10)
    for value in (0.0, -1.0, np.nan, np.inf, 1j):
        with pytest.raises(ValueError, match="finite and positive"):
            system.solve(rtol=value)
