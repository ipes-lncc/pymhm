"""Original-equation refinement against complete systems and physical gauges."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.core.contracts import LocalProblem
from pymhm.core.refinement import refine_hybrid
from pymhm.core.system import HybridSystem
from pymhm.linalg.linear import LinearSolveError, SolverUnavailableError


def perturbed(system, solution, fixed=()):
    """Perturb valid retained coordinates without changing prescribed trace data."""
    trace = solution.trace.copy()
    free = np.setdiff1d(np.arange(len(trace)), list(fixed))
    trace[free] += 0.003
    coarse = tuple(c + 0.002 for c in solution.coarse)
    fields = tuple(
        r.reconstruct(trace[r.problem.trace_dofs], c)
        for r, c in zip(system.responses, coarse, strict=True)
    )
    return replace(solution, trace=trace, coarse=coarse, fields=fields)


@pytest.mark.parametrize("gauged", [False, True])
@pytest.mark.parametrize("retained", ["none", "kernel", "general"])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_petrov_multiple_modes_match_uncondensed_physical_equations(retained, scale, gauged):
    """Resolve distinct test/trial spaces, nonzero weak boundary data and two modes."""
    rng = np.random.default_rng(814)
    left, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    right, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    eigen = np.arange(1.0, 8.0)
    if retained == "kernel":
        eigen[-2:] = 0
    matrix = scale * left @ np.diag(eigen) @ right.T
    coupling = scale * rng.normal(size=(7, 3))
    test = rng.normal(size=(7, 3))
    load = scale * rng.normal(size=7)
    boundary = rng.normal(size=3)
    options = {}
    if retained != "none":
        options = dict(
            constraints=np.diag(np.arange(1, 8)) @ right[:, -2:],
            test_constraints=np.diag(np.arange(2, 9)) @ left[:, -2:],
        )
        if retained == "kernel":
            options.update(kernel=right[:, -2:], left_kernel=left[:, -2:])
        else:
            options.update(coarse_basis=right[:, -2:], test_basis=left[:, -2:])
    p = LocalProblem(matrix, coupling, load, np.arange(3), test_coupling=test, **options)
    system = HybridSystem([p], boundary_load=boundary)
    initial = perturbed(system, system.solve())
    exact = np.linalg.solve(
        np.block([[matrix, coupling], [test.T, np.zeros((3, 3))]]), np.r_[load, boundary]
    )
    moments = []
    if gauged:
        weights = [rng.normal(size=7)]
        target = float(weights[0] @ exact[:7])
        moments = [(weights, target)]
        initial = perturbed(
            system, system.solve(constraints=[system.mean_constraint(weights, target)])
        )
    result = refine_hybrid(
        system, initial, boundary_load=boundary, moments=moments, rtol=1e-12, max_steps=2
    )
    assert len(result.residual_norms) > 1
    assert result.residual_norms[-1] <= 1e-12 * result.rhs_norm
    assert_allclose(result.solution.fields[0], exact[:7], rtol=1e-10, atol=1e-10)
    assert_allclose(result.solution.trace, exact[7:], rtol=1e-10, atol=1e-10)
    assert result.residual_norms[0] > result.residual_norms[-1] * 1e5
    assert result.solution.residual < 2e-12
    assert_allclose(result.solution.gauge_multipliers, 0, atol=2e-8)


def gauged():
    """Build a two-cell Neumann problem with a nonzero flux and pressure mean."""
    cells = [
        LocalProblem(
            [[1.0, -1], [-1, 1]],
            np.diag([1.0, -1]),
            load,
            np.array([i, i + 1]),
            kernel=np.ones((2, 1)),
        )
        for i, load in enumerate(([1.0, 0.0], [0.0, -1.0]))
    ]
    system = HybridSystem(cells)
    fixed = {0: 0.2, 2: 0.2}
    weights = [np.ones(2) / 2] * 2
    moments = [(weights, 2.0)]
    value = system.solve(fixed=fixed, constraints=[system.mean_constraint(weights, 2.0)])
    return system, value, fixed, moments


def test_prescribed_flux_and_physical_gauge_are_refined_together():
    """Preserve nonhomogeneous Neumann coefficients while restoring the pressure integral."""
    system, value, fixed, moments = gauged()
    initial = perturbed(system, value, fixed)
    result = refine_hybrid(system, initial, fixed=fixed, moments=moments, rtol=1e-13)
    assert_allclose(result.solution.fields, value.fields, atol=2e-13)
    assert_allclose(result.solution.trace, value.trace, atol=2e-13)
    assert_allclose(
        sum(w @ u for w, u in zip(moments[0][0], result.solution.fields, strict=True)),
        2.0,
        atol=2e-14,
    )
    with pytest.raises(ValueError, match="each original physical moment"):
        refine_hybrid(system, initial, fixed=fixed)


def simple():
    """Provide a small invertible problem with a nonzero original load."""
    system = HybridSystem(
        [LocalProblem([[2.0, 0.2], [0.2, 3.0]], np.eye(2), [1.0, 2.0], np.arange(2))],
        boundary_load=[2.0, 3.0],
    )
    return system, system.solve()


def test_zero_steps_and_exhaustion_do_not_hide_failed_original_equations():
    """Return an accepted initial state or explicitly reject it without any correction."""
    system, value = simple()
    assert (
        len(refine_hybrid(system, value, boundary_load=[2.0, 3.0], max_steps=0).residual_norms) == 1
    )
    with pytest.raises(LinearSolveError, match="after 0 corrections"):
        refine_hybrid(system, perturbed(system, value), boundary_load=[2.0, 3.0], max_steps=0)


def test_failed_correction_releases_factors_and_reports_original_defect(monkeypatch):
    """An ineffective local inverse cannot silently mark unresolved fields as converged."""
    from contextlib import contextmanager

    import pymhm.core.refinement as owner

    closed = []

    @contextmanager
    def ineffective(matrix, **options):
        """Exercise factor cleanup while deliberately returning no source correction."""
        try:
            yield SimpleNamespace(solve=lambda rhs, **kwargs: np.zeros_like(rhs))
        finally:
            closed.append(True)

    system, value = simple()
    bad = replace(value, fields=(value.fields[0] + [0.01, -0.02],))
    monkeypatch.setattr(owner, "factorize", ineffective)
    with pytest.raises(LinearSolveError, match="after 1 corrections"):
        refine_hybrid(system, bad, boundary_load=[2.0, 3.0], max_steps=1, rtol=1e-14)
    assert closed == [True]


@pytest.mark.parametrize(
    "options,match",
    [
        ({"max_steps": -1}, "nonnegative"),
        ({"rtol": -1}, "nonnegative"),
        ({"refinement_precision": "automatic"}, "precision"),
        ({"fixed": {True: 0}}, "valid trace"),
        ({"fixed": {3: 0}}, "valid trace"),
        ({"fixed": {0: np.nan}}, "finite"),
        ({"fixed": {0: 23}}, "supplied fixed"),
        ({"moments": [([], 1.0)]}, "weight array"),
        ({"moments": [([np.ones(2)], np.inf)]}, "finite moment"),
    ],
)
def test_invalid_refinement_contracts_fail_before_factoring(options, match):
    """Validate precision, tolerances, fixed data and physical moments explicitly."""
    system, value = simple()
    with pytest.raises(ValueError, match=match):
        refine_hybrid(system, value, **options)


def test_field_layout_and_extended_platform_validation(monkeypatch):
    """Reject incomplete fields and unsupported wide storage without a silent fallback."""
    import pymhm.core.refinement as owner

    system, value = simple()
    with pytest.raises(ValueError, match="one field"):
        refine_hybrid(system, replace(value, fields=()))
    actual = np.finfo
    monkeypatch.setattr(owner.np, "finfo", lambda dtype: actual(float))
    with pytest.raises(SolverUnavailableError, match="wider"):
        refine_hybrid(system, value, refinement_precision="extended")


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(float).eps, reason="requires wider accumulation"
)
def test_extended_fields_retain_extra_digits():
    """Retain explicit wide storage in fields, trace and corrected coarse coordinates."""
    system, value, fixed, moments = gauged()
    result = refine_hybrid(
        system,
        perturbed(system, value, fixed),
        fixed=fixed,
        moments=moments,
        refinement_precision="extended",
    )
    assert result.solution.trace.dtype == np.longdouble
    assert all(u.dtype == np.longdouble for u in result.solution.fields)


def test_all_prescribed_traces_use_the_original_eliminated_load_scale():
    """Count physical prescribed action when the volume source and free boundary load vanish."""
    system = HybridSystem(
        [LocalProblem([[2.0, 0.2], [0.2, 3.0]], np.eye(2), [0.0, 0.0], np.arange(2))]
    )
    fixed = {0: 1.0, 1: 2.0}
    value = system.solve(fixed=fixed)
    initial = replace(value, fields=(value.fields[0] + [0.01, -0.02],))
    result = refine_hybrid(system, initial, fixed=fixed, rtol=1e-13)
    assert result.rhs_norm == np.linalg.norm([1.0, 2.0])
    assert_allclose(result.solution.fields, value.fields, atol=2e-14)


def test_refined_field_archive_preserves_physical_replay(tmp_path):
    """Archive all corrected field coordinates instead of recomputing old response maps."""
    system, value, fixed, moments = gauged()
    result = refine_hybrid(system, perturbed(system, value, fixed), fixed=fixed, moments=moments)
    path = tmp_path / "fields.npz"
    np.savez(
        path,
        trace=result.solution.trace,
        **{f"field_{i}": u for i, u in enumerate(result.solution.fields)},
    )
    with np.load(path, allow_pickle=False) as saved:
        replay = tuple(saved[f"field_{i}"] for i in range(len(result.solution.fields)))
        for field, restored in zip(result.solution.fields, replay, strict=True):
            assert np.array_equal(field, restored)
            assert field.dtype == restored.dtype
        restored_solution = replace(result.solution, trace=saved["trace"], fields=replay)
    checked = refine_hybrid(system, restored_solution, fixed=fixed, moments=moments, max_steps=0)
    assert checked.residual_norms == (result.residual_norms[-1],)


@pytest.mark.parametrize("max_steps", [0, 2])
def test_a_physical_gauge_cannot_be_added_after_the_initial_solve(max_steps):
    """Reject retrospective constraints even when the supplied field meets their target."""
    system, value = simple()
    weights = [np.ones_like(field) for field in value.fields]
    target = sum(w @ field for w, field in zip(weights, value.fields, strict=True))
    with pytest.raises(ValueError, match="each original physical moment"):
        refine_hybrid(system, value, moments=[(weights, target)], max_steps=max_steps)
