"""Compare general coarse elimination with the full saddle algebra and the Stokes limit."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, HybridSystem, LocalProblem, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.vector import solve_brinkman


@pytest.mark.parametrize("kind", ["spd", "indefinite", "nonsymmetric"])
def test_general_coarse_basis_matches_full_saddle_and_gauge(kind: str) -> None:
    """Check physical equations, nonsymmetric blocks and reconstructed integral constraints."""
    rng = np.random.default_rng(731)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    if kind == "spd":
        matrix = q @ np.diag([1, 2, 3, 5, 7, 9]) @ q.T
    elif kind == "indefinite":
        matrix = q @ np.diag([-5, -2, 1, 3, 7, 9]) @ q.T
    else:
        matrix = rng.normal(size=(6, 6)) + 4 * np.eye(6)
    basis = rng.normal(size=(6, 2))
    constraints = np.diag(np.arange(1, 7)) @ basis
    coupling = rng.normal(size=(6, 2))
    load, boundary = rng.normal(size=6), rng.normal(size=2)
    problem = LocalProblem(
        matrix, coupling, load, np.arange(2), constraints=constraints, coarse_basis=basis
    )
    assert problem.kernel.shape == (6, 0)
    system = HybridSystem([problem], boundary_load=boundary)
    monolithic = np.block([[matrix, coupling], [coupling.T, np.zeros((2, 2))]])
    expected = np.linalg.solve(monolithic, np.r_[load, boundary])
    actual = system.solve()
    assert_allclose(actual.fields[0], expected[:6], rtol=1e-11, atol=1e-12)
    assert_allclose(actual.trace, expected[6:], rtol=1e-11, atol=1e-12)
    response = system.responses[0]
    assert_allclose(constraints.T @ response.source, 0, atol=1e-12)
    assert_allclose(constraints.T @ response.lifts, 0, atol=1e-12)
    assert_allclose(constraints.T @ response.retained_basis, constraints.T @ basis, atol=1e-12)
    assert_allclose(matrix @ actual.fields[0] + coupling @ actual.trace, load, atol=1e-11)
    if kind == "nonsymmetric":
        assert not np.allclose(system.matrix.toarray(), system.matrix.toarray().T)
    weights = rng.normal(size=6)
    target = float(weights @ expected[:6])
    row, right = system.mean_constraint([weights], target)
    assert_allclose(row[2:], response.retained_basis.T @ weights, atol=1e-12)
    constrained = system.solve(constraints=[(row, right)])
    assert_allclose(constrained.fields[0], expected[:6], rtol=1e-11, atol=1e-12)
    assert_allclose(constrained.gauge_multipliers, 0, atol=1e-11)


def test_true_kernel_keeps_original_rhs_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use one real LU per local problem, adding A Z loads only for a general basis."""
    import pymhm.core.condensation as hybrid

    widths = []
    original = hybrid.factorize

    @contextmanager
    def record_factorization(matrix: Any, *, solver: str) -> Any:
        """Record the actual multiple-RHS solve while preserving native factor ownership."""
        with original(matrix, solver=solver) as factor:

            def solve(rhs: np.ndarray, **kwargs: Any) -> np.ndarray:
                """Record the load count and execute the actual SciPy factorization."""
                widths.append(rhs.shape[1])
                return factor.solve(rhs, **kwargs)

            yield SimpleNamespace(solve=solve)

    monkeypatch.setattr(hybrid, "factorize", record_factorization)
    laplacian = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)
    kernel = LocalProblem(laplacian, np.eye(3), np.ones(3), np.arange(3), np.ones((3, 1)))
    general = LocalProblem(
        laplacian + np.eye(3), np.eye(3), np.ones(3), np.arange(3), coarse_basis=np.ones((3, 1))
    )
    first, second = kernel.condense(), general.condense()
    assert first.coarse_vectors is None
    assert second.coarse_vectors is not None
    assert widths == [4, 5]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"kernel": np.ones((2, 1))}, "mutually exclusive"),
        ({"coarse_basis": [1, 1]}, "must be a matrix"),
        ({"coarse_basis": np.ones((3, 1))}, "shape"),
        ({"coarse_basis": np.ones((2, 1), dtype=complex)}, "real"),
        ({"constraints": np.zeros((2, 1))}, "nonsingularly"),
    ],
)
def test_general_coarse_basis_validation(change: dict[str, Any], match: str) -> None:
    """Reject conflicting kernels, malformed bases and degenerate moment constraints."""
    values = {"coarse_basis": np.ones((2, 1))}
    values.update(change)
    with pytest.raises(ValueError, match=match):
        LocalProblem(np.eye(2), np.eye(2), np.ones(2), np.arange(2), **values)


@pytest.mark.parametrize("solver", ["pyamg", "amgx"])
def test_general_coarse_basis_rejects_unsupported_amg(solver: str) -> None:
    """Reject the unsupported projection before importing or invoking an optional AMG backend."""
    problem = LocalProblem([[2.0]], [[1.0]], [1.0], np.array([0]), coarse_basis=[[1.0]])
    with pytest.raises(ValueError, match="AMG.*general coarse_basis"):
        problem.condense(solver)


@pytest.mark.parametrize("formulation", ["taylor-hood", "usfem"])
@pytest.mark.parametrize("drag", [1e-8, 1e-12, 1e-16, 1.0, 100.0])
def test_near_stokes_pressure_patch_and_nonzero_gauge(formulation: str, drag: float) -> None:
    """Recover zero velocity and a linear pressure without inverse-drag cancellation."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    with threadpool_limits(limits=1):
        result = solve_brinkman(
            mesh,
            drag=drag,
            source=[1.0, -1.0],
            skeleton=skeleton,
            formulation=formulation,
            local_refinement=4,
            mean_pressure=2.3,
        )
        assert result.l2_error([0, 0]) < 1e-12
        assert result.pressure_l2_error(lambda x: x[:, 0] - x[:, 1] + 2.3) < 1e-11
        assert result.divergence_l2() < 1e-11
    assert_allclose(result.hybrid.gauge_multipliers, 0, atol=1e-11)


@pytest.mark.parametrize("formulation", ["taylor-hood", "usfem"])
@pytest.mark.parametrize("resolution", [2, 4])
def test_manufactured_brinkman_approaches_discrete_stokes_limit(
    formulation: str, resolution: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare non-affine fields at three positive drags with a zero-drag reference."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    exact = importlib.import_module("manufactured")
    mesh = TriangleMesh.unit_square(resolution)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    options = {
        "skeleton": skeleton,
        "formulation": formulation,
        "local_refinement": 4,
        "quadrature_order": 6,
        "mean_pressure": 2.3,
    }
    with threadpool_limits(limits=1):
        stokes = solve_brinkman(mesh, source=exact.stokes_source, **options)
        stokes_velocity_error = stokes.l2_error(exact.stokes_velocity, order=8)
        stokes_pressure_error = stokes.pressure_l2_error(
            lambda x: exact.stokes_pressure(x) + 2.3, order=8
        )
        for drag in (1e-8, 1e-12, 1e-16):
            result = solve_brinkman(
                mesh,
                drag=drag,
                source=lambda x, value=drag: exact.stokes_source(x, drag=value),
                **options,
            )
            assert_allclose(result.values, stokes.values, rtol=1e-9, atol=1e-9)
            assert_allclose(result.pressure, stokes.pressure, rtol=1e-9, atol=1e-9)
            assert (
                abs(result.l2_error(exact.stokes_velocity, order=8) - stokes_velocity_error) < 1e-9
            )
            error = result.pressure_l2_error(lambda x: exact.stokes_pressure(x) + 2.3, order=8)
            assert abs(error - stokes_pressure_error) < 1e-9
            assert_allclose(result.hybrid.gauge_multipliers, 0, atol=1e-10)


def test_true_kernel_and_near_kernel_share_one_global_system() -> None:
    """Couple an exact nullspace and a positive-reaction coarse mode across one interface."""
    laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
    mass = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6
    basis = np.ones((2, 1))
    first = LocalProblem(laplacian, np.eye(2), np.zeros(2), np.array([0, 1]), basis, mass @ basis)
    reaction = 1e-12
    second = LocalProblem(
        laplacian + reaction * mass,
        np.diag([-1.0, 1.0]),
        reaction * mass @ [1.0, 2.0],
        np.array([1, 2]),
        constraints=mass @ basis,
        coarse_basis=basis,
    )
    system = HybridSystem([first, second], boundary_load=[0.0, 0.0, 2.0])
    result = system.solve()
    assert_allclose(result.fields, [[0.0, 1.0], [1.0, 2.0]], atol=1e-12)
    assert_allclose(result.trace, [1.0, -1.0, -1.0], atol=1e-12)
    assert_allclose(result.coarse, [[0.5], [1.5]], atol=1e-12)
