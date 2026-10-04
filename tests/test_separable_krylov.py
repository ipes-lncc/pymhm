"""Verify matrix-free factors, refined-grid SPD preconditioning and complete Qk solves."""

import numpy as np
import pytest

from pymhm._legacy.models.darcy.separable import SeparableField, solve_separable_diffusion
from pymhm.fem.scalar.quadrilateral import qk_space, quadrilateral_operators
from pymhm.linalg.linear import LinearSolveError
from pymhm.linalg.separable import solve_separable_krylov, tensor_diffusion_operator
from pymhm.meshes.cartesian import CartesianMacroMesh


def coefficient() -> SeparableField:
    """Return a positive, variable, explicitly separated test coefficient."""
    return SeparableField(((2, 1), (lambda x: x, 1), (0.5, lambda y: y)))


def forcing() -> SeparableField:
    """Derive -div(K grad p) for p=1+x+xy+y² and K=2+x+y/2."""
    return SeparableField(((-5, 1), (lambda x: -2.5 * x, 1), (-3, lambda y: y)))


def exact(points: np.ndarray) -> np.ndarray:
    """Evaluate the quadratic physical solution with inhomogeneous exterior values."""
    x, y = points.T
    return 1 + x + x * y + y**2


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_matrix_free_matches_native_and_is_spd(degree: int) -> None:
    """Compare each operator/load entry, and prove finite-dimensional SPD on a small mesh."""
    mesh = CartesianMacroMesh(2, 3, (-0.25, 0.75, 0.1, 1.4))
    operator = tensor_diffusion_operator(
        (2, 3), mesh.bounds, degree, permeability=coefficient(), source=forcing(), order=7
    )
    expected = quadrilateral_operators(
        mesh, degree, permeability=coefficient(), source=forcing(), order=7
    )
    actual = operator.assemble()
    np.testing.assert_allclose(actual.toarray(), expected[0].toarray(), rtol=8e-12, atol=5e-13)
    np.testing.assert_allclose(operator.load, expected[2], rtol=8e-12, atol=5e-13)
    vector = np.sin(np.arange(actual.shape[0]))
    np.testing.assert_allclose(operator.apply(vector), actual @ vector, rtol=2e-12, atol=2e-13)
    interior = operator.interior().assemble().toarray()
    np.testing.assert_allclose(interior, interior.T, atol=2e-13)
    assert np.linalg.eigvalsh(interior).min() > 0.01
    low = tensor_diffusion_operator(
        (2 * degree, 3 * degree),
        mesh.bounds,
        1,
        permeability=coefficient(),
        source=forcing(),
        order=3,
    )
    assert np.linalg.eigvalsh(low.interior().assemble().toarray()).min() > 0.001


@pytest.mark.parametrize("degree", [2, 3, 5])
def test_complete_solver_matches_exact_and_direct(degree: int) -> None:
    """Retain load and inhomogeneous boundary elimination while changing only the solver."""
    mesh = CartesianMacroMesh(3, 2)
    options = dict(
        degree=degree,
        permeability=coefficient(),
        source=forcing(),
        dirichlet=exact,
        quadrature_order=7,
    )
    result = solve_separable_krylov(mesh, **options)
    direct = solve_separable_diffusion(mesh, **options)
    np.testing.assert_allclose(result.pressure, exact(qk_space(mesh, degree)[1]), atol=2e-10)
    np.testing.assert_allclose(result.pressure, direct.pressure, atol=2e-10)
    assert result.relative_equation_residual <= 1e-10
    assert result.iterations > 0
    assert result.preconditioner_levels > 0


def test_extended_residual_and_zero_rhs() -> None:
    """Check wide accumulation and zero-source homogeneous boundary cases explicitly."""
    mesh = CartesianMacroMesh(3)
    if np.finfo(np.longdouble).eps < np.finfo(float).eps:
        result = solve_separable_krylov(
            mesh,
            degree=3,
            permeability=coefficient(),
            source=forcing(),
            dirichlet=exact,
            refinement_precision="extended",
            rtol=1e-14,
        )
        assert result.pressure.dtype == np.dtype(np.longdouble)
        assert result.relative_equation_residual <= 1e-14
    result = solve_separable_krylov(
        mesh, degree=2, permeability=coefficient(), source=SeparableField(())
    )
    assert not result.pressure.any()
    assert result.relative_equation_residual == 0
    result = solve_separable_krylov(
        CartesianMacroMesh(1),
        degree=1,
        permeability=coefficient(),
        source=forcing(),
        dirichlet=exact,
    )
    np.testing.assert_array_equal(result.pressure, exact(qk_space(result.mesh, 1)[1]))
    assert result.iterations == 0


def test_real_nonconvergence_is_an_error() -> None:
    """Never return a field that exceeds the stated original-equation residual criterion."""
    with pytest.raises(LinearSolveError, match="failed to converge"):
        solve_separable_krylov(
            CartesianMacroMesh(4), degree=3, permeability=coefficient(), source=forcing(), maxiter=1
        )


@pytest.mark.parametrize("bounds", [(0, 1, 0), (0, 1, 0, np.inf), (0, 1, 1, 0), (0, 1j, 0, 1)])
def test_invalid_geometry(bounds: tuple) -> None:
    """Reject invalid rectangular bounds before forming matrix factors."""
    with pytest.raises(ValueError, match="bounds"):
        tensor_diffusion_operator(
            (2, 2), bounds, 2, permeability=coefficient(), source=forcing(), order=4
        )


def test_invalid_contracts() -> None:
    """Reject unsupported fields, spaces, iteration counts and array contracts."""
    options = dict(permeability=coefficient(), source=forcing(), order=4)
    with pytest.raises(ValueError, match="two"):
        tensor_diffusion_operator((2,), (0, 1, 0, 1), 2, **options)
    with pytest.raises(ValueError, match="nonempty"):
        tensor_diffusion_operator(
            (2, 2), (0, 1, 0, 1), 2, **dict(options, permeability=SeparableField(()))
        )
    with pytest.raises(TypeError, match="source"):
        tensor_diffusion_operator((2, 2), (0, 1, 0, 1), 2, **dict(options, source=1))
    operator = tensor_diffusion_operator((2, 2), (0, 1, 0, 1), 2, **options)
    for bad in (np.zeros(2), np.full(25, np.nan)):
        with pytest.raises(ValueError, match="input"):
            operator.apply(bad)
    settings = dict(degree=2, permeability=coefficient(), source=forcing())
    with pytest.raises(TypeError, match="Cartesian"):
        solve_separable_krylov(None, **settings)
    with pytest.raises(ValueError, match="refinement_precision"):
        solve_separable_krylov(CartesianMacroMesh(2), **settings, refinement_precision="invalid")
    with pytest.raises(ValueError, match="integer"):
        solve_separable_krylov(CartesianMacroMesh(2), **settings, maxiter=0)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="finite"):
        tensor_diffusion_operator(
            (2, 2), (0, 1, 0, 1), 2, **dict(options, source=SeparableField(((1e308, 1e308),)))
        )


def test_unachievable_true_residual_is_rejected() -> None:
    """A sub-roundoff requested criterion must fail after bounded correction attempts."""
    with pytest.raises(LinearSolveError, match="true residual"):
        solve_separable_krylov(
            CartesianMacroMesh(2),
            degree=3,
            permeability=coefficient(),
            source=forcing(),
            rtol=1e-30,
            refinement_precision="extended",
        )


def test_narrow_extended_platform_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the documented rejection when long double supplies no additional digits."""
    original = np.finfo
    monkeypatch.setattr(
        np, "finfo", lambda dtype: original(float if dtype is np.longdouble else dtype)
    )
    from pymhm.linalg.linear import SolverUnavailableError

    with pytest.raises(SolverUnavailableError, match="wider"):
        solve_separable_krylov(
            CartesianMacroMesh(2),
            degree=2,
            permeability=coefficient(),
            source=forcing(),
            refinement_precision="extended",
        )


def test_matrix_factor_overflow_contract() -> None:
    """Reject an unrepresentable stiffness while the physical input coefficient is finite."""
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="finite"):
        tensor_diffusion_operator(
            (4, 4),
            (0, 1, 0, 1),
            2,
            permeability=SeparableField(((1e308, 1),)),
            source=SeparableField(()),
            order=4,
        )


def test_native_amg_cycle_is_symmetric_positive_on_control_operator() -> None:
    """Check the actual symmetric smoother/Galerkin cycle used by CG, including coarse solve."""
    import pyamg

    low = (
        tensor_diffusion_operator(
            (10, 12), (0, 1, 0, 1), 1, permeability=coefficient(), source=forcing(), order=3
        )
        .interior()
        .assemble()
    )
    hierarchy = pyamg.smoothed_aggregation_solver(
        low,
        symmetry="hermitian",
        smooth=("jacobi", {"weighting": "local"}),
        presmoother=("gauss_seidel", {"sweep": "symmetric"}),
        postsmoother=("gauss_seidel", {"sweep": "symmetric"}),
    )
    action = hierarchy.aspreconditioner(cycle="V")
    matrix = np.column_stack([action @ column for column in np.eye(low.shape[0])])
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-14, rtol=3e-14)
    assert np.linalg.eigvalsh(matrix).min() > 0
