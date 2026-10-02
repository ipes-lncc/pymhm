"""Physical signs, exact complex patches, materials and local Helmholtz inversion."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pymhm.helmholtz import solve_helmholtz
from pymhm.helmholtz_forms import (
    acoustic_space,
    complex_vector,
    positive_values,
    real_matrix,
    real_vector,
    volume_forms,
)
from pymhm.helmholtz_spaces import OscillatoryFaceSpace, helmholtz_skeleton
from pymhm.mesh import SkeletonSpace, TriangleMesh
from pymhm.planar_material import PlanarMaterial, PlanarRegion
from pymhm.polygon import PolygonMesh
from pymhm.quadrilateral import CartesianMacroMesh
from pymhm.reservoir import CartesianCellField
from pymhm.solvers import LinearSolveError


@pytest.mark.parametrize("rectangle", [False, True])
def test_oblique_acoustic_material_integration(rectangle):
    """Exact plane cuts preserve analytical mass/gradient moments without fitting the space."""
    mesh = CartesianMacroMesh(1) if rectangle else TriangleMesh.unit_square(1)
    density = PlanarMaterial(1, (PlanarRegion([[1, 1]], [0.83], 4),))
    modulus = PlanarMaterial(2, (PlanarRegion([[1, 1]], [0.61], 3),))
    k, m, _ = volume_forms(mesh, 2, density, modulus, 0, 6)
    _, nodes = acoustic_space(mesh, 2)
    x, one = nodes[:, 0], np.ones(len(nodes))
    assert x @ k @ x == pytest.approx(1 - 0.75 * 0.83**2 / 2, abs=2e-14)
    assert one @ m @ one == pytest.approx(0.5 - 0.61**2 / 12, abs=2e-14)
    other_k, other_m, _ = volume_forms(mesh, 2, density, modulus, 0, 7)
    np.testing.assert_allclose(k.toarray(), other_k.toarray(), atol=2e-14)
    np.testing.assert_allclose(m.toarray(), other_m.toarray(), atol=2e-14)
    result = solve_helmholtz(mesh, omega=2, density=density, bulk_modulus=modulus, degree=2)
    assert result.l2_error(0) == 0
    np.testing.assert_allclose(positive_values(density, [[0.1, 0.1], [0.9, 0.9]]), [4, 1])
    anisotropic = PlanarMaterial([[1, 0], [0, 2]], (PlanarRegion([[1, 1]], [0.5], 1),))
    with pytest.raises(ValueError, match="isotropic"):
        volume_forms(mesh, 2, anisotropic, 1, 0, 4)


def pressure(points):
    """Complex quadratic with independently differentiated gradient and Laplacian."""
    x, y = points.T
    return 1 + 2j + (2 - 1j) * x + (1 + 3j) * y + (1 - 2j) * x * y + x**2


def gradient(points):
    """Exact physical gradient of the complex quadratic pressure."""
    x, y = points.T
    return np.column_stack((2 - 1j + (1 - 2j) * y + 2 * x, 1 + 3j + (1 - 2j) * x))


def force(points):
    """Return -Delta(p)-omega²*p with omega=2, rho=kappa=1."""
    return -2 - 4 * pressure(points)


def impedance(points, normals):
    """Return du/dn-i*omega*u with the outgoing-wave convention."""
    return np.sum(gradient(points) * normals, axis=1) - 2j * pressure(points)


@pytest.mark.parametrize("rectangle", [False, True])
def test_macro_geometry_is_evaluated_once_before_local_dispatch(rectangle, monkeypatch):
    """Global geometry cost is independent of the number of dispatched local faces."""
    mesh = CartesianMacroMesh(3) if rectangle else TriangleMesh.unit_square(3)
    skeleton = helmholtz_skeleton(mesh, 2)
    exterior = mesh.boundary_faces
    calls = dict.fromkeys(("normals", "lengths", "boundary_faces"), 0)
    for name in calls:
        original = getattr(type(mesh), name).fget

        def counted(instance, original=original, name=name):
            """Count global evaluations while preserving all local mesh geometry."""
            if instance is mesh:
                calls[name] += 1
            return original(instance)

        monkeypatch.setattr(type(mesh), name, property(counted))
    result = solve_helmholtz(
        mesh,
        omega=2,
        source=force,
        degree=2,
        skeleton=skeleton,
        dirichlet=pressure,
        absorbing={int(exterior[0]): impedance},
        neumann={int(exterior[1]): lambda p, n: -np.sum(gradient(p) * n, axis=1)},
    )
    assert calls == {"normals": 1, "lengths": 1, "boundary_faces": 3}
    assert result.l2_error(pressure) < 3e-12
    assert result.gradient_l2_error(gradient) < 4e-12


@pytest.mark.parametrize("rectangle", [False, True])
def test_exact_flux_projection_has_linear_geometry_work(rectangle, monkeypatch):
    """Polynomial physical traces reconstruct the exact field without repeated mesh scans."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    diagnostic = importlib.import_module("examples.helmholtz_stability")
    mesh = CartesianMacroMesh(3) if rectangle else TriangleMesh.unit_square(3)
    result = solve_helmholtz(mesh, omega=2, degree=2, source=force, absorbing=impedance)
    original = type(mesh).normals.fget
    calls = []

    def counted(instance):
        """Observe each complete normal-array evaluation for the projected physical field."""
        if instance is mesh:
            calls.append(instance)
        return original(instance)

    monkeypatch.setattr(type(mesh), "normals", property(counted))
    projected = diagnostic.projected_solution(result, SimpleNamespace(gradient=gradient))
    assert len(calls) == 1
    assert projected.l2_error(pressure) < 3e-12
    np.testing.assert_allclose(projected.trace, result.trace, atol=3e-12)


@pytest.mark.parametrize("kind", ["triangle", "rectangle", "polygon"])
@pytest.mark.parametrize("boundary", ["absorbing", "dirichlet", "neumann", "mixed"])
def test_complex_quadratic_patch(kind, boundary):
    """Every boundary partition reproduces the same physical nonzero complex field."""
    mesh = TriangleMesh.unit_square(1) if kind == "triangle" else CartesianMacroMesh(1)
    if kind == "polygon":
        mesh = PolygonMesh(
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), ((0, 1, 2, 3),)
        )
    options = {"absorbing": None, "dirichlet": pressure}
    if boundary == "absorbing":
        options["absorbing"] = impedance
    elif boundary == "neumann":
        options["neumann"] = lambda p, n: -np.sum(gradient(p) * n, axis=1)
    elif boundary == "mixed":
        options["absorbing"] = {int(mesh.boundary_faces[0]): impedance}
        options["neumann"] = {
            int(mesh.boundary_faces[1]): lambda p, n: -np.sum(gradient(p) * n, axis=1)
        }
    result = solve_helmholtz(mesh, omega=2, source=force, degree=2, **options)
    assert result.l2_error(pressure) < 3e-12
    assert result.gradient_l2_error(gradient) < 4e-12
    assert np.max(abs(result.conservation_residuals())) < 2e-12
    reference = np.array([[0.2, 0.3, 0.5]]) if kind != "rectangle" else np.array([[0.2, 0.3]])
    for cell in range(len(mesh.cells)):
        points, values, derivatives = result.sample(cell, reference)
        np.testing.assert_allclose(values.ravel(), pressure(points.reshape(-1, 2)), atol=3e-12)
        np.testing.assert_allclose(
            derivatives.reshape(-1, 2), gradient(points.reshape(-1, 2)), atol=4e-12
        )


@pytest.mark.parametrize("point_sources", [(), [(0.5, 0.5, 1), (0.1, 0.7, -0.3)]])
def test_realification_and_uncondensed_complex_saddle(point_sources):
    """Realification and condensation match direct complex equations, including impedance."""
    mesh = TriangleMesh.unit_square(1)
    result = solve_helmholtz(
        mesh,
        omega=2.1,
        source=1 + 3j,
        point_sources=point_sources,
        dirichlet=pressure,
        absorbing={int(mesh.boundary_faces[0]): impedance},
        degree=2,
    )
    diagonal, loads, blocks = [], [], []
    boundary = np.zeros(result.skeleton.size // 2, dtype=complex)
    fixed = []
    for response, (_, _, data, prescribed) in zip(
        result.system.responses, result.system.local_metadata, strict=True
    ):
        problem = response.problem
        matrix = problem.matrix[::2, ::2] + 1j * problem.matrix[1::2, ::2]
        dofs = problem.trace_dofs[::2] // 2
        coupling = sparse.lil_matrix((len(problem.load) // 2, len(boundary)), dtype=complex)
        coupling[:, dofs] = problem.coupling[::2, ::2]
        diagonal.append(matrix)
        loads.append(complex_vector(problem.load))
        blocks.append(coupling.tocsc())
        boundary[dofs] += complex_vector(data)
        fixed.extend(dofs[list(prescribed)])
    free = np.setdiff1d(np.arange(len(boundary)), fixed)
    a, b = sparse.block_diag(diagonal), sparse.vstack(blocks)[:, free]
    matrix = sparse.bmat([[a, b], [b.T, None]], format="csc")
    solution = spsolve(matrix, np.r_[np.concatenate(loads), boundary[free]])
    np.testing.assert_allclose(solution[: a.shape[0]], np.concatenate(result.pressure), atol=3e-12)
    np.testing.assert_allclose(solution[a.shape[0] :], result.trace[free], atol=3e-12)
    z = np.arange(matrix.shape[0]) * (1 + 0.3j)
    np.testing.assert_allclose(
        real_matrix(matrix) @ real_vector(z), real_vector(matrix @ z), atol=1e-11
    )


@pytest.mark.parametrize("kind", ["triangle", "rectangle", "polygon"])
def test_discrete_acoustic_point_load_and_macro_balance(kind):
    """Dirac strengths appear exactly once in local test functionals and physical balances."""
    mesh = TriangleMesh.unit_square(2) if kind == "triangle" else CartesianMacroMesh(2)
    if kind == "polygon":
        mesh = PolygonMesh(mesh.points, tuple(mesh.cells))
    sources = np.array([[0.5, 0.5, 1], [0.173, 0.281, -0.25]])
    solution = solve_helmholtz(mesh, omega=2, point_sources=sources, degree=3)
    assert sum(p[:, 2].sum() for p in solution.point_sources) == pytest.approx(0.75)
    np.testing.assert_allclose(solution.conservation_residuals(), 0, atol=3e-13)
    total = 0.0
    for fine, response in zip(solution.local_meshes, solution.system.responses, strict=True):
        _, nodes = acoustic_space(fine, 3)
        load = complex_vector(response.problem.load)
        total += load @ (nodes[:, 0] ** 3 + 2 * nodes[:, 1] ** 2)
    expected = sources[:, 2] @ (sources[:, 0] ** 3 + 2 * sources[:, 1] ** 2)
    assert total == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize("rectangles", [False, True])
def test_submacro_material_interface_and_cut_integration(rectangles):
    """Exact transmission and separate rho/kappa partitions retain physical flux signs."""
    mesh = CartesianMacroMesh(1) if rectangles else TriangleMesh.unit_square(1)
    rho = CartesianCellField(np.array([[1.0], [4.0]]), (0.5, 1.0))
    kappa = CartesianCellField(np.array([[2.0, 3.0, 5.0]]), (1.0, 1 / 3))

    def exact(p):
        """Continuous pressure with density-scaled one-sided gradients."""
        return (1 + 2j) * (1 + p[:, 0] + 3 * np.maximum(p[:, 0] - 0.5, 0))

    def exact_gradient(p):
        """Return the one-sided gradient of the transmission solution."""
        return np.column_stack(((1 + 2j) * np.where(p[:, 0] < 0.5, 1.0, 4.0), np.zeros(len(p))))

    def source(p):
        """Diffusive flux is constant, leaving only the negative mass source."""
        return -4 * exact(p) / kappa(p)

    result = solve_helmholtz(
        mesh,
        omega=2,
        density=rho,
        bulk_modulus=kappa,
        source=source,
        dirichlet=exact,
        absorbing=None,
        degree=2,
        local_refinement=6,
    )
    assert result.l2_error(exact, 10) < 2e-11
    assert result.gradient_l2_error(exact_gradient, 10) < 5e-11
    # Nonaligned cells: coefficient integration is invariant under higher Gauss orders.
    fine = mesh.submesh(0, 1)
    low = volume_forms(fine, 2, rho, kappa, 0j, 4)
    high = volume_forms(fine, 2, rho, kappa, 0j, 7)
    for a, b in zip(low[:2], high[:2], strict=True):
        np.testing.assert_allclose(a.toarray(), b.toarray(), atol=3e-14)


def test_local_resonance_is_rejected_without_shift():
    """A local discrete Neumann eigenfrequency is not silently damped or gauged."""
    mesh = CartesianMacroMesh(1)
    # Q1 on one unit square has generalized Neumann eigenvalues 0,12,12,24.
    with pytest.raises(LinearSolveError, match="rank|singular|residual"):
        solve_helmholtz(mesh, omega=np.sqrt(12), absorbing=None, degree=1, local_refinement=1)


@pytest.mark.parametrize("callback", [False, True])
def test_diagonal_complex_pml_patch(callback):
    """PML multiplies diffusion and mass by their distinct coordinate-transform factors."""
    stretch = np.array([1 + 0.3j, 1 + 0.6j])
    tensor = stretch[::-1] / stretch
    mass = np.prod(stretch)
    data = (lambda p: np.broadcast_to(stretch, (len(p), 2))) if callback else stretch
    result = solve_helmholtz(
        TriangleMesh.unit_square(1),
        omega=2,
        degree=2,
        source=lambda p: -2 * tensor[0] - 4 * mass * pressure(p),
        dirichlet=pressure,
        absorbing=None,
        pml_stretch=data,
    )
    assert result.l2_error(pressure) < 3e-12
    assert result.gradient_l2_error(gradient) < 3e-12
    assert max(abs(result.conservation_residuals())) < 2e-12
    for invalid in ((1 - 1j, 1), (0, 1), [1, 2, 3], [np.nan, 1]):
        with pytest.raises(ValueError, match="stretches"):
            solve_helmholtz(
                TriangleMesh.unit_square(1), omega=2, absorbing=None, pml_stretch=invalid
            )
    with pytest.raises(ValueError, match="PML"):
        solve_helmholtz(TriangleMesh.unit_square(1), omega=2, pml_stretch=(1, 1))


def test_oscillatory_span_projection_and_polynomial_limit():
    """The executed basis spans paper exponentials and has deterministic orthonormal moments."""
    for degree in range(5):
        space = OscillatoryFaceSpace((0.0, 0.3, 1.0), (degree, degree), False, 4.2)
        x, w = space.quadrature(32)
        basis = space.evaluate(x)
        np.testing.assert_allclose(basis @ space.constant_coefficients(), 1.0, atol=2e-13)
        for segment in range(2):
            block = slice(segment * (degree + 1), (segment + 1) * (degree + 1))
            np.testing.assert_allclose(
                basis[:, block].T @ (w[:, None] * basis[:, block]),
                np.eye(degree + 1) * np.diff(space.breaks)[segment],
                atol=2e-12,
            )
    space = OscillatoryFaceSpace((0.0, 1.0), (2,), False, 2.0)
    t, w = space.quadrature(20)
    basis = space.evaluate(t)
    exact = np.exp(1j * np.sqrt(2) * t)
    coefficients = np.linalg.solve(basis.T @ (w[:, None] * basis), basis.T @ (w * exact))
    np.testing.assert_allclose(basis @ coefficients, exact, atol=1e-13)
    assert helmholtz_skeleton(CartesianMacroMesh(1), 2, degree=2, oscillatory=True).components == 2
    with pytest.raises(ValueError, match="unresolved"):
        OscillatoryFaceSpace((0.0, 1.0), (8,), False, 1e-5)


def test_spawn_and_sampling_contracts():
    """Portable worker assembly preserves the complex coefficient representation."""
    mesh = TriangleMesh.unit_square(1)
    serial = solve_helmholtz(mesh, omega=2, source=force, absorbing=impedance, degree=2)
    parallel = solve_helmholtz(
        mesh, omega=2, source=force, absorbing=impedance, degree=2, backend="process", workers=2
    )
    np.testing.assert_array_equal(serial.trace, parallel.trace)
    np.testing.assert_allclose(serial.l2_error(0), parallel.l2_error(lambda p: np.zeros(len(p))))
    np.testing.assert_allclose(
        serial.gradient_l2_error([0, 0]), parallel.gradient_l2_error(lambda p: np.zeros_like(p))
    )
    for reference in ([[1, 0]], [[-1, 1, 1]], [[0.2, 0.2, 0.2]], [[1j, 0, 1]]):
        with pytest.raises(ValueError):
            serial.sample(0, reference)
    with pytest.raises(ValueError):
        serial.sample(3, [[1, 0, 0]])
    with pytest.raises(ValueError):
        serial.l2_error(np.nan)
    with pytest.raises(ValueError):
        serial.gradient_l2_error([np.nan, 0])


@pytest.mark.parametrize(
    "options",
    [
        {"omega": 0},
        {"omega": 1j},
        {"density": -1},
        {"bulk_modulus": 1j},
        {"source": [1, 2]},
        {"absorbing": {999: 0}},
        {"neumann": 0},
        {"skeleton": SkeletonSpace(TriangleMesh.unit_square(1))},
    ],
)
def test_invalid_contracts(options):
    """Invalid data, overlaps and real/imaginary layouts are rejected explicitly."""
    arguments = {"omega": 2} | options
    with pytest.raises(ValueError):
        solve_helmholtz(TriangleMesh.unit_square(1), **arguments)
    with pytest.raises(TypeError):
        solve_helmholtz(None, omega=2)
    with pytest.raises(ValueError):
        helmholtz_skeleton(TriangleMesh.unit_square(1), 0)
    with pytest.raises(ValueError):
        OscillatoryFaceSpace((0.0, 1.0), (1,), True)
    with pytest.raises(ValueError):
        OscillatoryFaceSpace().evaluate([-1.0])
