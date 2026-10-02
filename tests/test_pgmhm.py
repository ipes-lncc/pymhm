"""Variational and physical invariants for residual-enriched Petrov-Galerkin MHM."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.darcy import solve_darcy
from pymhm.elements import boundary_data
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.pgmhm import _trace_matrix, solve_pgmhm
from pymhm.polygon import PolygonMesh
from pymhm.reservoir import CartesianCellField


@pytest.fixture(autouse=True)
def one_thread():
    """Keep small variational checks independent of native thread oversubscription."""
    with threadpool_limits(1):
        yield


def quadratic(points):
    """Nonharmonic quadratic with nonzero Dirichlet boundary trace."""
    x, y = points.T
    return 1 + x**2 + x * y + 2 * y**2


def quadratic_flux(points):
    """Exact anisotropic physical flux for the quadratic patch."""
    x, y = points.T
    return -np.column_stack((2 * x + y, x + 4 * y)) @ np.array([[3, 0.4], [0.4, 2]])


@pytest.mark.parametrize("polygonal", [False, True])
def test_nonzero_dirichlet_anisotropic_patch_and_enriched_flux_orientation(polygonal):
    if polygonal:
        points = np.array([[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1], [1, 1.0]])
        mesh = PolygonMesh(points, (np.array([0, 1, 2, 3, 4, 5]), np.array([3, 2, 6, 4])))
    else:
        mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    result = solve_pgmhm(
        mesh,
        skeleton=skeleton,
        degree=3,
        local_refinement=2,
        permeability=np.array([[3, 0.4], [0.4, 2]]),
        source=-14.8,
        dirichlet=quadratic,
        stabilization_parameter=0.1,
    )
    for enriched in (False, True):
        assert result.l2_error(quadratic, enriched=enriched) < 4e-12
        assert result.flux_l2_error(quadratic_flux, enriched=enriched) < 6e-11
    assert max(abs(result.conservation_residuals())) < 5e-12
    for cell, faces in enumerate(mesh.cell_faces):
        for side, face in enumerate(faces):
            parameter = np.linspace(0, 1, 7)
            a, b = mesh.points[mesh.faces[face]]
            expected = quadratic_flux(a + parameter[:, None] * (b - a)) @ mesh.normals[face]
            expected *= mesh.signs[cell, side]
            assert_allclose(result.normal_flux(cell, int(face), parameter), expected, atol=5e-11)


def test_only_enriched_multiplier_conserves_and_enriched_pressure_matches_weak_trace():
    mesh = TriangleMesh.unit_square(2)
    result = solve_pgmhm(mesh, source=1, stabilization_parameter=0.1)
    assert max(abs(result.conservation_residuals(enriched=False))) > 1e-6
    assert max(abs(result.conservation_residuals())) < 3e-14
    moments = np.zeros(result.skeleton.size)
    for response, base, enriched in zip(
        result.system.responses, result.pressure, result.enriched_pressure, strict=True
    ):
        problem = response.problem
        assert_allclose(problem.constraints[:, 0] @ (enriched - base), 0, atol=1e-17)
        np.add.at(moments, problem.trace_dofs, problem.coupling.T @ enriched)
    assert_allclose(moments, 0, atol=2e-15)
    for face in np.flatnonzero(mesh.face_cells[:, 1] >= 0):
        left, right = mesh.face_cells[face]
        parameter = np.linspace(0, 1, 7)
        assert_allclose(
            result.normal_flux(left, int(face), parameter),
            -result.normal_flux(right, int(face), parameter),
            atol=2e-15,
        )


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(float).eps,
    reason="extended accumulation requires a wider long-double type",
)
def test_extended_local_precision_preserves_laminated_physical_patch():
    """Both response and enrichment retain wide coefficients on a high-contrast laminate."""
    mesh = TriangleMesh.unit_square(2)
    material = CartesianCellField(np.array([[1e-7], [1.0]]), (0.5, 1.0))
    normals = mesh.normals
    neumann = {int(face): 0.0 for face in mesh.boundary_faces if abs(normals[face, 0]) > 0.5}

    def pressure(points):
        """A continuous affine pressure satisfies the tangential permeability jump."""
        return 1 - points[:, 1]

    def flux(points):
        """Evaluate physical Darcy flux without differentiating the discrete approximation."""
        return np.column_stack((np.zeros(len(points)), np.where(points[:, 0] < 0.5, 1e-7, 1)))

    result = solve_pgmhm(
        mesh,
        stabilization_parameter=0.1,
        permeability=material,
        dirichlet=pressure,
        neumann=neumann,
        local_refinement_precision="extended",
    )
    for enriched in (False, True):
        assert result.l2_error(pressure, enriched=enriched) < 2e-12
        assert result.flux_l2_error(flux, enriched=enriched) < 2e-12
    assert all(field.dtype == np.longdouble for field in result.pressure)
    assert all(field.dtype == np.longdouble for field in result.enriched_pressure)
    assert max(abs(result.conservation_residuals())) < 2e-12


def test_local_precision_is_an_explicit_validated_choice():
    """Invalid precision requests fail before any local problem is assembled."""
    with pytest.raises(ValueError, match="local_refinement_precision"):
        solve_pgmhm(
            TriangleMesh.unit_square(),
            stabilization_parameter=0.1,
            local_refinement_precision="automatic",
        )


def test_nonmatching_fine_traces_and_nonpolynomial_boundary_moments():
    """Check original equations and physical balance at their separate floating-point scales."""
    mesh = TriangleMesh.unit_square()

    def source(x):
        """Return a nonpolynomial volume source."""
        return np.sin(x[:, 0] + 2 * x[:, 1])

    def boundary(x):
        """Return a nonpolynomial prescribed pressure."""
        return np.exp(x[:, 0] + x[:, 1])

    result = solve_pgmhm(
        mesh,
        source=source,
        dirichlet=boundary,
        stabilization_parameter=0.1,
        local_meshes=(mesh.submesh(0, 2), mesh.submesh(1, 3)),
        quadrature_order=10,
    )
    moments = np.zeros(result.skeleton.size)
    for response, field in zip(result.system.responses, result.enriched_pressure, strict=True):
        np.add.at(moments, response.problem.trace_dofs, response.problem.coupling.T @ field)
    expected = boundary_data(result.skeleton, boundary, order=10)[0]
    assert_allclose(moments, expected, atol=8e-15)
    balances = result.conservation_residuals()
    for response, field, balance in zip(
        result.system.responses, result.pressure, balances, strict=True
    ):
        matrix = response.problem.matrix
        # The represented stiffness need not annihilate constants exactly. Keep
        # its action in the original local equation rather than zeroing balance.
        action = matrix.toarray().astype(np.longdouble) @ field.astype(np.longdouble)
        assert abs(balance + np.sum(action)) < 1e-14
        # Gamma bounds rounding of a sparse dot product in physical load units.
        # It is independent of the corrected condensation implementation.
        terms = int(matrix.getnnz(axis=0).max()) + 1
        rounding = terms * np.finfo(float).eps
        gamma = rounding / (1 - rounding)
        physical_scale = np.sum(abs(matrix) @ abs(field))
        assert abs(balance) <= gamma * physical_scale + 1e-14


@pytest.mark.parametrize("pure", [False, True])
def test_variable_coefficient_mixed_and_pure_neumann_physical_gauge(pure):
    mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    boundary = {}
    for face in mesh.boundary_faces:
        normal = mesh.normals[face].copy()
        if pure or abs(normal[1]) > 0.5:
            boundary[int(face)] = lambda x, n=normal: -(2 + x[:, 0]) * (n[0] + 2 * n[1])

    def exact(x):
        """Return the affine pressure with mean 0.7."""
        return x[:, 0] + 2 * x[:, 1] - 0.8

    result = solve_pgmhm(
        mesh,
        permeability=lambda x: 2 + x[:, 0],
        ellipticity_lower_bound=2,
        source=-1,
        dirichlet=exact,
        neumann=boundary,
        degree=3,
        skeleton=skeleton,
        stabilization_parameter=0.1,
        mean_pressure=0.7 if pure else 0,
    )
    assert result.l2_error(exact, enriched=True) < 2e-12
    assert max(abs(result.conservation_residuals())) < 2e-14
    for face in boundary:
        cell = int(mesh.face_cells[face, 0])
        t = np.linspace(0, 1, 5)
        start, end = mesh.points[mesh.faces[face]]
        assert_allclose(
            result.normal_flux(cell, face, t),
            boundary[face](start + t[:, None] * (end - start)),
            atol=1e-13,
        )


def test_permeability_source_scaling_and_same_space_mhm_limit():
    mesh = TriangleMesh.unit_square(2)
    a = solve_pgmhm(mesh, source=1, stabilization_parameter=0.1)
    b = solve_pgmhm(mesh, permeability=1e-4, source=1e-4, stabilization_parameter=0.1)
    assert_allclose(np.concatenate(a.pressure), np.concatenate(b.pressure), atol=5e-15)
    assert_allclose(
        np.concatenate(a.enriched_pressure), np.concatenate(b.enriched_pressure), atol=5e-15
    )
    assert_allclose(a.hybrid.trace * 1e-4, b.hybrid.trace, atol=5e-18)
    reference = solve_darcy(mesh, source=1, degree=2, local_refinement=1)
    errors = []
    for alpha in (0.02, 0.01, 0.005):
        result = solve_pgmhm(mesh, source=1, stabilization_parameter=alpha)
        errors.append(
            np.linalg.norm(np.concatenate(result.pressure) - np.concatenate(reference.pressure))
        )
    assert 1.98 < errors[0] / errors[1] < 2.02
    assert 1.98 < errors[1] / errors[2] < 2.02


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_parallel_original_local_assembly(backend):
    mesh = TriangleMesh.unit_square()
    result = solve_pgmhm(mesh, source=1, stabilization_parameter=0.1, backend=backend, workers=2)
    serial = solve_pgmhm(mesh, source=1, stabilization_parameter=0.1)
    assert_allclose(result.hybrid.trace, serial.hybrid.trace, atol=1e-15)
    assert_allclose(
        np.concatenate(result.enriched_pressure),
        np.concatenate(serial.enriched_pressure),
        atol=1e-15,
    )


def test_invalid_parameters_and_incompatible_neumann_data():
    mesh = TriangleMesh.unit_square()
    for parameter in (0, -1, np.nan, 1j, [0.1]):
        with pytest.raises(ValueError, match="stabilization_parameter"):
            solve_pgmhm(mesh, stabilization_parameter=parameter)
    with pytest.raises(TypeError, match="macrocells"):
        solve_pgmhm(None, stabilization_parameter=0.1)
    with pytest.raises(ValueError, match="scalar skeleton"):
        solve_pgmhm(
            mesh, stabilization_parameter=0.1, skeleton=SkeletonSpace(TriangleMesh.unit_square())
        )
    rich = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
    with pytest.raises(ValueError, match="local degree"):
        solve_pgmhm(mesh, stabilization_parameter=0.1, skeleton=rich, degree=2)
    for mean in (np.inf, 1):
        with pytest.raises(ValueError, match="mean_pressure"):
            solve_pgmhm(mesh, stabilization_parameter=0.1, mean_pressure=mean)
    with pytest.raises(ValueError, match="one local mesh"):
        solve_pgmhm(mesh, stabilization_parameter=0.1, local_meshes=())
    with pytest.raises(ValueError, match="sampled"):
        solve_pgmhm(
            mesh,
            stabilization_parameter=0.1,
            permeability=lambda x: np.ones(len(x)),
            ellipticity_lower_bound=2,
        )
    with pytest.raises(ValueError, match="incompatible"):
        solve_pgmhm(
            mesh,
            source=1e-12,
            stabilization_parameter=0.1,
            neumann={int(f): 0 for f in mesh.boundary_faces},
        )
    result = solve_pgmhm(TriangleMesh.unit_square(2), stabilization_parameter=0.1)
    with pytest.raises(ValueError, match="incident"):
        result.normal_flux(0, len(result.skeleton.mesh.faces) - 1, [0.5])
    face = int(result.skeleton.mesh.cell_faces[0, 0])
    for parameter in ([[0.5]], [-0.1], [1.1], [np.nan]):
        with pytest.raises(ValueError, match="parameters"):
            result.normal_flux(0, face, parameter)
    result.normal_flux(0, face, [0.5], enriched=False)
    with pytest.raises(ValueError, match="does not cover"):
        _trace_matrix(mesh, mesh.submesh(0, 1), 0, 2, np.array([2.0]))
