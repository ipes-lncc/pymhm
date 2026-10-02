"""Physical Neumann Robin elimination and polygonal three-field identities."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.lagrange import nodal_space
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.mh import solve_mh
from pymhm.mh2m import PressureTraceSpace, solve_mh2m
from pymhm.polygon import PolygonMesh
from pymhm.scalar_trace_integration import integrate_dirichlet_trace


@pytest.fixture(autouse=True)
def one_thread():
    """Avoid oversubscription in the small independent boundary checks."""
    with threadpool_limits(1):
        yield


def polygons():
    """Partition the unit square into one nonconvex L and one square."""
    points = np.array([[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1], [1, 1.0]])
    return PolygonMesh(points, (np.array([0, 1, 2, 3, 4, 5]), np.array([3, 2, 6, 4])))


def pressure(points):
    """Return a nonharmonic polynomial with physical unit-square mean 9/4."""
    x, y = points.T
    return 1 + x * x + x * y + 2 * y * y


def flux(points):
    """Differentiate pressure independently for the anisotropic material."""
    x, y = points.T
    return -np.column_stack((2 * x + y, x + 4 * y)) @ np.array([[3.0, 0.4], [0.4, 2.0]])


@pytest.mark.parametrize("geometry", ["triangular", "polygonal"])
@pytest.mark.parametrize("boundary", ["mixed", "neumann"])
def test_mh_quadratic_physical_flux_and_integral_mean(geometry, boundary):
    """A constant Robin normal coefficient may vanish without singular division."""
    mesh = TriangleMesh.unit_square() if geometry == "triangular" else polygons()
    faces = mesh.boundary_faces if boundary == "neumann" else mesh.boundary_faces[:2]
    natural = {int(f): lambda x, normal=mesh.normals[f]: flux(x) @ normal for f in faces}
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
    result = solve_mh(
        mesh,
        skeleton=skeleton,
        degree=2,
        local_refinement=2,
        permeability=np.array([[3.0, 0.4], [0.4, 2.0]]),
        source=-14.8,
        dirichlet=pressure if boundary == "mixed" else None,
        neumann=natural,
        mean_pressure=2.25 if boundary == "neumann" else 0,
        robin_parameter=0.25,
    )
    assert result.l2_error(pressure) < 3e-11
    assert result.flux_l2_error(flux) < 3e-11
    assert_allclose(result.conservation_residuals(), 0, atol=3e-12)
    assert_allclose(
        result.global_matrix @ result.global_coefficients, result.global_rhs, atol=4e-12
    )
    assert_allclose(result.global_matrix.toarray(), result.global_matrix.toarray().T, atol=1e-13)
    for face, coefficients in result.boundary_pressure.items():
        t = np.linspace(0, 1, 5)
        a, b = mesh.points[mesh.faces[face]]
        points = a + t[:, None] * (b - a)
        assert_allclose(
            skeleton.faces[face].evaluate(t) @ coefficients, pressure(points), atol=1e-10
        )
        cell = int(mesh.face_cells[face, 0])
        assert_allclose(
            result.normal_flux(cell, face, t), flux(points) @ mesh.normals[face], atol=1e-10
        )


@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "neumann"])
def test_mh2m_nonconvex_polygons_preserve_three_field_equations(boundary):
    """Polygon boundaries share Gamma vertices and retain independent Lambda moments."""
    mesh = polygons()
    faces = (
        mesh.boundary_faces
        if boundary == "neumann"
        else (mesh.boundary_faces[:2] if boundary == "mixed" else [])
    )
    natural = {int(f): lambda x, normal=mesh.normals[f]: flux(x) @ normal for f in faces}
    result = solve_mh2m(
        mesh,
        permeability=np.array([[3.0, 0.4], [0.4, 2.0]]),
        source=-14.8,
        dirichlet=pressure,
        neumann=natural,
        pressure_trace=PressureTraceSpace.uniform(mesh, 2),
        flux_space=SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces)),
        degree=2,
        local_refinement=4,
        mean_pressure=2.25 if boundary == "neumann" else 0,
    )
    assert result.l2_error(pressure) < 3e-12
    assert result.flux_l2_error(flux) < 3e-11
    assert_allclose(result.conservation_residuals(), 0, atol=1e-12)
    for p, data, equation, continuity in zip(
        result.pressure,
        result.local,
        result.local_equation_residuals(),
        result.trace_moment_residuals(),
        strict=True,
    ):
        assert_allclose(p, pressure(nodal_space(data.mesh, 2)[1]), atol=4e-12)
        assert_allclose(equation, 0, atol=6e-13)
        assert_allclose(continuity, 0, atol=6e-13)


@pytest.mark.parametrize("source", [1.0, 1e-12])
def test_mh_neumann_compatibility_uses_physical_scale(source):
    """An arbitrarily small unbalanced source cannot be hidden by the pressure gauge."""
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="incompatible Neumann"):
        solve_mh(mesh, source=source, neumann={int(f): 0 for f in mesh.boundary_faces})


def test_mh_gauge_and_cancellation_do_not_replace_physical_compatibility():
    """Constant shifts are represented while resolvable load imbalance is rejected."""
    mesh = TriangleMesh.unit_square()
    natural = {int(f): 0 for f in mesh.boundary_faces}
    result = solve_mh(mesh, neumann=natural, mean_pressure=3, dirichlet=None)
    assert result.l2_error(3) < 2e-12
    assert result.flux_l2_error((0, 0)) < 3e-12
    # Opposite physical boundary fluxes cancel; the source 2 does not.
    natural = {int(f): 1e10 * mesh.normals[f, 0] for f in mesh.boundary_faces}
    with pytest.raises(ValueError, match="incompatible Neumann"):
        solve_mh(mesh, source=2, neumann=natural)


def test_invalid_neumann_metadata_and_original_equation_gate(monkeypatch):
    """Reject invalid faces, irrelevant means and inaccurate global boundary solutions."""
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="exterior"):
        solve_mh(mesh, neumann={int(np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0]): 0})
    with pytest.raises(ValueError, match="pure Neumann"):
        solve_mh(mesh, mean_pressure=2)
    for mean in (np.nan, 1j, [1, 2]):
        with pytest.raises(ValueError, match="finite and real"):
            solve_mh(mesh, mean_pressure=mean)
    result = solve_mh(mesh)
    for faces in ((int(mesh.boundary_faces[0]),) * 2, (-1,)):
        with pytest.raises(ValueError, match="distinct exterior"):
            integrate_dirichlet_trace(result.skeleton, result.local_meshes, 0, 2, 4, faces=faces)
    monkeypatch.setattr(
        "pymhm.mh_boundary.solve_linear", lambda matrix, rhs, **kwargs: np.zeros_like(rhs)
    )
    with pytest.raises(ValueError, match="original boundary equations"):
        solve_mh(mesh, source=1, neumann={int(mesh.boundary_faces[0]): 0})
