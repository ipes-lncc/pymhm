"""Canonical RT3D duality, represented traces and continuous-test equilibrium."""

import hashlib
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from examples import reconstruction3d_resolution
from pymhm._legacy.models.darcy.primal_3d import solve_darcy_3d
from pymhm.fem.hdiv.family_3d import (
    cell_quadrature,
    face_polynomials,
    face_quadrature,
    face_shape,
    reference_faces,
    reference_vertices,
)
from pymhm.fem.hdiv.rt_3d import RTTetraFamily, rt3d_interior_tests
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.io.workspace import source_file
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.recovery.moments_3d import reconstruct_darcy_moments_3d


@pytest.fixture(autouse=True)
def one_thread():
    """Avoid nested native threads in small reference moment matrices."""
    with threadpool_limits(1):
        yield


def test_resolution_campaign_creates_fresh_output_directory(tmp_path, monkeypatch):
    """The first published state captures its literal basis in a new output directory.

    Geometry, Gaussian data, quadrature and residual criteria are unchanged.
    The acquisition stops before its second solve; this tests fresh capture IO,
    while the notebook separately acquires the complete four-state study.
    """
    original_geometry = source_file("examples/data/reconstruction3d-macro.json")
    data = json.loads(original_geometry.read_text())
    mesh = TetraMesh(np.asarray(data["macro_points"]), np.asarray(data["macro_cells"]))
    geometry = tmp_path / "examples/data/reconstruction3d-macro.json"
    geometry.parent.mkdir(parents=True)
    geometry.write_bytes(original_geometry.read_bytes())
    output = tmp_path / "examples/results/reconstruction3d"
    assert not output.exists()
    monkeypatch.setattr(reconstruction3d_resolution, "ROOT", tmp_path)
    monkeypatch.setattr(reconstruction3d_resolution, "OUTPUT", output)
    solve = reconstruction3d_resolution.solve_darcy_3d
    calls = 0

    class FirstCaptureComplete(Exception):
        """Stop this IO regression before repeating the remaining published solves."""

    def first_solve(*args, **kwargs):
        """Execute the first actual solve and stop before constructing the second case."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FirstCaptureComplete
        return solve(*args, **kwargs)

    monkeypatch.setattr(reconstruction3d_resolution, "solve_darcy_3d", first_solve)
    with pytest.raises(FirstCaptureComplete):
        reconstruction3d_resolution.run()
    record = json.loads((output / "resolution.json").read_text())
    assert "source_changed_during_run" not in record  # The four-state acquisition is incomplete.
    owner = source_file("examples/reconstruction3d_resolution.py")
    assert record["source_sha256"]["examples/reconstruction3d_resolution.py"] == (
        hashlib.sha256(owner.read_bytes()).hexdigest()
    )
    assert len(record["rows"]) == 1
    for row in record["rows"]:
        assert row["macro_cells"] == 162
        assert row["local_refinement"] == 2
        assert row["error_quadrature_change"] <= 2e-9
        assert row["macro_balance"] <= 1e-9
        archive_path = output / row["archive"]
        assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == row["archive_sha256"]
        with np.load(archive_path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["macro_points"], mesh.points)
            np.testing.assert_array_equal(archive["macro_cells"], mesh.cells)
            assert int(archive["local_degree"]) == row["local_degree"]
            assert int(archive["reconstruction_degree"]) == row["reconstruction_degree"]
            current = RTTetraFamily(row["reconstruction_degree"])
            np.testing.assert_array_equal(archive["rt_basis"], current.coefficients)


@pytest.mark.parametrize("degree", range(4))
def test_rt_moment_duality_and_divergence(degree):
    """Independent integration recovers every RT face/volume degree of freedom."""
    family = RTTetraFamily(degree)
    vertices = reference_vertices("tetrahedron")
    rows = []
    for face in reference_faces("tetrahedron"):
        nodes = vertices[list(face)]
        uv, w = face_quadrature(3, degree + 4)
        normal = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
        normal *= np.sign(normal @ (nodes.mean(0) - vertices.mean(0)))
        value = family.tabulate(face_shape(uv, 3) @ nodes)[0] @ normal
        rows.append(face_polynomials(uv, 3, degree).T @ (w[:, None] * value))
    points, w = cell_quadrature("tetrahedron", degree + 4)
    value, div, pressure = family.tabulate(points)
    rows.append(np.einsum("q,qia,qja->ij", w, rt3d_interior_tests(points, degree), value))
    assert_allclose(np.vstack(rows), np.eye(family.local_size), atol=8e-12)
    assert np.linalg.matrix_rank(pressure.T @ (w[:, None] * div)) == family.pressure_size
    assert_allclose(pressure @ np.linalg.lstsq(pressure, div, rcond=None)[0], div, atol=2e-9)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_affine_flux_and_archived_rt_coordinates(degree):
    """Affine pressure has exact reconstructed flux at every supported test order."""
    mesh = TetraMesh.unit_cube()
    solution = solve_darcy_3d(mesh, degree=max(1, degree), dirichlet=lambda x: x.sum(axis=1))
    result = reconstruct_darcy_moments_3d(solution, degree=degree)
    assert result.flux_l2_error([-1, -1, -1]) < 3e-11
    assert max(np.max(abs(r)) for r in result.normal_flux_residuals()) == 0
    assert max(np.max(abs(r)) for r in result.continuous_moment_residuals()) < 1e-12
    assert max(np.max(abs(r)) for r in result.fine_conservation_residuals()) < 1e-12
    points = np.array([[0.2, 0.1, 0.3]])
    current = result.family.tabulate(points)
    replayed = result.family.tabulate(points, coefficients=result.family.coefficients.copy())
    for first, second in zip(current, replayed, strict=True):
        assert_allclose(first, second, atol=0, rtol=0)


def test_quadratic_trace_and_cubic_pressure():
    """A genuinely quadratic normal flux needs the complete P2 skeletal density."""
    mesh = TetraMesh.unit_cube()
    skeleton = TriangularSkeleton(mesh, degree=2)

    def pressure(x):
        """Cubic polynomial potential."""
        return x[:, 0] ** 3 + x[:, 1] ** 2 * x[:, 2]

    def source(x):
        """Negative Laplacian of the prescribed potential."""
        return -6 * x[:, 0] - 2 * x[:, 2]

    def flux(x):
        """Exact physical negative gradient."""
        return np.column_stack((-3 * x[:, 0] ** 2, -2 * x[:, 1] * x[:, 2], -(x[:, 1] ** 2)))

    solution = solve_darcy_3d(mesh, skeleton=skeleton, degree=3, dirichlet=pressure, source=source)
    result = reconstruct_darcy_moments_3d(solution, degree=2)
    assert solution.l2_error(pressure) < 2e-12
    assert result.flux_l2_error(flux) < 3e-11
    assert max(np.max(abs(r)) for r in result.continuous_moment_residuals()) < 1e-12


def test_continuous_equilibrium_is_distinct_from_fine_cell_equilibrium():
    """Canonical averaging satisfies continuous P2 tests without imposing fine DG0 balance."""
    mesh = TetraMesh.unit_cube()
    solution = solve_darcy_3d(
        mesh, degree=2, source=lambda x: np.sin(x[:, 0] + 2 * x[:, 1]), quadrature_order=7
    )
    result = reconstruct_darcy_moments_3d(solution, degree=2, quadrature_order=7)
    assert max(np.max(abs(r)) for r in result.continuous_moment_residuals()) < 2e-12
    assert max(np.max(abs(r)) for r in result.fine_conservation_residuals()) > 1e-5


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_parallel_moments_match_serial(backend):
    """Independent workers preserve canonical coordinates and one-sided fields."""
    solution = solve_darcy_3d(TetraMesh.unit_cube(), source=1)
    serial = reconstruct_darcy_moments_3d(solution, degree=1)
    parallel = reconstruct_darcy_moments_3d(solution, degree=1, backend=backend, workers=2)
    assert_allclose(serial.flux, parallel.flux, atol=2e-14)


def test_invalid_rt_and_skeletal_contracts():
    """Reject nonreal geometry, malformed archives, underintegration and incompatible spaces."""
    family = RTTetraFamily(1)
    for points in ([1, 2], [[1j, 0, 0]], [[np.nan, 0, 0]]):
        with pytest.raises(ValueError, match="triples"):
            family.tabulate(points)
    with pytest.raises(ValueError, match="archived"):
        family.tabulate([[0, 0, 0]], coefficients=np.ones((2, 3)))
    solution = solve_darcy_3d(TetraMesh.unit_cube(), degree=1)
    with pytest.raises(ValueError, match="test space"):
        reconstruct_darcy_moments_3d(solution, degree=2)
    with pytest.raises(ValueError, match="quadrature"):
        reconstruct_darcy_moments_3d(solution, quadrature_order=1)
    with pytest.raises(ValueError, match="triples"):
        solution.skeleton.basis(0, [[1j, 0, 0]])


def test_unaligned_or_nonincident_boundary_is_rejected():
    """Reject jumps inside a fine face and boundary faces outside their macrocell."""
    from pymhm.meshes.mixed import AffineMixedMesh
    from pymhm.recovery.moments_3d import _boundary_moments

    mesh = TetraMesh.unit_cube()
    fine = mesh.submesh(0, 1)
    family = RTTetraFamily(0)
    skeleton = TriangularSkeleton(mesh, subdivisions=2)
    with pytest.raises(ValueError, match="aligned"):
        _boundary_moments(
            skeleton,
            0,
            AffineMixedMesh(fine.points, fine.cells),
            family,
            np.zeros(skeleton.size),
            3,
        )
    shifted = AffineMixedMesh(fine.points + [2, 3, 4], fine.cells)
    with pytest.raises(ValueError, match="containing macroface"):
        _boundary_moments(skeleton, 0, shifted, family, np.zeros(skeleton.size), 3)
