"""Physical tetrahedral Robin equations and continuous pressure-skeleton geometry."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.darcy3d import TriangularSkeleton
from pymhm.mesh import TriangleMesh
from pymhm.mh3d import solve_mh_3d
from pymhm.mh_trace3d import PressureTraceSpace3D, boundary_rules, broken_face_basis
from pymhm.tetrahedral import TetraMesh


@pytest.fixture(autouse=True)
def single_thread():
    """Keep small matrix checks independent of threaded BLAS scheduling."""
    with threadpool_limits(1):
        yield


@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "neumann"])
def test_anisotropic_quadratic_all_boundary_types(boundary):
    """Check volume fields, physical flux, the full operator and the pressure mean."""
    mesh = TetraMesh.unit_cube()
    tensor = np.array([[3.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.0]])

    def exact(x):
        """Return the nonharmonic pressure with unit-cube mean two."""
        return 1 + np.sum(x * x, axis=1)

    def flux(x):
        """Differentiate the exact pressure with the full tensor."""
        return -2 * x @ tensor

    faces = (
        mesh.boundary_faces
        if boundary == "neumann"
        else (mesh.boundary_faces[:2] if boundary == "mixed" else [])
    )
    natural = {int(f): lambda x, n=mesh.normals[f]: flux(x) @ n for f in faces}
    skeleton = TriangularSkeleton(mesh, degree=2)
    result = solve_mh_3d(
        mesh,
        permeability=tensor,
        source=-12,
        dirichlet=exact,
        neumann=natural,
        mean_pressure=2 if boundary == "neumann" else 0,
        skeleton=skeleton,
    )
    assert result.l2_error(exact, 4) < 2e-11
    assert result.flux_l2_error(flux, 4) < 2e-11
    values, numerical_flux = result.evaluate(0, np.array([[0.25] * 4]))
    fine = result.local_meshes[0]
    centers = fine.points[fine.cells].mean(axis=1)
    assert_allclose(values[:, 0], exact(centers), atol=2e-11)
    assert_allclose(numerical_flux[:, 0], flux(centers), atol=2e-11)
    assert_allclose(result.conservation_residuals(), 0, atol=3e-12)
    assert_allclose(
        result.global_matrix @ result.global_coefficients, result.global_rhs, atol=3e-12
    )
    for cell in range(len(mesh.cells)):
        total = sum(result.normal_flux_moments(cell, int(f)).sum() for f in mesh.cell_faces[cell])
        assert_allclose(total, -12 * mesh.volumes[cell], atol=2e-12)
    with pytest.raises(ValueError, match="incident"):
        result.normal_flux_moments(
            0, next(f for f in range(len(mesh.faces)) if f not in mesh.cell_faces[0])
        )


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("subdivisions", [1, 2])
def test_pressure_surface_polynomials_and_shared_edge_identities(degree, subdivisions):
    """All face traces agree on macroedges through topological node identity."""
    mesh = TetraMesh.unit_cube()
    gamma = PressureTraceSpace3D(mesh, degree, subdivisions)
    assert gamma.size == len(np.unique(gamma.nodes, axis=0))
    bary = np.array([[0.2, 0.3, 0.5], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    for face in range(len(mesh.faces)):
        data = 1 + np.sum(gamma.nodes[gamma.face_dofs[face]], axis=1) ** degree
        expected = 1 + np.sum(bary @ mesh.points[mesh.faces[face]], axis=1) ** degree
        assert_allclose(gamma.evaluate(face, bary) @ data, expected, atol=1e-13)
    assert len(gamma.cell_dofs(0)) > 3


def test_physical_boundary_rules_and_partition_contracts():
    """Fine-facet quadrature preserves physical area and polynomial mass."""
    mesh = TetraMesh.unit_cube()
    fine = mesh.submesh(0, 2)
    rules = boundary_rules(mesh, 0, fine, 2, 4)
    assert_allclose(
        sum(weights.sum() for _, _, _, _, weights, _ in rules),
        mesh.areas[mesh.cell_faces[0]].sum(),
        atol=2e-15,
    )
    skeleton = TriangularSkeleton(mesh, 2, degree=1)
    for face, _, basis, points, _, bary in rules:
        assert_allclose(basis.sum(axis=1), 1, atol=2e-15)
        assert_allclose(bary @ mesh.points[mesh.faces[face]], points, atol=2e-15)
        assert_allclose(broken_face_basis(skeleton, face, bary).sum(axis=1), 1, atol=2e-15)
    with pytest.raises(TypeError, match="tetrahedral"):
        PressureTraceSpace3D(TriangleMesh.unit_square())
    gamma = PressureTraceSpace3D(mesh)
    for bad in (np.ones((1, 2)), [[1j, 0, 1]], [[np.nan, 0, 1]], [[-0.1, 0, 1.1]], [[0, 0, 0]]):
        with pytest.raises(ValueError, match="barycentric"):
            gamma.evaluate(0, np.asarray(bad))
    with pytest.raises(ValueError, match="conormal"):
        broken_face_basis(skeleton, 0, np.array([[-1.0, -1, 3]]))
    with pytest.raises(ValueError, match="partition"):
        PressureTraceSpace3D(mesh, subdivisions=2).evaluate(
            0, np.array([[-7.5e-13, 0, 1 + 7.5e-13]])
        )


def test_neumann_compatibility_small_scale_and_input_contracts():
    """Incompatible forcing and nonphysical gauges are rejected without regularization."""
    mesh = TetraMesh.unit_cube()
    natural = {int(f): 0.0 for f in mesh.boundary_faces}
    for source in (1.0, 1e-12):
        with pytest.raises(ValueError, match="incompatible"):
            solve_mh_3d(mesh, source=source, neumann=natural)
    result = solve_mh_3d(mesh, neumann=natural, mean_pressure=3)
    assert result.l2_error(3, 3) < 1e-11
    with pytest.raises(TypeError, match="tetrahedral"):
        solve_mh_3d(TriangleMesh.unit_square())
    invalid = [
        ({"skeleton": TriangularSkeleton(TetraMesh.unit_cube())}, "skeleton"),
        ({"skeleton": TriangularSkeleton(mesh, 4)}, "subdivisions"),
        ({"neumann": {-1: 0}}, "exterior"),
        ({"mean_pressure": np.nan}, "finite"),
        ({"mean_pressure": 1j}, "finite"),
        ({"mean_pressure": 1}, "pure Neumann"),
        ({"permeability": lambda x: 1 + x[:, 0]}, "certified"),
        ({"ellipticity_lower_bound": -1}, "positive"),
        ({"origin": [0, 0]}, "three-dimensional"),
        ({"robin_parameter": 0}, "coercivity"),
        ({"ellipticity_lower_bound": 2}, "sampled"),
    ]
    for options, message in invalid:
        with pytest.raises(ValueError, match=message):
            solve_mh_3d(mesh, **options)


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_picklable_local_robin_factory_preserves_physical_fields(backend):
    """Exercise the complete factory through both supported concurrent maps."""
    mesh = TetraMesh.unit_cube()
    serial = solve_mh_3d(mesh, source=1, dirichlet=0.5, degree=1)
    parallel = solve_mh_3d(mesh, source=1, dirichlet=0.5, degree=1, backend=backend, workers=2)
    assert_allclose(parallel.hybrid.trace, serial.hybrid.trace, atol=2e-13)
    for found, expected in zip(parallel.pressure, serial.pressure, strict=True):
        assert_allclose(found, expected, atol=2e-13)
