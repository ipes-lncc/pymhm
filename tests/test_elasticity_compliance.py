"""Anisotropic mixed stress, physical hydrostatic identity and material contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.elasticity_compliance import stress_compliance_values
from pymhm.elasticity_mixed import solve_elasticity_mixed
from pymhm.elasticity_tensor_rt import solve_elasticity_tensor_rt
from pymhm.mesh import TriangleMesh
from pymhm.polygon import PolygonMesh, solve_elasticity_mixed_polygons
from pymhm.quadrilateral import CartesianMacroMesh


def anisotropic_compliance():
    """Independently extend an anisotropic symmetric-strain material to skew tensors."""
    kelvin = np.array([[6.0, 2, 0.7], [2, 4, -0.4], [0.7, -0.4, 3]])
    basis = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1], [0, 1, 0, 0]], float)
    basis[:, 2:] /= np.sqrt(2)
    modal = np.zeros((4, 4))
    modal[:3, :3] = np.linalg.inv(kelvin)
    modal[3, 3] = 0.5
    return basis @ modal @ basis.T


def displacement(points):
    """Affine displacement with nonzero dilation, shear and rigid rotation."""
    return points @ np.array([[1.0, -1], [2, 3]])


@pytest.mark.parametrize("triangular", [True, False])
@pytest.mark.parametrize("boundary", ["displacement", "mixed", "traction"])
def test_anisotropic_patch_with_physical_boundary_conditions(triangular, boundary):
    """Stress, displacement and rotation match an independently inverted material law."""
    mesh = TriangleMesh.unit_square(2) if triangular else CartesianMacroMesh(2)
    solve = solve_elasticity_mixed if triangular else solve_elasticity_tensor_rt
    material = anisotropic_compliance()
    strain = np.array([[1, 0.5], [0.5, 3]])
    sigma = np.linalg.solve(material, strain.ravel()).reshape(2, 2)
    options = dict(compliance=material, local_refinement=1, dirichlet=displacement)
    if boundary != "displacement":
        faces = mesh.boundary_faces[:1] if boundary == "mixed" else mesh.boundary_faces
        options["traction"] = {int(f): sigma @ mesh.normals[f] for f in faces}
        options["rigid_moments"] = (1.5, 1.0, -0.25)
    solution = solve(mesh, **options)
    if triangular:
        errors = [solution.l2_error(displacement), solution.stress_l2_error(sigma)]
        errors.append(solution.rotation_l2_error(1.5))
    else:
        errors = list(solution.errors(displacement, sigma, (0, 0), 1.5).values())
    assert max(errors) < 2e-10
    assert np.max(np.abs(solution.equilibrium_residuals())) < 2e-11


@pytest.mark.parametrize("triangular", [True, False])
def test_variable_anisotropic_material_and_force(triangular):
    """A smooth compliance gives linear stress with exactly differentiated body force."""
    mesh = TriangleMesh.unit_square() if triangular else CartesianMacroMesh()
    solve = solve_elasticity_mixed if triangular else solve_elasticity_tensor_rt
    material = anisotropic_compliance()
    sigma = np.linalg.solve(material, np.array([1, 0.5, 0.5, 3])).reshape(2, 2)
    solution = solve(
        mesh,
        compliance=lambda x: material / (1 + x[:, 0, None, None]),
        source=-sigma[:, 0],
        dirichlet=displacement,
        quadrature_order=12,
    )

    def exact(x):
        """Evaluate the independently prescribed linear stress."""
        return (1 + x[:, 0, None, None]) * sigma

    if triangular:
        assert solution.stress_l2_error(exact, 12) < 2e-10
        assert solution.l2_error(displacement) < 2e-10
    else:
        assert max(solution.errors(displacement, exact, sigma[:, 0], 1.5, 12).values()) < 2e-10


@pytest.mark.parametrize("triangular", [True, False])
def test_isotropic_tensor_agrees_with_lame_formulation(triangular):
    """The full-tensor option preserves the existing isotropic mixed method."""
    mesh = TriangleMesh.unit_square() if triangular else CartesianMacroMesh()
    solve = solve_elasticity_mixed if triangular else solve_elasticity_tensor_rt
    identity = np.array([1.0, 0, 0, 1])
    material = 0.5 * np.eye(4) - np.outer(identity, identity) / 8
    a = solve(mesh, dirichlet=displacement)
    b = solve(mesh, dirichlet=displacement, compliance=material)
    for left, right in zip(a.hybrid.fields, b.hybrid.fields, strict=True):
        assert_allclose(left, right, atol=2e-11, rtol=0)
    with pytest.raises(ValueError, match="incompressible"):
        solve(mesh, compliance=material, mean_pressure=1)


def test_tensor_coordinates_and_callback_preserve_full_operator():
    """Fourth-order Cartesian and flattened forms implement the same contraction."""
    matrix = anisotropic_compliance()
    points = np.array([[0.1, 0.2], [0.6, 0.9]])
    expected = np.broadcast_to(matrix, (2, 4, 4))
    assert_allclose(stress_compliance_values(matrix.reshape(2, 2, 2, 2), points), expected)
    assert_allclose(stress_compliance_values(lambda x: expected, points), expected)


@pytest.mark.parametrize(
    "matrix,message",
    [
        (np.eye(4) * 1j, "real"),
        (np.eye(3), "shape"),
        (np.full((4, 4), np.nan), "finite"),
        (np.eye(4) + np.triu(np.ones((4, 4)), 1), "self-adjoint"),
        (np.diag([1, 2, 3, 4]), "preserve"),
        (np.diag([1, 1, 1, 0]), "positive definite"),
    ],
)
def test_invalid_compliance_contract(matrix, message):
    """Reject operators incompatible with a real coercive weak-symmetry method."""
    with pytest.raises(ValueError, match=message):
        stress_compliance_values(matrix, np.zeros((1, 2)))


@pytest.mark.parametrize("pure_traction", [False, True])
def test_polygonal_mixed_elasticity_keeps_original_edges_and_rigid_gauges(pure_traction):
    """A nonconvex L and a square recover anisotropic fields and physical rigid moments."""
    mesh = PolygonMesh(
        np.array(((0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1), (1, 1))),
        (np.array((0, 1, 2, 3, 4, 5)), np.array((3, 2, 6, 4))),
    )
    material = anisotropic_compliance()
    sigma = np.linalg.solve(material, np.array([1, 0.5, 0.5, 3])).reshape(2, 2)
    traction = (
        {int(f): sigma @ mesh.normals[f] for f in mesh.boundary_faces} if pure_traction else None
    )
    result = solve_elasticity_mixed_polygons(
        mesh,
        compliance=material,
        dirichlet=displacement,
        traction=traction,
        rigid_moments=(1.5, 1.0, -0.25),
    )
    assert result.skeleton.mesh is mesh
    assert result.l2_error(displacement) < 2e-10
    assert result.stress_l2_error(sigma) < 2e-10
    assert np.max(np.abs(result.equilibrium_residuals())) < 2e-11
