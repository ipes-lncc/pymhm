"""Three-dimensional general stiffness, six rigid modes and physical elasticity gauges."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.darcy3d import TriangularSkeleton
from pymhm.elasticity3d import _KELVIN3, constitutive_values_3d, rigid_modes_3d, solve_elasticity_3d
from pymhm.tetrahedral import TetraMesh


@pytest.fixture(autouse=True)
def bounded_blas_threads():
    """Keep these small dense local operators from oversubscribing lightweight CI."""
    with threadpool_limits(1):
        yield


GRADIENT = np.array([[1.0, 2, 3], [-1, 3, 2], [2, -3, 1]])
STIFFNESS = np.diag([5.0, 6, 7, 2, 3, 4]) + 0.1 * np.ones((6, 6))
STRAIN = np.einsum("aij,ij->a", _KELVIN3, GRADIENT)
SIGMA = np.einsum("a,aij->ij", STIFFNESS @ STRAIN, _KELVIN3)


def exact(points: np.ndarray) -> np.ndarray:
    """Return an affine field with all six strain components present."""
    return points @ GRADIENT.T


@pytest.mark.parametrize("degree,refinement", [(1, 4), (2, 2), (3, 1), (4, 1)])
def test_variable_general_tensor_affine_patch(degree: int, refinement: int) -> None:
    """Match physical tensor gradients and every macro force/moment equation."""

    def material(x: np.ndarray) -> np.ndarray:
        """Evaluate a positive variable scalar multiple of the anisotropic stiffness."""
        return (1 + x @ np.array([1.0, 2, 3]))[:, None, None] * STIFFNESS

    cube = TetraMesh.unit_cube()
    mesh = TetraMesh(cube.points, cube.cells[:2])
    result = solve_elasticity_3d(
        mesh,
        degree=degree,
        local_refinement=refinement,
        constitutive=material,
        dirichlet=exact,
        source=-SIGMA @ np.array([1.0, 2, 3]),
        quadrature_order=degree + 3,
    )
    errors = result.errors(exact, lambda x: (1 + x @ np.array([1.0, 2, 3]))[:, None, None] * SIGMA)
    assert max(errors.values()) < 2e-9
    assert np.max(np.abs(result.equilibrium_residuals())) < 2e-10
    assert_allclose(
        result.evaluate(0, np.array([[0.1, 0.2, 0.3, 0.4]]))[1],
        np.broadcast_to(GRADIENT, (len(result.local_meshes[0].cells), 1, 3, 3)),
        atol=2e-10,
    )


def test_six_modes_are_exact_translations_and_centered_rotations() -> None:
    """The last three columns have zero symmetric gradient and the declared axis order."""
    points = np.array([[1.0, 2, 3], [2, 4, 1]])
    center = np.array([0.5, 0.5, 0.5])
    modes = rigid_modes_3d(points, center)
    assert_allclose(modes[:, :, :3], np.broadcast_to(np.eye(3), (2, 3, 3)))
    for axis in range(3):
        assert_allclose(modes[:, :, 3 + axis], np.cross(np.eye(3)[axis], points - center))


def test_full_traction_preserves_six_physical_moments() -> None:
    """Stress determines strain while six integrated moments choose the rigid representative."""
    mesh = TetraMesh.unit_cube()
    result = solve_elasticity_3d(
        mesh,
        constitutive=STIFFNESS,
        traction={int(f): SIGMA @ mesh.normals[f] for f in mesh.boundary_faces},
        rigid_moments=(3, 2, 0, -5 / 12, 1 / 12, -1 / 4),
    )
    assert max(result.errors(exact, SIGMA).values()) < 2e-10


def test_mixed_traction_and_cartesian_tensor() -> None:
    """Cartesian symmetries preserve Kelvin energy and outward physical traction signs."""
    cube = TetraMesh.unit_cube()
    mesh = TetraMesh(cube.points, cube.cells[:2])
    material = np.einsum("aij,ab,bkl->ijkl", _KELVIN3, STIFFNESS, _KELVIN3)
    assert_allclose(constitutive_values_3d(material, np.zeros((1, 3)))[0], STIFFNESS, atol=2e-15)
    face = int(mesh.boundary_faces[0])
    result = solve_elasticity_3d(
        mesh, constitutive=material, dirichlet=exact, traction={face: SIGMA @ mesh.normals[face]}
    )
    assert max(result.errors(exact, SIGMA).values()) < 2e-10


def test_segmented_p1_face_modes_preserve_rigid_traces() -> None:
    """Resolved dyadic subtriangles keep every vector component's signed orientation."""
    cube = TetraMesh.unit_cube()
    mesh = TetraMesh(cube.points, cube.cells[:2])
    skeleton = TriangularSkeleton(mesh, 2, degree=1)
    result = solve_elasticity_3d(
        mesh,
        skeleton=skeleton,
        degree=3,
        local_refinement=2,
        dirichlet=exact,
        constitutive=STIFFNESS,
    )
    assert max(result.errors(exact, SIGMA).values()) < 3e-9


@pytest.mark.parametrize(
    "material,match",
    [
        (np.eye(6) * 1j, "real"),
        (np.full((6, 6), np.nan), "finite"),
        (np.eye(3), "shape"),
        (np.triu(np.ones((6, 6))), "symmetric"),
        (np.diag([1, 1, 1, 1, 1, 0]), "positive definite"),
    ],
)
def test_tensor_contract(material: np.ndarray, match: str) -> None:
    """Reject nonphysical or dimensionally incompatible constitutive operators."""
    with pytest.raises(ValueError, match=match):
        constitutive_values_3d(material, np.zeros((1, 3)))


def test_minor_symmetry_and_lame_contracts() -> None:
    """Skew-strain coupling and nonpositive shear cannot be silently projected away."""
    tensor = np.einsum("aij,ab,bkl->ijkl", _KELVIN3, STIFFNESS, _KELVIN3)
    tensor[0, 1, 0, 0] += 0.1
    with pytest.raises(ValueError, match="minor symmetries"):
        constitutive_values_3d(tensor, np.zeros((1, 3)))
    for options in ({"lame_mu": 0.0}, {"lame_lambda": -1.0}):
        with pytest.raises(ValueError, match="Lame"):
            constitutive_values_3d(None, np.zeros((1, 3)), **options)


@pytest.mark.parametrize("moments", [(0,) * 5, [0j] * 6, [np.nan] * 6])
def test_traction_gauge_contract(moments) -> None:
    """Require all six finite real integrated moments."""
    mesh = TetraMesh.unit_cube()
    with pytest.raises(ValueError, match="six finite"):
        solve_elasticity_3d(
            mesh, traction={int(f): (0, 0, 0) for f in mesh.boundary_faces}, rigid_moments=moments
        )


def test_unbalanced_force_is_not_hidden_by_rigid_gauge() -> None:
    """A prescribed displacement gauge cannot balance an incompatible Neumann force."""
    cube = TetraMesh.unit_cube()
    mesh = TetraMesh(cube.points, cube.cells[:2])
    with pytest.raises(ValueError, match="incompatib"):
        solve_elasticity_3d(
            mesh, source=(1, 0, 0), traction={int(f): (0, 0, 0) for f in mesh.boundary_faces}
        )


def test_skeleton_contracts() -> None:
    """P1 rigid traces must belong to the same mesh and fit the local face partition."""
    mesh = TetraMesh.unit_cube()
    for skeleton, match in [
        (TriangularSkeleton(TetraMesh.unit_cube(), degree=1), "belong"),
        (TriangularSkeleton(mesh), "rigid-motion"),
        (TriangularSkeleton(mesh, 4, degree=1), "resolve"),
    ]:
        with pytest.raises(ValueError, match=match):
            solve_elasticity_3d(mesh, skeleton=skeleton)
    result = solve_elasticity_3d(mesh)
    for stress in (np.eye(3) * 1j, np.full((3, 3), np.inf)):
        with pytest.raises(ValueError, match="exact stress"):
            result.errors((0, 0, 0), stress)


def test_explicit_local_refinement_precision_contract() -> None:
    """Extended local accumulation preserves the physical affine solution and strict gate."""
    mesh = TetraMesh.unit_cube()
    with pytest.raises(ValueError, match="refinement_precision"):
        solve_elasticity_3d(mesh, local_refinement_precision="unknown")
    if np.finfo(np.longdouble).eps == np.finfo(float).eps:
        with pytest.raises(ValueError, match="extended"):
            solve_elasticity_3d(mesh, local_refinement_precision="extended")
    else:
        result = solve_elasticity_3d(
            mesh, local_refinement_precision="extended", constitutive=STIFFNESS, dirichlet=exact
        )
        assert max(result.errors(exact, SIGMA).values()) < 2e-10
