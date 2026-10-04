"""Conservative BDM2/P1 Darcy patches, boundary gauges and high-order convergence."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm._legacy.models.darcy.mixed_bdm import solve_darcy_bdm
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


def pressure(points):
    """A cubic pressure whose anisotropic flux is exactly quadratic."""
    x, y = points.T
    return x**3 + y**3 + x * y


def gradient(points):
    """Gradient of the cubic pressure, differentiated independently."""
    x, y = points.T
    return np.column_stack((3 * x * x + y, 3 * y * y + x))


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_cubic_pressure_quadratic_flux_patch(backend):
    """Exact BDM2 flux coexists with the non-exact DG1 pressure projection."""
    mesh = TriangleMesh.unit_square()
    tensor = np.array([[2.0, 0.3], [0.3, 1.0]])
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))

    def source(x):
        """Divergence of the exact quadratic anisotropic flux."""
        return -12 * x[:, 0] - 6 * x[:, 1] - 0.6

    result = solve_darcy_bdm(
        mesh,
        permeability=tensor,
        dirichlet=pressure,
        source=source,
        skeleton=skeleton,
        backend=backend,
        workers=2,
    )
    assert result.flux_l2_error(lambda x: -gradient(x) @ tensor) < 3e-12
    assert result.divergence_l2_error(source) < 3e-12
    assert result.l2_error(pressure) > 1e-3
    bary, weights = triangle_quadrature(5)
    mass = np.einsum("q,qi,qj->ij", weights, bary, bary)
    for fine, values in zip(result.local_meshes, result.pressure, strict=True):
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        moments = np.einsum(
            "q,qi,tq->ti", weights, bary, pressure(points.reshape(-1, 2)).reshape(points.shape[:2])
        )
        assert_allclose(values, np.linalg.solve(mass, moments.T).T, atol=3e-12, rtol=0)
    assert_allclose(result.conservation_residuals(), 0, atol=3e-12)
    for method in (
        result.fine_equilibrium_residuals,
        result.fine_conservation_residuals,
        result.normal_flux_residuals,
    ):
        for defect in method():
            assert_allclose(defect, 0, atol=3e-12)


def test_spatial_anisotropic_permeability():
    """An affine pressure with variable tensor has nonzero constant divergence."""

    def permeability(x):
        value = np.broadcast_to(np.array([[1.0, 0.2], [0.2, 2.0]]), (len(x), 2, 2)).copy()
        value[:, 0, 0] += x[:, 0]
        value[:, 1, 1] += x[:, 1]
        return value

    result = solve_darcy_bdm(
        TriangleMesh.unit_square(2),
        permeability=permeability,
        dirichlet=lambda x: x[:, 0] + 2 * x[:, 1],
        source=-3.0,
    )
    assert result.l2_error(lambda x: x[:, 0] + 2 * x[:, 1]) < 2e-12
    assert (
        result.flux_l2_error(lambda x: np.column_stack((-1.4 - x[:, 0], -4.2 - 2 * x[:, 1])))
        < 3e-12
    )
    assert result.divergence_l2_error(-3.0) < 3e-12


def test_pure_neumann_and_mixed_boundary_gauges():
    """Outward physical flux and mean pressure preserve an affine solution."""
    mesh = TriangleMesh.unit_square()

    def exact(x):
        """Affine pressure with prescribed mean 3.2."""
        return 1.7 + x[:, 0] + 2 * x[:, 1]

    neumann = {int(f): float(mesh.normals[f] @ [-1.0, -2.0]) for f in mesh.boundary_faces}
    result = solve_darcy_bdm(mesh, neumann=neumann, mean_pressure=3.2)
    assert result.l2_error(exact) < 3e-12
    assert result.flux_l2_error([-1.0, -2.0]) < 3e-12
    partial = {next(iter(neumann)): next(iter(neumann.values()))}
    mixed = solve_darcy_bdm(mesh, neumann=partial, dirichlet=exact)
    assert mixed.l2_error(exact) < 3e-12
    with pytest.raises(ValueError, match="incompatible"):
        solve_darcy_bdm(mesh, neumann=neumann, source=1.0)


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_aligned_segmented_trace_spaces(degree):
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(degree, 2) for _ in mesh.faces))
    result = solve_darcy_bdm(mesh, skeleton=skeleton, dirichlet=lambda x: 1 + x[:, 0] - 2 * x[:, 1])
    assert result.l2_error(lambda x: 1 + x[:, 0] - 2 * x[:, 1]) < 3e-12
    assert result.flux_l2_error([-1, 2]) < 3e-12


def test_smooth_pressure_flux_and_divergence_convergence():
    def exact(x):
        return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

    def flux(x):
        return -np.pi * np.column_stack(
            (
                np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]),
                np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]),
            )
        )

    def source(x):
        """Analytical negative Laplacian of the pressure."""
        return 2 * np.pi**2 * exact(x)

    errors = []
    for n in (2, 4, 8):
        result = solve_darcy_bdm(TriangleMesh.unit_square(n), source=source, quadrature_order=6)
        errors.append(
            (
                result.l2_error(exact, 8),
                result.flux_l2_error(flux, 8),
                result.divergence_l2_error(source, 8),
            )
        )
    assert np.all(np.log2(np.asarray(errors[:-1]) / errors[1:]) > 1.7)


def test_aligned_layer_high_contrast_patch():
    def permeability(x):
        return np.where(x[:, 0] < 0.5, 1.0, 1000.0)

    def exact(x):
        return np.where(x[:, 0] <= 0.5, 1 - x[:, 0], 0.5 - (x[:, 0] - 0.5) / 1000)

    result = solve_darcy_bdm(
        TriangleMesh.unit_square(2), permeability=permeability, dirichlet=exact
    )
    assert result.l2_error(exact) < 2e-12
    assert result.flux_l2_error([1, 0]) < 2e-10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"local_refinement": 0},
        {"quadrature_order": 2},
        {"permeability": -1},
        {"permeability": [[1, 2], [0, 1]]},
        {"permeability": np.nan},
    ],
)
def test_validation(kwargs):
    with pytest.raises(ValueError):
        solve_darcy_bdm(TriangleMesh.unit_square(), **kwargs)


def test_invalid_skeletons():
    mesh = TriangleMesh.unit_square()
    for skeleton in (SkeletonSpace(mesh, components=2), SkeletonSpace(TriangleMesh.unit_square())):
        with pytest.raises(ValueError, match="skeleton"):
            solve_darcy_bdm(mesh, skeleton=skeleton)
    for face in (FaceSpace.uniform(3), FaceSpace((0.0, 0.500004, 1.0), (1, 1))):
        skeleton = SkeletonSpace(mesh, tuple(face for _ in mesh.faces))
        with pytest.raises(ValueError, match="aligned"):
            solve_darcy_bdm(mesh, skeleton=skeleton)
