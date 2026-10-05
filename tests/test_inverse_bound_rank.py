"""Stabilization constants require an identifiable physical energy quotient."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.elasticity.mixed_pressure import _inverse_constant, _strain_and_divergence
from pymhm._legacy.models.elasticity.pressure_forms_3d import tetra_elasticity_pressure_operators
from pymhm.fem.inequalities import laplacian_inverse_bound
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_element_tabulate, tetrahedron_quadrature
from pymhm.fem.scalar.triangle import tabulate
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("operator", ["flow", "elasticity"])
def test_triangle_rejects_unresolved_physical_strain_modes(operator) -> None:
    """An anisotropic P2 element cannot discard positive modes as extra kernels."""
    mesh = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1e-6]]), np.array([[0, 1, 2]]))
    bary, weights = triangle_quadrature(5)
    _, _, _, gradient, hessian = tabulate(mesh, 2, bary)
    lengths = mesh.lengths[mesh.cell_faces].max(axis=1)
    with threadpool_limits(1), pytest.raises(ValueError, match="resolved kernel"):
        if operator == "flow":
            laplacian_inverse_bound(gradient, hessian, weights, lengths)
        else:
            strain, _, strong = _strain_and_divergence(gradient, hessian)
            _inverse_constant(strain, strong, weights, lengths, float(lengths.max()))


def test_tetra_rejects_unresolved_physical_strain_modes() -> None:
    """Exactly six rigid modes must be separated before computing the GaLS bound."""
    mesh = TetraMesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1e-3, 0.0], [0.0, 0.0, 1.0]]),
        [[0, 1, 2, 3]],
    )
    with threadpool_limits(1), pytest.raises(ValueError, match="6 resolved kernel"):
        tetra_elasticity_pressure_operators(mesh, degree=2)


def test_triangle_p2_sharp_laplacian_bound() -> None:
    """The right-triangle P2 quotient has sharp m=1/96, including its boundary."""
    mesh = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), [[0, 1, 2]])
    bary, weights = triangle_quadrature(5)
    _, _, _, gradient, hessian = tabulate(mesh, 2, bary)
    diameter = mesh.lengths[mesh.cell_faces].max(axis=1)
    with threadpool_limits(1):
        bound = laplacian_inverse_bound(gradient, hessian, weights, diameter)
    np.testing.assert_allclose(bound, 1 / 96, rtol=3e-14)
    stiffness = np.einsum("q,tqia,tqja->tij", weights, gradient, gradient)[0]
    laplacian = np.trace(hessian, axis1=-2, axis2=-1)[0]
    strong = np.einsum("q,qi,qj->ij", weights, laplacian, laplacian)
    admissible = np.linalg.eigvalsh(stiffness - bound[0] * diameter[0] ** 2 * strong)
    excluded = np.linalg.eigvalsh(stiffness - 1.001 * bound[0] * diameter[0] ** 2 * strong)
    assert admissible.min() > -2e-14
    assert excluded.min() < -1e-4


def test_tetra_laplacian_rejects_unresolved_constant_quotient() -> None:
    """Three-dimensional flow must also resolve every nonconstant energy mode."""
    mesh = TetraMesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1e-6, 0.0], [0.0, 0.0, 1.0]]),
        [[0, 1, 2, 3]],
    )
    bary, weights = tetrahedron_quadrature(5)
    _, _, _, gradient, hessian = tetra_element_tabulate(mesh, 2, bary)
    diameter = np.array(
        [np.max(np.linalg.norm(mesh.points[:, None] - mesh.points[None, :], axis=-1))]
    )
    with threadpool_limits(1), pytest.raises(ValueError, match="1 resolved kernel"):
        laplacian_inverse_bound(gradient, hessian, weights, diameter)
