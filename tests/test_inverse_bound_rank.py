"""Stabilization constants require an identifiable physical energy quotient."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.elasticity.mixed_pressure import _inverse_constant, _strain_and_divergence
from pymhm._legacy.models.elasticity.pressure_forms_3d import tetra_elasticity_pressure_operators
from pymhm._legacy.models.flow.solver import _laplacian_inverse_bound
from pymhm.fem.scalar.operators import triangle_quadrature
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
            _laplacian_inverse_bound(gradient, hessian, weights, lengths)
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
