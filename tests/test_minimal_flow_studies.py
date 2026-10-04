"""Check physical volume and component conventions of the new Jacobian norm."""

from types import SimpleNamespace

import numpy as np
import pytest

from examples.minimal_flow_studies import vector_gradient_error
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("order", [3, 4, 5])
def test_affine_nonsymmetric_jacobian_is_integrated_in_physical_component_order(order):
    mesh = TriangleMesh.unit_square(1)
    gradient = np.array([[2.0, 3.0], [5.0, 7.0]])
    points = nodal_space(mesh, 1)[1]
    solution = SimpleNamespace(local_meshes=(mesh,), values=(points @ gradient.T,), degree=1)

    def exact(points):
        return np.broadcast_to(gradient, (len(points), 2, 2))

    assert vector_gradient_error(solution, exact, order) < 1e-14

    def zero(points):
        return np.zeros((len(points), 2, 2))

    assert vector_gradient_error(solution, zero, order) == pytest.approx(np.sqrt(87))
