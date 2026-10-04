"""Verify the reference-quadrature Jacobian convention for the new physical norm."""

from types import SimpleNamespace

import numpy as np
import pytest

from examples.minimal_elasticity_mixed3d import physical_errors
from pymhm.meshes.mixed import AffineMixedMesh


@pytest.mark.parametrize("order", [3, 4, 5])
def test_constant_unit_divergence_has_unit_physical_norm_on_unit_cube(order):
    mesh = AffineMixedMesh.unit_cube(1)

    def evaluate(cell, reference):
        divergence = np.zeros((len(mesh.cells), len(reference), 3))
        divergence[..., 0] = 1
        return None, None, None, divergence

    solution = SimpleNamespace(
        local_meshes=[mesh],
        evaluate=evaluate,
        errors=lambda *args, **kwargs: {},
    )
    data = SimpleNamespace(
        displacement=0,
        stress=0,
        rotation=0,
        source=lambda points: np.zeros_like(points),
    )
    assert physical_errors(solution, data, order)["divergence_l2"] == pytest.approx(1.0)
