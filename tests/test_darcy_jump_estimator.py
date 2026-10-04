"""Independent side-multiplicity and scaling checks for the Darcy jump indicator."""

from dataclasses import replace

import numpy as np
import pytest

from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.estimators.darcy_jump import estimate_darcy_jumps
from pymhm.meshes.triangle import TriangleMesh


def test_constant_fields_and_interior_multiplicity():
    """Integrate known constant residuals analytically, counting an interior face twice."""
    mesh = TriangleMesh.unit_square(1)
    solution = solve_darcy(mesh, local_refinement=2, permeability=4.0)
    constant = tuple(np.ones_like(values) for values in solution.pressure)
    same = replace(solution, pressure=constant)
    result = estimate_darcy_jumps(same, calibration=3.0)
    assert result.total == pytest.approx(12.0)
    opposite = replace(solution, pressure=(constant[0], -constant[1]))
    result = estimate_darcy_jumps(opposite, calibration=3.0, order=4)
    assert result.total == pytest.approx(6 * np.sqrt(6))
    assert np.sum(result.face_squared) == pytest.approx(5 * 36)
    assert result.coefficient_scale == 2.0
    natural = {int(face): 0.0 for face in mesh.boundary_faces}
    result = estimate_darcy_jumps(opposite, neumann=natural, ellipticity_lower_bound=[4, 4])
    assert result.total == pytest.approx(np.sqrt(8))
    assert estimate_darcy_jumps(same, dirichlet=1.0).total < 1e-13


def test_affine_patch_and_validation():
    """A represented affine solution has zero jumps; invalid contracts are rejected."""
    mesh = TriangleMesh.unit_square(1)

    def exact(x):
        """Represented affine boundary field."""
        return 1 + x[:, 0] + 2 * x[:, 1]

    solution = solve_darcy(mesh, dirichlet=exact, local_refinement=2)
    assert estimate_darcy_jumps(solution, dirichlet=exact).total < 1e-13
    with pytest.raises(ValueError, match="primal"):
        estimate_darcy_jumps(replace(solution, formulation="mixed"))
    for value in (0, -1, np.inf, 1j):
        with pytest.raises(ValueError, match="calibration"):
            estimate_darcy_jumps(solution, calibration=value)
    interior = int(np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0])
    with pytest.raises(ValueError, match="exterior"):
        estimate_darcy_jumps(solution, neumann={interior: 0})
    with pytest.raises(ValueError, match="order"):
        estimate_darcy_jumps(solution, order=0)
