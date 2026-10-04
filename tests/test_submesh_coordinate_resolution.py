"""Partition validation separates coordinate uncertainty from resolved geometry defects."""

import numpy as np
import pytest

from pymhm.fem.quadrature.material import fit_material_mesh
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.refinement import validate_submesh
from pymhm.meshes.triangle import TriangleMesh


def fitted_triangle(area_case=False):
    """Construct a small physical triangle far from the origin, intersected by pixels."""
    vertices = np.array(
        [
            [502.005176879, 2138.308634],
            [503.760684705, 2142.38628098],
            [497.538100337, 2141.23400816],
        ]
    )
    if area_case:
        vertices = np.array(
            [
                [203.298366299, 1646.89836962],
                [207.703654529, 1646.70367417],
                [203.728073806, 1651.30796978],
            ]
        )
    macro = TriangleMesh(vertices, np.array([[0, 1, 2]]))
    material = CartesianCellField(np.ones((60, 220)), (20.0, 10.0))
    fine = fit_material_mesh(macro.submesh(0, 4), material)
    return macro, fine


@pytest.mark.parametrize("area_case", [False, True])
def test_material_fitted_membership_survives_absolute_coordinate_roundoff(area_case):
    """A valid cut partition passes at the reservoir coordinates and after translation."""
    macro, fine = fitted_triangle(area_case)
    validate_submesh(macro, 0, fine)
    # Translation of represented vertices cannot undo their original roundoff;
    # an independently constructed translated partition is checked instead.
    shift = macro.points[0]
    translated = TriangleMesh(macro.points - shift, macro.cells)
    material = CartesianCellField(np.ones((60, 220)), (20.0, 10.0), origin=tuple(-shift))
    local = fit_material_mesh(translated.submesh(0, 4), material)
    validate_submesh(translated, 0, local)


def test_resolvable_displacement_and_area_change_remain_rejected():
    """Roundoff-aware membership does not admit a shifted or enlarged physical domain."""
    macro, fine = fitted_triangle()
    shifted = TriangleMesh(fine.points + [1e-6, 0], fine.cells)
    with pytest.raises(ValueError, match="cover"):
        validate_submesh(macro, 0, shifted)
    center = fine.points.mean(axis=0)
    enlarged = TriangleMesh(center + (1 + 1e-6) * (fine.points - center), fine.cells)
    with pytest.raises(ValueError, match="cover"):
        validate_submesh(macro, 0, enlarged)
    # Inset vertices satisfy membership, but the resolvable area loss must fail
    # independently of the boundary-containment check.
    inset = TriangleMesh(center + (1 - 1e-6) * (fine.points - center), fine.cells)
    with pytest.raises(ValueError, match="cover"):
        validate_submesh(macro, 0, inset)
