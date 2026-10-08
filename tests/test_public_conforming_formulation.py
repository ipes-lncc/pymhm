"""Public conforming/factor primitives retain physical boundary and quadrature conventions."""

import numpy as np
import pytest

from examples.formulations.conforming import conforming_quadrilateral, conforming_separated
from pymhm._legacy.models.darcy.conforming import solve_conforming_quadrilateral
from pymhm._legacy.models.darcy.separable import solve_separable_diffusion
from pymhm.fem.scalar.separable import (
    interval_nodal_quadrature,
    interval_weighted_operators,
    require_positive_separated,
)
from pymhm.materials.separable import SeparableField
from pymhm.meshes.cartesian import CartesianMacroMesh


@pytest.mark.parametrize("boundary", ["essential", "mixed", "natural"])
def test_public_global_reference_preserves_full_polynomial_and_mean(boundary):
    mesh = CartesianMacroMesh(2)

    def exact(points):
        return 1 + (points**2).sum(axis=1)

    natural = {
        int(face): lambda x, n=mesh.normals[face]: -2 * x @ n for face in mesh.boundary_faces
    }
    if boundary == "essential":
        natural = {}
    elif boundary == "mixed":
        natural = {min(natural): natural[min(natural)]}
    options = dict(degree=2, source=-4.0, dirichlet=exact, neumann=natural, mean_pressure=5 / 3)
    actual = conforming_quadrilateral(mesh, **options)
    original = solve_conforming_quadrilateral(mesh, **options)
    np.testing.assert_allclose(actual.pressure, original.pressure, atol=1e-12, rtol=1e-10)
    points = np.array([[0.2, 0.4], [0.7, 0.9]])
    np.testing.assert_allclose(actual.evaluate(points)[0], exact(points), atol=1e-12, rtol=1e-10)
    assert actual.residual < 1e-10
    if boundary == "natural":
        with pytest.raises(ValueError, match="incompatible"):
            conforming_quadrilateral(mesh, **dict(options, source=0))
        with pytest.raises(ValueError, match="finite"):
            conforming_quadrilateral(mesh, **dict(options, mean_pressure=np.nan))


def test_public_separated_reference_matches_the_same_element_operator():
    mesh = CartesianMacroMesh(2, 3, (2, 4, -1, 2))
    coefficient = SeparableField(((2.0, 1.0), (lambda x: x**2, lambda y: 1 + y**2)))
    source = SeparableField(((np.sin, np.cos),))
    actual = conforming_separated(mesh, degree=3, permeability=coefficient, source=source)
    original = solve_separable_diffusion(mesh, degree=3, permeability=coefficient, source=source)
    element = conforming_quadrilateral(mesh, degree=3, permeability=coefficient, source=source)
    for other in (original, element):
        np.testing.assert_allclose(actual.pressure, other.pressure, atol=1e-12, rtol=1e-10)


def test_interval_operators_retain_physical_scaling_and_reject_invalid_data():
    data = interval_nodal_quadrature(2, (2.0, 4.0), 2, 4)
    mass, stiffness, load = interval_weighted_operators(data, np.ones_like(data[0]))
    ones = np.ones(5)
    np.testing.assert_allclose(mass @ ones, load, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(stiffness @ ones, 0, atol=1e-12, rtol=1e-10)
    assert abs(load.sum() - 2) < 1e-12
    for bounds in ((0, 0), (1, 0), (np.nan, 1), (0j, 1), (0, 1, 2)):
        with pytest.raises(ValueError, match="interval"):
            interval_nodal_quadrature(2, bounds, 2, 4)
    with pytest.raises(ValueError, match="quadrature points"):
        interval_weighted_operators(data, np.ones(2))
    with pytest.raises(ValueError, match="nonempty"):
        require_positive_separated([])
