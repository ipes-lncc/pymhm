"""Explicit 2017 minimum-eigenvalue and 2025 tensor-safe stabilization conventions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_brinkman
from pymhm.flow import _minimum_resistance
from pymhm.reservoir import CartesianCellField


@pytest.mark.parametrize(
    "field,minimum",
    [
        (0.0, 0.0),
        (2.5, 2.5),
        (np.diag([1.5, 10.0]), 1.5),
        (CartesianCellField(np.array([[3.0, 7.0], [1.0, 2.0]]), (0.5, 0.5)), 1.0),
        (CartesianCellField(np.tile(np.diag([2.0, 11.0]), (2, 3, 1, 1)), (0.5, 1 / 3)), 2.0),
    ],
)
def test_exact_global_minimum_is_explicit_for_constant_and_cartesian_fields(field, minimum):
    """Use all pixels and the smallest tensor eigenvalue, not a sampled centroid."""
    assert _minimum_resistance(field, None) == minimum
    assert _minimum_resistance(field, minimum / 2) == minimum / 2
    with pytest.raises(ValueError, match="minimum eigenvalue"):
        _minimum_resistance(field, minimum + 1)


@pytest.mark.parametrize("bound", [-1.0, np.nan, np.inf, 1j, [1.0], "bad", True])
def test_invalid_minimum_bound_is_rejected(bound):
    """Reject nonphysical bounds and nonscalar/nonreal contracts before assembly."""
    with pytest.raises(ValueError, match="finite nonnegative scalar"):
        _minimum_resistance(2.0, bound)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"stabilization": "unknown"}, "stabilization must"),
        ({"gamma_min": 0.0}, "only used"),
        ({"formulation": "taylor-hood", "stabilization": "minimum-2017"}, "Taylor-Hood"),
        ({"stabilization": "minimum-2017", "drag": lambda points: 2.0}, "explicit gamma_min"),
        (
            {"stabilization": "minimum-2017", "gamma_min": 3.0, "drag": lambda points: 2.0},
            "sampled material",
        ),
    ],
)
def test_stabilization_selector_and_callback_bound_contracts(kwargs, match):
    """Never infer a spatial convention from a callback or ignore unused parameters."""
    options = dict(formulation="usfem", local_refinement=4)
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        solve_brinkman(TriangleMesh.unit_square(), **options)


@pytest.mark.parametrize("mode", ["constant", "cartesian", "callback"])
@pytest.mark.parametrize("rule", ["minimum-2017", "pointwise-2017"])
def test_minimum_2017_reproduces_affine_flow_with_nonhomogeneous_data(mode, rule):
    """Preserve consistency of the full stabilized source and pressure-gradient terms."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace(degrees=(1,)) for _ in mesh.faces), 2)
    tensor = np.array([[2.0, 0.3], [0.3, 4.0]])
    coefficient = tensor
    bound = None
    if mode == "cartesian":
        coefficient = CartesianCellField(np.tile(tensor, (3, 5, 1, 1)), (1 / 3, 0.2))
    if mode == "callback":

        def coefficient(points):
            """Evaluate the same constant tensor through the callback API."""
            return np.tile(tensor, (len(points), 1, 1))

        bound = float(np.linalg.eigvalsh(tensor)[0]) if rule == "minimum-2017" else None

    def velocity(points):
        """Return an incompressible affine field representable by all local spaces."""
        return points * [1, -1]

    def pressure(points):
        """Return affine zero-mean pressure whose gradient enters the exact force."""
        return points[:, 0] - 0.5

    def force(points):
        """Compute the full tensor drag plus pressure gradient independently."""
        return velocity(points) @ tensor.T + [1.0, 0.0]

    solution = solve_brinkman(
        mesh,
        viscosity=0.3,
        drag=coefficient,
        source=force,
        dirichlet=velocity,
        skeleton=skeleton,
        formulation="usfem",
        degree=3,
        local_refinement=1,
        stabilization=rule,
        gamma_min=bound,
    )
    assert solution.l2_error(velocity) < 3e-13
    assert solution.pressure_l2_error(pressure) < 3e-12


@pytest.mark.parametrize("rule", ["minimum-2017", "pointwise-2017"])
def test_both_stabilization_rules_have_the_same_continuous_stokes_limit(rule):
    """Recover identical fields when gamma_min and every drag eigenvalue vanish."""
    mesh = TriangleMesh.unit_square()
    first = solve_brinkman(mesh, source=(1.0, 2.0), formulation="usfem", degree=2)
    second = solve_brinkman(
        mesh, source=(1.0, 2.0), formulation="usfem", degree=2, stabilization=rule
    )
    assert_allclose(np.asarray(first.values), np.asarray(second.values), atol=1e-14)
    assert_allclose(np.asarray(first.pressure), np.asarray(second.pressure), atol=1e-13)
