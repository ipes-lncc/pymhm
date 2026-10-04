"""Exact Oseen patches and explicit coefficient contracts for variable transport."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.flow.solver import _flow_local
from pymhm._legacy.models.vector import solve_brinkman


@pytest.mark.parametrize("formulation", ["taylor-hood", "oseen"])
@pytest.mark.parametrize("variable", [False, True])
def test_affine_patch_with_nonzero_advection_divergence(formulation: str, variable: bool) -> None:
    """An exact affine solution checks the skew-divergence correction and pressure gauge."""
    mesh = TriangleMesh.unit_square()
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)

    def exact(points: np.ndarray) -> np.ndarray:
        """Return an affine divergence-free velocity."""
        return points * np.array([1.0, -1.0])

    beta = (lambda points: points) if variable else (1.0, 2.0)

    def force(points: np.ndarray) -> np.ndarray:
        """Evaluate -nu Delta u + beta.grad(u) + 2u + grad(p) analytically."""
        convection = points if variable else np.tile(beta, (len(points), 1))
        return 2 * exact(points) + convection * np.array([1.0, -1.0]) + [1.0, 2.0]

    solution = solve_brinkman(
        mesh,
        formulation=formulation,
        degree=2,
        local_refinement=3,
        skeleton=space,
        drag=2.0,
        advection=beta,
        advection_divergence=2.0 if variable else None,
        advection_bound=np.sqrt(2) if variable else None,
        source=force,
        dirichlet=exact,
        mean_pressure=0.7,
    )
    assert solution.l2_error(exact) < 3e-12
    assert solution.pressure_l2_error(lambda x: x[:, 0] + 2 * x[:, 1] - 0.8) < 3e-11
    assert solution.divergence_l2() < 3e-11


def test_zero_advection_oseen_is_stokes_usfem() -> None:
    """The published delta formula has a defined zero-drag and zero-advection limit."""
    mesh = TriangleMesh.unit_square()
    space = SkeletonSpace(mesh, components=2)
    options = dict(
        mesh=mesh,
        skeleton=space,
        degree=2,
        refinement=2,
        viscosity=0.3,
        drag=0.0,
        beta=np.zeros(2),
        source=(1.0, 2.0),
        order=5,
    )
    stokes = _flow_local(0, formulation="usfem", **options).problem
    oseen = _flow_local(0, formulation="oseen", beta_bound=0.0, **options).problem
    assert_allclose(oseen.matrix.toarray(), stokes.matrix.toarray(), atol=3e-15)
    assert_allclose(oseen.load, stokes.load, atol=3e-15)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"advection": lambda x: x}, "advection_divergence"),
        ({"advection": lambda x: x, "advection_divergence": 2}, "advection_bound"),
        ({"advection_divergence": 1}, "zero divergence"),
        ({"advection_divergence": lambda x: 0}, "zero divergence"),
        ({"advection_bound": -1}, "finite nonnegative"),
        ({"advection_bound": np.inf}, "finite nonnegative"),
        ({"advection_bound": 1j}, "finite nonnegative"),
        ({"advection_bound": [1]}, "finite nonnegative"),
        ({"advection_bound": "a"}, "finite nonnegative"),
        ({"advection": (2, 0), "advection_bound": 1}, "sampled advection"),
        ({"drag": np.eye(2)}, "constant scalar"),
        ({"drag": lambda x: 1}, "constant scalar"),
        ({"drag": 1j}, "constant scalar"),
        ({"drag": -1}, "nonnegative"),
    ],
)
def test_oseen_coefficient_contracts(options: dict[str, Any], match: str) -> None:
    """Reject missing differential data, uncertified bounds and unsupported material tensors."""
    with pytest.raises(ValueError, match=match):
        solve_brinkman(TriangleMesh.unit_square(), formulation="oseen", **options)


def test_variable_solenoidal_taylor_hood_needs_no_supremum() -> None:
    """A smooth prescribed vector and its divergence are sufficient for the unstabilized form."""
    solution = solve_brinkman(
        TriangleMesh.unit_square(),
        advection=lambda x: np.column_stack((x[:, 1], -x[:, 0])),
        advection_divergence=lambda x: np.zeros(len(x)),
        drag=1.0,
    )
    assert solution.l2_error((0, 0)) == 0


def test_variable_advection_natural_boundary_uses_skew_pseudotraction() -> None:
    """A nonzero outlet fixes pressure through nu grad(u)n-pn-u(beta.n)/2."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    outlet = next(
        int(face) for face in mesh.boundary_faces if np.all(mesh.points[mesh.faces[face], 0] == 1)
    )

    def exact(points: np.ndarray) -> np.ndarray:
        """Evaluate the solenoidal affine velocity."""
        return points * [1.0, -1.0]

    solution = solve_brinkman(
        mesh,
        formulation="oseen",
        degree=2,
        local_refinement=3,
        skeleton=skeleton,
        drag=2,
        advection=lambda x: x,
        advection_divergence=2,
        advection_bound=np.sqrt(2),
        dirichlet=exact,
        source=lambda x: 3 * exact(x) + [1, 2],
        traction={outlet: lambda x: np.column_stack((-0.5 - 2 * x[:, 1], x[:, 1] / 2))},
    )
    assert solution.l2_error(exact) < 2e-12
    assert solution.pressure_l2_error(lambda x: x[:, 0] + 2 * x[:, 1]) < 2e-11
