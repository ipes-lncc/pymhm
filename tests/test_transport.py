"""Robin transport identities and implicit heat evolution verification."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.transport import solve_heat, solve_transport


def affine(x):
    """Return a scalar linear field whose Robin flux is facewise affine."""
    return 1 + x[:, 0] + 2 * x[:, 1]


@pytest.mark.parametrize(
    "velocity,reaction", [((0.0, 0.0), 0.0), ((1.0, -0.25), 0.0), ((1.0, 0.5), 2.0)]
)
def test_robin_rad_patch(velocity, reaction):
    mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))

    def source(x):
        return reaction * affine(x) + velocity[0] + 2 * velocity[1]

    result = solve_transport(
        mesh,
        velocity=velocity,
        reaction=reaction,
        source=source,
        dirichlet=affine,
        skeleton=skeleton,
    )
    assert result.l2_error(affine) < 2e-11
    for face in range(len(mesh.faces)):
        t = np.array([0.25, 0.75])
        points = mesh.points[mesh.faces[face, 0]] + t[:, None] * (
            mesh.points[mesh.faces[face, 1]] - mesh.points[mesh.faces[face, 0]]
        )
        exact = (
            -np.array([1, 2]) + affine(points)[:, None] * np.array(velocity)[None] / 2
        ) @ mesh.normals[face]
        computed = skeleton.faces[face].evaluate(t) @ result.hybrid.trace[skeleton.dofs(face)]
        assert_allclose(computed, exact, atol=5e-10)


def test_diffusion_limit_equivalence():
    mesh = TriangleMesh.unit_square(2)
    scalar = solve_transport(mesh, source=1, dirichlet=affine)
    darcy = solve_darcy(mesh, source=1, dirichlet=affine)
    assert_allclose(scalar.hybrid.trace, darcy.hybrid.trace, atol=1e-12)


def test_heat_linear_space_time_patch_and_zero_solution():
    mesh = TriangleMesh.unit_square(2)
    times = [0.0, 0.01, 0.03, 0.1]
    results = solve_heat(
        mesh,
        times,
        initial=affine,
        source=lambda x, t: np.ones(len(x)),
        dirichlet=lambda x, t: affine(x) + t,
    )
    for result, time in zip(results, times[1:], strict=True):
        assert result.l2_error(lambda x, time=time: affine(x) + time) < 3e-12
    assert solve_heat(mesh, [0, 1])[0].l2_error(0) == 0


def test_heat_energy_decay_and_temporal_refinement():
    mesh = TriangleMesh.unit_square(3)

    def initial(x):
        return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

    errors = []
    for steps in (2, 4):
        result = solve_heat(
            mesh, np.linspace(0, 0.1, steps + 1), initial=initial, local_refinement=3
        )[-1]
        errors.append(result.l2_error(lambda x: initial(x) * np.exp(-2 * np.pi**2 * 0.1)))
        assert result.l2_error(0) < 0.5
    assert errors[1] < errors[0]


@pytest.mark.parametrize("times", [[], [0], [1, 0], [0, np.nan], [[0, 1]]])
def test_heat_time_validation(times):
    with pytest.raises(ValueError, match="times"):
        solve_heat(TriangleMesh.unit_square(), times)


def test_transport_and_heat_validators():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="reaction"):
        solve_transport(mesh, reaction=-1)
    with pytest.raises(ValueError, match="skeleton"):
        solve_transport(mesh, skeleton=SkeletonSpace(mesh, components=2))
    with pytest.raises(ValueError, match="skeleton"):
        solve_heat(mesh, [0, 1], skeleton=SkeletonSpace(mesh, components=2))
