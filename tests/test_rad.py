"""Conservative variable-coefficient transport and consistent SUPG verification."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.transport.solver import solve_heat, solve_transport


def quadratic(points):
    """Return a quadratic pressure with variable-coefficient polynomial flux."""
    return np.sum(points**2, axis=1)


def permeability(points):
    """Return an SPD diagonal tensor with independently known divergence."""
    result = np.zeros((len(points), 2, 2))
    result[:, 0, 0] = 1 + points[:, 0]
    result[:, 1, 1] = 2 + points[:, 1]
    return result


def velocity(points):
    """Return an affine compressible velocity field with divergence two."""
    return points + [1.0, -0.5]


@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
def test_variable_conservative_transport_patch_and_robin_flux(stabilization):
    mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(3) for _ in mesh.faces))
    solution = solve_transport(
        mesh,
        degree=3,
        skeleton=skeleton,
        diffusion=permeability,
        velocity=velocity,
        velocity_divergence=2.0,
        diffusion_divergence=(1.0, 1.0),
        reaction=1.0,
        source=lambda x: -6 - 2 * x[:, 0] - 5 * x[:, 1] + 5 * np.sum(x * x, axis=1),
        dirichlet=quadratic,
        stabilization=stabilization,
    )
    assert solution.l2_error(quadratic) < 3e-12
    for face in range(len(mesh.faces)):
        parameter, _ = skeleton.faces[face].quadrature(6)
        start, end = mesh.points[mesh.faces[face]]
        points = start + parameter[:, None] * (end - start)
        flux = (
            -np.einsum("qij,qj->qi", permeability(points), 2 * points)
            + velocity(points) * quadratic(points)[:, None] / 2
        )
        actual = (
            skeleton.faces[face].evaluate(parameter) @ solution.hybrid.trace[skeleton.dofs(face)]
        )
        assert_allclose(actual, flux @ mesh.normals[face], atol=2e-10)


@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
def test_transport_robin_and_neumann_boundaries(stabilization):
    mesh = TriangleMesh.unit_square(2)
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))

    def exact(x):
        return 1 + x[:, 0] + 2 * x[:, 1]

    neumann = {int(face): -mesh.normals[face] @ [1.0, 2.0] for face in mesh.boundary_faces}
    pure = solve_transport(
        mesh, neumann=neumann, mean_value=2.5, degree=2, stabilization=stabilization
    )
    assert pure.l2_error(exact) < 2e-12
    beta = np.array([1.0, -0.2])
    face = int(mesh.boundary_faces[0])
    robin = {
        face: lambda x: (
            (-np.array([1.0, 2.0]) + beta[None] * exact(x)[:, None] / 2) @ mesh.normals[face]
        )
    }
    mixed = solve_transport(
        mesh,
        skeleton=space,
        velocity=beta,
        reaction=3.0,
        source=lambda x: 3 * exact(x) + 0.6,
        dirichlet=exact,
        neumann=robin,
        stabilization=stabilization,
    )
    assert mixed.l2_error(exact) < 2e-12


@pytest.mark.parametrize(
    "options,match",
    [
        ({"stabilization": "gls"}, "stabilization"),
        ({"velocity": velocity}, "velocity_divergence"),
        ({"stabilization": "supg", "diffusion": permeability}, "diffusion_divergence"),
        ({"velocity_divergence": 1.0}, "constant velocity"),
        ({"diffusion_divergence": (1.0, 0.0)}, "constant diffusion"),
        ({"mean_value": np.nan}, "finite"),
        ({"mean_value": 1.0}, "gauge"),
        ({"velocity": lambda x: -x, "velocity_divergence": -2.0}, "nonnegative"),
    ],
)
def test_transport_validation(options, match):
    with pytest.raises(ValueError, match=match):
        solve_transport(TriangleMesh.unit_square(), **options)


@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
@pytest.mark.parametrize("reaction", [-1.0, -0.5])
def test_negative_reaction_compensated_by_conservative_divergence(stabilization, reaction):
    """L12 requires c+div(beta)/2>=0, including its zero endpoint."""
    mesh = TriangleMesh.unit_square()
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))

    def exact(x):
        return 1 + x[:, 0] + 2 * x[:, 1]

    result = solve_transport(
        mesh,
        skeleton=space,
        degree=3,
        local_refinement=2,
        velocity=lambda x: x,
        velocity_divergence=2.0,
        reaction=reaction,
        dirichlet=exact,
        source=lambda x: x[:, 0] + 2 * x[:, 1] + (2 + reaction) * exact(x),
        stabilization=stabilization,
    )
    assert result.l2_error(exact, order=6) < 2e-12


def test_negative_effective_reaction_remains_rejected():
    with pytest.raises(ValueError, match="reaction\\+div"):
        solve_transport(
            TriangleMesh.unit_square(),
            velocity=lambda x: x,
            velocity_divergence=2.0,
            reaction=-1.01,
        )


@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
@pytest.mark.parametrize("enforcement", ["weak", "strong"])
def test_diffusive_boundary_flux_is_not_the_robin_multiplier(stabilization, enforcement):
    mesh = TriangleMesh.unit_square(2)
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    right = [
        int(face)
        for face in mesh.boundary_faces
        if np.allclose(mesh.points[mesh.faces[face], 0], 1)
    ]
    result = solve_transport(
        mesh,
        skeleton=space,
        velocity=(1.0, 0.0),
        reaction=1.0,
        source=lambda x: 2 + x[:, 0],
        dirichlet=lambda x: 1 + x[:, 0],
        diffusive_flux=dict.fromkeys(right, -1.0),
        dirichlet_enforcement=enforcement,
        stabilization=stabilization,
    )
    assert result.l2_error(lambda x: 1 + x[:, 0]) < 2e-12
    for face in right:
        assert_allclose(
            space.faces[face].evaluate(np.array([0.2, 0.8]))
            @ result.hybrid.trace[space.dofs(face)],
            -1,
            atol=1e-11,
        )


@pytest.mark.parametrize(
    "options,match",
    [
        ({"dirichlet_enforcement": "other"}, "dirichlet_enforcement"),
        ({"neumann": {0: 0}, "diffusive_flux": {0: 0}}, "disjoint"),
    ],
)
def test_extended_boundary_validation(options, match):
    with pytest.raises(ValueError, match=match):
        solve_transport(TriangleMesh.unit_square(), **options)


@pytest.mark.parametrize("degree", [2, 3])
def test_heat_high_order_space_and_time_patch(degree):
    mesh = TriangleMesh.unit_square(2)
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(degree - 1) for _ in mesh.faces))

    def initial(x):
        return x[:, 0] ** degree + x[:, 1] ** degree

    def laplacian(x):
        return degree * (degree - 1) * (x[:, 0] ** (degree - 2) + x[:, 1] ** (degree - 2))

    times = [0.0, 0.01, 0.04, 0.1]
    results = solve_heat(
        mesh,
        times,
        degree=degree,
        skeleton=space,
        initial=initial,
        dirichlet=lambda x, t: initial(x) + t,
        source=lambda x, t: 1 - laplacian(x),
    )
    for solution, time in zip(results, times[1:], strict=True):
        assert solution.degree == degree
        assert solution.l2_error(lambda x, time=time: initial(x) + time) < 3e-12


@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
def test_tangential_divergence_free_transport_has_a_constant_gauge(stabilization):
    """The physical Robin trace of a constant state must be representable too."""
    mesh = TriangleMesh.unit_square()

    def tangent(points):
        x, y = points.T
        return np.column_stack((x * (1 - x) * (1 - 2 * y), -(1 - 2 * x) * y * (1 - y)))

    robin = {int(face): 0.0 for face in mesh.boundary_faces}
    compatible = SkeletonSpace(mesh, tuple(FaceSpace.uniform(3) for _ in mesh.faces))
    solution = solve_transport(
        mesh,
        velocity=tangent,
        velocity_divergence=0.0,
        degree=3,
        skeleton=compatible,
        neumann=robin,
        mean_value=1.0,
        stabilization=stabilization,
    )
    assert solution.l2_error(1.0) < 2e-12
    with pytest.raises(ValueError, match="physical equations"):
        solve_transport(
            mesh,
            velocity=tangent,
            velocity_divergence=0.0,
            degree=3,
            neumann=robin,
            mean_value=1.0,
            stabilization=stabilization,
        )


def test_non_tangential_robin_advection_has_no_constant_gauge():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="gauge"):
        solve_transport(
            mesh,
            velocity=(1.0, 0.0),
            neumann={int(face): 0.0 for face in mesh.boundary_faces},
            mean_value=1.0,
        )
