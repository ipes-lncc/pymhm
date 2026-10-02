"""Scalar UNUSUAL residual signs, inverse bounds and physical boundary patches."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_transport
from pymhm.rad import _rad_local
from pymhm.reservoir import CartesianCellField
from pymhm.unusual import UnusualParameters


@pytest.fixture(autouse=True)
def single_thread():
    """Avoid threaded BLAS startup in small polynomial element tests."""
    with threadpool_limits(1):
        yield


@pytest.mark.parametrize("reaction", [0.0, 1.0, 1000.0])
def test_p1_matrix_and_load_equal_closed_form_moments(reaction):
    """For P1 and constant A, the stabilization reduces the reaction mass exactly."""
    mesh = TriangleMesh(np.array([[0.0, 0], [2, 0], [0, 1]]), np.array([[0, 1, 2]]))
    assembly = _rad_local(
        0,
        mesh=mesh,
        skeleton=SkeletonSpace(mesh),
        degree=1,
        refinement=1,
        diffusion=2.0,
        diffusion_divergence=(0.0, 0.0),
        velocity=(0.0, 0.0),
        velocity_divergence=0.0,
        reaction=reaction,
        source=3.0,
        stabilization="unusual",
        order=4,
    )
    nodes = assembly.metadata[0].points
    gradient = np.linalg.inv(np.column_stack((np.ones(3), nodes)))[1:].T
    mass = (np.ones((3, 3)) + np.eye(3)) / 12
    tau = (5 / 3) / (max(reaction * 5 / 3, 4) + 4)
    expected = 2 * gradient @ gradient.T + (reaction - tau * reaction**2) * mass
    assert_allclose(assembly.problem.matrix.toarray(), expected, atol=3e-14, rtol=3e-13)
    assert_allclose(assembly.problem.load, np.full(3, 1 - tau * reaction), atol=4e-15)
    if reaction:
        assert np.linalg.eigvalsh(expected).min() > 0
    else:
        assert_allclose(expected @ np.ones(3), 0, atol=1e-15)


def exact(points):
    """A nonaffine polynomial with nonzero boundary displacement and flux."""
    x, y = points.T
    return 1 + x + 2 * y + x * y + y * y


def gradient(points):
    """Its independently differentiated gradient."""
    x, y = points.T
    return np.column_stack((1 + y, 2 + x + 2 * y))


TENSOR = np.array([[2.0, 0.3], [0.3, 1.0]])


def material(points):
    """A variable tensor with off-diagonal entries and nonzero divergence."""
    return (1 + points.sum(axis=1))[:, None, None] * TENSOR


def reaction(points):
    """A positive affine reaction with global upper bound four."""
    return 3 + points[:, 0]


def source(points):
    """Evaluate c*u-div(A grad u) from analytic derivatives, independently of assembly."""
    return (
        reaction(points) * exact(points)
        - 2.6 * (1 + points.sum(axis=1))
        - gradient(points) @ (TENSOR @ np.ones(2))
    )


@pytest.mark.parametrize("enforcement", ["weak", "strong"])
@pytest.mark.parametrize("degree", [2, 3])
def test_variable_tensor_quadratic_patch_with_mixed_boundaries(degree, enforcement):
    """Complete coefficient derivatives and negative residual load preserve exactness."""
    mesh = TriangleMesh.unit_square()
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
    right = next(f for f in mesh.boundary_faces if np.allclose(mesh.points[mesh.faces[f], 0], 1))
    result = solve_transport(
        mesh,
        degree=degree,
        local_refinement=3,
        skeleton=space,
        diffusion=material,
        diffusion_divergence=TENSOR @ np.ones(2),
        reaction=reaction,
        source=source,
        dirichlet=exact,
        dirichlet_enforcement=enforcement,
        neumann={
            int(right): lambda x: (
                -np.einsum("qij,qj->qi", material(x), gradient(x)) @ mesh.normals[right]
            )
        },
        stabilization="unusual",
        unusual_parameters=UnusualParameters(diffusion_lower=0.8, reaction_upper=4.0),
        quadrature_order=6,
    )
    assert result.l2_error(exact, order=6) < 2e-11
    assert result.hybrid.residual < 1e-11


def test_poisson_neumann_limit_keeps_physical_mean_and_flux():
    """The zero-reaction limit retains the exact null mode without a parameter floor."""
    mesh = TriangleMesh.unit_square()
    result = solve_transport(
        mesh,
        degree=2,
        stabilization="unusual",
        neumann={int(f): -mesh.normals[f] @ [1.0, 2.0] for f in mesh.boundary_faces},
        mean_value=2.5,
    )
    assert result.l2_error(lambda x: 1 + x[:, 0] + 2 * x[:, 1]) < 3e-12


def test_unusual_does_not_collapse_to_supg_without_advection():
    """Reaction stabilization remains active when the streamline term vanishes."""
    mesh = TriangleMesh.unit_square()
    options = dict(reaction=1000.0, source=1.0, local_refinement=4)
    galerkin = solve_transport(mesh, **options)
    supg = solve_transport(mesh, stabilization="supg", **options)
    unusual = solve_transport(mesh, stabilization="unusual", **options)
    for first, second in zip(galerkin.values, supg.values, strict=True):
        assert_allclose(first, second, atol=0, rtol=0)
    assert (
        max(np.max(abs(a - b)) for a, b in zip(galerkin.values, unusual.values, strict=True)) > 1e-5
    )


def test_cartesian_material_requires_resolved_interfaces():
    """Strong residuals require elementwise regular coefficients, not quadrature cuts alone."""
    mesh = TriangleMesh.unit_square()
    coefficient = CartesianCellField(np.array([[1.0], [10.0]]), (0.5, 1.0))
    with pytest.raises(ValueError, match="interfaces aligned"):
        solve_transport(mesh, stabilization="unusual", diffusion=coefficient, local_refinement=1)
    result = solve_transport(
        mesh,
        stabilization="unusual",
        diffusion=coefficient,
        local_refinement=2,
        reaction=3.0,
        source=3.0,
        dirichlet=1.0,
    )
    assert result.l2_error(1.0) < 2e-12


@pytest.mark.parametrize("value", [0.0, -1.0, 0.34, np.nan, np.inf])
def test_invalid_inverse_constant_is_rejected(value):
    """An arbitrary inverse parameter cannot bypass the admissibility contract."""
    with pytest.raises(ValueError, match="inverse_constant"):
        UnusualParameters(inverse_constant=value)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"velocity": (1.0, 0.0)}, "zero advection"),
        ({"velocity_divergence": 1.0}, "zero advection"),
        ({"velocity_divergence": lambda x: np.zeros(len(x))}, "zero advection"),
        ({"velocity": lambda x: 0 * x}, "zero advection"),
        ({"diffusion": material}, "diffusion_lower"),
        ({"reaction": reaction}, "reaction_upper"),
        ({"unusual_parameters": {}}, "UnusualParameters"),
        ({"unusual_parameters": UnusualParameters(2.0)}, "diffusion_lower"),
        ({"unusual_parameters": UnusualParameters(0.0)}, "diffusion_lower"),
        ({"unusual_parameters": UnusualParameters(reaction_upper=-1.0)}, "reaction_upper"),
        (
            {"reaction": 2.0, "unusual_parameters": UnusualParameters(reaction_upper=1.0)},
            "reaction_upper",
        ),
        (
            {"degree": 2, "unusual_parameters": UnusualParameters(inverse_constant=1 / 3)},
            "coercivity bound",
        ),
        (
            {"diffusion": material, "unusual_parameters": UnusualParameters(0.8)},
            "diffusion_divergence",
        ),
    ],
)
def test_invalid_stabilization_inputs_fail_before_a_false_solution(options, match):
    """Bounds, material derivatives and the zero-advection model are explicit contracts."""
    with pytest.raises(ValueError, match=match):
        solve_transport(TriangleMesh.unit_square(), stabilization="unusual", **options)


def test_optional_parameters_cannot_be_silently_ignored():
    """UNUSUAL-specific input cannot modify or masquerade as Galerkin/SUPG."""
    with pytest.raises(ValueError, match="requires UNUSUAL"):
        solve_transport(TriangleMesh.unit_square(), unusual_parameters=UnusualParameters())
    result = solve_transport(
        TriangleMesh.unit_square(),
        stabilization="unusual",
        source=1.0,
        reaction=1.0,
        unusual_parameters=UnusualParameters(
            diffusion_lower=lambda x: np.ones(len(x)),
            reaction_upper=lambda x: np.ones(len(x)),
            inverse_constant=1 / 6,
        ),
    )
    assert result.hybrid.residual < 1e-12


@pytest.mark.parametrize("reaction", [0.0, 1e-16, 1.0, 1000.0])
def test_explicit_parameter_has_positive_element_energy(reaction):
    """An explicit stable m may exceed the conservative automatic inverse bound."""
    from scipy.linalg import eigvalsh

    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh)
    assembly = _rad_local(
        0,
        mesh=mesh,
        skeleton=skeleton,
        degree=2,
        refinement=1,
        diffusion=1.0,
        diffusion_divergence=(0.0, 0.0),
        velocity=(0.0, 0.0),
        velocity_divergence=0.0,
        reaction=reaction,
        source=0.0,
        stabilization="unusual",
        order=8,
        unusual_parameters=UnusualParameters(inverse_constant=1 / 48),
    )
    matrix = assembly.problem.matrix.toarray()
    eigenvalues = eigvalsh(matrix)
    assert eigenvalues[1] > 0.05
    assert eigenvalues[0] >= -1e-14
