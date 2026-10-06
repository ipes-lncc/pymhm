"""Independent polynomial, conservation and native UFL checks for tetrahedral RAD."""

import numpy as np
import pytest

from pymhm._legacy.models.darcy.primal_3d import solve_darcy_3d
from pymhm._legacy.models.transport.rad_3d import (
    _boundary_tangent_3d,
    solve_rad_3d,
    solve_rad_3d_conforming,
    tetra_rad_operators,
)
from pymhm.fem.scalar.tetrahedron import (
    tetra_element_tabulate,
    tetra_nodal_space,
    tetrahedron_quadrature,
)
from pymhm.fem.scalar.tetrahedron_topology import (
    continuous_tetra_nodes,
    tetra_indices,
    tetra_polynomials,
)
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.materials.evaluation import vector_values_3d
from pymhm.meshes.tetrahedron import TetraMesh


def affine(points):
    """Return a nonzero affine field for all boundary orientations."""
    return 1 + points @ np.array([1.0, 2.0, 3.0])


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_cardinality_partition_hessian_and_topological_continuity(degree):
    indices = tetra_indices(degree)
    basis, _, _ = tetra_polynomials(degree, indices / degree)
    np.testing.assert_allclose(basis, np.eye(len(indices)), atol=3e-14)
    mesh = TetraMesh.unit_cube(1)
    dofs, nodes = continuous_tetra_nodes(mesh, degree)
    assert len(nodes) == (degree + 1) ** 3
    assert len(np.unique(nodes, axis=0)) == len(nodes)
    bary, _ = tetrahedron_quadrature(3)
    dofs, nodes, values, gradient, hessian = tetra_element_tabulate(mesh, degree, bary)
    np.testing.assert_allclose(values.sum(axis=1), 1, atol=5e-14)
    np.testing.assert_allclose(gradient.sum(axis=2), 0, atol=2e-13)
    np.testing.assert_allclose(hessian.sum(axis=2), 0, atol=2e-12)
    actual = np.einsum("ti,tqia->tqa", affine(nodes)[dofs], gradient)
    np.testing.assert_allclose(actual, np.broadcast_to([1, 2, 3], actual.shape), atol=2e-13)
    np.testing.assert_allclose(
        np.einsum("ti,tqiab->tqab", affine(nodes)[dofs], hessian), 0, atol=3e-12
    )
    if degree >= 2:
        coefficients = np.sum(nodes**2, axis=1)[dofs]
        actual = np.einsum("ti,tqiab->tqab", coefficients, hessian)
        np.testing.assert_allclose(actual, np.broadcast_to(2 * np.eye(3), actual.shape), atol=3e-12)


@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
def test_nonhomogeneous_affine_transport_reaction_patch(degree, stabilization):
    mesh = TetraMesh.unit_cube(1)
    skeleton = TriangularSkeleton(mesh, degree=1)
    solution = solve_rad_3d(
        mesh,
        degree=degree,
        local_refinement=2,
        skeleton=skeleton,
        velocity=[1, -0.5, 0.25],
        reaction=0.3,
        source=lambda p: 0.75 + 0.3 * affine(p),
        dirichlet=affine,
        stabilization=stabilization,
    )
    assert solution.l2_error(affine) < 2e-11
    assert solution.h1_seminorm_error([1, 2, 3]) < 2e-10
    assert (
        solution.flux_l2_error(
            lambda p: -np.array([1, 2, 3]) + np.array([1, -0.5, 0.25]) * affine(p)[:, None]
        )
        < 2e-10
    )
    assert solution.hybrid.residual < 1e-11


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_classical_affine_strong_boundary_and_fully_constrained(degree):
    solution = solve_rad_3d_conforming(
        TetraMesh.unit_cube(1), degree=degree, dirichlet=affine, velocity=[1, 0, 0], source=1.0
    )
    assert solution.l2_error(affine) < 3e-13
    assert (
        solution.flux_l2_error(
            lambda p: -np.array([1, 2, 3]) + np.array([1, 0, 0]) * affine(p)[:, None]
        )
        < 5e-12
    )
    assert solution.residual < 1e-12


def test_independent_face_degrees_partitions_and_robin_data():
    mesh = TetraMesh.unit_cube(1)
    degree = np.ones(len(mesh.faces), dtype=int)
    skeleton = TriangularSkeleton(mesh, subdivisions=2, degree=degree)
    assert skeleton.size == len(mesh.faces) * 12
    assert np.array_equal(skeleton.constant_coefficients, np.ones(skeleton.size))
    neumann = {int(face): float(-mesh.normals[face] @ [1, 2, 3]) for face in mesh.boundary_faces}
    solution = solve_darcy_3d(
        mesh, degree=3, local_refinement=2, skeleton=skeleton, neumann=neumann, mean_pressure=4.0
    )
    assert solution.l2_error(affine) < 3e-12
    np.testing.assert_allclose(solution.conservation_residuals(), 0, atol=1e-12)
    rad = solve_rad_3d(
        mesh, degree=3, local_refinement=2, skeleton=skeleton, neumann=neumann, mean_value=4.0
    )
    assert rad.l2_error(affine) < 3e-12
    with pytest.raises(ValueError, match="incompatible"):
        solve_rad_3d(mesh, degree=3, skeleton=skeleton, neumann=dict.fromkeys(neumann, 0), source=1)


def test_mixed_face_degrees_and_pure_robin_transport():
    """Exercise face-wise degree/refinement and the nongauged Robin transport solution."""
    mesh = TetraMesh.unit_cube(1)
    degrees = np.arange(len(mesh.faces)) % 2
    skeleton = TriangularSkeleton(mesh, subdivisions=1 + degrees, degree=degrees)
    solution = solve_darcy_3d(
        mesh, degree=3, local_refinement=2, skeleton=skeleton, dirichlet=affine
    )
    assert solution.l2_error(affine) < 3e-12
    robin = {int(face): float(mesh.normals[face, 0] / 2) for face in mesh.boundary_faces}
    transported = solve_rad_3d(
        mesh, degree=4, local_refinement=1, velocity=[1, 0, 0], neumann=robin
    )
    assert transported.l2_error(1.0) < 1e-11


@pytest.mark.parametrize("scale", [1e-8, 1e-16])
def test_retained_constants_preserve_vanishing_transport_and_reaction(scale):
    solution = solve_rad_3d(
        TetraMesh.unit_cube(1),
        degree=4,
        local_refinement=1,
        velocity=[scale, 0, 0],
        reaction=scale,
        source=lambda p: scale * (1 + affine(p)),
        dirichlet=affine,
    )
    assert solution.l2_error(affine) < 2e-11


def test_variable_coefficients_full_conservative_residual():
    def diffusion(p):
        """Return a positive variable scalar conductivity with known divergence."""
        return 1 + p[:, 0]

    def velocity(p):
        """Return a variable field of divergence three."""
        return np.column_stack((1 + p[:, 0], p[:, 1], p[:, 2]))

    def force(p):
        """Apply the conservative strong operator to the affine field."""
        return -1 + velocity(p) @ np.array([1, 2, 3]) + 3.5 * affine(p)

    solution = solve_rad_3d(
        TetraMesh.unit_cube(1),
        degree=4,
        local_refinement=1,
        diffusion=diffusion,
        diffusion_divergence=[1, 0, 0],
        velocity=velocity,
        velocity_divergence=3.0,
        reaction=0.5,
        source=force,
        dirichlet=affine,
        stabilization="supg",
    )
    assert solution.l2_error(affine) < 2e-11
    assert solution.h1_seminorm_error([1, 2, 3]) < 1e-10


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize(
    ("coefficients", "message"),
    [
        ({"velocity": lambda points: points}, "velocity_divergence"),
        (
            {"diffusion": lambda points: 1 + points[:, 0], "stabilization": "supg"},
            "diffusion_divergence",
        ),
        ({"velocity_divergence": 1.0}, "constant velocity"),
        ({"diffusion_divergence": [1.0, 0.0, 0.0]}, "constant diffusion"),
    ],
)
def test_classical_and_local_derivative_contracts(classical, coefficients, message):
    """Require explicit consistent coefficient derivatives in every assembly entry point."""
    mesh = TetraMesh.unit_cube(1)
    assemble = solve_rad_3d_conforming if classical else tetra_rad_operators
    with pytest.raises(ValueError, match=message):
        assemble(mesh, degree=2, **coefficients)


def test_classical_variable_coefficients_with_declared_derivatives():
    """Check a nonconstant conservative SUPG operator against an exact affine field."""
    solution = solve_rad_3d_conforming(
        TetraMesh.unit_cube(1),
        degree=2,
        diffusion=lambda points: 1 + points[:, 0],
        diffusion_divergence=[1, 0, 0],
        velocity=lambda points: points,
        velocity_divergence=3.0,
        reaction=0.5,
        source=lambda points: -1 + points @ np.array([1, 2, 3]) + 3.5 * affine(points),
        dirichlet=affine,
        stabilization="supg",
    )
    assert solution.l2_error(affine) < 3e-13
    assert solution.h1_seminorm_error([1, 2, 3]) < 2e-12


def test_invalid_data_and_degree_contracts():
    mesh = TetraMesh.unit_cube(1)
    assert not _boundary_tangent_3d(TriangularSkeleton(mesh), [1, 0, 0], 3)
    for invalid in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            tetra_indices(invalid)
    for bary in ([1, 0, 0], [[1j, 0, 0, 0]], [[np.nan, 0, 0, 0]]):
        with pytest.raises(ValueError):
            tetra_polynomials(3, bary)
    with pytest.raises(ValueError, match="three components"):
        vector_values_3d([1, 2], mesh.points)
    with pytest.raises(ValueError, match="one integer"):
        TriangularSkeleton(mesh, degree=[0, 1])
    with pytest.raises(ValueError, match="degree"):
        TriangularSkeleton(mesh, degree=-1)
    with pytest.raises(ValueError, match="stabilization"):
        tetra_rad_operators(mesh, 1, stabilization="bad")
    with pytest.raises(ValueError, match="nonnegative"):
        tetra_rad_operators(mesh, 1, reaction=-1)
    options = [
        ({"velocity": lambda p: p}, "velocity_divergence"),
        ({"diffusion": lambda p: np.ones(len(p)), "stabilization": "supg"}, "diffusion_divergence"),
        ({"velocity_divergence": 1}, "constant velocity"),
        ({"diffusion_divergence": [1, 0, 0]}, "constant diffusion"),
        ({"mean_value": np.nan}, "finite"),
        ({"mean_value": 1}, "all-Robin"),
        ({"skeleton": TriangularSkeleton(TetraMesh.unit_cube(1))}, "belong"),
        ({"skeleton": TriangularSkeleton(mesh, 4)}, "resolve"),
    ]
    for kwargs, message in options:
        with pytest.raises(ValueError, match=message):
            solve_rad_3d(mesh, **kwargs)
    solution = solve_rad_3d_conforming(mesh, degree=1)
    with pytest.raises(ValueError, match="cell outside"):
        solution.evaluate(1, np.full((1, 4), 0.25))


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_factory_parallel_fields_match_serial(backend):
    kwargs = dict(degree=4, local_refinement=1, source=1.0, velocity=[1, 0, 0])
    mesh = TetraMesh.unit_cube(1)
    reference = solve_rad_3d(mesh, **kwargs)
    actual = solve_rad_3d(mesh, backend=backend, workers=2, **kwargs)
    for first, second in zip(reference.values, actual.values, strict=True):
        np.testing.assert_allclose(first, second, atol=2e-12)


@pytest.mark.fem
@pytest.mark.parametrize("degree", [3, 4])
@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
def test_native_dolfinx_tetrahedral_rad_operator(degree, stabilization):
    """Compare independent Basix/UFL high-order matrix and complete SUPG load."""
    dolfinx = pytest.importorskip("dolfinx")
    ufl = pytest.importorskip("ufl")
    basix = pytest.importorskip("basix.ufl")
    mpi = pytest.importorskip("mpi4py.MPI")
    from scipy.spatial import cKDTree

    mesh = TetraMesh(
        [[0.1, 0.2, 0.3], [1.3, 0.1, 0.2], [0.2, 1.1, 0.4], [0.3, 0.1, 1.4]], [[0, 1, 2, 3]]
    )
    domain = dolfinx.mesh.create_mesh(
        mpi.COMM_SELF,
        mesh.cells,
        x=mesh.points,
        e=ufl.Mesh(basix.element("Lagrange", "tetrahedron", 1, shape=(3,))),
    )
    space = dolfinx.fem.functionspace(
        domain,
        basix.element(
            "Lagrange",
            "tetrahedron",
            degree,
            lagrange_variant=__import__("basix").LagrangeVariant.equispaced,
        ),
    )
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(domain)
    K = ufl.as_matrix([[2.0, 0.1, 0], [0.1, 3.0, 0.2], [0, 0.2, 4.0]])
    tensor = np.array([[2.0, 0.1, 0], [0.1, 3.0, 0.2], [0, 0.2, 4.0]])
    beta = ufl.as_vector([1.0, -0.5, 0.25])
    c = 0.3
    f = 1 + x[0] + 2 * x[1] - x[2]
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * degree + 2})
    form = (
        ufl.inner(K * ufl.grad(u), ufl.grad(v))
        + 0.5 * (ufl.dot(beta, ufl.grad(u)) * v - u * ufl.dot(beta, ufl.grad(v)))
        + c * u * v
    ) * dx
    rhs = f * v * dx
    if stabilization == "supg":
        diameter = max(
            np.linalg.norm(first - second) for first in mesh.points for second in mesh.points
        )
        tau = 1 / np.sqrt(
            (2 * np.linalg.norm([1.0, -0.5, 0.25]) / diameter) ** 2
            + (4 * np.linalg.eigvalsh(tensor)[-1] / diameter**2) ** 2
            + c**2
        )
        strong = -ufl.div(K * ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) + c * u
        form += tau * strong * ufl.dot(beta, ufl.grad(v)) * dx
        rhs += tau * f * ufl.dot(beta, ufl.grad(v)) * dx
    assembled = dolfinx.fem.assemble_matrix(dolfinx.fem.form(form))
    assembled.scatter_reverse()
    load = dolfinx.fem.assemble_vector(dolfinx.fem.form(rhs)).array
    matrix, force, _, _ = tetra_rad_operators(
        mesh,
        degree,
        diffusion=tensor,
        velocity=[1.0, -0.5, 0.25],
        reaction=c,
        source=lambda p: 1 + p[:, 0] + 2 * p[:, 1] - p[:, 2],
        stabilization=stabilization,
    )
    nodes = tetra_nodal_space(mesh, degree)[1]
    distance, order = cKDTree(space.tabulate_dof_coordinates()).query(nodes)
    assert distance.max() < 2e-13
    np.testing.assert_allclose(
        matrix.toarray(),
        assembled.to_scipy().toarray()[np.ix_(order, order)],
        atol=1e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(force, load[order], atol=2e-14)
