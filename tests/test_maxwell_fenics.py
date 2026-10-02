"""Independent native UFL central-DG Maxwell volume, facet and material operators."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm.darcy3d import TriangularSkeleton
from pymhm.lagrange import multiindices
from pymhm.maxwell_dg import MaxwellSkeleton, assemble_local, derivatives, physical_points
from pymhm.mesh import SkeletonSpace, TriangleMesh
from pymhm.quadrilateral import CartesianMacroMesh
from pymhm.tetra_lagrange import tetra_indices
from pymhm.tetrahedral import TetraMesh

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("kind", ["triangle", "quadrilateral", "tetrahedron"])
@pytest.mark.parametrize("degree", [1, 2])
def test_native_dg_curl_and_tensor_masses(kind, degree):
    """Independently assemble central jumps/averages and variable SPD mass tensors."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix
    import basix.ufl
    import ufl
    from mpi4py import MPI

    dimension = 3 if kind == "tetrahedron" else 2
    rectangle = kind == "quadrilateral"
    macro = (
        TetraMesh.unit_cube()
        if dimension == 3
        else CartesianMacroMesh(1)
        if rectangle
        else TriangleMesh.unit_square(1)
    )
    fine = macro.submesh(0, 2)
    base = SkeletonSpace(macro) if dimension == 2 else TriangularSkeleton(macro)

    def material(points):
        """Polynomial SPD tensor with nonzero symmetric off-diagonal components."""
        tensor = np.broadcast_to(
            np.eye(dimension) * 3 + 0.2, (len(points), dimension, dimension)
        ).copy()
        tensor[:, 0, 0] += points[:, 0]
        return tensor

    local = assemble_local(
        fine,
        MaxwellSkeleton(base),
        0,
        degree,
        (lambda p: 2 + p[:, 0]) if dimension == 2 else material,
        material,
        5,
    )
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        fine.cells[:, [0, 1, 3, 2]] if rectangle else fine.cells,
        fine.points,
        ufl.Mesh(basix.ufl.element("Lagrange", kind, 1, shape=(dimension,))),
    )
    space = dolfinx.fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange",
            kind,
            degree,
            discontinuous=True,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        ),
    )
    reference = (
        np.array(
            [(x, y) for y in np.linspace(0, 1, degree + 1) for x in np.linspace(0, 1, degree + 1)]
        )
        if rectangle
        else (multiindices(degree) if dimension == 2 else tetra_indices(degree)) / degree
    )
    own_points = physical_points(fine, reference)
    native_points = space.tabulate_dof_coordinates()[:, :dimension]
    native_vertices = domain.geometry.x[domain.geometry.dofmap][:, :, :dimension]
    own_centers = fine.points[fine.cells].mean(axis=1)
    native_centers = native_vertices.mean(axis=1)
    permutation = []
    for cell, center in enumerate(own_centers):
        native_cell = int(np.argmin(np.linalg.norm(native_centers - center, axis=1)))
        dofs = space.dofmap.cell_dofs(native_cell)
        order = np.argmin(
            np.linalg.norm(own_points[cell, :, None] - native_points[dofs][None], axis=2), axis=1
        )
        permutation.extend(dofs[order])
    permutation = np.asarray(permutation)
    assert len(np.unique(permutation)) == len(permutation)
    assert_allclose(native_points[permutation], own_points.reshape(-1, dimension), atol=1e-14)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    x, normal = ufl.SpatialCoordinate(domain), ufl.FacetNormal(domain)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 10})
    ds = ufl.Measure("dS", domain=domain, metadata={"quadrature_degree": 10})

    def matrix(form):
        """Assemble the native operator and permute by cell identity and physical nodes."""
        value = dolfinx.fem.assemble_matrix(dolfinx.fem.form(form))
        value.scatter_reverse()
        return value.to_scipy().toarray()[np.ix_(permutation, permutation)]

    native_derivatives = []
    for axis, actual in enumerate(derivatives(fine, degree, 5)):
        form = v * u.dx(axis) * dx
        form += (ufl.avg(u) * ufl.jump(v, normal)[axis] - ufl.jump(u * v, normal)[axis]) * ds
        expected = matrix(form)
        native_derivatives.append(expected)
        assert_allclose(actual.toarray(), expected, atol=5e-15)
    scalar_size = len(permutation)
    native_mass = np.zeros((scalar_size * dimension,) * 2)
    for a in range(dimension):
        for b in range(dimension):
            coefficient = (3 if a == b else 0) + 0.2 + (x[0] if a == b == 0 else 0)
            native_mass[a::dimension, b::dimension] = matrix(coefficient * u * v * dx)
    assert_allclose(local.magnetic_mass.toarray(), native_mass, atol=4e-15)
    expected_electric = matrix((2 + x[0]) * u * v * dx) if dimension == 2 else native_mass
    assert_allclose(local.electric_mass.toarray(), expected_electric, atol=4e-15)
    if dimension == 2:
        expected_curl = sparse.kron(-native_derivatives[1], [[1], [0]]) + sparse.kron(
            native_derivatives[0], [[0], [1]]
        )
    else:
        expected_curl = sparse.csr_matrix((scalar_size * 3,) * 2)
        for axis, derivative in enumerate(native_derivatives):
            expected_curl -= sparse.kron(derivative, np.cross(np.eye(3)[axis], np.eye(3)).T)
    assert_allclose(local.curl.toarray(), expected_curl.toarray(), atol=5e-15)
