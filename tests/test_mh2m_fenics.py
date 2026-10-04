"""Independent DOLFINx assembly of the local MH2M Neumann operators."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.fem.scalar.triangle import nodal_space
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.crisscross import crisscross_submesh
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.three_field import PressureTraceSpace, solve_mh2m

pytestmark = pytest.mark.fem


@pytest.mark.parametrize(
    ("degree", "polygonal", "crossed"),
    [(1, False, False), (2, False, False), (3, False, False), (2, True, False), (1, False, True)],
)
def test_variable_tensor_neumann_lifts_match_independent_ufl(degree, polygonal, crossed):
    """Compare native stiffness, load, boundary mean and a nonconstant Neumann lift."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = (
        PolygonMesh(
            np.array([[0.0, 0.0], [1, 0], [1, 0.4], [0.4, 0.4], [0.4, 1], [0, 1]]),
            (np.arange(6),),
        )
        if polygonal
        else TriangleMesh(np.array([[0.1, 0.2], [1.2, 0.1], [0.3, 0.9]]), np.array([[0, 1, 2]]))
    )

    def material(points):
        """Positive definite affine tensor with a nonzero off-diagonal component."""
        values = np.broadcast_to([[2.0, 0.25], [0.25, 3.0]], (len(points), 2, 2)).copy()
        values[:, 0, 0] += points[:, 0]
        values[:, 1, 1] += points[:, 1]
        return values

    result = solve_mh2m(
        mesh,
        degree=degree,
        local_refinement=2,
        permeability=material,
        source=lambda x: 1 + x[:, 0] ** 2,
        pressure_trace=PressureTraceSpace.uniform(mesh),
        flux_space=SkeletonSpace(
            mesh, tuple(FaceSpace.uniform(int(degree > 1)) for _ in mesh.faces)
        ),
        quadrature_order=8,
        local_meshes=(crisscross_submesh(mesh, 0, 2),) if crossed else None,
    )
    local = result.local[0]
    coordinate = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF, local.mesh.cells, local.mesh.points, coordinate
    )
    element = basix.ufl.element(
        "Lagrange", "triangle", degree, lagrange_variant=basix.LagrangeVariant.equispaced
    )
    space = dolfinx.fem.functionspace(domain, element)
    native_coordinates = space.tabulate_dof_coordinates()[:, :2]
    _, coordinates = nodal_space(local.mesh, degree)
    ordering = np.argmin(
        np.linalg.norm(coordinates[:, None] - native_coordinates[None], axis=2), axis=1
    )
    assert len(np.unique(ordering)) == len(coordinates)
    assert_allclose(native_coordinates[ordering], coordinates, atol=1e-14)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(domain)
    tensor = ufl.as_matrix(((2 + x[0], 0.25), (0.25, 3 + x[1])))
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
    ds = ufl.Measure("ds", domain=domain, metadata={"quadrature_degree": 8})
    assembled = dolfinx.fem.assemble_matrix(
        dolfinx.fem.form(ufl.inner(tensor * ufl.grad(u), ufl.grad(v)) * dx)
    )
    assembled.scatter_reverse()
    stiffness = assembled.to_scipy().toarray()[np.ix_(ordering, ordering)]
    load = dolfinx.fem.assemble_vector(dolfinx.fem.form((1 + x[0] ** 2) * v * dx)).array[ordering]
    boundary = dolfinx.fem.assemble_vector(dolfinx.fem.form(v * ds)).array[ordering]
    assert_allclose(stiffness, local.stiffness.toarray(), atol=6e-13)
    assert_allclose(load, local.load, atol=3e-15)
    constants = np.concatenate([face.constant_coefficients() for face in result.flux_space.faces])
    assert_allclose(boundary, local.boundary_coupling @ constants, atol=3e-15)
    # Independent saddle with an integral boundary constraint, without pinned elimination.
    augmented = np.block([[stiffness, boundary[:, None]], [boundary[None, :], np.zeros((1, 1))]])
    lift = np.linalg.solve(augmented, np.r_[load, 0.0])[:-1]
    assert_allclose(lift, local.source_lift, atol=3e-14, rtol=3e-12)
