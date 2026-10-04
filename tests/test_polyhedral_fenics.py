"""Independent native Basix/UFL integration of original polygonal macroface moments."""

import numpy as np
import pytest

from pymhm._legacy.models.transport.polyhedral import PolygonalSkeleton3D, polygonal_trace_coupling
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.polyhedral import PolyhedralMesh


@pytest.mark.fem
@pytest.mark.parametrize("degree", [3, 4])
@pytest.mark.parametrize("family", ["cube", "shared", "hexagon", "coplanar"])
def test_native_polygonal_face_coupling(degree: int, family: str) -> None:
    """Match signed affine face loads, without splitting one polygon into independent traces."""
    dolfinx = pytest.importorskip("dolfinx")
    ufl = pytest.importorskip("ufl")
    basix_ufl = pytest.importorskip("basix.ufl")
    basix = pytest.importorskip("basix")
    mpi = pytest.importorskip("mpi4py.MPI")
    from scipy.spatial import Delaunay, cKDTree

    if family == "cube":
        mesh = PolyhedralMesh.cubes()
        cell = 0
    elif family == "shared":
        mesh = PolyhedralMesh.cubes(2)
        cell = 1
    elif family == "hexagon":
        angles = np.arange(6) * np.pi / 3
        base = PolygonMesh(np.column_stack((np.cos(angles), np.sin(angles))), (np.arange(6),))
        mesh = PolyhedralMesh.extrude(base)
        cell = 0
    else:
        base = PolygonMesh(
            np.array([[0, 0], [0.5, 0], [1, 0], [0, 1], [0.5, 1], [1, 1]]),
            (np.array([0, 1, 4, 3]), np.array([1, 2, 5, 4])),
        )
        divided = PolyhedralMesh.extrude(base)
        exterior = [divided.faces[f] for f in divided.boundary_faces]
        mesh = PolyhedralMesh(divided.points, exterior, [np.arange(len(exterior))])
        cell = 0
    fine = mesh.submesh(cell, 1)
    domain = dolfinx.mesh.create_mesh(
        mpi.COMM_SELF,
        fine.cells,
        fine.points,
        ufl.Mesh(basix_ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
    )
    space = dolfinx.fem.functionspace(
        domain,
        basix_ufl.element(
            "Lagrange", "tetrahedron", degree, lagrange_variant=basix.LagrangeVariant.equispaced
        ),
    )
    indices = []
    labels = []
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        domain, 2, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    midpoints = dolfinx.mesh.compute_midpoints(domain, 2, boundary_facets)
    for side, face in enumerate(mesh.cell_faces[cell]):
        normal = mesh.normals[face]
        origin = mesh.face_origins[face]
        tangents = mesh.face_tangents[face]
        polygon = (mesh.points[mesh.faces[face]] - origin) @ tangents.T
        inside = Delaunay(polygon).find_simplex((midpoints - origin) @ tangents.T) >= 0
        facets = boundary_facets[inside & (np.abs((midpoints - origin) @ normal) < 1e-12)]
        indices.extend(facets)
        labels.extend([side] * len(facets))
    sort = np.argsort(indices)
    tags = dolfinx.mesh.meshtags(
        domain,
        2,
        np.asarray(indices, dtype=np.int32)[sort],
        np.asarray(labels, dtype=np.int32)[sort],
    )
    ds = ufl.Measure(
        "ds", domain=domain, subdomain_data=tags, metadata={"quadrature_degree": degree + 2}
    )
    x = ufl.SpatialCoordinate(domain)
    v = ufl.TestFunction(space)
    coefficients = np.array(
        [[1 + 0.1 * side, -0.2 + 0.04 * side, 0.3] for side in range(len(mesh.cell_faces[cell]))]
    )
    form = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        local = [
            sum(
                (x[a] - mesh.face_origins[face, a]) * mesh.face_tangents[face, b, a]
                for a in range(3)
            )
            / np.sqrt(mesh.areas[face])
            for b in range(2)
        ]
        data = coefficients[side, 0] + sum(coefficients[side, b + 1] * local[b] for b in range(2))
        form += mesh.signs[cell][side] * data * v * ds(side)
    expected = dolfinx.fem.assemble_vector(dolfinx.fem.form(form)).array
    nodes = tetra_nodal_space(fine, degree)[1]
    distance, order = cKDTree(space.tabulate_dof_coordinates()).query(nodes)
    assert distance.max() < 3e-13
    skeleton = PolygonalSkeleton3D(mesh)
    actual = polygonal_trace_coupling(mesh, cell, fine, skeleton, degree) @ coefficients.ravel()
    np.testing.assert_allclose(actual, expected[order], atol=2e-13, rtol=5e-12)
