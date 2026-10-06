"""Native UFL verification of material-fitted P4 locals and segmented P2 traces."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.spatial import cKDTree

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.backends.fenics import from_ufl
from pymhm.fem.quadrature.material import fit_material_faces, fit_material_mesh
from pymhm.fem.scalar.triangle import nodal_space, scalar_operators, trace_coupling
from pymhm.linalg.linear import solve_linear
from pymhm.materials.cartesian import CartesianCellField


@pytest.mark.fem
@pytest.mark.parametrize("split", [False, True])
def test_p4_cut_material_segmented_p2_matches_full_ufl_saddle(split):
    """Match complete native matrices, sources, trace loads and uncondensed fields."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    macro = TriangleMesh.unit_square()
    material = CartesianCellField(np.array([[10.0, 1.0]]), (1.0, 0.5))
    skeleton = SkeletonSpace(macro, tuple(FaceSpace.uniform(2) for _ in macro.faces))
    if split:
        skeleton = fit_material_faces(skeleton, material)
    locals_ = tuple(fit_material_mesh(macro.submesh(cell, 2), material) for cell in range(2))
    matrices, loads, couplings, permutations = [], [], [], []
    for cell, fine in enumerate(locals_):
        domain = dolfinx.mesh.create_mesh(
            MPI.COMM_SELF,
            fine.cells,
            x=fine.points,
            e=ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))),
        )
        space = dolfinx.fem.functionspace(
            domain,
            basix.ufl.element(
                "Lagrange", "triangle", 4, lagrange_variant=basix.LagrangeVariant.equispaced
            ),
        )
        x = ufl.SpatialCoordinate(domain)
        u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
        coefficient = ufl.conditional(ufl.lt(x[1], 0.5), 10.0, 1.0)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 12})
        a, load = coefficient * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx, v * dx
        entities, labels, segments = [], [], []
        for side, face in enumerate(macro.cell_faces[cell]):
            start, end = macro.points[macro.faces[face]]
            tangent = end - start
            for left, right in zip(
                skeleton.faces[face].breaks[:-1], skeleton.faces[face].breaks[1:], strict=True
            ):
                tag = len(segments) + 1

                def on_segment(points, start=start, tangent=tangent, left=left, right=right):
                    """Identify native boundary facets by their physical supporting segment."""
                    relative = points[:2].T - start
                    parameter = relative @ tangent / (tangent @ tangent)
                    return (
                        (abs(relative[:, 0] * tangent[1] - relative[:, 1] * tangent[0]) < 1e-12)
                        & (parameter >= left - 1e-12)
                        & (parameter <= right + 1e-12)
                    )

                facets = dolfinx.mesh.locate_entities_boundary(domain, 1, on_segment)
                entities.extend(facets)
                labels.extend([tag] * len(facets))
                parameter = ufl.dot(x - ufl.as_vector(start), ufl.as_vector(tangent)) / float(
                    tangent @ tangent
                )
                coordinate = 2 * (parameter - left) / (right - left) - 1
                segments.append((tag, float(macro.signs[cell, side]), coordinate))
        ordering = np.argsort(entities)
        tags = dolfinx.mesh.meshtags(
            domain,
            1,
            np.asarray(entities, np.int32)[ordering],
            np.asarray(labels, np.int32)[ordering],
        )
        ds = ufl.Measure(
            "ds", domain=domain, subdomain_data=tags, metadata={"quadrature_degree": 12}
        )
        traces = [
            sign * polynomial * v * ds(tag)
            for tag, sign, coordinate in segments
            for polynomial in (1, coordinate, (3 * coordinate**2 - 1) / 2)
        ]
        native = from_ufl(a, load, traces, skeleton.cell_dofs(cell))
        _, nodes = nodal_space(fine, 4)
        distance, permutation = cKDTree(nodes).query(space.tabulate_dof_coordinates()[:, :2])
        assert distance.max() < 3e-14
        assert len(np.unique(permutation)) == len(nodes)
        matrix, _, force = scalar_operators(fine, 4, diffusion=material, source=1, order=7)
        trace = trace_coupling(macro, cell, fine, skeleton, 4)
        dofs = skeleton.cell_dofs(cell)
        assert_allclose(
            native.matrix.toarray(),
            matrix.toarray()[permutation][:, permutation],
            rtol=2e-12,
            atol=2e-12,
        )
        assert_allclose(native.load, force[permutation], rtol=2e-12, atol=2e-14)
        assert_allclose(native.coupling, trace[permutation], rtol=2e-12, atol=2e-14)
        columns = np.zeros((len(nodes), skeleton.size))
        columns[:, dofs] = native.coupling
        matrices.append(native.matrix)
        loads.append(native.load)
        couplings.append(sparse.csr_matrix(columns))
        permutations.append(permutation)
    matrix, coupling = sparse.block_diag(matrices, format="csr"), sparse.vstack(couplings)
    full = sparse.bmat([[matrix, coupling], [coupling.T, None]], format="csr")
    fields = solve_linear(full, np.r_[np.concatenate(loads), np.zeros(skeleton.size)])
    solution = solve_darcy(
        macro,
        degree=4,
        permeability=material,
        source=1,
        local_meshes=locals_,
        skeleton=skeleton,
        quadrature_order=7,
    )
    start = 0
    for values, permutation in zip(solution.pressure, permutations, strict=True):
        assert_allclose(fields[start : start + len(values)], values[permutation], atol=3e-12)
        start += len(values)
    assert_allclose(fields[start:], solution.hybrid.trace, atol=3e-11)
    assert_allclose(solution.conservation_residuals(), 0, atol=3e-13)
