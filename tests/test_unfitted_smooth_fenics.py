"""Native global verification of the smooth 16-macro crisscross trace study."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from examples.unfitted_convergence import error_norms, smooth_field, smooth_source
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.backends.fenics import from_ufl
from pymhm.fem.scalar.triangle import nodal_space, scalar_operators, trace_coupling
from pymhm.linalg.linear import solve_linear


def _compare(trace_degree: int, segments_count: int) -> tuple[dict, np.ndarray]:
    """Assemble P6/r2 native locals and solve their uncondensed full saddle system."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    points = [[x / 2, y / 2] for y in range(3) for x in range(3)]
    triangles = []
    for j in range(2):
        for i in range(2):
            vertices = (3 * j + i, 3 * j + i + 1, 3 * (j + 1) + i + 1, 3 * (j + 1) + i)
            center = len(points)
            points.append([(i + 0.5) / 2, (j + 0.5) / 2])
            triangles.extend([vertices[k], vertices[(k + 1) % 4], center] for k in range(4))
    macro = TriangleMesh(points, triangles)
    skeleton = SkeletonSpace(
        macro, tuple(FaceSpace.uniform(trace_degree, segments_count) for _ in macro.faces)
    )
    locals_ = tuple(macro.submesh(cell, 2) for cell in range(len(macro.cells)))
    matrices, loads, couplings, permutations = [], [], [], []
    for cell, fine in enumerate(locals_):
        domain = dolfinx.mesh.create_mesh(
            MPI.COMM_SELF,
            fine.cells,
            fine.points,
            ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))),
        )
        space = dolfinx.fem.functionspace(
            domain,
            basix.ufl.element(
                "Lagrange", "triangle", 6, lagrange_variant=basix.LagrangeVariant.equispaced
            ),
        )
        x = ufl.SpatialCoordinate(domain)
        u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
        forcing = 8 * np.pi**2 * ufl.sin(2 * np.pi * x[0]) * ufl.sin(2 * np.pi * x[1])
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 28})
        a, load = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx, forcing * v * dx
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
            for polynomial in (1, coordinate)[: trace_degree + 1]
        ]
        native = from_ufl(a, load, traces, skeleton.cell_dofs(cell))
        _, nodes = nodal_space(fine, 6)
        distance, permutation = cKDTree(nodes).query(space.tabulate_dof_coordinates()[:, :2])
        assert distance.max() < 3e-14
        assert len(np.unique(permutation)) == len(nodes)
        matrix, _, force = scalar_operators(fine, 6, source=smooth_source, order=16)
        trace = trace_coupling(macro, cell, fine, skeleton, 6)
        dofs = skeleton.cell_dofs(cell)
        assert_allclose(
            native.matrix.toarray(),
            matrix.toarray()[permutation][:, permutation],
            rtol=3e-11,
            atol=3e-11,
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
    rhs = np.r_[np.concatenate(loads), np.zeros(skeleton.size)]
    fields = solve_linear(full, rhs, solver="petsc")
    solution = solve_darcy(
        macro,
        degree=6,
        source=smooth_source,
        local_meshes=locals_,
        skeleton=skeleton,
        quadrature_order=16,
    )
    start = 0
    maximum_pressure_difference = 0.0
    for values, permutation in zip(solution.pressure, permutations, strict=True):
        delta = fields[start : start + len(values)] - values[permutation]
        maximum_pressure_difference = max(maximum_pressure_difference, float(np.max(abs(delta))))
        assert_allclose(fields[start : start + len(values)], values[permutation], atol=2e-11)
        start += len(values)
    assert_allclose(fields[start:], solution.hybrid.trace, atol=2e-10)
    assert_allclose(solution.conservation_residuals(), 0, atol=3e-12)
    return {
        "macro_cells": len(macro.cells),
        "local_degree": 6,
        "local_refinement": 2,
        "trace_degree": trace_degree,
        "trace_segments": segments_count,
        "native_full_dofs": full.shape[0],
        "maximum_pressure_coefficient_difference": maximum_pressure_difference,
        "maximum_trace_coefficient_difference": float(
            np.max(abs(fields[start:] - solution.hybrid.trace))
        ),
        "native_original_relative_residual": float(
            np.linalg.norm(full @ fields - rhs) / np.linalg.norm(rhs)
        ),
        "physical_norms": error_norms(solution, smooth_field, 16),
    }, fields[:start]


@pytest.mark.fem
@pytest.mark.parametrize("segments", [1, 2])
def test_smooth_p6_full_native_saddle_and_trace_symmetry(segments):
    """Confirm all sixteen macro couplings, forcing and the unrefined P0/P1 plateau."""
    with threadpool_limits(1):
        constant, first = _compare(0, segments)
        linear, second = _compare(1, segments)
    assert constant["native_original_relative_residual"] < 1e-10
    assert linear["native_original_relative_residual"] < 1e-10
    if segments == 1:
        assert_allclose(first, second, rtol=0, atol=2e-11)
    else:
        assert (
            linear["physical_norms"]["gradient_absolute"]
            < constant["physical_norms"]["gradient_absolute"] / 4
        )
