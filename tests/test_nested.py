"""Recursive condensation preserves physical fields, gauges and oriented traces."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, HybridSystem, LocalProblem, SkeletonSpace
from pymhm._legacy.models.darcy.cartesian import (
    _assemble_quad,
    _QuadTask,
    solve_darcy_quadrilateral,
)
from pymhm.core.nested import nest_hybrid_system, nested_trace_map
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.quadrilateral import qk_space
from pymhm.meshes.cartesian import CartesianMacroMesh


@pytest.mark.parametrize("natural", [False, True])
@pytest.mark.parametrize("source", [0.0, -4.0])
def test_recursive_darcy_matches_flat_discretization(natural, source):
    """Two MHM levels recover the identical broken Q2 fields and physical mean."""
    outer_mesh = CartesianMacroMesh(2)
    outer = SkeletonSpace(outer_mesh, tuple(FaceSpace.uniform(1, 2) for _ in outer_mesh.faces))
    nested, means, offsets = [], [], []
    for cell in range(4):
        inner_mesh = outer_mesh.submesh(cell, 2)
        inner = SkeletonSpace(inner_mesh, tuple(FaceSpace.uniform(1) for _ in inner_mesh.faces))
        system = HybridSystem.from_local_factory(
            _assemble_quad,
            [_QuadTask(inner_mesh, i, (2, 2), inner, 2, 1.0, source, 5) for i in range(4)],
        )
        modes = np.r_[np.zeros(inner.size), np.ones(4)][:, None]
        moments = [metadata[1] for metadata in system.local_metadata]
        row = system.mean_constraint(moments)[0][:, None]
        selected, mapping = nested_trace_map(outer, cell, inner)
        item = nest_hybrid_system(
            system, selected, mapping, outer.cell_dofs(cell), kernel=modes, constraints=row
        )
        mean, offset = item.moment(moments)
        nested.append(item)
        means.append(mean)
        offsets.append(offset)

    def exact(x):
        """Representable patch with nonzero boundary data and physical average."""
        return 1 + x[:, 0] + (x[:, 0] ** 2 + x[:, 1] ** 2 if source else 0)

    def outward(mesh):
        """Construct every outward normal derivative independently from the field."""
        return {
            int(f): lambda x, n=mesh.normals[f]: (
                -(np.array([1.0, 0.0]) + (2 * x if source else 0)) @ n
            )
            for f in mesh.boundary_faces
        }

    neumann = outward(outer_mesh) if natural else None
    boundary, fixed = boundary_data(outer, exact, neumann)
    parent = HybridSystem([item.problem for item in nested], boundary_load=boundary)
    physical_mean = 1.5 + (2 / 3 if source else 0)
    constraints = [parent.mean_constraint(means, physical_mean - sum(offsets))] if natural else None
    result = parent.solve(fixed=fixed, constraints=constraints)
    flat_mesh = CartesianMacroMesh(4)
    flat = solve_darcy_quadrilateral(
        flat_mesh,
        skeleton=SkeletonSpace(flat_mesh, tuple(FaceSpace.uniform(1) for _ in flat_mesh.faces)),
        degree=2,
        local_refinement=2,
        source=source,
        dirichlet=exact,
        neumann=outward(flat_mesh) if natural else None,
        mean_pressure=physical_mean,
    )
    for cell, (item, field) in enumerate(zip(nested, result.fields, strict=True)):
        reconstructed = item.reconstruct(field)
        assert reconstructed.interior_residual < 2e-14
        assert_allclose(
            reconstructed.unknowns[item.boundary_dofs],
            item.trace_map @ result.trace[outer.cell_dofs(cell)],
            atol=3e-12,
        )
        for subcell, values in enumerate(reconstructed.fields):
            i, j = 2 * (cell % 2) + subcell % 2, 2 * (cell // 2) + subcell // 2
            assert_allclose(values, flat.pressure[j * 4 + i], atol=2e-11)
            assert_allclose(values, exact(qk_space(flat.local_meshes[j * 4 + i], 2)[1]), atol=2e-11)


def test_nested_invalid_contracts_and_source_moment():
    """Reject invisible outer modes, malformed boundaries and non-kernel lifts."""
    inner = HybridSystem([LocalProblem(np.eye(2), np.eye(2), [1.0, 2.0], [0, 1])])
    item = nest_hybrid_system(inner, [0, 1], np.eye(2), [0, 1])
    assert_allclose(item.moment([np.ones(2)])[0], [-1, -1, 0, 0])
    assert item.moment([np.ones(2)])[1] == 3
    for selected in ([0, 0], [], [2], [0.5], [[0]]):
        with pytest.raises(ValueError, match="boundary_dofs"):
            nest_hybrid_system(inner, selected, [[1]], [0])
    with pytest.raises(ValueError, match="injective"):
        nest_hybrid_system(inner, [0], [[1, 1]], [0, 1])
    with pytest.raises(ValueError, match="vector"):
        nest_hybrid_system(inner, [0], [[1]], [[0]])
    with pytest.raises(ValueError, match="kernel must be a matrix"):
        nest_hybrid_system(inner, [0], [[1]], [0], kernel=[1, 0])
    with pytest.raises(ValueError, match="nullspace"):
        nest_hybrid_system(inner, [0], [[1]], [0], kernel=np.ones((2, 1)))
    with pytest.raises(ValueError, match="nested field"):
        item.reconstruct([1])


def test_trace_restriction_rejects_incompatible_geometry_and_spaces():
    """A discontinuous parent trace cannot be projected into an unsplit constant child."""
    mesh = CartesianMacroMesh()
    outer = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 2) for _ in mesh.faces))
    with pytest.raises(ValueError, match="cannot represent"):
        nested_trace_map(outer, 0, SkeletonSpace(mesh))
    with pytest.raises(ValueError, match="components"):
        nested_trace_map(outer, 0, SkeletonSpace(mesh, components=2))
    shifted = CartesianMacroMesh(bounds=(0.1, 1.1, 0, 1))
    with pytest.raises(ValueError, match="exactly one parent"):
        nested_trace_map(outer, 0, SkeletonSpace(shifted))


@pytest.mark.parametrize("degree", [0, 1])
@pytest.mark.parametrize("affine", [False, True])
def test_triangle_segment_restriction_preserves_independent_one_sided_polynomials(degree, affine):
    """Translated/skew normal traces restrict to the same mathematical half-edges."""
    from pymhm.meshes.triangle import TriangleMesh

    macro = TriangleMesh.unit_square(1)
    if affine:
        origin = np.array([0.8826192971400124, 0.3242537859615011])
        vertices = np.array(
            [[0.4921243078878566, -0.16874548542199747], [0.9575255425243296, -1.5247462502871534]]
        )
        macro = TriangleMesh(origin + macro.points @ (vertices - origin), macro.cells)
    outer = SkeletonSpace(macro, tuple(FaceSpace.uniform(degree, 2) for _ in macro.faces))
    coefficients = np.arange(1.0, outer.size + 1)
    for cell in range(len(macro.cells)):
        fine = macro.submesh(cell, 2)
        inner = SkeletonSpace(fine, tuple(FaceSpace.uniform(degree) for _ in fine.faces))
        boundary, transform = nested_trace_map(outer, cell, inner)
        child_coefficients = transform @ coefficients[outer.cell_dofs(cell)]
        offset = 0
        for face in fine.boundary_faces:
            child = inner.faces[face]
            parameters, _ = child.quadrature(4)
            vertices = fine.points[fine.faces[face]]
            physical = vertices[0] + parameters[:, None] * (vertices[1] - vertices[0])
            matching = []
            for parent in macro.cell_faces[cell]:
                endpoints = macro.points[macro.faces[parent]]
                tangent = endpoints[1] - endpoints[0]
                t = (physical - endpoints[0]) @ tangent / (tangent @ tangent)
                delta = physical - endpoints[0] - t[:, None] * tangent
                if np.max(abs(delta)) < 1e-14 and t.min() >= 0 and t.max() <= 1:
                    matching.append((parent, t))
            assert len(matching) == 1
            parent, t = matching[0]
            sign = fine.normals[face] @ macro.normals[parent]
            expected = sign * outer.faces[parent].evaluate(t) @ coefficients[outer.dofs(parent)]
            actual = child.evaluate(parameters) @ child_coefficients[offset : offset + child.size]
            assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
            assert np.array_equal(boundary[offset : offset + child.size], inner.dofs(int(face)))
            offset += child.size


@pytest.mark.parametrize("degree", [0, 1])
@pytest.mark.parametrize("displacement", [8 * 128 * np.finfo(float).eps, 1e-12])
def test_affine_segment_restriction_rejects_genuinely_distinct_trace_cut(degree, displacement):
    """A true cut inside a child edge is not hidden by geometric endpoint snapping."""
    from pymhm.meshes.triangle import TriangleMesh

    macro = TriangleMesh.unit_square(1)
    origin = np.array([0.8826192971400124, 0.3242537859615011])
    vertices = np.array(
        [[0.4921243078878566, -0.16874548542199747], [0.9575255425243296, -1.5247462502871534]]
    )
    macro = TriangleMesh(origin + macro.points @ (vertices - origin), macro.cells)
    displaced_cut = 0.5 + displacement
    parent_face = FaceSpace((0.0, displaced_cut, 1.0), (degree, degree))
    outer = SkeletonSpace(macro, tuple(parent_face for _ in macro.faces))
    for cell in range(len(macro.cells)):
        fine = macro.submesh(cell, 2)
        inner = SkeletonSpace(fine, tuple(FaceSpace.uniform(degree) for _ in fine.faces))
        with pytest.raises(ValueError, match="cannot represent the restricted parent trace"):
            nested_trace_map(outer, cell, inner)


def test_three_recursive_levels_preserve_nonsymmetric_petrov_fields():
    """Condensation is associative for unequal trial/test trace pairings."""
    problem = LocalProblem(
        [[2.0, 0.4], [0.1, 1.3]],
        [[1.0, 0.3], [0.2, 1.0]],
        [2.0, -1.0],
        [0, 1],
        test_coupling=[[1.1, -0.2], [0.0, 0.7]],
    )
    expected = HybridSystem([problem], boundary_load=[1.0, 2.0]).solve()
    inner = HybridSystem([problem])
    levels = []
    for _ in range(3):
        item = nest_hybrid_system(inner, [0, 1], np.eye(2), [0, 1])
        levels.append(item)
        inner = HybridSystem([item.problem])
    solved = HybridSystem([levels[-1].problem], boundary_load=[1.0, 2.0]).solve()
    field = solved.fields[0]
    for item in reversed(levels):
        reconstruction = item.reconstruct(field)
        assert reconstruction.interior_residual < 2e-14
        field = reconstruction.fields[0]
    assert_allclose(field, expected.fields[0], atol=2e-13)
    assert_allclose(solved.trace, expected.trace, atol=2e-13)


def test_nested_trace_rejects_true_cut_inside_nonzero_roundoff_size_child_edge():
    """Endpoint normalization preserves a short edge straddling an actual trace jump."""
    from pymhm.meshes.triangle import TriangleMesh

    macro = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    offset = 16 * np.finfo(float).eps
    child = TriangleMesh(
        np.array([[0.0, 0.0], [0.5 - offset, 0.0], [0.5 + offset, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4]]),
    )
    assert np.all(child.areas > 0)
    outer = SkeletonSpace(
        macro,
        (FaceSpace.uniform(0, 2), FaceSpace.uniform(0), FaceSpace.uniform(0)),
    )
    with pytest.raises(ValueError, match="cannot represent the restricted parent trace"):
        nested_trace_map(outer, 0, SkeletonSpace(child))
