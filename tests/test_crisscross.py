"""Geometry, boundary moments, and complete MH2M crisscross patches."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.fem.scalar.triangle import trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.crisscross import crisscross_submesh
from pymhm.meshes.refinement import validate_submesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.three_field import PressureTraceSpace, solve_mh2m


@pytest.mark.parametrize("count", [1, 2, 3, 4, 8])
def test_affine_area_orientation_and_exact_boundary_counts(count):
    mesh = TriangleMesh(np.array([[10.0, -4.0], [12, -3.0], [9, -1.0]]), np.array([[0, 2, 1]]))
    for side in range(3):
        fine = crisscross_submesh(mesh, 0, count, diagonal_side=side)
        validate_submesh(mesh, 0, fine)
        assert len(fine.cells) == 2 * count**2
        assert len(fine.boundary_faces) == 4 * count
        assert np.all(fine.areas > 0)
        assert_allclose(fine.areas.sum(), mesh.areas[0], rtol=3e-15)
        vertices = mesh.points[mesh.cells[0]]
        lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
        expected = np.r_[
            np.repeat(lengths[side] / (2 * count), 2 * count),
            np.repeat(lengths[(side + 1) % 3] / count, count),
            np.repeat(lengths[(side + 2) % 3] / count, count),
        ]
        assert_allclose(np.sort(fine.lengths[fine.boundary_faces]), np.sort(expected), atol=3e-15)


@pytest.mark.parametrize("count", [2, 4, 8])
def test_equal_conormal_and_cartesian_resolution_has_injective_pairing(count):
    mesh = TriangleMesh.unit_square()
    flux = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, count) for _ in mesh.faces))
    for cell in range(2):
        fine = crisscross_submesh(mesh, cell, count)
        coupling = trace_coupling(mesh, cell, fine, flux, 1)
        singular = np.linalg.svd(coupling, compute_uv=False)
        assert len(singular) == 3 * count
        assert singular[-1] > 1e-3 / count
        standard = trace_coupling(mesh, cell, mesh.submesh(cell, count), flux, 1)
        assert np.linalg.svd(standard, compute_uv=False)[-1] < 1e-15


@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "neumann"])
def test_complete_affine_physical_patch_with_explicit_partitions(boundary):
    mesh = TriangleMesh.unit_square(2)
    fine = tuple(crisscross_submesh(mesh, cell, 2) for cell in range(len(mesh.cells)))
    selected = (
        mesh.boundary_faces
        if boundary == "neumann"
        else mesh.boundary_faces[:2]
        if boundary == "mixed"
        else []
    )
    tensor = np.array([[3.0, 0.4], [0.4, 2.0]])
    gradient = np.array([1.0, -2.0])
    physical_flux = -tensor @ gradient
    result = solve_mh2m(
        mesh,
        permeability=tensor,
        dirichlet=lambda x: 1 + x[:, 0] - 2 * x[:, 1],
        neumann={int(f): physical_flux @ mesh.normals[f] for f in selected},
        mean_pressure=0.5 if boundary == "neumann" else 0,
        pressure_trace=PressureTraceSpace.uniform(mesh, 1, 1 if boundary == "neumann" else 2),
        flux_space=SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 2) for _ in mesh.faces)),
        local_meshes=fine,
    )
    assert result.l2_error(lambda x: 1 + x[:, 0] - 2 * x[:, 1]) < 2e-13
    assert result.flux_l2_error(physical_flux) < 2e-12
    assert_allclose(result.conservation_residuals(), 0, atol=2e-13)
    assert all(a.mesh is b for a, b in zip(result.local, fine, strict=True))


def test_explicit_default_partitions_preserve_all_operators_bitwise():
    mesh = TriangleMesh.unit_square()
    options = dict(source=lambda x: 1 + x[:, 0], dirichlet=lambda x: x[:, 1] ** 2)
    standard = solve_mh2m(mesh, **options)
    explicit = solve_mh2m(mesh, local_meshes=tuple(mesh.submesh(i, 4) for i in range(2)), **options)
    assert np.array_equal(standard.trace, explicit.trace)
    for a, b in zip(standard.local, explicit.local, strict=True):
        assert np.array_equal(a.stiffness.toarray(), b.stiffness.toarray())
        assert np.array_equal(a.pressure_lift, b.pressure_lift)
        assert np.array_equal(a.pressure_source, b.pressure_source)


def test_invalid_partition_inputs():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(TypeError, match="TriangleMesh"):
        crisscross_submesh(None, 0, 1)
    with pytest.raises(ValueError, match="outside"):
        crisscross_submesh(mesh, 2, 1)
    with pytest.raises(ValueError, match="subdivisions"):
        crisscross_submesh(mesh, 0, 0)
    with pytest.raises(ValueError, match="diagonal_side"):
        crisscross_submesh(mesh, 0, 1, diagonal_side=3)
    with pytest.raises(ValueError, match="one partition"):
        solve_mh2m(mesh, local_meshes=())
    with pytest.raises(ValueError, match="cover"):
        solve_mh2m(mesh, local_meshes=(mesh, mesh))
