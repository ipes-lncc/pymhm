"""Moment-level and closed-loop checks for fixed-macro transport adaptation."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.transport.solver import solve_transport
from pymhm.adaptivity.transport import (
    FaceIndicators,
    TransportBounds,
    estimate_transport_faces,
    refine_skeleton_faces,
    solve_adaptive_transport,
)
from pymhm.fem.traces.scalar import scalar_trace


@pytest.mark.parametrize("continuous", [False, True])
def test_bisection_preserves_polynomial_degrees_and_components(continuous):
    mesh = TriangleMesh.unit_square()
    face = FaceSpace((0.0, 0.3, 1.0), (1, 2), continuous)
    space = SkeletonSpace(mesh, (face,) * len(mesh.faces), components=2)
    masks = tuple(np.array([False, True]) for _ in mesh.faces)
    result = refine_skeleton_faces(space, masks)
    assert result.mesh is mesh
    assert result.components == 2
    assert result.faces[0].breaks == (0.0, 0.3, 0.65, 1.0)
    assert result.faces[0].degrees == (1, 2, 2)
    assert result.faces[0].continuous == continuous


def test_exact_jump_integral_uses_macro_length_and_one_sided_values():
    mesh = TriangleMesh.unit_square()
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 2) for _ in mesh.faces))
    solution = solve_transport(mesh, skeleton=space, degree=2, dirichlet_enforcement="strong")
    # Prescribed broken constants give |R|=1; eta^2=segment_fraction with C=1.
    solution = replace(
        solution, values=(np.ones_like(solution.values[0]), -np.ones_like(solution.values[1]))
    )
    indicators = estimate_transport_faces(solution, TransportBounds(1, 0, 0), order=5)
    interior = int(np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0])
    assert_allclose(indicators.values[interior], np.sqrt(0.5))
    assert_allclose(indicators.total, 1)
    assert sum(mask.sum() for mask in indicators.mark()) == 2
    with pytest.raises(ValueError, match="macroface"):
        scalar_trace(
            mesh, solution.local_meshes[0], interior, 2, solution.values[0], np.array([-0.1])
        )


def test_indicator_scaling_and_rejecting_unstated_boundary_assumption():
    mesh = TriangleMesh.unit_square()
    result = solve_transport(mesh)
    with pytest.raises(ValueError, match="strong Dirichlet"):
        estimate_transport_faces(result, TransportBounds(1, 0, 0))
    assert_allclose(TransportBounds(1, 2, 3).scale(result.skeleton), 6)
    zeros = FaceIndicators(result.skeleton, tuple(np.zeros(1) for _ in mesh.faces), 1)
    assert not any(mask.any() for mask in zeros.mark())
    for theta in (0, 1, np.nan):
        with pytest.raises(ValueError, match="theta"):
            zeros.mark(theta)


@pytest.mark.parametrize("bounds", [(0, 0, 0), (1, -1, 0), (1, 0, -1), (1, 0, np.nan)])
def test_bounds_must_be_declared_valid(bounds):
    with pytest.raises(ValueError, match="bounds"):
        TransportBounds(*bounds)


def test_marking_shape_is_not_silently_broadcast():
    space = SkeletonSpace(TriangleMesh.unit_square())
    with pytest.raises(ValueError, match="one marking"):
        refine_skeleton_faces(space, ())
    for mask in (np.array([1]), np.array([True, False])):
        with pytest.raises(ValueError, match="boolean"):
            refine_skeleton_faces(space, (mask,) * len(space.faces))


def test_adaptive_loop_reduces_manufactured_error_without_changing_local_mesh():
    mesh = TriangleMesh.unit_square(2)

    def exact(x):
        return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

    def source(x):
        return 2 * np.pi**2 * exact(x)

    result = solve_adaptive_transport(
        SkeletonSpace(mesh),
        TransportBounds(1, 0, 0),
        iterations=3,
        degree=2,
        local_refinement=4,
        source=source,
    )
    assert result.stop_reason == "iterations"
    assert len(result.solutions) == 3
    assert result.solutions[-1].skeleton.size > result.solutions[0].skeleton.size
    assert result.solutions[-1].l2_error(exact, 7) < result.solutions[0].l2_error(exact, 7)
    for solution in result.solutions:
        assert len(solution.local_meshes[0].cells) == 16
    capped = solve_adaptive_transport(
        SkeletonSpace(mesh),
        TransportBounds(1, 0, 0),
        iterations=2,
        source=source,
        max_trace_dofs=SkeletonSpace(mesh).size,
    )
    assert capped.stop_reason == "max_trace_dofs"
    zero = solve_adaptive_transport(SkeletonSpace(mesh), TransportBounds(1, 0, 0))
    assert zero.stop_reason == "tolerance"
    assert len(zero.solutions) == 1


@pytest.mark.parametrize(
    "options,match",
    [
        ({"iterations": 0}, "iterations"),
        ({"theta": 1}, "theta"),
        ({"tolerance": -1}, "tolerance"),
        ({"max_trace_dofs": 0}, "max_trace_dofs"),
        ({"dirichlet_enforcement": "weak"}, "strong"),
    ],
)
def test_adaptive_loop_validation(options, match):
    with pytest.raises(ValueError, match=match):
        solve_adaptive_transport(
            SkeletonSpace(TriangleMesh.unit_square()), TransportBounds(1, 0, 0), **options
        )


@pytest.mark.parametrize("convention", ["neumann", "diffusive_flux"])
def test_mixed_boundary_patch_keeps_physical_and_robin_fluxes_distinct(convention):
    """Nonzero normal advection distinguishes the two accepted natural data."""
    mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    velocity = np.array([0.7, 0.4])

    def exact(points):
        """An affine field with nonzero data and nonzero wall-normal gradient."""
        return 1 + points[:, 0] + 2 * points[:, 1]

    natural = {}
    for face in mesh.boundary_faces:
        normal = mesh.normals[face]
        if abs(normal[1]) == 1:
            flux = -np.array([1, 2]) @ normal
            natural[int(face)] = (
                float(flux)
                if convention == "diffusive_flux"
                else lambda points, flux=flux, normal=normal: (
                    flux + (velocity @ normal) * exact(points) / 2
                )
            )
    result = solve_adaptive_transport(
        skeleton,
        TransportBounds(1, np.linalg.norm(velocity), 1),
        iterations=3,
        tolerance=1e-11,
        diffusion=1,
        velocity=velocity,
        reaction=1,
        source=lambda points: 1.5 + exact(points),
        dirichlet=exact,
        degree=2,
        local_refinement=2,
        **{convention: natural},
    )
    assert result.stop_reason == "tolerance"
    assert result.solutions[0].l2_error(exact, 5) < 2e-12
    assert set(result.solutions[0].natural_faces) == set(natural)
    for face in mesh.boundary_faces:
        assert_allclose(result.indicators[0].values[face], 0, atol=0)
        assert not result.indicators[0].mark()[face].any()


def test_mixed_boundary_adaptation_preserves_walls_and_reduces_error():
    """L11 wall data survive several segment refinements without local changes."""
    mesh = TriangleMesh.unit_square(2)
    walls = {int(f): 0.0 for f in mesh.boundary_faces if abs(mesh.normals[f, 1]) == 1}

    def exact(points):
        """Smooth nonpolynomial scalar with homogeneous diffusive wall data."""
        return np.sin(np.pi * points[:, 0])

    result = solve_adaptive_transport(
        SkeletonSpace(mesh),
        TransportBounds(0.2, 1, 0),
        iterations=3,
        diffusion=0.2,
        velocity=(1, 0),
        source=lambda x: 0.2 * np.pi**2 * exact(x) + np.pi * np.cos(np.pi * x[:, 0]),
        diffusive_flux=walls,
        degree=2,
        local_refinement=4,
    )
    assert len(result.solutions) == 3
    first, last = result.solutions[0], result.solutions[-1]
    assert last.skeleton.size > first.skeleton.size
    assert last.l2_error(exact, 8) < first.l2_error(exact, 8)
    for state, indicator in zip(result.solutions, result.indicators, strict=True):
        assert set(state.natural_faces) == set(walls)
        for a, b in zip(first.local_meshes, state.local_meshes, strict=True):
            assert_allclose(a.points, b.points, atol=0, rtol=0)
            assert_allclose(a.cells, b.cells, atol=0, rtol=0)
        for face in mesh.boundary_faces:
            assert state.skeleton.faces[face] == first.skeleton.faces[face]
            assert_allclose(indicator.values[face], 0, atol=0)
