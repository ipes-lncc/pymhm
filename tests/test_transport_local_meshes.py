"""Physical time stepping and Darcy coupling preserve supplied fine geometry."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm._legacy.models.transport.dispersion import (
    HydrodynamicDispersion,
    RT0DarcyVelocity,
    solve_darcy_transport,
)
from pymhm._legacy.models.transport.solver import solve_heat, solve_transport
from pymhm._legacy.models.transport.transient import MacroCoefficient, solve_transient_transport
from pymhm.fem.scalar.triangle import nodal_space


def _partitions(mesh, kind):
    """Make nonuniform valid partitions, including four cells with moved edge nodes."""
    result = []
    for cell, indices in enumerate(mesh.cells):
        vertices = mesh.points[indices]
        if kind == "fan":
            points = np.vstack((vertices, vertices.mean(axis=0)))
            fine = TriangleMesh(points, np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]]))
        else:
            fine = mesh.submesh(cell, 2)
            points = fine.points.copy()
            midpoint = (vertices[0] + vertices[1]) / 2
            index = np.argmin(np.linalg.norm(points - midpoint, axis=1))
            points[index] = 0.35 * vertices[0] + 0.65 * vertices[1]
            fine = TriangleMesh(points, fine.cells.copy())
        result.append(fine)
    return tuple(result)


def _quadratic(points):
    """Return a pressure with exactly known Laplacian four and nonzero boundary."""
    return 1 + points[:, 0] + np.sum(points**2, axis=1)


@pytest.mark.parametrize("kind", ["fan", "moved-midpoint"])
@pytest.mark.parametrize("solver", [solve_heat, solve_transient_transport])
def test_custom_heat_meshes_preserve_nonzero_time_patch(kind, solver):
    """Every operation uses the supplied P3 space, even with equal cell counts."""
    mesh = TriangleMesh.unit_square()
    meshes = _partitions(mesh, kind)
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    options = dict(
        skeleton=space,
        degree=3,
        local_refinement=2,
        local_meshes=meshes,
        initial=_quadratic,
        source=lambda x, t: _quadratic(x) - 4 * (1 + t),
        dirichlet=lambda x, t: (1 + t) * _quadratic(x),
        quadrature_order=6,
    )
    if solver is solve_transient_transport:
        options["dirichlet_enforcement"] = "weak"
    result = solver(mesh, [0, 0.125, 0.25], **options)
    solutions = result.solutions if solver is solve_transient_transport else result
    for time, solution in zip((0.125, 0.25), solutions, strict=True):
        for supplied, actual, values in zip(
            meshes, solution.local_meshes, solution.values, strict=True
        ):
            assert actual is supplied
            assert_allclose(
                values, (1 + time) * _quadratic(nodal_space(supplied, 3)[1]), atol=3e-12
            )
        assert solution.l2_error(lambda x, t=time: (1 + t) * _quadratic(x), 6) < 2e-12


@pytest.mark.parametrize("solver", [solve_heat, solve_transient_transport])
def test_invalid_transport_partitions_are_rejected_before_assembly(solver):
    """An omitted macro partition or an exterior fine vertex cannot be ignored."""
    mesh = TriangleMesh.unit_square()
    fine = _partitions(mesh, "fan")
    with pytest.raises(ValueError, match="one local mesh"):
        solver(mesh, [0, 1], local_meshes=fine[:1])
    shifted = TriangleMesh(fine[0].points + (0.1, 0), fine[0].cells)
    with pytest.raises(ValueError, match="cover its macro triangle"):
        solver(mesh, [0, 1], local_meshes=(shifted, fine[1]))


@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
def test_custom_transient_mesh_and_consistent_previous_mass(stabilization):
    """Nonpolynomial forcing matches stationary RAD on the identical selected mesh."""
    mesh = TriangleMesh.unit_square()
    meshes = _partitions(mesh, "moved-midpoint")
    options = dict(
        local_meshes=meshes,
        local_refinement=2,
        degree=2,
        velocity=(0.7, 0.2),
        reaction=2.0,
        stabilization=stabilization,
        dirichlet_enforcement="weak",
    )
    evolved = solve_transient_transport(
        mesh,
        [0, 0.25],
        initial=_quadratic,
        source=lambda x, t: np.sin(x[:, 0]) + t,
        dirichlet=lambda x, t: _quadratic(x) + t,
        **options,
    ).solutions[0]
    options["reaction"] = 6.0
    stationary = solve_transport(
        mesh,
        source=lambda x: np.sin(x[:, 0]) + 0.25 + 4 * _quadratic(x),
        dirichlet=lambda x: _quadratic(x) + 0.25,
        **options,
    )
    assert_allclose(evolved.hybrid.trace, stationary.hybrid.trace, atol=3e-12)
    for actual, expected in zip(evolved.values, stationary.values, strict=True):
        assert_allclose(actual, expected, atol=3e-12)


def _custom_darcy(mesh, meshes):
    """Supply an exact affine RT0 field q=(1+x,1+y) on each actual fine partition."""
    template = solve_darcy(mesh, formulation="mixed", local_refinement=2)
    fluxes = []
    for fine in meshes:
        points = fine.points[fine.faces].mean(axis=1)
        fluxes.append(np.sum((1 + points) * fine.normals, axis=1) * fine.lengths)
    return replace(template, local_meshes=meshes, flux=tuple(fluxes))


@pytest.mark.parametrize("kind", ["fan", "moved-midpoint"])
def test_darcy_coupling_preserves_actual_meshes_and_coefficient_sides(kind):
    """The Darcy archive geometry is used by advection, dispersion and old-state mass."""
    mesh = TriangleMesh.unit_square()
    meshes = _partitions(mesh, kind)
    darcy = _custom_darcy(mesh, meshes)
    options = dict(
        degree=2,
        initial=_quadratic,
        source=lambda x, t: np.sin(x[:, 0]) + t,
        dirichlet=lambda x, t: _quadratic(x) + t,
        stabilization="supg",
        molecular=0.1,
        longitudinal=0.2,
        transverse=0.1,
    )
    coupled = solve_darcy_transport(darcy, [0, 0.1], **options)
    velocities = tuple(
        RT0DarcyVelocity(fine, q) for fine, q in zip(meshes, darcy.flux, strict=True)
    )
    materials = tuple(HydrodynamicDispersion(v, 0.1, 0.2, 0.1) for v in velocities)
    direct_options = {
        k: v for k, v in options.items() if k not in ("molecular", "longitudinal", "transverse")
    }
    direct = solve_transient_transport(
        mesh,
        [0, 0.1],
        local_meshes=meshes,
        velocity=MacroCoefficient(velocities),
        velocity_divergence=MacroCoefficient(tuple(v.divergence for v in velocities)),
        diffusion=MacroCoefficient(materials),
        diffusion_divergence=MacroCoefficient(tuple(d.divergence for d in materials)),
        **direct_options,
    )
    for fine, actual in zip(meshes, coupled.solutions[0].local_meshes, strict=True):
        assert actual is fine
    assert_array_equal(coupled.solutions[0].hybrid.trace, direct.solutions[0].hybrid.trace)
    for actual, expected in zip(
        coupled.solutions[0].values, direct.solutions[0].values, strict=True
    ):
        assert_array_equal(actual, expected)
    assert_allclose(coupled.balance_residuals, 0, atol=3e-12)
    copies = tuple(TriangleMesh(fine.points.copy(), fine.cells.copy()) for fine in meshes)
    same = solve_darcy_transport(darcy, [0, 0.1], local_meshes=copies, **options)
    assert_array_equal(same.solutions[0].hybrid.trace, coupled.solutions[0].hybrid.trace)
    if kind == "fan":
        with pytest.raises(ValueError, match="local_refinement"):
            solve_darcy_transport(darcy, [0, 0.1], local_refinement=2)


def test_darcy_transport_rejects_same_count_different_geometry_or_connectivity():
    """Neither cell counts nor matching coordinates alone identify an allowed override."""
    mesh = TriangleMesh.unit_square()
    meshes = _partitions(mesh, "moved-midpoint")
    darcy = _custom_darcy(mesh, meshes)
    wrong_geometry = tuple(mesh.submesh(cell, 2) for cell in range(len(mesh.cells)))
    wrong_cells = tuple(TriangleMesh(fine.points, fine.cells[::-1]) for fine in meshes)
    for supplied in (meshes[:1], wrong_geometry, wrong_cells):
        with pytest.raises(ValueError, match="points and connectivity"):
            solve_darcy_transport(darcy, [0, 0.1], local_meshes=supplied)
