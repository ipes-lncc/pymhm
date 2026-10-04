"""Manufactured, heterogeneous and conservative Darcy verification."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy


def affine(x):
    """Affine pressure patch with nonzero tangential and normal gradients."""
    return 0.7 + x[:, 0] - 2 * x[:, 1]


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
@pytest.mark.parametrize("tensor", [2.0, [[3, 0.5], [0.5, 1.0]]])
def test_anisotropic_affine_patch(formulation, tensor):
    mesh = TriangleMesh.unit_square(2, 1)
    result = solve_darcy(mesh, dirichlet=affine, permeability=tensor, formulation=formulation)
    k = np.eye(2) * tensor if np.ndim(tensor) == 0 else np.array(tensor)
    assert result.flux_l2_error(-k @ [1, -2]) < 2e-12
    assert_allclose(result.conservation_residuals(), 0, atol=2e-12)
    if formulation == "primal":
        assert result.l2_error(affine) < 2e-13
        with pytest.raises(ValueError, match="RT0"):
            result.fine_conservation_residuals()
    else:
        for local, p in zip(result.local_meshes, result.pressure, strict=True):
            assert_allclose(p, affine(local.points[local.cells].mean(axis=1)), atol=2e-12)
        for residual in result.fine_conservation_residuals():
            assert_allclose(residual, 0, atol=2e-12)


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
def test_manufactured_convergence(formulation):
    def exact(x):
        return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

    def source(x):
        return 2 * np.pi**2 * exact(x)

    errors = []
    flux_errors = []

    def flux(x):
        return -np.pi * np.column_stack(
            (
                np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]),
                np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]),
            )
        )

    for n in (2, 4, 8):
        result = solve_darcy(
            TriangleMesh.unit_square(n), source=source, local_refinement=3, formulation=formulation
        )
        errors.append(result.l2_error(exact))
        flux_errors.append(result.flux_l2_error(flux))
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    assert min(rates) > (1.8 if formulation == "primal" else 1.1)
    assert flux_errors[-1] < flux_errors[0] / 2


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
def test_boundary_conditions_neumann_compatibility(formulation):
    mesh = TriangleMesh.unit_square(2)
    neumann = {int(f): float(np.array([-1.0, 2]) @ mesh.normals[f]) for f in mesh.boundary_faces}
    result = solve_darcy(
        mesh, dirichlet=affine, neumann=neumann, mean_pressure=0.2, formulation=formulation
    )
    assert result.flux_l2_error([-1, 2]) < 1e-11
    mixed_bc = {next(iter(neumann)): next(iter(neumann.values()))}
    mixed_result = solve_darcy(mesh, dirichlet=affine, neumann=mixed_bc, formulation=formulation)
    assert mixed_result.flux_l2_error([-1, 2]) < 1e-11
    with pytest.raises(ValueError, match="incompatible"):
        solve_darcy(mesh, source=1, neumann=neumann, formulation=formulation)


def test_piecewise_anisotropic_layer_interface():
    mesh = TriangleMesh.unit_square(2)

    def permeability(x):
        return np.where(x[:, 0] < 0.5, 1.0, 1000.0)

    def pressure(x):
        return np.where(x[:, 0] <= 0.5, 1 - x[:, 0], 0.5 - (x[:, 0] - 0.5) / 1000)

    for formulation in ("primal", "mixed"):
        result = solve_darcy(
            mesh, permeability=permeability, dirichlet=pressure, formulation=formulation
        )
        assert result.flux_l2_error([1, 0]) < 5e-10
        assert_allclose(result.conservation_residuals(), 0, atol=1e-11)


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
def test_nonzero_source_conservation(formulation):
    mesh = TriangleMesh.unit_square(2)

    def exact(x):
        return x[:, 0] ** 2 + x[:, 1] ** 2

    result = solve_darcy(mesh, source=-4, dirichlet=exact, formulation=formulation)
    assert_allclose(result.conservation_residuals(), 0, atol=2e-12)
    if formulation == "mixed":
        for residual in result.fine_conservation_residuals():
            assert_allclose(residual, 0, atol=2e-12)


def test_independent_face_degrees_and_partitions():
    mesh = TriangleMesh.unit_square(1)
    faces = tuple(FaceSpace.uniform(i % 2, 2) for i in range(len(mesh.faces)))
    space = SkeletonSpace(mesh, faces)
    result = solve_darcy(mesh, dirichlet=affine, skeleton=space, local_refinement=5)
    assert result.l2_error(affine) < 5e-13
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 2) for _ in mesh.faces))
    result = solve_darcy(mesh, dirichlet=affine, skeleton=space, formulation="mixed")
    assert result.flux_l2_error([-1, 2]) < 5e-12


def test_validation():
    mesh = TriangleMesh.unit_square()
    for kwargs in (
        {"formulation": "bogus"},
        {"formulation": "mixed", "degree": 2},
        {"local_refinement": 0},
        {"skeleton": SkeletonSpace(mesh, components=2)},
        {"skeleton": SkeletonSpace(TriangleMesh.unit_square())},
    ):
        with pytest.raises(ValueError):
            solve_darcy(mesh, **kwargs)
    for face in (FaceSpace.uniform(1), FaceSpace.uniform(0, 3)):
        space = SkeletonSpace(mesh, tuple(face for _ in mesh.faces))
        with pytest.raises(ValueError, match="RT0"):
            solve_darcy(mesh, skeleton=space, formulation="mixed")


@pytest.mark.parametrize("offset", [-4e-6, 4e-6])
def test_rt0_rejects_trace_breaks_near_but_not_on_fine_edges(offset):
    """RT0 normal traces cannot jump inside a fine edge, even near its endpoint."""
    mesh = TriangleMesh.unit_square()
    face = FaceSpace((0.0, 0.5 + offset, 1.0), (0, 0))
    skeleton = SkeletonSpace(mesh, tuple(face for _ in mesh.faces))
    with pytest.raises(ValueError, match="aligned with fine edges"):
        solve_darcy(
            mesh, skeleton=skeleton, local_refinement=2, formulation="mixed", dirichlet=affine
        )


def test_pyamg_local_neumann_solver_matches_lu():
    mesh = TriangleMesh.unit_square(2)
    amg = solve_darcy(mesh, dirichlet=affine, source=1, local_solver="pyamg")
    lu = solve_darcy(mesh, dirichlet=affine, source=1)
    assert_allclose(amg.hybrid.trace, lu.hybrid.trace, atol=1e-9)
    assert_allclose(amg.pressure, lu.pressure, atol=1e-9)


def test_quadrature_can_separate_discrete_balance_from_source_integration_error():
    mesh = TriangleMesh.unit_square()
    result = solve_darcy(mesh, source=lambda x: x[:, 0] ** 7, quadrature_order=5)
    assert_allclose(result.conservation_residuals(), 0, atol=1e-13)
    assert_allclose(result.conservation_residuals(order=8), 0, atol=1e-13)


@pytest.mark.parametrize("order", [0, -1, 2.5, True])
def test_darcy_rejects_invalid_quadrature_before_applying_degree_floor(order):
    """The shared floor must not conceal invalid volume/boundary integration requests."""
    with pytest.raises(ValueError, match="quadrature_order"):
        solve_darcy(TriangleMesh.unit_square(), quadrature_order=order)
