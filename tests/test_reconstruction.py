"""Constrained flux reconstruction preserves traces and fine mass balance."""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.elements import face_integration
from pymhm.reconstruction import equilibrate_flux


def test_equilibrated_affine_patch_and_pointwise_flux():
    mesh = TriangleMesh.unit_square(2)
    result = solve_darcy(mesh, dirichlet=lambda x: x[:, 0] + 2 * x[:, 1])
    flux = equilibrate_flux(result)
    assert flux.l2_error([-1, -2]) < 1e-12
    for r in flux.conservation_residuals():
        assert_allclose(r, 0, atol=1e-12)


def test_equilibration_with_source_and_boundary_fluxes():
    mesh = TriangleMesh.unit_square(2)
    result = solve_darcy(mesh, source=-4, dirichlet=lambda x: x[:, 0] ** 2 + x[:, 1] ** 2)
    flux = equilibrate_flux(result)
    for cell, (fine, q, residual) in enumerate(
        zip(flux.meshes, flux.coefficients, flux.conservation_residuals(), strict=True)
    ):
        assert_allclose(residual, 0, atol=2e-12)
        expected = (
            face_integration(mesh, cell, fine, result.skeleton)[1]
            @ result.hybrid.trace[result.skeleton.cell_dofs(cell)]
        )
        assert_allclose(q[fine.boundary_faces], expected, atol=2e-12)


def test_equilibration_validation():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="primal"):
        equilibrate_flux(solve_darcy(mesh, formulation="mixed"))
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    with pytest.raises(ValueError, match="degree-zero"):
        equilibrate_flux(solve_darcy(mesh, skeleton=space))
    result = solve_darcy(mesh)
    altered = dataclasses.replace(result.hybrid, trace=np.ones(result.skeleton.size))
    with pytest.raises(ValueError, match="incompatible"):
        equilibrate_flux(dataclasses.replace(result, hybrid=altered))


@pytest.mark.parametrize("scale", [1e-12, 1e-100])
def test_equilibration_rejects_incompatible_trace_in_small_units(scale):
    result = solve_darcy(TriangleMesh.unit_square())
    altered = dataclasses.replace(result.hybrid, trace=np.full(result.skeleton.size, scale))
    with pytest.raises(ValueError, match="incompatible"):
        equilibrate_flux(dataclasses.replace(result, hybrid=altered))
