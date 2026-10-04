"""Independent analytical data and norm contracts of the PGMHM campaign."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.petrov_galerkin import solve_pgmhm


@pytest.fixture
def campaign(monkeypatch):
    """Import public example functions independently of the pytest launcher path."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.pgmhm_campaign")


def test_source_flux_derivatives_and_crisscross_geometry(campaign):
    points = np.array([[0.12, 0.23], [0.46, 0.68], [0.85, 0.79]])
    divergence = np.zeros(len(points))
    for axis in range(2):
        shifted = points.astype(complex)
        shifted[:, axis] += 1e-25j
        assert_allclose(
            -np.imag(campaign.exact(shifted)) / 1e-25,
            campaign.exact_flux(points)[:, axis],
            rtol=2e-14,
        )
        divergence += np.imag(campaign.exact_flux(shifted)[:, axis]) / 1e-25
    assert_allclose(divergence, campaign.source(points), rtol=2e-14)
    mesh = campaign.crisscross(2)
    assert len(mesh.cells) == 16
    assert_allclose(mesh.areas.sum(), 1, atol=1e-15)
    assert_allclose(mesh.lengths.max(), 0.5, atol=1e-15)


def test_zero_field_has_nonzero_exact_pressure_flux_divergence_norms(campaign):
    result = solve_pgmhm(TriangleMesh.unit_square(), stabilization_parameter=0.1)
    values = campaign.norms(result, 20)
    assert_allclose(values["pressure_l2"], 0.5, rtol=2e-13)
    assert_allclose(values["flux_l2"], np.sqrt(2) * np.pi, rtol=2e-13)
    assert_allclose(values["enriched_divergence_l2"], 4 * np.pi**2, rtol=2e-13)
    assert_allclose(
        values["enriched_hdiv_standard_broken"], np.sqrt(2 * np.pi**2 + 16 * np.pi**4), rtol=2e-13
    )


def test_submacro_fitted_permeability_jump_exact_piecewise_pressure():
    """A material interface cuts both macros while the local fine meshes fit it."""
    material = CartesianCellField(
        np.array([[1.0], [1000.0], [1000.0], [1000.0]]), (0.25, 1.0), (0.0, 0.0)
    )

    def pressure(points):
        """Return the continuous layered potential with constant physical flux."""
        x = points[:, 0]
        return np.minimum(x, 0.25) + np.maximum(x - 0.25, 0) / 1000

    result = solve_pgmhm(
        TriangleMesh.unit_square(),
        permeability=material,
        dirichlet=pressure,
        local_refinement=4,
        stabilization_parameter=0.1,
        quadrature_order=6,
    )
    assert result.l2_error(pressure, enriched=True) < 2e-11
    assert result.flux_l2_error((-1.0, 0.0), enriched=True) < 2e-9
    assert max(abs(result.conservation_residuals())) < 2e-11
