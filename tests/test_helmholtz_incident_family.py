"""Prepared incident-wave factors preserve all original full-domain equations."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.helmholtz_article import Configuration, solve_configuration, solve_direction_family
from examples.helmholtz_campaign import AcousticWave, norms
from examples.helmholtz_incident_family import IncidentFamily
from pymhm.helmholtz import solve_helmholtz
from pymhm.helmholtz_spaces import helmholtz_skeleton
from pymhm.hybrid import HybridSystem, LocalProblem
from pymhm.quadrilateral import CartesianMacroMesh


@pytest.mark.parametrize("oscillatory", [False, True])
@pytest.mark.parametrize(
    "precision",
    [
        "double",
        pytest.param(
            "extended",
            marks=pytest.mark.skipif(
                np.finfo(np.longdouble).eps >= np.finfo(float).eps,
                reason="extended correction requires a wider host mantissa",
            ),
        ),
    ],
)
def test_repeated_incident_data_match_direct_fields_without_new_harmonic_responses(
    oscillatory, precision, monkeypatch
):
    """Changed nonzero absorption and homogeneous data preserve fields and physical norms."""
    mesh = CartesianMacroMesh(3)
    wave = AcousticWave(8.1, np.pi / 13)
    skeleton = helmholtz_skeleton(mesh, wave.omega, degree=2, oscillatory=oscillatory)
    options = dict(
        omega=wave.omega,
        degree=4,
        local_refinement=2,
        quadrature_order=10,
        skeleton=skeleton,
        local_refinement_precision=precision,
    )
    prepared = solve_helmholtz(mesh, absorbing=wave.absorbing, **options)
    expected = []
    for angle in (0, np.pi / 4):
        other = AcousticWave(wave.omega, angle)
        expected.append((other, solve_helmholtz(mesh, absorbing=other.absorbing, **options)))

    def unexpected_condensation(*args, **kwargs):
        raise AssertionError("The prepared harmonic columns must be reused")

    monkeypatch.setattr(LocalProblem, "condense", unexpected_condensation)
    owner = IncidentFamily(prepared)
    with pytest.raises(RuntimeError, match="must be open"):
        owner.solve(0j)
    with owner:
        factors = [*owner._local_factors.values(), owner._global_factor]
        for other, direct in expected:
            actual = owner.solve(other.absorbing)
            assert owner.last_field_diagnostics["original_field_trace_residual"] < 1e-12
            assert owner.last_field_diagnostics["original_local_equation_residual_max"] < 1e-12
            assert actual.pressure[0].dtype == direct.pressure[0].dtype
            assert_allclose(actual.pressure, direct.pressure, rtol=2e-11, atol=2e-12)
            assert_allclose(actual.trace, direct.trace, rtol=2e-11, atol=2e-12)
            assert_allclose(actual.conservation_residuals(), 0, atol=3e-12)
            assert_allclose(
                list(norms(actual, other).values()),
                list(norms(direct, other).values()),
                rtol=3e-11,
                atol=3e-12,
            )
        homogeneous = owner.solve(0j)
        assert owner.last_field_diagnostics["original_field_trace_residual"] == 0
        assert_allclose(homogeneous.pressure, 0, atol=2e-14)
        with pytest.raises(RuntimeError, match="already open"):
            owner.__enter__()
    for factor in factors:
        if factor is not None:
            with pytest.raises(RuntimeError, match="closed"):
                factor.solve(np.empty(0))


def test_incident_family_rejects_changed_boundary_operator():
    """A different boundary partition cannot reuse the prepared absorption matrix."""
    mesh = CartesianMacroMesh(1)
    solution = solve_helmholtz(mesh, omega=2, absorbing=None, degree=2)
    with pytest.raises(ValueError, match="full exterior absorption"):
        IncidentFamily(solution)


@pytest.mark.parametrize("oscillatory", [False, True])
def test_incident_family_rejects_corrupted_physical_field_even_with_small_schur_residual(
    oscillatory, monkeypatch
):
    mesh = CartesianMacroMesh(2)
    skeleton = helmholtz_skeleton(mesh, 2.0, degree=1, oscillatory=oscillatory)
    prepared = solve_helmholtz(mesh, omega=2.0, skeleton=skeleton, degree=3, absorbing=1 + 0.3j)
    original = HybridSystem.solve

    def corrupted(self, **kwargs):
        value = original(self, **kwargs)
        assert value.residual < 1e-12
        return replace(value, fields=(value.fields[0] + 0.1, *value.fields[1:]))

    monkeypatch.setattr(HybridSystem, "solve", corrupted)
    with IncidentFamily(prepared) as owner:
        with pytest.raises(ValueError, match="original Helmholtz local"):
            owner.solve(1 + 0.3j)
        assert owner.last_field_diagnostics is None


def test_incident_family_rejects_changed_trace_injection_map(monkeypatch):
    from pymhm.helmholtz import _HelmholtzFactory

    mesh = CartesianMacroMesh(2)
    prepared = solve_helmholtz(mesh, omega=2.0, degree=3, absorbing=1j)
    original = _HelmholtzFactory.__call__

    def different_map(self, cell):
        value = original(self, cell)
        changed = np.roll(value.problem.trace_dofs, 2)
        altered = LocalProblem(
            value.problem.matrix, value.problem.coupling, value.problem.load, changed
        )
        return replace(value, problem=altered)

    with IncidentFamily(prepared) as owner:
        monkeypatch.setattr(_HelmholtzFactory, "__call__", different_map)
        with pytest.raises(ValueError, match="operator or injection map"):
            owner.solve(1j)


def test_angular_family_records_complete_norms_and_one_shared_condensation(tmp_path, monkeypatch):
    """Every acquired angle agrees with the direct case while offline work is counted once."""
    cases = [Configuration("direction", 2, 2, True, 4.1, angle) for angle in (0, np.pi / 13)]
    expected = [solve_configuration(c, tmp_path) for c in cases]
    calls = []
    original = LocalProblem.condense

    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LocalProblem, "condense", counted)
    rows = list(solve_direction_family(cases, tmp_path))
    assert len(calls) == 4
    assert len(rows) == len(cases)
    for row, direct in zip(rows, expected, strict=True):
        for key in ("key", "n", "ell", "omega", "angle", "assembly_order", "error_order"):
            assert row[key] == direct[key]
        for key in ("pressure_relative_error", "gradient_relative_error", "energy_relative_error"):
            assert row[key] == pytest.approx(direct[key], rel=2e-11, abs=3e-13)
        assert (
            row["offline_preparation_seconds_shared"]
            == rows[0]["offline_preparation_seconds_shared"]
        )
    assert list(solve_direction_family([], tmp_path)) == []
    with pytest.raises(ValueError, match="fixed angular operator"):
        list(
            solve_direction_family(
                [cases[0], Configuration("direction", 3, 2, True, 4.1, 0)], tmp_path
            )
        )
