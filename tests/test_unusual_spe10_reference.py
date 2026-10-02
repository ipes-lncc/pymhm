"""Portable interpolation and native independent diffusion--reaction reference checks."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose


def example():
    """Load the original example without changing the import path."""
    name = "solve_unusual_spe10_reference"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def quadratic_field(nx=2, ny=3):
    """Create complete nodal coefficients of an independently differentiated quadratic."""
    x, y = np.meshgrid(np.linspace(0, 2, 2 * nx + 1), np.linspace(0, 3, 2 * ny + 1))
    return example().CG2Field(
        1 + x + 2 * y + x * y + 3 * x**2 - y**2, np.full((1, 1), 2.0), (0, 2, 0, 3)
    )


def test_canonical_polynomial_and_gradient_on_both_triangles(tmp_path):
    """P2 replay, its flux and its archive retain exact physical derivatives."""
    field = quadratic_field()
    points = np.array([[0.1, 0.5], [0.8, 0.2], [1.9, 2.8], [2.0, 3.0]])
    x, y = points.T
    value, gradient, flux = field.evaluate(points)
    assert_allclose(value, 1 + x + 2 * y + x * y + 3 * x**2 - y**2, atol=2e-14)
    target = np.column_stack((1 + y + 6 * x, 2 + x - 2 * y))
    assert_allclose(gradient, target, atol=3e-14)
    assert_allclose(flux, -2 * target, atol=6e-14)
    path = tmp_path / "field.npz"
    np.savez(
        path, coefficients=field.coefficients, permeability=field.permeability, bounds=field.bounds
    )
    assert_allclose(example().CG2Field.load(path).evaluate(points)[0], value)


def test_integrated_nested_polynomials_and_physical_energy():
    """A doubled field has unit relative errors against itself as the fine reference."""
    fine = quadratic_field(4, 6)
    zero = quadratic_field()
    zero = example().CG2Field(np.zeros_like(zero.coefficients), zero.permeability, zero.bounds)
    norms = example().integrated_difference(fine, zero)
    for name in ("pressure", "flux", "energy"):
        assert norms[f"{name}_relative"] == pytest.approx(1.0)
    same = example().integrated_difference(fine, quadratic_field())
    assert max(same[f"{name}_relative"] for name in ("pressure", "flux", "energy")) < 1e-14


@pytest.mark.parametrize("points", [np.array([[3.0, 0.0]]), np.array([[np.nan, 0.0]]), np.ones(2)])
def test_bad_evaluation_points_rejected(points):
    """A reference cannot silently extrapolate across the prescribed physical domain."""
    with pytest.raises(ValueError):
        quadratic_field().evaluate(points)


def test_nonmatching_material_and_mesh_rejected():
    """Exact nested integration requires shared material and an actual nested partition."""
    fine, coarse = quadratic_field(3, 3), quadratic_field()
    with pytest.raises(ValueError, match="nested"):
        example().integrated_difference(fine, coarse)
    with pytest.raises(ValueError, match="positive"):
        example().CG2Field(np.ones((3, 3)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="alignment"):
        example().solve(2, 3)


def test_graded_coordinates_preserve_nested_polynomials():
    """Nonuniform physical rectangles retain exact P2 values and nested diagonal integration."""
    fields = []
    for factor in (1, 2):
        xa = example().subdivide_axis(np.array([0.0, 0.02, 0.5, 2.0]), factor)
        ya = example().subdivide_axis(np.array([0.0, 0.003, 0.1, 3.0]), factor)
        x, y = np.meshgrid(example().subdivide_axis(xa, 2), example().subdivide_axis(ya, 2))
        fields.append(example().CG2Field(1 + x * y + x**2, np.ones((1, 1)), (0, 2, 0, 3), xa, ya))
    comparison = example().integrated_difference(fields[1], fields[0])
    assert comparison["pressure_relative"] < 2e-15
    assert comparison["flux_relative"] < 3e-14
    assert not example().axis_aligned(np.array([0.0, 0.2, 1.0]), 2)


@pytest.mark.parametrize("horizontal", [False, True])
def test_reaction_grading_retains_all_material_interfaces(horizontal):
    """Both graded families keep exact pixels and isotropically nested physical rectangles."""
    coarse = example().graded_axes(1, horizontal=horizontal)
    fine = example().graded_axes(2, horizontal=horizontal)
    for axis, refined, pixels in zip(coarse, fine, (60, 220), strict=True):
        assert example().axis_aligned(axis, pixels)
        assert np.min(np.diff(axis)) > 0
        assert_allclose(refined[::2], axis, atol=0, rtol=0)


@pytest.mark.fem
@pytest.mark.parametrize("graded", [False, True])
def test_native_cg2_diffusion_reaction_inhomogeneous_affine_patch(graded):
    """Independent UFL/MUMPS preserves affine values, derivatives, reaction and energy."""
    pytest.importorskip("dolfinx")
    axes = (
        (np.array([0.0, 0.03, 40.0, 1200.0]), np.array([0.0, 0.001, 1.0, 100.0, 2200.0]))
        if graded
        else None
    )
    field, report = example().solve(3, 4, threads=1, patch=True, axes=axes)
    points = np.array([[10.0, 10.0], [530.0, 1300.0], [1199.0, 2199.0]])
    value, gradient, flux = field.evaluate(points)
    assert_allclose(value, 1 + points[:, 1] / 2200, atol=3e-13)
    assert_allclose(gradient, np.broadcast_to([0, 1 / 2200], gradient.shape), atol=2e-15)
    assert_allclose(flux, -2 * gradient, atol=1e-16)
    assert report["original_free_residual_relative"] < 1e-12
    assert report["energy_identity_relative"] < 1e-12
