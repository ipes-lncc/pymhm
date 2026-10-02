"""Native-reference correction components preserve the original high-contrast equations."""

import importlib
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import splu


def modules(monkeypatch):
    """Import the application arithmetic without adding it to the portable package API."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return tuple(
        importlib.import_module(f"examples.{name}")
        for name in ("compensated_reference", "solve_pgmhm_inclusions_reference")
    )


def decimal_residual(matrix, rhs, high, low):
    """Evaluate exact binary64 inputs using independent 80-digit decimal arithmetic."""
    result = []
    with localcontext() as context:
        context.prec = 80
        for row in range(len(rhs)):
            value = Decimal.from_float(float(rhs[row]))
            for index in range(matrix.indptr[row], matrix.indptr[row + 1]):
                column = matrix.indices[index]
                coefficient = Decimal.from_float(float(high[column])) + Decimal.from_float(
                    float(low[column])
                )
                value -= Decimal.from_float(float(matrix.data[index])) * coefficient
            result.append(float(value))
    return np.asarray(result)


def test_original_high_contrast_residual_and_portable_components(monkeypatch):
    """Correction accumulation exceeds one float's digits without changing the criterion."""
    arithmetic, _ = modules(monkeypatch)
    matrix = sparse.csr_matrix(
        [[1e12 + 1, -1e12, 0], [-1e12, 2e12 + 2, -1e12], [0, -1e12, 1e12 + 1]]
    )
    rhs = np.array([1.0, 2.0, 3.0])
    factor = splu(matrix.tocsc())
    high, low, history = arithmetic.refine_components(matrix, rhs, factor.solve, rtol=1e-14)
    assert history[-1] <= 1e-14
    actual = arithmetic.component_residual(matrix, rhs, high, low)
    expected = decimal_residual(matrix, rhs, high, low)
    assert_allclose(actual, expected, atol=2e-19, rtol=1e-10)
    assert np.linalg.norm(expected) <= 1e-14 * np.linalg.norm(rhs)
    assert np.linalg.norm(decimal_residual(matrix, rhs, high, np.zeros_like(low))) > (
        1e-8 * np.linalg.norm(rhs)
    )
    with pytest.raises(ArithmeticError, match="criterion failed"):
        arithmetic.refine_components(matrix, rhs, lambda forcing: np.zeros_like(forcing))
    zeros = np.zeros(3)
    assert arithmetic.refine_components(matrix, zeros, factor.solve)[2] == [0.0]


def test_compensated_products_match_decimal_for_cancelled_rows(monkeypatch):
    """Product rounding is retained as well as cancellation between sparse row terms."""
    arithmetic, _ = modules(monkeypatch)
    rng = np.random.default_rng(183)
    raw = rng.normal(size=(17, 17)) * np.geomspace(1e-8, 1e12, 17)
    matrix = sparse.csr_matrix(raw)
    high = rng.normal(size=17)
    low = np.spacing(high) * rng.uniform(-0.4, 0.4, size=17)
    rhs = raw @ high
    residual = arithmetic.component_residual(matrix, rhs, high, low)
    assert_allclose(residual, decimal_residual(matrix, rhs, high, low), rtol=2e-11, atol=1e-20)
    with pytest.raises(ArithmeticError, match="nonfinite"):
        arithmetic.component_residual(
            sparse.eye(1, format="csr"), np.array([np.nan]), np.ones(1), np.zeros(1)
        )
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ArithmeticError, match="finite arithmetic range"),
    ):
        arithmetic.two_product(np.array([1e308]), np.array([2.0]))


def test_component_polynomials_and_archive_keep_the_low_field(monkeypatch, tmp_path):
    """Evaluate the small polynomial separately even beside a much larger constant."""
    _, reference = modules(monkeypatch)
    axis = np.linspace(0, 1, 3)
    x, y = np.meshgrid(axis, axis)
    high = np.ones((3, 3), dtype=np.longdouble)
    low = 1e-20 * (x * x + y)
    field = reference.InclusionField(high, np.ones((1, 1)), (0, 1, 0, 1), correction=low)
    points = np.array([[0.25, 0.5], [0.75, 0.125]])
    _, gradient, flux = field.evaluate(points)
    expected = 1e-20 * np.column_stack((2 * points[:, 0], np.ones(2)))
    assert_allclose(gradient, expected, atol=1e-35, rtol=1e-14)
    assert_allclose(flux, -expected, atol=1e-35, rtol=1e-14)
    path = tmp_path / "reference.npz"
    np.savez(
        path,
        coefficients=high.astype(float),
        coefficients_correction=low,
        coefficients_tail=np.zeros_like(low),
        permeability=field.permeability,
        bounds=field.bounds,
        x_axis=field.x_axis,
        y_axis=field.y_axis,
    )
    assert_allclose(reference.load_field(path).evaluate(points)[1], gradient, atol=0, rtol=0)
    with pytest.raises(ValueError, match="principal shape"):
        reference.InclusionField(high, np.ones((1, 1)), (0, 1, 0, 1), correction=np.zeros(2))
