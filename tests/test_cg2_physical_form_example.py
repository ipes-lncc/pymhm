"""Physical CG2 moment action is independent of rounded CSR constant modes."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.sparse.linalg import splu

requires_extended = pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(float).eps,
    reason="physical-form accumulation requires a wider long-double type",
)


def modules(monkeypatch):
    """Load original reference owners without exposing optional native dependencies."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return tuple(
        importlib.import_module(f"examples.{name}")
        for name in (
            "cg2_physical_form",
            "compensated_reference",
            "solve_pgmhm_inclusions_reference",
        )
    )


def nodal_axes(x, y):
    """Return tensor coordinates including edge midpoints for canonical CG2 arrays."""
    return np.meshgrid(
        np.sort(np.r_[x, (x[:-1] + x[1:]) / 2]), np.sort(np.r_[y, (y[:-1] + y[1:]) / 2])
    )


@requires_extended
def test_physical_constants_quadratic_patch_and_component_replay(monkeypatch, tmp_path):
    """Exact moments preserve a constant and the low polynomial beside a huge datum."""
    owner, _, reference = modules(monkeypatch)
    x = np.array([0, 1 / 64, 1 / 8, 1 / 2, 1.0], dtype=np.longdouble)
    y = np.array([0, 1 / 4, 3 / 4, 1.0], dtype=np.longdouble)
    xx, yy = nodal_axes(x, y)
    form = owner.CG2DiffusionForm(x, y, np.full((3, 4), 2.0), source=-8)
    high = np.full(xx.shape, 1e200)
    assert np.count_nonzero(form.action(high)) == 0
    low = 1e-20 * (xx**2 + yy**2)
    assert_array_equal(form.action(high, low), form.action(np.zeros_like(high), low))
    quadratic = xx**2 + yy**2
    residual = form.action(quadratic) - form.load()
    assert_allclose(residual[1:-1, 1:-1], 0, atol=4e-18, rtol=0)
    assert_allclose(form.load().sum(), -8, atol=1e-18, rtol=0)
    field = reference.InclusionField(
        np.zeros(xx.shape), np.full((1, 1), 2.0), (0, 1, 0, 1), x, y, quadratic
    )
    check = reference.physical_diagnostics(field, -8)
    assert check["residual"] < 1e-16
    assert check["relative_energy_work_defect"] < 1e-16
    path = tmp_path / "field.npz"
    np.savez(
        path,
        coefficients=field.coefficients,
        coefficients_correction=np.asarray(quadratic, dtype=float),
        coefficients_tail=np.zeros_like(quadratic, dtype=float),
        permeability=field.permeability,
        bounds=field.bounds,
        x_axis=x,
        y_axis=y,
    )
    replay = reference.load_field(path)
    assert reference.physical_diagnostics(replay, -8) == check
    points = np.array([[0.003, 0.11], [0.71, 0.31]])
    for first, second in zip(field.evaluate(points), replay.evaluate(points), strict=True):
        assert_array_equal(first, second)
    for values in (np.zeros_like(xx), np.ones_like(xx), xx):
        harmonic = reference.InclusionField(values, np.ones((1, 1)), (0, 1, 0, 1), x, y)
        checks = reference.physical_diagnostics(harmonic, 0)
        assert checks["residual"] < 1e-16
        assert checks["relative_energy_work_defect"] < 1e-16


@pytest.mark.parametrize("target_dtype", [float, np.longdouble])
def test_target_form_refinement_keeps_csr_as_preconditioner(monkeypatch, target_dtype):
    """A perturbed CSR can precondition the target without being its acceptance criterion."""
    _, arithmetic, _ = modules(monkeypatch)
    target = np.array([[3.0, -1.0], [-1.0, 2.0]], dtype=target_dtype)
    matrix = sparse.csr_matrix(np.asarray(target, float) + np.diag([0.01, 0.02]))
    factor = splu(matrix.tocsc())
    rhs = np.array([1.0, 3.0])
    original = matrix.copy()

    def residual(high, low):
        """Evaluate the independent target in extended precision."""
        return rhs - target @ (high.astype(target_dtype) + low)

    high, low, history = arithmetic.refine_components(
        matrix, rhs, factor.solve, residual=residual, residual_scale=float(np.linalg.norm(rhs))
    )
    assert len(history) > 2 and history[-1] < 1e-10
    assert np.linalg.norm(arithmetic.component_residual(matrix, rhs, high, low)) > 1e-3
    assert_array_equal(matrix.data, original.data)
    for scale in (-1.0, np.inf):
        with pytest.raises(ValueError, match="scale"):
            arithmetic.refine_components(matrix, rhs, factor.solve, residual_scale=scale)


@requires_extended
def test_physical_form_validation_and_readonly_geometry(monkeypatch):
    """Geometry, coefficient dimensions and available arithmetic have explicit contracts."""
    owner, _, _ = modules(monkeypatch)
    x = np.array([0.0, 1.0])
    k = np.ones((1, 1))
    form = owner.CG2DiffusionForm(x, x, k)
    x[0] = -0.1
    k[0, 0] = 2
    assert form.x_axis[0] == 0 and form.permeability[0, 0] == 1
    assert not form.x_axis.flags.writeable
    for a, b, c, s in (
        ([0, np.nan], [0, 1], [[1]], 1),
        ([0, 0], [0, 1], [[1]], 1),
        ([[0, 1]], [0, 1], [[1]], 1),
        ([0], [0, 1], [[1]], 1),
        ([0, 1], [0, 1], [1], 1),
        ([0, 1], [0, 1], [[0]], 1),
        ([0, 1], [0, 1], [[1]], np.nan),
    ):
        with pytest.raises(ValueError):
            owner.CG2DiffusionForm(a, b, c, s)
    for high, low in (
        (np.zeros(2), None),
        (np.zeros((3, 3)), np.zeros(2)),
        (np.full((3, 3), np.nan), None),
        (np.zeros((3, 3)), np.full((3, 3), np.inf)),
    ):
        with pytest.raises(ValueError):
            form.action(high, low)


@requires_extended
def test_physical_action_against_decimal_vertex_moments(monkeypatch):
    """Independent affine-gradient vertex moments resolve a low field under a huge datum."""
    from decimal import Decimal, localcontext

    owner, _, _ = modules(monkeypatch)
    x, y = np.array([0.0, 0.125, 1.0]), np.array([0.0, 1.0])
    xx, yy = nodal_axes(x, y)
    high = np.full(xx.shape, 1e100)
    low = np.asarray(1e-20 * (xx**2 + xx * yy + 2 * yy**2), float)
    material = np.array([[1.0, 1e5]])
    form = owner.CG2DiffusionForm(x, y, material)
    answer = [[Decimal(0) for _ in range(5)] for _ in range(3)]
    # For linear gradients, integration is the barycentric mass moment
    # area*(1+delta_ab)/12 at the THREE VERTICES, not the action's Gauss points.
    patterns = (
        ([(0, 0), (2, 0), (2, 2), (1, 0), (2, 1), (1, 1)], [[-1, 0], [1, -1], [0, 1]]),
        ([(0, 0), (0, 2), (2, 2), (0, 1), (1, 2), (1, 1)], [[0, -1], [-1, 1], [1, 0]]),
    )
    with localcontext() as context:
        context.prec = 80
        for rectangle in range(2):
            dx = Decimal.from_float(float(x[rectangle + 1] - x[rectangle]))
            dy = Decimal(1)
            area = dx * dy / 2
            k = Decimal.from_float(float(material[0, rectangle]))
            for offsets, raw in patterns:
                d = [[Decimal(v) / h for v, h in zip(row, (dx, dy), strict=True)] for row in raw]
                gradients = []
                for vertex in range(3):
                    lam = [Decimal(int(i == vertex)) for i in range(3)]
                    row = [[(4 * lam[i] - 1) * v for v in d[i]] for i in range(3)]
                    row += [
                        [4 * (lam[a] * d[b][j] + lam[b] * d[a][j]) for j in range(2)]
                        for a, b in ((0, 1), (1, 2), (2, 0))
                    ]
                    gradients.append(row)
                coefficients = [
                    Decimal.from_float(float(low[b, 2 * rectangle + a])) for a, b in offsets
                ]
                gp = [
                    [sum(gradients[v][i][j] * coefficients[i] for i in range(6)) for j in range(2)]
                    for v in range(3)
                ]
                for i, (a, b) in enumerate(offsets):
                    result = (
                        sum(
                            (1 + int(v == w)) * sum(gradients[v][i][j] * gp[w][j] for j in range(2))
                            for v in range(3)
                            for w in range(3)
                        )
                        * area
                        * k
                        / 12
                    )
                    answer[b][2 * rectangle + a] += result
    expected = np.array([[np.longdouble(str(v)) for v in row] for row in answer])
    assert_allclose(form.action(high, low), expected, rtol=2e-17, atol=1e-36)


def test_physical_form_rejects_absent_extended_precision(monkeypatch):
    """Explicitly reject a missing arithmetic capability instead of emulating extra digits."""
    owner, _, _ = modules(monkeypatch)
    monkeypatch.setattr(owner.np, "finfo", lambda dtype: SimpleNamespace(eps=1e-16))
    with pytest.raises(RuntimeError, match="extended"):
        owner.CG2DiffusionForm([0, 1], [0, 1], [[1]])
