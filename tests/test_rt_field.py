"""Immutable RT fields validate global coefficients once and preserve physical replay."""

import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import pymhm.rt as rt
from pymhm.elements import triangle_quadrature
from pymhm.mesh import TriangleMesh


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_repeated_field_batches_match_public_evaluation_without_rescanning(degree, monkeypatch):
    """Every batch preserves the public evaluator bitwise and scans coefficients only once."""
    mesh = TriangleMesh.unit_square(3)
    size = (degree + 1) * len(mesh.faces) + degree * (degree + 1) * len(mesh.cells)
    coefficients = np.random.default_rng(17).normal(size=size)
    bary, _ = triangle_quadrature(4)
    points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells]).reshape(-1, 2)
    cells = np.repeat(np.arange(len(mesh.cells)), len(bary))
    expected_batches = [
        rt.rt_evaluate_points(
            mesh, coefficients, degree, points[first : first + 13], cells[first : first + 13]
        )
        for first in range(0, len(points), 13)
    ]
    expected = tuple(
        np.concatenate([batch[index] for batch in expected_batches]) for index in (0, 1)
    )
    validate = rt._validated_coefficients
    calls = []

    def counted(*args):
        """Count complete coefficient-vector validations rather than wall-clock timings."""
        calls.append(len(args[1]))
        return validate(*args)

    monkeypatch.setattr(rt, "_validated_coefficients", counted)
    field = rt.RTField(mesh, coefficients, degree)
    values, divergence = [], []
    for first in range(0, len(points), 13):
        result = field.evaluate(points[first : first + 13], cells[first : first + 13])
        values.append(result[0])
        divergence.append(result[1])
    assert calls == [size]
    assert_array_equal(np.concatenate(values), expected[0])
    assert_array_equal(np.concatenate(divergence), expected[1])


def test_field_owns_immutable_precision_preserving_storage_and_survives_pickle():
    """Neither the caller nor serialized replay can mutate previously validated coefficients."""
    mesh = TriangleMesh.unit_square()
    coefficients = np.arange(len(mesh.faces), dtype=np.longdouble) + np.longdouble("0.125")
    expected = coefficients.copy()
    field = rt.RTField(mesh, coefficients, 0)
    coefficients[:] = np.nan
    for current in (field, pickle.loads(pickle.dumps(field))):
        assert current.coefficients.dtype == np.longdouble
        assert_array_equal(current.coefficients, expected)
        with pytest.raises(ValueError, match="read-only"):
            current.coefficients[0] = 0
        with pytest.raises(ValueError, match="WRITEABLE"):
            current.coefficients.setflags(write=True)


def test_field_rejects_bad_coefficients_and_still_validates_incident_points():
    """One-time coefficient checks do not bypass coordinate or cell-membership checks."""
    mesh = TriangleMesh.unit_square()
    good = np.ones(len(mesh.faces))
    for values in (good[:-1], good * np.nan, good.astype(complex)):
        with pytest.raises(ValueError, match="coefficients"):
            rt.RTField(mesh, values, 0)
    with pytest.raises(ValueError, match="degree"):
        rt.RTField(mesh, good, -1)
    field = rt.RTField(mesh, good, 0)
    with pytest.raises(ValueError, match="incident"):
        field.evaluate(np.zeros((1, 2)), np.array([-1]))
    with pytest.raises(ValueError, match="outside"):
        field.evaluate(np.array([[2.0, 2.0]]), np.array([0]))
