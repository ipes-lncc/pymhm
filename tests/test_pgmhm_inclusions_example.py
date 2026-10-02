"""Square-annulus geometry and original P2 comparison norms are explicit and reproducible."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.lagrange import nodal_space
from pymhm.mesh import TriangleMesh
from pymhm.refinement import validate_submesh


def modules(monkeypatch):
    """Load original application modules without making them optional-core dependencies."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return tuple(
        importlib.import_module(f"examples.{name}")
        for name in (
            "pgmhm_inclusion_data",
            "compare_pgmhm_inclusions",
            "solve_pgmhm_inclusions_reference",
        )
    )


def test_square_annulus_measure_and_polygon_local_geometry(monkeypatch):
    """The material has hollow square cores and all 32 local partitions fit the L mesh."""
    model, _, _ = modules(monkeypatch)
    material = model.material_array()
    assert_allclose(np.mean(material > 1), 0.10546875, atol=0, rtol=0)
    center = np.full((1, 2), 0.5 / 27)
    assert model.coefficient(center) == 1
    center[0, 0] += (model.INNER_RADIUS + model.OUTER_RADIUS) / 2
    assert model.coefficient(center) == 1e5
    macro, locals_ = model.local_meshes(1)
    assert len(macro.cells) == 32
    assert sum(len(mesh.cells) for mesh in locals_) == 30752
    for cell, mesh in enumerate(locals_):
        validate_submesh(macro, cell, mesh)
    with pytest.raises(ValueError, match="positive integer"):
        model.axis(0)


def test_p2_ragged_fields_and_nonzero_exact_norms(monkeypatch, tmp_path):
    """Simultaneous comparisons retain the physical area, flux factor and each field order."""
    _, comparator, reference = modules(monkeypatch)
    mesh = TriangleMesh.unit_square()
    _, nodes = nodal_space(mesh, 2)
    fields = {name: (i + 1) * nodes[:, 0] for i, name in enumerate(("mhm", "pgmhm", "enriched"))}
    path = tmp_path / "mhm.npz"
    arrays = dict(
        point_offsets=[0, 4],
        cell_offsets=[0, 2],
        coefficient_offsets=[0, len(nodes)],
        local_points=mesh.points,
        local_cells=mesh.cells,
    )
    for name, values in fields.items():
        arrays.update(
            {
                name: values,
                name + "_correction": np.zeros_like(values),
                name + "_tail": np.zeros_like(values),
            }
        )
    np.savez(path, **arrays)
    field = comparator.InclusionMHMField(path)
    classical = reference.InclusionField(np.ones((3, 3)), np.full((1, 1), 2.0), (0, 1, 0, 1))
    monkeypatch.setattr(comparator, "_FIELDS", (field, classical))
    monkeypatch.setattr(comparator, "coefficient", lambda points: np.full(len(points), 2.0))
    expected = [1 / 3, 4, 2, 1 / 3, 16, 8, 1, 36, 18, 1, 0, 0]
    assert_allclose(comparator.integrate((0, 3)), expected, atol=2e-14, rtol=1e-14)
    assert_allclose(comparator.integrate((0, 4)), expected, atol=2e-14, rtol=1e-14)
    with pytest.raises(ValueError, match="outside"):
        field.evaluate(0, 0, np.array([[2.0, 2.0]]))


def test_reference_grading_preserves_material_and_isotropic_nested_refinement(monkeypatch):
    """Only the new graded sequence is nested; its material boundaries are unchanged."""
    model, _, _ = modules(monkeypatch)
    grading = importlib.import_module("examples.inclusion_grading")
    coarse, fine = grading.graded_axis(1), grading.graded_axis(2)
    assert_allclose(fine[::2], coarse, rtol=0, atol=0)
    assert_allclose(fine[1::2], (coarse[:-1] + coarse[1:]) / 2, rtol=0, atol=2e-16)
    assert_allclose(coarse[::7], model.axis(1), rtol=0, atol=0)
    assert np.all(np.diff(fine) > 0)
    with pytest.raises(ValueError, match="positive integer"):
        grading.graded_axis(0)


def test_classical_norms_require_nested_diagonal_grids(monkeypatch):
    """A nonzero physical norm checks weights and rejects unrelated graded triangulations."""
    _, _, reference = modules(monkeypatch)
    fine = reference.InclusionField(
        2 + np.tile(np.linspace(0, 1, 9), (9, 1)), np.ones((1, 1)), (0, 1, 0, 1)
    )
    coarse = reference.InclusionField(
        1 + np.tile(np.linspace(0, 1, 5), (5, 1)), np.ones((1, 1)), (0, 1, 0, 1)
    )
    result = reference.difference(fine, coarse, 3)
    assert_allclose(result["pressure_absolute"], 1.0, atol=2e-14, rtol=0)
    assert_allclose(result["flux_absolute"], 0.0, atol=2e-14, rtol=0)
    unrelated = reference.InclusionField(np.ones((7, 7)), np.ones((1, 1)), (0, 1, 0, 1))
    with pytest.raises(ValueError, match="isotropically nested"):
        reference.difference(unrelated, coarse, 3)
    moved = reference.InclusionField(
        np.ones((9, 9)),
        np.ones((1, 1)),
        (0, 1, 0, 1),
        np.array([0.0, 0.2, 0.5, 0.75, 1.0]),
        np.linspace(0, 1, 5),
    )
    with pytest.raises(ValueError, match="physical axes"):
        reference.difference(moved, coarse, 3)
