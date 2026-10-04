"""Analytical geometry and moment checks for archived SPE10 comparison norms."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pymhm.fem.hdiv.rt import rt_interpolate
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.triangle import TriangleMesh


@pytest.fixture
def modules(monkeypatch):
    """Import original example helpers without adding a runtime dependency."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return (
        importlib.import_module("examples.spe10_adaptive"),
        importlib.import_module("examples.spe10_adaptive_norms"),
    )


def test_overlay_preserves_sixth_degree_moment_on_non_nested_grids(modules):
    """Resolve unrelated material lines, reference diagonals and an oblique fine cell."""
    triangle = np.array([[150.0, 270.0], [351.0, 281.0], [171.0, 433.0]])
    reference = SimpleNamespace(nx=37, ny=111)
    points, weights = modules[1].overlay_quadrature(triangle, reference, 4)
    bary, base_weights = triangle_quadrature(6)
    exact_points = bary @ triangle
    area = abs(np.linalg.det((triangle[1:] - triangle[0]).T)) / 2
    monomial = (points[:, 0] / 1200) ** 4 * (points[:, 1] / 2200) ** 2
    exact = (exact_points[:, 0] / 1200) ** 4 * (exact_points[:, 1] / 2200) ** 2
    assert weights.sum() == pytest.approx(area, rel=3e-14)
    assert weights @ monomial == pytest.approx(area * (base_weights @ exact), rel=3e-14)
    assert np.all(weights > 0)


@pytest.mark.parametrize("double_only", [False, True])
def test_classical_norm_constant_physical_fields(modules, monkeypatch, double_only):
    """Check physical and material-weighted norms across unresolved coefficient cuts."""
    api = modules[0]
    if double_only:
        monkeypatch.setattr(api.np, "longdouble", np.float64)
    scales = np.arange(1.0, 7.0).reshape(3, 2)
    tensors = scales[..., None, None] * np.diag([2.0, 5.0])
    material = CartesianCellField(tensors, tuple(api.DOMAIN / [3, 2]))
    monkeypatch.setattr(api, "load_layer", lambda: material)
    references = []
    for n in (1, 2):
        mesh = api.mesh_rectangle(n, n)
        references.append(
            api.StructuredRT(n, n, np.ones((len(mesh.cells), 6)), rt_interpolate(mesh, [3, 4], 2))
        )
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        norms = api.reference_integrals(references[1], references[0])
    assert norms["pressure_l2"] == pytest.approx(np.sqrt(np.prod(api.DOMAIN)), rel=1e-13)
    assert norms["flux_l2"] == pytest.approx(5 * np.sqrt(np.prod(api.DOMAIN)), rel=1e-13)
    assert norms["previous_pressure_relative_difference"] < 1e-13
    assert norms["previous_flux_relative_difference"] < 1e-13
    energy_squared = np.prod(api.DOMAIN) * (9 / 2 + 16 / 5) * np.mean(1 / scales)
    assert norms["flux_energy_norm"] == pytest.approx(np.sqrt(energy_squared), rel=1e-13)
    assert norms["previous_flux_energy_relative_difference"] < 1e-13
    with pytest.raises(ValueError, match="dyadic"):
        api.reference_integrals(references[1], references[1])


@pytest.mark.parametrize("double_only", [False, True])
def test_positive_integral_retains_many_small_fragments(modules, monkeypatch, double_only):
    """Resolve a known positive tail even where longdouble has float64 precision."""
    api = modules[0]
    if double_only:
        monkeypatch.setattr(api.np, "longdouble", np.float64)
    weights = np.r_[1.0, np.full(65536, 2.0**-54)]
    actual = api._weighted_integral(weights, np.ones_like(weights))
    expected = 1.0 + 2.0**-38
    assert abs(actual - expected) <= 4 * np.finfo(np.float64).eps


@pytest.mark.parametrize("portable", [False, True])
def test_variable_local_archives_preserve_extended_pressure(
    modules, monkeypatch, tmp_path, portable
):
    """Unequal local arrays preserve one-sided P2 values and RT2 moments."""
    api = modules[1]
    monkeypatch.setattr(api, "load_layer", lambda: lambda x: np.tile(np.eye(2), (len(x), 1, 1)))
    macro = TriangleMesh.unit_square()
    meshes = tuple(macro.submesh(i, i + 1) for i in range(2))
    points = tuple(mesh.points for mesh in meshes)
    cells = tuple(mesh.cells for mesh in meshes)
    values = tuple((1 + nodal_space(mesh, 2)[1] @ [2, 3]).astype(np.longdouble) for mesh in meshes)
    values[0][3] = np.nextafter(values[0][3], np.longdouble(np.inf))
    flux = tuple(rt_interpolate(mesh, [-2, -3], 2) for mesh in meshes)

    def offsets(parts):
        """Return exact segment boundaries for non-object archive storage."""
        return np.r_[0, np.cumsum([len(part) for part in parts])]

    path = tmp_path / "variable.npz"
    coefficients = {"pressure": np.concatenate(values)}
    if portable:
        high, low, tail = importlib.import_module("examples.archive_precision").split_precision(
            coefficients["pressure"]
        )
        coefficients = {"pressure": high, "pressure_correction": low, "pressure_tail": tail}
    np.savez(
        path,
        macro_points=macro.points,
        macro_cells=macro.cells,
        local_points=np.concatenate(points),
        point_offsets=offsets(points),
        local_cells=np.concatenate(cells),
        cell_offsets=offsets(cells),
        **coefficients,
        pressure_offsets=offsets(values),
        reconstructed_flux=np.concatenate(flux),
        flux_offsets=offsets(flux),
    )
    restored = api.BrokenP2(path)
    assert restored.pressure[0].base is restored.pressure[1].base
    assert restored.flux[0].base is restored.flux[1].base
    assert all(value.dtype == np.dtype(np.longdouble) for value in restored.pressure)
    for actual, expected in zip(restored.pressure, values, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for cell, mesh in enumerate(meshes):
        centers = mesh.points[mesh.cells].mean(axis=1)
        pressure, raw, recovered = restored.evaluate_local(cell, centers)
        np.testing.assert_allclose(pressure, 1 + centers @ [2, 3], atol=2e-14)
        np.testing.assert_allclose(raw, np.tile([-2, -3], (len(centers), 1)), atol=2e-13)
        np.testing.assert_allclose(recovered, raw, atol=2e-13)
    monkeypatch.setattr(modules[0], "DOMAIN", np.ones(2))
    monkeypatch.setattr(modules[1], "DOMAIN", np.ones(2))
    reference_mesh = modules[0].mesh_rectangle(1, 1)
    reference = modules[0].StructuredRT(
        1, 1, np.ones((2, 6)), rt_interpolate(reference_mesh, [-2, -3], 2)
    )
    norms = api.compare(restored, reference)
    assert norms["pressure_difference_l2"] == pytest.approx(np.sqrt(22 / 3), rel=1e-13)
    assert norms["raw_flux_energy_difference"] < 2e-13
    assert norms["reconstructed_flux_energy_difference"] < 2e-13
    assert norms["reference_flux_energy_norm"] == pytest.approx(np.sqrt(13), rel=1e-13)
