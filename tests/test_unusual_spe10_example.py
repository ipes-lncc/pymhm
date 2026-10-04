"""Physical field and exact-overlay contracts of the SPE10 UNUSUAL controls."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.triangle import TriangleMesh


def modules(monkeypatch):
    """Load application examples without making them portable-core dependencies."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return tuple(
        importlib.import_module(f"examples.{name}")
        for name in (
            "solve_unusual_spe10",
            "compare_unusual_spe10",
            "solve_unusual_spe10_reference",
        )
    )


def test_p1_archive_replays_physical_one_sided_fields(monkeypatch, tmp_path):
    """Stored variable partitions preserve pressure, gradients and the coefficient in flux."""
    driver, _, _ = modules(monkeypatch)
    monkeypatch.setattr(
        driver, "load_layer", lambda: CartesianCellField((2 * np.eye(2))[None, None], (1.0, 1.0))
    )
    mesh = TriangleMesh.unit_square()
    local = tuple(mesh.submesh(i, 2) for i in range(2))
    path = tmp_path / "field.npz"
    np.savez(
        path,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        local_points=np.concatenate([fine.points for fine in local]),
        local_cells=np.concatenate([fine.cells for fine in local]),
        pressure=np.concatenate(
            [1 + fine.points[:, 0] + 3 * fine.points[:, 1] + i for i, fine in enumerate(local)]
        ),
        point_offsets=np.r_[0, np.cumsum([len(fine.points) for fine in local])],
        cell_offsets=np.r_[0, np.cumsum([len(fine.cells) for fine in local])],
    )
    field = driver.UnusualSPE10Field(path)
    for macro, fine in enumerate(local):
        points = fine.points[fine.cells].mean(axis=1)
        p, gradient, flux = field.evaluate_local(macro, points, np.arange(len(fine.cells)))
        assert_allclose(p, 1 + points[:, 0] + 3 * points[:, 1] + macro, atol=2e-15)
        assert_allclose(gradient, np.tile([1, 3], (len(points), 1)), atol=2e-15)
        assert_allclose(flux, np.tile([-2, -6], (len(points), 1)), atol=4e-15)
    with pytest.raises(ValueError, match="declared fine cells"):
        field.evaluate_local(0, np.array([[2.0, 3.0]]), np.array([0]))


def test_nonuniform_overlay_integrates_moments_and_nonzero_energy(monkeypatch):
    """Independent nonzero norms detect missing Jacobians, weights and reaction terms."""
    _, compare, baseline = modules(monkeypatch)
    x_axis, y_axis = np.array([0.0, 0.1, 2.0]), np.array([0.0, 0.04, 3.0])
    # Constant pressure two has reaction norm two times sqrt(area), no flux.
    field = baseline.CG2Field(
        np.full((5, 5), 2.0), np.full((1, 1), 4.0), (0, 2, 0, 3), x_axis, y_axis
    )
    mesh = TriangleMesh(TriangleMesh.unit_square().points * [2, 3], [[0, 1, 3], [0, 3, 2]])
    vertices = mesh.points[mesh.cells]
    totals = np.zeros(6)
    for triangle in vertices:
        points, weights = compare.overlay_quadrature(triangle, field, 3)
        totals += weights @ np.column_stack(
            (
                np.ones(len(points)),
                points[:, 0],
                points[:, 1],
                points[:, 0] ** 2,
                points[:, 1] ** 2,
                points.prod(axis=1),
            )
        )
    assert_allclose(totals, [6, 6, 9, 8, 18, 9], rtol=3e-14)

    def zero(macro, points, owners):
        """The zero candidate has known differences from the nonzero constant reference."""
        return np.zeros(len(points)), np.zeros((len(points), 2)), np.zeros((len(points), 2))

    mhm = SimpleNamespace(vertices=(vertices,), evaluate_local=zero)
    result = compare.integrate_cells(mhm, field, 0, 3)
    assert_allclose(result, [24, 0, 24, 24, 0, 24], rtol=3e-14, atol=1e-24)


def test_reaction_layer_refinement_preserves_geometry_and_physical_metric(monkeypatch):
    """Local refinement changes h only and satisfies the declared h/sqrt(K) budget."""
    modules(monkeypatch)
    module = importlib.import_module("examples.unusual_spe10_refinement")
    mesh = TriangleMesh.unit_square(2)
    material = CartesianCellField([[0.25]], (1.0, 1.0))
    refined, history = module.reaction_layer_mesh(mesh, material)
    assert_allclose(refined.areas.sum(), 1.0, atol=2e-15)
    assert len(refined.cells) > len(mesh.cells)
    assert history[-1]["marked"] == 0
    assert history[-1]["largest_active_h_over_length"] <= 0.5
    with pytest.raises(RuntimeError, match="geometry budget"):
        module.reaction_layer_mesh(mesh, material, max_steps=0)
    with pytest.raises(ValueError, match="positive geometric"):
        module.reaction_layer_mesh(mesh, material, relative_diameter=0)


def test_material_trace_control_keeps_affine_heterogeneous_patch(monkeypatch, tmp_path):
    """Added P0 moments resolve physical flux jumps without changing the UNUSUAL form."""
    driver, _, _ = modules(monkeypatch)
    from pymhm._legacy.models.transport.rad import solve_rad
    from pymhm.fem.quadrature.material import fit_material_mesh

    mesh = TriangleMesh.unit_square()
    material = CartesianCellField(np.array([[1.0, 10.0]]), (1.0, 0.5))
    base = driver.make_skeleton(mesh, material, 3, trace_fitted=False)
    skeleton = driver.make_skeleton(mesh, material, 3, trace_fitted=True)
    assert skeleton.size > base.size
    assert all(np.all(np.asarray(face.degrees) == 0) for face in skeleton.faces)

    def exact(points):
        """Tangential gradient has continuous zero normal flux at the material interface."""
        return 1 + points[:, 0]

    local = tuple(fit_material_mesh(mesh.submesh(i, 6), material) for i in range(2))
    solution = solve_rad(
        mesh,
        diffusion=material,
        diffusion_divergence=(0.0, 0.0),
        source=exact,
        reaction=1.0,
        dirichlet=exact,
        skeleton=skeleton,
        local_meshes=local,
        degree=1,
        stabilization="unusual",
    )
    for fine, values in zip(local, solution.values, strict=True):
        assert_allclose(values, exact(fine.points), rtol=0, atol=2e-12)
    monkeypatch.setattr(driver, "hashes", lambda: {})
    monkeypatch.setattr(driver, "load_layer", lambda: material)
    record = driver.save_solution(
        solution,
        refinement=6,
        segments=3,
        order=6,
        layer_resolution=None,
        trace_fitted=True,
        max_local_cells=1000,
        refinement_history=[],
        fitting_seconds=0.0,
        elapsed=0.0,
        workers=1,
        fingerprint={},
        output=tmp_path,
        reuse={"local_responses_are_shared": True},
    )
    replay = driver.UnusualSPE10Field(tmp_path / record["archive"])
    for stored, original in zip(replay.pressure, solution.values, strict=True):
        assert_allclose(stored, original, rtol=0, atol=0)
    with np.load(tmp_path / record["archive"]) as arrays:
        assert_allclose(arrays["trace"], solution.hybrid.trace, rtol=0, atol=0)
    assert record["trace_dofs"] == solution.skeleton.size
    assert record["response_reuse"]["local_responses_are_shared"]
