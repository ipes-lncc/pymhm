"""Scientific plotting invariants for current records and independent macro-side limits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from examples import plot_periodic
from pymhm._legacy.models.darcy.conforming import ConformingQuadrilateralSolution
from pymhm.meshes.cartesian import CartesianMacroMesh

pytestmark = pytest.mark.visualization


def test_profiles_keep_both_macro_side_values_and_mark_faces(monkeypatch):
    """A real nodal pressure jump stays two independent limits in the rendered plot."""
    mesh = CartesianMacroMesh(2)
    reference = ConformingQuadrilateralSolution(mesh, 1, np.zeros(9), 1.0, 0.0)
    collections = {}
    for segments, multiplier in ((1, 1.0), (32, 3.0)):
        collections[segments] = tuple(
            ConformingQuadrilateralSolution(
                mesh.submesh(cell, 2), 1, np.full(9, multiplier * (cell + 1)), 1.0, 0.0
            )
            for cell in range(4)
        )
    monkeypatch.setattr(plot_periodic, "load_reference", lambda _: (reference, Path("reference")))
    monkeypatch.setattr(
        plot_periodic,
        "load_fields",
        lambda macro, refinement, segments: (collections[segments], None),
    )
    figures = []
    monkeypatch.setattr(plot_periodic, "save", lambda figure, _: figures.append(figure))
    try:
        plot_periodic.profile_maps("1:2", macro=2, refinement=2, samples=16, ordinate=0.25)
        pressure = figures[0].axes[0]
        limits = []
        for line in pressure.lines:
            x, y = line.get_xdata(), line.get_ydata()
            if len(x) > 2:
                assert x.max() <= 0.5 or x.min() >= 0.5
                limits.extend(y[x == 0.5])
        assert set(limits) == {0.0, 1.0, 2.0, 3.0, 6.0}
        assert any(np.array_equal(line.get_xdata(), [0.5, 0.5]) for line in pressure.lines)
        assert figures[0].axes[3].get_ylabel() == "Darcy flux magnitude"
    finally:
        for figure in figures:
            plot_periodic.plt.close(figure)


def test_convergence_uses_selected_reference_resolution(monkeypatch, tmp_path):
    """Unrelated historical finer local grids cannot silently select an empty current curve."""
    selected = "5:256:10:lor"
    rows = [
        dict(
            field=f"mhm:r{refinement}:s{segments}",
            reference=selected,
            macro=8,
            refinement=refinement,
            segments=segments,
            relative_h1=1 / segments,
            relative_l2=0.1,
        )
        for refinement in (64, 128)
        for segments in (1, 32)
    ]
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(dict(comparisons=rows)))
    historical = dict(
        reference=[],
        mhm=[
            dict(
                macro=8,
                refinement=refinement,
                segments=1,
                reference_degree=1,
                reference_n=4096,
                relative_h1=999.0,
            )
            for refinement in (128, 512)
        ],
    )
    (tmp_path / "periodic-historical.json").write_text(json.dumps(historical))
    monkeypatch.setattr(plot_periodic, "COMPARISON", path)
    monkeypatch.setattr(plot_periodic, "RECORDS", tmp_path)
    figures = []
    monkeypatch.setattr(plot_periodic, "save", lambda figure, _: figures.append(figure))
    try:
        assert plot_periodic.convergence(selected) == 128
        face = figures[0].axes[0]
        assert "local r=128" in face.get_title()
        assert not any(line.get_label() == "MHM / conforming Q1 4096²" for line in face.lines)
        current = next(
            line for line in face.lines if line.get_label().startswith("MHM / conforming Q5")
        )
        assert np.array_equal(current.get_ydata(), [1.0, 1 / 32])
    finally:
        for figure in figures:
            plot_periodic.plt.close(figure)


@pytest.mark.parametrize("ordinate", [0.0, 0.5, 1.0, np.nan])
def test_profile_ordinate_rejects_ambiguous_macro_face(ordinate):
    """A horizontal macroface needs two traces and cannot become a single implicit side."""
    with pytest.raises(ValueError, match="strictly inside"):
        plot_periodic.profile_maps("1:2", macro=2, ordinate=ordinate)
