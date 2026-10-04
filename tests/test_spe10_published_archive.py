"""Light affine-field verification of the published-convention campaign archive."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.fem.scalar.triangle import nodal_space
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(float).eps,
    reason="the reservoir campaign explicitly selects wider local refinement precision",
)
def test_affine_pressure_and_four_local_triangles(tmp_path, monkeypatch):
    """A small exact patch verifies spaces, flux balance and portable physical coefficients."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    campaign = importlib.import_module("examples.solve_spe10_published")
    restore_precision = importlib.import_module("examples.archive_precision").restore_precision
    domain = importlib.import_module("examples.spe10_adaptive").DOMAIN
    unit = TriangleMesh.unit_square(2)
    mesh = TriangleMesh(unit.points * domain, unit.cells)
    monkeypatch.setattr(campaign, "mesh_rectangle", lambda nx, ny: mesh)
    monkeypatch.setattr(campaign, "load_layer", lambda: 2.0)
    with threadpool_limits(1):
        rows = campaign.acquire(tmp_path, levels=1, target_cells=8, workers=1)
    assert len(rows) == 1
    row = rows[0]
    assert row["estimator_convention"] == "published"
    assert row["fine_triangles"] == 4 * row["macro_triangles"]
    assert row["local_refinement"] == 2
    assert row["stop_reason"] == "macro_budget"
    assert not row["source_changed_during_solve"]
    assert_allclose(row["inflow"], 2 * domain[0] / domain[1], rtol=2e-12)
    assert row["estimator"] < 1e-9
    assert row["local_indicator"] < 1e-11
    with np.load(tmp_path / row["archive"]) as values:
        coefficients = restore_precision(
            values["pressure"], values["pressure_correction"], values["pressure_tail"]
        )
        for i in range(8):
            pfirst, plast = values["point_offsets"][i : i + 2]
            cfirst, clast = values["cell_offsets"][i : i + 2]
            first, last = values["pressure_offsets"][i : i + 2]
            local = TriangleMesh(
                values["local_points"][pfirst:plast], values["local_cells"][cfirst:clast]
            )
            nodes = nodal_space(local, 2)[1]
            assert_allclose(coefficients[first:last], 1 - nodes[:, 1] / domain[1], atol=2e-12)
    assert (
        json.loads((tmp_path / "adaptive.json").read_text())[0]["archive_sha256"]
        == row["archive_sha256"]
    )
