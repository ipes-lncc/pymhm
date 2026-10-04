"""A complete study label requires every acquired case and current mathematical identity."""

import json

import numpy as np
import pytest

from examples import helmholtz_article as article
from examples import helmholtz_local_control as fine
from examples.helmholtz_publication import support_rows


def checkpoint_row(case, directory):
    """Make an explicitly synthetic checkpoint fixture without running a numerical solve."""
    row = {
        **case.__dict__,
        "key": case.key,
        "local_degree": case.ell + 2,
        "requested_assembly_order": 10,
        "assembly_order": max(10, 2 * case.ell + 4),
        "error_order": 12,
        "macro_edge_length": 1 / case.n,
        "macro_diameter": np.sqrt(2) / case.n,
        "trace_basis": "oscillatory" if case.oscillatory else "polynomial",
        **dict.fromkeys(
            (
                "pressure_l2",
                "gradient_l2",
                "pressure_exact_l2",
                "gradient_exact_l2",
                "pressure_relative_error",
                "gradient_relative_error",
                "energy_relative_error",
                "residual",
                "macro_balance_max",
            ),
            0.1,
        ),
        "original_field_trace_residual": 1e-15,
        "original_local_equation_residual_max": 1e-15,
    }
    if case.archive_fields:
        path = directory / f"fixture-{case.key}.npz"
        path.write_bytes(b"Synthetic checkpoint fixture; not numerical acquisition")
        row.update(article.archive_identity(path))
    return row


def test_all_directions_convergence_and_controls_are_required_before_plotting(tmp_path):
    """Missing one angle or duplicating another fails even when total row counts match."""
    cases = article.configurations(["direction", "convergence", "local-control"])
    rows = [checkpoint_row(case, tmp_path) for case in cases]
    assert len(rows) == 1078
    record = {
        "source_sha256": article.article_hashes(),
        "requested_assembly_order": 10,
        "error_order": 12,
        "rows": rows,
    }
    path = tmp_path / "article.json"
    path.write_text(json.dumps(record))
    assert article.publication_rows(tmp_path) == rows
    for changed in (rows[:-1], [*rows[:-1], rows[0]]):
        path.write_text(json.dumps({**record, "rows": changed}))
        with pytest.raises(ValueError, match="case set"):
            article.publication_rows(tmp_path)
    path.write_text(json.dumps({**record, "source_sha256": {"unrelated": "source"}}))
    with pytest.raises(ValueError, match="sources differ"):
        article.publication_rows(tmp_path)


def test_fine_controls_require_all_four_effective_quadrature_contracts(tmp_path):
    """A Q6 control cannot retain the old requested-only quadrature convention."""
    rows = [checkpoint_row(case, tmp_path) for case in fine.configurations()]
    record = {"source_sha256": fine.hashes(), "rows": rows}
    path = tmp_path / "local-refinement-eight.json"
    path.write_text(json.dumps(record))
    assert fine.publication_rows(tmp_path) == rows
    rows[-1]["assembly_order"] = 10
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity"):
        fine.publication_rows(tmp_path)


def test_every_current_numerical_owner_is_in_the_acquisition_identity():
    """Quadrature/material, geometry and response precision helpers share one guard."""
    hashes = article.article_hashes()
    for name in (
        "io/reservoir",
        "materials/planar",
        "fem/quadrature/planar",
        "meshes/roundoff",
        "core/contracts",
        "linalg/linear",
    ):
        assert f"src/pymhm/{name}.py" in hashes
    assert "examples/helmholtz_incident_family.py" in hashes


def test_primary_and_native_controls_identify_the_actual_analytical_acquisition(tmp_path):
    """An old comparison or wrong native boundary frequency cannot accompany new curves."""
    rows = [checkpoint_row(case, tmp_path) for case in article.configurations(["convergence"])]
    path = tmp_path / "article.json"
    path.write_text(json.dumps({"rows": rows}))
    published = {
        "pymhm_source": {
            "source_sha256": article.article_hashes(),
            "sha256": article.archive_identity(path)["archive_sha256"],
        },
        "rows": [
            {
                "ell": row["ell"],
                "n": row["n"],
                "basis": row["trace_basis"],
                "field": field,
                "published_graph_value": 0.1,
                "published_graph_interval": [0.09, 0.11],
                "pymhm_case_key": row["key"],
                "pymhm_relative_error": row[f"{field}_relative_error"],
            }
            for row in rows
            for field in ("pressure", "gradient")
        ],
    }
    native = {
        "executed_source_sha256": article.article_hashes(),
        "runtime_provenance": {"test_fixture": True},
        "rows": [
            {
                "ell": ell,
                "n": n,
                "oscillatory": oscillatory,
                "local_degree": ell + 2,
                "refinement": 2,
                "omega": 10 * np.pi,
                "theta": np.pi / 13,
                "native_original_equation_relative_residual": 1e-14,
                "pressure_coefficient_relative_l2_difference": 1e-14,
                "native_relative_pressure_l2": 0.1,
                "native_relative_gradient_l2": 0.1,
                "pymhm_relative_pressure_l2": 0.1,
                "pymhm_relative_gradient_l2": 0.1,
            }
            for ell, sizes in ((2, (12, 16, 24, 32)), (3, (24, 32)))
            for n in sizes
            for oscillatory in (False, True)
        ],
    }
    published_path = tmp_path / "published-convergence.json"
    native_path = tmp_path / "native-convergence-verification.json"
    published_path.write_text(json.dumps(published))
    native_path.write_text(json.dumps(native))
    assert [len(value) for value in support_rows(tmp_path, rows)] == [44, 12]
    published["pymhm_source"]["sha256"] = "old-acquisition"
    published_path.write_text(json.dumps(published))
    with pytest.raises(ValueError, match="different analytical data"):
        support_rows(tmp_path, rows)
    published["pymhm_source"]["sha256"] = article.archive_identity(path)["archive_sha256"]
    published_path.write_text(json.dumps(published))
    native["rows"][0]["omega"] += 0.1
    native_path.write_text(json.dumps(native))
    with pytest.raises(ValueError, match="identity"):
        support_rows(tmp_path, rows)
