"""Reject stale field identities when collecting the physical local-resolution controls."""

import json

import pytest

pytest.importorskip("matplotlib")

from examples.plot_unfitted_convergence import (  # noqa: E402
    local_differences,
    printed_comparisons,
)

pytestmark = pytest.mark.visualization


def local_records(folder):
    """Create six completed comparisons with distinct field identities and nonzero norms."""
    comparisons = [(24, 32, f"ell{ell}-s32") for ell in range(4)]
    comparisons += [(24, 32, "ell4-s4"), (16, 32, "ell3-s16")]
    campaigns = {refinement: {"cases": []} for refinement in (16, 24, 32)}
    (folder / "local-resolution").mkdir()
    for first, second, name in comparisons:
        fields = []
        for refinement in (first, second):
            field = {"name": f"{name}-r{refinement}.npz", "sha256": f"{name}-{refinement}"}
            campaigns[refinement]["cases"].append(
                {"name": name, "archive": field["name"], "archive_sha256": field["sha256"]}
            )
            fields.append(field)
        for order in (9, 11):
            record = {
                "fields": fields,
                "quadrature_order": order,
                "source_changed_during_run": False,
                "source_sha256": {"evaluation.py": "executed-source"},
                "norms": {"pressure_l2": 0.01, "broken_gradient_l2": 0.02, "common_triangles": 16},
            }
            path = folder / "local-resolution" / f"{name}-r{first}-r{second}-q{order}.json"
            path.write_text(json.dumps(record))
    return campaigns


def test_collector_keeps_physical_differences_and_both_quadrature_records(tmp_path):
    """The summary preserves both actual norms and the exact records they came from."""
    rows = local_differences(tmp_path, local_records(tmp_path))
    assert len(rows) == 6
    assert rows[0]["local_refinements"] == [24, 32]
    assert rows[0]["physical_difference_by_quadrature"]["11"]["broken_gradient_l2"] == 0.02
    assert len(rows[0]["record_sha256"]) == 2


@pytest.mark.parametrize("mutation", ["field", "order", "changed", "source", "negative", "nan"])
def test_collector_rejects_mismatched_provenance_and_invalid_norms(tmp_path, mutation):
    """An unrelated archive, changed source, or invalid norm cannot enter the comparison."""
    campaigns = local_records(tmp_path)
    path = tmp_path / "local-resolution/ell0-s32-r24-r32-q9.json"
    record = json.loads(path.read_text())
    if mutation == "field":
        record["fields"][0]["sha256"] = "different-acquisition"
    elif mutation == "order":
        record["quadrature_order"] = 7
    elif mutation == "changed":
        record["source_changed_during_run"] = True
    elif mutation == "source":
        record["source_sha256"] = {}
    else:
        record["norms"]["pressure_l2"] = -0.1 if mutation == "negative" else float("nan")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="provenance mismatch|invalid physical"):
        local_differences(tmp_path, campaigns)


def test_printed_configurations_are_not_shifted_and_errors_are_not_rescaled():
    """Both h/p views select the stated trace counts and use the printed value denominator."""
    smooth = {
        "cases": [
            {
                "name": f"ell0-s{s}",
                "archive_sha256": f"field-s{s}",
                "norms": {
                    "quadrature_11": {"gradient_absolute": 30.0},
                    "quadrature_13": {"gradient_absolute": value},
                },
            }
            for s, value in ((1, 2.0), (2, 0.5))
        ]
    }
    contrast = {
        "cases": [
            {
                "name": "S2-contrast100",
                "archive_sha256": "contrast",
                "norms": {"quadrature_9": {"gradient_absolute": 0.4}},
            }
        ]
    }
    published = {
        2: {
            "series": [
                {
                    "trace_degree": 0,
                    "values": [
                        {"skeleton_subdivisions": 1, "value": 1.0, "graphical_interval": [0.9, 1.1]}
                    ],
                }
            ]
        },
        3: {
            "series": [
                {
                    "skeleton_subdivisions": 2,
                    "values": [{"trace_degree": 0, "value": 0.5, "graphical_interval": [0.4, 0.6]}],
                }
            ]
        },
        7: {
            "series": [
                {
                    "setting": "S2",
                    "values": [{"contrast": 100, "value": 0.2, "graphical_interval": [0.1, 0.3]}],
                }
            ]
        },
    }
    rows = printed_comparisons(smooth, contrast, published)
    assert [row["computed_absolute_gradient_error"] for row in rows] == [2.0, 0.5, 0.4]
    assert [row["difference_relative_to_published"] for row in rows] == [1.0, 0.0, 1.0]
    assert [row["inside_graphical_interval"] for row in rows] == [False, True, False]
