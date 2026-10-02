"""Scientific evidence contracts for a single fixed-trace resolution endpoint."""

import hashlib
import json
from pathlib import Path

import pytest

from examples import verify_transport_published as verification


def endpoint_record(tmp_path, monkeypatch):
    """Provide small immutable field bytes and the declared physical-space metadata."""
    archive = tmp_path / "endpoint.npz"
    archive.write_bytes(b"field-identity-fixture")
    row = dict(
        archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        epsilon=1.0,
        local_refinement=512,
        segments=16,
        free_trace_dofs=5888,
        retained_constant_coordinates=240,
        source_changed_during_run=False,
        quadrature={str(q): dict(l2_error=0.02, broken_h1_error=0.2) for q in (8, 12)},
    )
    report = dict(
        epsilon=1.0,
        macro_resolution=8,
        local_refinement=512,
        local_degree=1,
        trace_degree=0,
        prepared_segments=16,
        source_changed_during_run=False,
        gradient_dg0_projection_error={"8": 0.1, "12": 0.1},
        records=[row],
    )
    path = tmp_path / "endpoint.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(verification, "DATA", tmp_path)
    return path, report


def test_endpoint_has_only_two_observed_error_values(tmp_path, monkeypatch):
    """One scalar field associates with two norms, without fabricating a five-point family."""
    path, _ = endpoint_record(tmp_path, monkeypatch)
    rows = verification.checked_endpoint(path)["records"]
    published = dict(
        panels=[
            dict(
                series=[
                    dict(
                        method="space",
                        points=[
                            dict(
                                N1_graphical_interval=[count - 1, count + 1],
                                value=value,
                                graphical_interval=[0.99 * value, 1.01 * value],
                                visibility="visible_marker",
                            )
                            for count in (368, 736, 1472, 2944, 5888)
                        ],
                    )
                ]
            )
            for value in (0.02, 0.1)
        ]
    )
    compared = verification.compare_rows(rows, published, "space")
    assert len(compared) == 2
    assert [row["free_trace_dofs"] for row in compared] == [5888, 5888]
    assert [row["inside_graphical_interval"] for row in compared] == [True, False]
    assert [row["ratio_to_marker"] for row in compared] == [1.0, 2.0]


@pytest.mark.parametrize("change", ["coefficient", "trace", "missing_norm", "duplicate"])
def test_endpoint_rejects_a_different_physical_comparison(tmp_path, monkeypatch, change):
    """Matching an abscissa cannot substitute for coefficient, space or quadrature identity."""
    path, report = endpoint_record(tmp_path, monkeypatch)
    if change == "coefficient":
        report["records"][0]["epsilon"] = 0.1
    elif change == "trace":
        report["records"][0]["segments"] = 8
    elif change == "missing_norm":
        report["records"][0]["quadrature"].pop("12")
    else:
        report["records"].append(report["records"][0].copy())
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="preserve"):
        verification.checked_endpoint(path)


def test_endpoint_requires_the_recorded_field_bytes(tmp_path, monkeypatch):
    """A current norm record does not authorize changed coefficient archives."""
    path, report = endpoint_record(tmp_path, monkeypatch)
    (tmp_path / report["records"][0]["archive"]).write_bytes(b"different-field")
    with pytest.raises(ValueError, match="provenance"):
        verification.checked_endpoint(path)


def test_endpoint_uses_one_coherent_record_read(tmp_path, monkeypatch):
    """A replacement during field verification cannot mix metadata from two versions."""
    path, expected = endpoint_record(tmp_path, monkeypatch)
    original_read = Path.read_text
    reads = []

    def replace_after_read(self, *args, **kwargs):
        """Simulate an atomic writer replacing the record after its bytes were read."""
        value = original_read(self, *args, **kwargs)
        if self == path:
            reads.append(self)
            path.write_text(json.dumps({"records": [], "epsilon": 0.1}))
        return value

    monkeypatch.setattr(Path, "read_text", replace_after_read)
    assert verification.checked_endpoint(path) == expected
    assert reads == [path]
