"""Require physical-norm plots to identify the exact acquired reference and fields."""

import copy
import hashlib

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("pyvista")

from examples.plot_mh2m_cg3 import checked_reference, promoted_record

pytestmark = pytest.mark.visualization


def acquired(path, value):
    """Write a small distinct field payload and its acquisition identity."""
    path.write_bytes(value.encode())
    return {
        "archive": path.name,
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_changed_during_run": False,
        "resolution": 512,
    }


def test_increment_identity_requires_both_consecutive_acquisitions(tmp_path, monkeypatch):
    """A valid checksum elsewhere in the campaign cannot identify a different increment."""
    import examples.plot_mh2m_cg3 as module

    fields = [acquired(tmp_path / f"{n}.npz", str(n)) for n in range(5)]
    control = acquired(tmp_path / "control.npz", "q12")
    norms = {"quadrature_8": {}, "quadrature_10": {}}
    increments = [
        {
            "reference_archive": fine["archive"],
            "reference_sha256": fine["archive_sha256"],
            "other_archive": coarse["archive"],
            "other_sha256": coarse["archive_sha256"],
            "norms": norms,
        }
        for coarse, fine in zip(fields[:-1], fields[1:], strict=True)
    ]
    record = {
        "reference_acquisitions": fields,
        "increments": increments,
        "assembly_quadrature_control": {"acquisition": control},
    }
    monkeypatch.setattr(module.CubicTriangularField, "load", lambda path: path)
    _, finest = checked_reference(tmp_path, record)
    assert finest == fields[-1]
    corrupted = copy.deepcopy(record)
    corrupted["increments"][0]["other_sha256"] = fields[2]["archive_sha256"]
    with pytest.raises(ValueError, match="two acquired fields"):
        checked_reference(tmp_path, corrupted)
    corrupted = copy.deepcopy(record)
    corrupted["increments"][0]["norms"].pop("quadrature_8")
    with pytest.raises(ValueError, match="both norm quadratures"):
        checked_reference(tmp_path, corrupted)
    (tmp_path / fields[-1]["archive"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from its accepted acquisition"):
        checked_reference(tmp_path, record)


def test_case_requires_explicit_row_reference_or_verified_record_reference(tmp_path):
    """The structured schema declares its reference once; a crossed row must name it."""
    field = acquired(tmp_path / "field.npz", "physical field")
    reference = acquired(tmp_path / "reference.npz", "physical reference")
    row = {
        "archive": field["archive"],
        "archive_sha256": field["archive_sha256"],
        "norms": {"quadrature_8": {"pressure": 1}, "quadrature_10": {"pressure": 1}},
    }
    case = {"cases": [field]}
    with pytest.raises(ValueError, match="same fields"):
        promoted_record(case, [row], tmp_path, 1, reference)
    structured = promoted_record(case, [row], tmp_path, 1, reference, record_reference=reference)
    assert structured["cases"][0]["reference_archive_sha256"] == reference["archive_sha256"]
    crossed = {
        **row,
        "reference_archive": reference["archive"],
        "reference_sha256": reference["archive_sha256"],
    }
    assert promoted_record(case, [crossed], tmp_path, 1, reference) == structured
    with pytest.raises(ValueError, match="record-level reference"):
        promoted_record(case, [row], tmp_path, 1, reference, record_reference=field)
    corrupted = {**crossed, "reference_archive": "different.npz"}
    with pytest.raises(ValueError, match="same fields"):
        promoted_record(case, [corrupted], tmp_path, 1, reference)
