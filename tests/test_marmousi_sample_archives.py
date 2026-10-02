"""Preserve complex, one-sided pressure samples in lightweight notebook archives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from examples import marmousi_sample_archives as samples


def acquisition(folder: Path, width: int = 20, degree: int = 1) -> Path:
    """Write a small complex acquisition with visibly different incident values."""
    folder.mkdir(exist_ok=True)
    archive = folder / f"mhm-H{width}-ell{degree}-q9.npz"
    np.savez_compressed(
        archive,
        sample_points=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]),
        sample_pressure=np.arange(12).reshape(4, 3) + 1j * np.arange(12, 24).reshape(4, 3),
        incident_sides=np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]),
        pressure=np.ones((2, 30), dtype=complex),
    )
    record = archive.with_suffix(".json")
    record.write_text(
        json.dumps(
            {
                "H_m": width,
                "trace_degree": degree,
                "archive": archive.name,
                "archive_sha256": samples.digest(archive),
                "source_changed_during_run": False,
            }
        )
    )
    return record


def test_exact_sample_copy_excludes_full_coefficients(tmp_path: Path) -> None:
    """Complex values, side order and coordinates survive extraction bitwise."""
    record = acquisition(tmp_path)
    result = samples.export_samples(record, tmp_path / "samples")
    target = tmp_path / "samples" / result["archive"]
    with (
        np.load(target, allow_pickle=False) as archive,
        np.load(record.with_suffix(".npz")) as full,
    ):
        assert set(archive.files) == {"sample_points", "sample_pressure", "incident_sides"}
        for key in archive.files:
            np.testing.assert_array_equal(archive[key], full[key])
    assert result["archive_sha256"] == samples.digest(target)
    assert result["acquisition_record_sha256"] == samples.digest(record)
    assert result["acquisition_archive_sha256"] == samples.digest(record.with_suffix(".npz"))
    assert result["sample_count"] == 3
    assert result["archive_bytes"] == target.stat().st_size


@pytest.mark.parametrize(
    "failure", ["changed_sources", "digest", "changed_record", "changed_archive"]
)
def test_changed_acquisition_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Neither changed acquisition identities nor corrupted archives are published."""
    record = acquisition(tmp_path)
    metadata = json.loads(record.read_text())
    if failure == "changed_sources":
        metadata["source_changed_during_run"] = True
    elif failure == "digest":
        metadata["archive_sha256"] = "0" * 64
    else:
        original = samples.digest
        calls = 0

        def changing_digest(path: Path) -> str:
            """Simulate replacement of a file between its two checks."""
            nonlocal calls
            changed = record if failure == "changed_record" else record.with_suffix(".npz")
            if path == changed:
                calls += 1
                if calls > 1:
                    return "changed"
            return original(path)

        monkeypatch.setattr(samples, "digest", changing_digest)
    record.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        samples.export_samples(record, tmp_path / "samples")
    assert not (tmp_path / "samples").exists()


@pytest.mark.parametrize(
    "name", ["../field.npz", "/field.npz", "C:\\field.npz", "field.txt", "", 1]
)
def test_archive_path_is_local(tmp_path: Path, name: str | int) -> None:
    """Acquisition records cannot redirect extraction to another filesystem path."""
    record = acquisition(tmp_path)
    metadata = json.loads(record.read_text())
    metadata["archive"] = name
    record.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="local NPZ filename"):
        samples.export_samples(record, tmp_path / "samples")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("sample_points", np.zeros(3)),
        ("sample_points", np.zeros((3, 3))),
        ("sample_points", np.zeros((0, 2))),
        ("sample_pressure", np.zeros((3, 3))),
        ("incident_sides", np.ones((4, 1))),
        ("incident_sides", np.ones((4, 2))),
        ("sample_points", np.full((3, 2), np.nan)),
        ("sample_pressure", np.full((4, 3), np.inf)),
    ],
)
def test_invalid_samples_rejected(tmp_path: Path, key: str, value: np.ndarray) -> None:
    """Require finite coordinates and all four independent incident conventions."""
    record = acquisition(tmp_path)
    archive = record.with_suffix(".npz")
    with np.load(archive) as stored:
        arrays = dict(stored)
    arrays[key] = value
    np.savez_compressed(archive, **arrays)
    metadata = json.loads(record.read_text())
    metadata["archive_sha256"] = samples.digest(archive)
    record.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="four distinct"):
        samples.export_samples(record, tmp_path / "samples")


def test_complete_family_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Publish all fifteen sample archives with an explicit dependency manifest."""
    for width in (20, 40, 80):
        for degree in range(5):
            acquisition(tmp_path, width, degree)
    monkeypatch.setattr("sys.argv", ["samples", "--source", str(tmp_path)])
    samples.main()
    manifest = json.loads((tmp_path / "samples/manifest.json").read_text())
    assert len(manifest["rows"]) == 15
    assert {(row["H_m"], row["trace_degree"]) for row in manifest["rows"]} == {
        (width, degree) for width in (20, 40, 80) for degree in range(5)
    }
    assert manifest["source_sha256"][Path(samples.__file__).name] == samples.digest(
        Path(samples.__file__)
    )
    record = tmp_path / "mhm-H20-ell0-q9.json"
    metadata = json.loads(record.read_text())
    metadata["H_m"] = 40
    record.write_text(json.dumps(metadata))
    monkeypatch.setattr(
        "sys.argv", ["samples", "--source", str(tmp_path), "--output", str(tmp_path / "other")]
    )
    with pytest.raises(ValueError, match="discretization"):
        samples.main()
