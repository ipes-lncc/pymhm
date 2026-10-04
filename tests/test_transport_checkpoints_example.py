"""Canonical fields survive interrupted diagnostics without false acceptance."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples.transport_checkpoints import checkpoint_field, checkpoint_norm


def test_field_checkpoint_and_each_norm_keep_the_executed_field(tmp_path):
    """Progress is atomic, tied to field bytes and distinct from a final acquisition."""
    path = tmp_path / "field.npz"
    arrays = {"coefficients": np.arange(6.0).reshape(2, 3), "trace": np.array([1.0, -2.0])}
    record = checkpoint_field(
        path, arrays, {"physical_residual": 3e-15, "source_hashes": {"owner": "executed"}}
    )
    field_bytes = path.read_bytes()
    assert record["archive_sha256"] == hashlib.sha256(field_bytes).hexdigest()
    assert record["quadrature"] == {}
    assert "incomplete" in record["status"]
    assert not path.with_suffix(".json").exists()
    with np.load(path) as data:
        for key, value in arrays.items():
            np.testing.assert_array_equal(data[key], value)
    for order in (8, 12):
        checkpoint_norm(path, record, order, {"l2_error": 0.0125})
        assert path.read_bytes() == field_bytes
        assert json.loads(path.with_suffix(".progress.json").read_text()) == record
    assert not list(tmp_path.glob("*.tmp"))


def test_field_digest_streams_and_invalid_coefficients_do_not_replace_archive(
    tmp_path, monkeypatch
):
    """Large fields need bounded-memory hashes and cannot be overwritten by nonfinite data."""
    path = tmp_path / "field.npz"

    def forbidden_read(*args, **kwargs):
        raise AssertionError("archive digest must not copy the complete file into memory")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    record = checkpoint_field(path, {"values": np.arange(4.0)}, {})
    before = record["archive_sha256"]
    for values in (np.array([np.nan]), np.array([object()]), np.array([np.inf])):
        with pytest.raises(ValueError, match="finite numerical arrays"):
            checkpoint_field(path, {"values": values}, {})
    with pytest.raises(ValueError):
        checkpoint_field(path, {"values": np.arange(5.0)}, {"residual": np.nan})
    from examples.campaign_provenance import file_digest

    assert file_digest(path) == before
    assert not list(tmp_path.glob("*.tmp"))
