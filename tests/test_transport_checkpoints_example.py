"""Canonical fields survive interrupted diagnostics without false acceptance."""

import hashlib
import json

import numpy as np

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
