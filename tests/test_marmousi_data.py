"""Verify primary SEG-Y decoding, physical units and declared crop selection."""

import hashlib
import importlib
import struct
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def data(monkeypatch):
    """Load the example without making examples an installed runtime package."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.marmousi_data")


def test_ibm_decoding_and_trace_layout(tmp_path, data):
    """Known hexadecimal IBM numbers check signs, exponents and trace strides."""
    words = np.array([[0, 0x41100000, 0xC1200000], [0x40800000, 0x42100000, 0x43100000]])
    expected = np.array([[0.0, 1.0, -2.0], [0.5, 16.0, 256.0]])
    np.testing.assert_array_equal(data._ibm_values(words), expected)
    payload = bytearray(3600 + 2 * (240 + 12))
    payload[3220:3222] = struct.pack(">H", 3)
    payload[3224:3226] = struct.pack(">H", 1)
    for i in range(2):
        offset = 3600 + i * 252
        payload[offset + 114 : offset + 116] = struct.pack(">H", 3)
        payload[offset + 240 : offset + 252] = words[i].astype(">u4").tobytes()
    path = tmp_path / "fixture.segy"
    path.write_bytes(payload)
    np.testing.assert_array_equal(
        data._read_samples(path, [1, 0], [2, 0], shape=(2, 3)), expected[[1, 0]][:, [2, 0]]
    )
    for byte, value, message in [(3224, 5, "IBM"), (3220, 2, "binary"), (3714, 2, "trace")]:
        altered = payload.copy()
        altered[byte : byte + 2] = struct.pack(">H", value)
        path.write_bytes(altered)
        with pytest.raises(ValueError, match=message):
            data._read_samples(path, [0], [0], shape=(2, 3))
    path.write_bytes(payload[:-1])
    with pytest.raises(ValueError, match="size"):
        data._read_samples(path, [0], [0], shape=(2, 3))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"shape": (2,)}, "positive crop"),
        ({"shape": (2.0, 3.0)}, "positive crop"),
        ({"shape": (0, 2)}, "positive crop"),
        ({"origin": (0,)}, "positive crop"),
        ({"origin": (1j, 0)}, "positive crop"),
        ({"origin": (np.nan, 0)}, "positive crop"),
        ({"spacing": 1j}, "positive crop"),
        ({"spacing": np.inf}, "positive crop"),
        ({"spacing": 0}, "positive crop"),
        ({"origin": (0.1, 0)}, "coincide"),
        ({"origin": (-5, 0)}, "outside"),
        ({"origin": (17000, 0)}, "outside"),
    ],
)
def test_invalid_crop(tmp_path, kwargs, message, data):
    """Crop constraints fail before opening or modifying any primary data."""
    with pytest.raises(ValueError, match=message):
        data.load_marmousi_crop(tmp_path, **kwargs)


def test_physical_units_and_pinned_digest(tmp_path, monkeypatch, data):
    """Cell centres, SI conversion and bulk modulus are independently checked."""
    manifest = {}
    for name in ["vp", "density"]:
        content = name.encode()
        (tmp_path / name).write_bytes(content)
        manifest[name] = (name, f"https://example.com/{name}", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(data, "FILES", manifest)

    def samples(path, ix, iz, *, shape):
        """Stand in only for the separately tested low-level SEG-Y reader."""
        assert shape == (13601, 2801)
        np.testing.assert_array_equal(ix, [2718, 2722])
        np.testing.assert_array_equal(iz, [414, 418, 422])
        return np.full((2, 3), 3.0 if path.name == "vp" else 2.0)

    monkeypatch.setattr(data, "_read_samples", samples)
    material = data.load_marmousi_crop(tmp_path, shape=(2, 3))
    np.testing.assert_array_equal(material.velocity.values, np.full((2, 3), 3000.0))
    np.testing.assert_array_equal(material.density.values, np.full((2, 3), 2000.0))
    np.testing.assert_array_equal(material.bulk_modulus.values, np.full((2, 3), 18e9))
    assert material.provenance["historical_article_arrays_identified"] is False
    assert material.density.origin == (0.0, 0.0)
    (tmp_path / "vp").write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256"):
        data.load_marmousi_crop(tmp_path, shape=(2, 3))
    (tmp_path / "vp").write_bytes(b"vp")
    for value in [0.0, np.nan]:
        monkeypatch.setattr(
            data, "_read_samples", lambda *a, value=value, **k: np.full((2, 3), value)
        )
        with pytest.raises(ValueError, match="strictly positive"):
            data.load_marmousi_crop(tmp_path, shape=(2, 3))
