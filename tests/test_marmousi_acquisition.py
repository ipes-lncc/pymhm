"""Pinned primary data acquisition never accepts partial or corrupt SEG-Y caches."""

import hashlib
import io

import pytest

from examples import marmousi_data


@pytest.fixture
def primary(monkeypatch):
    """Use bounded synthetic primary bytes without downloading reservoir volumes."""
    payload = b"SEG-Y primary"
    identity = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(marmousi_data, "PRIMARY_FILE_BYTES", len(payload))
    monkeypatch.setattr(
        marmousi_data,
        "FILES",
        {
            "vp": ("vp.segy", "https://primary.example/vp.segy", identity),
            "density": ("density.segy", "https://primary.example/density.segy", identity),
        },
    )
    return payload


def test_pinned_acquisition_and_cache_reuse(primary, tmp_path, monkeypatch):
    """Both fields use their pinned URL and digest; repeat runs require no network."""
    calls = []

    def open_url(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(primary)

    monkeypatch.setattr(marmousi_data.urllib.request, "urlopen", open_url)
    fields = marmousi_data.download_marmousi_data(tmp_path, maximum_bytes_per_file=len(primary))
    assert set(fields) == {"vp", "density"}
    assert all(path.read_bytes() == primary for path in fields.values())
    assert calls == [(item[1], 60) for item in marmousi_data.FILES.values()]
    monkeypatch.setattr(
        marmousi_data.urllib.request, "urlopen", lambda *a, **k: pytest.fail("cache downloaded")
    )
    assert (
        marmousi_data.download_marmousi_data(tmp_path, maximum_bytes_per_file=len(primary))
        == fields
    )
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("response", [b"partial", b"wrong primary", b"SEG-Y primary overflow"])
def test_invalid_primary_transfer_leaves_no_cache(primary, tmp_path, monkeypatch, response):
    """Size and digest failures never publish a file under the trusted cache name."""
    monkeypatch.setattr(
        marmousi_data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(response)
    )
    with pytest.raises(ValueError, match="Marmousi"):
        marmousi_data.download_marmousi_data(tmp_path, maximum_bytes_per_file=len(primary))
    assert list(tmp_path.iterdir()) == []


def test_interrupted_transfer_removes_partial_file(primary, tmp_path, monkeypatch):
    """A failed network connection leaves the same empty cache as a fresh clone."""

    def unavailable(*args, **kwargs):
        raise OSError("connection interrupted")

    monkeypatch.setattr(marmousi_data.urllib.request, "urlopen", unavailable)
    with pytest.raises(OSError, match="interrupted"):
        marmousi_data.download_marmousi_data(tmp_path, maximum_bytes_per_file=len(primary))
    assert list(tmp_path.iterdir()) == []


def test_wrong_existing_primary_is_not_overwritten(primary, tmp_path, monkeypatch):
    """Cache identity failures preserve the caller's existing bytes for inspection."""
    destination = tmp_path / "vp.segy"
    destination.write_bytes(b"wrong primary")
    monkeypatch.setattr(
        marmousi_data.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("corrupt cache downloaded"),
    )
    with pytest.raises(ValueError, match="cache differs"):
        marmousi_data.download_marmousi_data(tmp_path, maximum_bytes_per_file=len(primary))
    assert destination.read_bytes() == b"wrong primary"


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_primary_budget_is_explicit(primary, tmp_path, budget):
    """Acquisition requires a positive integral budget for a complete primary file."""
    with pytest.raises(ValueError, match="byte budget"):
        marmousi_data.download_marmousi_data(tmp_path, maximum_bytes_per_file=budget)
