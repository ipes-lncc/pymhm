"""Release metadata must preserve source digests without private filesystem paths."""

import io
import zipfile
from pathlib import Path
from runpy import run_path

import pytest

_CHECKS = run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "check_distribution.py"))
validate_json_provenance = _CHECKS["validate_json_provenance"]
validate_compact_layer = _CHECKS["validate_compact_layer"]


def test_compact_layers_are_real_npz_payloads():
    """The three small material inputs remain usable when checkout skips Git LFS."""
    root = Path(__file__).resolve().parents[1]
    for name in _CHECKS["COMPACT_RESERVOIR_LAYERS"]:
        validate_compact_layer((root / name).read_bytes(), name)


@pytest.mark.parametrize(
    "payload",
    [
        b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n",
        b"PK\x03\x04truncated",
    ],
)
def test_compact_layer_rejects_pointer_or_invalid_zip(payload):
    """A present filename cannot substitute a pointer or truncated archive for its arrays."""
    with pytest.raises(SystemExit, match="NPZ"):
        validate_compact_layer(payload, "layer-1.npz")


@pytest.mark.parametrize("missing", [True, False])
def test_compact_layer_requires_numpy_members(missing):
    """ZIP containers must contain each material input as an actual NPY member."""
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name in ("permeability.npy", "porosity.npy", "spacing.npy"):
            if not (missing and name == "spacing.npy"):
                archive.writestr(name, b"not an array")
    with pytest.raises(SystemExit, match="NPZ"):
        validate_compact_layer(payload.getvalue(), "layer-1.npz")


@pytest.mark.parametrize(
    "path",
    [
        "/home/researcher/work/compare.py",
        "/Users/researcher/work/compare.py",
        r"C:\Users\researcher\work\compare.py",
        ".tmp/validation/compare.py",
        "file:///home/researcher/work/compare.py",
    ],
)
def test_private_paths_are_rejected_in_keys_and_nested_values(path):
    """Inspect both source-hash keys and recursively nested provenance values."""
    for record in ({"source_hashes": {path: "abc123"}}, {"runs": [{"source": path}]}):
        with pytest.raises(SystemExit, match="private filesystem path"):
            validate_json_provenance(record, "results.json")


def test_relative_sources_and_public_urls_preserve_actual_digests():
    """Keep portable source identities, ordinary metadata and public URL paths."""
    record = {
        "source_hashes": {
            "independent-unfitted-ufl-comparison-driver": "abc123",
            "src/pymhm/darcy.py": "def456",
        },
        "source_url": "https://example.org/home/project",
        "runs": [None, 1, False, {"residual": 1e-14}],
    }
    validate_json_provenance(record, "results.json")
    assert record["source_hashes"]["independent-unfitted-ufl-comparison-driver"] == "abc123"
