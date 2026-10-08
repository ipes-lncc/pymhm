"""Check one-sided display geometry and fresh acquisition source identity."""

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")
from examples import plot_nested as plotting

pytestmark = pytest.mark.visualization


def test_independent_leaf_pixels_do_not_overlap_or_mix_macroface_values():
    """Pixel bounds conserve each leaf interval while its two face values stay separate."""
    left = plotting.sample_edges(np.linspace(0, 0.5, 17))
    right = plotting.sample_edges(np.linspace(0.5, 1, 17))
    assert left[-1] == right[0] == 0.5
    assert np.sum(np.diff(left)) == np.sum(np.diff(right)) == 0.5
    assert left[-2] < 0.5 < right[1]
    # Independent constant one-sided fields integrate over their own pixels,
    # even if their values disagree at the common macroface.
    assert np.diff(left) @ np.ones(17) + np.diff(right) @ np.full(17, 3) == 2


@pytest.mark.parametrize("samples", [[0, 0], [0, np.nan], [0, np.inf], [1, 0], [0]])
def test_invalid_sample_geometry_is_rejected(samples):
    """Nonfinite or non-increasing coordinates cannot define physical sample pixels."""
    with pytest.raises(ValueError, match="increasing"):
        plotting.sample_edges(np.asarray(samples))


def test_fresh_producer_source_identity_is_not_retagged_as_previous_snapshot(tmp_path, monkeypatch):
    """A fresh rendered acquisition retains its own executed sources and rejects mixtures."""
    folder = tmp_path / "examples/results/fresh"
    folder.mkdir(parents=True)
    output = tmp_path / "figures"
    output.mkdir()
    archive = folder / "n4-homogeneous.npz"
    archive.write_bytes(b"executed archive")
    sources = {"src/pymhm/core/nested.py": "fresh-source-digest"}
    record = dict(
        schema=2,
        rows=[
            dict(
                n=n,
                boundary_case=boundary,
                source_sha256=sources,
                archive=archive.name,
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            )
            for n in (1, 2, 4, 8, 16)
            for boundary in ("sine_zero_dirichlet", "affine_plus_sine")
        ],
    )
    record_path = folder / "nested.json"
    record_path.write_text(json.dumps(record))
    monkeypatch.setattr(plotting, "ROOT", tmp_path)
    monkeypatch.setattr(plotting, "__file__", str(tmp_path / "plot.py"))
    (tmp_path / "plot.py").write_text("fresh plot owner")
    monkeypatch.setattr(
        plotting,
        "read_archive",
        lambda path: (
            {"n": 4, "boundary_case": "sine_zero_dirichlet", "acquisition_uuid": "fresh"},
            {},
        ),
    )
    monkeypatch.setattr(plotting, "convergence", lambda record, output: {})
    monkeypatch.setattr(plotting, "fields", lambda arrays, output: ({}, {"points": "saved"}))
    provenance = plotting.render(record_path, output)
    expected = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    assert provenance["numerical_source_identity_sha256"] == expected
    assert provenance["acquisition_uuid"] == "fresh"
    assert provenance["actual_display_digests"] == {"points": "saved"}
    record["rows"][0]["source_sha256"] = {"src/pymhm/core/nested.py": "changed"}
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unchanged executed source"):
        plotting.render(record_path, output)
