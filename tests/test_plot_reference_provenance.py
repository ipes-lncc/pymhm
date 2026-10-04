"""Completed physical comparison records control every plotted reference selection."""

import json
from types import SimpleNamespace

import pytest

from examples import spe10_plot_records as records
from pymhm.meshes.triangle import TriangleMesh


def comparison(tmp_path, reference="classical-rt2-480x1760.npz"):
    """Use small opaque field bytes to exercise provenance without a large PDE solve."""
    field = tmp_path / "mhm.npz"
    field.write_bytes(b"one physical MHM field")
    ref = tmp_path / reference
    ref.write_bytes(reference.encode())
    row = dict(
        mhm=field.name,
        mhm_sha256=records.digest(field),
        reference=ref.name,
        reference_sha256=records.digest(ref),
        norms=[dict(order=q) for q in (4, 5)],
        source_changed_during_run=False,
    )
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(row))
    return path, row


def test_reference_comes_from_verified_norm_record(tmp_path):
    """A completed finer reference supersedes the old fixed 240-cell filename."""
    path, row = comparison(tmp_path)
    checked = records.checked_comparison(path, tmp_path)
    assert records.comparison_reference(tmp_path, [checked]) == tmp_path / row["reference"]


@pytest.mark.parametrize("defect", ["order", "source", "mhm", "reference"])
def test_incomplete_or_changed_comparison_is_rejected(tmp_path, defect):
    path, row = comparison(tmp_path)
    if defect == "order":
        row["norms"][1]["order"] = 4
    elif defect == "source":
        row["source_changed_during_run"] = True
    else:
        (tmp_path / row[defect]).write_bytes(b"different coefficients")
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        records.checked_comparison(path, tmp_path)


def test_reference_selection_rejects_different_denominators(tmp_path):
    _, first = comparison(tmp_path)
    _, second = comparison(tmp_path, "classical-rt2-240x880.npz")
    with pytest.raises(ValueError, match="same physical reference"):
        records.comparison_reference(tmp_path, [first, second])


def test_partial_paired_campaign_is_not_plotted(tmp_path):
    assert records.completed_cases(tmp_path) == records.CASES
    paths = [
        tmp_path / f"mhm-unusual-{stem}-q5{suffix}"
        for stem, _ in records.PAIRED_CASES
        for suffix in (".npz", ".json", "-comparison.json")
    ]
    paths[0].write_bytes(b"incomplete field fixture")
    with pytest.raises(ValueError, match="both paired"):
        records.completed_cases(tmp_path)
    for path in paths:
        path.touch()
    assert records.completed_cases(tmp_path) == (*records.CASES, *records.PAIRED_CASES)


def test_paired_claim_requires_same_responses_and_local_geometry():
    mesh = TriangleMesh.unit_square()
    field = SimpleNamespace(macro=mesh, meshes=(mesh,))
    metadata = dict(
        nominal_local_refinement=64,
        reaction_layer_resolution=0.25,
        local_degree=1,
        trace_degree=0,
        material_fitted_trace=True,
    )
    fine = dict(metadata, trace_segments=64, archive_sha256="shared lifts")
    coarse = dict(
        metadata,
        trace_segments=32,
        response_reuse=dict(
            prepared_archive_sha256="shared lifts", local_responses_are_shared=True
        ),
    )
    records.validate_pair([coarse, fine], [field, field])
    coarse["response_reuse"]["local_responses_are_shared"] = False
    with pytest.raises(ValueError, match="same declared local problem"):
        records.validate_pair([coarse, fine], [field, field])
    coarse["response_reuse"]["local_responses_are_shared"] = True
    other = SimpleNamespace(macro=mesh, meshes=(TriangleMesh(mesh.points * 0.5, mesh.cells),))
    with pytest.raises(ValueError, match="local geometry"):
        records.validate_pair([coarse, fine], [field, other])
