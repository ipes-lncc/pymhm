"""Generated contract fixtures for campaign resume and immutable acquired provenance."""

import copy
import hashlib
import json

import pytest

from examples.campaign_provenance import (
    file_digest,
    index_records,
    positive_integers,
    require_equal,
    source_validation,
    validation_record,
    verify_archive,
    verify_result,
)


@pytest.mark.parametrize("values", [[], [0], [-1], [True], [1.0], [2, 2], [4, 2]])
def test_invalid_mathematical_sequences(values):
    with pytest.raises(ValueError, match="distinct positive"):
        positive_integers(values, label="resolutions", increasing=True)


def test_exact_configuration_before_checkpoint(tmp_path):
    """Changes cannot relabel existing output, including reordered meshes and rules."""
    target = tmp_path / "campaign.json"
    configuration = dict(sizes=[2, 4], assembly=16, control=12, orders=[8, 10])
    target.write_text(json.dumps(configuration))
    before = target.read_bytes()
    assert positive_integers([10, 8], label="orders") == (10, 8)
    for key, changed in (
        ("sizes", [4, 2]),
        ("sizes", [2, 4, 8]),
        ("assembly", 10),
        ("control", 8),
        ("orders", [10, 8]),
        ("orders", [8, 8]),
        ("assembly", 16.0),
    ):
        requested = {**configuration, key: changed}
        with pytest.raises(ValueError, match="configuration changed"):
            require_equal(configuration, requested, label="configuration")
        assert target.read_bytes() == before
    require_equal(configuration, dict(reversed(list(configuration.items()))), label="config")
    require_equal({"value": 1}, {"value": 1}, label="config")
    with pytest.raises(ValueError):
        require_equal({"value": float("nan")}, {}, label="config")
    with pytest.raises(ValueError):
        require_equal({}, {"value": float("inf")}, label="config")
    with pytest.raises(ValueError):
        require_equal({"value": 1}, {"value": True}, label="config")


@pytest.fixture
def completed(tmp_path):
    """Create two tiny physical-file stand-ins and a fully identified norm row."""
    inputs = {}
    for name in ("field", "reference"):
        path = tmp_path / f"{name}.npz"
        path.write_bytes(f"generated-{name}".encode())
        inputs[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {
        "name": "case-1",
        "archive": "field.npz",
        "archive_sha256": inputs[tmp_path / "field.npz"],
        "reference_archive": "reference.npz",
        "reference_sha256": inputs[tmp_path / "reference.npz"],
    }
    return (
        {
            **identity,
            "norms": {"quadrature_8": {"difference": 0.1}, "quadrature_10": {"difference": 0.1}},
        },
        identity,
        inputs,
    )


def test_completed_row_validates_files_before_skip(completed):
    """A retained result cannot hide a changed case or changed reference payload."""
    row, identity, inputs = completed
    original = copy.deepcopy(row)
    verify_result(row, identity, archives=inputs, norm_orders=[8, 10])
    assert row == original
    for path, expected in inputs.items():
        before = path.read_bytes()
        assert verify_archive(path, expected) == file_digest(path)
        path.write_bytes(before + b"changed")
        with pytest.raises(ValueError, match="archive digest changed"):
            verify_result(row, identity, archives=inputs, norm_orders=[8, 10])
        path.write_bytes(before)
    with pytest.raises(FileNotFoundError):
        verify_archive(next(iter(inputs)).with_name("absent.npz"), "0" * 64)


def test_completed_row_identity_and_norm_contract(completed):
    row, identity, inputs = completed
    for changed in (
        {**identity, "reference_sha256": "changed"},
        {**identity, "archive": "other.npz"},
        {**identity, "new_missing_field": 1},
        {},
    ):
        with pytest.raises(ValueError):
            verify_result(row, changed, archives=inputs, norm_orders=[8, 10])
    for norms in (None, [], {"quadrature_8": {}}, {**row["norms"], "quadrature_12": {}}):
        with pytest.raises(ValueError, match="quadratures"):
            verify_result({**row, "norms": norms}, identity, archives=inputs, norm_orders=[8, 10])
    with pytest.raises(ValueError, match="input archives"):
        verify_result(row, identity, archives={}, norm_orders=[8, 10])


@pytest.mark.parametrize("bad", [{}, {"name": ""}, {"name": 2}, {"name": "a"}])
def test_duplicate_or_invalid_completed_identity(bad):
    with pytest.raises(ValueError, match="identity"):
        index_records([{"name": "a"}, bad], key="name")


def test_append_case_preserves_existing_rows(completed):
    row, identity, inputs = completed
    original = copy.deepcopy(row)
    rows = [row]
    indexed = index_records(rows, key="name")
    verify_result(indexed["case-1"], identity, archives=inputs, norm_orders=[8, 10])
    assert "case-2" not in indexed
    rows.append({**row, "name": "case-2"})
    assert rows[0] == original
    assert set(index_records(rows, key="name")) == {"case-1", "case-2"}


def test_source_review_preserves_acquisition_and_records_validation_separately(tmp_path):
    executed = {"numerical.py": "1" * 64, "driver.py": "2" * 64}
    current = {**executed, "driver.py": "3" * 64, "loader.py": "4" * 64}
    reviews = {
        "driver.py": {"before": "2" * 64, "after": "3" * 64, "reason": "resume validation"},
        "loader.py": {"before": None, "after": "4" * 64, "reason": "newly observed guard"},
    }
    target = tmp_path / "comparison.json"
    target.write_text(json.dumps({"source_sha256": executed, "cases": [{"value": 3}]}))
    acquired = target.read_bytes()
    checked = source_validation(
        executed, current, numerical_sources=["numerical.py", "loader.py"], reviewed=reviews
    )
    assert checked["executed_source_sha256"] == executed
    assert checked["newly_observed_sources"] == ["loader.py"]
    report = validation_record(
        target, configuration={"sizes": [2, 4]}, sources=checked, checked_results=["case-1"]
    )
    assert report["acquired_manifest_sha256"] == hashlib.sha256(acquired).hexdigest()
    assert report == validation_record(
        target, configuration={"sizes": [2, 4]}, sources=checked, checked_results=["case-1"]
    )
    assert target.read_bytes() == acquired
    assert json.loads(acquired)["source_sha256"] == executed
    assert (
        source_validation(executed, executed, numerical_sources=[], reviewed={})[
            "newly_observed_sources"
        ]
        == []
    )


def test_source_guard_cannot_be_dropped_or_numerically_relabelled():
    executed = {"numerical.py": "a", "driver.py": "b"}
    with pytest.raises(ValueError, match="removed"):
        source_validation(executed, {}, numerical_sources=[], reviewed={})
    with pytest.raises(ValueError, match="numerical source changed"):
        source_validation(
            executed,
            {**executed, "numerical.py": "c"},
            numerical_sources=["numerical.py"],
            reviewed={"numerical.py": {"before": "a", "after": "c", "reason": "not permitted"}},
        )
    with pytest.raises(ValueError, match="exact, explicit"):
        source_validation(executed, executed, numerical_sources=[], reviewed={"unused": {}})
    with pytest.raises(ValueError, match="exact, explicit"):
        source_validation(executed, {**executed, "new": "c"}, numerical_sources=[], reviewed={})
    for review in (
        {},
        {"before": "wrong", "after": "c", "reason": "guard"},
        {"before": "b", "after": "wrong", "reason": "guard"},
        {"before": "b", "after": "c", "reason": " "},
        {"before": "b", "after": "c", "reason": None},
    ):
        with pytest.raises(ValueError, match="observed bytes"):
            source_validation(
                executed,
                {**executed, "driver.py": "c"},
                numerical_sources=[],
                reviewed={"driver.py": review},
            )


def test_reported_norm_order_must_match_its_key(completed):
    """Reject contradictory nested quadrature metadata before retaining a norm result."""
    row, identity, inputs = completed
    row["norms"]["quadrature_8"]["quadrature_order"] = 8
    verify_result(row, identity, archives=inputs, norm_orders=[8, 10])
    for changed in (None, {"quadrature_order": 10}, {"quadrature_order": 8.0}):
        row["norms"]["quadrature_8"] = changed
        with pytest.raises(ValueError):
            verify_result(row, identity, archives=inputs, norm_orders=[8, 10])


def test_empty_or_nonfinite_norm_map_is_not_a_completed_result(completed):
    """Reject missing numerical content and nonfinite values, including nested metadata."""
    row, identity, inputs = completed
    for invalid in ({}, {"difference": float("nan")}, {"metadata": [float("inf")]}):
        changed = copy.deepcopy(row)
        changed["norms"]["quadrature_8"] = invalid
        with pytest.raises(ValueError):
            verify_result(changed, identity, archives=inputs, norm_orders=[8, 10])
