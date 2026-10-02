"""Generated fixtures for explicit generation of notebook fields."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def api():
    """Load the helper without optional libraries or archived fields."""
    path = Path(__file__).resolve().parents[1] / "scripts/notebook_data.py"
    spec = importlib.util.spec_from_file_location("notebook_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_fields_can_be_inventoried_before_generation(api, tmp_path):
    """Absent computed files are listed once even when several notebooks read them."""
    path = tmp_path / "examples/results/case/field.npz"
    plan = api.dependency_plan(tmp_path, {"01": {path}, "02": {path}})
    assert plan["archive_count"] == 1
    assert plan["payload_bytes"] == 0
    assert plan["missing"] == ["examples/results/case/field.npz"]
    assert plan["notebooks"]["01"] == plan["notebooks"]["02"]
    with pytest.raises(ValueError, match="Generate the corresponding public cases"):
        api.validate_archives(tmp_path, plan, 100)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"PK-data")
    assert api.archive_size(path) == 7
    api.validate_archives(tmp_path, plan, 7)
    generated = api.dependency_plan(tmp_path, {"01": {path}})
    assert generated["payload_bytes"] == 7
    assert generated["missing"] == []


@pytest.mark.parametrize(
    "name", ["outside.npz", "examples/results/a*.npz", "examples/results/a.csv"]
)
def test_invalid_archive_names(api, tmp_path, name):
    """Unrelated paths and wildcard patterns cannot broaden the declared selection."""
    with pytest.raises(ValueError, match="archive path"):
        api.dependency_plan(tmp_path, {"01": {tmp_path / name}})


def test_placeholders_and_checkout_escape_are_rejected(api, tmp_path):
    """A legacy pointer cannot masquerade as a computed physical field."""
    path = tmp_path / "bad.npz"
    path.write_bytes(api.PLACEHOLDER_HEADER + b"size 123\n")
    with pytest.raises(ValueError, match="placeholder"):
        api.archive_size(path)
    with pytest.raises(ValueError):
        api.dependency_plan(tmp_path / "nested", {"01": {path}})


@pytest.mark.parametrize("budget", [0, -1, 4])
def test_budget_is_explicit(api, tmp_path, budget):
    """Restoration does not silently enlarge the notebook execution budget."""
    path = tmp_path / "examples/results/field.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"PK123")
    plan = api.dependency_plan(tmp_path, {"01": {path}})
    with pytest.raises(ValueError):
        api.validate_archives(tmp_path, plan, budget)


def test_empty_selection_requires_no_computed_fields(api, tmp_path):
    """A records-only notebook can be validated without requesting unrelated data."""
    api.validate_archives(tmp_path, api.dependency_plan(tmp_path, {}), 1)


def test_record_location_is_independent_of_payload_presence(api, tmp_path):
    """Versioned records select the same relative path before and after generation."""
    record = tmp_path / "controls/comparison.json"
    record.parent.mkdir()
    own = record.parent / "own.npz"
    inherited = tmp_path / "reference.npz"
    own.with_suffix(".json").write_text("{}")
    inherited.with_suffix(".json").write_text("{}")
    assert api.record_archive(record, "own.npz") == own
    assert api.record_archive(record, "reference.npz") == inherited
    assert api.record_archive(record, "unlisted.npz") == record.parent / "unlisted.npz"


def test_cli_notebook_selection(api, tmp_path, monkeypatch, capsys):
    """Unknown notebook identifiers fail, while a records-only notebook needs no payloads."""
    folder = tmp_path / "notebooks"
    folder.mkdir()
    (folder / "47_well.ipynb").write_text("{}")
    monkeypatch.setattr(api, "__file__", str(tmp_path / "scripts/notebook_data.py"))
    monkeypatch.setattr(api, "required_archives", lambda root: {})
    monkeypatch.setattr(sys, "argv", ["notebook_data.py", "--notebook", "47", "--check"])
    api.main()
    assert json.loads(capsys.readouterr().out)["archive_count"] == 0
    monkeypatch.setattr(sys, "argv", ["notebook_data.py", "--notebook", "99"])
    with pytest.raises(SystemExit, match="2"):
        api.main()


def test_cli_missing_data_reports_explicit_generation(api, tmp_path, monkeypatch):
    """A source-only checkout fails clearly before attempting notebook execution."""
    monkeypatch.setattr(api, "__file__", str(tmp_path / "scripts/notebook_data.py"))
    monkeypatch.setattr(
        api, "required_archives", lambda root: {"01": {root / "examples/results/field.npz"}}
    )
    monkeypatch.setattr(sys, "argv", ["notebook_data.py", "--check"])
    with pytest.raises(SystemExit, match="2"):
        api.main()
