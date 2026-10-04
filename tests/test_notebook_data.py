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
    assert api.required_archives(tmp_path, set()) == {}
    assert api.required_archives(tmp_path, {"47"}) == {}


def test_selected_notebook_ignores_unrelated_missing_manifests(api, tmp_path):
    """A selected family inventories exact payloads without unrelated scientific records."""
    folder = tmp_path / "examples/results/neopz"
    folder.mkdir(parents=True)
    (folder / "comparison.json").write_text(
        json.dumps({"rows": [{"archive": "field.npz", "archive_sha256": "recorded-hash"}]})
    )
    dependencies = api.required_archives(tmp_path, {"14", "21"})
    assert dependencies == {
        "14": {folder / "field.npz"},
        "21": {tmp_path / "examples/results/spe10/layer-36.npz"},
    }
    plan = api.dependency_plan(tmp_path, dependencies)
    assert plan["missing"] == [
        "examples/results/neopz/field.npz",
        "examples/results/spe10/layer-36.npz",
    ]
    with pytest.raises(ValueError, match="Missing 2 computed notebook field"):
        api.validate_archives(tmp_path, plan, 100)
    with pytest.raises(FileNotFoundError, match="elasticity-reference.json"):
        api.required_archives(tmp_path)
    with pytest.raises(FileNotFoundError, match="elasticity-reference.json"):
        api.required_archives(tmp_path, {"14", "15"})


def test_quarter_notebook_preflight_includes_point_fields_and_series(api, tmp_path):
    """Physical point replay and convergence inputs are checked before the kernel."""
    point = tmp_path / "examples/results/quarter-five-spot"
    reference = point / "reference"
    reference.mkdir(parents=True)
    (reference / "comparison.json").write_text(json.dumps({"cases": [{"fields": "native.npz"}]}))
    (reference / "classical-convergence.json").write_text(
        json.dumps({"levels": [{"fields": "classical.npz"}], "mhm_trace_enrichment": []})
    )
    (point / "point-wells.json").write_text(
        json.dumps({"rows": [{"field_archive": "point-fields.npz"}]})
    )
    (point / "point-convergence.json").write_text(
        json.dumps({"rows": [{"archive": "point-convergence-n8.npz"}]})
    )
    dependencies = api.required_archives(tmp_path, {"22"})
    expected = {
        reference / "native.npz",
        reference / "classical.npz",
        point / "point-fields.npz",
        point / "point-convergence-n8.npz",
    }
    assert dependencies == {"22": expected}
    plan = api.dependency_plan(tmp_path, dependencies)
    assert len(plan["missing"]) == 4
    with pytest.raises(ValueError, match="Missing 4 computed"):
        api.validate_archives(tmp_path, plan, 100)


def test_nested_notebook_checks_selected_archive_and_each_displayed_image(api, tmp_path):
    """A selected current field and its PNGs are required before the notebook kernel."""
    folder = tmp_path / "examples/results"
    folder.mkdir(parents=True)
    selected = folder / "nested-current/selected.npz"
    (folder / "nested.json").write_text(
        json.dumps({"display_field": {"archive": "nested-current/selected.npz"}})
    )
    dependencies = api.required_archives(tmp_path, {"36"})
    assert dependencies == {"36": {selected}}
    images = api.required_images(tmp_path, {"36"})
    assert len(images["36"]) == 2
    plan = api.dependency_plan(tmp_path, dependencies, images)
    with pytest.raises(ValueError, match="Missing 1 computed"):
        api.validate_archives(tmp_path, plan, 100)
    selected.parent.mkdir()
    selected.write_bytes(b"PK-data")
    plan = api.dependency_plan(tmp_path, dependencies, images)
    assert not plan["missing"]
    with pytest.raises(ValueError, match="Missing 2 notebook image"):
        api.validate_archives(tmp_path, plan, 100)
    for image in images["36"]:
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"PNG-data")
    api.validate_archives(tmp_path, api.dependency_plan(tmp_path, dependencies, images), 100)
    # The selector must stay tied to the current record, without collecting an
    # unrelated absent campaign or all ten numerical payloads.
    (folder / "nested.json").unlink()
    with pytest.raises(FileNotFoundError, match="nested.json"):
        api.required_archives(tmp_path, {"36"})


@pytest.mark.parametrize("notebook,name", [("33", "darcy-rt"), ("41", "rad3d")])
def test_selected_grouped_notebook_reads_only_its_manifest(api, tmp_path, notebook, name):
    """A selected dimension family preserves the notebook's last-row field selector."""
    folder = tmp_path / "examples/results"
    folder.mkdir(parents=True)
    (folder / f"{name}.json").write_text(
        json.dumps({"rows": [{"fields": "coarse.npz"}, {"fields": "fine.npz"}]})
    )
    assert api.required_archives(tmp_path, {notebook}) == {notebook: {folder / "fine.npz"}}


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
    for path in api.required_images(tmp_path, {"47"})["47"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    monkeypatch.setattr(sys, "argv", ["notebook_data.py", "--notebook", "47", "--check"])
    api.main()
    assert json.loads(capsys.readouterr().out)["archive_count"] == 0
    monkeypatch.setattr(sys, "argv", ["notebook_data.py", "--notebook", "99"])
    with pytest.raises(SystemExit, match="2"):
        api.main()


def test_cli_selection_uses_only_requested_manifests(api, tmp_path, monkeypatch, capsys):
    """The CLI selects manifests before loading and still requires generated payloads."""
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    (notebooks / "14_neopz.ipynb").write_text("{}")
    folder = tmp_path / "examples/results/neopz"
    folder.mkdir(parents=True)
    (folder / "comparison.json").write_text(json.dumps({"rows": [{"archive": "field.npz"}]}))
    monkeypatch.setattr(api, "__file__", str(tmp_path / "scripts/notebook_data.py"))
    monkeypatch.setattr(sys, "argv", ["notebook_data.py", "--notebook", "14"])
    api.main()
    assert json.loads(capsys.readouterr().out)["missing"] == ["examples/results/neopz/field.npz"]
    monkeypatch.setattr(sys, "argv", ["notebook_data.py", "--notebook", "14", "--check"])
    with pytest.raises(SystemExit, match="2"):
        api.main()
    capsys.readouterr()
    (folder / "field.npz").write_bytes(b"PK-data")
    for path in api.required_images(tmp_path, {"14"})["14"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    api.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["payload_bytes"] == 7
    assert plan["missing"] == []


def test_cli_missing_data_reports_explicit_generation(api, tmp_path, monkeypatch):
    """A source-only checkout fails clearly before attempting notebook execution."""
    adaptive = tmp_path / "examples/results/spe10-adaptive/published/adaptive.json"
    adaptive.parent.mkdir(parents=True)
    adaptive.write_text("[]")
    monkeypatch.setattr(api, "__file__", str(tmp_path / "scripts/notebook_data.py"))
    monkeypatch.setattr(
        api,
        "required_archives",
        lambda root, notebooks=None: {"01": {root / "examples/results/field.npz"}},
    )
    monkeypatch.setattr(sys, "argv", ["notebook_data.py", "--check"])
    with pytest.raises(SystemExit, match="2"):
        api.main()


def test_present_fields_do_not_hide_missing_notebook_images(api, tmp_path):
    """A selected notebook fails before execution when its displayed PNG is absent."""
    field = tmp_path / "examples/results/spe10/layer-36.npz"
    field.parent.mkdir(parents=True)
    field.write_bytes(b"PK-data")
    image = tmp_path / "docs/figures/spe10/layers.png"
    images = {"21": {image}, "23": {image}}
    plan = api.dependency_plan(tmp_path, {"21": {field}}, images)
    assert plan["missing"] == []
    assert plan["image_count"] == 1
    assert plan["missing_images"] == ["docs/figures/spe10/layers.png"]
    with pytest.raises(ValueError, match="Missing 1 notebook image asset"):
        api.validate_archives(tmp_path, plan, 100)
    image.parent.mkdir(parents=True)
    image.write_bytes(b"PNG-data")
    available = api.dependency_plan(tmp_path, {"21": {field}}, images)
    assert available["image_bytes"] == 8
    assert available["notebook_images"]["21"] == available["notebook_images"]["23"]
    api.validate_archives(tmp_path, available, 15)
    with pytest.raises(ValueError, match="exceeding"):
        api.validate_archives(tmp_path, available, 14)


@pytest.mark.parametrize("name", ["outside.png", "docs/figures/a*.png", "docs/figures/a.csv"])
def test_invalid_notebook_image_names(api, tmp_path, name):
    """Only explicit figure paths can enter the selected notebook input budget."""
    with pytest.raises(ValueError, match="image path"):
        api.dependency_plan(tmp_path, {}, {"21": {tmp_path / name}})


def test_image_selection_precedes_unrelated_adaptive_manifest(api, tmp_path):
    """Selecting layer36 does not require the adaptive campaign's mesh-page record."""
    expected = {
        tmp_path / "docs/figures/spe10/volume-and-slice.png",
        tmp_path / "docs/figures/spe10/layers.png",
        tmp_path / "docs/figures/reservoir-papers/l07-figure-5.png",
        tmp_path / "docs/figures/reservoir-papers/l13-figure-18.png",
    }
    assert api.required_images(tmp_path, {"21"}) == {"21": expected}
    assert api.required_images(tmp_path, set()) == {}
    record = tmp_path / "examples/results/spe10-adaptive/published/adaptive.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps([{}] * 5))
    images = api.required_images(tmp_path, {"40"})["40"]
    assert tmp_path / "docs/figures/spe10-adaptive/meshes.png" in images
    assert tmp_path / "docs/figures/spe10-adaptive/meshes-2.png" in images
    assert tmp_path / "docs/figures/spe10-adaptive/meshes-3.png" not in images
    assert "40" in api.required_images(tmp_path)
