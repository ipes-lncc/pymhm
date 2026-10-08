"""Generated fixtures for explicit generation of notebook fields."""

import importlib
import json

import pytest


@pytest.fixture
def api(monkeypatch):
    """Load the downloaded inventory without acquiring any remote fixture inputs."""
    module = importlib.import_module("scripts.notebook_data")
    monkeypatch.setattr(module, "local_resource", lambda path, **kwargs: path)
    monkeypatch.setattr(
        module,
        "resource_glob",
        lambda folder, pattern, *, recursive=False, root=None: sorted(
            folder.rglob(pattern) if recursive else folder.glob(pattern)
        ),
    )
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


def test_nested_group_selection_and_ambiguous_basename(api, tmp_path):
    """Parent groups include descendants, while equal tutorial names require full paths."""
    base = tmp_path / "notebooks"
    sources = [base / "waves/helmholtz/tutorial.ipynb", base / "waves/maxwell/tutorial.ipynb"]
    for source in sources:
        source.parent.mkdir(parents=True)
        source.write_text("{}")
    assert api.select_notebooks(tmp_path, ["waves", "waves/maxwell"]) == sources
    assert api.select_notebooks(tmp_path, ["notebooks"]) == sources
    assert api.select_notebooks(tmp_path, ["waves/maxwell/tutorial.ipynb"]) == [sources[1]]
    with pytest.raises(ValueError, match="Ambiguous"):
        api.select_notebooks(tmp_path, ["tutorial"])


def test_historical_notebook_ids_cannot_be_duplicated_between_problem_groups(api, tmp_path):
    """One scientific archive contract cannot silently refer to two separate notebooks."""
    for name in ["darcy/14_neopz.ipynb", "elasticity/14_reference.ipynb"]:
        source = tmp_path / "notebooks" / name
        source.parent.mkdir(parents=True)
        source.write_text("{}")
    with pytest.raises(ValueError, match="Duplicate notebook identifier 14"):
        api.discover_notebooks(tmp_path)


@pytest.mark.parametrize(
    "code",
    [
        "from scripts.notebook_reproduction import notebook_workspace\n"
        "ROOT = notebook_workspace('darcy/21_spe10_data.ipynb')",
        "from scripts.notebook_reproduction import notebook_workspace as workspace\n"
        "ROOT = workspace(selector='notebooks/darcy/21_spe10_data.ipynb')",
        "import pymhm.io.workspace as ws\n"
        "ROOT = ws.notebook_workspace('darcy/21_spe10_data.ipynb')",
    ],
)
def test_downloaded_notebook_declares_its_catalogue_identity(api, tmp_path, code):
    """A renamed downloaded source uses its literal downloaded resource selector."""
    source = tmp_path / "downloaded.ipynb"
    source.write_text(json.dumps({"cells": [{"cell_type": "code", "source": code}]}))
    root = tmp_path / "work"
    assert api.notebook_selector(root, source) == "darcy/21_spe10_data.ipynb"
    assert api.selected_notebook_ids(root, [source]) == {"21"}


def test_ambiguous_or_undeclared_downloaded_identity(api, tmp_path):
    """Names and dynamic Python calls alone never inherit another notebook's recipe."""
    source = tmp_path / "21_downloaded.ipynb"
    source.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ("notebook_workspace('darcy/21_spe10_data.ipynb')"),
                    }
                ]
            }
        )
    )
    assert api.notebook_selector(tmp_path / "work", source) is None
    source.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": (
                            "from scripts.notebook_reproduction import notebook_workspace\n"
                            "notebook_workspace('darcy/21_spe10_data.ipynb')\n"
                            "notebook_workspace('darcy/40_spe10_adaptive.ipynb')"
                        ),
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="Conflicting"):
        api.notebook_selector(tmp_path / "work", source)


def test_conditioning_replay_inventories_only_its_original_native_fields(api, tmp_path):
    """Historical conditioning excludes unrelated native boundary-layer archives."""
    folder = tmp_path / "examples/results/rad-native"
    folder.mkdir(parents=True)
    (folder / "verification.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"kind": "conditioning", "archive": "first.npz"},
                    {"kind": "layer", "archive": "unrelated.npz"},
                    {"kind": "conditioning", "archive": "second.npz"},
                ]
            }
        )
    )
    assert api.required_archives(tmp_path, {"45"}) == {
        "45": {folder / "first.npz", folder / "second.npz"}
    }


def test_tetra_pk_inventory_includes_resolution_fields_used_by_profiles(api, tmp_path):
    """The displayed P4/P2/RT2 control is a dependency of the P5/RT3 notebook."""
    folder = tmp_path / "examples/results/tetra-pk"
    previous = tmp_path / "examples/results/reconstruction3d"
    folder.mkdir(parents=True)
    previous.mkdir(parents=True)
    (folder / "uniform.json").write_text(json.dumps({"rows": [{"archive": "uniform.npz"}]}))
    (folder / "fixed.json").write_text(json.dumps({"rows": [{"archive": "fixed.npz"}]}))
    (folder / "reconstruction-order.json").write_text(
        json.dumps({"archive": "rt3.npz", "parent_archive": "fixed.npz"})
    )
    (previous / "resolution.json").write_text(
        json.dumps(
            {"rows": [{"archive": "resolution-p2.npz"}, {"archive": "resolution-p4-rt2.npz"}]}
        )
    )
    expected = {
        folder / "uniform.npz",
        folder / "fixed.npz",
        folder / "rt3.npz",
        previous / "resolution-p2.npz",
        previous / "resolution-p4-rt2.npz",
    }
    dependencies = api.required_archives(tmp_path, {"64"})
    assert dependencies == {"64": expected}
    plan = api.dependency_plan(tmp_path, dependencies)
    assert plan["archive_count"] == 5
    with pytest.raises(ValueError, match="Missing 5 computed notebook field"):
        api.validate_archives(tmp_path, plan, 100)
