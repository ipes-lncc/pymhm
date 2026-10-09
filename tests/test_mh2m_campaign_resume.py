"""Small generated fields verify public campaign resume and archive-validation contracts."""

import copy
import importlib
import json
import sys
from types import SimpleNamespace

import pytest


def physical_norm(order):
    """Generate the complete physical-integrator schema without computing any FEM field."""
    result = dict(quadrature_order=order, overlay_area=1.0)
    for name in ("pressure", "flux", "gradient", "energy"):
        result[f"{name}_difference"] = 0.125
        result[f"reference_{name}_norm"] = 2.0
        result[f"{name}_relative_difference"] = 0.0625
    return result


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Prepare only byte-sized field stand-ins; numerical integration is not invoked."""
    module = importlib.reload(importlib.import_module("examples.compare_mh2m_cg3"))
    data = tmp_path / "data"
    directory = data / "cg3"
    directory.mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "DATA", data)
    # Model downloaded case companions with their own editable source bytes.
    package = importlib.import_module("examples")
    monkeypatch.setattr(package, "__file__", str(tmp_path / "examples/__init__.py"))
    for name in module.SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    for n, order in ((2, 16), (4, 16), (4, 12), (512, 16)):
        archive = directory / f"reference-cg3-n{n}-q{order}.npz"
        archive.write_bytes(archive.name.encode())
        metadata = dict(
            source_changed_during_run=False,
            archive=archive.name,
            archive_sha256=module.digest(archive),
            resolution=n,
            quadrature_degree=order,
            degree=3,
        )
        archive.with_suffix(".json").write_text(json.dumps(metadata))
    (data / "reference-n1024.npz").write_bytes(b"classical-cg1-fixture")
    cross = data / "crisscross"
    cross.mkdir()
    case = cross / "case.npz"
    case.write_bytes(b"case-fixture")
    row = dict(
        name="case",
        archive=case.name,
        archive_sha256=module.digest(case),
        method="MH2M",
        family="fixture",
    )
    (cross / "comparison.json").write_text(json.dumps({"cases": [row]}))
    (data / "case.npz").write_bytes(case.read_bytes())
    (data / "comparison.json").write_text(json.dumps({"cases": [row]}))
    calls = []
    monkeypatch.setattr(
        module.CubicTriangularField, "load", lambda path: SimpleNamespace(resolution=1)
    )
    monkeypatch.setattr(module, "load_field", lambda path: SimpleNamespace(resolution=1))
    monkeypatch.setattr(module, "common_triangles", lambda *args: None)
    monkeypatch.setattr(
        module.CrossedP1, "from_arrays", lambda *args: SimpleNamespace(resolution=1)
    )

    class Arrays:
        """Supply the archive context interface without allocating numerical arrays."""

        def __enter__(self):
            return {"vertices": None, "pressure": None}

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(module.np, "load", lambda *a, **k: Arrays())

    def difference(*args, **kwargs):
        """Record whether resume incorrectly recomputes a completed comparison."""
        calls.append(kwargs)
        return physical_norm(args[3])

    monkeypatch.setattr(module, "difference", difference)
    argv = ["compare", "--data", str(data), "--sizes", "2", "4"]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return module, data, calls, argv, difference


@pytest.mark.parametrize(
    "option",
    [
        ["--sizes", "4", "2"],
        ["--sizes", "2", "4", "8"],
        ["--assembly-order", "18"],
        ["--quadrature-control", "14"],
        ["--orders", "10", "8"],
        ["--orders", "8", "8"],
    ],
)
def test_changed_configuration_fails_without_checkpoint(campaign, monkeypatch, option):
    module, data, calls, argv, _ = campaign
    target = data / "cg3/comparison.json"
    before, count = target.read_bytes(), len(calls)
    monkeypatch.setattr(sys, "argv", argv + option)
    with pytest.raises(ValueError):
        module.main()
    assert target.read_bytes() == before
    assert len(calls) == count


def test_identical_resume_and_execution_options_do_not_recompute(campaign, monkeypatch):
    module, data, calls, argv, _ = campaign
    target = data / "cg3/comparison.json"
    before, count = target.read_bytes(), len(calls)
    monkeypatch.setattr(
        sys, "argv", argv + ["--workers", "7", "--stage", "cases", "--names", "case"]
    )
    module.main()
    assert target.read_bytes() == before
    assert len(calls) == count
    report = json.loads((data / "cg3/comparison-resume-validation.json").read_text())
    assert (target.parent / report["acquired_manifest"]).read_bytes() == before
    assert (
        module.digest(target.parent / report["acquired_manifest"])
        == report["acquired_manifest_sha256"]
    )


@pytest.mark.parametrize("changed", ["case", "reference", "metadata", "orders"])
def test_completed_data_change_cannot_hide_behind_skip(campaign, changed):
    module, data, calls, argv, _ = campaign
    target = data / "cg3/comparison.json"
    if changed == "case":
        (data / "crisscross/case.npz").write_bytes(b"changed")
    elif changed == "reference":
        (data / "cg3/reference-cg3-n4-q16.npz").write_bytes(b"changed")
    elif changed == "metadata":
        path = data / "cg3/reference-cg3-n4-q16.json"
        record = json.loads(path.read_text())
        record["new_acquisition_information"] = "changed"
        path.write_text(json.dumps(record))
    else:
        record = json.loads(target.read_text())
        record["cases"][0]["norms"].pop("quadrature_10")
        target.write_text(json.dumps(record))
    before, count = target.read_bytes(), len(calls)
    with pytest.raises(ValueError):
        module.main()
    assert target.read_bytes() == before
    assert len(calls) == count


def test_append_preserves_old_rows_and_validation_snapshot(campaign):
    module, data, calls, argv, _ = campaign
    target = data / "cg3/comparison.json"
    before = target.read_bytes()
    old = json.loads(before)
    path = data / "crisscross/comparison.json"
    manifest = json.loads(path.read_text())
    row = copy.deepcopy(manifest["cases"][0])
    row["name"] = "added"
    row["archive"] = "added.npz"
    (path.parent / row["archive"]).write_bytes(b"new-field")
    row["archive_sha256"] = module.digest(path.parent / row["archive"])
    manifest["cases"].append(row)
    path.write_text(json.dumps(manifest))
    count = len(calls)
    module.main()
    after = json.loads(target.read_text())
    assert after["cases"][:-1] == old["cases"]
    assert after["source_sha256"] == old["source_sha256"]
    assert len(calls) == count + 2
    report = json.loads((target.parent / "comparison-resume-validation.json").read_text())
    assert (target.parent / report["acquired_manifest"]).read_bytes() == before
    assert after["cases"][-1]["postprocessing_source_sha256"] == after["source_sha256"]


def test_complete_legacy_manifest_is_retained(campaign):
    module, data, calls, argv, _ = campaign
    target = data / "cg3/comparison.json"
    record = json.loads(target.read_text())
    record.pop("mathematical_configuration")
    target.write_text(json.dumps(record))
    before = target.read_bytes()
    module.main()
    assert target.read_bytes() == before


def test_reviewed_orchestration_keeps_acquired_source_hash(campaign, monkeypatch):
    module, data, calls, argv, _ = campaign
    target = data / "cg3/comparison.json"
    before, count = target.read_bytes(), len(calls)
    name = "examples/compare_mh2m_cg3.py"
    source = module.ROOT / name
    old_digest = module.digest(source)
    source.write_text("reviewed-validation-only-change")
    review = {name: dict(before=old_digest, after=module.digest(source), reason="resume checks")}
    review_path = module.ROOT / "review.json"
    review_path.write_text(json.dumps(review))
    monkeypatch.setattr(sys, "argv", argv + ["--resume-review", str(review_path)])
    module.main()
    assert target.read_bytes() == before
    assert len(calls) == count
    report = json.loads((target.parent / "comparison-resume-validation.json").read_text())
    assert report["source_validation"]["executed_source_sha256"][name] == old_digest
    assert report["source_validation"]["validation_source_sha256"][name] == module.digest(source)
    numeric = module.ROOT / "examples/mh2m_heterogeneous.py"
    numeric.write_text("changed-mathematical-loader")
    with pytest.raises(ValueError, match="numerical source changed"):
        module.main()
    assert target.read_bytes() == before


def test_controls_validate_completed_archive(campaign, monkeypatch):
    main, data, calls, argv, difference = campaign
    monkeypatch.setitem(sys.modules, "examples.compare_mh2m_cg3", main)
    module = importlib.reload(importlib.import_module("examples.compare_mh2m_cg3_controls"))
    path = main.ROOT / "examples/compare_mh2m_cg3_controls.py"
    path.write_text("control-fixture")
    monkeypatch.setattr(module, "load_field", lambda path: object())
    monkeypatch.setattr(module, "difference", difference)
    monkeypatch.setattr(sys, "argv", ["controls"])
    module.main()
    target = data / "cg3/structured-comparison.json"
    before, count = target.read_bytes(), len(calls)
    monkeypatch.setattr(sys, "argv", ["controls", "--workers", "9"])
    module.main()
    assert target.read_bytes() == before
    assert len(calls) == count

    (data / "case.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive digest"):
        module.main()
    assert target.read_bytes() == before
    assert len(calls) == count


def test_separate_output_preserves_archived_comparison(campaign, monkeypatch):
    module, data, calls, argv, _ = campaign
    target = data / "cg3/comparison.json"
    before = target.read_bytes()
    fresh = data / "independent-replay/comparison.json"
    monkeypatch.setattr(sys, "argv", argv + ["--output", str(fresh)])
    module.main()
    assert target.read_bytes() == before
    assert (
        json.loads(fresh.read_text())["cases"][0]["norms"]
        == json.loads(before)["cases"][0]["norms"]
    )


def test_public_validation_does_not_require_a_private_review(campaign, monkeypatch):
    cg3, data, calls, argv, _ = campaign
    module = importlib.reload(importlib.import_module("examples.validate_mh2m_campaign"))
    monkeypatch.setattr(module, "ROOT", cg3.ROOT)
    for name in ("validate_mh2m_campaign", "mh2m_crisscross_campaign", "compare_mh2m_cg3_controls"):
        (cg3.ROOT / f"examples/{name}.py").write_text("generated-validator-source")
    target = data / "cg3/comparison.json"
    acquired = json.loads(target.read_text())
    acquired["source_changed_during_run"] = True
    acquired["acquisition_sources"] = [
        {"source_sha256": acquired["source_sha256"], "source_changed_during_run": True}
    ]
    target.write_text(json.dumps(acquired))
    before, count = target.read_bytes(), len(calls)
    report = module.validate_campaign(data, "cg3")
    assert report["checked_results"] == 4
    assert target.read_bytes() == before
    assert len(calls) == count
    checked = json.loads((target.parent / "comparison-resume-validation.json").read_text())
    assert (
        checked["source_validation"]["executed_source_sha256"]
        == json.loads(before)["source_sha256"]
    )
    assert "not a numerical replay" in checked["source_validation"]["scope"]
    assert checked["source_validation"]["recorded_source_changed_during_run"] is True
    assert (
        checked["source_validation"]["acquisition_source_phases"] == acquired["acquisition_sources"]
    )


def test_validation_rejects_manifest_change_during_check(campaign):
    module, data, calls, argv, _ = campaign
    target = data / "cg3/comparison.json"
    before = module.digest(target)
    target.write_text(target.read_text() + " ")
    with pytest.raises(ValueError, match="changed during validation"):
        module.save_validation(target, {}, {}, [], before)


@pytest.fixture
def related_record(tmp_path):
    """Create a material-independent record with no numerical solve or field allocation."""
    from examples.campaign_provenance import file_digest

    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"reference")
    field = tmp_path / "mh2m-n1-r16-g1-l8.npz"
    field.write_bytes(b"field")
    references = [
        dict(
            resolution=2,
            assembly_order=8,
            archive=reference.name,
            archive_sha256=file_digest(reference),
        )
    ]
    row = dict(
        name="case",
        archive=field.name,
        archive_sha256=file_digest(field),
        method="MH2M",
        family="global-refinement",
        macro_resolution=1,
        local_refinement=16,
        local_degree=1,
        Gamma_degree=1,
        Gamma_segments=1,
        Lambda_degree=0,
        Lambda_segments=8,
        assembly_order=8,
        reference_resolution=2,
        norms=physical_norm(6),
        norms_quadrature_check=physical_norm(8),
    )
    configuration = dict(references=[2], resolutions=[1], assembly_order=8, norm_orders=[6, 8])
    return tmp_path, dict(
        references=references, cases=[row], mathematical_configuration=configuration
    )


def test_structured_existing_fields_checked_before_reuse(related_record):
    from examples.mh2m_heterogeneous import validate_campaign_resume

    output, record = related_record
    original = copy.deepcopy(record)
    config = validate_campaign_resume(record, output)
    assert record == original
    assert config == record["mathematical_configuration"]
    legacy = {key: value for key, value in record.items() if key != "mathematical_configuration"}
    assert validate_campaign_resume(legacy, output) == config
    with pytest.raises(ValueError, match="configuration changed"):
        validate_campaign_resume(record, output, {**config, "assembly_order": 10})
    for key in ("cases", "references"):
        archive = output / record[key][0]["archive"]
        before = archive.read_bytes()
        archive.write_bytes(b"changed")
        with pytest.raises(ValueError, match="archive digest"):
            validate_campaign_resume(record, output)
        archive.write_bytes(before)
    record["cases"][0].pop("norms_quadrature_check")
    with pytest.raises(ValueError, match="quadratures"):
        validate_campaign_resume(record, output)


def test_crisscross_existing_fields_and_reference_checked_before_skip(related_record, monkeypatch):
    import examples.mh2m_crisscross_campaign as module

    output, record = related_record
    record["cases"][0]["assembly_order"] = 10
    monkeypatch.setattr(module, "configurations", lambda: [("case", "MH2M", 1, 16, 1, 8)])
    reference = copy.deepcopy(record["references"])
    module.validate_campaign_resume(record, output, reference)
    changed = copy.deepcopy(reference)
    changed[0]["resolution"] = 4
    with pytest.raises(ValueError, match="reference acquisitions"):
        module.validate_campaign_resume(record, output, changed)
    field = output / record["cases"][0]["archive"]
    before = field.read_bytes()
    field.write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive digest"):
        module.validate_campaign_resume(record, output, reference)
    field.write_bytes(before)
    record["cases"][0]["Lambda_segments"] = 4
    with pytest.raises(ValueError, match="discretization"):
        module.validate_campaign_resume(record, output, reference)


def test_current_data_attribution_is_separate_from_acquisition_flags():
    from examples.mh2m_heterogeneous import data_conventions

    conventions = data_conventions()
    assert conventions["gamma"] == 1.8
    assert conventions["epsilon"] == "1/14"
    assert "Barros (2022)" in conventions["gamma_provenance"]
    assert "(4.1)" in conventions["source_provenance"]
    assert "original figure arrays are not supplied" in conventions["historical_scope"]


def test_crisscross_new_output_retains_valid_reference_paths(tmp_path, monkeypatch):
    import examples.mh2m_crisscross_campaign as module
    from examples.campaign_provenance import file_digest

    references = tmp_path / "examples/results/mh2m-heterogeneous"
    references.mkdir(parents=True)
    archive = references / "reference-n2.npz"
    archive.write_bytes(b"reference")
    metadata = dict(resolution=2, archive=archive.name, archive_sha256=file_digest(archive))
    (references / "comparison.json").write_text(json.dumps({"references": [metadata]}))
    output = tmp_path / "new-acquisition"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "hashes", lambda: {"generated-fixture": "unchanged"})
    monkeypatch.setattr(module, "configurations", lambda: [("case", "MHM", 1, 2, 1, 1)])

    def acquire(*args):
        """Supply a complete generated acquisition for the pre-checkpoint validator."""
        output.mkdir(parents=True, exist_ok=True)
        field = output / "case.npz"
        field.write_bytes(b"generated-field")
        return dict(
            name="case",
            method="MHM",
            macro_resolution=1,
            local_refinement=2,
            local_degree=1,
            Gamma_degree=None,
            Gamma_segments=None,
            Lambda_degree=0,
            Lambda_segments=1,
            assembly_order=10,
            archive=field.name,
            archive_sha256=file_digest(field),
        )

    monkeypatch.setattr(module, "acquire", acquire)
    monkeypatch.setattr(sys, "argv", ["campaign", "--stage", "acquire", "--output", str(output)])
    module.main()
    record = json.loads((output / "comparison.json").read_text())
    assert (output / record["references"][0]["archive"]).resolve() == archive.resolve()


def test_native_reference_evidence_is_linked_without_reexecution(tmp_path):
    from examples.campaign_provenance import file_digest
    from examples.validate_mh2m_campaign import independent_reference_evidence

    assert independent_reference_evidence(tmp_path) is None
    archive = tmp_path / "reference.npz"
    archive.write_bytes(b"native-verified-field")
    path = tmp_path / "cg1-native-verification.json"
    record = dict(
        method="Independent native assembly",
        scope="Discrete field comparison",
        source_changed_during_comparison=False,
        fields=[dict(archive=archive.name, archive_sha256=file_digest(archive))],
    )
    path.write_text(json.dumps(record))
    evidence = independent_reference_evidence(tmp_path)
    assert evidence["record_sha256"] == file_digest(path)
    assert "does not execute the native solver" in evidence["linkage_scope"]
    archive.write_bytes(b"changed-field")
    with pytest.raises(ValueError, match="archive digest"):
        independent_reference_evidence(tmp_path)
    record["source_changed_during_comparison"] = True
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="source guard"):
        independent_reference_evidence(tmp_path)


def test_physical_schema_requires_complete_nonnegative_norms():
    from examples.mh2m_campaign_contracts import validate_difference_norm

    valid = physical_norm(8)
    valid["metadata"] = {"name": "kept", "samples": [1, 2, 3], "available": True}
    before = copy.deepcopy(valid)
    validate_difference_norm(valid, 8)
    assert valid == before
    for field in valid.keys() - {"metadata"}:
        incomplete = dict(valid)
        incomplete.pop(field)
        with pytest.raises(ValueError, match="missing required"):
            validate_difference_norm(incomplete, 8)
    for field, value in (
        ("quadrature_order", 10),
        ("overlay_area", 0),
        ("overlay_area", float("inf")),
        ("pressure_difference", -1),
        ("flux_difference", None),
        ("reference_pressure_norm", float("nan")),
        ("reference_gradient_norm", True),
        ("energy_difference", "1.0"),
        ("gradient_difference", 10**400),
        ("energy_relative_difference", -1),
        ("flux_relative_difference", None),
        ("pressure_relative_difference", float("inf")),
    ):
        with pytest.raises(ValueError):
            validate_difference_norm({**valid, field: value}, 8)


def test_zero_reference_norm_keeps_undefined_ratio_valid():
    from examples.mh2m_campaign_contracts import validate_difference_norm

    values = physical_norm(8)
    for name in ("pressure", "flux", "gradient", "energy"):
        values[f"reference_{name}_norm"] = 0.0
        values[f"{name}_relative_difference"] = None
    before = copy.deepcopy(values)
    validate_difference_norm(values, 8)
    assert values == before
    with pytest.raises(ValueError, match="zero reference norm"):
        validate_difference_norm({**values, "pressure_relative_difference": 0.0}, 8)
    with pytest.raises(ValueError, match="positive reference norm"):
        validate_difference_norm({**values, "reference_pressure_norm": 1.0}, 8)


@pytest.mark.parametrize("defect", ["empty", "missing", "nan", "infinite-metadata"])
def test_incomplete_numeric_checkpoint_cannot_be_skipped(campaign, defect):
    module, data, calls, argv, _ = campaign
    target = data / "cg3/comparison.json"
    record = json.loads(target.read_text())
    norms = record["cases"][0]["norms"]
    if defect == "empty":
        record["cases"][0]["norms"] = {key: {} for key in norms}
    elif defect == "missing":
        norms["quadrature_8"].pop("pressure_difference")
    elif defect == "nan":
        norms["quadrature_8"]["flux_difference"] = float("nan")
    else:
        norms["quadrature_8"]["metadata"] = {"values": [float("inf")]}
    target.write_text(json.dumps(record))
    before, count = target.read_bytes(), len(calls)
    with pytest.raises(ValueError):
        module.main()
    assert target.read_bytes() == before
    assert len(calls) == count


def test_new_nonfinite_norm_is_rejected_before_checkpoint(campaign, monkeypatch):
    module, data, calls, argv, _ = campaign
    archived = data / "cg3/comparison.json"
    original = archived.read_bytes()
    fresh = data / "new-invalid-comparison.json"

    def invalid(*args, **kwargs):
        """Emulate an integration failure without executing a numerical solve."""
        values = physical_norm(args[3])
        values["flux_difference"] = float("nan")
        return values

    monkeypatch.setattr(module, "difference", invalid)
    monkeypatch.setattr(sys, "argv", argv + ["--output", str(fresh)])
    with pytest.raises(ValueError):
        module.main()
    assert archived.read_bytes() == original
    assert json.loads(fresh.read_text())["increments"] == []
