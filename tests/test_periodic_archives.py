"""Lightweight acquisition-manifest checks for the periodic scientific example."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_array_equal


def _example(name):
    """Load example dependencies explicitly under the portable test entry point."""
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_example("periodic_norms")
_example("verify_periodic")
_example("periodic_reference")
compare_periodic = _example("compare_periodic")


def _archive(tmp_path, monkeypatch):
    """Create distinct legacy/current fields so an implicit filename cannot pass."""
    monkeypatch.setattr(compare_periodic, "ROOT", tmp_path)
    monkeypatch.setattr(compare_periodic, "ARTIFACTS", tmp_path)
    folder = tmp_path / "examples/results"
    folder.mkdir(parents=True)
    path = tmp_path / "reference-q2-2-threads8.npz"
    np.savez(path, pressure=np.arange(25.0), residual=1e-12)
    np.savez(tmp_path / "reference-q2-2-order3-lor.npz", pressure=np.zeros(25), residual=0.0)
    record = {
        "degree": 2,
        "n": 2,
        "quadrature_order": 3,
        "solver": "low-order-refined-pyamg-cg",
        "archive": path.name,
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "relative_equation_residual": 1e-12,
    }
    metadata = folder / "periodic-reference-q2-2-order3.json"
    metadata.write_text(json.dumps(record))
    return path, metadata, record


def test_manifest_selects_and_verifies_the_executed_archive(tmp_path, monkeypatch):
    """The declared hash/filename selects current coefficients, with legacy syntax preserved."""
    path, metadata, _ = _archive(tmp_path, monkeypatch)
    field, actual = compare_periodic.load_reference("2:2:3:lor")
    assert actual == path
    assert_array_equal(field.pressure, np.arange(25.0))
    metadata.unlink()
    field, actual = compare_periodic.load_reference("2:2:3:lor")
    assert actual.name == "reference-q2-2-order3-lor.npz"
    assert_array_equal(field.pressure, np.zeros(25))


def test_reference_records_can_stay_in_a_separate_review_directory(tmp_path, monkeypatch):
    path, metadata, _ = _archive(tmp_path, monkeypatch)
    records = tmp_path / "review-records"
    records.mkdir()
    metadata.rename(records / metadata.name)
    monkeypatch.setattr(compare_periodic, "REFERENCE_RECORDS", records)
    field, actual = compare_periodic.load_reference("2:2:3:lor")
    assert actual == path
    assert_array_equal(field.pressure, np.arange(25.0))


@pytest.mark.parametrize("mutation", ["hash", "degree", "solver", "parent", "residual"])
def test_manifest_mismatch_is_rejected(tmp_path, monkeypatch, mutation):
    """Never silently load stale, differently discretized or relocated acquisition data."""
    _, metadata, record = _archive(tmp_path, monkeypatch)
    key, value = {
        "hash": ("archive_sha256", "0" * 64),
        "degree": ("degree", 3),
        "solver": ("solver", "other"),
        "parent": ("archive", "../outside.npz"),
        "residual": ("relative_equation_residual", 0.0),
    }[mutation]
    record[key] = value
    metadata.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest|digest|arrays"):
        compare_periodic.load_reference("2:2:3:lor")


@pytest.mark.parametrize(
    "pressure,residual",
    [(np.zeros(24), 0), (np.full(25, np.nan), 0), (np.zeros(25), np.inf), (np.zeros(25), -1)],
)
def test_invalid_legacy_arrays_are_rejected(tmp_path, monkeypatch, pressure, residual):
    """Dimension and finiteness contracts apply even to archives without a manifest."""
    monkeypatch.setattr(compare_periodic, "ARTIFACTS", tmp_path)
    np.savez(tmp_path / "reference-q2-2.npz", pressure=pressure, residual=residual)
    with pytest.raises(ValueError, match="arrays"):
        compare_periodic.load_reference("2:2")


@pytest.mark.parametrize("changed", ["coefficient", "epsilon", "source", "boundary", "provenance"])
def test_reference_material_and_case_provenance_are_checked(tmp_path, monkeypatch, changed):
    _, manifest, record = _archive(tmp_path, monkeypatch)
    case = _example("verify_periodic")
    record["source_sha256"] = {"examples/verify_periodic.py": case.fingerprint(Path(case.__file__))}
    record["case_conventions"] = case.case_conventions()
    if changed == "provenance":
        record["source_sha256"]["examples/verify_periodic.py"] = "0" * 64
    else:
        record["case_conventions"][changed] = 1.0 if changed == "epsilon" else "different"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="material/case"):
        compare_periodic.load_reference("2:2:3:lor")


def test_legacy_material_provenance_is_explicitly_unverified(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)
    _, _, verified = compare_periodic._load_reference("2:2:3:lor")
    assert verified is False
    case = _example("verify_periodic")
    with pytest.raises(ValueError, match="unavailable"):
        case.validate_case_provenance({})
    assert case.validate_case_provenance({}, allow_legacy=True) is False


def _sidecar_archive(tmp_path, monkeypatch, *, degree=2, order=4, assembly="elementwise"):
    """Persist a complete verified sidecar independently of the requested syntax."""
    monkeypatch.setattr(compare_periodic, "ROOT", tmp_path)
    monkeypatch.setattr(compare_periodic, "ARTIFACTS", tmp_path)
    suffix = "" if order is None else f"-order{order}"
    if assembly != "elementwise":
        suffix += f"-{assembly}"
    path = tmp_path / f"reference-q{degree}-2{suffix}.npz"
    np.savez(path, pressure=np.arange((2 * degree + 1) ** 2, dtype=float), residual=1e-12)
    case = _example("verify_periodic")
    record = dict(
        degree=degree,
        n=2,
        quadrature_order=max(4, degree + 1) if order is None else order,
        assembly=assembly,
        archive=path.name,
        archive_sha256=case.fingerprint(path),
        residual=1e-12,
        source_sha256={"examples/verify_periodic.py": case.fingerprint(Path(case.__file__))},
        case_conventions=case.case_conventions(),
    )
    manifest = path.with_suffix(".json")
    manifest.write_text(json.dumps(record))
    return path, manifest, record


@pytest.mark.parametrize(
    "specification,degree,order,assembly",
    [
        ("2:2", 2, None, "elementwise"),
        ("5:2", 5, None, "elementwise"),
        ("2:2:6", 2, 6, "elementwise"),
        ("2:2:6:separable", 2, 6, "separable"),
        ("2:2:6:lor", 2, 6, "lor"),
    ],
)
def test_sidecar_matches_default_or_explicit_quadrature_and_assembly(
    tmp_path, monkeypatch, specification, degree, order, assembly
):
    path, _, _ = _sidecar_archive(
        tmp_path, monkeypatch, degree=degree, order=order, assembly=assembly
    )
    field, actual, verified = compare_periodic._load_reference(specification)
    assert actual == path
    assert_array_equal(field.pressure, np.arange((2 * degree + 1) ** 2, dtype=float))
    assert verified is True


@pytest.mark.parametrize("key", ["quadrature_order", "assembly"])
@pytest.mark.parametrize("missing", [False, True])
def test_sidecar_rejects_different_or_missing_quadrature_and_assembly(
    tmp_path, monkeypatch, key, missing
):
    _, manifest, record = _sidecar_archive(tmp_path, monkeypatch, order=None)
    if missing:
        record.pop(key)
    else:
        record[key] = 6 if key == "quadrature_order" else "separable"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest"):
        compare_periodic.load_reference("2:2")


@pytest.mark.parametrize("key", ["quadrature_order", "assembly"])
def test_sidecar_rejects_acquisition_data_different_from_the_explicit_request(
    tmp_path, monkeypatch, key
):
    _, manifest, record = _sidecar_archive(tmp_path, monkeypatch, order=6, assembly="separable")
    record[key] = 4 if key == "quadrature_order" else "elementwise"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest"):
        compare_periodic.load_reference("2:2:6:separable")


def test_nodal_archive_without_sidecar_leaves_acquisition_provenance_unverified(
    tmp_path, monkeypatch
):
    path, manifest, _ = _sidecar_archive(tmp_path, monkeypatch, order=None)
    manifest.unlink()
    field, actual, verified = compare_periodic._load_reference("2:2")
    assert actual == path
    assert_array_equal(field.pressure, np.arange(25.0))
    assert verified is False


def test_reference_acquisition_accepts_review_destinations_and_reuses_verified_coefficients(
    tmp_path, monkeypatch
):
    acquisition = _example("periodic_reference")
    calls = []

    def solve(mesh, **kwargs):
        calls.append((mesh, kwargs))
        pressure = np.zeros((11, 11))
        pressure[1:-1, 1:-1] = np.arange(81).reshape(9, 9)
        return SimpleNamespace(
            pressure=pressure.ravel(),
            residual=1e-12,
            relative_equation_residual=1e-12,
            iterations=3,
            preconditioner_levels=2,
        )

    monkeypatch.setattr(acquisition, "solve_separable_krylov", solve)
    artifacts, records = tmp_path / "arrays", tmp_path / "records"
    acquisition.run(2, 10, refinement_precision="double", artifacts=artifacts, records=records)
    acquisition.run(2, 10, refinement_precision="double", artifacts=artifacts, records=records)
    assert len(calls) == 1
    assert calls[0][1]["degree"] == 5
    assert calls[0][1]["quadrature_order"] == 10
    record = json.loads((records / "periodic-reference-q5-2-order10.json").read_text())
    path = artifacts / record["archive"]
    assert record["archive_sha256"] == acquisition.fingerprint(path)
    assert record["source_sha256"] == acquisition.sources()
    assert record["assembly"] == "lor"
    assert record["refinement_precision"] == "double"
    assert record["pressure_dtype"] == np.dtype(float).str
    assert record["coefficient_precision_bits"] == np.finfo(float).nmant + 1
    assert record["operator_precision_bits"] == 53
    assert record["lockfile_sha256"] == acquisition.fingerprint(acquisition.ROOT / "pixi.lock")
    assert record["git_revision"]
    expected = np.zeros((11, 11))
    expected[1:-1, 1:-1] = np.arange(81).reshape(9, 9)
    with np.load(path) as arrays:
        assert_array_equal(arrays["pressure"], expected.ravel())
        assert_array_equal(
            arrays["basis_cardinal_matrix"],
            acquisition._executed_basis(5, 10)["basis_cardinal_matrix"],
        )


@pytest.mark.parametrize("degree,order", [(1, 2), (5, 6)])
@pytest.mark.parametrize("precision", ["double", "extended"])
def test_reference_acquisition_uses_the_requested_qk_space(tmp_path, degree, order, precision):
    """The public Q1/Q5 path matches direct conforming assembly on the same data."""
    from pymhm._legacy.models.darcy.separable import SeparableField, solve_separable_diffusion
    from pymhm.linalg.linear import SolverUnavailableError
    from pymhm.meshes.cartesian import CartesianMacroMesh

    acquisition = _example("periodic_reference")
    artifacts, records = tmp_path / "arrays", tmp_path / "records"
    if precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        with pytest.raises(SolverUnavailableError, match="wider"):
            acquisition.run(
                2,
                order,
                degree=degree,
                refinement_precision=precision,
                artifacts=artifacts,
                records=records,
            )
        assert not artifacts.exists() and not records.exists()
        return
    acquisition.run(
        2,
        order,
        degree=degree,
        refinement_precision=precision,
        artifacts=artifacts,
        records=records,
    )
    record = json.loads((records / f"periodic-reference-q{degree}-2-order{order}.json").read_text())
    direct = solve_separable_diffusion(
        CartesianMacroMesh(2),
        degree=degree,
        permeability=SeparableField(((1.0, 1.0), (acquisition.factor_x, acquisition.factor_y))),
        source=SeparableField(((np.sin, np.sin),)),
        quadrature_order=order,
    )
    with np.load(artifacts / record["archive"], allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["pressure"], direct.pressure, rtol=2e-9, atol=2e-13)
        assert arrays["basis_values_1d"].shape == (order, degree + 1)
        assert arrays["basis_cardinal_matrix"].shape == ((degree + 1) ** 2,) * 2
        assert record["basis_sha256"] == {
            name: acquisition._array_digest(arrays[name]) for name in record["basis_sha256"]
        }
    assert record["relative_equation_residual"] <= 1e-10
    assert record["degree"] == degree
    assert record["quadrature_order"] == order
    assert record["refinement_precision"] == precision
    for name, digest in record["source_sha256"].items():
        assert (artifacts / "acquisition-sources" / f"{digest}.py").read_bytes() == (
            acquisition.ROOT / name
        ).read_bytes()


@pytest.mark.parametrize(
    "arguments",
    [
        dict(n=0, order=10),
        dict(n=True, order=10),
        dict(n=2.0, order=10),
        dict(n=2, degree=0, order=10),
        dict(n=2, degree=True, order=10),
        dict(n=2, degree=1.5, order=10),
        dict(n=2, degree=1, order=1),
        dict(n=2, degree=5, order=5),
        dict(n=2, degree=1, order=2.5),
        dict(n=2, order=10, native_threads=0),
        dict(n=2, order=10, native_threads=True),
        dict(n=2, order=10, refinement_precision="automatic"),
    ],
)
def test_reference_acquisition_rejects_immediately_excluded_inputs(tmp_path, arguments):
    acquisition = _example("periodic_reference")
    with pytest.raises(ValueError, match="integer|precision"):
        acquisition.run(**arguments, artifacts=tmp_path, records=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("changed", ["sources", "lockfile"])
def test_reference_acquisition_rejects_changes_during_solve(tmp_path, monkeypatch, changed):
    """Publication cannot attach old provenance to a field executed during edits."""
    acquisition = _example("periodic_reference")
    original_sources, original_fingerprint = acquisition.sources, acquisition.fingerprint
    state = {"solved": False}

    def solve(*args, **kwargs):
        state["solved"] = True
        return SimpleNamespace(pressure=np.zeros(9), relative_equation_residual=0)

    def sources():
        return (
            {"changed": "0" * 64}
            if state["solved"] and changed == "sources"
            else original_sources()
        )

    def fingerprint(path):
        return (
            "0" * 64
            if state["solved"] and changed == "lockfile" and path.name == "pixi.lock"
            else original_fingerprint(path)
        )

    monkeypatch.setattr(acquisition, "solve_separable_krylov", solve)
    monkeypatch.setattr(acquisition, "sources", sources)
    monkeypatch.setattr(acquisition, "fingerprint", fingerprint)
    with pytest.raises(RuntimeError, match="sources or lockfile"):
        acquisition.run(
            2, 2, degree=1, refinement_precision="double", artifacts=tmp_path, records=tmp_path
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "changed", ["basis", "bits", "degree", "order", "operator", "boundary", "precision"]
)
def test_reference_reuse_checks_the_archived_basis_and_precision(tmp_path, changed):
    """A rehashed archive cannot substitute a different coefficient coordinate system."""
    acquisition = _example("periodic_reference")
    artifacts, records = tmp_path / "arrays", tmp_path / "records"
    acquisition.run(
        2, 2, degree=1, refinement_precision="double", artifacts=artifacts, records=records
    )
    record_path = records / "periodic-reference-q1-2-order2.json"
    record = json.loads(record_path.read_text())
    path = artifacts / record["archive"]
    if changed in {"basis", "boundary"}:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if changed == "basis":
            arrays["basis_cardinal_matrix"] = arrays["basis_cardinal_matrix"][:, ::-1]
            record["basis_sha256"]["basis_cardinal_matrix"] = acquisition._array_digest(
                arrays["basis_cardinal_matrix"]
            )
        else:
            arrays["pressure"][0] = 1
        np.savez_compressed(path, **arrays)
        record["archive_sha256"] = acquisition.fingerprint(path)
    else:
        key, value = {
            "bits": ("coefficient_precision_bits", np.finfo(float).nmant + 2),
            "degree": ("degree", 2),
            "order": ("quadrature_order", 3),
            "operator": ("operator_precision_bits", 64),
            "precision": ("refinement_precision", "extended"),
        }[changed]
        record[key] = value
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="contract|Dirichlet"):
        acquisition.run(
            2, 2, degree=1, refinement_precision="double", artifacts=artifacts, records=records
        )


def test_reference_cli_keeps_q5_and_order10_defaults(monkeypatch):
    acquisition = _example("periodic_reference")
    calls = []
    monkeypatch.setattr(sys, "argv", ["periodic_reference.py"])
    monkeypatch.setattr(acquisition, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    acquisition.main()
    assert [args for args, _ in calls] == [(512, 10, 1), (1024, 10, 1), (2048, 10, 1)]
    assert all(options["degree"] == 5 for _, options in calls)


@pytest.mark.parametrize(
    "changed", [None, "matrix", "nodes", "bits", "degree", "orientation", "owner", "missing"]
)
def test_new_reference_consumer_requires_the_executed_basis_contract(
    tmp_path, monkeypatch, changed
):
    """Compare/plot callers reject a rehashed coordinate or evaluator substitution."""
    acquisition = _example("periodic_reference")
    artifacts, records = tmp_path / "arrays", tmp_path / "records"
    acquisition.run(
        2, 2, degree=1, refinement_precision="double", artifacts=artifacts, records=records
    )
    monkeypatch.setattr(compare_periodic, "ARTIFACTS", artifacts)
    monkeypatch.setattr(compare_periodic, "REFERENCE_RECORDS", records)
    record_path = records / "periodic-reference-q1-2-order2.json"
    record = json.loads(record_path.read_text())
    path = artifacts / record["archive"]
    if changed in {"matrix", "nodes"}:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        name = "basis_values_1d" if changed == "matrix" else "basis_nodes"
        arrays[name] = arrays[name][::-1].copy()
        record["basis_sha256"][name] = acquisition._array_digest(arrays[name])
        np.savez_compressed(path, **arrays)
        record["archive_sha256"] = acquisition.fingerprint(path)
    elif changed == "bits":
        record["coefficient_precision_bits"] += 1
    elif changed == "degree":
        record["degree"] = 2
    elif changed == "orientation":
        record["basis_convention"] = "Q1 equidistant nodal cardinal functions; y index fastest"
    elif changed == "owner":
        record["source_sha256"]["src/pymhm/_legacy/models/darcy/cartesian.py"] = "0" * 64
    elif changed == "missing":
        record.pop("basis_sha256")
    record_path.write_text(json.dumps(record))
    if changed is not None:
        with pytest.raises(ValueError, match="contract|requested field"):
            compare_periodic.load_reference("1:2:2:lor")
    else:
        field, actual = compare_periodic.load_reference("1:2:2:lor")
        assert actual == path
        with np.load(path, allow_pickle=False) as arrays:
            np.testing.assert_allclose(
                field.evaluate(np.array([[0.5, 0.5]]))[0], arrays["pressure"][[4]], rtol=0, atol=0
            )


@pytest.mark.parametrize("changed", [None, "matrix", "bits"])
def test_verify_reference_sidecar_flow_enforces_the_new_basis_contract(tmp_path, changed):
    """The reference-only CLI validates a complete new sidecar before using its field."""
    acquisition = _example("periodic_reference")
    artifacts, records = tmp_path / "arrays", tmp_path / "records"
    acquisition.run(
        2, 2, degree=1, refinement_precision="double", artifacts=artifacts, records=records
    )
    record = json.loads((records / "periodic-reference-q1-2-order2.json").read_text())
    source = artifacts / record["archive"]
    path = artifacts / "reference-q1-2-order2-lor.npz"
    path.write_bytes(source.read_bytes())
    record["archive"] = path.name
    if changed == "matrix":
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["basis_reference_derivatives_1d"] *= -1
        record["basis_sha256"]["basis_reference_derivatives_1d"] = acquisition._array_digest(
            arrays["basis_reference_derivatives_1d"]
        )
        np.savez_compressed(path, **arrays)
    elif changed == "bits":
        record["coefficient_precision_bits"] += 1
    record["archive_sha256"] = acquisition.fingerprint(path)
    path.with_suffix(".json").write_text(json.dumps(record))
    output = tmp_path / "verification.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.verify_periodic",
            "--reference-only",
            "--reference-levels",
            "2",
            "--reference-degree",
            "1",
            "--reference-quadrature",
            "2",
            "--reference-assembly",
            "lor",
            "--artifacts",
            str(artifacts),
            "--output",
            str(output),
        ],
        cwd=acquisition.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if changed is None:
        assert result.returncode == 0, result.stderr
        actual = json.loads(output.read_text())["reference"][0]
        assert actual["basis_sha256"] == record["basis_sha256"]
        assert actual["coefficient_precision_bits"] == record["coefficient_precision_bits"]
    else:
        assert result.returncode != 0
        assert "basis or precision" in result.stderr
        assert not output.exists()


@pytest.mark.parametrize("name", ["verify_periodic", "periodic_reference", "compare_periodic"])
@pytest.mark.parametrize("module", [False, True])
def test_periodic_cli_supports_file_and_module_entrypoints(name, module):
    """A fresh interpreter resolves every example dependency in either supported form."""
    root = Path(__file__).resolve().parents[1]
    command = ["-m", f"examples.{name}"] if module else [str(root / "examples" / f"{name}.py")]
    result = subprocess.run(
        [sys.executable, *command, "--help"], cwd=root, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
