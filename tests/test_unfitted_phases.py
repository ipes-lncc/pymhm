"""Original saddle acceptance, bounded local lifetime and executed-field replay."""

import json
import shutil
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import examples.unfitted_phases as owner
from examples.archive_precision import precision_fields
from examples.unfitted_convergence import error_norms, smooth_field, smooth_source
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.core.contracts import LocalProblem
from pymhm.linalg.linear import LinearSolveError, SolverUnavailableError


def acquisition(directory: Path, *, threads: int = 1, precision: str = "double"):
    """Use small admissible trace/local pairs on the complete printed macro mesh."""
    return owner.UnfittedAcquisition(
        directory,
        refinement=2,
        names=["ell0-s1", "ell1-s2"],
        degree=3,
        assembly_order=5,
        norm_orders=[5, 7],
        native_threads=threads,
        refinement_precision=precision,
    )


@pytest.fixture(scope="module")
def corrected_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Inject a complementary arithmetic defect; preserve source, moments and spaces."""
    run = acquisition(tmp_path_factory.mktemp("unfitted-corrected"))
    run.condense()
    run.solve()
    reconstruct = LocalProblem.reconstruct

    def complementary(problem, *args, **kwargs):
        pressure = reconstruct(problem, *args, **kwargs)
        if np.any(problem.load):
            direction = (np.arange(len(problem.load)) % 2).astype(pressure.dtype)
            direction -= problem.kernel[:, 0] * (problem.constraints.T @ direction)[0]
            return pressure + np.asarray(1e-7, dtype=pressure.dtype) * direction[:, None]
        return pressure

    with pytest.MonkeyPatch.context() as context:
        context.setattr(LocalProblem, "reconstruct", complementary)
        run.reconstruct()
    assert run.record["refinement"]["accepted"]
    assert 0 < run.record["refinement"]["steps"] <= 2
    return run.directory


@pytest.fixture
def corrected_acquisition(tmp_path: Path, corrected_snapshot: Path) -> owner.UnfittedAcquisition:
    """Reload private corrected archives so mutation tests share no files or records."""
    shutil.copytree(corrected_snapshot, tmp_path, dirs_exist_ok=True)
    return acquisition(tmp_path)


def rewrite_archive(path: Path, **arrays) -> str:
    """Replace one acquisition file while preserving explicit caller-selected entries."""
    with np.load(path, allow_pickle=False) as original:
        data = {key: original[key] for key in original.files}
    data.update(arrays)
    owner._npz(path, **data)
    return owner.fingerprint(path)


@pytest.mark.parametrize("precision", ["double", "extended"])
def test_two_pass_fields_match_full_shared_darcy_and_original_equations(tmp_path, precision):
    if precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("requires a wider native long-double significand")
    run = acquisition(tmp_path, precision=precision)
    for stage in (run.condense, run.solve, run.reconstruct, run.norms):
        stage()
    assert run.record["complete"]
    for name, (ell, segments) in run.cases.items():
        expected = solve_darcy(
            run.mesh,
            skeleton=run._skeleton(ell, segments),
            degree=3,
            local_refinement=2,
            source=smooth_source,
            quadrature_order=5,
            local_refinement_precision=precision,
        )
        arrays = run._field(name)
        assert_array_equal(arrays["macro_points"], expected.skeleton.mesh.points)
        actual = run._solution(name)
        assert_allclose(actual["trace"], expected.hybrid.trace, rtol=2e-11, atol=2e-11)
        for cell, pressure in enumerate(expected.pressure):
            assert_allclose(
                owner._restore(arrays, f"pressure_{cell}", run.dtype),
                pressure,
                rtol=2e-11,
                atol=2e-11,
            )
            retained = owner._restore(arrays, f"retained_basis_{cell}", run.dtype)
            stored = owner._restore(run._cell(cell), "retained_basis", run.dtype)
            assert_array_equal(retained, stored)
        original = run.record["original_equations"][name]
        assert original["accepted"] and original["original_free_equation_relative_residual"] < 1e-10
        assert len(original["cells"]) == 16
        row = next(row for row in run.record["cases"] if row["name"] == name)
        assert_allclose(
            row["norms"]["quadrature_7"]["gradient_absolute"],
            error_norms(expected, smooth_field, 7)["gradient_absolute"],
            rtol=2e-12,
        )
    resumed = acquisition(tmp_path, precision=precision)
    resumed.condense()
    resumed.solve()
    resumed.reconstruct()
    assert resumed.record["acquisition_id"] == run.record["acquisition_id"]


def test_replay_uses_acquisition_operator_and_basis_under_two_threads(tmp_path):
    run = acquisition(tmp_path)
    run.condense()
    run.solve()
    run.reconstruct()
    expected = {name: run._field(name) for name in run.cases}
    replay = acquisition(tmp_path, threads=2)
    replay.record["fields"] = {}
    replay.reconstruct()
    for name in run.cases:
        actual = replay._field(name)
        for cell in range(16):
            assert_allclose(
                owner._restore(actual, f"pressure_{cell}", replay.dtype),
                owner._restore(expected[name], f"pressure_{cell}", run.dtype),
                rtol=2e-11,
                atol=2e-11,
            )
        assert replay.record["original_equations"][name]["accepted"]


def test_no_previous_macro_lifts_or_original_matrix_survive_condensation(tmp_path, monkeypatch):
    """A new native factor starts only after the preceding response is released."""
    previous = []
    condense = LocalProblem.condense

    def tracking(problem, *args, **kwargs):
        assert all(reference() is None for reference in previous)
        response = condense(problem, *args, **kwargs)
        previous[:] = [weakref.ref(response), weakref.ref(problem.matrix)]
        return response

    monkeypatch.setattr(LocalProblem, "condense", tracking)
    acquisition(tmp_path).condense()
    assert all(reference() is None for reference in previous)


def test_original_physical_saddle_gate_rejects_a_constant_pressure_shift(tmp_path, monkeypatch):
    """Kernel shifts leave volume rows intact but violate original weak boundary rows."""
    run = acquisition(tmp_path)
    run.condense()
    run.solve()
    reconstruct = LocalProblem.reconstruct

    def shifted(problem, *args, **kwargs):
        return reconstruct(problem, *args, **kwargs) + 0.1

    monkeypatch.setattr(LocalProblem, "reconstruct", shifted)
    with pytest.raises(LinearSolveError, match="original unfitted saddle"):
        run.reconstruct()
    assert not run.record["fields"]
    assert all(not row["accepted"] for row in run.record["original_equations"].values())
    assert not list(run.cell_directory.glob("*.npy.part"))


@pytest.mark.parametrize("stage", ["solve", "reconstruct", "norms"])
def test_phase_requires_its_executed_predecessor(tmp_path, stage):
    with pytest.raises(ValueError, match="incomplete"):
        getattr(acquisition(tmp_path), stage)()


def test_precision_sources_configuration_and_archives_are_verified(tmp_path, monkeypatch):
    run = acquisition(tmp_path)
    run.condense()
    with pytest.raises(ValueError, match="configuration"):
        owner.UnfittedAcquisition(
            tmp_path,
            refinement=2,
            names=["ell0-s1"],
            degree=3,
            assembly_order=5,
            norm_orders=[5, 7],
            refinement_precision="double",
        )
    with monkeypatch.context() as context:
        context.setattr(owner, "source_hashes", lambda: {})
        with pytest.raises(ValueError, match="source or lockfile"):
            run.solve()
    cell = run.cell_directory / "cell-0.npz"
    original = cell.read_bytes()
    cell.write_bytes(original + b"corruption")
    with pytest.raises(ValueError, match="archive digest"):
        run.solve()
    cell.write_bytes(original)
    arrays = run._cell(0)
    arrays.update(
        precision_fields("retained_basis", -owner._restore(arrays, "retained_basis", run.dtype))
    )
    owner._npz(cell, **arrays)
    run.record["cells"]["0"]["sha256"] = owner.fingerprint(cell)
    with pytest.raises(ValueError, match="retained basis"):
        run.solve()
    with pytest.raises(ValueError, match="double or extended"):
        acquisition(tmp_path / "bad", precision="automatic")
    actual = np.finfo
    with monkeypatch.context() as context:
        context.setattr(owner.np, "finfo", lambda dtype: actual(float))
        with pytest.raises(SolverUnavailableError, match="wider"):
            acquisition(tmp_path / "unsupported", precision="extended")


def test_narrower_host_coefficient_precision_cannot_silently_replay(tmp_path):
    """The complete recorded significand width participates in constructor identity."""
    run = acquisition(tmp_path)
    record = json.loads(run.path.read_text())
    record["configuration"]["coefficient_precision_bits"] += 1
    run.path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="configuration"):
        acquisition(tmp_path)


@pytest.mark.parametrize("names", [[], ["ell0-s1", "ell0-s1"], ["ell3-s3"], ["ell5-s1"]])
def test_only_declared_printed_trace_configurations_are_accepted(tmp_path, names):
    with pytest.raises(ValueError, match="printed smooth"):
        owner.UnfittedAcquisition(tmp_path, refinement=2, names=names)


def test_equal_boundary_dimension_nulltrace_remains_excluded(tmp_path):
    run = owner.UnfittedAcquisition(tmp_path, refinement=1, names=["ell3-s2"])
    run.condense()
    with pytest.raises(LinearSolveError, match="rank"):
        run.solve()


def test_individual_cli_phases_publish_a_complete_reproducible_record(tmp_path):
    command = [
        sys.executable,
        "-m",
        "examples.unfitted_phases",
        "--output",
        str(tmp_path),
        "--refinement",
        "2",
        "--degree",
        "3",
        "--names",
        "ell0-s1",
        "ell1-s2",
        "--assembly-order",
        "5",
        "--norm-orders",
        "5",
        "7",
        "--refinement-precision",
        "double",
    ]
    for stage in ("condense", "solve", "reconstruct", "norms", "all"):
        result = subprocess.run(
            command + ["--stage", stage], cwd=owner.ROOT, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "smooth-p3-r2-q5-phases.json").read_text())
    assert record["complete"] and len(record["cases"]) == 2
    assert len(record["cells"]) == 16
    assert record["execution"]["host_exclusive"] is False
    assert record["configuration"]["assembly_order"] == 5


def test_shared_original_correction_replays_actual_history_without_new_local_solves(
    corrected_acquisition, monkeypatch
):
    """Executed defects correct the same physical saddle and replay across BLAS counts."""
    run = corrected_acquisition
    expected = {name: run._field(name) for name in run.cases}
    for name, (ell, segments) in run.cases.items():
        native = solve_darcy(
            run.mesh,
            skeleton=run._skeleton(ell, segments),
            degree=3,
            local_refinement=2,
            source=smooth_source,
            quadrature_order=5,
        )
        for cell, field in enumerate(native.pressure):
            assert_allclose(
                owner._restore(expected[name], f"pressure_{cell}", run.dtype),
                field,
                rtol=2e-11,
                atol=2e-11,
            )
        assert run.record["original_equations"][name]["accepted"]
        assert run.record["refinement"]["cases"][name]["original_residual_norms"][0] > 1e-10
    records = run.record["refinement"]["records"]
    assert {key.split("-")[0] for key in records} == {
        "initial",
        "load",
        "forcing",
        "source",
        "correction",
    }
    replay = acquisition(run.directory, threads=2)
    replay.record["fields"] = {}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("replay must use the executed initial and physical increments")

    monkeypatch.setattr(LocalProblem, "reconstruct", forbidden)
    replay.reconstruct()
    for name in run.cases:
        actual = replay._field(name)
        for cell in range(16):
            assert_array_equal(
                owner._restore(actual, f"pressure_{cell}", replay.dtype),
                owner._restore(expected[name], f"pressure_{cell}", run.dtype),
            )
    assert not list(run.cell_directory.glob("*.npy.part"))


@pytest.mark.parametrize(
    "key,reason",
    [
        ("initial_pressure_0", "initial coefficients"),
        ("executed_delta_0_0", "correction coefficients"),
        ("pressure_0", "do not replay"),
        ("coarse", "field coordinates"),
    ],
)
def test_actual_ordered_field_history_is_checked_after_outer_digest_update(
    corrected_acquisition, key, reason
):
    """A changed initial, increment or final field cannot inherit an accepted record."""
    run = corrected_acquisition
    name = next(iter(run.cases))
    arrays = run._field(name)
    values = arrays[key].copy()
    values.flat[0] += 0.01
    row = run.record["fields"][name]
    row["sha256"] = rewrite_archive(run.directory / row["archive"], **{key: values})
    with pytest.raises(ValueError, match=reason):
        run._field(name)


@pytest.mark.parametrize("key", ["acquisition_id", "case_order", "values"])
def test_portable_correction_store_preserves_identity_order_and_executed_digits(
    corrected_acquisition, key
):
    """Outer digest updates do not change an executed correction's data contract."""
    run = corrected_acquisition
    size = run._cell(0)["kernel"].shape[0]
    fields = {name: np.empty((16, size), dtype=run.dtype) for name in run.cases}
    store = owner._RefinementStore(run, fields)
    store.records = run.record["refinement"]["records"]
    row = store.records["initial-0-cell-0"]
    path = store.directory / row["archive"]
    with np.load(path, allow_pickle=False) as archive:
        values = archive[key].copy()
    if key == "acquisition_id":
        values = np.asarray("another acquisition")
    elif key == "case_order":
        values = values[::-1]
    else:
        values.flat[0] += 0.01
    row["sha256"] = rewrite_archive(path, **{key: values})
    with pytest.raises(ValueError, match="acquisition|ordering|numerical digest"):
        store.read_record("initial", 0, 0)


def test_refined_coordinate_replay_checks_each_executed_increment(corrected_acquisition):
    """Final trace/coarse coordinates stay tied to the original archived coefficient basis."""
    run = corrected_acquisition
    name = next(iter(run.cases))
    row = run.record["refinement"]["cases"][name]
    path = run.directory / row["archive"]
    with np.load(path, allow_pickle=False) as archive:
        changed = archive["delta_coarse_0"].copy()
    changed.flat[0] += 0.01
    row["sha256"] = rewrite_archive(path, delta_coarse_0=changed)
    with pytest.raises(ValueError, match="coordinate increments"):
        run._refined_solution(name, run._solution(name))


@pytest.mark.parametrize("key", ["kernel", "constraints", "nodal_points"])
def test_actual_physical_local_contract_is_verified_after_outer_digest_update(tmp_path, key):
    """Current original moments/cardinal coordinates anchor the executed local basis."""
    run = acquisition(tmp_path)
    run.condense()
    arrays = run._cell(0)
    changed = arrays[key].copy()
    changed.flat[0] += 0.01
    row = run.record["cells"]["0"]
    row["sha256"] = rewrite_archive(run.cell_directory / "cell-0.npz", **{key: changed})
    if key == "kernel":
        row["kernel_sha256"] = owner.array_digest(changed)
    with pytest.raises(ValueError, match="operator, nodal basis"):
        run._restored_local(0)


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_original_refinement_limit_is_explicit_and_nonnegative(tmp_path, steps):
    """The original-equation stopping threshold is fixed independently of iteration count."""
    with pytest.raises(ValueError, match="nonnegative integer"):
        owner.UnfittedAcquisition(
            tmp_path, refinement=2, names=["ell0-s1"], original_refinement_steps=steps
        )
