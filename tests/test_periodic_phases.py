"""Executed constant kernels and oriented coefficients replay cellwise Q1 Darcy."""

import json
import shutil
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_info

import examples.periodic_phases as owner
from examples.periodic_phases import PeriodicAcquisition, array_digest, load_fields
from examples.verify_periodic import material, source
from pymhm import FaceSpace, SkeletonSpace
from pymhm._legacy.models.darcy.cartesian import solve_darcy_quadrilateral
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.fem.scalar.quadrilateral import quadrilateral_trace_coupling
from pymhm.linalg.linear import LinearSolveError, SolverUnavailableError, _accurate_residual
from pymhm.meshes.cartesian import CartesianMacroMesh


def _acquisition(directory, threads=1, precision="double"):
    return PeriodicAcquisition(
        directory,
        macro=2,
        refinement=4,
        segments=[1, 2],
        native_threads=threads,
        refinement_precision=precision,
    )


@pytest.fixture(scope="module")
def completed_snapshots(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Acquire each supported precision once for read-only replay test setup.

    Every consumer copies the entire completed tree before loading a run, so
    replays and corruptions exercise independent files and coefficient records.
    Partial-stage, native-lifetime and independent producer checks create their
    own acquisitions instead of using these completed snapshots.
    """
    snapshots = {}
    precisions = ["double"]
    if np.finfo(np.longdouble).eps < np.finfo(float).eps:
        precisions.append("extended")
    for precision in precisions:
        acquisition = _acquisition(
            tmp_path_factory.mktemp(f"periodic-{precision}"), precision=precision
        )
        acquisition.condense()
        acquisition.solve()
        acquisition.reconstruct()
        snapshots[precision] = acquisition.directory
    return snapshots


def _copied_acquisition(
    tmp_path: Path, snapshots: dict[str, Path], precision: str = "double"
) -> PeriodicAcquisition:
    """Return a new run and manifest over byte-identical private executed archives."""
    shutil.copytree(snapshots[precision], tmp_path, dirs_exist_ok=True)
    return _acquisition(tmp_path, precision=precision)


@pytest.mark.parametrize("precision", ["double", "extended"])
def test_phases_match_original_operator_rhs_fields_and_macro_balance(tmp_path, precision):
    if precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("requires wider accumulation")
    acquisition = _acquisition(tmp_path, precision=precision)
    acquisition.condense()
    acquisition.solve()
    acquisition.reconstruct()
    assert not list(tmp_path.rglob("*.part"))
    for segments in [1, 2]:
        mesh = CartesianMacroMesh(2)
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, segments) for _ in mesh.faces))
        full = solve_darcy_quadrilateral(
            mesh, permeability=material, source=source, skeleton=skeleton, local_refinement=4
        )
        path = tmp_path / f"mhm-2-r4-s{segments}.npz"
        fields = load_fields(path, macro=2, refinement=4, segments=segments)
        with np.load(path) as data:
            dtype = np.dtype(np.longdouble if precision == "extended" else float)
            assert all(data[name].dtype == dtype for name in ("fields", "trace", "coarse"))
            if precision == "extended":
                assert np.any(data["fields"] != data["fields"].astype(float).astype(dtype))
            assert_allclose(data["fields"], full.pressure, rtol=3e-13, atol=3e-16)
            assert_allclose(data["trace"], full.hybrid.trace, rtol=3e-12, atol=3e-14)
            assert_array_equal(data["kernel"], np.ones((25, 1)))
            assert_allclose(data["coarse"], full.hybrid.coarse, rtol=3e-13, atol=3e-16)
            defects, loads = [], []
            for cell, pressure in enumerate(data["fields"]):
                problem = owner._assemble_quad(acquisition._task(cell, skeleton)).problem
                forcing = problem.load.astype(np.longdouble) - problem.coupling.astype(
                    np.longdouble
                ) @ data["trace"][skeleton.cell_dofs(cell)].astype(np.longdouble)
                defects.append(_accurate_residual(problem.matrix.tocsr(), forcing, pressure))
                loads.append(problem.load)
            assert np.linalg.norm(np.concatenate(defects)) <= 1e-10 * np.linalg.norm(
                np.concatenate(loads)
            )
        for cell, field in enumerate(fields):
            points = field.mesh.points[field.mesh.cells].mean(axis=1)
            assert_allclose(field.physical_flux(points), full.evaluate(cell, points)[1], atol=3e-15)
        assert max(abs(full.conservation_residuals())) < 2e-16
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert metadata["acquisition_id"] == acquisition.record["acquisition_id"]
        assert metadata["kernel_sha256"] == array_digest(np.ones((25, 1)))
    pool_records = acquisition.record["condensation_threadpools"]
    assert pool_records and all(pool["num_threads"] == 1 for pool in pool_records)
    assert all("filepath" not in pool for pool in pool_records)
    # Verify checkpoints, instead of invoking local factorization, on a resumed pass.
    acquisition.condense()
    acquisition.solve()


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("precision", ["double", "extended"])
def test_replay_uses_executed_basis_across_threads_and_equivalent_nullspace_rotation(
    tmp_path, completed_snapshots, monkeypatch, threads, precision
):
    if precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("requires wider accumulation")
    _copied_acquisition(tmp_path, completed_snapshots, precision)
    path = tmp_path / "mhm-2-r4-s2.npz"
    with np.load(path) as data:
        original = data["fields"].copy()
    assemble = owner._assemble_quad

    def rotated(task):
        assembled = assemble(task)
        p = assembled.problem
        return LocalAssembly(
            LocalProblem(p.matrix, p.coupling, p.load, p.trace_dofs, -p.kernel, -p.constraints),
            assembled.metadata,
        )

    monkeypatch.setattr(owner, "_assemble_quad", rotated)
    resumed = _acquisition(tmp_path, threads, precision)
    resumed.reconstruct()
    with np.load(path) as data:
        assert_allclose(data["fields"], original, rtol=2e-14, atol=2e-17)
        assert_array_equal(data["kernel"], np.ones((25, 1)))
    assert all(
        pool["num_threads"] == threads for pool in resumed.record["reconstruction_threadpools"]
    )
    assert all(pool["num_threads"] >= 1 for pool in threadpool_info())


def test_replay_uses_archived_retained_matrix_without_materializing_trace_lifts(
    tmp_path, completed_snapshots, monkeypatch
):
    """The second pass uses the archived E and solves only actual combined sources."""
    acquisition = _copied_acquisition(tmp_path, completed_snapshots)
    path = tmp_path / "mhm-2-r4-s1.npz"
    with np.load(path) as data:
        expected = data["fields"].copy()
    reconstruct = LocalProblem.reconstruct
    retained = []

    def checked_reconstruction(problem, *args, **kwargs):
        basis = kwargs["retained_basis"]
        assert_array_equal(basis, acquisition._cell(len(retained))["retained_basis"])
        retained.append(array_digest(basis))
        return reconstruct(problem, *args, **kwargs)

    def forbidden_condensation(*args, **kwargs):
        raise AssertionError("replay must not materialize all trace responses")

    monkeypatch.setattr(LocalProblem, "condense", forbidden_condensation)
    monkeypatch.setattr(LocalProblem, "reconstruct", checked_reconstruction)
    acquisition.reconstruct()
    with np.load(path) as data:
        assert_array_equal(data["fields"], expected)
    assert len(retained) == len(acquisition.mesh.cells)
    arrays = acquisition._cell(0)
    assert (
        array_digest(arrays["retained_basis"])
        == acquisition.record["cells"]["0"]["retained_basis_sha256"]
    )


def test_original_full_saddle_gate_rejects_fields_before_atomic_publication(tmp_path, monkeypatch):
    """A constant field error violates weak Dirichlet rows despite the local kernel."""
    acquisition = PeriodicAcquisition(
        tmp_path, macro=2, refinement=4, segments=[1, 2], original_refinement_steps=0
    )
    acquisition.condense()
    acquisition.solve()
    reconstruct = LocalProblem.reconstruct

    def wrong_boundary_pressure(*args, **kwargs):
        return reconstruct(*args, **kwargs) + 0.125

    monkeypatch.setattr(LocalProblem, "reconstruct", wrong_boundary_pressure)
    with pytest.raises(LinearSolveError, match="original periodic saddle residual"):
        acquisition.reconstruct()
    assert not (tmp_path / "mhm-2-r4-s1.npz").exists()
    assert not list(tmp_path.rglob("*.part"))
    for values in acquisition.record["original_equations"].values():
        assert not values["accepted"]
        assert values["original_saddle_relative_residual"] > 1e-10
        assert values["weak_pressure_continuity_residual_norm"] > 0.0
        assert len(values["cells"]) == len(acquisition.mesh.cells)


def test_retained_matrix_digest_and_dtype_are_verified(tmp_path):
    """Coarse replay requires the archived executed E matrix with its real dtype."""
    acquisition = _acquisition(tmp_path)
    acquisition.condense()
    path = acquisition.cell_directory / "cell-0.npz"
    arrays = acquisition._cell(0)
    arrays["retained_basis"] *= -1
    owner._npz(path, **arrays)
    acquisition.record["cells"]["0"]["sha256"] = owner.fingerprint(path)
    with pytest.raises(ValueError, match="retained basis"):
        acquisition.solve()


def test_precision_is_explicit_and_requires_supported_storage(tmp_path, monkeypatch):
    """Reject unknown modes, unsupported hosts and differently configured replay."""
    with pytest.raises(ValueError, match="double or extended"):
        _acquisition(tmp_path, precision="automatic")
    actual = np.finfo
    monkeypatch.setattr(owner.np, "finfo", lambda dtype: actual(float))
    with pytest.raises(SolverUnavailableError, match="wider"):
        _acquisition(tmp_path, precision="extended")
    monkeypatch.undo()
    acquisition = _acquisition(tmp_path)
    if np.finfo(np.longdouble).eps < np.finfo(float).eps:
        with pytest.raises(ValueError, match="configuration"):
            _acquisition(tmp_path, precision="extended")
    assert acquisition.config["refinement_precision"] == "double"


def test_replay_rejects_sources_lock_or_configuration_mismatch(tmp_path, monkeypatch):
    acquisition = _acquisition(tmp_path)
    acquisition.condense()
    acquisition.solve()
    for stage in [acquisition.condense, acquisition.solve, acquisition.reconstruct]:
        with monkeypatch.context() as context:
            context.setattr(owner, "source_hashes", lambda: {})
            with pytest.raises(ValueError, match="source or lockfile"):
                stage()
    acquisition.record["lockfile_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source or lockfile"):
        acquisition.reconstruct()
    with pytest.raises(ValueError, match="configuration"):
        PeriodicAcquisition(tmp_path, macro=2, refinement=4, segments=[1])


@pytest.mark.parametrize("stage", ["solve", "reconstruct"])
def test_phase_requires_its_executed_predecessor(tmp_path, stage):
    acquisition = _acquisition(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        getattr(acquisition, stage)()


def test_cell_and_coefficient_hashes_are_verified(tmp_path):
    acquisition = _acquisition(tmp_path)
    acquisition.condense()
    cell = acquisition.cell_directory / "cell-0.npz"
    original = cell.read_bytes()
    cell.write_bytes(original + b"corruption")
    with pytest.raises(ValueError, match="cell archive digest"):
        acquisition.solve()
    cell.write_bytes(original)
    acquisition.solve()
    coefficient = tmp_path / acquisition.record["solutions"]["1"]["archive"]
    coefficient.write_bytes(coefficient.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="coefficient archive digest"):
        acquisition.reconstruct()


@pytest.mark.parametrize(
    "change", ["kernel", "signs", "fields", "residual", "configuration", "field_dtype", "precision"]
)
def test_final_field_contract_detects_basis_orientation_and_array_mismatches(
    tmp_path, completed_snapshots, change
):
    _copied_acquisition(tmp_path, completed_snapshots)
    path = tmp_path / "mhm-2-r4-s1.npz"
    manifest = path.with_suffix(".json")
    record = json.loads(manifest.read_text())
    if change == "configuration":
        record["configuration"]["macro"] = 3
    elif change == "precision":
        record["configuration"]["coefficient_precision_bits"] = 1
    else:
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        if change == "fields":
            arrays["fields"][0, 0] = np.nan
        elif change == "residual":
            arrays["residual"] = -1.0
        elif change == "field_dtype":
            arrays["fields"] = arrays["fields"].astype(np.float32)
        else:
            arrays[change] *= -1
        owner._npz(path, **arrays)
        record["archive_sha256"] = owner.fingerprint(path)
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest|basis|arrays|precision"):
        load_fields(path, macro=2, refinement=4, segments=1)


def test_cli_phases_and_fresh_norm_record_replace_historical_summaries(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "results.json"
    output.write_text(
        json.dumps({"reference": [{"n": 4, "source_hashes": {"obsolete": "value"}}], "mhm": []})
    )
    common = [
        sys.executable,
        "-m",
        "examples.verify_periodic",
        "--artifacts",
        str(tmp_path),
        "--output",
        str(output),
        "--reference-levels",
        "4",
        "--reference-solver",
        "scipy",
        "--macro",
        "2",
        "--local-refinement",
        "4",
        "--segments",
        "1",
        "2",
    ]
    for stage in ["condense", "solve", "reconstruct", "all", "norms"]:
        result = subprocess.run(
            common + ["--stage", stage], cwd=root, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text())
    assert len(record["reference"]) == 1
    assert "obsolete" not in record["reference"][0]["source_hashes"]
    assert len(record["mhm"]) == 2
    assert all(row["acquisition_id"] for row in record["mhm"])
    output.unlink()
    result = subprocess.run(common + ["--stage", "norms"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(output.read_text())["reference"]) == 1


def test_equal_q1_resolution_and_p0_segments_have_an_exact_global_nulltrace(tmp_path):
    """The alternating trace is invisible to all Q1 nodal boundary test rows."""
    mesh = CartesianMacroMesh(2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 4) for _ in mesh.faces))
    trace = np.tile([1.0, -1.0, 1.0, -1.0], len(mesh.faces))
    assert np.linalg.norm(trace) > 0
    for cell in range(len(mesh.cells)):
        fine = mesh.submesh(cell, 4)
        coupling = quadrilateral_trace_coupling(mesh, cell, fine, skeleton, 1)
        # Independently integrate the two linear endpoint cardinals on every
        # boundary interval: each integral is physical interval length / 2.
        direct = np.zeros_like(coupling)
        for side, face in enumerate(mesh.cell_faces[cell]):
            endpoints = mesh.points[mesh.faces[face]]
            nodes = endpoints[0] + np.arange(5)[:, None] / 4 * (endpoints[1] - endpoints[0])
            ij = np.rint((nodes - fine.points[0]) / fine.spacing).astype(int)
            indices = ij[:, 1] * 5 + ij[:, 0]
            for segment in range(4):
                direct[indices[segment : segment + 2], side * 4 + segment] = (
                    mesh.signs[cell, side] * mesh.lengths[face] / 8
                )
        assert_array_equal(coupling, direct)
        assert_array_equal(direct @ trace[skeleton.cell_dofs(cell)], np.zeros(25))
    with pytest.raises(LinearSolveError, match="rank"):
        solve_darcy_quadrilateral(mesh, skeleton=skeleton, local_refinement=4, source=1.0)
    acquisition = PeriodicAcquisition(tmp_path, macro=2, refinement=4, segments=[1, 2, 4])
    acquisition.condense()
    with pytest.raises(LinearSolveError, match="rank"):
        acquisition.solve()


def test_each_cell_releases_volumetric_lifts_before_the_next_factorization(tmp_path, monkeypatch):
    acquisition = _acquisition(tmp_path)
    condense = LocalProblem.condense
    restrict = owner.restrict_response
    volumetric = []

    def tracked_condensation(problem, *args, **kwargs):
        assert all(reference() is None for reference in volumetric)
        response = condense(problem, *args, **kwargs)
        volumetric.extend([weakref.ref(response.source), weakref.ref(response.lifts)])
        return response

    def tracked_restriction(*args, **kwargs):
        response = restrict(*args, **kwargs)
        volumetric.append(weakref.ref(response.lifts))
        return response

    monkeypatch.setattr(LocalProblem, "condense", tracked_condensation)
    monkeypatch.setattr(owner, "restrict_response", tracked_restriction)
    acquisition.condense()
    assert all(reference() is None for reference in volumetric)
    acquisition.solve()
    acquisition.reconstruct()
    assert all(reference() is None for reference in volumetric)


@pytest.mark.parametrize("changed", ["coefficient", "epsilon", "source", "boundary", "provenance"])
def test_manifested_pressure_cannot_use_a_different_material_or_physical_case(
    tmp_path, completed_snapshots, changed
):
    _copied_acquisition(tmp_path, completed_snapshots)
    path = tmp_path / "mhm-2-r4-s1.npz"
    manifest = path.with_suffix(".json")
    record = json.loads(manifest.read_text())
    if changed == "provenance":
        record["source_hashes"]["examples/verify_periodic.py"] = "0" * 64
    else:
        record["configuration"][changed] = 1.0 if changed == "epsilon" else "different"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="material/case"):
        load_fields(path, macro=2, refinement=4, segments=1)


@pytest.mark.parametrize("threads", [1, 2])
def test_actual_original_equation_correction_is_archived_and_replayed(
    tmp_path, monkeypatch, threads
):
    """Same-operator correction preserves actual fields and coordinate increments."""
    acquisition = _acquisition(tmp_path)
    acquisition.condense()
    acquisition.solve()
    reconstruct = LocalProblem.reconstruct

    def perturbed_initial(problem, *args, **kwargs):
        field = reconstruct(problem, *args, **kwargs)
        return field + 0.001 if np.any(problem.load) else field

    monkeypatch.setattr(LocalProblem, "reconstruct", perturbed_initial)
    acquisition.reconstruct()
    assert acquisition.record["refinement"]["accepted"]
    assert acquisition.record["refinement"]["steps"] == 1
    saved = {}
    for segments in [1, 2]:
        path = tmp_path / f"mhm-2-r4-s{segments}.npz"
        with np.load(path, allow_pickle=False) as data:
            saved[segments] = data["fields"].copy()
        result = acquisition.record["refinement"]["cases"][str(segments)]
        assert result["original_residual_norms"][0] > result["original_residual_norms"][-1] * 1e5
        assert result["original_residual_norms"][-1] <= 1e-10 * result["original_rhs_norm"]
        fields = load_fields(path, macro=2, refinement=4, segments=segments)
        mesh = CartesianMacroMesh(2)
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, segments) for _ in mesh.faces))
        full = solve_darcy_quadrilateral(
            mesh, permeability=material, source=source, skeleton=skeleton, local_refinement=4
        )
        assert_allclose(np.array([field.pressure for field in fields]), full.pressure, atol=3e-15)
    resumed = _acquisition(tmp_path, threads)
    resumed.reconstruct()
    for segments in [1, 2]:
        with np.load(tmp_path / f"mhm-2-r4-s{segments}.npz", allow_pickle=False) as data:
            assert_array_equal(data["fields"], saved[segments])
    assert all(
        pool["num_threads"] == threads for pool in resumed.record["reconstruction_threadpools"]
    )


def test_ordered_correction_record_digest_is_checked_before_replay(tmp_path, completed_snapshots):
    """A corrupted executed initial record cannot be replaced by a fresh conditional solve."""
    acquisition = _copied_acquisition(tmp_path, completed_snapshots)
    values = acquisition.record["refinement"]["records"]["initial-0-cell-0"]
    path = acquisition.cell_directory / "original-refinement" / values["archive"]
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="correction archive digest"):
        acquisition.reconstruct()
    assert not list(tmp_path.rglob("*.part"))


def test_declared_portable_field_components_are_validated(tmp_path, completed_snapshots):
    """Consumers verify both native coefficients and their portable decomposition."""
    _copied_acquisition(tmp_path, completed_snapshots)
    path = tmp_path / "mhm-2-r4-s1.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["fields_portable_correction"][0, 0] = 0.5
    owner._npz(path, **arrays)
    manifest = path.with_suffix(".json")
    record = json.loads(manifest.read_text())
    record["archive_sha256"] = owner.fingerprint(path)
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="portable precision components"):
        load_fields(path, macro=2, refinement=4, segments=1)


@pytest.mark.parametrize("steps", [-1, True, 0.5])
def test_original_refinement_budget_is_explicit(tmp_path, steps):
    with pytest.raises(ValueError, match="nonnegative integer"):
        PeriodicAcquisition(
            tmp_path, macro=2, refinement=4, segments=[1], original_refinement_steps=steps
        )
