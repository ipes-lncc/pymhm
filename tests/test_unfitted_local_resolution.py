"""Check physical common-grid integration independently of an acquired MHM solve."""

import hashlib
import json
import runpy
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.archive_precision import restore_precision
from examples.unfitted_convergence import save_field
from examples.unfitted_local_resolution import containing_cells, difference, read_uniform, template
from examples.unfitted_phases import UnfittedAcquisition, fingerprint
from pymhm import solve_darcy
from pymhm.lagrange import nodal_space
from pymhm.mesh import TriangleMesh


def archive(path, refinement, degree, polynomial):
    """Persist an exactly representable field with the campaign's complete geometry."""
    macro = TriangleMesh.unit_square(1)
    arrays = dict(macro_points=macro.points, macro_cells=macro.cells, degree=degree)
    for cell in range(len(macro.cells)):
        mesh = macro.submesh(cell, refinement)
        _, nodes = nodal_space(mesh, degree)
        arrays.update(
            {
                f"points_{cell}": mesh.points,
                f"cells_{cell}": mesh.cells,
                f"pressure_{cell}": polynomial(nodes),
            }
        )
    np.savez(path, **arrays)


def test_nonnested_exact_quadratic_and_distinct_nonzero_norms(tmp_path):
    """Use r2/r3 with equal nonzero fields and a separately integrated zero comparison."""
    first, second, zero = (tmp_path / f"{name}.npz" for name in ("a", "b", "zero"))

    def polynomial(points):
        """Return x²+2y², whose global norms are available by monomial moments."""
        return points[:, 0] ** 2 + 2 * points[:, 1] ** 2

    archive(first, 2, 2, polynomial)
    archive(second, 3, 3, polynomial)
    archive(zero, 3, 2, lambda points: np.zeros(len(points)))
    same = difference(first, second, order=5)
    assert same["pressure_l2"] < 3e-15
    assert same["broken_gradient_l2"] < 3e-14
    assert same["common_triangles"] == 72
    for order in (4, 6):
        measured = difference(first, zero, order=order)
        assert_allclose(measured["pressure_l2"], np.sqrt(13 / 9), rtol=3e-15)
        assert_allclose(measured["broken_gradient_l2"], np.sqrt(20 / 3), rtol=3e-15)


def test_portable_campaign_archive_preserves_complete_field_coordinates(tmp_path):
    """Replay complete fields in their executed cardinal basis and physical trace maps."""
    solution = solve_darcy(TriangleMesh.unit_square(), source=1, degree=2, local_refinement=2)

    def extend(values):
        """Exercise representable digits beyond float64 whenever the platform provides them."""
        return np.asarray(values, dtype=np.longdouble) + np.longdouble(2) ** -59

    pressure = tuple(extend(p) for p in solution.pressure)
    hybrid = replace(
        solution.hybrid,
        fields=pressure,
        trace=extend(solution.hybrid.trace),
        coarse=tuple(extend(c) for c in solution.hybrid.coarse),
    )
    solution = replace(solution, pressure=pressure, hybrid=hybrid)
    path = tmp_path / "portable.npz"
    assert save_field(solution, path) == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as arrays:
        assert all(arrays[name].dtype.itemsize <= 8 for name in arrays.files)
        assert arrays["field_archive_version"] == 2
        assert arrays["coefficient_precision_bits"] == np.finfo(np.longdouble).nmant + 1
        assert not any(name.startswith("coarse") for name in arrays.files)
        for name, expected in (("trace", hybrid.trace),):
            actual = restore_precision(
                arrays[name], arrays[f"{name}_correction"], arrays[f"{name}_tail"]
            )
            np.testing.assert_array_equal(actual, expected)
        for cell, mesh in enumerate(solution.local_meshes):
            dofs, nodes = nodal_space(mesh, solution.degree)
            np.testing.assert_array_equal(arrays[f"nodal_dofs_{cell}"], dofs)
            np.testing.assert_array_equal(arrays[f"nodal_points_{cell}"], nodes)
        for face, space in enumerate(solution.skeleton.faces):
            np.testing.assert_array_equal(arrays[f"trace_breaks_{face}"], space.breaks)
            np.testing.assert_array_equal(arrays[f"trace_degrees_{face}"], space.degrees)
            np.testing.assert_array_equal(
                arrays[f"trace_dofs_{face}"], solution.skeleton.dofs(face)
            )
        np.testing.assert_array_equal(arrays["macro_normals"], solution.skeleton.mesh.normals)
        np.testing.assert_array_equal(arrays["macro_signs"], solution.skeleton.mesh.signs)
    assert not path.with_suffix(".npz.part").exists()
    _, _, degree, actual = read_uniform(path)
    assert degree == 2
    np.testing.assert_array_equal(actual, np.stack(pressure))
    with pytest.raises(ValueError, match="primal cardinal"):
        save_field(replace(solution, formulation="mixed"), tmp_path / "invalid-p0.npz")
    assert not (tmp_path / "invalid-p0.npz").exists()
    with pytest.raises(ValueError, match="declared nodal basis"):
        save_field(
            replace(solution, pressure=(pressure[0][:-1], *pressure[1:])),
            tmp_path / "invalid-cardinal.npz",
        )
    assert not (tmp_path / "invalid-cardinal.npz").exists()


def test_wider_archive_rejects_silent_precision_loss(tmp_path, monkeypatch):
    """A platform without wider arithmetic can read components but cannot claim exact replay."""
    path = tmp_path / "wider.npz"
    archive(path, 1, 1, lambda points: np.ones(len(points)))
    with np.load(path) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    for cell in range(2):
        arrays[f"pressure_{cell}_correction"] = np.full(3, 2.0**-59)
        arrays[f"pressure_{cell}_tail"] = np.zeros(3)
    np.savez(path, **arrays)
    monkeypatch.setattr(np, "longdouble", np.float64)
    with pytest.raises(ValueError, match="requires wider longdouble"):
        read_uniform(path)
    for cell in range(2):
        arrays[f"pressure_{cell}_correction"][:] = 0
    np.savez(path, **arrays)
    np.testing.assert_array_equal(read_uniform(path)[3], np.ones((2, 3)))


def test_archive_declared_significand_rejects_a_narrower_extended_consumer(tmp_path):
    """Portable data on a narrower host cannot silently discard executed digits."""
    path = tmp_path / "wider-extended.npz"
    archive(path, 1, 1, lambda points: np.ones(len(points)))
    with np.load(path) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["coefficient_precision_bits"] = np.finfo(np.longdouble).nmant + 2
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="declared precision"):
        read_uniform(path)


@pytest.mark.parametrize("schema", ["legacy", "monolithic-v2"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1j])
def test_raw_coefficients_require_finite_real_values(tmp_path, schema, invalid):
    """A valid nodal map cannot authorize nonphysical raw real-pressure coefficients."""
    path = tmp_path / "raw.npz"
    if schema == "legacy":
        archive(path, 1, 1, lambda nodes: nodes[:, 0])
    else:
        solution = solve_darcy(TriangleMesh.unit_square(), source=1, degree=1)
        save_field(solution, path)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays.pop("pressure_0_correction", None)
    arrays.pop("pressure_0_tail", None)
    np.savez(path, **arrays)
    read_uniform(path)
    arrays["pressure_0"] = arrays["pressure_0"].astype(np.result_type(invalid))
    arrays["pressure_0"][0] = invalid
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="finite real nodal"):
        read_uniform(path)


@pytest.mark.parametrize(
    "changed",
    [
        "basis_multiindices",
        "macro_faces",
        "macro_face_cells",
        "macro_normals",
        "macro_signs",
        "macro_cell_faces",
        "nodal_dofs_0",
        "nodal_points_0",
        "missing_macro_map",
        "missing_nodal_map",
        "field",
        "version",
        "missing_version",
    ],
)
def test_replay_checks_executed_cardinal_and_orientation_contract(tmp_path, changed):
    """Equal geometry and dimensions cannot authorize changed coefficient coordinates."""
    path = tmp_path / "declared-cardinal.npz"
    solution = solve_darcy(TriangleMesh.unit_square(), source=1, degree=2, local_refinement=2)
    save_field(solution, path)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    if changed == "missing_macro_map":
        arrays.pop("macro_normals")
    elif changed == "missing_nodal_map":
        arrays.pop("nodal_points_0")
    elif changed == "field":
        arrays["pressure_0"] = arrays["pressure_0"][:-1]
    elif changed == "version":
        arrays["field_archive_version"] = np.array(3)
    elif changed == "missing_version":
        arrays.pop("field_archive_version")
    elif changed == "nodal_dofs_0":
        arrays[changed].flat[:2] = arrays[changed].flat[:2][::-1]
    else:
        arrays[changed].flat[0] += 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="archive version|archived cardinal|nodal basis"):
        read_uniform(path)


@pytest.fixture(scope="module")
def phase_archive(tmp_path_factory):
    """Acquire one complete small phase case with its actually executed bases."""
    directory = tmp_path_factory.mktemp("uniform-phase")
    run = UnfittedAcquisition(
        directory,
        refinement=1,
        names=["ell0-s1"],
        degree=2,
        assembly_order=4,
        norm_orders=[4],
        refinement_precision="double",
    )
    for stage in (run.condense, run.solve, run.reconstruct, run.norms):
        stage()
    return run


def test_phase_archive_preserves_executed_coordinates_and_zero_self_difference(phase_archive):
    """Accepted phase fields use their receipt, ordered history and cardinal basis."""
    run = phase_archive
    path = run.directory / run.record["fields"]["ell0-s1"]["archive"]
    _, _, degree, actual = read_uniform(path)
    assert degree == 2
    expected = run._field("ell0-s1")
    np.testing.assert_array_equal(actual, np.stack([expected[f"pressure_{i}"] for i in range(16)]))
    assert difference(path, path, order=4)["broken_gradient_l2"] == 0


@pytest.mark.parametrize(
    "changed",
    [
        "nodal_dofs_0",
        "nodal_points_0",
        "multiindices_0",
        "constraints_0",
        "retained_basis_0",
        "missing_nodal_map",
        "degree",
        "schema",
        "incomplete",
        "missing_receipt",
        "duplicate_receipt",
        "missing_acquisition_id",
    ],
)
def test_phase_replay_rejects_contradictory_bases_with_updated_digest(
    tmp_path, phase_archive, changed
):
    """Digest-consistent dimensions cannot substitute a different executed field basis."""
    directory = tmp_path / "phase"
    shutil.copytree(phase_archive.directory, directory)
    receipt = directory / phase_archive.path.name
    record = json.loads(receipt.read_text())
    row = record["fields"]["ell0-s1"]
    path = directory / row["archive"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if changed == "schema":
        record["schema"] = "pymhm-unfitted-phases-v3"
    elif changed == "incomplete":
        record["complete"] = False
    elif changed == "missing_nodal_map":
        arrays.pop("nodal_points_0")
    elif changed == "missing_acquisition_id":
        arrays.pop("acquisition_id")
    elif changed in {"missing_receipt", "duplicate_receipt"}:
        pass
    elif changed == "nodal_dofs_0":
        arrays[changed][0, :2] = arrays[changed][0, :2][::-1]
    else:
        arrays[changed].flat[0] += 1
    np.savez(path, **arrays)
    row["sha256"] = fingerprint(path)
    record["cases"][0]["archive_sha256"] = row["sha256"]
    receipt.write_text(json.dumps(record))
    if changed == "missing_receipt":
        receipt.unlink()
    elif changed == "duplicate_receipt":
        shutil.copyfile(receipt, directory / "duplicate-phases.json")
    with pytest.raises(
        ValueError, match="cardinal|local map|retained basis|receipt|schema|archived basis"
    ):
        read_uniform(path)


def test_containment_proves_vertices_and_falls_back_from_spatial_candidates(monkeypatch):
    """An unsuccessful nearest-cell candidate is never treated as an owning cell."""
    import examples.unfitted_local_resolution as module

    class WrongTree:
        """Return a deliberately incorrect candidate for every common triangle."""

        def __init__(self, points):
            self.count = len(points)

        def query(self, centers, k):
            return np.zeros(len(centers)), np.full((len(centers), 1), self.count - 1)

    monkeypatch.setattr(module, "cKDTree", WrongTree)
    mesh, common = template(2), template(4)
    owners = containing_cells(mesh, common.points[common.cells])
    assert len(np.unique(owners)) == 4
    with pytest.raises(ValueError, match="not contained"):
        containing_cells(mesh, np.array([[[0, 0], [1.1, 0], [0, 1]]]))


def test_archive_geometry_and_macro_contracts_are_not_inferred_from_counts(tmp_path):
    """Reject shifted fine coordinates and differing macrogeometry, even with equal sizes."""
    path, other = tmp_path / "a.npz", tmp_path / "b.npz"
    archive(path, 2, 2, lambda points: points[:, 0])
    with np.load(path) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["points_0"] = arrays["points_0"] + [1e-6, 0]
    np.savez(other, **arrays)
    with pytest.raises(ValueError, match="archived geometry"):
        read_uniform(other)
    with np.load(path) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["macro_points"] = arrays["macro_points"] + [1, 0]
    for cell in range(2):
        arrays[f"points_{cell}"] = arrays[f"points_{cell}"] + [1, 0]
    np.savez(other, **arrays)
    with pytest.raises(ValueError, match="identical macro"):
        difference(path, other)
    arrays["cells_0"] = arrays["cells_0"][:-1]
    np.savez(other, **arrays)
    with pytest.raises(ValueError, match="uniform local"):
        read_uniform(other)


def test_cli_records_actual_fields_and_nonzero_physical_norms(tmp_path, monkeypatch):
    """Exercise the public executable and its field/provenance output contract."""
    first, second, output = (tmp_path / name for name in ("a.npz", "b.npz", "out/data.json"))
    archive(first, 1, 1, lambda points: points[:, 0])
    archive(second, 2, 1, lambda points: np.zeros(len(points)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unfitted_local_resolution",
            str(first),
            str(second),
            "--order",
            "3",
            "--output",
            str(output),
        ],
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module("examples.unfitted_local_resolution", run_name="__main__")
    data = json.loads(output.read_text())
    assert data["fields"][0]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert_allclose(data["norms"]["pressure_l2"], 1 / np.sqrt(3), rtol=3e-15)
    assert_allclose(data["norms"]["broken_gradient_l2"], 1, rtol=3e-15)


@pytest.mark.parametrize("changed", ["field", "source"])
def test_cli_rejects_changed_input_or_evaluation_owner(tmp_path, monkeypatch, changed):
    """A result is not published if its archive or numerical-source bytes change."""
    import examples.unfitted_local_resolution as module

    first, second, output = (tmp_path / name for name in ("a.npz", "b.npz", "out.json"))
    archive(first, 1, 1, lambda points: points[:, 0])
    archive(second, 1, 1, lambda points: np.zeros(len(points)))
    monkeypatch.setattr(
        sys, "argv", ["unfitted_local_resolution", str(first), str(second), "--output", str(output)]
    )
    read = Path.read_bytes
    calls = 0

    def changed_bytes(path):
        """Simulate only a second guarded read; do not edit an actual numerical owner."""
        nonlocal calls
        data = read(path)
        target = path == first if changed == "field" else path.name == "lagrange.py"
        if target:
            calls += 1
            if calls == 2:
                return data + b"changed"
        return data

    monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    with pytest.raises(RuntimeError, match="source or acquired field changed"):
        module.main()
    assert not output.exists()
