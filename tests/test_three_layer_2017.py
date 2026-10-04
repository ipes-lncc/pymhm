"""Check physical input identity and original equations without a full 341-cell campaign."""

import json
import shutil
from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from examples import three_layer_2017_acquire as acquisition
from examples.three_layer_2017 import DATA, assembly_order, load_case
from pymhm import TriangleMesh
from pymhm._legacy.models.waves.elastodynamics import ElastodynamicStepper, _make_local
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace


def test_selected_geometry_wave_speeds_units_and_unsmoothed_radial_force():
    """The archived geometry is unchanged and both physical wave speeds survive conversion."""
    case = load_case()
    raw = json.loads((DATA / "macro-mesh.json").read_text())
    np.testing.assert_array_equal(case.mesh.points, raw["points"])
    np.testing.assert_array_equal(case.mesh.cells, raw["cells"])
    density, stiffness = case.materials()
    physical = stiffness.values * case.scales[1] * case.scales[0] ** 2 / case.scales[3] ** 2
    table = json.loads((DATA / "case.json").read_text())["materials"]
    np.testing.assert_allclose(
        np.sqrt(physical[:, 0, 0] / case.density), [r["vp_m_s"] for r in table], rtol=3e-15
    )
    np.testing.assert_allclose(
        np.sqrt(physical[:, 2, 2] / (2 * case.density)), [r["vs_m_s"] for r in table], rtol=3e-15
    )
    np.testing.assert_array_equal(density.heights, stiffness.heights)
    points = np.array([[0.51, 0.225], [0.5, 0.225], [0.521, 0.225]])
    np.testing.assert_allclose(
        case.source().at_time(0.2)(points), [[0.0005, 0], [0, 0], [0, 0]], rtol=0, atol=1e-18
    )
    assert len(case.mesh.cells) == 341
    assert case.skeleton().size == 25728
    assert not case.density.flags.writeable
    assert case.metadata()["historical_literal_reproduction"] is False
    assert "Negative mean" in case.metadata()["trace_convention"]


def test_selected_ormsby_agrees_with_independent_spectral_integral():
    """Integrate the trapezoidal spectrum directly, independently of the sinc formula."""
    case = load_case()
    abscissae, weights = leggauss(48)
    times = np.array([0.053, 0.107, 0.2, 0.271, 0.399])
    numerator, denominator = np.zeros_like(times), 0.0
    for left, right, rising in ((5, 10, 1), (10, 15, 0), (15, 20, -1)):
        frequencies = (left + right) / 2 + (right - left) / 2 * abscissae
        spectrum = np.ones_like(frequencies)
        if rising == 1:
            spectrum = (frequencies - left) / (right - left)
        elif rising == -1:
            spectrum = (right - frequencies) / (right - left)
        measure = weights * (right - left) / 2
        denominator += measure @ spectrum
        numerator += np.cos(2 * np.pi * (times[:, None] - 0.2) * frequencies) @ (measure * spectrum)
    np.testing.assert_allclose(case.wavelet(times), numerator / denominator, rtol=0, atol=8e-15)
    np.testing.assert_array_equal(case.wavelet([-0.1, 0, 0.2, 0.4, 0.5]), [0, 0, 1, 0, 0])
    for invalid in (np.nan, np.inf, 1j):
        with pytest.raises(ValueError, match="time"):
            case.wavelet(invalid)


@pytest.mark.parametrize("defect", ["hash", "degree", "lambda", "corners", "force", "scale"])
def test_input_validation_rejects_altered_data_or_inconsistent_physics(tmp_path, defect):
    """Changes are declared through new provenance, never silently substituted into the case."""
    directory = tmp_path / "inputs"
    shutil.copytree(DATA, directory)
    path = directory / "case.json"
    contract = json.loads(path.read_text())
    if defect == "hash":
        (directory / "horizons.csv").write_text("different data\n")
    elif defect == "degree":
        contract["discretization"]["trace_degree"] = 3
    elif defect == "lambda":
        contract["materials"][0]["lambda_Pa"] *= 1.01
    elif defect == "corners":
        contract["source"]["ormsby_corner_frequencies_Hz"] = [5, 10, 10, 20]
    elif defect == "force":
        contract["source"]["peak_amplitude_N_m3"] = True
    else:
        contract["nondimensionalization"]["reference_length_m"] = 0
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        load_case(directory)


def test_portable_input_archive_and_resume_rejects_changed_payload(tmp_path):
    """Only numeric physical arrays are archived; matching source/input identities precede reuse."""
    case = load_case()
    record = acquisition.prepare(case, tmp_path)
    with np.load(tmp_path / record["archive"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["macro_points"], case.mesh.points)
        np.testing.assert_array_equal(archive["density_kg_m3"], case.density)
        np.testing.assert_array_equal(archive["wavelet"], case.wavelet(archive["physical_time_s"]))
    assert acquisition.prepare(case, tmp_path) == record
    assert record["scientific_acceptance"] is False
    (tmp_path / record["archive"]).write_bytes(b"changed numeric payload")
    with pytest.raises(ValueError, match="archive digest"):
        acquisition.prepare(case, tmp_path)


def test_original_physical_checks_preserve_forced_work_and_rigid_momentum():
    """A nonzero physical force performs work and drives the unrestrained rigid mode."""
    mesh = TriangleMesh.unit_square()
    with ElastodynamicStepper(
        mesh,
        time_step=0.001,
        degree=2,
        local_refinement=2,
        traction={int(face): np.zeros(2) for face in mesh.boundary_faces},
    ) as stepper:
        initial = stepper.initialize()
        solution = stepper.advance([1.0, 0.3])
        checks = acquisition.original_step_checks(stepper, initial, solution, [1.0, 0.3])
        assert checks["source_work_increment_nondimensional"] > 0
        np.testing.assert_allclose(
            checks["linear_angular_momentum_nondimensional"][:2],
            [0.001, 0.0003],
            rtol=0,
            atol=3e-18,
        )
        assert (
            max(
                checks[name]
                for name in (
                    "original_local_momentum_relative_residual",
                    "physical_work_energy_relative_residual",
                    "linear_angular_momentum_relative_residual",
                )
            )
            < 1e-12
        )
        broken = tuple(value.copy() for value in solution.velocity)
        broken[0][0] += 1e-4
        with pytest.raises(RuntimeError, match="physical"):
            acquisition.original_step_checks(
                stepper, initial, replace(solution, velocity=broken), [1.0, 0.3]
            )
    with ElastodynamicStepper(mesh, time_step=0.001, degree=2) as displacement_boundary:
        initial = displacement_boundary.initialize()
        with pytest.raises(ValueError, match="zero traction"):
            acquisition.original_step_checks(displacement_boundary, initial, initial, 0.0)


def test_excluded_assembly_and_acquisition_requests_fail_before_operator_construction(tmp_path):
    """The immediately insufficient P3 quadrature and invalid index cannot trigger a solve."""
    assert assembly_order(5) == 5
    with pytest.raises(ValueError, match="quadrature"):
        assembly_order(4)
    with pytest.raises(ValueError, match="macro"):
        acquisition.acquire(load_case(), tmp_path, (341,))
    with pytest.raises(ValueError, match="steps"):
        acquisition.trajectory(load_case(), tmp_path, steps=301)


def test_p3_endpoint_functional_and_actual_vector_pairing_nullity():
    """Independent polynomial moments identify the alternating closed-contour kernel."""
    coordinate, weights = leggauss(6)
    endpoint_representer = 1 + 5 * (3 * coordinate**2 - 1) / 2
    for power in range(4):
        np.testing.assert_allclose(
            weights @ (endpoint_representer * coordinate**power),
            (-1) ** power + 1,
            rtol=0,
            atol=1.1e-14,
        )
    case = load_case()
    density, stiffness = case.materials()
    skeleton = case.skeleton()
    local = _make_local(7, case.mesh, skeleton, 3, 8, 5, density, stiffness, None, None)
    analytic_kernel = np.zeros((144, 2))
    for side, face in enumerate(case.mesh.cell_faces[7]):
        segment_length = case.mesh.lengths[face] / 8
        for segment in range(8):
            for component in range(2):
                index = side * 48 + segment * 6 + component
                analytic_kernel[index, component] = (-1) ** segment / segment_length
                analytic_kernel[index + 4, component] = 5 * (-1) ** segment / segment_length
    relative = np.linalg.norm(local.coupling @ analytic_kernel) / (
        np.linalg.norm(local.coupling) * np.linalg.norm(analytic_kernel)
    )
    assert relative < 8 * np.finfo(float).eps
    report = acquisition.trace_pairing_diagnostic(local, case.mesh, skeleton, 7)
    assert report["measured_mass_trace_L2_scaled_rank"] == 142
    assert report["local_vector_kernel_dimension"] == 2
    assert report["smallest_positive_to_largest_singular_value"] > 1e-3
    assert max(report["two_null_to_largest_singular_values"]) < 1e-14
    assert min(report["kernel_full_face_restriction_minimum_singular_values"]) > 0.1
    with pytest.raises(ValueError, match="pairing"):
        acquisition.trace_pairing_diagnostic(replace(local, degree=4), case.mesh, skeleton, 7)
    with pytest.raises(ValueError, match="pairing"):
        acquisition.trace_pairing_diagnostic(
            local,
            case.mesh,
            SkeletonSpace(case.mesh, tuple(FaceSpace.uniform(2, 4) for _ in case.mesh.faces), 2),
            7,
        )


def test_prescribed_exterior_removes_pairing_modes_without_removing_rigid_displacement():
    """A true P3/r8-P2/s8 global operator has full free rank under the stated traction BC."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2, 8) for _ in mesh.faces), components=2)
    with ElastodynamicStepper(
        mesh,
        time_step=0.001,
        degree=3,
        local_refinement=8,
        skeleton=skeleton,
        quadrature_order=5,
        traction={int(face): np.zeros(2) for face in mesh.boundary_faces},
    ) as stepper:
        free_matrix = stepper.matrix[stepper.free][:, stepper.free].toarray()
        positive = np.linalg.eigvalsh((free_matrix + free_matrix.T) / 2)
        assert len(positive) == 48
        assert positive[0] > 1e-4 * positive[-1]
        initial = stepper.initialize()
        solution = stepper.advance([1, 0])
        checks = acquisition.original_step_checks(stepper, initial, solution, [1, 0])
        np.testing.assert_allclose(
            checks["linear_angular_momentum_nondimensional"][:2], [0.001, 0], atol=1e-17
        )
    case = load_case()
    argument = acquisition.boundary_propagation_diagnostic(
        case.mesh, tuple(int(face) for face in case.mesh.boundary_faces)
    )
    assert argument["macros_reachable_from_prescribed_exterior"] == 341
    assert argument["global_numerical_rank_verified"] is False
    points = np.vstack([mesh.points, mesh.points + [2, 0]])
    separated = TriangleMesh(points, np.vstack([mesh.cells, mesh.cells + len(mesh.points)]))
    first_component_faces = tuple(
        int(face) for face in separated.boundary_faces if separated.face_cells[face, 0] < 2
    )
    with pytest.raises(ValueError, match="every macro component"):
        acquisition.boundary_propagation_diagnostic(separated, first_component_faces)
    with pytest.raises(ValueError, match="exterior"):
        acquisition.boundary_propagation_diagnostic(mesh, ())


def test_local_acquisition_archives_executed_basis_and_expected_nullity(tmp_path):
    """Numeric operator records retain nodes/maps and declare the actual local rank."""
    case = load_case()
    rows = acquisition.acquire(case, tmp_path, (7,))
    assert acquisition.acquire(case, tmp_path, (7,)) == rows
    assert rows[0]["trace_pairing"]["local_vector_kernel_dimension"] == 2
    with np.load(tmp_path / rows[0]["archive"], allow_pickle=False) as archive:
        assert archive["local_nodes"].shape == (325, 2)
        assert archive["cell_dofs"].shape == (64, 10)
        assert archive["coupling"].shape == (650, 144)
        np.testing.assert_array_equal(archive["trace_dofs"], case.skeleton().cell_dofs(7))
        assert archive["basis_degree"] == 3
    assert rows[0]["scientific_acceptance"] is False
    basis = rows[0]["executed_field_basis"]
    from examples.three_layer_field_archive import validate_field_basis

    record = json.loads((tmp_path / basis["metadata"]).read_text())
    with np.load(tmp_path / basis["archive"], allow_pickle=False) as payload:
        validate_field_basis(record, payload)


def test_trajectory_archives_actual_prepared_source_and_forced_states(tmp_path):
    """A single-macro fixture preserves the full driver/source/archive and physical checks."""
    case = replace(
        load_case(), mesh=TriangleMesh([[0.46, 0.18], [0.58, 0.18], [0.46, 0.30]], [[0, 1, 2]])
    )
    record = acquisition.trajectory(case, tmp_path, steps=2)
    assert record["persisted_time_states"] == 3
    assert record["source_preparation_seconds"] >= 0
    assert record["scientific_acceptance"] is False
    setup = json.loads((tmp_path / "setup.json").read_text())
    assert "original unweighted nodal" in setup["source_contract"]
    assert setup["executed_field_basis"] == record["executed_field_basis"]
    with np.load(tmp_path / "operator-0.npz", allow_pickle=False) as archive:
        contract = acquisition.operator_contract_digest(
            {key: archive[key] for key in archive.files}
        )
        assert setup["executed_operator_contract_sha256"]["operator-0.npz"] == contract
        vector = archive["prepared_spatial_source_load"].reshape(-1, 2)
        np.testing.assert_allclose(vector.sum(axis=0), 0, atol=2e-17)
        source = case.source().spatial_field()
        np.testing.assert_allclose(
            (archive["local_nodes"] - source.center).T @ vector,
            np.eye(2) * source.amplitude * np.pi * source.radius**3 / 3,
            rtol=2e-9,
            atol=2e-17,
        )
    for index in (1, 2):
        checks = json.loads((tmp_path / f"state-{index:03}.json").read_text())
        assert checks["original_local_momentum_relative_residual"] < 1e-10
        assert checks["physical_work_energy_relative_residual"] < 1e-10
        assert "Negative mean" in checks["trace_convention"]
        assert (
            checks["executed_operator_contract_sha256"]
            == setup["executed_operator_contract_sha256"]
        )
        assert checks["executed_field_basis"] == setup["executed_field_basis"]
        measurement = checks["measurement"]
        assert measurement["step"] == index
        assert all(value >= 0 for key, value in measurement.items() if key != "step")
    with np.load(tmp_path / "state-002.npz", allow_pickle=False) as archive:
        assert np.linalg.norm(archive["velocity"]) > 0


def test_executed_operator_contract_binds_maps_dtype_and_every_mantissa_bit():
    """A coefficient vector cannot retain its contract after an exact basis/map change."""
    arrays = {
        "nodes": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "maps": np.array([0, 1], dtype=np.int64),
        "mass": np.array([[2.0, 0.0], [0.0, 3.0]]),
    }
    expected = acquisition.operator_contract_digest(arrays)
    assert expected == acquisition.operator_contract_digest(dict(reversed(tuple(arrays.items()))))
    for name, values in (
        ("maps", arrays["maps"][::-1]),
        ("maps", arrays["maps"].astype(np.int32)),
        ("mass", arrays["mass"].reshape(4)),
        ("mass", np.nextafter(arrays["mass"], np.inf)),
    ):
        assert expected != acquisition.operator_contract_digest({**arrays, name: values})
    for invalid in (np.array([np.inf]), np.array([np.nan]), np.array([object()])):
        with pytest.raises(ValueError, match="finite non-object"):
            acquisition.operator_contract_digest({"invalid": invalid})
