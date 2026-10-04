"""Physical and persisted-basis invariants of the finite three-dimensional method."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from examples import core_extension_data as exact
from examples.mshho3d_field_archive import (
    field_arrays,
    original_checks,
    read_field,
    replay,
    restore,
    validate_arrays,
    write_field,
)
from examples.mshho3d_ideal_p0 import ideal_p0_audit
from examples.mshho3d_sections import capture_section, replay_section, validate_section
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space, tetra_operators
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.methods.hho_3d import solve_mshho_3d

PRECISION = "extended" if np.finfo(np.longdouble).nmant > np.finfo(float).nmant else "double"


@pytest.fixture(params=("tetra", "cube"))
def example(request):
    """Return the actual finite P2/P0 homogeneous solution and its complete tables."""
    tetra = request.param == "tetra"
    mesh = TetraMesh.unit_cube(1) if tetra else PolyhedralMesh.cubes(1)
    solution = solve_mshho_3d(
        mesh,
        source=exact.source3d,
        quadrature_order=9,
        local_refinement=2 if tetra else 1,
        local_refinement_precision=PRECISION,
    )
    return solution, field_arrays(solution, assembly_order=9, source=exact.source3d)


def test_original_rows_csr_and_executed_energy(example):
    """Persist actual sparse row action, without substituting its transpose."""
    solution, arrays = example
    assert original_checks(arrays)["original_projected_source_relative_residual"] < 1e-10
    for cell, local in enumerate(solution.local):
        original, _, _ = tetra_operators(local.mesh, 2, source=exact.source3d, order=9)
        restored = sparse.csr_matrix(
            (
                arrays[f"audit_a_data_{cell}"],
                arrays[f"audit_a_indices_{cell}"],
                arrays[f"audit_a_indptr_{cell}"],
            ),
            shape=original.shape,
        )
        np.testing.assert_array_equal(restored.toarray(), original.toarray())
        np.testing.assert_array_equal(restore(arrays, f"executed_energy_{cell}"), local.energy)


def test_all_executed_lifts_equal_selected_ideal_space(example):
    """Every local lift column, including unused moments, belongs to the ideal U00 space."""
    solution, arrays = example
    kind = "cube" if isinstance(solution.skeleton.mesh, PolyhedralMesh) else "tetra"
    certificate = ideal_p0_audit(arrays, kind)
    assert certificate["accepted"]
    assert certificate["maxima"]["lift_relative_linf"] < 1e-12
    assert not certificate["executed_lift_replaced"]
    for forbidden in ("degree", "cell_degree"):
        changed = dict(arrays)
        changed[forbidden] = np.asarray(1)
        with pytest.raises(ValueError, match="P2"):
            ideal_p0_audit(changed, kind)
    changed = dict(arrays)
    changed["q10_material_0"] = 2 * arrays["q10_material_0"]
    with pytest.raises(ValueError, match="identity"):
        ideal_p0_audit(changed, kind)


def test_replay_norms_match_actual_coefficients(example):
    """Saved cardinal derivatives and volume maps reproduce physical field norms."""
    solution, arrays = example
    for order in (10, 12):
        pressure_error = flux_error = np.longdouble(0)
        for cell in range(len(solution.local)):
            pressure, gradient, flux = replay(arrays, cell, order)
            np.testing.assert_array_equal(flux, -gradient)
            points = arrays[f"q{order}_physical_points_{cell}"]
            measure = arrays[f"volumes_{cell}"][:, None] * arrays[f"q{order}_weights"]
            target = exact.pressure3d(points.reshape(-1, 3)).reshape(pressure.shape)
            target_flux = exact.flux3d(points.reshape(-1, 3)).reshape(flux.shape)
            pressure_error += np.sum(measure * (pressure - target) ** 2)
            flux_error += np.sum(measure * np.sum((flux - target_flux) ** 2, axis=-1))
        assert float(np.sqrt(pressure_error)) == pytest.approx(
            solution.l2_error(exact.pressure3d, order), rel=2e-14
        )
        assert float(np.sqrt(flux_error)) == pytest.approx(
            solution.flux_l2_error(exact.flux3d, order), rel=2e-14
        )


@pytest.mark.parametrize("kind", ("tetra", "cube"))
def test_physical_nonhomogeneous_quadratic(kind):
    """P2 volume/P0 planar normal flux retain the absolute Dirichlet pressure level."""

    def pressure(x):
        return np.sum(x * (1 - x), axis=1)

    mesh = TetraMesh.unit_cube(1) if kind == "tetra" else PolyhedralMesh.cubes(1)
    solution = solve_mshho_3d(
        mesh,
        source=6.0,
        dirichlet=pressure,
        quadrature_order=9,
        local_refinement=2 if kind == "tetra" else 1,
        local_refinement_precision=PRECISION,
    )
    arrays = field_arrays(solution, assembly_order=9, source=6.0, dirichlet=pressure)
    assert original_checks(arrays)["original_projected_source_relative_residual"] < 1e-10
    for field, local in zip(solution.pressure, solution.local, strict=True):
        _, nodes = tetra_nodal_space(local.mesh, 2)
        np.testing.assert_allclose(field, pressure(nodes), rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "corruption", ("cardinal", "derivative", "nodes", "mapping", "moment", "energy")
)
def test_physical_contract_rejects_changed_basis(example, corruption):
    """A new checksum does not make inconsistent physical basis data admissible."""
    _, arrays = example
    arrays = {k: v.copy() for k, v in arrays.items()}
    key = {
        "cardinal": "q10_cardinal_values",
        "derivative": "q10_cardinal_derivatives",
        "nodes": "nodes_0",
        "mapping": "barycentric_gradients_0",
        "moment": "executed_moments_0",
        "energy": "executed_energy_0",
    }[corruption]
    arrays[key].flat[0] += 0.01
    with pytest.raises(ValueError):
        validate_arrays(arrays)


def test_atomic_archive_digest_and_no_fresh_reconstruction(example, tmp_path, monkeypatch):
    """Data-only replay needs the archived matrices and rejects changed coefficient bytes."""
    solution, _ = example
    owner = Path(__file__).resolve().parents[1] / "examples/mshho3d_field_archive.py"
    path = tmp_path / "field.npz"
    write_field(
        path,
        solution,
        {},
        acquisition_uuid="test-original-physical-field",
        source_sha256={
            "examples/mshho3d_field_archive.py": hashlib.sha256(owner.read_bytes()).hexdigest()
        },
        assembly_order=9,
        source=exact.source3d,
    )
    import examples.mshho3d_field_archive as archive

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay called a fresh basis or operator assembly")

    monkeypatch.setattr(archive, "tetra_basis", forbidden)
    monkeypatch.setattr(archive, "tetra_operators", forbidden)
    arrays, record = read_field(path)
    assert record["coefficient_nmant"] == np.finfo(solution.pressure[0].dtype).nmant
    replay(arrays, 0, 10)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity"):
        read_field(path)


def test_section_replays_actual_tables_without_new_basis(example, monkeypatch):
    """Private fine-cell section vertices keep the producer's actual sampling basis."""
    solution, arrays = example
    arrays.update(capture_section(solution, 2))
    validate_arrays(arrays)
    import examples.mshho3d_sections as sections

    def forbidden(*args, **kwargs):
        raise AssertionError("Section replay called a new basis")

    monkeypatch.setattr(sections, "tetra_basis", forbidden)
    fields = replay_section(arrays)
    np.testing.assert_allclose(
        fields["actual"], restore(arrays, "display_actual"), rtol=1e-14, atol=1e-14
    )
    assert fields["macro_edges"].shape[1:] == (2, 2)
    assert np.all(
        arrays["display_owners"][fields["cells"]]
        == arrays["display_owners"][fields["cells"][:, :1]]
    )


def test_section_consumer_matches_terminal_acquired_error_quadrature(example):
    """The coarse cubic error distinguishes the acquired q12 norm from its q10 control."""
    from examples.sample_core_sections import mshho_fields

    solution, arrays = example
    arrays.update(capture_section(solution, 2))
    name = "cube-p0" if isinstance(solution.skeleton.mesh, PolyhedralMesh) else "tetra-p0"
    _, errors, _ = mshho_fields(name, 1, 2, archive=arrays)
    assert errors["pressure_l2"] == pytest.approx(
        solution.l2_error(exact.pressure3d, 12), rel=2e-14
    )
    assert errors["flux_l2"] == pytest.approx(solution.flux_l2_error(exact.flux3d, 12), rel=2e-14)


def test_section_rejects_basis_and_interface_merging(example):
    """A mixed-owner triangle would smooth an interface and is inadmissible."""
    solution, original = example
    original = dict(original)
    original.update(capture_section(solution, 2))
    arrays = {k: v.copy() for k, v in original.items()}
    arrays["display_cardinal_derivatives"].flat[0] += 0.01
    with pytest.raises(ValueError, match="derivative"):
        validate_section(arrays)
    arrays = {k: v.copy() for k, v in original.items()}
    first = arrays["display_cells"][0, 0]
    other = np.flatnonzero(
        np.any(arrays["display_owners"] != arrays["display_owners"][first], axis=1)
    )[0]
    arrays["display_cells"][0, 1] = other
    with pytest.raises(ValueError, match="merge"):
        validate_section(arrays)
