"""Analytical geometry, volume/surface norm and persisted-basis checks for the well replay."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.hdiv3d_family import HDiv3DFamily
from pymhm.hdiv3d_mesh import AffineMixedMesh, hdiv3d_dofs


@pytest.fixture(autouse=True)
def single_native_thread():
    """Keep small algebraic checks independent of native thread oversubscription."""
    with threadpool_limits(1):
        yield


@pytest.fixture
def verification(monkeypatch):
    """Load the original replay driver without changing the installed package boundary."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("verify_mixed_well_fields")


class ConstantExact:
    """Constant exact state with nonzero physical pressure drop and horizontal flux."""

    outer_pressure = 1.0
    height = 10.0

    def pressure(self, points):
        """Return a constant pressure with exactly known L2 norm."""
        return np.full(len(points), 3.0)

    def flux(self, points):
        """Return a constant vector with squared magnitude five."""
        return np.broadcast_to([1.0, 2.0, 0.0], (len(points), 3))


def field(verification, kind, pressure, flux):
    """Represent a physical constant field on a unit cube whose top lies at z=5."""
    mesh = AffineMixedMesh.unit_cube(kind=kind)
    mesh = AffineMixedMesh(mesh.points + [0, 0, 4], mesh.cells, kind)
    qp = verification.powers(kind, 2)
    pp = verification.powers(kind, 1)
    coefficients = np.zeros((len(mesh.cells), 3, len(qp)))
    coefficients[:, :, 0] = mesh.determinants[:, None] * np.einsum("tab,b->ta", mesh.inverse, flux)
    p = np.zeros((len(mesh.cells), len(pp)))
    p[:, 0] = pressure
    faces = [f for f in mesh.boundary_faces if np.all(mesh.points[mesh.faces[f], 2] == 5)]
    return verification.ArchivedField(
        record={"kind": kind, "pressure_degree": 1},
        vertices=mesh.points[mesh.cells],
        jacobian=mesh.jacobian,
        inverse=mesh.inverse,
        determinants=mesh.determinants,
        flux=coefficients,
        pressure=p,
        q_powers=qp,
        p_powers=pp,
        top_cells=np.array([mesh.incidence[f][0][0] for f in faces]),
        top_vertices=np.array([mesh.points[mesh.faces[f]] for f in faces]),
    )


@pytest.mark.parametrize("kind", ["tetrahedron", "prism"])
def test_nonzero_volume_surface_norms_and_cross_term(kind, monkeypatch, verification):
    """Exact constant fields test physical measures, denominators and the sign of the cross term."""
    monkeypatch.setattr(verification, "WellData", ConstantExact)
    candidate = field(verification, kind, 4.0, [2.0, 2.0, 0.0])
    reference = field(verification, kind, 3.0, [1.5, 2.0, 0.0])
    result = verification.integrated_pair(candidate, reference, 4)
    assert_allclose(result["pressure_difference_over_reference"], 1 / 3, rtol=2e-14)
    assert_allclose(result["mhm_drawdown_error_relative"], 0.5, rtol=2e-14)
    assert_allclose(result["mhm_flux_error_relative"], 1 / np.sqrt(5), rtol=2e-14)
    assert_allclose(result["classical_flux_error_relative"], 0.5 / np.sqrt(5), rtol=2e-14)
    assert_allclose(result["flux_cross_term_over_exact_squared"], 0.25 / 5, rtol=2e-14)
    assert_allclose(result["pythagoras_relative_defect"], 0.5, rtol=2e-14)
    surface = verification.top_surface_error(candidate, 4)
    assert_allclose(surface["flux_error_l2"], 1, rtol=2e-14)
    assert_allclose(surface["exact_flux_l2"], np.sqrt(5), rtol=2e-14)
    assert surface["vertical_flux_l2"] == 0
    reference.vertices[0, 0, 0] += 0.02
    with pytest.raises(ValueError, match="partitions"):
        verification.matching_cells(candidate, reference)


@pytest.mark.parametrize("kind", ["tetrahedron", "prism"])
def test_archive_read_preserves_stored_coefficients_and_rejects_changed_digest(
    kind, tmp_path, verification
):
    """The executed basis is mandatory and physically consistent with polynomial replay."""
    family = HDiv3DFamily(kind)
    mesh = AffineMixedMesh.unit_cube(kind=kind)
    mesh = AffineMixedMesh(mesh.points + [0, 0, 4], mesh.cells, kind)
    vector = np.random.default_rng(66).normal(size=int(hdiv3d_dofs(mesh, family).max()) + 1)
    coefficients = family.coefficients
    filename = tmp_path / "field.npz"
    np.savez(
        filename,
        local_points=mesh.points[None],
        local_cells=mesh.cells[None],
        pressure=np.zeros((1, len(mesh.cells), family.pressure_size)),
        flux=vector[None],
        flux_basis_coefficients=coefficients,
    )
    record = {
        "kind": kind,
        "pressure_degree": 1,
        "archive_schema": 2,
        "archive": filename.name,
        "sha256": verification.digest(filename),
        "flux_basis_sha256": hashlib.sha256(coefficients.tobytes()).hexdigest(),
    }
    metadata = tmp_path / "field.json"
    metadata.write_text(json.dumps(record))
    loaded = verification.read_field("field", tmp_path)
    from pymhm.hdiv3d_mesh import hdiv3d_basis

    points = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    expected = np.einsum(
        "tqia,ti->tqa", hdiv3d_basis(mesh, family, points)[0], vector[hdiv3d_dofs(mesh, family)]
    )
    actual = loaded.evaluate(
        np.arange(len(mesh.cells)), np.broadcast_to(points, (len(mesh.cells), *points.shape))
    )[1]
    assert_allclose(actual, expected, atol=2e-12)
    record["flux_basis_sha256"] = "wrong"
    metadata.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="basis digest"):
        verification.read_field("field", tmp_path)
    record["archive_schema"] = 1
    metadata.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="schema 2"):
        verification.read_field("field", tmp_path)
