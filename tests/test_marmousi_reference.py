"""Independent native acoustic patches and point-load reference checks."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.materials.cartesian import CartesianCellField

pytestmark = pytest.mark.fem


@pytest.fixture
def reference(monkeypatch):
    """Load the original reference application with optional native imports lazy."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.marmousi_reference")


@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("layered", [False, True])
def test_native_complex_constant_and_physical_boundary(reference, degree, layered):
    """Check both complex signs, strong pressure and heterogeneous impedance exactly."""
    pytest.importorskip("dolfinx")
    from mpi4py import MPI

    rho = CartesianCellField(np.array([[2.0], [20.0]]), (0.5, 1.0)) if layered else 2.0
    pressure, omega, modulus = 1 + 0.3j, 2.3, 7.0
    result = reference.native_reference(
        (4, 4),
        degree=degree,
        bounds=(0, 1, 0, 1),
        omega=omega,
        density=rho,
        bulk_modulus=modulus,
        volume_source=-(omega**2) / modulus * pressure,
        impedance_pressure=pressure,
        dirichlet=pressure,
        comm=MPI.COMM_SELF,
    )
    assert_allclose(result["pressure"], pressure, rtol=2e-12, atol=2e-12)
    assert result["stored_original_residual"] < 1e-10
    assert_allclose(result["pressure_l2"], abs(pressure), rtol=2e-12)


def test_native_point_load_matches_complex_conforming_solve(reference):
    """Compare the symmetric real native system to independently assembled complex CG1."""
    pytest.importorskip("dolfinx")
    from mpi4py import MPI
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    from pymhm.fem.scalar.helmholtz import volume_forms
    from pymhm.meshes.triangle import TriangleMesh

    n, omega = 4, 2.3
    mesh = TriangleMesh.unit_square(n)
    stiffness, mass, _ = volume_forms(mesh, 1, 2.0, 7.0, 0.0, 6)
    matrix = (stiffness - omega**2 * mass).astype(complex).tolil()
    for face in mesh.boundary_faces:
        vertices = mesh.points[mesh.faces[face]]
        if np.all(vertices[:, 1] == 0):
            continue
        block = (
            -1j
            * omega
            / np.sqrt(14)
            * np.linalg.norm(vertices[1] - vertices[0])
            / 6
            * np.array([[2, 1], [1, 2]])
        )
        matrix[np.ix_(mesh.faces[face], mesh.faces[face])] += block
    rhs = np.zeros(len(mesh.points), dtype=complex)
    source = np.flatnonzero(np.all(mesh.points == [0.5, 0.5], axis=1))[0]
    rhs[source] = 1
    free = np.flatnonzero(mesh.points[:, 1] != 0)
    expected = np.zeros(len(mesh.points), dtype=complex)
    expected[free] = spsolve(sparse.csc_matrix(matrix)[free][:, free], rhs[free])
    result = reference.native_reference(
        (n, n),
        degree=1,
        bounds=(0, 1, 0, 1),
        omega=omega,
        density=2.0,
        bulk_modulus=7.0,
        point_sources=((0.5, 0.5, 1.0),),
        comm=MPI.COMM_SELF,
    )
    ids = np.rint(result["coordinates"][:, 0] * n).astype(int) + (n + 1) * np.rint(
        result["coordinates"][:, 1] * n
    ).astype(int)
    assert_allclose(result["pressure"], expected[ids], atol=3e-13, rtol=3e-12)


def test_reject_unresolved_material(reference):
    """A coarse classical cell may not average unresolved material pixels."""
    field = CartesianCellField(np.ones((4, 4)), (0.25, 0.25))
    with pytest.raises(ValueError, match="resolve|larger"):
        reference.validate_material_grid((2, 2), (0, 1, 0, 1), field)


def test_native_reference_matches_condensed_quadrilateral_patch(reference):
    """Check the independent reference against condensed Q3 with the same complex PDE."""
    pytest.importorskip("dolfinx")
    from mpi4py import MPI

    from pymhm._legacy.models.waves.helmholtz import solve_helmholtz
    from pymhm.fem.traces.helmholtz import helmholtz_skeleton
    from pymhm.meshes.cartesian import CartesianMacroMesh

    pressure, omega = 1.0 + 0.3j, 2.3
    density, modulus = 2.0, 7.0
    native = reference.native_reference(
        (4, 4),
        degree=3,
        bounds=(0, 1, 0, 1),
        omega=omega,
        density=density,
        bulk_modulus=modulus,
        volume_source=-(omega**2) / modulus * pressure,
        impedance_pressure=pressure,
        dirichlet=pressure,
        comm=MPI.COMM_SELF,
    )
    mesh = CartesianMacroMesh(2, 2)
    absorbing = {
        int(face): -1j * omega / np.sqrt(density * modulus) * pressure
        for face in mesh.boundary_faces
        if not np.all(mesh.points[mesh.faces[face], 1] == 0)
    }
    mhm = solve_helmholtz(
        mesh,
        omega=omega,
        density=density,
        bulk_modulus=modulus,
        source=-(omega**2) / modulus * pressure,
        dirichlet=pressure,
        absorbing=absorbing,
        skeleton=helmholtz_skeleton(mesh, omega, degree=1),
        degree=3,
        local_refinement=2,
    )
    assert_allclose(native["pressure"], pressure, atol=2e-12, rtol=2e-12)
    for local in mhm.pressure:
        assert_allclose(local, pressure, atol=2e-12, rtol=2e-12)
    assert mhm.hybrid.residual < 1e-10
