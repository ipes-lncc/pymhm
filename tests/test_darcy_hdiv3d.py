"""Physical mixed Darcy patches, fine-grid equivalence and affine 3D boundary contracts."""

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from threadpoolctl import threadpool_limits

from pymhm.darcy_hdiv3d import Mixed3DSkeleton, hdiv3d_operators, solve_darcy_hdiv3d
from pymhm.hdiv3d_family import (
    HDiv3DFamily,
    cell_quadrature,
    face_polynomials,
    face_quadrature,
    face_shape,
)
from pymhm.hdiv3d_mesh import AffineMixedMesh, hdiv3d_basis, hdiv3d_dofs
from pymhm.solvers import solve_linear


@dataclass(frozen=True)
class Analytic:
    """Smooth affine or quadratic polynomial pressure with an exact Darcy field."""

    quadratic: bool = False

    def pressure(self, x):
        """Return a pressure with genuinely nonhomogeneous boundary values."""
        return 1 + x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2] + (x[:, 0] ** 2 if self.quadratic else 0)

    def flux(self, x):
        """Return negative gradient, component by component."""
        result = np.tile([-1.0, -2.0, -3.0], (len(x), 1))
        if self.quadratic:
            result[:, 0] -= 2 * x[:, 0]
        return result


@pytest.fixture(autouse=True)
def one_native_thread():
    """Keep numerical contracts small and deterministic in portable CI."""
    with threadpool_limits(1):
        yield


@pytest.mark.parametrize("kind,degree", [("tetrahedron", 1), ("tetrahedron", 2), ("prism", 1)])
@pytest.mark.parametrize("neumann", [False, True])
def test_affine_patch_and_physical_mean(kind, degree, neumann):
    """Every family preserves affine pressure, physical normal data and its requested mean."""
    mesh = AffineMixedMesh.unit_cube(kind=kind)
    data = Analytic()
    flux = data.flux(np.zeros((1, 3)))[0]
    boundary = (
        {int(f): float(flux @ mesh.normals[f]) for f in mesh.boundary_faces} if neumann else None
    )
    result = solve_darcy_hdiv3d(
        mesh,
        pressure_degree=degree,
        local_refinement=1,
        dirichlet=data.pressure,
        neumann=boundary,
        mean_pressure=4,
    )
    error = result.errors(data.pressure, data.flux)
    assert max(error.values()) < 3e-12
    assert np.max(result.physical_residuals) < 2e-12
    assert max(np.max(abs(row)) for row in result.equilibrium_residuals()) < 2e-12


def _fine_mesh(macro, r):
    """Glue the actual fine geometry by exact dyadic Cartesian coordinates."""
    pieces = [macro.submesh(i, r) for i in range(len(macro.cells))]
    points = np.vstack([piece.points for piece in pieces])
    offsets = np.r_[0, np.cumsum([len(piece.points) for piece in pieces])]
    cells = np.vstack([piece.cells + offsets[i] for i, piece in enumerate(pieces)])
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    return AffineMixedMesh(unique, inverse[cells], macro.kind)


@pytest.mark.parametrize("kind,degree", [("tetrahedron", 1), ("tetrahedron", 2), ("prism", 1)])
def test_complete_trace_equals_classical_conforming_system(kind, degree):
    """All fine-face moments recover a separately assembled uncondensed mixed system."""
    macro = AffineMixedMesh.unit_cube(kind=kind)
    data = Analytic(True)
    result = solve_darcy_hdiv3d(
        macro,
        pressure_degree=degree,
        local_refinement=2,
        subdivisions=2,
        source=-2.0,
        dirichlet=data.pressure,
    )
    fine = _fine_mesh(macro, 2)
    family = HDiv3DFamily(kind, degree)
    mass, div, load, _ = hdiv3d_operators(fine, family, source=-2.0)
    rhs = np.zeros(mass.shape[0])
    for face in fine.boundary_faces:
        uv, w = face_quadrature(len(fine.faces[face]), 5)
        tests = face_polynomials(uv, len(fine.faces[face]))
        dual = tests @ np.linalg.inv(tests.T @ (w[:, None] * tests))
        physical = face_shape(uv, len(fine.faces[face])) @ fine.points[fine.faces[face]]
        rhs[fine.face_offsets[face] : fine.face_offsets[face + 1]] = -dual.T @ (
            w * data.pressure(physical)
        )
    matrix = sparse.bmat([[mass, -div.T], [-div, None]], format="csc")
    field = solve_linear(matrix, np.r_[rhs, -load])
    centers = fine.points[fine.cells].mean(axis=1)
    points, _ = cell_quadrature(kind, 3)
    fullp = field[mass.shape[0] :].reshape(len(fine.cells), -1)
    for cell, local in enumerate(result.local_meshes):
        indices = np.array(
            [
                np.argmin(np.linalg.norm(centers - x, axis=1))
                for x in local.points[local.cells].mean(axis=1)
            ]
        )
        expectedp = fullp[indices] @ family.tabulate(points)[2].T
        expectedq = np.einsum(
            "tqia,ti->tqa",
            hdiv3d_basis(fine, family, points)[0][indices],
            field[: mass.shape[0]][hdiv3d_dofs(fine, family)[indices]],
        )
        actualp, actualq, _ = result.evaluate(cell, points)
        assert_allclose(actualp, expectedp, atol=3e-12)
        assert_allclose(actualq, expectedq, atol=3e-11)
    if degree == 2:
        assert max(result.errors(data.pressure, data.flux).values()) < 5e-12


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_factory_parallel_field_equality(backend):
    """Workers and callers with different BLAS settings reconstruct identical physical fields."""
    from pymhm.hdiv3d_family import _coefficients

    mesh = AffineMixedMesh.unit_cube(kind="prism")
    data = Analytic()
    _coefficients.cache_clear()
    try:
        # Spawned workers construct their basis under map_local's one-thread
        # limit, while this caller has already constructed its four-thread basis.
        with threadpool_limits(4):
            _ = HDiv3DFamily("prism").coefficients
        serial = solve_darcy_hdiv3d(mesh, local_refinement=1, dirichlet=data.pressure)
        parallel = solve_darcy_hdiv3d(
            mesh, local_refinement=1, dirichlet=data.pressure, backend=backend, workers=2
        )
        assert_allclose(parallel.hybrid.trace, serial.hybrid.trace, atol=2e-12)
        points, _ = cell_quadrature("prism", 3)
        for cell in range(len(mesh.cells)):
            for actual, expected in zip(
                parallel.evaluate(cell, points), serial.evaluate(cell, points), strict=True
            ):
                assert_allclose(actual, expected, atol=2e-11)
        assert max(parallel.errors(data.pressure, data.flux).values()) < 2e-11
    finally:
        _coefficients.cache_clear()


def test_coarse_traces_mixed_boundary_and_invalid_contracts():
    """P0 traces, physical side flux and incompatible pure Neumann data remain explicit."""
    mesh = AffineMixedMesh.unit_cube(kind="prism")
    zero = {int(f): 0.0 for f in mesh.boundary_faces if abs(mesh.normals[f, 2]) < 0.1}
    result = solve_darcy_hdiv3d(
        mesh, trace_degree=0, local_refinement=1, dirichlet=lambda x: x[:, 2], neumann=zero
    )
    assert max(result.errors(lambda x: x[:, 2], [0.0, 0.0, -1.0]).values()) < 2e-12
    for kwargs in [
        dict(trace_degree=2),
        dict(local_refinement=1, subdivisions=2),
        dict(neumann={-1: 0}),
        dict(neumann=dict.fromkeys(mesh.boundary_faces, 0.0), mean_pressure=np.nan),
        dict(neumann=dict.fromkeys(mesh.boundary_faces, 0.0), source=1.0),
    ]:
        with pytest.raises(ValueError):
            solve_darcy_hdiv3d(mesh, **kwargs)
    skeleton = Mixed3DSkeleton(mesh, 1, 2)
    assert len(skeleton.cell_dofs(0)) == 72


def test_trace_alignment_and_missing_parent_diagnostics():
    """A fine face crossing a subface or leaving the macro boundary cannot define a trace map."""
    from pymhm.darcy_hdiv3d import _partition, _trace_mapping

    partition = _partition(3, 2, 1, 1.0)
    with pytest.raises(ValueError, match="align"):
        partition.locate(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    mesh = AffineMixedMesh.unit_cube(kind="prism")
    fine = mesh.submesh(0, 1)
    outside = AffineMixedMesh(fine.points + [3, 3, 3], fine.cells, "prism")
    with pytest.raises(ValueError, match="containing"):
        _trace_mapping(Mixed3DSkeleton(mesh), 0, outside)


def test_corrupted_reconstruction_fails_physical_block_gate(monkeypatch):
    """A returned field must satisfy every physical mixed block, independently of solver status."""
    from dataclasses import replace

    from pymhm.hybrid import HybridSystem

    original = HybridSystem.solve

    def corrupted(self, **kwargs):
        """Inject a reconstruction defect to exercise the independent physical check."""
        result = original(self, **kwargs)
        fields = [field.copy() for field in result.fields]
        fields[0][0] += 1.0
        return replace(result, fields=tuple(fields))

    monkeypatch.setattr(HybridSystem, "solve", corrupted)
    with pytest.raises(ValueError, match="physical mixed block"):
        solve_darcy_hdiv3d(
            AffineMixedMesh.unit_cube(kind="prism"),
            local_refinement=1,
            dirichlet=Analytic().pressure,
            boundary_quadrature_order=6,
        )
