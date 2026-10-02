"""Three-field equality, boundary signs and independent polynomial MH2M checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pymhm.lagrange import nodal_space
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.mh2m import PressureTraceSpace, solve_mh2m
from pymhm.polygon import PolygonMesh
from pymhm.solvers import LinearSolveError


def polynomial(x):
    """A nonharmonic quadratic with nonzero boundary and mean 5/4."""
    return x[:, 0] ** 2 + x[:, 0] * x[:, 1] + 2 * x[:, 1] ** 2


def gradient(x):
    """Analytical derivative of the independent quadratic pressure."""
    return np.column_stack((2 * x[:, 0] + x[:, 1], x[:, 0] + 4 * x[:, 1]))


@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "neumann"])
def test_anisotropic_quadratic_boundary_conventions(boundary):
    mesh = TriangleMesh.unit_square(2)
    tensor = np.array([[3.0, 0.4], [0.4, 2.0]])
    source = -(2 * tensor[0, 0] + 2 * tensor[0, 1] + 4 * tensor[1, 1])
    selected = (
        mesh.boundary_faces
        if boundary == "neumann"
        else mesh.boundary_faces[:2]
        if boundary == "mixed"
        else []
    )
    neumann = {int(f): lambda x, n=mesh.normals[f]: (-gradient(x) @ tensor) @ n for f in selected}
    result = solve_mh2m(
        mesh,
        permeability=tensor,
        source=source,
        dirichlet=polynomial,
        neumann=neumann,
        pressure_trace=PressureTraceSpace.uniform(mesh, 2),
        flux_space=SkeletonSpace(
            mesh, tuple(FaceSpace.uniform(1, 2 if boundary == "neumann" else 1) for _ in mesh.faces)
        ),
        degree=2,
        local_refinement=4 if boundary == "neumann" else 2,
        mean_pressure=1.25 if boundary == "neumann" else 0,
    )
    assert result.l2_error(polynomial) < 2e-12
    assert result.gradient_l2_error(gradient) < 2e-11
    assert result.flux_l2_error(lambda x: -gradient(x) @ tensor) < 6e-11
    assert_allclose(result.conservation_residuals(), 0, atol=2e-13)
    for local, pressure, continuity, residual in zip(
        result.local,
        result.pressure,
        result.trace_moment_residuals(),
        result.local_equation_residuals(),
        strict=True,
    ):
        assert_allclose(continuity, 0, atol=3e-13)
        assert_allclose(residual, 0, atol=3e-13)
        assert_allclose(local.zero_mean_basis.T @ local.flux_integrals, 0, atol=2e-16)
        assert_allclose(local.stiffness.toarray(), local.stiffness.toarray().T, atol=1e-14)
        _, nodes = nodal_space(local.mesh, 2)
        assert_allclose(pressure, polynomial(nodes), atol=2e-12)
    if boundary == "neumann":
        mean = sum(
            data.volume_moments @ p for data, p in zip(result.local, result.pressure, strict=True)
        )
        assert_allclose(mean, 1.25, atol=2e-13)
    else:
        assert (
            np.linalg.eigvalsh(result.matrix[result.free_dofs][:, result.free_dofs].toarray()).min()
            > 0
        )


@pytest.mark.parametrize("polygonal", [False, True])
def test_three_field_monolithic_equivalence_with_variable_source(polygonal):
    mesh = (
        PolygonMesh(
            np.array([[0.0, 0.0], [0.5, 0], [1, 0], [0, 1], [0.5, 1], [1, 1]]),
            (np.array([0, 1, 4, 3]), np.array([1, 2, 5, 4])),
        )
        if polygonal
        else TriangleMesh.unit_square(2)
    )
    gamma = PressureTraceSpace.uniform(mesh, 2)
    flux = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 3) for _ in mesh.faces))
    result = solve_mh2m(
        mesh,
        source=lambda x: np.exp(x[:, 0] - x[:, 1]),
        dirichlet=lambda x: x[:, 0] - 2 * x[:, 1],
        pressure_trace=gamma,
        flux_space=flux,
        degree=2,
        local_refinement=6,
    )
    sizes = [len(data.load) for data in result.local]
    counts = [len(data.flux_integrals) for data in result.local]
    total_p, total_l = sum(sizes), sum(counts)
    # Build the uncondensed three-field equations independently, without Gh,
    # zero-average bases, source decomposition, or the condensed global matrix.
    a = sparse.block_diag([data.stiffness for data in result.local])
    b = sparse.block_diag([sparse.csc_matrix(data.boundary_coupling) for data in result.local])
    d = np.zeros((total_l, gamma.size))
    offset = 0
    for count, data in zip(counts, result.local, strict=True):
        d[offset : offset + count, data.trace_dofs] = data.trace_pairing
        offset += count
    operator = sparse.bmat([[a, -b, None], [-b.T, None, d], [None, d.T, None]], format="csc")
    rhs = np.r_[
        np.concatenate([data.load for data in result.local]), np.zeros(total_l + gamma.size)
    ]
    fixed = (
        np.unique(np.concatenate([gamma.face_dofs[f] for f in mesh.boundary_faces]))
        + total_p
        + total_l
    )
    expected = np.zeros(len(rhs))
    expected[fixed] = result.trace[fixed - total_p - total_l]
    free = np.setdiff1d(np.arange(len(rhs)), fixed)
    expected[free] = spsolve(operator[free][:, free], (rhs - operator @ expected)[free])
    actual = np.r_[np.concatenate(result.pressure), np.concatenate(result.conormal), result.trace]
    assert_allclose(actual, expected, atol=3e-11, rtol=2e-11)
    assert result.residual < 2e-13


def test_remark16_even_k_and_single_triangle_all_prescribed():
    mesh = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    result = solve_mh2m(
        mesh, degree=1, local_refinement=1, dirichlet=lambda x: 1 + x[:, 0] - 3 * x[:, 1]
    )
    assert len(result.free_dofs) == 0
    assert result.l2_error(lambda x: 1 + x[:, 0] - 3 * x[:, 1]) < 1e-14
    assert result.flux_l2_error((-1, 3)) < 1e-14


def test_independent_face_degrees_share_vertices_and_interpolate_constants():
    mesh = TriangleMesh.unit_square()
    faces = tuple(FaceSpace((0.0, 0.2, 1.0), (1, 3), continuous=True) for _ in mesh.faces)
    gamma = PressureTraceSpace(mesh, faces)
    assert gamma.size == len(mesh.points) + 3 * len(mesh.faces)
    for f, ids in enumerate(gamma.face_dofs):
        assert_allclose(gamma.nodes[ids[[0, 2]]], mesh.points[mesh.faces[f]])
        assert_allclose(faces[f].evaluate(np.linspace(0, 1, 7)).sum(axis=1), 1)
    # Independent Lambda cuts need not coincide with Gamma or fine-edge cuts.
    flux = SkeletonSpace(
        mesh, tuple(FaceSpace((0.0, 0.3, 0.7, 1.0), (1, 1, 1)) for _ in mesh.faces)
    )
    result = solve_mh2m(
        mesh, pressure_trace=gamma, flux_space=flux, degree=3, local_refinement=8, dirichlet=2
    )
    assert result.l2_error(2) < 2e-12


@pytest.mark.parametrize("source", [1.0, 1e-12])
def test_incompatible_pure_neumann_rejected_at_physical_source_scale(source):
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="incompatible Neumann"):
        solve_mh2m(mesh, source=source, neumann={int(f): 0 for f in mesh.boundary_faces})


def test_zero_neumann_and_nonzero_pressure_gauge():
    mesh = TriangleMesh.unit_square()
    result = solve_mh2m(mesh, neumann={int(f): 0 for f in mesh.boundary_faces}, mean_pressure=3)
    assert result.l2_error(3) < 1e-12
    assert_allclose(result.conservation_residuals(), 0, atol=1e-12)


def test_original_trace_equations_reject_an_inaccurate_backend(monkeypatch):
    """An inaccurate backend return must not bypass the original physical equations."""
    monkeypatch.setattr("pymhm.mh2m.solve_linear", lambda a, b, **kw: np.zeros_like(b))
    with pytest.raises(ValueError, match="original trace equations"):
        solve_mh2m(TriangleMesh.unit_square(2), source=1.0)


def test_invalid_discrete_spaces_and_inputs():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(TypeError, match="TriangleMesh"):
        PressureTraceSpace(None)
    with pytest.raises(ValueError, match="Gamma"):
        PressureTraceSpace(mesh, (FaceSpace(),))
    with pytest.raises(TypeError, match="TriangleMesh"):
        solve_mh2m(None)
    with pytest.raises(ValueError, match="supplied mesh"):
        solve_mh2m(mesh, pressure_trace=PressureTraceSpace(TriangleMesh.unit_square()))
    with pytest.raises(ValueError, match="discontinuous"):
        solve_mh2m(
            mesh,
            flux_space=SkeletonSpace(
                mesh, tuple(FaceSpace.uniform(1, continuous=True) for _ in mesh.faces)
            ),
        )
    with pytest.raises(ValueError, match="exterior"):
        solve_mh2m(mesh, neumann={int(np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0]): 0})
    with pytest.raises(ValueError, match="pure Neumann"):
        solve_mh2m(mesh, mean_pressure=1)
    with pytest.raises(ValueError, match="finite"):
        solve_mh2m(mesh, mean_pressure=np.nan)
    with pytest.raises(ValueError, match="injectivity"):
        solve_mh2m(
            mesh,
            degree=1,
            local_refinement=1,
            flux_space=SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces)),
        )
    with pytest.raises(LinearSolveError):
        solve_mh2m(mesh, pressure_trace=PressureTraceSpace.uniform(mesh, 4), local_refinement=4)


@pytest.mark.parametrize("segments", [2, 4])
def test_even_boundary_cycle_requires_more_fine_edges_than_constant_conormals(segments):
    """A P0 conormal on each P1 fine edge admits an alternating zero-load mode."""
    mesh = TriangleMesh.unit_square()
    flux = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, segments) for _ in mesh.faces))
    with pytest.raises(ValueError, match="injectivity"):
        solve_mh2m(mesh, flux_space=flux, degree=1, local_refinement=segments)
    result = solve_mh2m(
        mesh,
        flux_space=flux,
        degree=1,
        local_refinement=2 * segments,
        dirichlet=lambda x: 1 + x[:, 0] + 2 * x[:, 1],
    )
    assert result.l2_error(lambda x: 1 + x[:, 0] + 2 * x[:, 1]) < 2e-13
    assert result.flux_l2_error([-1.0, -2.0]) < 3e-12
