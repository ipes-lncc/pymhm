"""Robin formulation, physical flux, ellipticity and MHM-limit checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.traces.integration import integrate_dirichlet_trace
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.linalg.linear import LinearSolveError
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.robin import solve_mh


@pytest.fixture(autouse=True)
def one_native_thread():
    """Keep small verification systems independent of BLAS oversubscription."""
    with threadpool_limits(1):
        yield


def quadratic(x):
    """Nonharmonic polynomial with nonzero boundary values."""
    return 1 + x[:, 0] ** 2 + x[:, 0] * x[:, 1] + 2 * x[:, 1] ** 2


def quadratic_flux(x):
    """Physical flux for K=[[3,.4],[.4,2]], differentiated independently."""
    return -np.column_stack((2 * x[:, 0] + x[:, 1], x[:, 0] + 4 * x[:, 1])) @ np.array(
        [[3.0, 0.4], [0.4, 2.0]]
    )


@pytest.mark.parametrize("kind", ["triangle", "nonconvex"])
def test_exact_nonhomogeneous_quadratic_and_physical_normal_flux(kind):
    if kind == "triangle":
        mesh = TriangleMesh.unit_square(2)
    else:
        points = np.array([[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1], [1, 1.0]])
        mesh = PolygonMesh(points, (np.array([0, 1, 2, 3, 4, 5]), np.array([3, 2, 6, 4])))
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
    result = solve_mh(
        mesh,
        skeleton=skeleton,
        degree=2,
        local_refinement=2,
        permeability=np.array([[3.0, 0.4], [0.4, 2.0]]),
        source=-14.8,
        dirichlet=quadratic,
        robin_parameter=0.25,
    )
    assert result.l2_error(quadratic) < 3e-11
    assert result.flux_l2_error(quadratic_flux) < 3e-11
    assert np.max(abs(result.conservation_residuals())) < 3e-12
    assert len(result.hybrid.coarse[0]) == 0
    assert np.linalg.eigvalsh(result.system.matrix.toarray()).min() > 0
    for cell, faces in enumerate(mesh.cell_faces):
        for side, face in enumerate(faces):
            t = np.linspace(0, 1, 5)
            a, b = mesh.points[mesh.faces[face]]
            expected = quadratic_flux(a + t[:, None] * (b - a)) @ mesh.normals[face]
            expected *= mesh.signs[cell, side]
            assert_allclose(result.normal_flux(cell, int(face), t), expected, atol=3e-10)
    for response, (fine, robin) in zip(
        result.system.responses, result.system.local_metadata, strict=True
    ):
        ones = np.ones(len(response.problem.load))
        assert np.linalg.eigvalsh(response.problem.matrix.toarray()).min() > 0
        assert_allclose(ones @ robin @ ones, 0.25 * fine.areas.sum(), atol=1e-15)
    with pytest.raises(ValueError, match="incident"):
        solve_mh(TriangleMesh.unit_square(2)).normal_flux(
            0, len(TriangleMesh.unit_square(2).faces) - 1, [0.5]
        )


def test_robin_condensation_matches_full_saddle_nonpolynomial_data():
    mesh = TriangleMesh.unit_square(2)
    result = solve_mh(mesh, source=lambda x: np.exp(x[:, 0] + x[:, 1]), dirichlet=quadratic)
    problems = [response.problem for response in result.system.responses]
    operator = sparse.block_diag([p.matrix for p in problems], format="csc")
    coupling = sparse.vstack(
        [
            sparse.coo_matrix(
                (
                    p.coupling.ravel(),
                    (
                        np.repeat(np.arange(len(p.load)), p.coupling.shape[1]),
                        np.tile(p.trace_dofs, len(p.load)),
                    ),
                ),
                shape=(len(p.load), result.skeleton.size),
            )
            for p in problems
        ]
    ).tocsc()
    full = sparse.bmat([[operator, coupling], [coupling.T, None]], format="csc")
    boundary = boundary_data(result.skeleton, quadratic, None, order=8)[0]
    expected = spsolve(full, np.r_[np.concatenate([p.load for p in problems]), boundary])
    actual = np.r_[np.concatenate(result.pressure), result.hybrid.trace]
    assert_allclose(actual, expected, atol=6e-11, rtol=2e-11)
    for response, pressure in zip(result.system.responses, result.pressure, strict=True):
        problem = response.problem
        assert_allclose(
            problem.matrix @ pressure + problem.coupling @ result.hybrid.trace[problem.trace_dofs],
            problem.load,
            atol=8e-13,
        )


def test_nonhomogeneous_kink_uses_fine_boundary_cuts_without_enriching_trace():
    """Boundary moments agree with analytical integrals of a broken affine datum."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))

    def datum(points):
        """Put a derivative jump at fine-edge midpoints of every exterior macroface."""
        return abs(points[:, 0] - 0.5) + 2 * abs(points[:, 1] - 0.5)

    result = solve_mh(
        mesh, skeleton=skeleton, degree=2, local_refinement=2, quadrature_order=3, dirichlet=datum
    )
    moments = np.zeros(skeleton.size)
    for face in mesh.boundary_faces:
        vertices = mesh.points[mesh.faces[face]]
        horizontal = vertices[0, 1] == vertices[1, 1]
        moments[skeleton.dofs(int(face))] = (1.25 if horizontal else 1, 0)
    assert_allclose(result.system.rhs, -moments, atol=8e-15)
    assembled = np.zeros(skeleton.size)
    for response, pressure in zip(result.system.responses, result.pressure, strict=True):
        problem = response.problem
        np.add.at(assembled, problem.trace_dofs, problem.coupling.T @ pressure)
    assert_allclose(assembled, moments, atol=3e-13)
    assert all(len(space.breaks) == 2 for space in skeleton.faces)
    coarse_moments = boundary_data(skeleton, datum, order=4)[0]
    assert np.max(abs(coarse_moments - moments)) > 0.01
    with pytest.raises(ValueError, match="one skeleton component"):
        integrate_dirichlet_trace(
            SkeletonSpace(mesh, components=2), result.local_meshes, datum, 2, 4
        )


def test_mh_approaches_same_discrete_mhm_linearly_in_robin_parameter():
    mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    reference = solve_darcy(mesh, source=1, skeleton=skeleton, degree=3, local_refinement=2)
    errors = []
    for parameter in (0.25, 0.125, 0.0625):
        result = solve_mh(
            mesh,
            source=1,
            skeleton=skeleton,
            degree=3,
            local_refinement=2,
            robin_parameter=parameter,
        )
        errors.append(
            np.linalg.norm(np.concatenate(result.pressure) - np.concatenate(reference.pressure))
        )
    assert 1.98 < errors[0] / errors[1] < 2.02
    assert 1.98 < errors[1] / errors[2] < 2.02


def test_literal_cartesian_and_certified_variable_tensor_materials():
    mesh = TriangleMesh.unit_square()
    for material in (
        CartesianCellField(np.array([[1.0, 2.0], [3.0, 4.0]]), (0.5, 0.5)),
        CartesianCellField(np.broadcast_to([[2.0, 0.1], [0.1, 3.0]], (2, 2, 2, 2)), (0.5, 0.5)),
    ):
        result = solve_mh(mesh, permeability=material, dirichlet=2)
        assert result.l2_error(2) < 3e-12
    result = solve_mh(
        mesh,
        permeability=lambda x: 2 + x[:, 0],
        ellipticity_lower_bound=2,
        dirichlet=1,
        origin=(-0.1, -0.1),
    )
    assert result.l2_error(1) < 3e-12
    with pytest.raises(ValueError, match="sampled"):
        solve_mh(mesh, permeability=lambda x: np.ones(len(x)), ellipticity_lower_bound=2)


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps == np.finfo(float).eps,
    reason="explicit extended accumulation requires a wider native long-double type",
)
def test_explicit_extended_responses_and_fields_preserve_precision():
    result = solve_mh(
        TriangleMesh.unit_square(),
        source=1,
        local_refinement_precision="extended",
        refinement_precision="extended",
    )
    assert result.pressure[0].dtype == np.dtype(np.longdouble)
    assert result.system.responses[0].lifts.dtype == np.dtype(np.longdouble)
    assert result.hybrid.trace.dtype == np.dtype(np.longdouble)
    assert np.max(abs(result.conservation_residuals())) < 1e-12


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_portable_parallel_local_factory(backend):
    mesh = TriangleMesh.unit_square()
    result = solve_mh(mesh, source=1, backend=backend, workers=2)
    reference = solve_mh(mesh, source=1)
    assert_allclose(result.hybrid.trace, reference.hybrid.trace, atol=1e-13)
    assert_allclose(np.concatenate(result.pressure), np.concatenate(reference.pressure), atol=1e-13)


def test_inputs_and_incompatible_trace_spaces_are_rejected():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(TypeError, match="two-dimensional"):
        solve_mh(None)
    with pytest.raises(ValueError, match="scalar skeleton"):
        solve_mh(mesh, skeleton=SkeletonSpace(mesh, components=2))
    with pytest.raises(ValueError, match="certified"):
        solve_mh(mesh, permeability=lambda x: np.ones(len(x)))
    for lower in (0, np.nan, [1, 2], 1j):
        with pytest.raises(ValueError, match="positive scalar"):
            solve_mh(mesh, ellipticity_lower_bound=lower)
    for parameter in (0, -1, 100, np.nan, 1j):
        with pytest.raises(ValueError, match="coercivity"):
            solve_mh(mesh, robin_parameter=parameter)
    for origin in ((0, 1, 2), (np.nan, 0), (1j, 0)):
        with pytest.raises(ValueError, match="origin"):
            solve_mh(mesh, origin=origin)
    with pytest.raises(LinearSolveError):
        solve_mh(
            mesh,
            degree=1,
            local_refinement=1,
            skeleton=SkeletonSpace(mesh, tuple(FaceSpace.uniform(3) for _ in mesh.faces)),
        )
    result = solve_mh(mesh)
    for parameters in ([-0.1], [1.1], [np.nan], [[0.5]]):
        with pytest.raises(ValueError, match="parameters"):
            result.normal_flux(0, int(mesh.cell_faces[0, 0]), parameters)
