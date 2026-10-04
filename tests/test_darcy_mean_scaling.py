"""Physical mean constraints are invariant under auxiliary multiplier coordinates."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.darcy.primal import _DarcyLocalFactory, solve_darcy
from pymhm.core.contracts import LocalProblem
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


@pytest.fixture(autouse=True)
def one_thread():
    """Keep small linear algebra checks independent of native oversubscription."""
    with threadpool_limits(1):
        yield


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
@pytest.mark.parametrize("length", [1.0, 1000.0])
def test_local_average_preserves_physical_integrals_and_schur_operator(formulation, length):
    """Replacing an integral by its average changes only the auxiliary multiplier."""
    unit = TriangleMesh.unit_square()
    mesh = TriangleMesh(length * unit.points, unit.cells)
    skeleton = SkeletonSpace(mesh)
    factory = _DarcyLocalFactory(
        mesh,
        skeleton,
        np.array([[3.0, 0.4], [0.4, 2.0]]),
        1 / length**2,
        tuple(np.empty((0, 3)) for _ in mesh.cells),
        2,
        None,
        1,
        formulation,
        5,
    )
    item = factory(0)
    problem = item.problem
    physical_mean = item.metadata[1]
    assert_allclose(physical_mean @ problem.kernel[:, 0], mesh.areas[0], rtol=3e-15)
    assert_allclose(problem.constraints.T @ problem.kernel, [[1.0]], rtol=3e-15)
    assert_allclose(problem.constraints[:, 0] * mesh.areas[0], physical_mean, rtol=3e-15)
    original = LocalProblem(
        problem.matrix,
        problem.coupling,
        problem.load,
        problem.trace_dofs,
        problem.kernel,
        physical_mean[:, None],
    )
    normalized_response = problem.condense()
    original_response = original.condense()
    assert_allclose(normalized_response.source, original_response.source, rtol=2e-11, atol=2e-13)
    assert_allclose(normalized_response.lifts, original_response.lifts, rtol=2e-11, atol=2e-10)
    new = normalized_response.global_contribution(np.array([skeleton.size]))
    old = original_response.global_contribution(np.array([skeleton.size]))
    assert_allclose(new[1], old[1], rtol=2e-11, atol=2e-10)
    assert_allclose(new[2], old[2], rtol=2e-11, atol=2e-10)


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
@pytest.mark.parametrize("all_neumann", [False, True])
def test_domain_units_preserve_pressure_flux_and_physical_gauge(formulation, all_neumann):
    """Changing length units preserves pressure and rescales the physical Darcy flux."""
    unit = TriangleMesh.unit_square(2)
    results = []
    for length in (1.0, 1000.0):
        mesh = TriangleMesh(length * unit.points, unit.cells)

        def pressure(points, length=length):
            """Affine pressure has physical domain average 0.4 in either length unit."""
            return 0.9 + points[:, 0] / length - 2 * points[:, 1] / length

        exact_flux = np.array([-1.0, 2.0]) / length
        faces = mesh.boundary_faces if all_neumann else mesh.boundary_faces[:1]
        prescribed = {int(face): float(exact_flux @ mesh.normals[face]) for face in faces}
        result = solve_darcy(
            mesh,
            formulation=formulation,
            dirichlet=pressure,
            neumann=prescribed,
            mean_pressure=0.4,
            local_refinement=2,
        )
        results.append(result)
        assert result.flux_l2_error(exact_flux) < 2e-11
        assert_allclose(result.conservation_residuals(), 0.0, atol=2e-11)
    for first, second in zip(results[0].pressure, results[1].pressure, strict=True):
        assert_allclose(first, second, rtol=2e-11, atol=2e-11)
    assert_allclose(results[0].hybrid.trace, 1000 * results[1].hybrid.trace, atol=2e-11)
