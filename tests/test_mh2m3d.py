"""Three-field tetrahedral equations, independent skeleton spaces and physical gauges."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.darcy3d import TriangularSkeleton
from pymhm.mesh import TriangleMesh
from pymhm.mh2m3d import solve_mh2m_3d
from pymhm.mh_trace3d import PressureTraceSpace3D
from pymhm.solvers import LinearSolveError
from pymhm.tetrahedral import TetraMesh


@pytest.fixture(autouse=True)
def single_thread():
    """Avoid threaded-BLAS scheduling costs in the small physical checks."""
    with threadpool_limits(1):
        yield


@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "neumann"])
def test_quadratic_patch_on_independent_subface_spaces(boundary):
    """Check all local equations and complete source reconstruction with an anisotropic tensor."""
    mesh = TetraMesh.unit_cube()
    tensor = np.array([[3.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.0]])

    def pressure(x):
        """Return a quadratic with physical volume mean two."""
        return 1 + np.sum(x * x, axis=1)

    def flux(x):
        """Return its anisotropic physical flux."""
        return -2 * x @ tensor

    faces = [
        int(f)
        for f in mesh.boundary_faces
        if boundary == "neumann"
        or (boundary == "mixed" and np.all(mesh.points[mesh.faces[f], 0] == 1))
    ]
    natural = {f: lambda x, n=mesh.normals[f]: flux(x) @ n for f in faces}
    result = solve_mh2m_3d(
        mesh,
        permeability=tensor,
        source=-12,
        dirichlet=pressure,
        neumann=natural,
        mean_pressure=2 if boundary == "neumann" else 0,
        pressure_trace=PressureTraceSpace3D(mesh, 2),
        flux_space=TriangularSkeleton(mesh, 2, degree=1),
        local_refinement=4,
    )
    assert result.l2_error(pressure, 4) < 2e-11
    assert result.flux_l2_error(flux, 4) < 2e-11
    assert_allclose(result.conservation_residuals(), 0, atol=2e-12)
    for residual in (*result.trace_moment_residuals(), *result.local_equation_residuals()):
        assert_allclose(residual, 0, atol=3e-12)
    values, q = result.evaluate(0, np.array([[0.25] * 4]))
    points = result.local_meshes[0].points[result.local_meshes[0].cells].mean(axis=1)
    assert_allclose(values[:, 0], pressure(points), atol=1e-11)
    assert_allclose(q[:, 0], flux(points), atol=1e-11)


def test_nonaffine_full_three_field_equations_before_condensation():
    """Solve the complete volume/conormal/pressure-trace system independently."""
    mesh = TetraMesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [0.2, 1, 0], [0.1, 0.2, 1]]), np.array([[0, 1, 2, 3]])
    )
    h = float(mesh.volumes.sum() / mesh.areas.sum())
    result = solve_mh2m_3d(
        mesh,
        source=1,
        permeability=lambda x: 2 + x[:, 0],
        neumann={int(f): h for f in mesh.boundary_faces},
        mean_pressure=1.75,
    )
    local = result.local[0]
    a, b, d = local.stiffness.toarray(), local.boundary_coupling, local.trace_pairing
    nv, nl, ng = len(local.load), b.shape[1], d.shape[1]
    full = np.block(
        [
            [a, -b, np.zeros((nv, ng))],
            [-b.T, np.zeros((nl, nl)), d],
            [np.zeros((ng, nv)), d.T, np.zeros((ng, ng))],
        ]
    )
    gauge = np.r_[local.volume_moments, np.zeros(nl + ng)]
    full = np.block([[full, gauge[:, None]], [gauge[None], np.zeros((1, 1))]])
    rhs = np.r_[local.load, np.zeros(nl), -h * d.sum(axis=0), 1.75 * mesh.volumes.sum()]
    vector = np.linalg.solve(full, rhs)
    assert_allclose(vector[:nv], result.pressure[0], atol=2e-12)
    assert_allclose(vector[nv : nv + nl], result.conormal[0], atol=2e-12)
    assert_allclose(vector[nv + nl : -1], result.trace, atol=2e-12)


def test_all_pressure_trace_nodes_prescribed_on_one_tetrahedron():
    """Recover an affine pressure when every Gamma coefficient is Dirichlet."""
    mesh = TetraMesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [0.2, 1, 0], [0.1, 0.2, 1]]),
        np.array([[0, 1, 2, 3]]),
    )

    def pressure(points):
        """Return an affine boundary datum represented by Gamma."""
        return 1 + points @ np.array([1.0, 2.0, 3.0])

    result = solve_mh2m_3d(mesh, dirichlet=pressure)
    assert len(result.free_dofs) == 0
    assert result.l2_error(pressure) < 1e-12
    assert result.flux_l2_error([-1.0, -2.0, -3.0]) < 1e-12


def test_rank_and_boundary_contracts(monkeypatch):
    """Distinguish insufficient conormal spaces from the physical constant gauge."""
    mesh = TetraMesh.unit_cube()
    natural = {int(f): 0.0 for f in mesh.boundary_faces}
    with pytest.raises(LinearSolveError, match="rank"):
        solve_mh2m_3d(
            mesh,
            neumann=natural,
            pressure_trace=PressureTraceSpace3D(mesh, 2),
            flux_space=TriangularSkeleton(mesh, degree=1),
        )
    for source in (1.0, 1e-12):
        with pytest.raises(ValueError, match="incompatible"):
            solve_mh2m_3d(mesh, source=source, neumann=natural)
    with pytest.raises(TypeError, match="tetrahedral"):
        solve_mh2m_3d(TriangleMesh.unit_square())
    invalid = [
        ({"pressure_trace": PressureTraceSpace3D(TetraMesh.unit_cube())}, "supplied"),
        ({"pressure_trace": PressureTraceSpace3D(mesh, subdivisions=4)}, "partitions"),
        ({"neumann": {-1: 0}}, "exterior"),
        ({"mean_pressure": np.nan}, "finite"),
        ({"mean_pressure": 1j}, "finite"),
        ({"mean_pressure": 1}, "pure Neumann"),
    ]
    for options, message in invalid:
        with pytest.raises(ValueError, match=message):
            solve_mh2m_3d(mesh, **options)
    monkeypatch.setattr("pymhm.mh2m3d.solve_linear", lambda matrix, rhs, **kw: np.zeros(len(rhs)))
    with pytest.raises(ValueError, match="trace equations"):
        solve_mh2m_3d(
            mesh,
            source=lambda x: 1 + x[:, 0],
            neumann={int(f): 0.25 for f in mesh.boundary_faces},
            mean_pressure=1,
        )
