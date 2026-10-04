"""AFW3D physical patches, rigid modes, incompressibility and native assembly contracts."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.elasticity.stress_3d import (
    TractionSkeleton3D,
    _Factory,
    solve_elasticity_mixed_3d,
)
from pymhm._legacy.models.elasticity.stress_forms_3d import (
    compliance_action,
    mixed_elasticity_operators_3d,
    rigid_values,
)
from pymhm.fem.hdiv.family_3d import HDiv3DFamily, cell_quadrature
from pymhm.meshes.mixed import AffineMixedMesh


@pytest.fixture(autouse=True)
def native_threads():
    """Avoid oversubscribing the small mixed systems in these numerical regressions."""
    with threadpool_limits(1):
        yield


def analytical_data(mu=1.0):
    """Load the original analytical example without changing the module search path."""
    name = "mixed_elasticity3d_data"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].SolenoidalElasticity3D(mu)


G = np.array([[0.2, 0.3, -0.1], [0.1, -0.4, 0.2], [0.3, -0.2, 0.2]])


def displacement(points):
    """Affine deformation with symmetric strain, skew rotation and translation."""
    return points @ G.T + [1, 2, 3]


def rotation():
    """Axial skew coordinates with the solver's explicitly declared orientation."""
    return np.array([G[1, 2] - G[2, 1], G[2, 0] - G[0, 2], G[0, 1] - G[1, 0]]) / 2


def rigid_integrals(mesh, function):
    """Independently integrate displacement against the six centered rigid fields."""
    xi, weights = cell_quadrature("tetrahedron", 5)
    physical = mesh.geometry(xi)
    center = mesh.volumes @ mesh.points[mesh.cells].mean(axis=1) / mesh.volumes.sum()
    return np.einsum(
        "t,q,tqa,tqak->k",
        mesh.determinants,
        weights,
        function(physical.reshape(-1, 3)).reshape(physical.shape),
        rigid_values(physical, center),
    )


@pytest.mark.parametrize("degree,refinement", [(2, 1), (2, 2), (3, 1)])
@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "traction"])
def test_affine_deformation_all_boundary_types(degree, refinement, boundary):
    """All six rigid coordinates, full traction signs and stress-row orientations are physical."""
    mesh = AffineMixedMesh.unit_cube()
    sigma = G + G.T
    natural = {}
    if boundary != "dirichlet":
        faces = mesh.boundary_faces if boundary == "traction" else mesh.boundary_faces[:4]
        natural = {int(f): sigma @ mesh.normals[f] for f in faces}
    result = solve_elasticity_mixed_3d(
        mesh,
        stress_degree=degree,
        local_refinement=refinement,
        dirichlet=displacement,
        traction=natural,
        rigid_moments=rigid_integrals(mesh, displacement),
    )
    errors = result.errors(displacement, sigma, rotation())
    assert max(errors.values()) < 3e-11
    for residual in (
        *result.fine_force_residuals(),
        *result.weak_symmetry_residuals(),
        *result.normal_traction_residuals(),
    ):
        assert np.max(abs(residual)) < 3e-11


def test_local_neumann_has_exactly_six_rigid_modes():
    """The full local stress/displacement/rotation saddle has exactly six physical null modes."""
    mesh = AffineMixedMesh.unit_cube()
    family = HDiv3DFamily("tetrahedron", 1, 2)
    problem = _Factory(TractionSkeleton3D(mesh), family, 1, 2.0, 0.7, (1, 2, 3), None, 5)(0).problem
    a = problem.matrix.toarray()
    singular = np.linalg.svd(a, compute_uv=False)
    assert np.count_nonzero(singular < 2e-11 * singular[0]) == 6
    assert_allclose(a @ problem.kernel, 0, atol=2e-14)
    assert np.linalg.matrix_rank(problem.constraints.T @ problem.kernel) == 6


def test_incompressible_pressure_gauge_and_rigid_rotation():
    """Mean hydrostatic pressure and all three rigid rotations obey separate physical roles."""
    mesh = AffineMixedMesh.unit_cube()
    omega = np.array([0.2, -0.3, 0.4])

    def exact(points):
        """Rigid motion with zero symmetric strain and exactly zero boundary volume flux."""
        return np.cross(omega, points - 0.5) + [1, 2, 3]

    result = solve_elasticity_mixed_3d(
        mesh, lame_lambda=np.inf, mean_pressure=2.3, dirichlet=exact, local_refinement=1
    )
    assert max(result.errors(exact, -2.3 * np.eye(3), -omega).values()) < 3e-11
    with pytest.raises(ValueError, match="incompatible incompressible"):
        solve_elasticity_mixed_3d(
            mesh, lame_lambda=np.inf, dirichlet=lambda x: x, local_refinement=1
        )


def test_quadratic_displacement_nonzero_source():
    """BDM3/P2/P2 reproduces an inhomogeneous quadratic deformation and affine stress."""
    mesh = AffineMixedMesh.unit_cube()

    def exact(points):
        """Quadratic vector displacement with nonzero volume force."""
        return points**2

    def stress(points):
        """Cauchy stress for lambda2 and mu0.7."""
        result = np.zeros((len(points), 3, 3))
        for i in range(3):
            result[:, i, i] = 2.8 * points[:, i] + 4 * points.sum(axis=1)
        return result

    solution = solve_elasticity_mixed_3d(
        mesh,
        stress_degree=3,
        local_refinement=1,
        lame_lambda=2,
        lame_mu=0.7,
        source=(-6.8,) * 3,
        dirichlet=exact,
    )
    assert max(solution.errors(exact, stress, (0, 0, 0)).values()) < 3e-11


def test_anisotropic_full_compliance_affine_patch():
    """A self-adjoint full tensor law, including a skew extension, matches the exact deformation."""
    mesh = AffineMixedMesh.unit_cube()
    material = np.eye(9) * 0.4
    material[np.ix_([0, 4, 8], [0, 4, 8])] = [
        [0.3, -0.04, 0.01],
        [-0.04, 0.4, -0.02],
        [0.01, -0.02, 0.5],
    ]
    strain = (G + G.T) / 2
    sigma = np.linalg.solve(material, strain.ravel()).reshape(3, 3)
    solution = solve_elasticity_mixed_3d(
        mesh, compliance=material, dirichlet=displacement, local_refinement=1
    )
    assert max(solution.errors(displacement, sigma, rotation()).values()) < 3e-11


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_factory_backend_parity(backend):
    """Spawn-safe local assembly and condensation reproduce the same physical field."""
    result = solve_elasticity_mixed_3d(
        AffineMixedMesh.unit_cube(),
        dirichlet=displacement,
        local_refinement=1,
        backend=backend,
        workers=2,
    )
    assert max(result.errors(displacement, G + G.T, rotation()).values()) < 3e-11


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"stress_degree": 1}, "stress degree"),
        ({"trace_degree": 3}, "trace degree"),
        ({"subdivisions": 3}, "subdivisions"),
        ({"mean_pressure": np.nan}, "finite"),
        ({"mean_pressure": 1}, "incompressible limit"),
        ({"traction": {0: (0, 0, 0)}, "mean_pressure": 1}, "full displacement"),
        ({"traction": {999: (0, 0, 0)}}, "exterior"),
        ({"lame_lambda": -1}, "Lamé"),
        ({"lame_mu": 0}, "Lamé"),
        ({"lame_lambda": 1j}, "real"),
        ({"lame_lambda": np.nan}, "Lamé"),
    ],
)
def test_invalid_solver_contracts(kwargs, match):
    """Reject incompatible finite-element spaces, material laws and gauge requests explicitly."""
    with pytest.raises(ValueError, match=match):
        solve_elasticity_mixed_3d(AffineMixedMesh.unit_cube(), local_refinement=1, **kwargs)


@pytest.mark.parametrize("moments", [(1, 2), np.full(6, np.nan), np.ones(6) * 1j])
def test_invalid_rigid_moments(moments):
    """Only six finite real physical integrals can fix a pure-traction displacement."""
    mesh = AffineMixedMesh.unit_cube()
    with pytest.raises(ValueError, match="six finite"):
        solve_elasticity_mixed_3d(
            mesh,
            traction={int(f): (0, 0, 0) for f in mesh.boundary_faces},
            rigid_moments=moments,
            local_refinement=1,
        )


@pytest.mark.parametrize(
    "material,match",
    [
        (np.eye(9) * 1j, "real"),
        (np.eye(6), "shape"),
        (np.full((9, 9), np.nan), "finite"),
        (-np.eye(9), "positive"),
        (np.eye(9) + np.eye(9, k=1), "self-adjoint"),
        (np.diag([1, 2, 1, 3, 1, 1, 1, 1, 1]), "symmetric and skew"),
    ],
)
def test_invalid_full_compliance(material, match):
    """A weak-symmetry law must define a positive self-adjoint extension on all tensors."""
    with pytest.raises(ValueError, match=match):
        compliance_action(np.zeros((1, 1, 1, 3, 3)), np.zeros((1, 3)), 1, 1, material)


def test_geometry_family_and_error_contracts():
    """Reject non-tetrahedral AFW geometry, truncated stress families and invalid exact fields."""
    prism = AffineMixedMesh.unit_cube(kind="prism")
    with pytest.raises(ValueError, match="tetrahedral"):
        TractionSkeleton3D(prism)
    with pytest.raises(ValueError, match="complete tetrahedral"):
        mixed_elasticity_operators_3d(AffineMixedMesh.unit_cube(), HDiv3DFamily())
    result = solve_elasticity_mixed_3d(AffineMixedMesh.unit_cube(), local_refinement=1)
    for bad in [np.eye(3) * 1j, np.eye(3) * np.nan]:
        with pytest.raises(ValueError, match="real and finite"):
            result.errors((0, 0, 0), bad, (0, 0, 0))


@pytest.mark.parametrize("boundary", ["dirichlet", "mixed"])
def test_variable_lame_affine_patch(boundary):
    """Spatial Lamé fields use the full compliance, body force and physical variable traction."""
    mesh = AffineMixedMesh.unit_cube()
    strain_twice = G + G.T

    def stress(points):
        """Affine Cauchy stress with independently differentiated constant body force."""
        return (1 + points[:, 0, None, None]) * strain_twice

    natural = {}
    if boundary == "mixed":
        for f in mesh.boundary_faces[:4]:
            normal = mesh.normals[f].copy()
            natural[int(f)] = lambda x, normal=normal: stress(x) @ normal
    result = solve_elasticity_mixed_3d(
        mesh,
        local_refinement=1,
        lame_mu=lambda x: 1 + x[:, 0],
        lame_lambda=lambda x: 2 + x[:, 1],
        source=-strain_twice[:, 0],
        dirichlet=displacement,
        traction=natural,
        quadrature_order=12,
    )
    assert max(result.errors(displacement, stress, rotation()).values()) < 2e-11


def test_incompatible_force_is_not_removed_by_rigid_gauges():
    """Six displacement gauges cannot make an unbalanced external force admissible."""
    mesh = AffineMixedMesh.unit_cube()
    with pytest.raises(ValueError, match="incompatible|compatibility"):
        solve_elasticity_mixed_3d(
            mesh,
            local_refinement=1,
            source=(1, 0, 0),
            traction={int(f): (0, 0, 0) for f in mesh.boundary_faces},
        )


def test_manufactured_force_independent_complex_step():
    """The analytical load equals minus stress divergence and displacement has zero divergence."""
    data = analytical_data(0.7)
    points = np.random.default_rng(913).uniform(0.05, 0.95, (21, 3))
    derivative, divergence = np.zeros((len(points), 3, 3)), np.zeros_like(points)
    for axis in range(3):
        shifted = points.astype(complex)
        shifted[:, axis] += 1e-30j
        derivative[:, :, axis] = data.displacement(shifted).imag / 1e-30
        divergence += data.stress(shifted).imag[:, :, axis] / 1e-30
    assert_allclose(derivative, data.fields(points)[1], atol=3e-14, rtol=3e-14)
    assert_allclose(np.trace(derivative, axis1=-2, axis2=-1), 0, atol=3e-14)
    assert_allclose(-divergence, data.source(points), atol=3e-12, rtol=3e-14)


def test_nonaffine_bulk_modulus_limit():
    """Finite bulk moduli approach the incompressible solution without amplifying a fixed load."""
    mesh = AffineMixedMesh.unit_cube()
    data = analytical_data()
    solutions = []
    for lam in (1.0, 1e4, 1e8, np.inf):
        result = solve_elasticity_mixed_3d(
            mesh, lame_lambda=lam, source=data.source, local_refinement=2, quadrature_order=8
        )
        assert (
            max(result.errors(data.displacement, data.stress, data.rotation, order=10).values())
            < 100
        )
        solutions.append(result)
    last = solutions[-1]
    for field_name in ("stress", "displacement", "rotation"):
        reference = np.concatenate([a.ravel() for a in getattr(last, field_name)])
        errors = [
            np.linalg.norm(np.concatenate([a.ravel() for a in getattr(s, field_name)]) - reference)
            for s in solutions[:-1]
        ]
        assert errors[2] < 2e-4 * errors[1] < 2e-3 * errors[0]


@pytest.mark.parametrize("lam", [2.0, np.inf])
def test_nonaffine_complementary_energy_identity(lam):
    """Stress energy equals body-force work with homogeneous displacement and weak symmetry."""
    from pymhm.fem.hdiv.family_3d import cell_quadrature

    def force(points):
        """A nonconservative affine body force with exact polynomial work quadrature."""
        return points[:, [1, 2, 0]]

    solution = solve_elasticity_mixed_3d(
        AffineMixedMesh.unit_cube(),
        lame_lambda=lam,
        source=force,
        quadrature_order=5,
    )
    points, weights = cell_quadrature("tetrahedron", 5)
    energy, work = 0.0, 0.0
    for cell, fine in enumerate(solution.local_meshes):
        displacement, stress, _, _ = solution.evaluate(cell, points)
        trace = np.trace(stress, axis1=-2, axis2=-1)
        deviator = stress - trace[..., None, None] * np.eye(3) / 3
        density = np.sum(deviator**2, axis=(-2, -1)) / 2
        density += trace**2 / (3 * (2 + 3 * lam))
        body = force(fine.geometry(points).reshape(-1, 3)).reshape(displacement.shape)
        energy += float(fine.determinants @ (density @ weights))
        work += float(fine.determinants @ (np.sum(body * displacement, axis=-1) @ weights))
    assert_allclose(energy, work, rtol=2e-11, atol=0)
