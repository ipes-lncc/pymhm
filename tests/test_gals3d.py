"""Physical patches, gauges, incompressibility and contracts for mixed 3D elasticity."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.darcy3d import TriangularSkeleton
from pymhm.elasticity3d import rigid_modes_3d
from pymhm.gals3d import solve_elasticity_gals_3d
from pymhm.gals3d_forms import tetra_elasticity_pressure_operators
from pymhm.tetrahedral import TetraMesh, tetrahedron_quadrature


@pytest.fixture(autouse=True)
def one_thread():
    """Keep small independent operator checks deterministic and inexpensive."""
    with threadpool_limits(1):
        yield


def displacement(x):
    """Affine divergence-free displacement with nonzero boundary values."""
    return np.column_stack((1 + x[:, 0], 2 - x[:, 1], x[:, 0] + x[:, 1]))


GRAD = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0]])


@pytest.mark.parametrize(
    "formulation,degree",
    [("gals", k) for k in range(1, 5)] + [("taylor-hood", k) for k in range(2, 5)],
)
def test_affine_incompressible_patch(formulation, degree):
    """All supported orders preserve nonzero affine displacement and pressure mean."""
    result = solve_elasticity_gals_3d(
        TetraMesh.unit_cube(),
        formulation=formulation,
        degree=degree,
        lame_lambda=np.inf,
        dirichlet=displacement,
        mean_pressure=2.0,
    )
    assert result.l2_error(displacement, 5) < 1e-11
    assert result.pressure_l2_error(2.0, 5) < 2e-10
    assert result.h1_seminorm_error(GRAD, 5) < 2e-11
    assert result.stress_l2_error(GRAD + GRAD.T - 2 * np.eye(3), 5) < 2e-10
    assert result.compressibility_l2(5) < 2e-11
    assert all(alpha > 0 for alpha in result.stabilization) == (formulation == "gals")


@pytest.mark.parametrize("lam", [1.0, 1e4, 1e8, np.inf])
def test_nearly_incompressible_pressure_family(lam):
    """Pressure stays O(1) as lambda increases without growing the body force."""
    inverse = 0.0 if np.isinf(lam) else 1 / lam

    def exact(x):
        """Manufactured quadratic displacement with div(u)=-p/lambda."""
        return displacement(x) - inverse * np.column_stack(
            ((x[:, 0] - 0.5) ** 2 / 2, np.zeros((len(x), 2)))
        )

    def pressure(x):
        """Linear zero-mean Herrmann pressure."""
        return x[:, 0] - 0.5

    result = solve_elasticity_gals_3d(
        TetraMesh.unit_cube(),
        degree=2,
        lame_lambda=lam,
        source=(1 + 2 * inverse, 0.0, 0.0),
        dirichlet=exact,
    )
    assert result.l2_error(exact) < 3e-11
    assert (
        result.pressure_l2_error(pressure) < 3e-7
        if lam == 1e8
        else result.pressure_l2_error(pressure) < 3e-10
    )
    assert result.compressibility_l2() < 2e-11


@pytest.mark.parametrize("formulation", ["gals", "taylor-hood"])
def test_variable_lame_coefficients(formulation):
    """Analytic variable mu/lambda retain their derivatives and weighted pressure identity."""

    def mu(x):
        """Affine positive shear."""
        return 1 + x[:, 0] + x[:, 2]

    def lam(x):
        """Affine positive first Lame modulus."""
        return 2 + x[:, 1]

    def u(x):
        """Affine displacement with divergence two."""
        return np.column_stack((x[:, 0], 2 * x[:, 1], -x[:, 2]))

    def p(x):
        """Physical pressure fixed by finite compressibility."""
        return -2 * lam(x)

    # -div(2mu epsilon(u)) + grad(p).
    result = solve_elasticity_gals_3d(
        TetraMesh.unit_cube(),
        degree=2,
        formulation=formulation,
        lame_mu=mu,
        lame_lambda=lam,
        lame_mu_gradient=(1.0, 0.0, 1.0),
        shear_bounds=(1.0, 3.0, np.sqrt(2)),
        source=(-2.0, -2.0, 2.0),
        dirichlet=u,
    )
    assert result.l2_error(u) < 2e-11
    assert result.pressure_l2_error(p) < 2e-10
    assert result.compressibility_l2() < 2e-11


@pytest.mark.parametrize("lam", [2.0, np.inf])
def test_pure_traction_rigid_and_pressure_modes(lam):
    """Six independent rigid moments are physical gauges; traction fixes pressure."""
    mesh = TetraMesh.unit_cube()
    amplitudes = np.array([1.0, 2.0, -1.0, 0.2, -0.3, 0.1])

    def exact(x):
        """Rigid displacement about the physical centroid."""
        return rigid_modes_3d(x, np.array([0.5, 0.5, 0.5])) @ amplitudes

    bary, weights = tetrahedron_quadrature(4)
    points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
    modes = rigid_modes_3d(points, np.array([0.5, 0.5, 0.5]))
    values = exact(points.reshape(-1, 3)).reshape(points.shape)
    moments = np.einsum("t,q,tqaj,tqa->j", mesh.volumes, weights, modes, values)
    result = solve_elasticity_gals_3d(
        mesh,
        degree=2,
        lame_lambda=lam,
        traction={int(f): (0.0, 0.0, 0.0) for f in mesh.boundary_faces},
        rigid_moments=moments,
    )
    assert result.l2_error(exact) < 2e-11
    assert result.pressure_l2_error(0.0) < 2e-10


def test_mixed_physical_cauchy_traction():
    """A full displacement face fixes rotations; remaining physical tractions give exact fields."""
    mesh = TetraMesh.unit_cube()
    sigma = GRAD + GRAD.T
    traction = {
        int(f): sigma @ mesh.normals[f]
        for f in mesh.boundary_faces
        if mesh.points[mesh.faces[f], 2].mean() > 0
    }
    result = solve_elasticity_gals_3d(mesh, degree=2, dirichlet=displacement, traction=traction)
    assert result.l2_error(displacement) < 2e-11
    assert result.stress_l2_error(sigma) < 2e-10


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_worker_factory_parity(backend):
    """Assembly and condensation in spawned workers preserve the complete field."""
    mesh = TetraMesh.unit_cube()
    serial = solve_elasticity_gals_3d(mesh, degree=2, dirichlet=displacement)
    parallel = solve_elasticity_gals_3d(
        mesh, degree=2, dirichlet=displacement, backend=backend, workers=2
    )
    for a, b in zip(serial.displacement, parallel.displacement, strict=True):
        assert_allclose(a, b, rtol=0, atol=3e-12)


def test_local_operator_kernel_and_stabilization_bound():
    """Six rigid vectors annihilate the operator and alpha satisfies its physical bound."""
    mesh = TetraMesh.unit_cube().submesh(0, 2)
    forms = tetra_elasticity_pressure_operators(mesh, degree=2)
    assert_allclose(forms.matrix @ forms.kernel, 0.0, atol=2e-14)
    assert forms.stabilization_alpha == forms.stabilization_bound / 2
    requested = tetra_elasticity_pressure_operators(
        mesh, degree=2, stabilization_alpha=forms.stabilization_bound / 3
    )
    assert requested.stabilization_alpha == forms.stabilization_bound / 3
    with pytest.raises(ValueError, match="strictly below"):
        tetra_elasticity_pressure_operators(
            mesh, degree=2, stabilization_alpha=forms.stabilization_bound
        )


@pytest.mark.parametrize(
    "options,message",
    [
        ({"degree": 0}, "degree"),
        ({"degree": 5}, "degree"),
        ({"formulation": "bad"}, "GaLS"),
        ({"formulation": "taylor-hood", "degree": 1}, "Taylor"),
        ({"lame_lambda": 0}, "lambda"),
        ({"lame_lambda": complex(1, 1)}, "real"),
        ({"lame_lambda": np.nan}, "lambda"),
        ({"lame_mu": 0}, "positive"),
        ({"stabilization_alpha": -1}, "positive"),
        ({"stabilization_alpha": [1]}, "scalar"),
        ({"formulation": "taylor-hood", "degree": 2, "stabilization_alpha": 0.1}, "only"),
        ({"lame_mu": lambda x: 1 + x[:, 0]}, "gradient"),
        ({"lame_mu_gradient": (1, 0, 0)}, "zero gradient"),
        ({"shear_bounds": (0, 2, 1)}, "lower"),
        ({"shear_bounds": (1, 2, -1)}, "lower"),
        ({"shear_bounds": (1, 2)}, "lower"),
        ({"shear_bounds": (2, 3, 0)}, "bound"),
        ({"macro_diameter": 0}, "diameter"),
    ],
)
def test_operator_contracts(options, message):
    """Reject invalid dimensions, coefficients and unproved stabilization data."""
    with pytest.raises(ValueError, match=message):
        tetra_elasticity_pressure_operators(TetraMesh.unit_cube(), **options)


def test_sampled_shear_contracts():
    """Bounds apply to evaluated values and derivatives, not merely input vertices."""
    mesh = TetraMesh.unit_cube()
    with pytest.raises(ValueError, match="integration"):
        tetra_elasticity_pressure_operators(
            mesh,
            lame_mu=lambda x: np.where(np.all((x > 0) & (x < 1), axis=1), -1.0, 1.0),
            lame_mu_gradient=(0, 0, 0),
            shear_bounds=(1, 1, 0),
        )
    with pytest.raises(ValueError, match="bound"):
        tetra_elasticity_pressure_operators(
            mesh,
            lame_mu=lambda x: 1 + x[:, 0],
            lame_mu_gradient=(1, 0, 0),
            shear_bounds=(1, 2, 0.5),
        )


def test_boundary_and_trace_contracts():
    """Gauge and geometric restrictions cannot silently regularize incompatible problems."""
    mesh = TetraMesh.unit_cube()
    for skeleton in [
        TriangularSkeleton(TetraMesh.unit_cube(), degree=1),
        TriangularSkeleton(mesh, degree=0),
    ]:
        with pytest.raises(ValueError, match="P1 face"):
            solve_elasticity_gals_3d(mesh, skeleton=skeleton)
    with pytest.raises(ValueError, match="subdivision"):
        solve_elasticity_gals_3d(
            mesh, local_refinement=1, skeleton=TriangularSkeleton(mesh, degree=1, subdivisions=2)
        )
    with pytest.raises(ValueError, match="scalar"):
        solve_elasticity_gals_3d(mesh, mean_pressure=[0])
    with pytest.raises(ValueError, match="full displacement"):
        solve_elasticity_gals_3d(
            mesh, traction={int(mesh.boundary_faces[0]): (0, 0, 0)}, mean_pressure=1.0
        )
    with pytest.raises(ValueError, match="finite lambda"):
        solve_elasticity_gals_3d(mesh, degree=2, mean_pressure=1.0)
    with pytest.raises(ValueError, match="incompatible"):
        solve_elasticity_gals_3d(mesh, degree=2, lame_lambda=np.inf, dirichlet=lambda x: x)
    with pytest.raises(ValueError, match="six"):
        solve_elasticity_gals_3d(
            mesh,
            degree=2,
            traction={int(f): (0, 0, 0) for f in mesh.boundary_faces},
            rigid_moments=[0] * 3,
        )
    result = solve_elasticity_gals_3d(mesh, degree=2)
    with pytest.raises(ValueError, match="outside"):
        result.evaluate(len(mesh.cells), np.ones((1, 4)) / 4)
    with pytest.raises(ValueError, match="real"):
        result.stress_l2_error(complex(1, 1))


@pytest.mark.parametrize("scale", [1e-12, 1e-20])
def test_incompressible_boundary_compatibility_below_unit_scale(scale):
    """The physical global equations reject small incompatible volume displacements."""
    with pytest.raises(ValueError, match="incompatible"):
        solve_elasticity_gals_3d(
            TetraMesh.unit_cube(),
            formulation="taylor-hood",
            degree=2,
            lame_lambda=np.inf,
            dirichlet=lambda x: scale * x,
        )
