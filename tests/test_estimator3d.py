"""Three-dimensional MHM error decomposition, physical scaling and boundary contracts."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.darcy3d import TriangularSkeleton, solve_darcy_3d
from pymhm.estimator3d import estimate_darcy_error_3d, recover_potential_3d
from pymhm.tetrahedral import TetraMesh


@pytest.fixture(autouse=True)
def one_thread():
    """Use one native thread for small finite element reference systems."""
    with threadpool_limits(1):
        yield


def exact(points):
    """Smooth tensor-product potential with homogeneous Dirichlet trace."""
    return np.prod(np.sin(np.pi * points), axis=1)


def gradient(points):
    """Independent analytical derivative in all spatial components."""
    return np.pi * np.column_stack(
        [
            np.cos(np.pi * points[:, i])
            * np.prod(np.sin(np.pi * points[:, [j for j in range(3) if j != i]]), axis=1)
            for i in range(3)
        ]
    )


@pytest.mark.parametrize("m", [0, 1, 2])
def test_estimator_terms_and_material_scaling(m):
    """Energy terms bound the measured error and scale with the square root of diffusivity."""
    mesh = TetraMesh.unit_cube()
    results = []
    for material in (1.0, 7.0):
        solution = solve_darcy_3d(
            mesh,
            degree=3,
            permeability=material,
            source=lambda x, material=material: material * 3 * np.pi**2 * exact(x),
            quadrature_order=7,
        )
        estimate = estimate_darcy_error_3d(solution, degree=m)
        assert estimate.total >= estimate.energy_error(gradient)
        assert estimate.potential.l2_error(exact) < 0.2
        assert max(estimate.equilibrium_defect) < 3e-12
        results.append(estimate)
    for term in ("flux_defect", "nonconformity", "divergence_defect", "oscillation", "total"):
        assert_allclose(
            getattr(results[1], term),
            np.sqrt(7) * getattr(results[0], term),
            atol=2e-10,
            rtol=2e-10,
        )


@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "neumann"])
def test_affine_anisotropic_potential_with_boundary_data(boundary):
    """Recovery honors only prescribed potential faces and leaves Neumann nodes unconstrained."""
    mesh = TetraMesh.unit_cube()
    tensor = np.array([[2.0, 0.2, 0.1], [0.2, 3, -0.1], [0.1, -0.1, 1]])
    exact_flux = -tensor @ np.array([1.0, 2, 3])
    natural = (
        {
            int(f): exact_flux @ mesh.normals[f]
            for f in (mesh.boundary_faces if boundary == "neumann" else mesh.boundary_faces[:1])
        }
        if boundary != "dirichlet"
        else None
    )

    def pressure(x):
        """Affine potential with physical mean four."""
        return 1 + x @ np.array([1, 2, 3])

    solution = solve_darcy_3d(
        mesh, degree=3, permeability=tensor, dirichlet=pressure, neumann=natural, mean_pressure=4
    )
    estimate = estimate_darcy_error_3d(solution, dirichlet=pressure, neumann=natural)
    assert estimate.total < 3e-12
    assert estimate.potential.l2_error(pressure) < 3e-13


def test_literal_and_energy_conventions_and_certified_callback():
    """Literal and weighted terms coincide for identity material, with explicit callback bounds."""
    solution = solve_darcy_3d(
        TetraMesh.unit_cube(), degree=3, permeability=lambda x: np.ones(len(x)), source=1
    )
    energy = estimate_darcy_error_3d(solution, ellipticity_lower_bound=1)
    literal = estimate_darcy_error_3d(solution, convention="published")
    assert_allclose(energy.local_squared, literal.local_squared, atol=1e-14)
    assert literal.ellipticity_lower_bounds.size == 0
    with pytest.raises(ValueError, match="certified"):
        estimate_darcy_error_3d(solution)
    with pytest.raises(ValueError, match="sampled"):
        estimate_darcy_error_3d(solution, ellipticity_lower_bound=1.1)


def test_incompatible_data_spaces_and_equilibrium_rejected():
    """Do not attach the energy estimate to unrepresented data or altered variational fields."""
    mesh = TetraMesh.unit_cube()
    solution = solve_darcy_3d(mesh, degree=3, source=1)
    for options, message in [
        ({"convention": "bad"}, "convention"),
        ({"ellipticity_lower_bound": 0}, "positive"),
        ({"ellipticity_lower_bound": np.nan}, "positive"),
        ({"ellipticity_lower_bound": 1j}, "scalar"),
        ({"ellipticity_lower_bound": [1, 2]}, "scalar"),
        ({"neumann": {-1: 0}}, "exterior"),
        ({"dirichlet": lambda x: np.exp(x[:, 0])}, "Dirichlet"),
        ({"neumann": {int(mesh.boundary_faces[0]): 1}}, "Neumann"),
    ]:
        with pytest.raises(ValueError, match=message):
            estimate_darcy_error_3d(solution, **options)
    bad = replace(solution, pressure=tuple(2 * p for p in solution.pressure))
    with pytest.raises(ValueError, match="equilibrium"):
        estimate_darcy_error_3d(bad)
    wrong_space = replace(solution, skeleton=TriangularSkeleton(mesh, degree=1))
    with pytest.raises(ValueError, match="spaces"):
        estimate_darcy_error_3d(wrong_space)
    with pytest.raises(ValueError, match=r"k>=ell\+3"):
        estimate_darcy_error_3d(solve_darcy_3d(mesh, degree=2, source=1))
    gap = TetraMesh(solution.local_meshes[0].points * 0.99, solution.local_meshes[0].cells)
    with pytest.raises(ValueError, match="globally conforming"):
        recover_potential_3d(replace(solution, local_meshes=(gap, *solution.local_meshes[1:])))
