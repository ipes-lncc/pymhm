"""Analytical forcing, cavity traces and nonzero physical norms in the L14 examples."""

import numpy as np
from numpy.testing import assert_allclose

from examples.solve_spe10_taylor_hood import TaylorHoodField
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.meshes.triangle import TriangleMesh


def test_solenoidal_data_and_smooth_lid():
    """Verify the manufactured divergence and the explicitly compatible cavity corners."""
    from examples.solve_stokes_adaptive import StokesData

    points = np.random.default_rng(16).uniform(0.1, 0.9, (17, 2))
    data = StokesData(viscosity=0.01)
    u, grad, lap = data.fields(points)
    assert_allclose(np.trace(grad, axis1=1, axis2=2), 0, atol=0)
    step = 1e-5
    for axis in range(2):
        shift = np.eye(2)[axis] * step
        finite = (data.velocity(points + shift) - data.velocity(points - shift)) / (2 * step)
        assert_allclose(finite, grad[:, :, axis], atol=2e-8)
    assert_allclose(data.source(points), -0.01 * lap + 150 * (points[:, ::-1] - 0.5))
    cavity = StokesData(cavity=True)
    boundary = np.array([[0, 1], [0.5, 1], [1, 1], [0.5, 0], [0, 0.4], [1, 0.4]])
    assert_allclose(cavity.boundary(boundary), [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    assert_allclose(cavity.source(points), 0)
    constant = StokesData(cavity=True, lid="constant")
    assert_allclose(constant.boundary(boundary), [[1, 0], [1, 0], [1, 0], [0, 0], [0, 0], [0, 0]])
    assert_allclose(data.options()["dirichlet"](points), u)


def test_cavity_comparison_norms_have_correct_area_and_denominators():
    """Affine vector and pressure fields provide independent nonzero L2/H1 integrals."""
    from examples.compare_stokes_cavity import norms

    n = 2
    y, x = np.meshgrid(
        np.arange(2 * n + 1) / (2 * n), np.arange(2 * n + 1) / (2 * n), indexing="ij"
    )
    u = np.stack((x + 2 * y, 3 * x - y), axis=-1)
    y, x = np.meshgrid(np.arange(n + 1) / n, np.arange(n + 1) / n, indexing="ij")
    reference = TaylorHoodField(u, 1 + x - 2 * y, (0.0, 1.0, 0.0, 1.0))
    mesh = TriangleMesh.unit_square()
    arrays = dict(
        macro_cells=np.array([[0, 1, 2]]),
        velocity_degree=np.array(2),
        pressure_degree=np.array(1),
        points_0=mesh.points,
        cells_0=mesh.cells,
        velocity_0=np.zeros((len(nodal_space(mesh, 2)[1]), 2)),
        pressure_0=np.zeros(len(nodal_space(mesh, 1)[1])),
    )
    measured = norms(arrays, reference, 4, drag=3.0)
    assert_allclose(measured["velocity_l2"], np.sqrt(4.5), rtol=2e-14)
    assert_allclose(measured["pressure_l2"], np.sqrt(2 / 3), rtol=2e-14)
    assert_allclose(measured["velocity_h1_seminorm"], np.sqrt(15), rtol=2e-14)
    assert_allclose(measured["velocity_energy"], np.sqrt(15 + 3 * 4.5), rtol=2e-14)
    assert_allclose(measured["velocity_energy_relative"], 1)
    assert_allclose(
        [
            measured[k]
            for k in ("velocity_relative", "pressure_relative", "velocity_h1_seminorm_relative")
        ],
        1,
    )
    cutout = 1 / 8
    domains = ((0, 1, 0, 1 - cutout), (cutout, 1 - cutout, 1 - cutout, 1))

    def moment(i, j):
        """Integrate a monomial analytically over the two retained rectangles."""
        return sum(
            (right ** (i + 1) - left ** (i + 1))
            * (top ** (j + 1) - bottom ** (j + 1))
            / ((i + 1) * (j + 1))
            for left, right, bottom, top in domains
        )

    clipped = norms(arrays, reference, 4, drag=3.0, corner_cutout=cutout)
    u2 = 10 * moment(2, 0) + 5 * moment(0, 2) - 2 * moment(1, 1)
    p2 = (
        moment(0, 0)
        + moment(2, 0)
        + 4 * moment(0, 2)
        + 2 * moment(1, 0)
        - 4 * moment(0, 1)
        - 4 * moment(1, 1)
    )
    assert_allclose(clipped["velocity_l2"] ** 2, u2, rtol=2e-14)
    assert_allclose(clipped["pressure_l2"] ** 2, p2, rtol=2e-14)
    assert_allclose(clipped["velocity_h1_seminorm"] ** 2, 15 * (1 - 2 * cutout**2))
    assert_allclose(clipped["velocity_energy"] ** 2, 15 * moment(0, 0) + 3 * u2)


def test_cavity_profiles_preserve_polynomials_and_macro_intersections():
    """Profiles evaluate each fine-cell polynomial separately, with physical macro crossings."""
    from examples.compare_stokes_cavity import profiles

    mesh = TriangleMesh.unit_square()
    axis = np.linspace(0, 1, 5)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    reference = TaylorHoodField(
        np.stack((x * x + y, x - y * y), axis=-1),
        1 + x[::2, ::2] - 2 * y[::2, ::2],
        (0.0, 1.0, 0.0, 1.0),
    )
    arrays = dict(
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        velocity_degree=np.array(2),
        pressure_degree=np.array(1),
    )
    for cell in range(len(mesh.cells)):
        fine = mesh.submesh(cell, 2)
        unodes, pnodes = nodal_space(fine, 2)[1], nodal_space(fine, 1)[1]
        arrays.update(
            {
                f"points_{cell}": fine.points,
                f"cells_{cell}": fine.cells,
                f"velocity_{cell}": np.column_stack(
                    (
                        unodes[:, 0] ** 2 + unodes[:, 1],
                        unodes[:, 0] - unodes[:, 1] ** 2,
                    )
                ),
                f"pressure_{cell}": 1 + pnodes[:, 0] - 2 * pnodes[:, 1],
            }
        )
    result = profiles(arrays, reference)
    for label in ("upper", "vertical"):
        for field in ("velocity", "pressure"):
            assert_allclose(
                result[f"profile_{label}_{field}"],
                result[f"profile_{label}_reference_{field}"],
                atol=1e-15,
            )
    assert_allclose(result["profile_vertical_macro_intersections"], [0, 0.5, 1])
    assert_allclose(result["profile_upper_macro_intersections"], [0, 0.99, 1])
