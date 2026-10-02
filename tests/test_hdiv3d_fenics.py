"""Independent Basix polynomial tabulations for the tetrahedral mixed spaces."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import null_space
from threadpoolctl import threadpool_limits

from pymhm.hdiv3d_family import HDiv3DFamily, cell_quadrature

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("pressure_degree,normal_degree", [(1, 1), (2, 1), (2, 2), (3, 3)])
def test_bdfm_mass_and_divergence_against_basix_bdm(pressure_degree, normal_degree):
    """An independent complete BDM tabulation spans the selected normal-P1 bubble family."""
    basix = pytest.importorskip("basix")
    with threadpool_limits(1):
        element = basix.create_element(
            basix.ElementFamily.BDM,
            basix.CellType.tetrahedron,
            pressure_degree + 1,
            basix.LagrangeVariant.legendre,
        )
        family = HDiv3DFamily("tetrahedron", pressure_degree, normal_degree)
        points, w = cell_quadrature("tetrahedron", 5)
        native = element.tabulate(1, points)
        values, div, p = family.tabulate(points)
        X = native[0].transpose(0, 2, 1).reshape(-1, element.dim)
        Y = values.transpose(0, 2, 1).reshape(-1, family.local_size)
        change = np.linalg.lstsq(X, Y, rcond=None)[0]
        check, v = cell_quadrature("tetrahedron", 6)
        external = element.tabulate(1, check)
        expected = np.einsum("qia,ij->qja", external[0], change)
        divergence = (
            sum(
                external[basix.index(*(1 if a == b else 0 for a in range(3)))][:, :, b]
                for b in range(3)
            )
            @ change
        )
        actual, actualdiv, pressure = family.tabulate(check)
        assert_allclose(expected, actual, atol=3e-11)
        assert_allclose(divergence, actualdiv, atol=3e-10)
        tensor = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 3.0]])
        mass = np.einsum("q,qia,ab,qjb->ij", v, expected, np.linalg.inv(tensor), expected)
        own = np.einsum("q,qia,ab,qjb->ij", v, actual, np.linalg.inv(tensor), actual)
        assert_allclose(mass, own, atol=3e-10, rtol=2e-12)
        assert_allclose(
            pressure.T @ (v[:, None] * divergence),
            pressure.T @ (v[:, None] * actualdiv),
            atol=3e-11,
        )


def test_prism_tensor_family_against_independent_basix_factors():
    """Independent triangular BDM restrictions and interval polynomials produce the same27 modes."""
    basix = pytest.importorskip("basix")
    with threadpool_limits(1):
        triangle = basix.create_element(
            basix.ElementFamily.BDM, basix.CellType.triangle, 2, basix.LagrangeVariant.legendre
        )
        scalar = basix.create_element(
            basix.ElementFamily.P, basix.CellType.triangle, 1, basix.LagrangeVariant.equispaced
        )
        line = basix.create_element(
            basix.ElementFamily.P, basix.CellType.interval, 2, basix.LagrangeVariant.equispaced
        )
        t, w = np.polynomial.legendre.leggauss(5)
        s = (t + 1) / 2
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        rows = []
        for i in range(3):
            a, b = vertices[i], vertices[(i + 1) % 3]
            normal = np.array([b[1] - a[1], a[0] - b[0]])
            normal_value = triangle.tabulate(0, a + s[:, None] * (b - a))[0] @ normal
            rows.append((w * (3 * t * t - 1) / 4) @ normal_value)
        restriction = null_space(np.array(rows))
        assert restriction.shape == (12, 9)
        points, weights = cell_quadrature("prism", 5)
        xy = triangle.tabulate(1, points[:, :2])
        tri = np.einsum("qia,ij->qja", xy[0], restriction)
        dtri = (xy[1, :, :, 0] + xy[2, :, :, 1]) @ restriction
        z = points[:, 2]
        pxy = scalar.tabulate(0, points[:, :2])[0, :, :, 0]
        pz = line.tabulate(1, points[:, 2:])
        values = np.zeros((len(points), 27, 3))
        div = np.zeros((len(points), 27))
        for a in range(2):
            factor = 1 - z if a == 0 else z
            values[:, 9 * a : 9 * (a + 1), :2] = tri * factor[:, None, None]
            div[:, 9 * a : 9 * (a + 1)] = dtri * factor[:, None]
        values[:, 18:, 2] = np.einsum("qi,qj->qij", pxy, pz[0, :, :, 0]).reshape(len(points), 9)
        div[:, 18:] = np.einsum("qi,qj->qij", pxy, pz[1, :, :, 0]).reshape(len(points), 9)
        family = HDiv3DFamily("prism")
        own, owndiv, _ = family.tabulate(points)
        change = np.linalg.lstsq(
            values.transpose(0, 2, 1).reshape(-1, 27),
            own.transpose(0, 2, 1).reshape(-1, 27),
            rcond=None,
        )[0]
        assert np.linalg.matrix_rank(change) == 27
        assert_allclose(np.einsum("qia,ij->qja", values, change), own, atol=2e-11)
        assert_allclose(div @ change, owndiv, atol=2e-11)
        gram = np.einsum("q,qia,qja->ij", weights, values, values)
        assert_allclose(
            change.T @ gram @ change, np.einsum("q,qia,qja->ij", weights, own, own), atol=2e-11
        )


@pytest.mark.parametrize("degree", [2, 3])
def test_high_order_prism_against_independent_basix_product(degree):
    """Independent BDM(triangle)/P(interval) factors verify the complete higher-order operator."""
    basix = pytest.importorskip("basix")
    with threadpool_limits(1):
        triangle = basix.create_element(
            basix.ElementFamily.BDM,
            basix.CellType.triangle,
            degree + 1,
            basix.LagrangeVariant.legendre,
        )
        scalar = basix.create_element(
            basix.ElementFamily.P, basix.CellType.triangle, degree, basix.LagrangeVariant.equispaced
        )
        lines = [
            basix.create_element(
                basix.ElementFamily.P, basix.CellType.interval, k, basix.LagrangeVariant.equispaced
            )
            for k in (degree, degree + 1)
        ]

        def tabulate(points):
            """Build independent physical vector factors and their Cartesian divergence."""
            xy = triangle.tabulate(1, points[:, :2])
            pxy = scalar.tabulate(0, points[:, :2])[0, :, :, 0]
            z = [line.tabulate(1, points[:, 2:]) for line in lines]
            horizontal = np.einsum("qia,qj->qija", xy[0], z[0][0, :, :, 0]).reshape(
                len(points), -1, 2
            )
            dhorizontal = np.einsum(
                "qi,qj->qij", xy[1, :, :, 0] + xy[2, :, :, 1], z[0][0, :, :, 0]
            ).reshape(len(points), -1)
            vertical = np.einsum("qi,qj->qij", pxy, z[1][0, :, :, 0]).reshape(len(points), -1)
            dvertical = np.einsum("qi,qj->qij", pxy, z[1][1, :, :, 0]).reshape(len(points), -1)
            count = horizontal.shape[1]
            values = np.zeros((len(points), count + vertical.shape[1], 3))
            values[:, :count, :2], values[:, count:, 2] = horizontal, vertical
            return values, np.column_stack((dhorizontal, dvertical))

        family = HDiv3DFamily("prism", degree, degree)
        points, _ = cell_quadrature("prism", degree + 3)
        native, _ = tabulate(points)
        own = family.tabulate(points)[0]
        change = np.linalg.lstsq(
            native.transpose(0, 2, 1).reshape(-1, native.shape[1]),
            own.transpose(0, 2, 1).reshape(-1, family.local_size),
            rcond=None,
        )[0]
        check, weights = cell_quadrature("prism", degree + 4)
        values, divergence = tabulate(check)
        expected = np.einsum("qia,ij->qja", values, change)
        actual, div, pressure = family.tabulate(check)
        assert_allclose(expected, actual, atol=8e-11)
        assert_allclose(divergence @ change, div, atol=5e-10)
        assert_allclose(
            np.einsum("q,qia,qja->ij", weights, expected, expected),
            np.einsum("q,qia,qja->ij", weights, actual, actual),
            atol=5e-11,
        )
        assert_allclose(
            pressure.T @ (weights[:, None] * (divergence @ change)),
            pressure.T @ (weights[:, None] * div),
            atol=2e-11,
        )
