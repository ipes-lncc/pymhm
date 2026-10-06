"""Independent native UFL verification of real and imaginary acoustic forms."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm._legacy.models.waves.helmholtz import solve_helmholtz
from pymhm.fem.scalar.helmholtz import acoustic_space, complex_vector
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.triangle import TriangleMesh

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("rectangle", [False, True])
@pytest.mark.parametrize("pml", [False, True])
def test_acoustic_local_operators_match_native_ufl(degree, rectangle, pml):
    """Compare variable diffusion/mass, imaginary impedance and both load components."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix
    import basix.ufl
    import ufl
    from mpi4py import MPI

    coarse = (
        CartesianMacroMesh(1)
        if rectangle
        else TriangleMesh(np.array([[0.1, 0.2], [1.2, 0.1], [0.3, 0.9]]), np.array([[0, 1, 2]]))
    )
    result = solve_helmholtz(
        coarse,
        omega=2.3,
        degree=degree,
        local_refinement=3,
        density=lambda p: (2 + p[:, 0]) ** -2,
        bulk_modulus=lambda p: (2 + p[:, 0]) ** -2,
        source=lambda p: 1 + p[:, 1] ** 2 + 1j * (2 - p[:, 0]),
        absorbing=None if pml else lambda p, n: p[:, 0] + 1j * p[:, 1],
        pml_stretch=(1 + 0.3j, 1 + 0.6j) if pml else None,
        quadrature_order=10,
    )
    fine = result.local_meshes[0]
    kind = "quadrilateral" if rectangle else "triangle"
    coordinate = ufl.Mesh(basix.ufl.element("Lagrange", kind, 1, shape=(2,)))
    cells = fine.cells[:, [0, 1, 3, 2]] if rectangle else fine.cells
    domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, cells, x=fine.points, e=coordinate)
    space = dolfinx.fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange",
            kind,
            degree,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        ),
    )
    _, points = acoustic_space(fine, degree)
    native_points = space.tabulate_dof_coordinates()[:, :2]
    permutation = np.argmin(np.linalg.norm(points[:, None] - native_points[None], axis=2), axis=1)
    assert_allclose(native_points[permutation], points, atol=1e-14)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(domain)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 20})
    ds = ufl.Measure("ds", domain=domain, metadata={"quadrature_degree": 20})

    def matrix(form):
        """Assemble a native scalar form and permute only physical nodal coordinates."""
        assembled = dolfinx.fem.assemble_matrix(dolfinx.fem.form(form))
        assembled.scatter_reverse()
        return assembled.to_scipy().toarray()[np.ix_(permutation, permutation)]

    stretch = np.array([1 + 0.3j, 1 + 0.6j]) if pml else np.ones(2)
    diagonal = stretch[::-1] / stretch
    mass = np.prod(stretch)
    real = matrix(
        (2 + x[0]) ** 2
        * (
            sum(float(diagonal[j].real) * u.dx(j) * v.dx(j) for j in range(2))
            - 2.3**2 * float(mass.real) * u * v
        )
        * dx
    )
    imaginary = matrix(
        (2 + x[0]) ** 2
        * (
            sum(float(diagonal[j].imag) * u.dx(j) * v.dx(j) for j in range(2))
            - 2.3**2 * float(mass.imag) * u * v
        )
        * dx
        - (0 if pml else 2.3) * (2 + x[0]) ** 2 * u * v * ds
    )
    load_real = dolfinx.fem.assemble_vector(
        dolfinx.fem.form((1 + x[1] ** 2) * v * dx + (0 if pml else 1) * x[0] * v * ds)
    ).array[permutation]
    load_imag = dolfinx.fem.assemble_vector(
        dolfinx.fem.form((2 - x[0]) * v * dx + (0 if pml else 1) * x[1] * v * ds)
    ).array[permutation]
    response = result.system.responses[0]
    actual = response.problem.matrix.toarray()
    assert_allclose(actual[::2, ::2], real, atol=2e-12)
    assert_allclose(actual[1::2, ::2], imaginary, atol=2e-12)
    assert_allclose(actual[::2, 1::2], -imaginary, atol=2e-12)
    assert_allclose(complex_vector(response.problem.load), load_real + 1j * load_imag, atol=3e-14)
    expected = np.linalg.solve(real + 1j * imaginary, load_real + 1j * load_imag)
    assert_allclose(complex_vector(response.source), expected, atol=2e-12, rtol=1e-11)
