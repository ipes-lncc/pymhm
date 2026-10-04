"""Independent UFL operators for tetrahedral Taylor--Hood, USFEM and Oseen."""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.flow.forms_3d import tetra_flow_operators
from pymhm.backends.fenics import from_ufl
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space, tetrahedron_quadrature
from pymhm.meshes.tetrahedron import TetraMesh


def _monomial_inverse_constant(vertices, degree):
    """Compute the sharp Laplacian inverse constant using independent physical monomials."""
    bary, weights = tetrahedron_quadrature(degree + 2)
    points = bary[:, 1:]
    powers = np.array([a for a in product(range(degree + 1), repeat=3) if 1 <= sum(a) <= degree])
    gradient = np.zeros((len(weights), len(powers), 3))
    second = np.zeros((len(weights), len(powers), 3, 3))
    for node, power in enumerate(powers):
        for axis in range(3):
            derivative = power.copy()
            derivative[axis] -= 1
            if derivative[axis] >= 0:
                gradient[:, node, axis] = power[axis] * np.prod(points**derivative, axis=1)
            for other in range(3):
                twice = derivative.copy()
                twice[other] -= 1
                if np.all(twice >= 0):
                    second[:, node, axis, other] = (
                        power[axis] * derivative[other] * np.prod(points**twice, axis=1)
                    )
    inverse = np.linalg.inv((vertices[1:] - vertices[0]).T)
    gradient = gradient @ inverse
    hessian = np.einsum("qnic,ia,cb->qnab", second, inverse, inverse)
    laplacian = np.trace(hessian, axis1=-2, axis2=-1)
    energy = np.einsum("q,qia,qja->ij", weights, gradient, gradient)
    strong = np.einsum("q,qi,qj->ij", weights, laplacian, laplacian)
    maximum = max(0.0, linalg.eigvalsh(strong, energy)[-1])
    diameter = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1).max()
    return 1 / 3 if maximum == 0 else min(1 / 3, 1 / (diameter**2 * maximum))


@pytest.mark.fem
@pytest.mark.parametrize(
    "formulation,degree,stabilization,variable",
    [
        ("taylor-hood", 2, "tensor-2025", False),
        ("taylor-hood", 3, "tensor-2025", True),
        ("taylor-hood", 4, "tensor-2025", True),
        ("usfem", 1, "tensor-2025", True),
        ("usfem", 2, "tensor-2025", True),
        ("usfem", 3, "pointwise-2017", True),
        ("usfem", 2, "minimum-2017", True),
        ("oseen", 1, "tensor-2025", True),
        ("oseen", 2, "tensor-2025", True),
        ("oseen", 3, "tensor-2025", False),
    ],
)
def test_tetrahedral_flow_operator_against_independent_ufl(
    formulation, degree, stabilization, variable
):
    """Check every matrix/load entry with independent derivatives and stabilization constants."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    with threadpool_limits(1):
        fine = TetraMesh(
            [[0.1, 0.2, -0.1], [1.2, 0.3, 0.0], [0.2, 1.1, 0.2], [0.0, 0.1, 1.2]], [[0, 1, 2, 3]]
        )
        domain = dolfinx.mesh.create_mesh(
            MPI.COMM_SELF,
            fine.cells,
            fine.points,
            ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
        )
        pk = degree - 1 if formulation == "taylor-hood" else degree
        variant = basix.LagrangeVariant.equispaced
        space = dolfinx.fem.functionspace(
            domain,
            basix.ufl.mixed_element(
                [
                    basix.ufl.element(
                        "Lagrange", "tetrahedron", degree, shape=(3,), lagrange_variant=variant
                    ),
                    basix.ufl.element("Lagrange", "tetrahedron", pk, lagrange_variant=variant),
                ]
            ),
        )
        x = ufl.SpatialCoordinate(domain)
        nu, gamma, beta_bound = 0.003, 2.0, 4.0
        tensor = np.array([[2.0, 0.3, 0.1], [0.3, 3.0, 0.2], [0.1, 0.2, 4.0]])

        def gamma_callback(p):
            """Evaluate the analytic variable resistance."""
            return (2 + p[:, 0] ** 2 + p[:, 2])[:, None, None] * tensor

        resistance = gamma if formulation == "oseen" else gamma_callback if variable else tensor
        material = (
            gamma * ufl.Identity(3)
            if formulation == "oseen"
            else (2 + x[0] ** 2 + x[2]) * ufl.as_matrix(tensor)
            if variable
            else ufl.as_matrix(tensor)
        )

        def beta_callback(p):
            """Evaluate convection with nonzero analytic divergence."""
            return np.column_stack((0.2 + p[:, 0], 0.3 - 0.5 * p[:, 1], 0.4 + 0.25 * p[:, 2]))

        advective = formulation in ("taylor-hood", "oseen")
        beta = (
            beta_callback
            if variable and advective
            else (0.2, 0.3, 0.4)
            if advective
            else (0.0, 0.0, 0.0)
        )
        beta_ufl = (
            ufl.as_vector((0.2 + x[0], 0.3 - 0.5 * x[1], 0.4 + 0.25 * x[2]))
            if variable and advective
            else ufl.as_vector(beta)
        )
        div_beta = 0.75 if variable and advective else 0.0

        def source(p):
            """Return polynomial body force in all Cartesian components."""
            return np.column_stack((1 + p[:, 0] ** 2, -2 + p[:, 0] * p[:, 1], 0.5 + p[:, 2] ** 2))

        source_ufl = ufl.as_vector((1 + x[0] ** 2, -2 + x[0] * x[1], 0.5 + x[2] ** 2))
        minimum = 1.0 if stabilization == "minimum-2017" else None
        native = tetra_flow_operators(
            fine,
            degree=degree,
            viscosity=nu,
            drag=resistance,
            advection=beta,
            advection_divergence=div_beta,
            advection_bound=beta_bound,
            source=source,
            formulation=formulation,
            stabilization=stabilization,
            gamma_min=minimum,
            order=degree + 2,
        )
        u, p = ufl.TrialFunctions(space)
        v, q = ufl.TestFunctions(space)
        form = (
            nu * ufl.inner(ufl.grad(u), ufl.grad(v))
            + ufl.inner(material * u, v)
            - p * ufl.div(v)
            - q * ufl.div(u)
        )
        form += (
            ufl.inner(ufl.grad(u) * beta_ufl, v) - ufl.inner(u, ufl.grad(v) * beta_ufl)
        ) / 2 - div_beta * ufl.inner(u, v) / 2
        load = ufl.inner(source_ufl, v)
        bary, weights = tetrahedron_quadrature(degree + 2)
        if formulation != "taylor-hood":
            vertices = fine.points[fine.cells[0]]
            h = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1).max()
            m = _monomial_inverse_constant(vertices, degree)
            viscous = 4 * nu / m
            if formulation == "oseen":
                delta = h * h / (max(gamma * h * h, viscous) + max(viscous, beta_bound * h))
                kappa = beta_bound * h * min(1, beta_bound * h / viscous)
                form += kappa * ufl.div(u) * ufl.div(v)
            elif stabilization == "minimum-2017":
                delta = h * h / (max(minimum * h * h, viscous) + viscous)
            elif stabilization == "pointwise-2017":
                smallest = np.linalg.eigvalsh(tensor)[0] * (2 + x[0] ** 2 + x[2])
                delta = h * h / (ufl.max_value(smallest * h * h, viscous) + viscous)
            else:
                physical = bary @ vertices
                values = gamma_callback(physical) if variable else tensor[None]
                maximum = np.linalg.eigvalsh(values)[..., -1].max()
                delta = h * h / (max(maximum * h * h, viscous) + viscous)
            trial = -nu * ufl.div(ufl.grad(u)) + material * u + ufl.grad(p)
            test = -nu * ufl.div(ufl.grad(v)) + material * v + ufl.grad(q)
            if formulation == "oseen":
                trial += ufl.grad(u) * beta_ufl
                test -= ufl.grad(v) * beta_ufl
            form -= delta * ufl.inner(trial, test)
            load -= delta * ufl.inner(source_ufl, test)
        dx = ufl.Measure(
            "dx",
            domain=domain,
            metadata={
                "quadrature_rule": "custom",
                "quadrature_points": bary[:, 1:],
                "quadrature_weights": weights / 6,
            },
        )
        independent = from_ufl(form * dx, load * dx, [], np.empty(0, dtype=int))
        permutation = np.empty(len(native.load), dtype=int)
        velocity_nodes = tetra_nodal_space(fine, degree)[1]
        for component, localdegree in [(0, degree), (1, pk)]:
            collapsed, mapping = space.sub(component).collapse()
            coordinates = collapsed.tabulate_dof_coordinates()
            nodes = tetra_nodal_space(fine, localdegree)[1]
            distance = np.linalg.norm(coordinates[:, None] - nodes[None], axis=2)
            ids = distance.argmin(axis=1)
            assert distance[np.arange(len(ids)), ids].max() < 2e-14
            permutation[mapping] = (
                (3 * ids[:, None] + np.arange(3)).ravel()
                if component == 0
                else 3 * len(velocity_nodes) + ids
            )
        expected = native.matrix.toarray()[permutation][:, permutation]
        assert_allclose(independent.matrix.toarray(), expected, atol=2e-11, rtol=3e-11)
        assert_allclose(independent.load, native.load[permutation], atol=2e-12, rtol=3e-11)
