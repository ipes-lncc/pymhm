"""Independent UFL mixed elasticity operators with full variable-shear residuals."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.elasticity.pressure_forms_3d import tetra_elasticity_pressure_operators
from pymhm.backends.fenics import from_ufl
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space, tetrahedron_quadrature
from pymhm.meshes.tetrahedron import TetraMesh


@pytest.mark.fem
@pytest.mark.parametrize(
    "formulation,degree,variable",
    [
        ("gals", 1, False),
        ("gals", 1, True),
        ("gals", 2, True),
        ("gals", 3, True),
        ("gals", 4, False),
        ("taylor-hood", 2, True),
        ("taylor-hood", 3, False),
        ("taylor-hood", 4, True),
    ],
)
def test_gals_tetra_operator_against_ufl(formulation, degree, variable):
    """Compare every entry using UFL strain/divergence derivatives and explicit alpha."""
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
            x=fine.points,
            e=ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
        )
        pdegree = degree if formulation == "gals" else degree - 1
        space = dolfinx.fem.functionspace(
            domain,
            basix.ufl.mixed_element(
                [
                    basix.ufl.element(
                        "Lagrange",
                        "tetrahedron",
                        degree,
                        shape=(3,),
                        lagrange_variant=basix.LagrangeVariant.equispaced,
                    ),
                    basix.ufl.element(
                        "Lagrange",
                        "tetrahedron",
                        pdegree,
                        lagrange_variant=basix.LagrangeVariant.equispaced,
                    ),
                ]
            ),
        )
        x = ufl.SpatialCoordinate(domain)

        def mu(points):
            """Strictly positive affine shear field."""
            return 2 + 0.2 * points[:, 0] + 0.1 * points[:, 2]

        def lam(points):
            """Variable compressibility independent of shear."""
            return 3 + points[:, 1]

        def source(points):
            """Independent polynomial vector load."""
            return np.column_stack(
                (1 + points[:, 0] ** 2, points[:, 0] * points[:, 1] - 2, 0.5 + points[:, 2])
            )

        alpha = 1e-5 if formulation == "gals" else None
        native = tetra_elasticity_pressure_operators(
            fine,
            degree=degree,
            formulation=formulation,
            lame_mu=mu if variable else 2.0,
            lame_lambda=lam if variable else np.inf,
            lame_mu_gradient=(0.2, 0, 0.1) if variable else None,
            shear_bounds=(1.9, 2.5, 0.23) if variable else None,
            source=source,
            stabilization_alpha=alpha,
            order=degree + 3,
        )
        u, p = ufl.TrialFunctions(space)
        v, q = ufl.TestFunctions(space)
        shear = 2 + 0.2 * x[0] + 0.1 * x[2] if variable else 2.0
        inverse = 1 / (3 + x[1]) if variable else 0.0
        sigma = 2 * shear * ufl.sym(ufl.grad(u)) - p * ufl.Identity(3)
        test_sigma = 2 * shear * ufl.sym(ufl.grad(v)) - q * ufl.Identity(3)
        force = ufl.as_vector((1 + x[0] ** 2, x[0] * x[1] - 2, 0.5 + x[2]))
        form = (
            2 * shear * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v)))
            - p * ufl.div(v)
            - q * ufl.div(u)
            - inverse * p * q
        )
        load = ufl.inner(force, v)
        if formulation == "gals":
            vertices = fine.points[fine.cells[0]]
            h = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1).max()
            form -= alpha * h * h * ufl.inner(ufl.div(sigma), ufl.div(test_sigma))
            load += alpha * h * h * ufl.inner(force, ufl.div(test_sigma))
        bary, weights = tetrahedron_quadrature(degree + 3)
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
        nv = len(native.displacement_nodes)
        for component, k in [(0, degree), (1, pdegree)]:
            collapsed, mapping = space.sub(component).collapse()
            coordinates = collapsed.tabulate_dof_coordinates()
            nodes = tetra_nodal_space(fine, k)[1]
            distances = np.linalg.norm(coordinates[:, None] - nodes[None], axis=-1)
            ids = distances.argmin(axis=1)
            assert distances[np.arange(len(ids)), ids].max() < 2e-14
            permutation[mapping] = (
                (3 * ids[:, None] + np.arange(3)).ravel() if component == 0 else 3 * nv + ids
            )
        assert_allclose(
            independent.matrix.toarray(),
            native.matrix.toarray()[permutation][:, permutation],
            atol=3e-11,
            rtol=3e-11,
        )
        assert_allclose(independent.load, native.load[permutation], atol=3e-12, rtol=3e-11)
