"""Independent conforming UFL saddle with polynomial traction restrictions on polygons."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm.elements import triangle_quadrature
from pymhm.fenics import from_ufl
from pymhm.polygon import solve_elasticity_mixed_polygons


def native_mumps_solve(matrix: sparse.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Solve the independent saddle through native PETSc/MUMPS and release all handles."""
    from petsc4py import PETSc

    csr = matrix.tocsr()
    operator = PETSc.Mat().createAIJ(
        size=csr.shape,
        csr=(csr.indptr.astype(PETSc.IntType), csr.indices.astype(PETSc.IntType), csr.data),
        comm=PETSc.COMM_SELF,
    )
    source, result = operator.createVecs()
    solver = PETSc.KSP().create(PETSc.COMM_SELF)
    try:
        source.array[:] = rhs
        solver.setOperators(operator)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.getPC().setFactorSolverType("mumps")
        solver.solve(source, result)
        assert solver.getConvergedReason() > 0
        return result.array.copy()
    finally:
        solver.destroy()
        source.destroy()
        result.destroy()
        operator.destroy()


def compare_polygon_problem(n: int) -> dict[str, Any]:
    """Assemble a native global AFW saddle and restrict only interior quadratic tractions."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    from examples import core_extension_data as data
    from examples.solve_core_extensions import polygon_grid

    macro = polygon_grid(n)
    native = solve_elasticity_mixed_polygons(
        macro,
        compliance=data.compliance(),
        source=data.force,
        dirichlet=data.displacement,
        local_refinement=1,
        quadrature_order=8,
    )
    all_points = np.vstack([mesh.points for mesh in native.local_meshes])
    points, inverse = np.unique(all_points, axis=0, return_inverse=True)
    cells, offset = [], 0
    for mesh in native.local_meshes:
        cells.extend(inverse[offset + mesh.cells])
        offset += len(mesh.points)
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        np.array(cells),
        points,
        ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))),
    )
    bdm = basix.ufl.element("BDM", "triangle", 2)
    scalar = basix.ufl.element(
        "DG", "triangle", 1, lagrange_variant=basix.LagrangeVariant.equispaced
    )
    vector = basix.ufl.element(
        "DG", "triangle", 1, shape=(2,), lagrange_variant=basix.LagrangeVariant.equispaced
    )
    space = dolfinx.fem.functionspace(domain, basix.ufl.mixed_element([bdm, bdm, vector, scalar]))
    first, second, displacement, rotation = ufl.TrialFunctions(space)
    test_first, test_second, test_displacement, test_rotation = ufl.TestFunctions(space)
    sigma, tau = ufl.as_tensor((first, second)), ufl.as_tensor((test_first, test_second))
    stress_flat = ufl.as_vector((sigma[0, 0], sigma[0, 1], sigma[1, 0], sigma[1, 1]))
    test_flat = ufl.as_vector((tau[0, 0], tau[0, 1], tau[1, 0], tau[1, 1]))
    compliance = ufl.as_matrix(data.compliance())
    x, normal = ufl.SpatialCoordinate(domain), ufl.FacetNormal(domain)
    prescribed = ufl.as_vector((x[0] ** 2 * x[1] ** 2, x[0] ** 3 * x[1]))
    strain = ufl.sym(ufl.grad(prescribed))
    strain_flat = ufl.as_vector((strain[0, 0], strain[0, 1], strain[1, 0], strain[1, 1]))
    exact_stress_flat = ufl.as_matrix(np.linalg.inv(data.compliance())) * strain_flat
    exact_stress = ufl.as_tensor(
        ((exact_stress_flat[0], exact_stress_flat[1]), (exact_stress_flat[2], exact_stress_flat[3]))
    )
    force = -ufl.div(exact_stress)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 12})
    ds = ufl.Measure("ds", domain=domain, metadata={"quadrature_degree": 12})
    a = (
        ufl.dot(compliance * stress_flat, test_flat)
        + ufl.dot(displacement, ufl.div(tau))
        + ufl.dot(ufl.div(sigma), test_displacement)
        + rotation * (tau[0, 1] - tau[1, 0])
        + test_rotation * (sigma[0, 1] - sigma[1, 0])
    ) * dx
    load = -ufl.dot(force, test_displacement) * dx + ufl.dot(prescribed, tau * normal) * ds
    problem = from_ufl(a, load, [], np.empty(0, dtype=int))

    # All local polygon edges have one fine facet at refinement one. The single
    # quadratic normal moment on each interior edge vanishes precisely for P1
    # tractions. Exterior tractions retain all BDM2 moments and their nonzero g.
    entities, labels, edge_polynomials = [], [], []
    interior = np.flatnonzero(macro.face_cells[:, 1] >= 0)
    for tag, face in enumerate(interior, start=1):
        start, end = macro.points[macro.faces[face]]
        tangent = end - start

        def on_edge(coordinates, start=start, tangent=tangent):
            """Select the unique native facet on the original physical polygon edge."""
            relative = coordinates[:2].T - start
            parameter = relative @ tangent / (tangent @ tangent)
            return (
                (abs(relative[:, 0] * tangent[1] - relative[:, 1] * tangent[0]) < 1e-12)
                & (parameter >= -1e-12)
                & (parameter <= 1 + 1e-12)
            )

        facets = dolfinx.mesh.locate_entities(domain, 1, on_edge)
        assert len(facets) == 1
        entities.extend(facets)
        labels.extend([tag])
        coordinate = (
            2 * ufl.dot(x - ufl.as_vector(start), ufl.as_vector(tangent)) / float(tangent @ tangent)
            - 1
        )
        edge_polynomials.append((3 * coordinate**2 - 1) / 2)
    order = np.argsort(entities)
    tags = dolfinx.mesh.meshtags(
        domain, 1, np.asarray(entities, np.int32)[order], np.asarray(labels, np.int32)[order]
    )
    dS = ufl.Measure("dS", domain=domain, subdomain_data=tags, metadata={"quadrature_degree": 12})
    restrictions = []
    for tag, polynomial in enumerate(edge_polynomials, start=1):
        traction = tau("+") * normal("+")
        for component in range(2):
            form = traction[component] * polynomial("+") * dS(tag)
            vector = dolfinx.fem.assemble_vector(dolfinx.fem.form(form))
            vector.scatter_reverse(dolfinx.la.InsertMode.add)
            restrictions.append(vector.array.copy())
    restriction = sparse.csc_matrix(np.column_stack(restrictions))
    complete = sparse.bmat([[problem.matrix, restriction], [restriction.T, None]], format="csc")
    rhs = np.r_[problem.load, np.zeros(restriction.shape[1])]
    coefficients = native_mumps_solve(complete, rhs)
    field = dolfinx.fem.Function(space)
    field.x.array[:] = coefficients[: problem.matrix.shape[0]]
    components = [part.collapse() for part in field.split()]
    original_to_local = np.argsort(domain.topology.original_cell_index)
    bary, weights = triangle_quadrature(6)
    differences, denominators = np.zeros(3), np.zeros(3)
    maximum = np.zeros(3)
    offset = 0
    for mesh, stress, displacement, rotation in zip(
        native.local_meshes, native.stress, native.displacement, native.rotation, strict=True
    ):
        coordinates = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 2)
        physical = np.column_stack((coordinates, np.zeros(len(coordinates))))
        indices = np.repeat(original_to_local[offset : offset + len(mesh.cells)], len(bary)).astype(
            np.int32
        )
        evaluated = [part.eval(physical, indices) for part in components]
        reference = [np.stack(evaluated[:2], axis=1), evaluated[2], evaluated[3].ravel()]
        values, _ = native.family.evaluate(mesh, stress, bary)
        computed = [
            values.reshape(-1, 2, 2),
            np.einsum("qi,tia->tqa", bary, displacement).reshape(-1, 2),
            (rotation @ bary.T).ravel(),
        ]
        factors = (mesh.areas[:, None] * weights).ravel()
        for index, (candidate, target) in enumerate(zip(computed, reference, strict=True)):
            delta = (candidate - target).reshape(len(factors), -1)
            differences[index] += factors @ np.sum(delta**2, axis=1)
            denominators[index] += factors @ np.sum(target.reshape(len(factors), -1) ** 2, axis=1)
            maximum[index] = max(maximum[index], np.max(abs(delta)))
        offset += len(mesh.cells)
    return {
        "n": n,
        "macrocells": len(macro.cells),
        "fine_triangles": len(cells),
        "native_unknowns": problem.matrix.shape[0],
        "traction_constraints": restriction.shape[1],
        "relative_l2_stress_displacement_rotation": np.sqrt(differences / denominators).tolist(),
        "absolute_l2_stress_displacement_rotation": np.sqrt(differences).tolist(),
        "maximum_components_stress_displacement_rotation": maximum.tolist(),
        "quadratic_traction_moment_linf": float(np.max(abs(restriction.T @ field.x.array))),
        "full_saddle_relative_residual": float(
            np.linalg.norm(complete @ coefficients - rhs) / np.linalg.norm(rhs)
        ),
        "mhm_macro_equilibrium_linf": float(np.max(abs(native.equilibrium_residuals()))),
        "dolfinx_version": dolfinx.__version__,
        "basix_version": basix.__version__,
        "independent_solver": "native PETSc LU with MUMPS",
    }


@pytest.mark.fem
def test_polygonal_anisotropic_fields_match_independent_full_ufl_saddle(monkeypatch):
    """Verify non-affine fields and quadratic normal moments with nonzero exterior data."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    result = compare_polygon_problem(1)
    assert_allclose(result["relative_l2_stress_displacement_rotation"], 0, atol=2e-11)
    assert result["quadratic_traction_moment_linf"] < 2e-12
    assert result["full_saddle_relative_residual"] < 2e-12
