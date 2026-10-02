"""Native DOLFINx assembly and global MHM reconstruction with signed traces."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.elements import boundary_data, face_integration, p1_operators
from pymhm.fenics import (
    brinkman_forms,
    elasticity_forms,
    from_ufl,
    mixed_darcy_forms,
    primal_darcy_forms,
    usfem_brinkman_forms,
)
from pymhm.hybrid import HybridSystem
from pymhm.mesh import SkeletonSpace, TriangleMesh

pytestmark = pytest.mark.fem


def require_fenics() -> tuple[Any, Any, Any, Any, Any]:
    """Skip only when the optional finite-element stack is not installed."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    return dolfinx.fem, dolfinx.mesh, basix.ufl, ufl, MPI


def test_two_subdomain_darcy_affine_reconstruction() -> None:
    """Match independent P1 assembly and reproduce p=1+x+2y across two cells."""
    fem, dmesh, basix_ufl, ufl, mpi = require_fenics()
    coarse = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(coarse)
    problems, coordinates = [], []
    for cell in range(2):
        fine = coarse.submesh(cell, 3)
        domain = ufl.Mesh(basix_ufl.element("Lagrange", "triangle", 1, shape=(2,)))
        local_mesh = dmesh.create_mesh(mpi.COMM_SELF, fine.cells, fine.points, domain)
        space = fem.functionspace(local_mesh, ("Lagrange", 1))
        test = ufl.TestFunction(space)
        a, load = primal_darcy_forms(space, 1.0, 0.0)
        entities, tags = [], []
        for side, face in enumerate(coarse.cell_faces[cell]):
            start, end = coarse.points[coarse.faces[face]]
            tangent = end - start

            def on_face(
                points: np.ndarray, start: Any = start, tangent: Any = tangent
            ) -> np.ndarray:
                return np.isclose(
                    (points[0] - start[0]) * tangent[1] - (points[1] - start[1]) * tangent[0], 0
                )

            facets = dmesh.locate_entities_boundary(local_mesh, 1, on_face)
            entities.extend(facets)
            tags.extend([side + 1] * len(facets))
        order = np.argsort(entities)
        markers = dmesh.meshtags(
            local_mesh,
            1,
            np.asarray(entities, dtype=np.int32)[order],
            np.asarray(tags, dtype=np.int32)[order],
        )
        measure = ufl.Measure("ds", domain=local_mesh, subdomain_data=markers)
        trace_forms = [
            float(coarse.signs[cell, side]) * test * measure(side + 1) for side in range(3)
        ]
        points = space.tabulate_dof_coordinates()[:, :2]
        coordinates.append(points)
        problem = from_ufl(
            a,
            load,
            trace_forms,
            skeleton.cell_dofs(cell),
            kernel=np.ones((len(points), 1)),
            constraint_forms=[test * ufl.dx],
        )
        problems.append(problem)
        permutation = np.argmin(
            np.linalg.norm(points[:, None] - fine.points[None, :], axis=2), axis=1
        )
        assert len(np.unique(permutation)) == len(points)
        native_matrix, native_mass, _ = p1_operators(fine)
        native_trace, _ = face_integration(coarse, cell, fine, skeleton)
        assert_allclose(
            problem.matrix.toarray(),
            native_matrix.toarray()[permutation][:, permutation],
            atol=2e-14,
        )
        assert_allclose(problem.coupling, native_trace[permutation], atol=2e-14)
        assert_allclose(
            problem.constraints[:, 0],
            np.asarray(native_mass.sum(axis=1)).ravel()[permutation],
            atol=2e-14,
        )
    boundary, _ = boundary_data(skeleton, lambda x: 1 + x[:, 0] + 2 * x[:, 1])
    solution = HybridSystem(problems, boundary_load=boundary).solve()
    for values, points in zip(solution.fields, coordinates, strict=True):
        assert_allclose(values, 1 + points[:, 0] + 2 * points[:, 1], atol=2e-12)
    assert_allclose(solution.trace, coarse.normals @ np.array([-1.0, -2.0]), atol=2e-12)
    assert solution.residual < 1e-12


@pytest.mark.parametrize("reaction", [1e-8, 1e-12])
def test_native_near_poisson_retained_coarse_mode(reaction: float) -> None:
    """Match an independent RAD saddle solve while retaining near-null constants."""
    fem, dmesh, _, ufl, mpi = require_fenics()
    domain = dmesh.create_unit_square(mpi.COMM_SELF, 2, 2)
    space = fem.functionspace(domain, ("Lagrange", 1))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(domain)
    exact = 1 + x[0] + 2 * x[1]
    resistance = reaction * (1 + x[0])
    velocity = ufl.as_vector((1e-8, -2e-8))
    a = (
        ufl.inner(ufl.grad(trial), ufl.grad(test))
        + ufl.dot(velocity, ufl.grad(trial)) * test
        + resistance * trial * test
    ) * ufl.dx
    load = (ufl.dot(velocity, ufl.grad(exact)) + resistance * exact) * test * ufl.dx
    entities, tags = [], []
    for marker, (axis, value) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1)), start=1):
        facets = dmesh.locate_entities_boundary(
            domain, 1, lambda points, axis=axis, value=value: np.isclose(points[axis], value)
        )
        entities.extend(facets)
        tags.extend([marker] * len(facets))
    order = np.argsort(entities)
    markers = dmesh.meshtags(
        domain,
        1,
        np.asarray(entities, dtype=np.int32)[order],
        np.asarray(tags, dtype=np.int32)[order],
    )
    measure = ufl.Measure("ds", domain=domain, subdomain_data=markers)
    traces = [test * measure(marker) for marker in range(1, 5)]
    points = space.tabulate_dof_coordinates()[:, :2]
    problem = from_ufl(
        a,
        load,
        traces,
        np.arange(4),
        coarse_basis=np.ones((len(points), 1)),
        constraint_forms=[test * ufl.dx],
    )
    boundary = np.array(
        [fem.assemble_scalar(fem.form(exact * measure(marker))) for marker in range(1, 5)]
    )
    assembled = fem.assemble_matrix(fem.form(a))
    assembled.scatter_reverse()
    matrix = assembled.to_scipy().toarray()
    coupling = np.column_stack([fem.assemble_vector(fem.form(form)).array for form in traces])
    source = fem.assemble_vector(fem.form(load)).array
    full = np.linalg.solve(
        np.block([[matrix, coupling], [coupling.T, np.zeros((4, 4))]]),
        np.r_[source, boundary],
    )
    solution = HybridSystem([problem], boundary_load=boundary).solve()
    assert problem.kernel.shape == (len(points), 0)
    assert_allclose(solution.fields[0], full[:-4], atol=2e-12, rtol=0)
    assert_allclose(solution.trace, full[-4:], atol=2e-12, rtol=0)
    assert_allclose(solution.fields[0], 1 + points[:, 0] + 2 * points[:, 1], atol=2e-12, rtol=0)
    assert_allclose(solution.trace, [1, -1, 2, -2], atol=2e-12, rtol=0)
    assert solution.residual < 1e-12


def test_native_mixed_darcy_pressure_boundary_sign() -> None:
    """A constant prescribed pressure has zero RT velocity and correct pressure."""
    fem, dmesh, basix_ufl, ufl, mpi = require_fenics()
    domain = dmesh.create_unit_square(mpi.COMM_SELF, 2, 2)
    element = basix_ufl.mixed_element(
        [
            basix_ufl.element("RT", domain.basix_cell(), 1),
            basix_ufl.element("DG", domain.basix_cell(), 0),
        ]
    )
    space = fem.functionspace(domain, element)
    a, load = mixed_darcy_forms(space, 1.0, 0.0)
    test_velocity, _ = ufl.TestFunctions(space)
    load -= 2.0 * ufl.dot(test_velocity, ufl.FacetNormal(domain)) * ufl.ds
    problem = from_ufl(a, load, [], np.empty(0, dtype=int))
    field = fem.Function(space)
    field.x.array[:] = problem.condense().source
    velocity, pressure = field.split()
    assert_allclose(velocity.collapse().x.array, 0, atol=1e-12)
    assert_allclose(pressure.collapse().x.array, 2, atol=1e-12)


def test_native_stokes_brinkman_elasticity_and_usfem_forms() -> None:
    """Check symmetric assembly and the negative USFEM pressure stabilization."""
    fem, dmesh, basix_ufl, ufl, mpi = require_fenics()
    domain = dmesh.create_unit_square(mpi.COMM_SELF, 2, 2)
    scalar = basix_ufl.element("Lagrange", domain.basix_cell(), 1)
    vector = basix_ufl.element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    equal_vector = basix_ufl.element("Lagrange", domain.basix_cell(), 1, shape=(2,))
    mixed = fem.functionspace(domain, basix_ufl.mixed_element([vector, scalar]))
    equal = fem.functionspace(domain, basix_ufl.mixed_element([equal_vector, scalar]))
    displacement = fem.functionspace(domain, vector)
    force = ufl.as_vector((1.0, 0.0))
    pairs = [
        brinkman_forms(mixed, 1.0, 0.0, force),
        brinkman_forms(mixed, 1.0, 2.0, force),
        elasticity_forms(displacement, 2.0, 1.0, force),
        usfem_brinkman_forms(equal, 1.0, 0.0, force, smallest_resistance=0.0),
    ]
    for a, load in pairs:
        problem = from_ufl(a, load, [], np.empty(0, dtype=int))
        assert_allclose(problem.matrix.toarray(), problem.matrix.toarray().T, atol=2e-13)
        assert np.linalg.norm(problem.load) > 0
    a, _ = pairs[-1]
    matrix = fem.assemble_matrix(fem.form(a)).to_scipy()
    _, pressure_map = equal.sub(1).collapse()
    pressure_block = matrix[pressure_map][:, pressure_map].toarray()
    assert np.trace(pressure_block) < 0
    assert np.linalg.eigvalsh(pressure_block).max() < 1e-12
    h = ufl.CellDiameter(domain)
    _, p = ufl.TrialFunctions(equal)
    _, q = ufl.TestFunctions(equal)
    expected = fem.assemble_matrix(
        fem.form(-(h**2) / 24 * ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx)
    ).to_scipy()
    assert_allclose(pressure_block, expected[pressure_map][:, pressure_map].toarray(), atol=1e-14)


@pytest.mark.parametrize("stabilized", [False, True])
def test_native_brinkman_constant_velocity_source_consistency(stabilized: bool) -> None:
    """Verify that matching drag/residual forcing reproduces constant velocity."""
    fem, dmesh, basix_ufl, ufl, mpi = require_fenics()
    domain = dmesh.create_unit_square(mpi.COMM_SELF, 2, 2)
    scalar = basix_ufl.element("Lagrange", domain.basix_cell(), 1)
    vector = basix_ufl.element("Lagrange", domain.basix_cell(), 1 if stabilized else 2, shape=(2,))
    space = fem.functionspace(domain, basix_ufl.mixed_element([vector, scalar]))
    force = ufl.as_vector((2.0, -2.0))
    if stabilized:
        a, load = usfem_brinkman_forms(space, 1.0, 2.0, force, smallest_resistance=2.0)
    else:
        a, load = brinkman_forms(space, 1.0, 2.0, force)
    problem = from_ufl(a, load, [], np.empty(0, dtype=int))
    exact = fem.Function(space)
    exact.sub(0).interpolate(lambda x: np.vstack((np.ones(x.shape[1]), -np.ones(x.shape[1]))))
    assert_allclose(problem.matrix @ exact.x.array, problem.load, atol=2e-14)
    assert_allclose(problem.condense().source, exact.x.array, atol=3e-11)
