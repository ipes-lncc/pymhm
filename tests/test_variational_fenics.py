"""Native local-form assembly with physical moments and signed macrofaces."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.backends.fenics import assemble_local_forms, from_ufl
from pymhm.core.system import HybridSystem
from pymhm.core.variational import LocalForm
from pymhm.fem.scalar.operators import boundary_data, face_integration, p1_operators
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("mode", ["kernel", "coarse_basis"])
def test_native_forms_preserve_operator_moments_and_signed_interfaces(mode: str) -> None:
    pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    coarse = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(coarse)
    problems, coordinates = [], []
    reaction = 0.0 if mode == "kernel" else 1e-4
    for cell in range(2):
        fine = coarse.submesh(cell, 2)
        domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
        local_mesh = mesh.create_mesh(MPI.COMM_SELF, fine.cells, x=fine.points, e=domain)
        space = fem.functionspace(local_mesh, ("Lagrange", 1))
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        x = ufl.SpatialCoordinate(local_mesh)
        dx = ufl.dx(domain=local_mesh)
        a = (ufl.inner(ufl.grad(trial), ufl.grad(test)) + reaction * trial * test) * dx
        load = reaction * (1 + x[0] + 2 * x[1]) * test * dx
        entities, tags = [], []
        for side, face in enumerate(coarse.cell_faces[cell]):
            start, end = coarse.points[coarse.faces[face]]
            tangent = end - start

            def on_face(points: Any, start: Any = start, tangent: Any = tangent) -> Any:
                return np.isclose(
                    (points[0] - start[0]) * tangent[1] - (points[1] - start[1]) * tangent[0], 0
                )

            facets = mesh.locate_entities_boundary(local_mesh, 1, on_face)
            entities.extend(facets)
            tags.extend([side + 1] * len(facets))
        order = np.argsort(entities)
        markers = mesh.meshtags(
            local_mesh,
            1,
            np.asarray(entities, dtype=np.int32)[order],
            np.asarray(tags, dtype=np.int32)[order],
        )
        ds = ufl.Measure("ds", domain=local_mesh, subdomain_data=markers)
        traces = tuple(float(coarse.signs[cell, side]) * test * ds(side + 1) for side in range(3))
        points = space.tabulate_dof_coordinates()[:, :2]
        basis = np.ones((len(points), 1))
        forms = LocalForm(
            a,
            load,
            traces,
            skeleton.cell_dofs(cell),
            moment_forms=(test * ufl.dx,),
            **{mode: basis},
        )
        problem = assemble_local_forms(forms)
        direct = from_ufl(
            a,
            load,
            traces,
            forms.trace_dofs,
            constraint_forms=forms.moment_forms,
            **{mode: basis},
        )
        assert_allclose(problem.matrix.toarray(), direct.matrix.toarray(), atol=0)
        assert_allclose(problem.coupling, direct.coupling, atol=0)
        assert_allclose(problem.constraints, direct.constraints, atol=0)
        assert_allclose(problem.load, direct.load, atol=0)
        permutation = np.argmin(
            np.linalg.norm(points[:, None] - fine.points[None, :], axis=2), axis=1
        )
        assert len(np.unique(permutation)) == len(points)
        stiffness, mass, _ = p1_operators(fine)
        coupling, _ = face_integration(coarse, cell, fine, skeleton)
        assert_allclose(
            problem.matrix.toarray(),
            (stiffness + reaction * mass).toarray()[permutation][:, permutation],
            atol=2e-14,
        )
        assert_allclose(problem.coupling, coupling[permutation], atol=2e-14)
        assert_allclose(
            problem.constraints[:, 0], np.asarray(mass.sum(axis=1)).ravel()[permutation], atol=2e-14
        )
        problems.append(problem)
        coordinates.append(points)
    boundary, _ = boundary_data(skeleton, lambda x: 1 + x[:, 0] + 2 * x[:, 1])
    solution = HybridSystem(problems, boundary_load=boundary).solve()
    for values, points in zip(solution.fields, coordinates, strict=True):
        assert_allclose(values, 1 + points[:, 0] + 2 * points[:, 1], atol=2e-12)
    assert_allclose(solution.trace, coarse.normals @ [-1.0, -2.0], atol=2e-12)
    assert solution.raw_residual is not None and solution.raw_residual < 1e-12
