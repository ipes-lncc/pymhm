"""Robin diffusion uses only public variational contexts in both dimensions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import (
    Equation,
    FaceSpace,
    MeshHierarchy,
    SkeletonSpace,
    TetraMesh,
    TriangleMesh,
    TriangularSkeleton,
    assemble,
    bind_interface,
    bind_problem,
    solve,
)
from pymhm.core.original import assemble_original_blocks
from pymhm.linalg.linear import solve_linear

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("dimension", [2, 3])
def test_user_robin_equations_reproduce_quadratic_pressure_and_physical_gradient(dimension):
    pytest.importorskip("dolfinx")
    import ufl

    macro = TriangleMesh.unit_square() if dimension == 2 else TetraMesh.unit_cube()
    skeleton = (
        SkeletonSpace(macro, tuple(FaceSpace.uniform(2) for _ in macro.faces))
        if dimension == 2
        else TriangularSkeleton(macro, degree=2)
    )
    hierarchy = MeshHierarchy(macro, lambda cell: macro.submesh(cell, 2))

    def local(ctx):
        native = ctx.native_space(degree=2)
        u, v = ufl.TrialFunction(native.space), ufl.TestFunction(native.space)
        x, n = ufl.SpatialCoordinate(native.mesh), ufl.FacetNormal(native.mesh)
        dx = ufl.Measure("dx", domain=native.mesh, metadata={"quadrature_degree": 6})
        ds = ufl.Measure("ds", domain=native.mesh, metadata={"quadrature_degree": 6})
        sigma = 0.25 * x / dimension
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + ufl.dot(sigma, n) * u * v * ds
        b = ctx.trace_pairings(lambda phi, face_ds: phi * v * face_ds)
        c = ctx.trace_pairings(lambda phi, face_ds: -phi * u * face_ds, axis="rows")
        ctx.field("pressure", native)
        return ctx.equations(a=a, L=-2 * dimension * v * dx, b=b, c=c)

    def global_(ctx):
        moments, _ = ctx.boundary_data(lambda points: 1 + np.sum(points**2, axis=1))
        return Equation(0, -ctx.trace_load(moments))

    problem = bind_problem(
        hierarchy,
        bind_interface(skeleton, convention="normal"),
        local,
        global_equation=global_,
        retained=0,
    )
    system = assemble(problem)
    solution = solve(system)
    original = assemble_original_blocks(system)
    independent = solve_linear(original.matrix, original.load)
    difference = np.r_[*solution.fields, solution.trace] - independent
    assert np.linalg.norm(difference) <= 1e-12 + 1e-10 * np.linalg.norm(independent)
    for record in system.cells:
        assert np.linalg.eigvalsh(record.equations.problem.matrix.toarray()).min() > 0
    assert np.linalg.eigvalsh(system.matrix.toarray()).min() > 0
    for field in solution.field("pressure"):
        points = field.mesh.points[field.mesh.cells].mean(axis=1)
        cells = np.arange(len(field.mesh.cells))
        assert_allclose(
            field.evaluate(points, cells=cells),
            1 + np.sum(points**2, axis=1),
            atol=1e-12,
            rtol=1e-10,
        )
        assert_allclose(-field.gradient(points, cells=cells), -2 * points, atol=1e-12, rtol=1e-10)
