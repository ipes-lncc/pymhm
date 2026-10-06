"""Real UFL local contexts, physical field descriptors and native ownership."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import Equation, ExecutionConfig, assemble, solve
from pymhm.core.context import LocalContext, bind_problem
from pymhm.core.spaces import MeshHierarchy, bind_interface
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh

pytestmark = pytest.mark.fem


def scalar_ufl_provider(ctx: LocalContext) -> Any:
    """Write reaction-diffusion with independently known affine field and normal data."""
    import ufl

    native = ctx.native_space(degree=2)
    u, v = ufl.TrialFunction(native.space), ufl.TestFunction(native.space)
    dx = ufl.Measure("dx", domain=native.mesh, metadata={"quadrature_degree": 6})
    ds = ufl.ds(domain=native.mesh)
    x, normal = ufl.SpatialCoordinate(native.mesh), ufl.FacetNormal(native.mesh)
    exact = ctx.cell + 1 + x[0] + 2 * x[1]
    ctx.field("pressure", native)
    return ctx.equations(
        a=(ufl.inner(ufl.grad(u), ufl.grad(v)) + u * v) * dx,
        L=exact * v * dx + ufl.dot(ufl.as_vector((1, 2)), normal) * v * ds,
        b=np.zeros((native.size, ctx.binding.trial_size)),
        c=np.zeros((ctx.binding.test_size, native.size)),
    )


def mixed_ufl_provider(ctx: LocalContext) -> Any:
    """Declare vector/scalar spaces and independent named physical components."""
    import basix
    import basix.ufl
    import ufl

    element = basix.ufl.mixed_element(
        [
            basix.ufl.element(
                "Lagrange",
                "triangle",
                2,
                shape=(2,),
                lagrange_variant=basix.LagrangeVariant.equispaced,
            ),
            basix.ufl.element(
                "Lagrange",
                "triangle",
                1,
                discontinuous=True,
                lagrange_variant=basix.LagrangeVariant.equispaced,
            ),
        ]
    )
    native = ctx.native_space(element)
    u, p = ufl.TrialFunctions(native.space)
    v, q = ufl.TestFunctions(native.space)
    dx = ufl.Measure("dx", domain=native.mesh, metadata={"quadrature_degree": 6})
    ctx.field("velocity", native, component=0)
    ctx.field("pressure", native, component=1)
    return ctx.equations(
        a=(ufl.inner(u, v) + p * q) * dx,
        L=(ufl.inner(ufl.as_vector((ctx.cell + 1, ctx.cell + 2)), v) + 3 * q) * dx,
        b=np.zeros((native.size, ctx.binding.trial_size)),
        c=np.zeros((ctx.binding.test_size, native.size)),
    )


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_native_context_forms_and_named_fields_survive_cpu_execution(backend: str) -> None:
    """The same user forms reconstruct scalar/vector/mixed fields in spawned workers."""
    pytest.importorskip("dolfinx")
    macro = TriangleMesh.unit_square()
    local = [macro.submesh(cell, 2) for cell in range(len(macro.cells))]
    hierarchy = MeshHierarchy(macro, local)
    interface = bind_interface(SkeletonSpace(macro), convention="normal")
    for provider in (scalar_ufl_provider, mixed_ufl_provider):
        problem = bind_problem(
            hierarchy,
            interface,
            provider,
            global_equation=Equation(np.eye(interface.size), np.arange(interface.size)),
        )
        system = assemble(problem, execution=ExecutionConfig(backend, workers=2, batch_size=1))
        solution = solve(system)
        assert_allclose(solution.trace, np.arange(interface.size), atol=2e-12)
        assert all(record.equations.trace_binding is not None for record in system.cells)
        for cell, field in enumerate(solution.field("pressure")):
            points = local[cell].points[local[cell].cells].mean(axis=1)
            expected = (
                3 if provider is mixed_ufl_provider else cell + 1 + points[:, 0] + 2 * points[:, 1]
            )
            assert_allclose(
                field.evaluate(points, cells=np.arange(len(points))),
                expected,
                atol=2e-10,
                rtol=2e-10,
            )
            assert field.basis_digest
            gradient = field.gradient(points, cells=np.arange(len(points)))
            expected_gradient = (
                np.zeros_like(points)
                if provider is mixed_ufl_provider
                else np.tile([1, 2], (len(points), 1))
            )
            assert_allclose(gradient, expected_gradient, atol=2e-10, rtol=2e-10)
            sampled_values, sampled_gradient = field.values_and_gradient(
                points, cells=np.arange(len(points))
            )
            assert_allclose(sampled_values, expected, atol=2e-10, rtol=2e-10)
            assert_allclose(sampled_gradient, expected_gradient, atol=2e-10, rtol=2e-10)
            descriptor = field.definition.descriptor
            assert_allclose(
                field.portable_coefficients, field.coefficients[descriptor.mapping], atol=2e-12
            )
        if provider is mixed_ufl_provider:
            for cell, field in enumerate(solution.field("velocity")):
                points = local[cell].points[local[cell].cells].mean(axis=1)
                assert_allclose(
                    field.evaluate(points),
                    np.tile([cell + 1, cell + 2], (len(points), 1)),
                    atol=2e-10,
                    rtol=2e-10,
                )
                assert_allclose(field.gradient(points), np.zeros((len(points), 2, 2)), atol=2e-10)


def test_native_context_resources_are_released_on_success_and_failure() -> None:
    """Worker-local bindings close on both compiled results and provider exceptions."""
    pytest.importorskip("dolfinx")
    macro = TriangleMesh.unit_square()
    hierarchy = MeshHierarchy(macro, [macro.submesh(cell, 1) for cell in range(2)])
    interface = bind_interface(SkeletonSpace(macro))

    class Captured:
        def __init__(self, fail: bool) -> None:
            self.fail = fail
            self.bindings: list[Any] = []

        def __call__(self, ctx: LocalContext) -> Any:
            native = ctx.native_space(degree=1, shape=(2,))
            self.bindings.append(native)
            if self.fail:
                raise RuntimeError("provider failed")
            ctx.field("velocity", native)
            return ctx.equations(
                a=np.eye(native.size),
                L=np.ones(native.size),
                b=np.zeros((native.size, ctx.binding.trial_size)),
                c=np.zeros((ctx.binding.test_size, native.size)),
            )

    for fail in (False, True):
        provider = Captured(fail)
        problem = bind_problem(
            hierarchy, interface, provider, global_equation=Equation(np.eye(interface.size), 0)
        )
        if fail:
            with pytest.raises(RuntimeError, match="provider failed"):
                assemble(problem)
        else:
            solution = solve(problem)
            assert_allclose(
                solution.field("velocity")[0].evaluate([[0.3, 0.1]]), [[1, 1]], atol=2e-10
            )
        assert provider.bindings
        assert all(
            binding.native_space is None and binding.native_mesh is None
            for binding in provider.bindings
        )
    import basix.ufl
    import ufl

    from pymhm.backends.spaces import bind_space

    local = problem.local_context(0)
    velocity = local.native_space(degree=2, shape=(2,))
    pressure = local.native_space(basix.ufl.element("Lagrange", "triangle", 1))
    assert velocity.mesh is pressure.mesh
    assert velocity.space.ufl_domain() is pressure.space.ufl_domain()
    vector, scalar = ufl.TestFunction(velocity.space), ufl.TrialFunction(pressure.space)
    pairing = ufl.inner(vector, ufl.grad(scalar)) * ufl.dx(domain=velocity.mesh)
    assert len(pairing.ufl_domains()) == 1
    borrowed = local.native_space(velocity.space)
    assert borrowed.space is velocity.space
    unrelated = bind_space(local.mesh, basix.ufl.element("Lagrange", "triangle", 1))
    with pytest.raises(ValueError, match="integration mesh"):
        local.native_space(unrelated.space)
    another = problem.local_context(0)
    accepted = another.native_space(unrelated.space)
    assert accepted.mesh is unrelated.mesh
    another.close()
    unrelated.close()
    local.close()
    assert local._native_mesh is None


@pytest.mark.parametrize("components,continuous", [(1, False), (2, True)])
def test_native_local_and_global_trace_forms_match_existing_physical_pairings(
    components: int,
    continuous: bool,
) -> None:
    """Partitioned face UFL forms preserve independent B/C and canonical incidence."""
    pytest.importorskip("dolfinx")
    import ufl

    from pymhm.core.equations import compile_local_equations
    from pymhm.fem.scalar.triangle import trace_coupling

    macro = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(
        macro,
        [FaceSpace.uniform(degree=2, subdivisions=2, continuous=continuous) for _ in macro.faces],
        components=components,
    )
    local_meshes = [macro.submesh(cell, 4) for cell in range(2)]
    problem = bind_problem(
        MeshHierarchy(macro, local_meshes),
        bind_interface(skeleton, convention="normal"),
        scalar_ufl_provider,
        retained=(1, 0),
    )
    for cell in range(2):
        ctx = problem.local_context(cell)
        native = ctx.native_space(degree=3, shape=() if components == 1 else (components,))
        u, v = ufl.TrialFunction(native.space), ufl.TestFunction(native.space)
        dx = ufl.dx(domain=native.mesh)
        b = ctx.trace_pairings(lambda phi, measure, v=v: ufl.inner(phi, v) * measure)
        c = ctx.trace_pairings(lambda phi, measure, u=u: -ufl.inner(phi, u) * measure, axis="rows")
        equations = ctx.equations(a=ufl.inner(u, v) * dx, L=0, b=b, c=c)
        compiled = compile_local_equations(equations)
        scalar = SkeletonSpace(macro, skeleton.faces)
        expected = np.kron(
            trace_coupling(macro, cell, local_meshes[cell], scalar, 3), np.eye(components)
        )
        assert_allclose(compiled.problem.coupling[native.mapping], expected, atol=2e-10, rtol=2e-10)
        # Native trial/test pairing signs remain independently supplied.
        assert_allclose(equations.c[:, native.mapping], -expected.T, atol=2e-10, rtol=2e-10)
        with pytest.raises(ValueError, match="axis"):
            ctx.trace_pairings(lambda phi, measure: phi * measure, axis="invalid")
        with pytest.raises(TypeError, match="callable"):
            ctx.trace_pairings(None)
        ctx.close()
    constant = 1 if components == 1 else ufl.as_vector([1, 2])
    equation = problem.context.interface_equation(
        lambda trace, test, dx: Equation(
            ufl.inner(trace, test) * dx, ufl.inner(constant, test) * dx
        )
    )
    matrix = np.zeros((problem.context.size, problem.context.size))
    load = np.zeros(problem.context.size)
    for face, declaration in enumerate(skeleton.faces):
        parameter, weights = declaration.quadrature(6)
        values = declaration.evaluate(parameter)
        weights *= macro.lengths[face]
        block = np.kron(values.T @ (weights[:, None] * values), np.eye(components))
        ids = skeleton.dofs(face)
        matrix[np.ix_(ids, ids)] = block
        integrated = values.T @ weights
        load[ids] = (integrated[:, None] * np.arange(1, components + 1)).ravel()
    assert_allclose(equation.a.toarray(), matrix, atol=2e-10, rtol=2e-10)
    assert_allclose(equation.L, load, atol=2e-10, rtol=2e-10)
    with pytest.raises(TypeError, match="builder"):
        problem.context.interface_equation(None)
    with pytest.raises(TypeError, match="return Equation"):
        problem.context.interface_equation(lambda trace, test, dx: 1)
