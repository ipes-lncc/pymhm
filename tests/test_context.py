"""Contextual and manual equations share the same hybrid numerical owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import Equation, ExecutionConfig, LocalEquations, MultiscaleProblem, assemble, solve
from pymhm.core.context import GlobalContext, LocalContext, bind_problem
from pymhm.core.equations import compile_local_equations
from pymhm.core.spaces import MeshHierarchy, TraceBinding, bind_interface
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class AlgebraicInterface:
    """Independent nonsymmetric trial/test changes of a shared global basis."""

    size: int = 2

    def binding(self, cell: int) -> TraceBinding:
        return TraceBinding(
            [0, 1],
            [[1, 0.2], [-0.3, 1.1]],
            [1, 0],
            [[0.7, -0.1], [0.2, 1.3]],
            basis_id=f"declared-face-basis-{cell}",
            require_injective=True,
        )


def contextual_algebra(ctx: LocalContext) -> LocalEquations:
    """Declare local mathematics and a field without handling global indices."""
    cell = ctx.cell
    ctx.field("potential", evaluator=constant_evaluator, basis_id="two-coordinate-polynomial")
    return ctx.equations(
        a=np.array([[3.0, 1.0], [0.0, 2.0]]) + cell * np.eye(2),
        L=[1 + cell, -0.3],
        b=[[1.0, 0.2], [-0.1, 0.8]],
        c=[[0.5, 0.3], [-0.2, 0.9]],
        d=[[0.4, 0.2], [-0.1, 0.5]],
        g=[0.2, -0.4],
        metadata=cell,
    )


def constant_evaluator(mesh: Any, coefficients: Any, points: Any, *, cells: Any = None) -> Any:
    """A user evaluator whose independent polynomial convention is explicit."""
    points = np.asarray(points)
    return coefficients[0] + coefficients[1] * points[:, 0]


def manual_algebra(cell: int) -> LocalEquations:
    """Independently form global-oriented blocks of the declared same equations."""
    r = np.array([[1, 0.2], [-0.3, 1.1]])
    s = np.array([[0.7, -0.1], [0.2, 1.3]])
    return LocalEquations(
        np.array([[3.0, 1.0], [0.0, 2.0]]) + cell * np.eye(2),
        [1 + cell, -0.3],
        np.array([[1.0, 0.2], [-0.1, 0.8]]) @ r,
        s.T @ np.array([[0.5, 0.3], [-0.2, 0.9]]),
        [0, 1],
        test_dofs=[1, 0],
        d=s.T @ np.array([[0.4, 0.2], [-0.1, 0.5]]) @ r,
        g=s.T @ np.array([0.2, -0.4]),
        metadata=cell,
    )


def _problem(**kwargs: Any) -> Any:
    """Bind portable mesh descriptions to the user-written local provider."""
    macro = TriangleMesh.unit_square()
    return bind_problem(
        MeshHierarchy(macro, [macro, macro]),
        AlgebraicInterface(),
        contextual_algebra,
        global_equation=Equation(4 * np.eye(2), [0.7, -0.1]),
        **kwargs,
    )


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_context_forms_match_manual_and_uncondensed_equations(backend: str) -> None:
    """Nonsymmetric B/C, shared coordinates and named fields survive all CPU modes."""
    problem = _problem()
    execution = ExecutionConfig(backend, workers=2, batch_size=1)
    system = assemble(problem, execution=execution)
    result = solve(system)
    manual = MultiscaleProblem(
        Equation(4 * np.eye(2), [0.7, -0.1]), manual_algebra, [0, 1], 2, (0, 0)
    )
    reference_system = assemble(manual)
    reference = solve(reference_system)
    assert_array_equal(system.matrix.toarray(), reference_system.matrix.toarray())
    assert_array_equal(system.rhs, reference_system.rhs)
    assert_allclose(result.trace, reference.trace, atol=2e-12, rtol=2e-12)
    assert_allclose(result.fields, reference.fields, atol=2e-12, rtol=2e-12)
    for index, binding in enumerate(system.trace_bindings):
        assert_allclose(result.local_trace(index), binding.trial_map @ result.trace[binding.dofs])
        assert_allclose(
            result.local_trace(index, test=True), binding.test_map @ result.trace[binding.test_dofs]
        )
        assert result.trace_bindings[index].basis_digest == binding.basis_digest
    with pytest.raises(ValueError, match="outside"):
        result.local_trace(2)
    with pytest.raises(TypeError, match="boolean"):
        result.local_trace(0, test=1)
    with pytest.raises(TypeError, match="no declared"):
        reference.local_trace(0)
    original = np.zeros((6, 6))
    rhs = np.zeros(6)
    original[4:, 4:] = 4 * np.eye(2)
    rhs[4:] = [0.7, -0.1]
    for cell in (0, 1):
        eq = manual_algebra(cell)
        ii = np.arange(2 * cell, 2 * cell + 2)
        jj, kk = 4 + eq.dofs, 4 + eq.test_dofs
        original[np.ix_(ii, ii)] = eq.a
        original[np.ix_(ii, jj)] = eq.b
        original[np.ix_(kk, ii)] = eq.c
        original[np.ix_(kk, jj)] += eq.d
        rhs[ii] = eq.L
        rhs[kk] += eq.g
    expected = np.linalg.solve(original, rhs)
    assert_allclose(np.r_[*result.fields, result.trace], expected, atol=2e-12, rtol=2e-12)
    for field, values in zip(result.field("potential"), result.fields, strict=True):
        assert_array_equal(field.coefficients, values)
        assert_allclose(
            field.evaluate([[0.2, 0.3], [0.7, 0.2]]), values[0] + values[1] * np.array([0.2, 0.7])
        )


def test_declared_boundaries_and_physical_gauges_reuse_existing_global_owner() -> None:
    """Fixed values and independent gauge rows preserve explicit global mathematics."""
    problem = _problem(fixed=lambda ctx: {0: 0.25})
    result = solve(problem)
    manual = MultiscaleProblem(
        Equation(4 * np.eye(2), [0.7, -0.1]), manual_algebra, [0, 1], 2, (0, 0), fixed={0: 0.25}
    )
    assert_allclose(result.fields, solve(manual).fields, atol=2e-12, rtol=2e-12)
    assert result.trace[0] == 0.25
    ungauged = solve(_problem())
    gauge = ((np.array([0.0, 1.0]), ungauged.trace[1]),)
    gauged = solve(_problem(constraints=gauge))
    assert_allclose(gauged.trace, ungauged.trace, atol=2e-12, rtol=2e-12)
    assert_allclose(gauged.gauge_multipliers, 0, atol=2e-12)


def test_declared_layout_and_optional_custom_boundary_capabilities() -> None:
    """Retained coordinates follow hierarchy order, while physical signs stay explicit."""
    macro = TriangleMesh.unit_square()
    hierarchy = MeshHierarchy(macro, [macro, macro], items=[1, 0])

    class MutableInterface:
        size = 2

        def __init__(self) -> None:
            self.calls = 0

        def binding(self, cell: int) -> TraceBinding:
            self.calls += 1
            return AlgebraicInterface().binding(cell)

    mutable = MutableInterface()
    snapshot = GlobalContext(hierarchy, mutable, [0, 0])
    mutable.size = 3
    assert snapshot.trace_size == 2 and snapshot.coarse_sizes == (0, 0)
    local_snapshot = LocalContext(snapshot, 0)
    assert local_snapshot.binding is local_snapshot.binding
    assert mutable.calls == 1
    local_snapshot.close()
    problem = bind_problem(
        hierarchy,
        AlgebraicInterface(),
        contextual_algebra,
        retained=(2, 1),
        global_equation=lambda ctx: Equation(np.eye(ctx.size), ctx.trace_load([1, 2])),
    )
    ctx = problem.context
    assert ctx.trace_size == 2 and ctx.size == 5
    assert_array_equal(ctx.retained_dofs(1), [2, 3])
    assert_array_equal(ctx.retained_dofs(0), [4])
    assert_array_equal(ctx.trace_load([1, 2]), [1, 2, 0, 0, 0])
    with pytest.raises(ValueError, match="not selected"):
        ctx.retained_dofs(2)

    class CustomBoundary(AlgebraicInterface):
        def boundary_data(self, value: Any, neumann: Any, *, order: int) -> Any:
            return np.array([value, order], dtype=float), dict(neumann)

    custom = GlobalContext(hierarchy, CustomBoundary(), (0, 0))
    load, fixed = custom.boundary_data(3.5, {1: 2.0}, order=4)
    assert_array_equal(load, [3.5, 4])
    assert fixed == {1: 2.0}
    with pytest.raises(TypeError, match="boundary-data capability"):
        ctx.boundary_data(0)
    with pytest.raises(TypeError, match="face-value projection"):
        ctx.fix_faces([0])
    from pymhm.fem.traces.triangle_3d import TriangularSkeleton
    from pymhm.meshes.tetrahedron import TetraMesh

    volume = TetraMesh.unit_cube()
    hierarchy_3d = MeshHierarchy(volume, [volume] * len(volume.cells))
    context_3d = GlobalContext(
        hierarchy_3d, bind_interface(TriangularSkeleton(volume)), (0,) * len(volume.cells)
    )
    load3d, fixed3d = context_3d.boundary_data(0)
    assert_array_equal(load3d, np.zeros(context_3d.trace_size))
    assert fixed3d == {}
    assert set(context_3d.fix_faces(volume.boundary_faces)) == {
        int(dof)
        for face in volume.boundary_faces
        for dof in context_3d.interface.space.dofs(int(face))
    }
    skeleton = SkeletonSpace(macro, [FaceSpace(degrees=(1,)) for _ in macro.faces])
    built = GlobalContext(hierarchy, bind_interface(skeleton), (0, 0))
    load, fixed = built.boundary_data(1.0, {int(macro.boundary_faces[0]): 2.0})
    from pymhm.fem.scalar.operators import boundary_data

    expected_load, expected_fixed = boundary_data(
        skeleton, 1.0, {int(macro.boundary_faces[0]): 2.0}
    )
    assert_allclose(load, expected_load, atol=2e-12, rtol=2e-12)
    assert fixed.keys() == expected_fixed.keys()
    assert_allclose(list(fixed.values()), list(expected_fixed.values()), atol=2e-12, rtol=2e-12)
    face = int(macro.boundary_faces[0])
    prescribed = built.fix_faces([face], 3.0)
    assert_array_equal(
        np.array([prescribed[int(index)] for index in skeleton.dofs(face)]),
        3 * skeleton.faces[face].constant_coefficients(),
    )
    with pytest.raises(ValueError, match="face index outside"):
        built.fix_faces([len(macro.faces)])
    from pymhm.methods.three_field import PressureTraceSpace

    pressure_trace = PressureTraceSpace.uniform(macro)
    shared = GlobalContext(hierarchy, bind_interface(pressure_trace), (0, 0))
    fixed = shared.fix_faces(macro.boundary_faces, 2.0)
    assert fixed == {
        int(index): 2.0 for face in macro.boundary_faces for index in pressure_trace.face_dofs[face]
    }


def test_context_mesh_is_lazy_and_provider_resources_are_released() -> None:
    """Local contexts allocate only requested meshes and forward worker lifecycle hooks."""
    macro = TriangleMesh.unit_square()
    calls: list[int] = []

    def local_mesh(cell: int) -> Any:
        calls.append(cell)
        return macro.submesh(cell, 1)

    class Provider:
        def __init__(self) -> None:
            self.events: list[str] = []

        def prepare_runtime(self) -> None:
            self.events.append("prepare")

        def close(self) -> None:
            self.events.append("close")

        def __call__(self, ctx: LocalContext) -> Any:
            return compile_local_equations(contextual_algebra(ctx))

    provider = Provider()
    problem = bind_problem(
        MeshHierarchy(macro, local_mesh),
        AlgebraicInterface(),
        provider,
        global_equation=Equation(4 * np.eye(2), [0.7, -0.1]),
    )
    assert calls == []
    local = problem.local_context(0)
    assert local.macro is macro
    assert local.mesh is local.mesh
    assert calls == [0]
    local.close()
    solve(problem)
    assert provider.events == ["prepare", "close"]
    assert calls == [0, 0, 1]
    with pytest.raises(ValueError, match="not selected"):
        problem.local_context(2)
    invalid = bind_problem(problem.context.hierarchy, AlgebraicInterface(), lambda ctx: None)
    with pytest.raises(TypeError, match="equation contract"):
        solve(invalid)
    with pytest.raises(ValueError, match="unique"):
        local.field("potential")
        local.field("potential")
    with pytest.raises(ValueError, match="bind a native"):
        local.trace_pairings(lambda value, measure: value * measure)


def test_problem_binding_rejects_incompatible_declarations_before_assembly() -> None:
    """Invalid mathematical layout declarations are rejected before solving any cell."""
    macro = TriangleMesh.unit_square()
    hierarchy = MeshHierarchy(macro, [macro, macro])
    for args, error, message in (
        ((None, AlgebraicInterface(), (0, 0)), TypeError, "MeshHierarchy"),
        ((hierarchy, object(), (0, 0)), TypeError, "callable"),
        ((hierarchy, AlgebraicInterface(), (0,)), ValueError, "hierarchy item order"),
        ((hierarchy, AlgebraicInterface(), (-1, 0)), ValueError, "retained modes"),
    ):
        with pytest.raises(error, match=message):
            GlobalContext(*args)
    for args, kwargs, error, message in (
        ((None, AlgebraicInterface(), contextual_algebra), {}, TypeError, "MeshHierarchy"),
        ((hierarchy, AlgebraicInterface(), None), {}, TypeError, "callable"),
        ((hierarchy, object(), contextual_algebra), {}, TypeError, "callable"),
        (
            (hierarchy, AlgebraicInterface(), contextual_algebra),
            {"retained": (0,)},
            ValueError,
            "hierarchy item order",
        ),
        (
            (hierarchy, AlgebraicInterface(), contextual_algebra),
            {"retained": -1},
            ValueError,
            "retained modes",
        ),
        (
            (hierarchy, AlgebraicInterface(), contextual_algebra),
            {"global_equation": 1},
            TypeError,
            "Equation",
        ),
    ):
        with pytest.raises(error, match=message):
            bind_problem(*args, **kwargs)
    other = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="same macro mesh"):
        bind_problem(hierarchy, bind_interface(SkeletonSpace(other)), contextual_algebra)


def test_custom_context_capabilities_and_explicit_global_coordinates() -> None:
    """Custom interfaces may own all representation and native-form capabilities."""
    macro = TriangleMesh.unit_square()

    class CustomInterface(AlgebraicInterface):
        def trace_pairings(self, context: Any, expression: Any, *, axis: str) -> Any:
            return expression(context.cell, axis)

        def interface_equation(self, context: Any, forms: Any) -> Equation:
            return forms(context.trace_size, context.size, None)

        def interface_pairing(self, context: Any, *, order: int) -> Any:
            return order * np.eye(context.binding.trial_size)

    problem = bind_problem(
        MeshHierarchy(macro, [macro, macro]),
        CustomInterface(),
        contextual_algebra,
        retained=np.int64(0),
    )
    local = problem.local_context(0)
    assert_array_equal(local.interface_pairing(CustomInterface(), order=3), 3 * np.eye(2))
    with pytest.raises(TypeError, match="boundary-pairing"):
        local.interface_pairing(AlgebraicInterface())
    assert local.trace_pairings(lambda cell, axis: (cell, axis), interface=CustomInterface()) == (
        0,
        "columns",
    )
    assert local.trace_pairings(lambda cell, axis: (cell, axis), axis="rows") == (0, "rows")
    global_equation = problem.context.interface_equation(
        lambda trace, size, measure: Equation(np.eye(size), np.ones(trace))
    )
    assert_array_equal(global_equation.a, np.eye(2))
    assert_array_equal(global_equation.L, [1, 1])
    eq = manual_algebra(0)
    direct = local.equations(a=eq.a, L=eq.L, b=eq.b, c=eq.c, d=eq.d, g=eq.g, coordinates="global")
    assert_array_equal(direct.b, eq.b)
    assert_array_equal(direct.c, eq.c)
    local.field("custom", mesh="declared-mesh", evaluator=constant_evaluator, basis_id="executed")
    assert local._fields[0].mesh == "declared-mesh"
    with pytest.raises(ValueError, match="outside the interface"):
        bind_problem(problem.context.hierarchy, AlgebraicInterface(1), contextual_algebra)
    with pytest.raises(ValueError, match="interface size"):
        bind_problem(problem.context.hierarchy, AlgebraicInterface(-1), contextual_algebra)
    local.close()


def test_nested_context_automatic_restriction_matches_explicit_multilevel_equations() -> None:
    """Recursive contexts delegate exactly to the existing oriented restriction owner."""
    from pymhm.core.multiscale import NestedEquations
    from pymhm.core.nested import nested_trace_map

    macro = TriangleMesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]])
    child_mesh = macro.submesh(0, 2)
    outer_space = SkeletonSpace(macro, [FaceSpace.uniform(degree=1) for _ in macro.faces])
    inner_space = SkeletonSpace(child_mesh, [FaceSpace.uniform(degree=1) for _ in child_mesh.faces])

    def child_provider(ctx: LocalContext) -> LocalEquations:
        return ctx.equations(
            a=[[2]],
            L=[1],
            b=np.zeros((1, ctx.binding.trial_size)),
            c=np.zeros((ctx.binding.test_size, 1)),
        )

    child = bind_problem(
        MeshHierarchy(child_mesh, [child_mesh] * len(child_mesh.cells)),
        bind_interface(inner_space, convention="normal"),
        child_provider,
        global_equation=Equation(np.eye(inner_space.size), np.arange(inner_space.size)),
    )
    hierarchy = MeshHierarchy(macro, [child_mesh])
    outer = bind_problem(
        hierarchy,
        bind_interface(outer_space, convention="normal"),
        lambda ctx: ctx.nested(child),
        global_equation=Equation(np.eye(outer_space.size), np.ones(outer_space.size)),
    )
    boundary, mapping = nested_trace_map(outer_space, 0, inner_space)
    explicit = MultiscaleProblem(
        outer.global_equation,
        lambda cell: NestedEquations(child, boundary, mapping, outer_space.cell_dofs(cell)),
        [0],
        outer_space.size,
        (0,),
    )
    automatic, manual = solve(outer), solve(explicit)
    assert_allclose(automatic.trace, manual.trace, atol=2e-12, rtol=2e-12)
    binding = outer.context.interface.binding(0)
    assert_allclose(automatic.local_trace(0), binding.trial_map @ automatic.trace[binding.dofs])
    assert_allclose(automatic.children[0].trace, manual.children[0].trace, atol=2e-12, rtol=2e-12)
    local = outer.local_context(0)
    declared = local.nested(child, boundary_dofs=boundary, trace_map=mapping)
    assert_array_equal(declared.trace_map, mapping)
    with pytest.raises(ValueError, match="declare both"):
        local.nested(child, boundary_dofs=boundary)
    incompatible = bind_problem(
        hierarchy, bind_interface(outer_space, convention="value"), child_provider
    )
    with pytest.raises(TypeError, match="explicit child trace restriction"):
        incompatible.local_context(0).nested(child)
    too_coarse = bind_problem(
        child.context.hierarchy,
        bind_interface(SkeletonSpace(child_mesh), convention="normal"),
        child_provider,
    )
    with pytest.raises(ValueError, match="cannot represent"):
        local.nested(too_coarse)
    local.close()
