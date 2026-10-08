"""Native user boundary forms agree with independently integrated trace owners."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import (
    ComponentTraceSpace,
    Equation,
    GlobalContext,
    LocalContext,
    MeshHierarchy,
    PressureTraceSpace,
    PressureTraceSpace3D,
    TangentialTraceSpace,
    TetraMesh,
    TriangleMesh,
    TriangularSkeleton,
    bind_interface,
    compile_form,
    trace_bilinear_form,
    trace_linear_form,
)
from pymhm.fem.traces.triangle_3d import tetra_trace_coupling

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("degree,segments,continuous", [(0, 1, False), (2, 2, False), (3, 2, True)])
def test_tetrahedral_native_trace_pairing_preserves_signed_subface_basis(
    degree, segments, continuous
):
    pytest.importorskip("dolfinx")
    import ufl

    macro = TetraMesh.unit_cube()
    space = TriangularSkeleton(macro, segments, degree=degree, continuous=continuous)
    cell, fine = 1, macro.submesh(1, 2)
    global_ = GlobalContext(
        MeshHierarchy(macro, lambda _: fine, items=(cell,)),
        bind_interface(space, convention="normal"),
        (0,),
    )
    ctx = LocalContext(global_, cell)
    try:
        native = ctx.native_space(degree=2)
        v = ufl.TestFunction(native.space)
        forms = ctx.trace_pairings(lambda phi, ds: phi * v * ds)
        actual = compile_form(forms, (native.size, ctx.binding.trial_size)) @ ctx.binding.trial_map
        expected = native.transport_coupling(tetra_trace_coupling(macro, cell, fine, space, 2))
        assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
        rows = ctx.trace_pairings(lambda phi, ds: -phi * v * ds, axis="rows")
        assert_allclose(
            ctx.binding.test_map.T @ compile_form(rows, (ctx.binding.test_size, native.size)),
            -expected.T,
            atol=1e-12,
            rtol=1e-10,
        )
    finally:
        ctx.close()


@pytest.mark.parametrize("kind", ["pressure2", "pressure3", "scalar", "vector", "tangent"])
def test_global_native_interface_forms_preserve_c0_shared_and_tangent_modes(kind):
    pytest.importorskip("dolfinx")
    import ufl

    macro = TriangleMesh.unit_square() if kind == "pressure2" else TetraMesh.unit_cube()
    scalar = (
        TriangularSkeleton(macro, 2, degree=2, continuous=True) if kind != "pressure2" else None
    )
    space = {
        "pressure2": lambda: PressureTraceSpace.uniform(macro, 2, 2),
        "pressure3": lambda: PressureTraceSpace3D(macro, 2, 2),
        "scalar": lambda: scalar,
        "vector": lambda: ComponentTraceSpace(scalar, 3),
        "tangent": lambda: TangentialTraceSpace(scalar),
    }[kind]()
    hierarchy = MeshHierarchy(macro, lambda cell: macro.submesh(cell, 2))
    context = GlobalContext(hierarchy, bind_interface(space), (0,) * len(macro.cells))
    coefficient = 2.5
    value = [1.0, -2, 3] if kind in {"vector", "tangent"} else 1.25
    actual = context.interface_equation(
        lambda u, v, dx: Equation(
            coefficient * ufl.inner(u, v) * dx,
            ufl.inner(ufl.as_vector(value), v) * dx if isinstance(value, list) else value * v * dx,
        )
    )
    assert_allclose(
        actual.a.toarray(),
        trace_bilinear_form(space, coefficient).toarray(),
        atol=1e-12,
        rtol=1e-10,
    )
    assert_allclose(actual.L, trace_linear_form(space, value), atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("kind", ["pressure", "vector", "tangent"])
def test_local_native_pressure_and_vector_basis_uses_actual_unique_cell_dofs(kind):
    pytest.importorskip("dolfinx")
    import ufl

    macro = TetraMesh.unit_cube()
    scalar = TriangularSkeleton(macro, 2, degree=1, continuous=True)
    space = (
        PressureTraceSpace3D(macro, 2, 2)
        if kind == "pressure"
        else ComponentTraceSpace(scalar, 3)
        if kind == "vector"
        else TangentialTraceSpace(scalar)
    )
    cell, fine = 2, macro.submesh(2, 2)
    global_ = GlobalContext(
        MeshHierarchy(macro, lambda _: fine, items=(cell,)), bind_interface(space), (0,)
    )
    ctx = LocalContext(global_, cell)
    try:
        native = ctx.native_space(degree=1, shape=() if kind == "pressure" else (3,))
        v = ufl.TestFunction(native.space)
        forms = ctx.trace_pairings(lambda phi, ds: ufl.inner(phi, v) * ds)
        actual = compile_form(forms, (native.size, ctx.binding.trial_size))
        weight = (
            np.ones(native.size) if kind == "pressure" else np.tile([1.0, -2, 3], native.size // 3)
        )
        expected = trace_linear_form(
            space, 1 if kind == "pressure" else [1, -2, 3], faces=macro.cell_faces[cell]
        )[ctx.binding.dofs]
        assert_allclose(weight @ actual, expected, atol=1e-12, rtol=1e-10)
    finally:
        ctx.close()
