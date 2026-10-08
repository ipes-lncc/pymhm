"""Native boundary forms match existing signed trace operators in executed bases."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import Equation, solve
from pymhm.core.context import GlobalContext, LocalContext
from pymhm.core.equations import compile_form
from pymhm.core.spaces import MeshHierarchy, bind_interface
from pymhm.fem.scalar.quadrilateral import quadrilateral_trace_coupling
from pymhm.fem.scalar.triangle import trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.triangle import TriangleMesh

pytestmark = pytest.mark.fem


def _context(macro, faces, fine, cell=0, components=1):
    skeleton = SkeletonSpace(macro, tuple(faces for _ in macro.faces), components)
    hierarchy = MeshHierarchy(macro, lambda _: fine, items=(cell,))
    return LocalContext(
        GlobalContext(hierarchy, bind_interface(skeleton, convention="normal"), (0,)), cell
    )


@pytest.mark.parametrize(
    "geometry,volume_degree,trace_degree,segments,continuous",
    [
        ("triangle", 1, 2, 2, False),
        ("rectangle", 2, 1, 2, True),
        ("rectangle", 1, 3, 1, False),
        ("triangle", 3, 0, 1, False),
    ],
)
def test_native_trace_forms_preserve_degree_partition_and_signed_coordinate_contract(
    geometry, volume_degree, trace_degree, segments, continuous
):
    pytest.importorskip("dolfinx")
    import ufl

    macro = TriangleMesh.unit_square() if geometry == "triangle" else CartesianMacroMesh(2, 1)
    cell = 1
    fine = macro.submesh(cell, 4)
    ctx = _context(
        macro, FaceSpace.uniform(trace_degree, segments, continuous=continuous), fine, cell
    )
    try:
        native = ctx.native_space(degree=volume_degree)
        u, v = ufl.TrialFunction(native.space), ufl.TestFunction(native.space)
        b = ctx.trace_pairings(lambda phi, ds: phi * v * ds)
        c = ctx.trace_pairings(lambda phi, ds: -2 * phi * v * ds, axis="rows")
        declaration = ctx.equations(a=u * v * ufl.dx, L=0, b=b, c=c)
        owner = trace_coupling if geometry == "triangle" else quadrilateral_trace_coupling
        expected = native.transport_coupling(
            owner(macro, cell, fine, ctx.global_context.interface.space, volume_degree)
        )
        assert_allclose(declaration.b, expected, rtol=1e-10, atol=1e-11)
        assert_allclose(declaration.c, -2 * expected.T, rtol=1e-10, atol=1e-11)
        local = compile_form(b, (native.size, ctx.binding.trial_size))
        assert_allclose(local @ ctx.binding.trial_map, expected, rtol=1e-10, atol=1e-11)
        if geometry == "triangle":
            from pymhm.backends.traces import trace_pairings
            from pymhm.methods.three_field import PressureTraceSpace

            # A global continuous pressure trace and a private conormal basis
            # share geometry without sharing coefficients or trial dimensions.
            pressure = bind_interface(PressureTraceSpace.uniform(macro))
            secondary = bind_interface(
                SkeletonSpace(macro, tuple(FaceSpace.uniform(1, 2) for _ in macro.faces)),
                convention="value",
            )
            alternate = LocalContext(
                GlobalContext(MeshHierarchy(macro, lambda _: fine, items=(cell,)), pressure, (0,)),
                cell,
            )
            plain = _context(macro, FaceSpace.uniform(1, 2), fine, cell)
            try:
                alternate.native_space(native.space)
                width = secondary.binding(cell).trial_size
                actual = alternate.trace_pairings(lambda phi, ds: phi * v * ds, interface=secondary)
                reference = trace_pairings(plain, native, lambda phi, ds: phi * v * ds)
                expected_unsigned = compile_form(reference, (native.size, width))
                assert_allclose(
                    compile_form(actual, (native.size, width)),
                    expected_unsigned,
                    rtol=1e-10,
                    atol=1e-11,
                )
                rows = alternate.trace_pairings(
                    lambda phi, ds: -3 * phi * v * ds, axis="rows", interface=secondary
                )
                assert_allclose(
                    compile_form(rows, (width, native.size)),
                    -3 * expected_unsigned.T,
                    rtol=1e-10,
                    atol=1e-11,
                )
                assert alternate.global_context.interface is pressure
                assert alternate.binding.trial_size != width
                wrong_macro = bind_interface(SkeletonSpace(TriangleMesh.unit_square()))
                with pytest.raises(ValueError, match="same macro mesh"):
                    alternate.trace_pairings(lambda phi, ds: phi * v * ds, interface=wrong_macro)
                pressure_forms = alternate.trace_pairings(lambda phi, ds: phi * v * ds)
                pressure_matrix = compile_form(
                    pressure_forms, (native.size, alternate.binding.trial_size)
                )
                assert np.linalg.norm(pressure_matrix) > 0
            finally:
                plain.close()
                alternate.close()
    finally:
        ctx.close()


@pytest.mark.parametrize("mixed", [False, True])
def test_vector_and_mixed_local_forms_use_component_interleaved_trace_basis(mixed):
    pytest.importorskip("dolfinx")
    import basix
    import basix.ufl
    import ufl

    macro = TriangleMesh.unit_square()
    cell = 1
    fine = macro.submesh(cell, 2)
    ctx = _context(macro, FaceSpace.uniform(1), fine, cell, components=2)
    try:
        velocity = basix.ufl.element(
            "Lagrange", "triangle", 1, shape=(2,), lagrange_variant=basix.LagrangeVariant.equispaced
        )
        element = (
            basix.ufl.mixed_element([velocity, basix.ufl.element("Lagrange", "triangle", 1)])
            if mixed
            else velocity
        )
        native = ctx.native_space(element)
        if mixed:
            u, p = ufl.TrialFunctions(native.space)
            v, q = ufl.TestFunctions(native.space)
            a = (ufl.inner(u, v) + p * q) * ufl.dx
        else:
            u, v = ufl.TrialFunction(native.space), ufl.TestFunction(native.space)
            a = ufl.inner(u, v) * ufl.dx
        declaration = ctx.equations(
            a=a,
            L=0,
            b=ctx.trace_pairings(lambda phi, ds: ufl.inner(phi, v) * ds),
            c=ctx.trace_pairings(lambda phi, ds: -ufl.inner(phi, v) * ds, axis="rows"),
        )
        scalar = trace_coupling(macro, cell, fine, ctx.global_context.interface.space, 1)
        expected = native.transport_coupling(
            np.kron(scalar, np.eye(2)), component=0 if mixed else None
        )
        assert_allclose(declaration.b, expected, atol=1e-11, rtol=1e-10)
        assert_allclose(declaration.c, -expected.T, atol=1e-11, rtol=1e-10)
    finally:
        ctx.close()


def test_translated_local_face_forms_share_a_kernel_signature_with_distinct_geometry_values():
    pytest.importorskip("dolfinx")
    import ufl

    macro = CartesianMacroMesh(2, 1)
    signatures = []
    for cell in range(2):
        ctx = _context(macro, FaceSpace.uniform(2), macro.submesh(cell, 2), cell)
        try:
            native = ctx.native_space()
            v = ufl.TestFunction(native.space)
            pairings = ctx.trace_pairings(lambda phi, ds, v=v: phi * v * ds)
            signatures.append(tuple(form.signature() for form in pairings.forms))
            owner = quadrilateral_trace_coupling(
                macro, cell, ctx.mesh, ctx.global_context.interface.space, 1
            )
            actual = (
                compile_form(pairings, (native.size, ctx.binding.trial_size))
                @ ctx.binding.trial_map
            )
            assert_allclose(actual, native.transport_coupling(owner), atol=1e-11, rtol=1e-10)
        finally:
            ctx.close()
    assert signatures[0] == signatures[1]


@pytest.mark.parametrize("origin,scale", [(0.0, 1e-12), (1e6, 1.0)])
def test_native_macroface_binding_uses_dimensionally_consistent_geometry_tolerances(origin, scale):
    pytest.importorskip("dolfinx")
    import ufl

    macro = CartesianMacroMesh(1, bounds=(origin, origin + scale, origin, origin + scale))
    ctx = _context(macro, FaceSpace(), macro.submesh(0, 4))
    try:
        native = ctx.native_space()
        v = ufl.TestFunction(native.space)
        forms = ctx.trace_pairings(lambda phi, ds: phi * v * ds)
        actual = compile_form(forms, (native.size, ctx.binding.trial_size)) @ ctx.binding.trial_map
        expected = native.transport_coupling(
            quadrilateral_trace_coupling(macro, 0, ctx.mesh, ctx.global_context.interface.space, 1)
        )
        assert_allclose(actual / scale, expected / scale, atol=1e-11, rtol=1e-10)
    finally:
        ctx.close()


def test_native_trace_adapter_rejects_unresolved_breaks_and_unassociated_geometry():
    pytest.importorskip("dolfinx")
    import ufl

    for breaks, fine, expected in [
        ((0.0, 0.5, 1.0), CartesianMacroMesh(1), "resolve"),
        ((0.0, 0.3, 1.0), CartesianMacroMesh(4), "align"),
        ((0.0, 1.0), CartesianMacroMesh(1, bounds=(0.1, 0.9, 0.0, 1.0)), "cover each"),
    ]:
        macro = CartesianMacroMesh(1)
        faces = FaceSpace(breaks, (0,) * (len(breaks) - 1))
        ctx = _context(macro, faces, fine)
        try:
            native = ctx.native_space()
            v = ufl.TestFunction(native.space)
            with pytest.raises(ValueError, match=expected):
                ctx.trace_pairings(lambda phi, ds, v=v: phi * v * ds)
        finally:
            ctx.close()


def test_trace_capability_validation_needs_no_native_import_or_coordinate_guess():
    from pymhm.backends.traces import trace_pairings

    with pytest.raises(ValueError, match="axis"):
        trace_pairings(None, None, lambda phi, ds: None, axis="invalid")
    with pytest.raises(TypeError, match="callable"):
        trace_pairings(None, None, None)
    context = SimpleNamespace(global_context=SimpleNamespace(interface=SimpleNamespace()))
    with pytest.raises(TypeError, match="capability"):
        trace_pairings(context, None, lambda phi, ds: None)
    with pytest.raises(TypeError, match="capability"):
        trace_pairings(context, None, lambda phi, ds: None, interface=SimpleNamespace())


@pytest.mark.parametrize("components,degree", [(1, 0), (2, 2)])
def test_global_native_interface_form_matches_independent_face_quadrature(components, degree):
    pytest.importorskip("dolfinx")
    import ufl

    from pymhm.backends.traces import interface_equation

    macro = CartesianMacroMesh(2, 1)
    faces = tuple(
        FaceSpace.uniform(
            degree - face % 3 if degree else 0,
            2,
            continuous=bool(degree and face % 3 != 2 and face % 2),
        )
        for face in range(len(macro.faces))
    )
    skeleton = SkeletonSpace(macro, faces, components)
    hierarchy = MeshHierarchy(macro, tuple(macro.submesh(cell, 2) for cell in range(2)))
    context = GlobalContext(hierarchy, bind_interface(skeleton, convention="normal"), (1, 2))

    def forms(lam, mu, dx):
        x = ufl.SpatialCoordinate(ufl.domain.extract_unique_domain(lam))
        value = 1 + x[0] - 2 * x[1] if components == 1 else ufl.as_vector([1 + x[0], 2 - x[1]])
        return ufl.inner(lam, mu) * dx, ufl.inner(value, mu) * dx

    equation = interface_equation(context, forms)
    expected_a = np.zeros((context.size, context.size))
    expected_load = np.zeros(context.size)
    for face, declaration in enumerate(faces):
        parameter, weights = declaration.quadrature(6)
        start, end = macro.points[macro.faces[face]]
        points = start + parameter[:, None] * (end - start)
        weights *= macro.lengths[face]
        values = declaration.evaluate(parameter)
        indices = skeleton.dofs(face)
        expected_a[np.ix_(indices, indices)] = np.kron(
            values.T @ (weights[:, None] * values), np.eye(components)
        )
        prescribed = (
            (1 + points[:, 0] - 2 * points[:, 1])[:, None]
            if components == 1
            else np.column_stack((1 + points[:, 0], 2 - points[:, 1]))
        )
        expected_load[indices] = (values.T @ (weights[:, None] * prescribed)).ravel()
    assert_allclose(equation.a.toarray(), expected_a, atol=1e-11, rtol=1e-10)
    assert_allclose(equation.L, expected_load, atol=1e-11, rtol=1e-10)


def test_global_robin_form_and_local_signed_equations_match_original_monolithic_system():
    pytest.importorskip("dolfinx")

    from pymhm.backends.traces import interface_equation
    from pymhm.core.context import bind_problem

    macro = CartesianMacroMesh(2, 1)
    skeleton = SkeletonSpace(macro)
    interface = bind_interface(skeleton, convention="normal")
    hierarchy = MeshHierarchy(macro, tuple(macro.submesh(cell, 1) for cell in range(2)))

    def local(ctx):
        size = ctx.binding.trial_size
        return ctx.equations(
            a=[[2.0]], L=[1.0 + ctx.cell], b=np.ones((1, size)), c=-np.ones((size, 1))
        )

    def global_forms(context):
        return interface_equation(
            context, lambda lam, mu, dx: Equation(3 * lam * mu * dx, 2 * mu * dx)
        )

    problem = bind_problem(hierarchy, interface, local, global_equation=global_forms)
    result = solve(problem)
    size = skeleton.size
    original = np.zeros((size + 2, size + 2))
    load = np.zeros(size + 2)
    original[2:, 2:] = problem.global_equation.a.toarray()
    load[2:] = problem.global_equation.L
    for cell in range(2):
        binding = interface.binding(cell)
        indices = 2 + binding.dofs
        coupling = np.diag(binding.trial_map)
        original[cell, cell] = 2
        original[cell, indices] = coupling
        original[indices, cell] = -coupling
        load[cell] = 1 + cell
    expected = np.linalg.solve(original, load)
    assert_allclose(
        np.r_[np.concatenate(result.fields), result.trace], expected, atol=1e-11, rtol=1e-10
    )


def test_global_interface_capability_and_builder_results_are_explicitly_checked():
    from pymhm.backends.traces import interface_equation

    with pytest.raises(TypeError, match="callable"):
        interface_equation(None, None)
    with pytest.raises(TypeError, match="capability"):
        interface_equation(SimpleNamespace(interface=SimpleNamespace()), lambda *args: None)
    pytest.importorskip("dolfinx")
    macro = CartesianMacroMesh(1)
    skeleton = SkeletonSpace(macro)
    context = GlobalContext(MeshHierarchy(macro, [macro]), bind_interface(skeleton), (0,))
    with pytest.raises(TypeError, match="return Equation"):
        interface_equation(context, lambda *args: None)


def test_native_trace_binding_rejects_unsupported_dimension_and_local_boundary_holes():
    pytest.importorskip("dolfinx")
    import ufl

    from pymhm.backends.traces import interface_equation
    from pymhm.meshes.tetrahedron import TetraMesh

    tetra = TetraMesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 1, 2, 3]])
    )
    ctx = _context(tetra, FaceSpace(), tetra)
    try:
        native = ctx.native_space()
        v = ufl.TestFunction(native.space)
        with pytest.raises(ValueError, match="planar"):
            ctx.trace_pairings(lambda phi, ds: phi * v * ds)
        with pytest.raises(ValueError, match="planar"):
            interface_equation(ctx.global_context, lambda *args: None)
    finally:
        ctx.close()
    macro = CartesianMacroMesh(1)
    fine = TriangleMesh.unit_square(3)
    punctured = TriangleMesh(fine.points, np.delete(fine.cells, [8, 9], axis=0))
    ctx = _context(macro, FaceSpace(), punctured)
    try:
        native = ctx.native_space()
        v = ufl.TestFunction(native.space)
        with pytest.raises(ValueError, match="cover the local"):
            ctx.trace_pairings(lambda phi, ds: phi * v * ds)
    finally:
        ctx.close()


def test_native_trace_pairings_reject_a_repeated_macroface_incidence():
    pytest.importorskip("dolfinx")
    import ufl

    from pymhm.backends.traces import trace_pairings

    macro = CartesianMacroMesh(1)
    ctx = _context(macro, FaceSpace(), macro)
    try:
        native = ctx.native_space()
        v = ufl.TestFunction(native.space)
        invalid_macro = SimpleNamespace(
            points=macro.points,
            faces=macro.faces,
            cell_faces=[np.r_[macro.cell_faces[0], macro.cell_faces[0, 0]]],
        )
        invalid = SimpleNamespace(
            global_context=SimpleNamespace(
                interface=SimpleNamespace(space=SkeletonSpace(invalid_macro))
            ),
            cell=0,
            macro=invalid_macro,
        )
        with pytest.raises(ValueError, match="multiple macrofaces"):
            trace_pairings(invalid, native, lambda phi, ds: phi * v * ds)
    finally:
        ctx.close()
