"""Native generic form reuse against independently created scalar/vector/mixed forms."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from pymhm.backends.workspace import (
    compile_form_bundle,
    create_workspace,
    update_workspace,
    workspace_key,
)
from pymhm.core.equations import LocalEquations, compile_local_equations

pytestmark = pytest.mark.fem


def _independent_assembly(form: Any) -> Any:
    """Use fresh native forms/buffers, independent of the workspace assembly owner."""
    from dolfinx import fem, la

    compiled = fem.form(form, dtype=np.float64)
    if len(form.arguments()) == 1:
        vector = fem.assemble_vector(compiled)
        vector.scatter_reverse(la.InsertMode.add)
        return vector.array.copy()
    matrix = fem.assemble_matrix(compiled)
    matrix.scatter_reverse()
    return sparse.csr_matrix(matrix.to_scipy(), copy=True)


@pytest.mark.parametrize("kind", ["scalar2", "vector2", "mixed2", "scalar3"])
def test_native_current_physical_data_geometry_and_owned_buffers(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A→B→A preserves arbitrary operators, sources, moments and oriented face loads."""
    pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    dim = 3 if kind == "scalar3" else 2
    domain = (
        mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2, mesh.CellType.hexahedron)
        if dim == 3
        else (mesh.create_unit_square(MPI.COMM_SELF, 2, 2))
    )
    cell = domain.basix_cell()
    if kind == "mixed2":
        element = basix.ufl.mixed_element(
            [
                basix.ufl.element("Lagrange", cell, 2, shape=(dim,)),
                basix.ufl.element("Lagrange", cell, 1),
            ]
        )
        space = fem.functionspace(domain, element)
        velocity, pressure = ufl.TrialFunctions(space)
        test_velocity, test_pressure = ufl.TestFunctions(space)
        field, test = velocity, test_velocity
    else:
        space = fem.functionspace(
            domain,
            basix.ufl.element(
                "Lagrange",
                cell,
                1,
                shape=(dim,) if kind == "vector2" else (),
            ),
        )
        field, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    coefficient_space = fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange",
            cell,
            1,
            shape=(dim, dim) if dim == 3 else (),
        ),
    )
    coefficient = fem.Function(coefficient_space)
    amplitude = fem.Constant(domain, np.float64(1.0))
    x = ufl.SpatialCoordinate(domain)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
    analytical = 2.0 + ufl.sin(1.7 * x[0] + 0.9 * x[1])
    operator = (
        (
            ufl.inner(coefficient * ufl.grad(field), ufl.grad(test))
            if dim == 3
            else coefficient * ufl.inner(ufl.grad(field), ufl.grad(test))
        )
        * analytical
        * dx
    )
    operator += amplitude * ufl.inner(field, test) * dx
    component = test[0] if kind in {"vector2", "mixed2"} else test
    load = amplitude * (1.0 + x[0] + x[1] * x[1]) * component * dx
    if kind == "mixed2":
        operator += (
            -pressure * ufl.div(test_velocity)
            - test_pressure * ufl.div(velocity)
            - pressure * test_pressure
        ) * dx
        load += (x[0] - x[1]) * test_pressure * dx
    entities, labels = [], []
    for axis in range(dim):
        for upper in (0, 1):
            facets = mesh.locate_entities_boundary(
                domain,
                dim - 1,
                lambda points, axis=axis, upper=upper: np.isclose(points[axis], upper),
            )
            entities.extend(facets)
            labels.extend([2 * axis + upper + 1] * len(facets))
    order = np.argsort(entities)
    tags = mesh.meshtags(
        domain,
        dim - 1,
        np.asarray(entities, dtype=np.int32)[order],
        np.asarray(labels, dtype=np.int32)[order],
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags, metadata={"quadrature_degree": 8})
    normal = ufl.FacetNormal(domain)
    declared = {"a": operator, "f": load, "moment": component * dx}
    for side in range(2 * dim):
        declared[f"face{side}"] = normal[side // 2] * component * ds(side + 1)
    count = 0
    native_compile = fem.compile_form

    def counting_compile(*args: Any, **kwargs: Any) -> Any:
        """Observe each data-independent kernel compilation, without affecting results."""
        nonlocal count
        count += 1
        return native_compile(*args, **kwargs)

    monkeypatch.setattr(fem, "compile_form", counting_compile)
    bundle = compile_form_bundle(declared, MPI.COMM_SELF)
    assert count == len(declared)
    original_geometry = domain.geometry.x.copy()
    original_key = workspace_key(domain, declared)
    saved: dict[str, Any] = {}
    with create_workspace(bundle, domain) as local:
        for index, value in enumerate((1.0, 2.3, 1.0)):
            geometry = original_geometry.copy()
            if index == 1:
                geometry[:, :dim] *= np.arange(1, dim + 1) * 0.3 + 0.7
                geometry[:, :dim] += 0.23

            def material(points: Any, value: float = value) -> Any:
                """Provide a nonperiodic positive scalar/tensor coefficient on physical points."""
                if dim == 2:
                    return value * (1.0 + 0.2 * points[0] + 0.1 * points[1])
                tensor = np.zeros((dim, dim, points.shape[1]))
                for i in range(dim):
                    tensor[i, i] = value * (1.0 + 0.3 * i + 0.1 * points[i])
                tensor[0, 1] = tensor[1, 0] = 0.07 * value
                return tensor.reshape(dim * dim, -1)

            update_workspace(
                local,
                geometry=geometry,
                coefficients={coefficient: material},
                constants={amplitude: np.array(value)},
            )
            assert workspace_key(domain, declared) == original_key
            assembled = {name: local.assemble(name) for name in declared}
            for name, form in declared.items():
                expected = _independent_assembly(form)
                actual = assembled[name]
                assert_allclose(
                    actual.toarray() if sparse.issparse(actual) else actual,
                    expected.toarray() if sparse.issparse(expected) else expected,
                    rtol=1e-13,
                    atol=1e-13,
                )
                if index == 2:
                    assert_array_equal(
                        actual.toarray() if sparse.issparse(actual) else actual,
                        saved[name].toarray() if sparse.issparse(saved[name]) else saved[name],
                    )
            # Same physical affine data, independently solved on the actual current matrix.
            right = assembled["f"] + 0.13 * assembled["moment"]
            right += sum(0.07 * (side + 1) * assembled[f"face{side}"] for side in range(2 * dim))
            field_coefficients = sparse.linalg.spsolve(assembled["a"], right)
            independent_matrix = _independent_assembly(operator)
            independent_rhs = _independent_assembly(load) + 0.13 * _independent_assembly(
                component * dx
            )
            independent_rhs += sum(
                0.07 * (side + 1) * _independent_assembly(declared[f"face{side}"])
                for side in range(2 * dim)
            )
            assert_allclose(
                field_coefficients,
                sparse.linalg.spsolve(independent_matrix, independent_rhs),
                rtol=1e-12,
                atol=1e-12,
            )
            assert np.linalg.norm(assembled["a"] @ field_coefficients - right) < 1e-10
            if index == 0:
                saved = assembled
            elif index == 1:
                assert np.linalg.norm(assembled["a"].toarray() - saved["a"].toarray()) > 0.1
        assert count == len(declared)
        assert len(local.buffers) == len(declared)


def test_native_data_independent_bundle_binds_new_mesh_and_rejects_stale_spaces() -> None:
    """Compiled kernels accept explicit compatible spaces on a distinct physical mesh."""
    pytest.importorskip("dolfinx")
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    prototype = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    physical = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    physical.geometry.x[:, :2] = physical.geometry.x[:, :2] * 0.4 + 0.2
    source_space = fem.functionspace(prototype, ("Lagrange", 1))
    actual_space = fem.functionspace(physical, ("Lagrange", 1))
    source_coefficient, actual_coefficient = fem.Function(source_space), fem.Function(actual_space)
    source_constant = fem.Constant(prototype, np.float64(1.0))
    actual_constant = fem.Constant(physical, np.float64(2.0))
    source_coefficient.interpolate(lambda x: 1.0 + x[0])
    actual_coefficient.interpolate(lambda x: 1.0 + x[0])
    source_trial, source_test = ufl.TrialFunction(source_space), ufl.TestFunction(source_space)
    source = (
        source_coefficient * source_trial * source_test
        + source_constant * ufl.inner(ufl.grad(source_trial), ufl.grad(source_test))
    ) * ufl.dx
    bundle = compile_form_bundle({"a": source}, MPI.COMM_SELF)
    with pytest.raises(ValueError, match="belong to the bound mesh"):
        create_workspace(bundle, physical)
    with create_workspace(
        bundle,
        physical,
        space_map={0: actual_space, 1: actual_space},
        coefficient_map={source_coefficient: actual_coefficient},
        constant_map={source_constant: actual_constant},
    ) as local:
        trial, test = ufl.TrialFunction(actual_space), ufl.TestFunction(actual_space)
        actual_form = (
            actual_coefficient * trial * test
            + actual_constant * ufl.inner(ufl.grad(trial), ufl.grad(test))
        ) * ufl.dx
        assert_allclose(
            local.assemble("a").toarray(),
            _independent_assembly(actual_form).toarray(),
            rtol=1e-13,
            atol=1e-13,
        )
    with pytest.raises(ValueError, match="compiled finite element"):
        wrong = fem.functionspace(physical, ("Lagrange", 2))
        create_workspace(bundle, physical, space_map={0: wrong, 1: wrong})


def test_native_coordinate_element_and_precision_exclusions() -> None:
    """The admissible affine binary64 mesh excludes the next geometry degree/precision."""
    pytest.importorskip("dolfinx")
    import basix
    import basix.ufl
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    prototype = mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    space = fem.functionspace(prototype, ("Lagrange", 1))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    source = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
    bundle = compile_form_bundle({"a": source}, MPI.COMM_SELF)
    with create_workspace(bundle, prototype) as local:
        assert_allclose(local.assemble("a").toarray(), _independent_assembly(source).toarray())
    element = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 2, basix.LagrangeVariant.equispaced
    )
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 2, shape=(2,)))
    quadratic = mesh.create_mesh(MPI.COMM_SELF, np.arange(6).reshape(1, 6), element.points, domain)
    quadratic_space = fem.functionspace(quadratic, ("Lagrange", 1))
    with pytest.raises(ValueError, match="compiled coordinate element"):
        create_workspace(bundle, quadratic, space_map={0: quadratic_space, 1: quadratic_space})
    low_precision = mesh.create_unit_square(MPI.COMM_SELF, 1, 1, dtype=np.float32)
    with pytest.raises(ValueError, match="binary64"):
        create_workspace(bundle, low_precision)
    low_space = fem.functionspace(low_precision, ("Lagrange", 1))
    low_trial, low_test = ufl.TrialFunction(low_space), ufl.TestFunction(low_space)
    with pytest.raises(ValueError, match="binary64"):
        compile_form_bundle({"a": low_trial * low_test * ufl.dx}, MPI.COMM_SELF)
    low_constant = fem.Constant(prototype, np.float32(1.0))
    constant_bundle = compile_form_bundle({"a": low_constant * source}, MPI.COMM_SELF)
    with pytest.raises(ValueError, match="binary64"):
        create_workspace(constant_bundle, prototype)


@pytest.mark.parametrize("dim", [2, 3])
def test_native_neumann_kernel_moments_and_reconstruction_match_full_kkt(dim: int) -> None:
    """Workspace-backed LocalEquations preserve gauge and homogeneous/nonzero physical data."""
    pytest.importorskip("dolfinx")
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    domain = (
        mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
        if dim == 2
        else (mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2, mesh.CellType.hexahedron))
    )
    space = fem.functionspace(domain, ("Lagrange", 1))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    x, normal = ufl.SpatialCoordinate(domain), ufl.FacetNormal(domain)
    material = fem.Constant(domain, np.float64(1.0))
    source = fem.Constant(domain, np.float64(0.0))
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
    a = (
        material
        * (2.0 + x[0] * x[0] + 0.3 * x[1])
        * ufl.inner(ufl.grad(trial), ufl.grad(test))
        * dx
    )
    declared = {"a": a, "f": source * (1.0 + x[0]) * test * dx, "C": test * dx}
    # Coordinate-polynomial traces are independent oriented face data.
    for axis in range(dim):
        declared[f"B{axis}"] = normal[axis] * (1.0 + 0.2 * x[axis]) * test * ufl.ds
    bundle = compile_form_bundle(declared, MPI.COMM_SELF)
    original = domain.geometry.x.copy()
    saved: dict[str, Any] = {}
    with create_workspace(bundle, domain) as local:
        for index, amplitude in enumerate((1.0, 2.3, 1.0)):
            physical = original.copy()
            if index == 1:
                physical[:, :dim] = 0.7 * physical[:, :dim] + 0.21
            update_workspace(
                local,
                geometry=physical,
                constants={
                    material: np.array(amplitude),
                    source: np.array(0.7 if index == 1 else 0.0),
                },
            )
            A, f = local.assemble("a"), local.assemble("f")
            C = local.assemble("C")[:, None]
            B = np.column_stack([local.assemble(f"B{axis}") for axis in range(dim)])
            Z = np.ones((len(f), 1))
            problem = compile_local_equations(
                LocalEquations(
                    a=A,
                    L=f,
                    b=B,
                    c=-B.T,
                    dofs=np.arange(dim),
                    kernel=Z,
                    moments=C,
                )
            ).problem
            response = problem.condense("scipy")
            independent = sparse.bmat([[_independent_assembly(a), C], [C.T, None]], format="csc")
            right = np.vstack((np.column_stack((f, B)), np.zeros((1, dim + 1))))
            solution = sparse.linalg.spsolve(independent, right)[: len(f)]
            assert_allclose(response.source, solution[:, 0], rtol=1e-12, atol=1e-12)
            assert_allclose(response.lifts, solution[:, 1:], rtol=1e-12, atol=1e-12)
            assert np.linalg.norm(A @ Z) < 1e-12
            assert_allclose(C.T @ response.source, 0.0, atol=1e-13)
            assert_allclose(C.T @ response.lifts, 0.0, atol=1e-13)
            # Trace coefficients enforce the original Neumann compatibility,
            # while the retained constant prescribes a nonzero physical mean.
            trace = np.arange(dim, dtype=float) * 0.13 + 0.2
            pairing = (Z.T @ B).ravel()
            pivot = int(np.argmax(np.abs(pairing)))
            trace[pivot] += float((Z.T @ (f - B @ trace)).item()) / pairing[pivot]
            field = response.reconstruct(trace, np.array([0.37]))
            assert np.linalg.norm(A @ field + B @ trace - f) < 1e-10
            assert_allclose((C.T @ field).item(), 0.37 * (C.T @ Z).item(), atol=1e-13)
            values = {
                "A": A.toarray(),
                "B": B,
                "f": f,
                "C": C,
                "Z": Z,
                "E": response.retained_basis,
                "field": field,
            }
            if index == 0:
                saved = values
            elif index == 2:
                for name, value in values.items():
                    assert_array_equal(value, saved[name])
