"""Native space binding, independent nodal transport and portable field replay."""

from __future__ import annotations

import pickle
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm.backends.spaces import (
    bind_space,
    coefficient_map,
    create_native_mesh,
    evaluate_descriptor,
    evaluate_descriptor_data,
    evaluate_descriptor_gradient,
)
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.hexahedron import HexMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh

pytestmark = pytest.mark.fem


def _element(cell: str, degree: int, **kwargs: Any) -> Any:
    """Declare a real equispaced Basix/UFL nodal element."""
    pytest.importorskip("dolfinx")
    import basix
    import basix.ufl

    return basix.ufl.element(
        "Lagrange", cell, degree, lagrange_variant=basix.LagrangeVariant.equispaced, **kwargs
    )


def _nodes(mesh: Any, degree: int) -> np.ndarray[Any, Any]:
    """Read coordinates from the existing portable continuous topology owner."""
    if isinstance(mesh, CartesianMacroMesh):
        from pymhm.fem.scalar.quadrilateral import qk_space

        return qk_space(mesh, degree)[1]
    if isinstance(mesh, TriangleMesh):
        from pymhm.fem.scalar.triangle import nodal_space

        return nodal_space(mesh, degree)[1]
    from pymhm.fem.scalar.tetrahedron import tetra_nodal_space

    return tetra_nodal_space(mesh, degree)[1]


@pytest.mark.parametrize(
    "mesh,cell",
    [
        (CartesianMacroMesh(2), "quadrilateral"),
        (TriangleMesh.unit_square(2), "triangle"),
        (TetraMesh.unit_cube(), "tetrahedron"),
    ],
)
def test_reference_topology_transport_reproduces_scalar_and_vector_polynomials(
    mesh: Any,
    cell: str,
) -> None:
    """DOF maps reproduce independent polynomials through P3/Q3 in 2D and 3D."""
    dimension = mesh.points.shape[1]
    centers = mesh.points[mesh.cells].mean(axis=1)
    for degree in (1, 2, 3):
        for shape in ((), (dimension,)):
            binding = bind_space(mesh, _element(cell, degree, shape=shape))
            nodes = _nodes(mesh, degree)
            scalar = 1 + nodes[:, 0] ** degree + np.arange(2, dimension + 1) @ nodes[:, 1:].T
            expected = (
                np.column_stack([scalar + i for i in range(dimension)]).ravel() if shape else scalar
            )
            callback = (
                (
                    lambda x, degree=degree: np.array(
                        [
                            1 + x[0] ** degree + np.arange(2, dimension + 1) @ x[1:dimension] + i
                            for i in range(dimension)
                        ]
                    )
                )
                if shape
                else (
                    lambda x, degree=degree: (
                        1 + x[0] ** degree + np.arange(2, dimension + 1) @ x[1:dimension]
                    )
                )
            )
            values = binding.interpolate(callback)
            assert_allclose(binding.to_portable(values), expected, rtol=2e-10, atol=2e-10)
            assert_array_equal(binding.to_native(binding.to_portable(values)), values)
            columns = np.column_stack((values, 2 * values))
            assert_array_equal(binding.to_native(binding.to_portable(columns)), columns)
            expected_centers = (
                1 + centers[:, 0] ** degree + np.arange(2, dimension + 1) @ centers[:, 1:].T
            )
            if shape:
                expected_centers = np.column_stack([expected_centers + i for i in range(dimension)])
            assert_allclose(
                binding.evaluate(values, centers), expected_centers, rtol=2e-10, atol=2e-10
            )
            assert_allclose(
                binding.evaluate(values, centers, cells=np.arange(len(centers))),
                expected_centers,
                rtol=2e-10,
                atol=2e-10,
            )
            descriptor = pickle.loads(pickle.dumps(binding.descriptor()))
            assert not descriptor.mapping.flags.writeable
            assert_allclose(
                evaluate_descriptor(descriptor, values, centers),
                expected_centers,
                rtol=2e-10,
                atol=2e-10,
            )
            assert descriptor.basis_digest == binding.descriptor().basis_digest
            expected_gradient = np.zeros((len(centers), dimension))
            expected_gradient[:, 0] = degree * centers[:, 0] ** (degree - 1)
            expected_gradient[:, 1:] = np.arange(2, dimension + 1)
            if shape:
                expected_gradient = np.repeat(expected_gradient[:, None, :], dimension, axis=1)
            replay_values, replay_gradient = evaluate_descriptor_data(descriptor, values, centers)
            assert_array_equal(replay_values, evaluate_descriptor(descriptor, values, centers))
            assert_allclose(replay_gradient, expected_gradient, atol=2e-10, rtol=2e-10)
            assert_allclose(
                evaluate_descriptor_gradient(
                    descriptor, values, centers, cells=np.arange(len(centers))
                ),
                expected_gradient,
                atol=2e-10,
                rtol=2e-10,
            )
            borrowed = bind_space(mesh, binding.space)
            assert_array_equal(borrowed.mapping, binding.mapping)
            borrowed.close()
            assert binding.size == len(values)
            binding.close()
            binding.close()
            with pytest.raises(RuntimeError, match="closed"):
                _ = binding.space
            with pytest.raises(RuntimeError, match="closed"):
                _ = binding.mesh


def test_native_matrix_and_signed_test_rows_match_independent_portable_assembly() -> None:
    """Native UFL and existing portable operators agree in the declared nodal basis."""
    import ufl

    from pymhm.backends.forms import assemble_form
    from pymhm.fem.scalar.quadrilateral import quadrilateral_operators
    from pymhm.fem.scalar.tetrahedron import tetra_operators
    from pymhm.fem.scalar.triangle import scalar_operators

    for mesh, cell, operator in (
        (CartesianMacroMesh(2), "quadrilateral", quadrilateral_operators),
        (TriangleMesh.unit_square(2), "triangle", scalar_operators),
        (TetraMesh.unit_cube(), "tetrahedron", tetra_operators),
    ):
        binding = bind_space(mesh, _element(cell, 2))
        u, v = ufl.TrialFunction(binding.space), ufl.TestFunction(binding.space)
        dx = ufl.Measure("dx", domain=binding.mesh, metadata={"quadrature_degree": 8})
        matrix = assemble_form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
        load = assemble_form(3 * v * dx)
        kwargs = {"source": 3.0, "order": 6}
        expected, _, force = operator(mesh, 2, **kwargs)
        mapping = binding.mapping
        assert_allclose(
            matrix[mapping][:, mapping].toarray(), expected.toarray(), rtol=2e-10, atol=2e-10
        )
        assert_allclose(load[mapping], force, rtol=2e-10, atol=2e-10)
        # Independent asymmetric row data must retain the user-selected signs.
        coupling = np.column_stack((np.arange(len(mapping)), -2 * np.arange(len(mapping))))
        transformed = binding.transport_coupling(coupling)
        assert_array_equal(transformed[mapping], coupling)
        binding.close()


def test_mixed_components_and_discontinuous_one_sided_replay() -> None:
    """Taylor-Hood and DG pressure use their own parent maps and one-sided cells."""
    pytest.importorskip("dolfinx")
    import basix.ufl

    mesh = TriangleMesh.unit_square()
    centers = mesh.points[mesh.cells].mean(axis=1)
    for pressure_degree, discontinuous in ((1, False), (1, True), (0, True)):
        element = basix.ufl.mixed_element(
            [
                _element("triangle", 2, shape=(2,)),
                _element("triangle", pressure_degree, discontinuous=discontinuous),
            ]
        )
        binding = bind_space(mesh, element)
        velocity = np.column_stack((1 + _nodes(mesh, 2)[:, 0], 2 + _nodes(mesh, 2)[:, 1])).ravel()
        vector = binding.to_native(velocity, component=0)
        pressure_map = coefficient_map(binding, component=1)
        pressure = np.arange(len(pressure_map), dtype=float) + 10
        vector += binding.to_native(pressure, component=1)
        assert_array_equal(binding.to_portable(vector, component=0), velocity)
        assert_array_equal(binding.to_portable(vector, component=1), pressure)
        expected_velocity = np.column_stack((1 + centers[:, 0], 2 + centers[:, 1]))
        assert_allclose(
            binding.evaluate(vector, centers, component=0), expected_velocity, atol=2e-10
        )
        for component in (0, 1):
            descriptor = binding.descriptor(component=component)
            assert_allclose(
                evaluate_descriptor(pickle.loads(pickle.dumps(descriptor)), vector, centers),
                binding.evaluate(vector, centers, component=component),
                atol=2e-10,
            )
        b = np.column_stack((velocity, -velocity))
        assert_array_equal(
            binding.transport_coupling(b, component=0)[coefficient_map(binding, component=0)], b
        )
        with pytest.raises(ValueError, match="explicit nodal component"):
            _ = binding.mapping
        with pytest.raises(ValueError, match="mixed evaluation"):
            binding.evaluate(vector, centers)
        if pressure_degree == 0:
            shared = np.repeat([[0.5, 0.5]], 2, axis=0)
            assert_allclose(
                binding.evaluate(vector, shared, cells=[0, 1], component=1), pressure, atol=2e-10
            )
            assert_allclose(
                evaluate_descriptor(binding.descriptor(component=1), vector, shared, cells=[0, 1]),
                pressure,
                atol=2e-10,
            )
            assert_allclose(
                evaluate_descriptor_gradient(
                    binding.descriptor(component=1), vector, shared, cells=[0, 1]
                ),
                np.zeros((2, 2)),
                atol=2e-10,
            )
        elif discontinuous:
            coordinates = mesh.points[mesh.cells]
            pressure = np.array(
                [
                    10 * cell + (cell + 1) * nodes[:, 0] + (2 * cell + 3) * nodes[:, 1]
                    for cell, nodes in enumerate(coordinates)
                ]
            ).ravel()
            vector = binding.to_native(pressure, component=1)
            shared = np.repeat([[0.5, 0.5]], 2, axis=0)
            assert_allclose(
                evaluate_descriptor_gradient(
                    binding.descriptor(component=1), vector, shared, cells=[0, 1]
                ),
                [[1, 3], [2, 5]],
                atol=2e-10,
                rtol=2e-10,
            )
        binding.close()


def test_native_non_nodal_spaces_and_hex_geometry_do_not_guess_portable_maps() -> None:
    """H(div), nonuniform nodal variants and unowned hex numbering stay explicit."""
    pytest.importorskip("dolfinx")
    import basix
    import basix.ufl

    mesh = TriangleMesh.unit_square()
    for family in ("RT", "N1E"):
        binding = bind_space(mesh, basix.ufl.element(family, "triangle", 1))
        with pytest.raises(ValueError, match="nodal Lagrange"):
            _ = binding.mapping
        # The native field remains usable: no Piola transformation is guessed.
        assert np.isfinite(binding.evaluate(np.ones(binding.size), [[0.2, 0.1]])).all()
        binding.close()
    binding = bind_space(
        mesh,
        basix.ufl.element(
            "Lagrange", "triangle", 3, lagrange_variant=basix.LagrangeVariant.gll_warped
        ),
    )
    with pytest.raises(ValueError, match="equispaced"):
        _ = binding.mapping
    binding.close()
    hexahedra = HexMesh.unit_cube()
    binding = bind_space(hexahedra, _element("hexahedron", 1))
    values = binding.interpolate(lambda x: x[0] + 2 * x[1] + 3 * x[2])
    assert_allclose(binding.evaluate(values, [[0.2, 0.3, 0.4]]), [2.0], atol=2e-10)
    with pytest.raises(ValueError, match="hexahedral"):
        _ = binding.mapping
    binding.close()
    with pytest.raises(ValueError, match="native binding"):
        create_native_mesh(object())


def test_bound_layout_and_evaluation_reject_invalid_contracts() -> None:
    """Malformed topology, coefficient shapes, components and points fail clearly."""
    fem = pytest.importorskip("dolfinx.fem")

    mesh = TriangleMesh.unit_square()
    binding = bind_space(mesh, _element("triangle", 2))
    for action, expression in (
        (lambda: binding.to_portable(np.ones(binding.size - 1)), "flattened native"),
        (lambda: binding.to_native(np.ones(binding.size + 1)), "portable nodal"),
        (lambda: binding.transport_coupling(np.ones(binding.size)), "coupling rows"),
        (lambda: binding.transport_coupling(np.ones((binding.size + 1, 1))), "coupling rows"),
        (lambda: binding.evaluate(np.ones(binding.size), [[0.2, 0.3, 0.4]]), "geometric_dimension"),
        (lambda: binding.evaluate(np.ones(binding.size), [[2.0, 2.0]]), "outside the native mesh"),
        (lambda: binding.evaluate(np.ones(binding.size), [[0.2, 0.3]], cells=[0.5]), "one integer"),
        (lambda: binding.evaluate(np.ones(binding.size), [[0.2, 0.3]], cells=[]), "one integer"),
        (lambda: binding.evaluate(np.ones(binding.size), [[0.2, 0.3]], cells=[-1]), "cell outside"),
        (lambda: binding.evaluate(np.ones(binding.size), [[0.2, 0.3]], cells=[2]), "cell outside"),
        (lambda: binding.evaluate(np.ones((binding.size, 2)), [[0.2, 0.3]]), "coefficient vector"),
        (lambda: coefficient_map(binding, component=1), "component outside"),
        (lambda: coefficient_map(binding, component=-1), "component"),
        (
            lambda: evaluate_descriptor(
                binding.descriptor(), np.ones(binding.size + 1), [[0.2, 0.3]]
            ),
            "executed dimension",
        ),
    ):
        with pytest.raises(ValueError, match=expression):
            action()
    stale_cells = replace(binding, native_cells=binding.native_cells[::-1])
    with pytest.raises(ValueError, match="native incidence"):
        _ = stale_cells.mapping
    stale_cells.close()
    descriptor = binding.descriptor()
    assert not descriptor.mapping.flags.writeable
    assert not descriptor.basis_matrix.flags.writeable
    assert not descriptor.basis_points.flags.writeable
    assert (
        replace(descriptor, basis_digest=descriptor.basis_digest).basis_digest
        == descriptor.basis_digest
    )
    for kwargs, message in (
        ({"mapping": np.array([-1])}, "valid executed"),
        ({"mapping": np.array([binding.size])}, "valid executed"),
        ({"mapping": np.array([[0]])}, "valid executed"),
        ({"mapping": np.array([0, 0])}, "independent coordinates"),
        ({"basis_matrix": np.array([1.0])}, "independent coordinates"),
        ({"basis_points": np.array([1.0])}, "independent coordinates"),
        ({"basis_digest": "stale"}, "basis digest"),
        ({"mapping": descriptor.mapping[:-1], "basis_digest": ""}, "portable component dimension"),
        ({"mapping": descriptor.mapping.astype(float), "basis_digest": ""}, "integer executed"),
        ({"basis_matrix": descriptor.basis_matrix[:-1], "basis_digest": ""}, "polynomial space"),
        ({"basis_matrix": descriptor.basis_matrix[:, :-1], "basis_digest": ""}, "polynomial space"),
        ({"basis_points": descriptor.basis_points[:-1], "basis_digest": ""}, "polynomial space"),
        ({"basis_points": descriptor.basis_points[:, :1], "basis_digest": ""}, "polynomial space"),
        ({"degree": -1}, "field degree"),
        ({"size": -1}, "field native size"),
        ({"cell_type": "quadrilateral"}, "mesh topology"),
        ({"value_shape": (0,)}, "field value dimension"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(descriptor, **kwargs)
    with pytest.raises(ValueError, match="interpolation nodes differ"):
        replace(descriptor, basis_points=descriptor.basis_points + 1, basis_digest="")
    with pytest.raises(ValueError, match="equispaced"):
        replace(descriptor, basis_points=descriptor.basis_points + 0.01, basis_digest="")
    with pytest.raises(ValueError, match="degree-zero"):
        replace(descriptor, degree=0, discontinuous=False)
    for kwargs, message in (
        ({"discontinuous": 1}, "discontinuous"),
        ({"value_shape": [1]}, "value_shape"),
        ({"basis_digest": None}, "basis_digest"),
    ):
        with pytest.raises(TypeError, match=message):
            replace(descriptor, **kwargs)
    altered = TriangleMesh(mesh.points + 0.1, mesh.cells)
    with pytest.raises(ValueError, match="native coordinates"):
        bind_space(altered, binding.space)
    other = TriangleMesh.unit_square(2)
    with pytest.raises(ValueError, match="same input cells"):
        bind_space(other, binding.space)
    reversed_cells = TriangleMesh(mesh.points, mesh.cells[::-1])
    with pytest.raises(ValueError, match="native vertex numbering"):
        bind_space(reversed_cells, binding.space)
    native = create_native_mesh(mesh)
    matching = fem.functionspace(native, _element("triangle", 1))
    assert bind_space(mesh, matching).size == len(mesh.points)
    for position in (2, len(mesh.points)):
        unused = TriangleMesh(
            np.insert(mesh.points, position, [3, 3], axis=0),
            np.where(mesh.cells >= position, mesh.cells + 1, mesh.cells),
        )
        incomplete = bind_space(unused, _element("triangle", 1))
        assert incomplete.size == len(mesh.points)
        with pytest.raises(ValueError, match="distinct portable coordinates"):
            _ = incomplete.mapping
        incomplete.close()
    binding.close()


def test_executed_basis_replay_uses_archived_matrix_across_equivalent_rotations() -> None:
    """Field replay follows executed polynomial coordinates, even after a basis rotation."""
    from threadpoolctl import threadpool_limits

    from pymhm.fem.reference import tabulate_archived_nodal_basis

    mesh = TriangleMesh([[0.1, 0.2], [1.1, 0.3], [0.3, 1.2]], [[0, 1, 2]])
    binding = bind_space(mesh, _element("triangle", 2))
    values = binding.interpolate(lambda x: 1 + x[0] * x[1] + 2 * x[0])
    descriptor = binding.descriptor()
    assert_array_equal(
        descriptor.basis_matrix, binding.space.element.basix_element.coefficient_matrix
    )
    reference = np.array([[0.1, 0.2], [0.3, 0.4]])
    physical = (1 - reference.sum(axis=1))[:, None] * mesh.points[0] + reference @ mesh.points[1:]
    expected = binding.evaluate(values, physical)
    rng = np.random.default_rng(103)
    rotation, _ = np.linalg.qr(rng.normal(size=descriptor.basis_matrix.shape))
    rotated = replace(descriptor, basis_matrix=rotation @ descriptor.basis_matrix, basis_digest="")
    native_coefficients = rotation @ values
    assert descriptor.basis_digest != rotated.basis_digest
    with threadpool_limits(limits=1):
        original = evaluate_descriptor(descriptor, values, physical)
        rotated_values = evaluate_descriptor(rotated, native_coefficients, physical)
    with threadpool_limits(limits=4):
        assert_allclose(
            evaluate_descriptor(descriptor, values, physical), original, rtol=2e-12, atol=2e-12
        )
        assert_allclose(
            evaluate_descriptor(rotated, native_coefficients, physical),
            rotated_values,
            rtol=2e-12,
            atol=2e-12,
        )
    assert_allclose(original, expected, rtol=2e-10, atol=2e-10)
    assert_allclose(rotated_values, expected, rtol=2e-10, atol=2e-10)
    expected_gradient = np.column_stack((physical[:, 1] + 2, physical[:, 0]))
    with threadpool_limits(limits=1):
        original_gradient = evaluate_descriptor_gradient(descriptor, values, physical)
    with threadpool_limits(limits=4):
        rotated_gradient = evaluate_descriptor_gradient(rotated, native_coefficients, physical)
    assert_allclose(original_gradient, expected_gradient, rtol=2e-10, atol=2e-10)
    assert_allclose(rotated_gradient, original_gradient, rtol=2e-12, atol=2e-12)
    for cell, degree, matrix, points, message in (
        ("unknown", 2, np.eye(3), reference, "reference cell"),
        ("triangle", 2, np.array([1]), reference, "finite real matrix"),
        ("triangle", 2, np.array([[1j]]), reference, "finite real matrix"),
        ("triangle", 2, np.array([[np.nan]]), reference, "finite real matrix"),
        ("triangle", 2, np.eye(3), reference, "polynomial dimension"),
    ):
        with pytest.raises(ValueError, match=message):
            tabulate_archived_nodal_basis(cell, degree, matrix, points)
    binding.close()
