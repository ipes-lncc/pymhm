"""Shared introductory support preserves fields, physical norms and executed coordinates."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from examples.introduction.scalar import (
    ScalarField,
    archive_cell,
    broken_quad_panel,
    compare_scalar_fields,
    dirichlet_solve,
    evaluate_bound_pressure,
    evaluate_incident_reference,
    evaluate_qk,
    evaluate_scalar,
    executed_array_digest,
    fine_line_profile,
    fine_quad_panel,
    grid_quadrature,
    load_spe10_layer,
    observed_rates,
    physical_errors,
    plot_convergence,
    plot_field_panels,
    plot_material,
    plot_reference_differences,
    plot_scalar_comparison,
    rectangle_panel,
    sample_field,
)
from pymhm.fem.scalar.quadrilateral import qk_space
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.fields import DiscreteField, FieldDefinition

ROOT = Path(__file__).resolve().parents[1]


def _linear(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independently declared exact linear pressure and constant raw gradient."""
    return points.sum(axis=1), np.ones_like(points)


def test_declared_quadrature_and_tensor_physical_norms() -> None:
    """Nonidentity material gives independent closed-form pressure/flux/energy norms."""
    points, weights = grid_quadrature((0.0, 1.0, 0.0, 1.0), (2, 3), order=4)

    def twice(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Double the exact pressure and gradient without using assembled data."""
        p, g = _linear(x)
        return 2 * p, 2 * g

    results = physical_errors(twice, _linear, np.diag([2.0, 3.0]), points, weights, batch_size=7)
    assert_allclose(weights.sum(), 1.0, atol=1e-12, rtol=1e-10)
    assert_allclose(
        [results["pressure_L2"], results["flux_L2"], results["flux_energy"]],
        np.sqrt([7 / 6, 13, 5]),
        atol=1e-12,
        rtol=1e-10,
    )
    assert_allclose(
        [results[key] for key in ("pressure_relative", "flux_relative", "energy_relative")], 1
    )


def test_triangular_fields_use_public_evaluation_with_original_incident_owners() -> None:
    """Quadratic evaluation, nested reference norms and fine-cell samples retain literal fields."""
    coarse, fine = TriangleMesh.unit_square(2), TriangleMesh.unit_square(4)
    _, coarse_nodes = nodal_space(coarse, 2)
    _, fine_nodes = nodal_space(fine, 2)
    field = ScalarField(coarse, 2, np.sum(coarse_nodes**2, axis=1))
    points = np.array([[0.12, 0.17], [0.8, 0.2], [0.5, 0.5]])
    pressure, gradient = evaluate_scalar(field, points)
    assert_allclose(pressure, np.sum(points**2, axis=1), atol=1e-12, rtol=1e-10)
    assert_allclose(gradient, 2 * points, atol=1e-12, rtol=1e-10)
    assert_array_equal(field.definition.coefficients, field.values)
    assert not field.definition.definition.basis_matrix.flags.writeable
    with pytest.raises(ValueError, match="outside"):
        evaluate_scalar(field, np.array([[2.0, 1.0]]))
    reference = ScalarField(fine, 2, np.sum(fine_nodes**2, axis=1))
    errors = compare_scalar_fields((field,), reference, lambda x: np.ones(len(x)), order=5)
    assert errors["pressure_L2"] < 1e-12
    assert errors["flux_L2"] < 1e-12
    samples = sample_field((field,), refinement=2)
    assert len(samples["cells"]) == 4 * len(coarse.cells)
    assert_allclose(
        samples["values"], np.sum(samples["points"] ** 2, axis=1), atol=1e-12, rtol=1e-10
    )
    assert_allclose(samples["gradient"], 2 * samples["points"], atol=1e-12, rtol=1e-10)
    with pytest.raises(ValueError, match="common"):
        sample_field(())


def test_cartesian_panels_profiles_and_material_sides_are_not_averaged() -> None:
    """Pixel-fitted incident cells give two distinct flux limits at a shared interface."""
    mesh = CartesianMacroMesh(1, 2)
    _, nodes = qk_space(mesh, 2)
    values = nodes.sum(axis=1)
    material = CartesianCellField(np.array([[1.0, 3.0]]), (1.0, 0.5))
    points = np.array([[0.2, 0.1], [0.7, 0.8]])
    assert_allclose(evaluate_qk(mesh, 2, values, points)[0], points.sum(axis=1))
    pressure_panel = fine_quad_panel(mesh, 2, values, material, "pressure")
    assert_allclose(pressure_panel[2], pressure_panel[0].sum(axis=1))
    flux_panel = fine_quad_panel(mesh, 2, values, material, "flux_magnitude")
    assert_allclose(flux_panel[2], np.repeat(np.sqrt(2) * np.array([1.0, 3.0]), 4))
    y, p, qy = fine_line_profile(mesh, 2, values, material, 0.25)
    assert np.isnan(y[[5, 11]]).all()
    assert_allclose(p[np.isfinite(p)], 0.25 + y[np.isfinite(y)])
    assert_allclose(qy[np.isfinite(qy)], np.repeat([-1.0, -3.0], 5))
    with pytest.raises(ValueError, match="inside"):
        fine_line_profile(mesh, 2, values, material, -1.0)
    with pytest.raises(ValueError, match="quantity"):
        fine_quad_panel(mesh, 2, values, material, "unknown")
    view = SimpleNamespace(mesh=mesh, portable_coefficients=values)
    solution = SimpleNamespace(field=lambda name: (view,))
    panel = broken_quad_panel(None, solution, 2, material, "pressure")
    assert_array_equal(panel[0], pressure_panel[0])
    assert_array_equal(panel[1], pressure_panel[1])
    assert_array_equal(panel[2], pressure_panel[2])


def test_named_macro_fields_and_explicit_essential_elimination() -> None:
    """Broken macro ownership is retained and arbitrary matrix lifting uses its given rows."""
    macro = CartesianMacroMesh(2, 1)
    fields = [
        SimpleNamespace(values_and_gradient=lambda x: (np.ones(len(x)), np.zeros_like(x))),
        SimpleNamespace(values_and_gradient=lambda x: (2 * np.ones(len(x)), np.zeros_like(x))),
    ]
    pressure, gradient = evaluate_bound_pressure(
        macro, tuple(fields), np.array([[0.2, 0.3], [0.7, 0.3]])
    )
    assert_array_equal(pressure, [1.0, 2.0])
    assert_array_equal(gradient, np.zeros((2, 2)))
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]])
    resolved = dirichlet_solve(matrix, np.array([0.0, 3.0]), np.array([0]), np.array([2.0]))
    assert_allclose(resolved, [2.0, 2.5])


def test_shared_graphics_preserve_panels_colorbars_and_convergence_labels() -> None:
    """The helper draws all original quantities and separate incident macroface overlays."""
    import matplotlib.pyplot as plt

    macro = CartesianMacroMesh(2)
    panel = rectangle_panel(macro, _linear, points_per_side=3)
    assert len(panel[0]) == 4 * 9
    for quantity in ("permeability_xx", "permeability_yy", "flux_magnitude"):
        rectangle_panel(
            macro, _linear, quantity=quantity, permeability=np.diag([2.0, 3.0]), points_per_side=3
        )
    figure = plot_field_panels(macro, {"Pressure": panel})
    assert len(figure.axes) == 2
    assert len(figure.axes[0].collections) == 2
    assert_array_equal(observed_rates([0.5, 0.25], [0.25, 0.0625]), [2.0])
    convergence = plot_convergence([0.5, 0.25], {"field": [0.25, 0.0625]})
    assert len(convergence.axes) == 2
    difference = plot_reference_differences(
        [0.5, 0.25],
        [
            {"pressure_relative": 0.25, "flux_relative": 0.5},
            {"pressure_relative": 0.0625, "flux_relative": 0.25},
        ],
    )
    assert "rate = 2.00" in difference.axes[0].get_legend().get_texts()[0].get_text()
    mesh = TriangleMesh.unit_square()
    _, nodes = nodal_space(mesh, 2)
    field = ScalarField(mesh, 2, nodes.sum(axis=1))
    material = plot_material(mesh, lambda x: np.full(len(x), 2.0), resolution=2)
    comparison = plot_scalar_comparison(
        mesh, (field,), field, lambda x: np.ones(len(x)), numerical_label="declared"
    )
    assert len(material.axes) == 2
    assert len(comparison.axes) == 12
    assert comparison.axes[1].get_title() == "Pressure: declared"
    plt.close("all")


def test_pinned_material_and_array_archive_contract() -> None:
    """The original complete SPE10 data and literal reconstruction arrays retain their digests."""
    material, record = load_spe10_layer(ROOT, 36)
    assert material.values.shape == (60, 220, 2, 2)
    assert record["selected_layer"]["layer_one_based"] == 36
    mesh = TriangleMesh.unit_square()
    _, nodes = nodal_space(mesh, 1)
    field = ScalarField(mesh, 1, nodes.sum(axis=1))
    arrays: dict[str, Any] = {}
    moment = np.arange(3.0)
    archive_cell(arrays, 0, field, moments=moment)
    assert_array_equal(arrays["moments_0"], moment)
    assert_array_equal(arrays["pressure_0"], field.values)
    assert executed_array_digest(moment) == executed_array_digest(moment.copy())
    assert executed_array_digest(moment, 2 * moment) != executed_array_digest(2 * moment, moment)
    with pytest.raises(TypeError, match="native descriptor"):
        archive_cell(arrays, 0, field, executed=field.definition)


@pytest.mark.fem
@pytest.mark.parametrize("kind", ["triangle", "quadrilateral"])
def test_native_mapping_field_archive_and_compiled_operator(kind: str, tmp_path: Path) -> None:
    """Actual UFL compilation and field snapshots retain the bound native coefficient map."""
    pytest.importorskip("basix")
    pytest.importorskip("dolfinx")
    import ufl

    from examples.introduction.scalar import native_scalar_space, save_scalar_report
    from pymhm.backends.spaces import bind_space
    from pymhm.core.equations import compile_form

    mesh = TriangleMesh.unit_square() if kind == "triangle" else CartesianMacroMesh(1)
    domain, space, mapping = native_scalar_space(mesh, 2)
    p, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})
    matrix = compile_form(ufl.inner(ufl.grad(p), ufl.grad(v)) * dx)
    assert np.linalg.norm(matrix @ np.ones(matrix.shape[0])) < 1e-12
    binding = bind_space(mesh, space)
    coefficients = binding.interpolate(lambda x: x[0] + x[1])
    executed = DiscreteField(
        FieldDefinition("pressure", descriptor=binding.descriptor()), coefficients
    )
    points = np.array([[0.12, 0.23], [0.62, 0.35]])
    assert_allclose(executed.evaluate(points), points.sum(axis=1), atol=1e-12, rtol=1e-10)
    assert_array_equal(mapping, executed.definition.descriptor.mapping)
    field = ScalarField(TriangleMesh.unit_square(), 1, np.array([0.0, 1.0, 1.0, 2.0]))
    arrays: dict[str, Any] = {}
    archive_cell(
        arrays,
        0,
        field,
        executed=executed,
        trace_binding=SimpleNamespace(trial_map=np.eye(2), dofs=np.array([1, 2])),
    )
    assert_array_equal(arrays["executed_pressure_mapping_0"], mapping)
    assert_array_equal(
        arrays["executed_pressure_basis_0"], executed.definition.descriptor.basis_matrix
    )
    assert_array_equal(arrays["trace_trial_map_0"], np.eye(2))
    binding.close()
    assert_allclose(executed.evaluate(points), points.sum(axis=1), atol=1e-12, rtol=1e-10)
    # Saving retains supplied scientific values, and records real checkout provenance.
    output = save_scalar_report(
        ROOT, "mshho_multiscale", {"field_error": 0.0}, arrays, {}, output_dir=tmp_path
    )
    assert (output / "report.json").is_file()
    provenance = json.loads((output / "report.json").read_text())
    assert set(provenance["support_sha256"]) == {
        "examples/introduction/scalar.py",
        "examples/field_sampling.py",
        "examples/introduction/provenance.py",
    }


@pytest.mark.fem
def test_mshho_notebook_constant_source_scales_the_physical_solution() -> None:
    """The visible P1 moments/P2 realization uses its declared source, including nonunit data."""
    pytest.importorskip("dolfinx")
    pytest.importorskip("basix")
    notebook = json.loads((ROOT / "notebooks/introduction/mshho_multiscale.ipynb").read_text())
    namespace: dict[str, Any] = {}
    for index in (1, 3):
        exec("".join(notebook["cells"][index]["source"]), namespace)
    namespace["macro"] = TriangleMesh.unit_square(1)
    namespace["local_refinement"] = 4
    for index in (6, 9, 11):
        exec("".join(notebook["cells"][index]["source"]), namespace)
    traces, pressures = [], []
    for source in (1.0, 2.75):
        namespace["source"] = source
        exec("".join(notebook["cells"][13]["source"]), namespace)
        solution, system = namespace["solution"], namespace["system"]
        traces.append(solution.trace.copy())
        pressures.append(np.concatenate([field.values for field in namespace["physical_fields"]]))
        for data in system.local_metadata:
            projected = data["R"].T @ data["F"]
            expected = np.r_[source, np.zeros(len(projected) - 1)]
            assert_allclose(projected, expected, atol=1e-12, rtol=1e-10)
    namespace.update(
        reference_dimensions=[],
        reference_refinement=[],
        errors_q10={},
        quadrature_difference=0.0,
        diagnostics={},
        basis_digests=[],
        form_equivalence={},
    )
    record_cell = ast.parse("".join(notebook["cells"][24]["source"]))
    assignment = ast.Module(body=[record_cell.body[0]], type_ignores=[])
    exec(compile(assignment, "notebook scientific record", "exec"), namespace)
    assert namespace["record"]["source"] == 2.75
    assert "=2.75," in namespace["record"]["physical_problem"]
    assert np.linalg.norm(pressures[0]) > 0
    assert_allclose(traces[1], 2.75 * traces[0], atol=1e-12, rtol=1e-10)
    assert_allclose(pressures[1], 2.75 * pressures[0], atol=1e-12, rtol=1e-10)


def test_reference_gradients_retain_opposite_incident_interface_sides() -> None:
    """A continuous piecewise linear cusp has two opposite gradients on its diagonal."""
    import matplotlib.pyplot as plt

    mesh = TriangleMesh.unit_square(1)
    _, nodes = nodal_space(mesh, 1)
    field = ScalarField(mesh, 1, abs(nodes[:, 0] - nodes[:, 1]))
    sampled = sample_field((field,), refinement=2)
    points = sampled["points"]
    pressure, gradient = evaluate_incident_reference(field, points, sampled["incident_centers"])
    assert_allclose(pressure, sampled["values"], atol=1e-12, rtol=1e-10)
    assert_allclose(gradient, sampled["gradient"], atol=1e-12, rtol=1e-10)
    interface = abs(points[:, 0] - points[:, 1]) < 1e-12
    assert len(np.unique(gradient[interface], axis=0)) == 2
    assert_array_equal(sampled["field_indices"], np.zeros(len(points), dtype=np.int64))
    assert_array_equal(sampled["incident_cells"], np.repeat([0, 1], len(points) // 2))
    figure = plot_scalar_comparison(
        mesh, (field,), field, lambda x: np.ones(len(x)), numerical_label="same field"
    )
    assert np.max(abs(figure.axes[5].collections[0].get_array())) < 1e-12
    plt.close(figure)


def test_essential_lifting_keeps_declared_boundary_coordinate_order() -> None:
    """A nonhomogeneous lifting cancels large boundary terms before adding a small term."""
    matrix = np.eye(4)
    matrix[3, :3] = [1e16, 1.0, -1e16]
    boundary = np.array([0, 2, 1])
    prescribed = np.ones(3)
    solution = dirichlet_solve(matrix, np.zeros(4), boundary, prescribed)
    # The exact boundary sum is 1; this supplied order cancels the large pair first.
    assert_allclose(solution, [1.0, 1.0, 1.0, -1.0], atol=1e-12, rtol=1e-10)
