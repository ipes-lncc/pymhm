"""Scientific trusted replay uses the literal persisted field coordinate contract."""

import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.field_archive import field_archive_arrays, load_trusted_field
from examples.plot_rad3d import slice_fields
from pymhm.fem.hdiv.rt import rt_basis, rt_dofs
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.fields import DiscreteField
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.piola import hdiv_field, piola_field


def test_cross_section_evaluation_without_plotting_dependencies() -> None:
    """Archived physical fields remain evaluable when Matplotlib imports are blocked."""
    code = """
import builtins
import numpy as np

original_import = builtins.__import__

def without_matplotlib(name, *args, **kwargs):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise ModuleNotFoundError("Matplotlib is unavailable in this numerical environment")
    return original_import(name, *args, **kwargs)

builtins.__import__ = without_matplotlib

from examples.plot_rad3d import slice_fields
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.postprocessing.fields import DiscreteField
from pymhm.postprocessing.nodal import nodal_field

mesh = TetraMesh.unit_cube(1)
field = DiscreteField(nodal_field("scalar", mesh, 1), np.ones(len(mesh.points)))
polygons, actual, expected = slice_fields(mesh, field.coefficients, 1, 0.37, field=field)
assert len(polygons) > 0
assert len(polygons) == len(actual) == len(expected)
np.testing.assert_allclose(actual[:, 0], 1, atol=1e-12, rtol=1e-10)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


def test_archived_one_sided_field_and_gradient_replay_without_basis_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the executed basis/map literally, even when fresh nodal builders are unavailable."""
    mesh = TetraMesh.unit_cube(1)
    _, nodes = tetra_nodal_space(mesh, 3)
    field = DiscreteField(nodal_field("scalar", mesh, 3), 1 + nodes @ np.array([1, 2, -1]))
    path = tmp_path / "field.npz"
    np.savez_compressed(path, **field_archive_arrays((field,)))

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("replay must use the persisted basis and topology")

    monkeypatch.setattr("pymhm.postprocessing.nodal.simplex_lagrange_basis", unavailable)
    monkeypatch.setattr("examples.plot_rad3d.tetra_nodal_space", unavailable)
    monkeypatch.setattr("examples.plot_rad3d.tetra_basis", unavailable)
    with np.load(path) as archive:
        replay = load_trusted_field(archive, 0)
        assert replay is not None
        assert replay.basis_digest == field.basis_digest
        np.testing.assert_array_equal(replay.definition.basis_matrix, field.definition.basis_matrix)
        points = mesh.points[mesh.cells].mean(axis=1)
        values = replay.evaluate(points, cells=np.arange(len(mesh.cells)))
        gradient = replay.gradient(points, cells=np.arange(len(mesh.cells)))
        np.testing.assert_allclose(values, 1 + points @ [1, 2, -1], atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(
            gradient, np.broadcast_to([1, 2, -1], gradient.shape), atol=1e-12, rtol=1e-10
        )
        polygons, physical, _ = slice_fields(mesh, replay.coefficients, 3, 0.37, field=replay)
        assert len(polygons) == len(physical)
        assert np.isfinite(physical).all()


def test_missing_historical_basis_and_mismatched_contract_are_explicit() -> None:
    """Distinguish historical coefficient-only data from a corrupt recorded definition."""
    assert load_trusted_field({}, 0) is None
    mesh = TetraMesh.unit_cube(1)
    _, nodes = tetra_nodal_space(mesh, 1)
    field = DiscreteField(nodal_field("scalar", mesh, 1), np.ones(len(nodes)))
    arrays = field_archive_arrays((field,))
    for key, value in [
        ("scalar_0_basis_matrix", np.zeros_like(arrays["scalar_0_basis_matrix"])),
        ("scalar_0_basis_digest", np.asarray("wrong")),
    ]:
        with pytest.raises(ValueError, match="executed basis"):
            load_trusted_field(arrays | {key: value}, 0)
    arrays["scalar_0_definition"] = np.frombuffer(pickle.dumps(None), dtype=np.uint8)
    with pytest.raises(TypeError, match="FieldDefinition"):
        load_trusted_field(arrays, 0)


def test_composed_piola_archive_replays_literal_orientation_without_single_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist moment and physical Piola maps, including oppositely oriented shared edges."""
    mesh = TriangleMesh.unit_square()
    dofs = rt_dofs(mesh, 2)
    coefficients = np.linspace(-0.4, 0.8, dofs.max() + 1)
    definition = hdiv_field("velocity", mesh, "RT", degree=2)
    divergence = hdiv_field("velocity_divergence", mesh, "RT", degree=2, divergence=True)
    evaluator = definition.evaluator
    assert evaluator is not None
    assert not np.array_equal(
        evaluator.orientation,
        np.broadcast_to(np.eye(dofs.shape[1]), evaluator.orientation.shape),
    )
    with pytest.raises(TypeError, match="single basis matrix"):
        _ = definition.basis_matrix
    fields = (DiscreteField(definition, coefficients), DiscreteField(divergence, coefficients))
    arrays = field_archive_arrays(fields, prefix="velocity")
    assert "velocity_0_basis_matrix" not in arrays
    assert "velocity_1_basis_matrix" not in arrays
    barycentric = np.array([[0.2, 0.3, 0.5]])
    points = np.einsum("qi,eia->eqa", barycentric, mesh.points[mesh.cells]).reshape(-1, 2)
    basis, div = rt_basis(mesh, 2, barycentric)
    expected = np.einsum("eqia,ei->eqa", basis, coefficients[dofs]).reshape(-1, 2)
    expected_div = np.einsum("eqi,ei->eq", div, coefficients[dofs]).ravel()
    orientation = evaluator.orientation.copy()
    orientation[0] *= -1
    changed = piola_field(
        "velocity",
        mesh,
        reference_basis=evaluator.reference_basis,
        dofs=dofs,
        orientation=orientation,
    )
    assert changed.basis_digest != definition.basis_digest
    with pytest.raises(ValueError, match="executed basis"):
        load_trusted_field(
            arrays | {"velocity_0_basis_matrix": np.eye(dofs.shape[1])}, 0, prefix="velocity"
        )
    path = tmp_path / "velocity.npz"
    np.savez_compressed(path, **arrays)

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("replay must consume the executed polynomial and moment maps")

    monkeypatch.setattr("pymhm.postprocessing.piola.create_reference_element", unavailable)
    monkeypatch.setattr("pymhm.postprocessing.piola.rt_moment_coefficients", unavailable)
    for threads in (1, 2):
        with threadpool_limits(threads), np.load(path) as archive:
            value = load_trusted_field(archive, 0, prefix="velocity")
            derivative = load_trusted_field(archive, 1, prefix="velocity")
            assert value is not None and derivative is not None
            assert value.basis_digest == fields[0].basis_digest
            assert derivative.basis_digest == fields[1].basis_digest
            owners = np.arange(len(mesh.cells))
            np.testing.assert_allclose(
                value.evaluate(points, cells=owners), expected, atol=1e-12, rtol=1e-10
            )
            np.testing.assert_allclose(
                derivative.evaluate(points, cells=owners), expected_div, atol=1e-12, rtol=1e-10
            )
