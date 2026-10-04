"""Scientific input identity, material partition and boundary conventions for L11."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from examples.transport_random_problem import (
    darcy_pressure,
    generate_permeability,
    load_realization,
    macro_mesh,
    natural_faces,
    require_fitted_partition,
    save_realization,
)


def test_archived_selected_material_replays_actual_values_and_seed(tmp_path):
    """Persist the whole 64-by-16 input, with no clipping or target extrema fitting."""
    path = tmp_path / "permeability.json"
    record = save_realization(path)
    restored, replay_record = load_realization(path)
    generated = generate_permeability()
    assert_array_equal(restored.values, generated.values)
    assert replay_record == record
    assert restored.values.shape == (64, 16)
    assert 1e-2 < restored.values.min() < restored.values.max() < 10
    assert not restored.values.flags.writeable
    assert not np.array_equal(generate_permeability(0).values, restored.values)
    centers = np.stack(
        np.meshgrid((np.arange(64) + 0.5) * 3 / 64, (np.arange(16) + 0.5) / 16, indexing="ij"),
        axis=-1,
    )
    assert_array_equal(restored(centers.reshape(-1, 2)), restored.values.ravel())
    assert_array_equal(restored(np.array([[3.0, 1.0]])), restored.values[-1:, -1])


@pytest.mark.parametrize(
    "key,value",
    [
        ("axes", ["y", "x"]),
        ("primary_source", {"doi": "10.1137/130938499", "section": "5.3"}),
        ("scope", "Literal reproduction of the historical random realization"),
        ("spacing", [1 / 64, 3 / 16]),
        ("units", "mD/ft"),
        ("shape", [16, 64]),
        ("darcy", {"pressure_left": 1.0}),
        ("transport", {"longitudinal": 1.0}),
        ("permeability_sha256", "0" * 64),
    ],
)
def test_changed_input_contract_or_digest_is_rejected(tmp_path, key, value):
    """Scientific data cannot silently acquire different axes, drive or identities."""
    path = tmp_path / "permeability.json"
    record = save_realization(path)
    record[key] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="contract|digest"):
        load_realization(path)


def test_changed_material_byte_and_invalid_seed_are_rejected(tmp_path):
    """Archive bytes rather than a regenerated seed define the represented material."""
    path = tmp_path / "permeability.json"
    record = save_realization(path)
    record["values"][0][0] *= 1.01
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="array or digest"):
        load_realization(path)
    for seed in (-1, True, 1.5):
        with pytest.raises(ValueError):
            generate_permeability(seed)


def test_material_fitted_resolution_and_immediately_excluded_case():
    """Keep the material grid fixed when selecting full or whole-domain pilot meshes."""
    require_fitted_partition(32, 8, 8)
    require_fitted_partition(8, 2, 8)
    with pytest.raises(ValueError, match="material interfaces"):
        require_fitted_partition(32, 8, 7)
    with pytest.raises(ValueError):
        require_fitted_partition(0, 8, 8)
    mesh = macro_mesh()
    assert len(mesh.cells) == 512
    assert_allclose(mesh.points.min(axis=0), [0, 0], atol=0)
    assert_allclose(mesh.points.max(axis=0), [3, 1], atol=0)
    assert_allclose(mesh.areas.sum(), 3, atol=2e-15)


def test_darcy_drive_and_transport_inflow_have_distinct_natural_faces():
    """Pressure vertical data and concentration left inflow use the stated normals."""
    mesh = macro_mesh(8, 2)
    pressure_natural = natural_faces(mesh, transport=False)
    concentration_natural = natural_faces(mesh, transport=True)
    for face in mesh.boundary_faces:
        normal = mesh.normals[face]
        assert (face in pressure_natural) == (normal[0] == 0)
        assert (face in concentration_natural) == (normal[0] != -1)
    left = mesh.points[:, 0] == 0
    right = mesh.points[:, 0] == 3
    assert_array_equal(darcy_pressure(mesh.points[left]), np.full(left.sum(), 3.0))
    assert_array_equal(darcy_pressure(mesh.points[right]), np.zeros(right.sum()))
