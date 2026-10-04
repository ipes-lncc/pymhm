"""Physical trajectory replay retains actual bases, precision and one-sided fields."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from examples.transport_trajectory import (
    coefficient_arrays,
    replay_scalar,
    replay_trace,
    scalar_geometry,
    transport_trace_geometry,
    uniform_trace_geometry,
    vector_coefficients,
)
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh


def configuration():
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2, 2) for _ in mesh.faces))
    meshes = tuple(mesh.submesh(cell, 2) for cell in range(len(mesh.cells)))
    return skeleton, meshes


def test_scalar_archive_matches_independent_cubic_and_preserves_macro_limits(monkeypatch):
    """Replay analytical cubic derivatives from stored matrices, regardless of a fresh basis."""
    skeleton, meshes = configuration()
    geometry, record = scalar_geometry(skeleton, meshes, 3)
    nodes = geometry["nodal_coordinates"]
    values = nodes[:, 0] ** 3 + 0.2 * nodes[:, 0] * nodes[:, 1] ** 2 - nodes[:, 0]
    offsets = geometry["nodal_offsets"]
    fields = tuple(
        values[a:b] + cell
        for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
    )
    arrays, field_record = coefficient_arrays("concentration", fields, geometry)
    monkeypatch.setattr(
        "examples.transport_trajectory.reference_basis",
        lambda *args: pytest.fail("replay must use the executed archived basis"),
    )
    for threads in (1, 2):
        with threadpool_limits(threads):
            samples, gradient = replay_scalar(geometry, record, arrays, field_record)
        points = np.einsum(
            "qi,tij->tqj",
            geometry["quadrature_barycentric"],
            geometry["fine_points"][geometry["fine_cells"]],
        )
        x, y = points[..., 0], points[..., 1]
        exact = x**3 + 0.2 * x * y**2 - x + geometry["fine_cell_macro"][:, None]
        exact_gradient = np.stack((3 * x**2 + 0.2 * y**2 - 1, 0.4 * x * y), axis=-1)
        assert_allclose(samples, exact, rtol=2e-14, atol=2e-14)
        assert_allclose(gradient, exact_gradient, rtol=2e-14, atol=2e-14)
    limited, _ = replay_scalar(geometry, record, arrays, field_record, first_cell=1, last_cell=3)
    assert_array_equal(limited, samples[1:3])
    changed = dict(geometry, reference_values=geometry["reference_values"] * 2)
    with pytest.raises(ValueError, match="digest differs: reference_values"):
        replay_scalar(changed, record, arrays, field_record)
    changed = dict(arrays, concentration=arrays["concentration"] + 1e-5)
    with pytest.raises(ValueError, match="digest differs: concentration"):
        replay_scalar(geometry, record, changed, field_record)


def test_scalar_archive_retains_wider_coefficient_digits_and_geometry_contract():
    """Separate remainders preserve physical low digits and reject incompatible geometry."""
    skeleton, meshes = configuration()
    geometry, record = scalar_geometry(skeleton, meshes, 3)
    increment = np.longdouble(2) ** -60 if np.finfo(np.longdouble).nmant > 52 else 0.0
    fields = tuple(
        np.full(b - a, np.longdouble(1) + increment, dtype=np.longdouble)
        for a, b in zip(geometry["nodal_offsets"][:-1], geometry["nodal_offsets"][1:], strict=True)
    )
    arrays, field_record = coefficient_arrays("concentration", fields, geometry)
    assert np.all(arrays["concentration_correction"] == increment)
    assert field_record["executed_mantissa_bits"] == np.finfo(np.longdouble).nmant
    values, gradients = replay_scalar(geometry, record, arrays, field_record)
    assert_allclose(values, 1 + increment, rtol=0, atol=4e-15)
    assert_allclose(gradients, 0, rtol=0, atol=4e-14)
    for options in ({"degree": 0}, {"degree": 3, "order": 3}):
        with pytest.raises(ValueError):
            scalar_geometry(skeleton, meshes, **options)
    with pytest.raises(ValueError, match="one fine mesh"):
        scalar_geometry(skeleton, meshes[:-1], 3)
    with pytest.raises(ValueError):
        scalar_geometry(skeleton, tuple(reversed(meshes)), 3)
    for changed in (fields[:-1], tuple(v[:-1] for v in fields)):
        with pytest.raises(ValueError, match="shapes differ"):
            coefficient_arrays("concentration", changed, geometry)
    with pytest.raises(ValueError, match="common executed dtype"):
        coefficient_arrays("concentration", (fields[0].astype(np.float32), fields[1]), geometry)
    with pytest.raises(ValueError, match="real floating dtype"):
        coefficient_arrays("concentration", tuple(v.astype(int) for v in fields), geometry)
    with pytest.raises(ValueError, match="identifier"):
        coefficient_arrays("", fields, geometry)
    for first, last in ((-1, 2), (0, 99), (2, 1), (True, 3), (0, 1.5)):
        with pytest.raises(ValueError):
            replay_scalar(geometry, record, arrays, field_record, first_cell=first, last_cell=last)


def test_trace_archive_uses_executed_face_basis_and_actual_normal_orientation():
    """Replay a constant physical vector's normal component on every oriented macroface."""
    skeleton, meshes = configuration()
    geometry, record = uniform_trace_geometry(skeleton, convention="Darcy physical normal flux")
    vector = np.array([2.0, -0.5])
    coefficients = np.zeros(skeleton.size)
    for face, normal in enumerate(skeleton.mesh.normals):
        dofs = skeleton.dofs(face)
        coefficients[dofs[::3]] = normal @ vector
    actual = replay_trace(geometry, record, coefficients)
    assert_allclose(actual, np.broadcast_to(skeleton.mesh.normals @ vector, actual.T.shape).T)
    for cell in range(len(meshes)):
        faces = skeleton.mesh.cell_faces[cell]
        outflow = np.sum(
            skeleton.mesh.signs[cell]
            * skeleton.mesh.lengths[faces]
            * (actual[faces] @ geometry["face_weights"])
        )
        assert abs(outflow) < 1e-14
    for changed in (coefficients[:-1], coefficients * np.nan):
        with pytest.raises(ValueError, match="coefficient vector"):
            replay_trace(geometry, record, changed)
    with pytest.raises(ValueError, match="digest differs: macro_signs"):
        replay_trace(dict(geometry, macro_signs=-geometry["macro_signs"]), record, coefficients)
    with pytest.raises(ValueError, match="convention"):
        uniform_trace_geometry(skeleton, convention="")
    faces = list(skeleton.faces)
    faces[0] = FaceSpace.uniform(1)
    with pytest.raises(ValueError, match="uniform face spaces"):
        uniform_trace_geometry(SkeletonSpace(skeleton.mesh, tuple(faces)), convention="Robin")


def test_portable_multiplier_vectors_keep_their_convention_and_precision():
    """Robin coefficients and nodal reactions do not acquire a physical flux meaning."""
    coefficients = np.array([1.0, -2.0], dtype=np.longdouble)
    arrays, record = vector_coefficients(
        "robin_trace", coefficients, convention="Half-advection Robin multiplier"
    )
    assert record["coefficient_convention"] == "Half-advection Robin multiplier"
    assert_array_equal(arrays["robin_trace"], coefficients)
    for values, convention in ((np.ones((2, 2)), "Robin"), (coefficients, "")):
        with pytest.raises(ValueError, match="vector and explicit convention"):
            vector_coefficients("trace", values, convention=convention)


def test_transport_trace_archive_distinguishes_natural_and_removed_exterior_data():
    """An interior Robin multiplier and a removed inflow trace cannot be called diffusive flux."""
    skeleton, _ = configuration()
    boundary = skeleton.mesh.boundary_faces
    strong = tuple(int(f) for f in boundary if skeleton.mesh.normals[f, 0] == -1)
    natural = tuple(int(f) for f in boundary if f not in strong)
    arrays, record = transport_trace_geometry(
        skeleton,
        diffusive_faces=natural,
        strong_faces=strong,
        numerical_normal_velocity="Darcy skeletal normal flux",
    )
    assert_array_equal(arrays["face_boundary_kind"][list(strong)], 2)
    assert_array_equal(arrays["face_boundary_kind"][list(natural)], 1)
    interior = np.setdiff1d(np.arange(len(skeleton.faces)), boundary)
    assert_array_equal(arrays["face_boundary_kind"][interior], 0)
    assert "no prescribed physical flux" in record["boundary_kind"]["2"]
    assert record["numerical_normal_velocity"] == "Darcy skeletal normal flux"
    assert "numerical_normal_velocity" in record["boundary_kind"]["0"]
    for diffusive, essential in (
        (natural, ()),
        (tuple(boundary), strong),
        (natural + (999,), strong),
    ):
        with pytest.raises(ValueError, match="partition the exterior"):
            transport_trace_geometry(
                skeleton,
                diffusive_faces=diffusive,
                strong_faces=essential,
                numerical_normal_velocity="Darcy skeletal normal flux",
            )
