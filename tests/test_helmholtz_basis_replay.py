"""Exercise executed oscillatory coordinates independently of a recomputed QR basis."""

import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.helmholtz_basis_archive import basis_payload, restore_trace_skeleton
from pymhm.fem.traces.helmholtz import OscillatoryFaceSpace, helmholtz_skeleton
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.cartesian import CartesianMacroMesh


def test_owned_rotated_coordinates_preserve_field_orientation_and_constant_moments(monkeypatch):
    """Keep physical fields under equivalent coordinate rotations and persisted coefficients."""
    original = OscillatoryFaceSpace((0.0, 0.3, 1.0), (1, 3), False, 4.2)
    matrices, coordinates = [], []
    expected_coefficients = np.arange(1, 7) + 1j * np.arange(7, 13)
    offset = 0
    for degree, matrix in zip(original.degrees, original.transforms, strict=True):
        rotation = np.eye(degree + 1)
        angle = 0.41
        rotation[np.ix_([0, degree], [0, degree])] = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
        matrices.append(matrix @ rotation)
        coordinates.extend(rotation.T @ expected_coefficients[offset : offset + degree + 1])
        offset += degree + 1
    points = np.linspace(0.0, 1.0, 65)
    expected = original.evaluate(points) @ expected_coefficients
    reversed_expected = original.evaluate(1 - points) @ expected_coefficients
    naive = original.evaluate(points) @ coordinates
    assert np.max(abs(naive - expected)) > 0.1

    def forbid_qr(*args, **kwargs):
        """Forbid silently recomputing an archived coordinate basis."""
        pytest.fail("restoration recomputed a QR transform")

    monkeypatch.setattr(np.linalg, "qr", forbid_qr)
    for count in (1, 2):
        with threadpool_limits(count):
            restored = OscillatoryFaceSpace.from_executed_transforms(
                original.breaks,
                original.degrees,
                matrices,
                wave_number_length=original.wave_number_length,
            )
            np.testing.assert_allclose(
                restored.evaluate(points) @ coordinates, expected, rtol=2e-14, atol=2e-13
            )
            np.testing.assert_allclose(
                restored.evaluate(1 - points) @ coordinates,
                reversed_expected,
                rtol=2e-14,
                atol=2e-13,
            )
            np.testing.assert_allclose(
                restored.evaluate(points) @ restored.constant_coefficients(),
                1.0,
                rtol=2e-14,
                atol=2e-13,
            )
            for stored, executed in zip(restored.transforms, matrices, strict=True):
                np.testing.assert_array_equal(stored, executed)
                assert not stored.flags.writeable
    first = restored.transforms[0].copy()
    matrices[0][:] = 0
    np.testing.assert_array_equal(restored.transforms[0], first)
    with pytest.raises(ValueError):
        restored.transforms[0][0, 0] = 0


@pytest.mark.parametrize(
    "values",
    [
        [],
        [np.eye(2), np.eye(2)],
        [np.ones((2, 3))],
        [np.eye(2, dtype=int)],
        [np.eye(2, dtype=np.float32)],
        [np.eye(2, dtype=complex)],
        [np.full((2, 2), np.nan)],
        [np.ones((2, 2))],
    ],
)
def test_restoration_rejects_incomplete_invalid_or_singular_coordinate_matrices(values):
    """A same-sized coefficient vector cannot excuse a changed precision or singular basis."""
    with pytest.raises(ValueError):
        OscillatoryFaceSpace.from_executed_transforms(
            (0.0, 1.0), (1,), values, wave_number_length=2.0
        )


@pytest.mark.parametrize("frequency", [0.0, -1.0, np.inf, np.nan, 1j])
def test_restoration_uses_shared_frequency_conditions(frequency):
    """Replay and fresh acquisition admit exactly the same physical frequency declarations."""
    with pytest.raises(ValueError, match="positive frequency"):
        OscillatoryFaceSpace.from_executed_transforms(
            (0.0, 1.0), (1,), [np.eye(2)], wave_number_length=frequency
        )
    with pytest.raises(ValueError, match="positive frequency"):
        OscillatoryFaceSpace((0.0, 1.0), (1,), False, frequency)


def test_restoration_rejects_unavailable_continuity_and_bad_partition():
    """Inherited segment validation and the unsupported continuous family remain shared."""
    with pytest.raises(ValueError, match="positive frequency"):
        OscillatoryFaceSpace.from_executed_transforms(
            (0.0, 1.0), (1,), [np.eye(2)], wave_number_length=2.0, continuous=True
        )
    with pytest.raises(ValueError, match="strictly"):
        OscillatoryFaceSpace.from_executed_transforms(
            (0.0, 0.0, 1.0), (1, 1), [np.eye(2), np.eye(2)], wave_number_length=2.0
        )


@pytest.mark.parametrize("degree", [-1, 0.5, True])
def test_restoration_excludes_invalid_degrees_at_the_zero_degree_boundary(degree):
    """Discontinuous constants are admissible; negative/fractional/boolean degrees are not."""
    with pytest.raises(ValueError, match="degree"):
        OscillatoryFaceSpace.from_executed_transforms(
            (0.0, 1.0), (degree,), [np.ones((1, 1))], wave_number_length=2.0
        )


def test_restoration_keeps_constant_endpoints_and_binary64_byte_order():
    """Degree zero and opposite-endian executed coordinates retain their numeric contract."""
    matrix = np.asarray([[2.0]], dtype=">f8")
    restored = OscillatoryFaceSpace.from_executed_transforms(
        (0.0, 1.0), (0,), [matrix], wave_number_length=2.0
    )
    assert restored.transforms[0].dtype.str == matrix.dtype.str
    assert restored.transforms[0].tobytes() == matrix.tobytes()
    np.testing.assert_array_equal(restored.evaluate([0.0, 1.0]), [[2.0], [2.0]])
    np.testing.assert_array_equal(restored.constant_coefficients(), [0.5])


@pytest.mark.parametrize("oscillatory", [False, True])
def test_persisted_complex_trace_replays_under_other_thread_count_without_new_qr(
    tmp_path, monkeypatch, oscillatory
):
    """Actual complex coefficients use every recorded face basis and original mesh ordering."""
    mesh = CartesianMacroMesh(2, 1)
    with threadpool_limits(1):
        original = helmholtz_skeleton(mesh, 4.2, degree=3, subdivisions=2, oscillatory=oscillatory)
        payload = basis_payload(original)
        coefficients = np.arange(original.size // 2) + 0.3j
        np.savez_compressed(
            tmp_path / "trace.npz",
            macro_points=mesh.points,
            macro_faces=mesh.faces,
            trace=coefficients,
            **payload,
        )
        points = np.linspace(0, 1, 65)
        expected = [
            face.evaluate(points)
            @ coefficients[original.offsets[i] // 2 : original.offsets[i + 1] // 2]
            for i, face in enumerate(original.faces)
        ]

    def forbid_qr(*args, **kwargs):
        pytest.fail("replay recomputed the executed numerical basis")

    monkeypatch.setattr(np.linalg, "qr", forbid_qr)
    with threadpool_limits(2), np.load(tmp_path / "trace.npz", allow_pickle=False) as archive:
        restored = restore_trace_skeleton(mesh, archive)
        for i, face in enumerate(restored.faces):
            actual = (
                face.evaluate(points)
                @ archive["trace"][restored.offsets[i] // 2 : restored.offsets[i + 1] // 2]
            )
            np.testing.assert_allclose(actual, expected[i], rtol=2e-14, atol=2e-13)


@pytest.mark.parametrize(
    "defect",
    ["matrix", "orientation", "schema", "kind", "generator", "size", "nonfinite", "dtype"],
)
def test_archive_rejects_matrix_and_geometry_identity_changes(defect):
    """Matrix digests, declared raw formulas and vertex ordering are part of the vector contract."""
    mesh = CartesianMacroMesh(1)
    skeleton = helmholtz_skeleton(mesh, 4.2, degree=2, oscillatory=True)
    archive = dict(
        macro_points=mesh.points.copy(),
        macro_faces=mesh.faces.copy(),
        trace=np.zeros(skeleton.size // 2, dtype=complex),
        **basis_payload(skeleton),
    )
    manifest = json.loads(archive["trace_basis_manifest"].item())
    if defect == "matrix":
        archive["face_0_segment_0_transform"] = archive["face_0_segment_0_transform"].copy()
        archive["face_0_segment_0_transform"][0, 0] += 0.01
    elif defect == "orientation":
        archive["macro_faces"][0] = archive["macro_faces"][0, ::-1]
    elif defect == "schema":
        manifest["schema"] = 2
    elif defect == "kind":
        manifest["faces"][0]["kind"] = "unrecorded basis"
    elif defect == "generator":
        manifest["faces"][0]["raw_generator_sha256"] = "0" * 64
    elif defect == "size":
        archive["trace"] = archive["trace"][:-1]
    elif defect == "nonfinite":
        archive["trace"][0] = np.nan
    else:
        archive["trace"] = np.zeros(skeleton.size // 2, dtype=int)
    archive["trace_basis_manifest"] = np.asarray(json.dumps(manifest))
    with pytest.raises(ValueError):
        restore_trace_skeleton(mesh, archive)


def test_custom_or_real_scalar_face_basis_is_not_misidentified():
    """Unknown callbacks cannot be labeled canonical Legendre coordinates in an archive."""
    mesh = CartesianMacroMesh(1)
    with pytest.raises(ValueError, match="two real components"):
        basis_payload(SkeletonSpace(mesh))

    class CustomFaceSpace(FaceSpace):
        """Represent a custom evaluation owner whose definition has not been archived."""

    custom = SkeletonSpace(mesh, tuple(CustomFaceSpace() for _ in mesh.faces), components=2)
    with pytest.raises(ValueError, match="custom"):
        basis_payload(custom)
