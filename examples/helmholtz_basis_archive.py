"""Persist and restore executed coordinates of a complex Helmholtz skeleton.

An oscillatory trace coefficient vector belongs to its executed real basis
matrices, oriented partitions and frequency. Polynomial coordinates use the
declared analytic Legendre/nodal basis. Source hashes do not replace these
numeric basis records.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from pymhm.fem.traces.helmholtz import OscillatoryFaceSpace
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace


def transform_digest(matrix: np.ndarray) -> str:
    """Hash the executed dtype, shape and C-ordered numeric matrix bytes exactly."""
    matrix = np.asarray(matrix)
    header = json.dumps({"dtype": matrix.dtype.str, "shape": list(matrix.shape)}, sort_keys=True)
    digest = hashlib.sha256(header.encode())
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def raw_generator_digest() -> str:
    """Identify the raw oscillatory evaluation formula independently of QR coordinates."""
    return hashlib.sha256(inspect.getsource(OscillatoryFaceSpace._raw).encode()).hexdigest()


def basis_payload(skeleton: SkeletonSpace) -> dict[str, np.ndarray]:
    """Archive every oriented face declaration and each executed oscillatory transform."""
    if skeleton.components != 2:
        raise ValueError("complex scalar Helmholtz archives require two real components")
    arrays = {}
    faces = []
    for index, space in enumerate(skeleton.faces):
        face: dict[str, Any] = {
            "breaks": list(space.breaks),
            "degrees": list(space.degrees),
            "continuous": space.continuous,
        }
        if type(space) is OscillatoryFaceSpace:
            face.update(
                kind="oscillatory",
                wave_number_length=float(space.wave_number_length),
                raw_generator_sha256=raw_generator_digest(),
            )
            entries = []
            for segment, matrix in enumerate(space.transforms):
                key = f"face_{index}_segment_{segment}_transform"
                arrays[key] = matrix
                entries.append({"array": key, "sha256": transform_digest(matrix)})
            face["transforms"] = entries
        elif type(space) is FaceSpace:
            face["kind"] = "polynomial"
        else:
            raise ValueError("archive the executed basis of a custom face space explicitly")
        faces.append(face)
    manifest = {
        "schema": 1,
        "components": 2,
        "faces": faces,
        "orientation": "Increasing parameter from mesh.faces[:,0] to mesh.faces[:,1]",
        "coefficients": "Complex scalar coordinates; real and imaginary parts use the same basis",
    }
    arrays["trace_basis_manifest"] = np.asarray(
        json.dumps(manifest, sort_keys=True, allow_nan=False)
    )
    return arrays


def restore_trace_skeleton(mesh: Any, archive: Mapping[str, np.ndarray]) -> SkeletonSpace:
    """Replay complex trace coordinates using verified persisted matrices and mesh orientation.

    The caller verifies the field archive's digest before loading it. This
    helper checks the stored macro point/face ordering, each basis matrix's
    own digest and the finite scalar coefficient vector's matching size.
    Oscillatory restoration delegates to the shared core owner;
    basis evaluation, reversed orientation and constant moments therefore
    all use the executed matrix rather than a new QR factorization.
    """
    if not np.array_equal(archive["macro_points"], mesh.points) or not np.array_equal(
        archive["macro_faces"], mesh.faces
    ):
        raise ValueError("archived macro geometry/orientation differs from the replay mesh")
    manifest = json.loads(np.asarray(archive["trace_basis_manifest"]).item())
    if (
        manifest.get("schema") != 1
        or manifest.get("components") != 2
        or len(manifest.get("faces", [])) != len(mesh.faces)
        or manifest.get("orientation")
        != "Increasing parameter from mesh.faces[:,0] to mesh.faces[:,1]"
    ):
        raise ValueError("the archive needs a complete oriented complex trace-basis manifest")
    faces = []
    for face in manifest["faces"]:
        breaks, degrees = tuple(face["breaks"]), tuple(face["degrees"])
        if face["kind"] == "oscillatory":
            if face.get("raw_generator_sha256") != raw_generator_digest():
                raise ValueError(
                    "the archived oscillatory generator differs from the replay formula"
                )
            matrices = []
            for entry in face["transforms"]:
                matrix = archive[entry["array"]]
                if transform_digest(matrix) != entry["sha256"]:
                    raise ValueError(
                        "executed oscillatory transform digest differs from its archive"
                    )
                matrices.append(matrix)
            space = OscillatoryFaceSpace.from_executed_transforms(
                breaks,
                degrees,
                matrices,
                wave_number_length=face["wave_number_length"],
                continuous=face["continuous"],
            )
        elif face["kind"] == "polynomial":
            space = FaceSpace(breaks, degrees, face["continuous"])
        else:
            raise ValueError("the executed face-basis kind is unsupported")
        faces.append(space)
    skeleton = SkeletonSpace(mesh, tuple(faces), components=2)
    coefficients = np.asarray(archive["trace"])
    if (
        coefficients.shape != (skeleton.size // 2,)
        or coefficients.dtype.kind not in "fc"
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("the archive needs finite scalar trace coefficients in its executed basis")
    return skeleton
