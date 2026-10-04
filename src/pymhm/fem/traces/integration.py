"""Scalar skeletal moments on the union of incident physical fine-edge partitions."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.core.validation import FloatArray
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.scalar import edge_pieces
from pymhm.materials.evaluation import scalar_values
from pymhm.meshes.triangle import TriangleMesh


def incident_face_breaks(
    skeleton: SkeletonSpace,
    local_meshes: Sequence[TriangleMesh] | Mapping[int, TriangleMesh],
    face: int,
    degree: int,
) -> FloatArray:
    """Return integration cuts without enriching the skeletal approximation space.

    The partition contains the skeletal breaks and all fine-edge endpoints on
    either incident side. Coincident physical endpoints are merged at the
    same floating-point threshold used by the edge trace assembly.
    """
    mesh = skeleton.mesh
    raw = [0.0, 1.0, *skeleton.faces[face].breaks]
    for cell in mesh.face_cells[face]:
        if cell >= 0:
            for positions, _ in edge_pieces(mesh, local_meshes[cell], face, degree):
                raw.extend(positions)
    breaks = np.sort(np.clip(raw, 0, 1))
    breaks = breaks[np.r_[True, np.diff(breaks) > 64 * np.finfo(float).eps]]
    breaks[0], breaks[-1] = 0, 1
    return breaks


def integrate_dirichlet_trace(
    skeleton: SkeletonSpace,
    local_meshes: Sequence[TriangleMesh],
    values: Any,
    degree: int,
    order: int,
    *,
    faces: Sequence[int] | None = None,
) -> FloatArray:
    """Integrate scalar pressure data against Lambda on every exterior macroface.

    Fine-edge cuts resolve prescribed data that are piecewise polynomial on
    the local boundary. They do not change Lambda's degrees or breakpoints.
    Additional discontinuities of an arbitrary callback still require an
    integration partition that resolves those discontinuities.
    ``faces`` selects exterior macrofaces; omitted faces have zero moments and
    the callback is never evaluated there.
    """
    if skeleton.components != 1:
        raise ValueError("scalar boundary integration requires one skeleton component")
    load = np.zeros(skeleton.size)
    mesh = skeleton.mesh
    selected = mesh.boundary_faces if faces is None else tuple(faces)
    if len(set(selected)) != len(selected) or any(f not in mesh.boundary_faces for f in selected):
        raise ValueError("boundary integration faces must be distinct exterior macrofaces")
    for face in selected:
        space = skeleton.faces[face]
        breaks = incident_face_breaks(skeleton, local_meshes, int(face), degree)
        points, weights = leggauss(max(order, max(space.degrees) + 2))
        parameter = (breaks[:-1, None] + (points + 1) / 2 * np.diff(breaks)[:, None]).ravel()
        weights = (np.diff(breaks)[:, None] * weights / 2 * mesh.lengths[face]).ravel()
        start, end = mesh.points[mesh.faces[face]]
        physical = start + parameter[:, None] * (end - start)
        load[skeleton.dofs(int(face))] = space.evaluate(parameter).T @ (
            weights * scalar_values(values, physical)
        )
    return load
