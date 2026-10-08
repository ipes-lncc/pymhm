"""Unsigned physical pairings between independently declared trace spaces."""

from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.core.validation import FloatArray, IntArray, positive_int, real_array
from pymhm.fem.quadrature.partitions import simplex_partition_refines
from pymhm.fem.traces.forms import trace_quadrature
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.polygon_3d import PolygonalSkeleton3D
from pymhm.fem.traces.pressure_2d import PressureTraceSpace
from pymhm.fem.traces.pressure_3d import PressureTraceSpace3D
from pymhm.fem.traces.triangle_3d import TriangularSkeleton


def evaluate_trace_basis(space: Any, face: int, points: FloatArray) -> tuple[IntArray, FloatArray]:
    """Evaluate literal face coordinates as physical scalar/vector basis values.

    Return shapes (n,) and (q,n,components), with no incidence normal sign.
    Cartesian-component and tangential wrappers retain their physical frame.
    Custom owners may supply trace_evaluation(face, points) with this contract.
    Points are physical; independently subdivided basis owners locate their own
    pieces. No projection, common-node identification or continuity is inferred.
    """
    face = positive_int(face, "face", 0)
    points = real_array(points, "physical trace points")
    custom = getattr(space, "trace_evaluation", None)
    if callable(custom):
        return custom(face, points)
    base = getattr(space, "base", space)
    declared_faces = getattr(base, "faces", ())
    edge_owner = bool(len(declared_faces)) and all(
        callable(getattr(item, "evaluate", None)) for item in declared_faces
    )
    if (
        not isinstance(
            base,
            (
                SkeletonSpace,
                PressureTraceSpace,
                TriangularSkeleton,
                PressureTraceSpace3D,
                PolygonalSkeleton3D,
            ),
        )
        and not edge_owner
    ):
        raise TypeError("custom trace space must declare trace_evaluation")
    if face >= len(base.mesh.faces):
        raise ValueError("face outside interface mesh")
    corners = base.mesh.points[base.mesh.faces[face]]
    if points.ndim != 2 or points.shape[1] != corners.shape[1]:
        raise ValueError("physical trace points need the mesh's spatial dimension")
    origin, edges = corners[0], corners[1:] - corners[0]
    coordinates = (points - origin) @ np.linalg.pinv(edges)
    residual = points - origin - coordinates @ edges
    scale = max(1.0, float(np.linalg.norm(edges)))
    if np.max(abs(residual), initial=0) > 1e-12 * scale:
        raise ValueError("physical trace points must lie on the declared face")
    if edge_owner:
        parameter = coordinates[:, 0]
        if np.any(parameter < -1e-12) or np.any(parameter > 1 + 1e-12):
            raise ValueError("physical trace points lie outside the declared edge")
        values = declared_faces[face].evaluate(parameter)
    elif isinstance(base, PolygonalSkeleton3D):
        values = base.basis(face, points)
    else:
        bary = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
        values = base.evaluate(face, bary)
    owner = getattr(space, "dofs", None)
    ids = owner(face) if callable(owner) else space.face_dofs[face]
    components = getattr(space, "components", 1)
    frames = getattr(space, "frames", ())
    frame = frames[face] if len(frames) else np.eye(components)
    basis = np.einsum("qi,ab->qiba", values, frame).reshape(len(points), -1, frame.shape[0])
    return np.asarray(ids, dtype=np.int64), basis


def interface_pairing(
    test_space: Any, trial_space: Any, cell: int, *, order: int = 6
) -> FloatArray:
    """Integrate unsigned test dot trial on a macrocell's complete boundary.

    Rows and columns follow each owner's literal cell_dofs(cell). Edge integration
    uses the union of independent breakpoints. Built-in triangular partitions
    must be nested, including explicitly declared nonuniform refinements. Their
    containment is checked before the finer owner supplies the common rule.
    Nonnested 3D partitions require a custom common_trace_quadrature(other,face,order)
    owner returning physical points/weights resolving both bases. Physical components are
    contracted explicitly. Outward signs, coefficients and PDE choices are caller
    data; this same mass pairing supports diffusion, elasticity and curl traces.
    """
    if not hasattr(test_space, "mesh") or not hasattr(trial_space, "mesh"):
        raise TypeError("trace pairings require mesh-associated coefficient owners")
    if test_space.mesh is not trial_space.mesh:
        raise ValueError("trace spaces must share the same macro mesh")
    cell, order = positive_int(cell, "cell", 0), positive_int(order, "order")
    if cell >= len(test_space.mesh.cells):
        raise ValueError("cell outside interface mesh")
    test_ids, trial_ids = test_space.cell_dofs(cell), trial_space.cell_dofs(cell)
    row = {int(value): i for i, value in enumerate(test_ids)}
    column = {int(value): i for i, value in enumerate(trial_ids)}
    result = np.zeros((len(test_ids), len(trial_ids)))
    test_base, trial_base = (
        getattr(test_space, "base", test_space),
        getattr(trial_space, "base", trial_space),
    )
    for face in test_space.mesh.cell_faces[cell]:
        face = int(face)
        corners = test_space.mesh.points[test_space.mesh.faces[face]]
        if len(corners) == 2:
            if not hasattr(test_base, "faces") or not hasattr(trial_base, "faces"):
                raise TypeError("edge pairing requires declared polynomial face bases")
            a, b = test_base.faces[face], trial_base.faces[face]
            breaks = np.unique(np.r_[a.breaks, b.breaks])
            q, w = leggauss(max(order, max(a.degrees) + max(b.degrees) + 1))
            parameter = (breaks[:-1, None] + (q + 1) / 2 * np.diff(breaks)[:, None]).ravel()
            points = corners[0] + parameter[:, None] * (corners[1] - corners[0])
            weights = (
                np.diff(breaks)[:, None] * w / 2 * np.linalg.norm(corners[1] - corners[0])
            ).ravel()
        else:

            def partition_count(base: Any, selected: int = face) -> int:
                """Count actual integration pieces independently of trace coordinates."""
                if isinstance(base, TriangularSkeleton):
                    return len(base.face_partition(selected))
                if isinstance(base, PressureTraceSpace3D):
                    return len(base.partitions[selected])
                return 0

            owner = (
                test_space
                if partition_count(test_base) >= partition_count(trial_base)
                else trial_space
            )
            custom_owner = next(
                (
                    candidate
                    for candidate in (test_space, trial_space)
                    if callable(getattr(candidate, "common_trace_quadrature", None))
                ),
                None,
            )
            if custom_owner is not None:
                owner = custom_owner
            elif isinstance(test_base, (TriangularSkeleton, PressureTraceSpace3D)) and isinstance(
                trial_base, (TriangularSkeleton, PressureTraceSpace3D)
            ):
                test_partition = (
                    test_base.face_partition(face)
                    if isinstance(test_base, TriangularSkeleton)
                    else test_base.partitions[face]
                )
                trial_partition = (
                    trial_base.face_partition(face)
                    if isinstance(trial_base, TriangularSkeleton)
                    else trial_base.partitions[face]
                )
                if simplex_partition_refines(test_partition, trial_partition):
                    owner = test_space
                elif simplex_partition_refines(trial_partition, test_partition):
                    owner = trial_space
                else:
                    raise ValueError("3D trace pairing requires nested integration partitions")
            degree_test = (
                int(test_base.degrees[face])
                if isinstance(test_base, TriangularSkeleton)
                else getattr(test_base, "degree", 0)
            )
            degree_trial = (
                int(trial_base.degrees[face])
                if isinstance(trial_base, TriangularSkeleton)
                else getattr(trial_base, "degree", 0)
            )
            common = getattr(owner, "common_trace_quadrature", None)
            if callable(common):
                other = trial_space if owner is test_space else test_space
                points, weights = common(other, face, max(order, degree_test + degree_trial + 2))
            else:
                _, points, weights, _ = trace_quadrature(
                    owner, face, order=max(order, degree_test + degree_trial + 2)
                )
        rows, test = evaluate_trace_basis(test_space, face, points)
        columns, trial = evaluate_trace_basis(trial_space, face, points)
        if test.shape[-1] != trial.shape[-1]:
            raise ValueError("trace pairing requires equal physical value dimensions")
        local = np.einsum("q,qia,qja->ij", weights, test, trial)
        result[np.ix_([row[int(i)] for i in rows], [column[int(i)] for i in columns])] += local
    return result
