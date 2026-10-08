"""Physical trace integration and projection independent of a PDE or hybrid method."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.core.validation import FloatArray, IntArray, positive_int, real_array
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.polygon_3d import PolygonalSkeleton3D, polygonal_face_quadrature
from pymhm.fem.traces.pressure_2d import PressureTraceSpace
from pymhm.fem.traces.pressure_3d import PressureTraceSpace3D
from pymhm.fem.traces.triangle_3d import TriangularSkeleton


def trace_quadrature(
    space: Any, face: int, *, order: int = 6
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray]:
    """Return face coordinates, physical points/weights and the executed value basis.

    Shapes are (n,), (q,dimension), (q,), (q,n,value_components). Basis values
    use canonical face coordinates; no incidence sign is inserted. The order
    floor integrates each polynomial Gram matrix exactly. Nonpolynomial data
    can require a larger order. Scalar edge/triangle, Cartesian-component and
    tangential wrappers retain their basis owner's coefficients and continuity.
    A custom space may implement ``trace_quadrature(face, order=...)`` with this
    same contract, including vector or tensor values flattened by component.
    """
    face, order = positive_int(face, "face", 0), positive_int(order, "order")
    if face >= len(space.mesh.faces):
        raise ValueError("face outside interface mesh")
    custom = getattr(space, "trace_quadrature", None)
    if callable(custom):
        return custom(face, order=order)
    base = getattr(space, "base", space)
    if not isinstance(
        base,
        (
            SkeletonSpace,
            PressureTraceSpace,
            TriangularSkeleton,
            PressureTraceSpace3D,
            PolygonalSkeleton3D,
        ),
    ):
        raise TypeError("custom trace space must declare trace_quadrature")
    owner = getattr(space, "dofs", None)
    dofs = owner(face) if callable(owner) else space.face_dofs[face]
    corners = base.mesh.points[base.mesh.faces[face]]
    if isinstance(base, (SkeletonSpace, PressureTraceSpace)):
        declaration = base.faces[face]
        q, w = leggauss(max(order, max(declaration.degrees) + 1))
        parameter = np.concatenate(
            [
                lo + (q + 1) * (hi - lo) / 2
                for lo, hi in zip(declaration.breaks[:-1], declaration.breaks[1:], strict=True)
            ]
        )
        weights = np.concatenate(
            [
                w * (hi - lo) / 2
                for lo, hi in zip(declaration.breaks[:-1], declaration.breaks[1:], strict=True)
            ]
        ) * np.linalg.norm(corners[1] - corners[0])
        points = corners[0] + parameter[:, None] * (corners[1] - corners[0])
        values = declaration.evaluate(parameter)
    elif isinstance(base, PolygonalSkeleton3D):
        points, weights = polygonal_face_quadrature(base.mesh, face, order)
        values = base.basis(face, points)
    else:
        degree = int(base.degrees[face]) if isinstance(base, TriangularSkeleton) else base.degree
        parts = (
            base.face_partition(face)
            if isinstance(base, TriangularSkeleton)
            else base.partitions[face]
        )
        q, w = triangle_quadrature(max(order, degree + 1))
        bary = np.einsum("qi,sij->sqj", q, parts).reshape(-1, 3)
        points = corners[0] + bary[:, 1:] @ (corners[1:] - corners[0])
        subvertices = parts @ corners
        areas = (
            np.linalg.norm(
                np.cross(
                    subvertices[:, 1] - subvertices[:, 0], subvertices[:, 2] - subvertices[:, 0]
                ),
                axis=1,
            )
            / 2
        )
        weights = (areas[:, None] * w).ravel()
        values = base.evaluate(face, bary)
    components = getattr(space, "components", 1)
    frames = getattr(space, "frames", ())
    frame = frames[face] if len(frames) else np.eye(components)
    # Kronecker columns are basis mode first and coefficient component second.
    basis = np.einsum("qi,ab->qiba", values, frame).reshape(len(points), -1, frame.shape[0])
    return np.asarray(dofs, dtype=np.int64), points, weights, basis


def _faces(space: Any, faces: Iterable[int] | None) -> tuple[int, ...]:
    """Select every declared face or validate an explicit integration subset."""
    selected = tuple(range(len(space.mesh.faces))) if faces is None else tuple(faces)
    if len(set(selected)) != len(selected):
        raise ValueError("integration faces must be distinct")
    return selected


def trace_linear_form(
    space: Any, value: Any, *, faces: Iterable[int] | None = None, order: int = 6
) -> FloatArray:
    """Assemble integral(value dot test) in the declared global trace basis.

    ``value`` is a scalar/vector constant or a callable evaluated at physical
    points with output (q,value_components). A scalar constant broadcasts to
    every component; scalar fields may return (q,). Signs and selected physical
    boundaries are caller choices. Shared coordinates accumulate contributions.
    """
    output = np.zeros(space.size)
    for face in _faces(space, faces):
        dofs, points, weights, basis = trace_quadrature(space, face, order=order)
        data = real_array(value(points) if callable(value) else value, "trace value")
        if basis.shape[-1] == 1 and data.shape == (len(points),):
            data = data[:, None]
        try:
            data = np.broadcast_to(data, (len(points), basis.shape[-1]))
        except ValueError as error:
            raise ValueError(
                "trace value must match physical points and value components"
            ) from error
        output[dofs] += np.einsum("q,qia,qa->i", weights, basis, data)
    return output


def trace_bilinear_form(
    space: Any, coefficient: Any = 1.0, *, faces: Iterable[int] | None = None, order: int = 6
) -> Any:
    """Assemble integral(test dot coefficient trial) as a sparse trace operator.

    A scalar coefficient gives an isotropic scalar/vector pairing. A constant
    or physical-point callable may instead supply a matrix in value components,
    with shape (m,m) or (q,m,m). This supports mass, reaction, Robin, impedance
    and tensor pairings without choosing a physical boundary convention.
    """
    rows, columns, entries = [], [], []
    for face in _faces(space, faces):
        dofs, points, weights, basis = trace_quadrature(space, face, order=order)
        data = real_array(
            coefficient(points) if callable(coefficient) else coefficient, "trace coefficient"
        )
        count, components = len(points), basis.shape[-1]
        if data.ndim == 0 or data.shape == (count,):
            data = np.broadcast_to(data, (count,))[:, None, None] * np.eye(components)
        try:
            data = np.broadcast_to(data, (count, components, components))
        except ValueError as error:
            raise ValueError(
                "trace coefficient must be scalar or a matrix in value components"
            ) from error
        block = np.einsum("q,qia,qab,qjb->ij", weights, basis, data, basis)
        ii, jj = np.nonzero(block)
        rows.extend(dofs[ii].tolist())
        columns.extend(dofs[jj].tolist())
        entries.extend(block[ii, jj].tolist())
    return sparse.coo_matrix((entries, (rows, columns)), shape=(space.size, space.size)).tocsc()


def project_trace(
    space: Any, value: Any, *, faces: Iterable[int] | None = None, order: int = 6
) -> dict[int, float]:
    """L2-project physical values onto selected trace coefficients with shared nodes.

    Solve the executed Gram matrix on exactly the union of selected face DOFs.
    The returned coefficient map can prescribe interface values in any problem.
    Normal densities, Robin pseudo-tractions and physical scalar/vector traces
    remain distinct choices of the user-supplied ``value`` and space.
    """
    selected = _faces(space, faces)
    load = trace_linear_form(space, value, faces=selected, order=order)
    matrix = trace_bilinear_form(space, faces=selected, order=order)
    active = (
        np.unique(
            np.concatenate([trace_quadrature(space, face, order=order)[0] for face in selected])
        )
        if selected
        else np.empty(0, dtype=np.int64)
    )
    coefficients = np.linalg.solve(matrix[active][:, active].toarray(), load[active])
    return dict(zip(active.tolist(), coefficients.tolist(), strict=True))


def trace_boundary_data(
    space: Any,
    value: Any,
    prescribed: Mapping[int, Any] | None = None,
    *,
    order: int = 6,
) -> tuple[FloatArray, dict[int, float]]:
    """Integrate exterior value data and project explicitly prescribed trace data.

    Faces in ``prescribed`` supply fixed trace coefficients; every other
    exterior face supplies the unsigned functional ``integral(value dot test)``.
    No PDE, outward sign or physical variable is inferred. Projection is joint
    across prescribed faces when their coefficient coordinates are shared.
    Face-dependent prescribed functions use physical scalar/vector values in
    the owner's actual basis. Interior faces cannot be prescribed here.
    """
    natural = {} if prescribed is None else dict(prescribed)
    faces = {positive_int(face, "boundary face", 0) for face in natural}
    boundary = set(space.mesh.boundary_faces)
    if not faces.issubset(boundary):
        raise ValueError("prescribed traces must identify exterior faces")
    load = trace_linear_form(space, value, faces=sorted(boundary - faces), order=order)
    if not faces:
        return load, {}
    projected_load = np.zeros(space.size)
    selected = []
    for face in sorted(faces):
        projected_load += trace_linear_form(space, natural[face], faces=[face], order=order)
        indices, _, _, _ = trace_quadrature(space, face, order=order)
        selected.extend(indices.tolist())
    coordinates = np.unique(selected)
    matrix = trace_bilinear_form(space, faces=sorted(faces), order=order)
    values = np.linalg.solve(
        matrix[coordinates][:, coordinates].toarray(), projected_load[coordinates]
    )
    return load, dict(zip(coordinates.tolist(), values.tolist(), strict=True))
