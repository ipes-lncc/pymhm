"""One-sided scalar trace evaluation and reduced jump integration on macrofaces.

The caller supplies executed local responses and the explicit coefficient
alpha*lower/(2H). This operation does not solve a method or choose its global
Equation. Adjacent fields preserve independent one-sided traces.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, IntArray
from pymhm.fem.reference import legendre_values
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.fem.traces.integration import incident_face_breaks
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.scalar import PreparedScalarTrace, edge_basis, prepare_scalar_trace
from pymhm.materials.evaluation import scalar_values
from pymhm.meshes.triangle import TriangleMesh


def scalar_trace_matrix(
    macro: Any,
    fine: TriangleMesh,
    face: int,
    degree: int,
    parameter: FloatArray,
    prepared: PreparedScalarTrace | None = None,
) -> Any:
    """Represent one-sided local nodal traces as a sparse evaluation operator."""
    count = len(nodal_space(fine, degree)[1])
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    covered = np.zeros(len(parameter), dtype=bool)
    prepared = prepare_scalar_trace(macro, fine, face, degree) if prepared is None else prepared
    for positions, ids, candidates in prepared.selections(parameter):
        selected = candidates[~covered[candidates]]
        local = (parameter[selected] - positions[0]) / (positions[1] - positions[0])
        basis = edge_basis(degree, local)
        rows.extend(np.repeat(selected, len(ids)))
        columns.extend(np.tile(ids, len(selected)))
        values.extend(basis.ravel())
        covered[selected] = True
    if not covered.all():
        raise ValueError("fine boundary does not cover the requested macroface")
    return sparse.coo_matrix((values, (rows, columns)), shape=(len(parameter), count)).tocsr()


@dataclass(frozen=True)
class FaceJumpForm:
    """Common face quadrature, signed jumps and projected Dirichlet moments."""

    face: int
    parameter: FloatArray
    weights: FloatArray
    coefficient: float
    indices: IntArray
    jump: FloatArray
    source_jump: FloatArray
    prescribed: FloatArray
    breaks: FloatArray
    projection: FloatArray
    sides: tuple[tuple[int, float, Any], ...]
    traces: tuple[PreparedScalarTrace, ...]

    def boundary_value(self, parameter: FloatArray) -> FloatArray:
        """Evaluate the piecewise Pk orthogonal boundary projection used in enrichment."""
        owners = np.clip(
            np.searchsorted(self.breaks, parameter, side="right") - 1, 0, len(self.breaks) - 2
        )
        t = 2 * (parameter - self.breaks[owners]) / np.diff(self.breaks)[owners] - 1
        return np.einsum(
            "qi,qi->q", legendre_values(t, self.projection.shape[1] - 1), self.projection[owners]
        )


def face_jump_form(
    system: HybridSystem,
    skeleton: SkeletonSpace,
    face: int,
    degree: int,
    order: int,
    alpha: float,
    lower: float,
    dirichlet: Any,
) -> FaceJumpForm:
    """Assemble one macroface jump on the common incident fine-edge partition."""
    mesh = skeleton.mesh
    neighbors = mesh.face_cells[face]
    neighbors = neighbors[neighbors >= 0]
    breaks = incident_face_breaks(
        skeleton,
        {int(cell): system.local_metadata[cell][0] for cell in neighbors},
        face,
        degree,
    )
    gauss, weights = leggauss(max(order, degree + 1))
    parameter = (breaks[:-1, None] + (gauss + 1) / 2 * np.diff(breaks)[:, None]).ravel()
    measure = (np.diff(breaks)[:, None] * weights / 2 * mesh.lengths[face]).ravel()
    ids = []
    for cell in neighbors:
        ids.extend(system.responses[cell].problem.trace_dofs)
        ids.append(system.kernel_offsets[cell])
    indices = np.unique(ids).astype(np.int64)
    dtype = np.result_type(*(system.responses[cell].source.dtype for cell in neighbors))
    jump = np.zeros((len(parameter), len(indices)), dtype=dtype)
    source_jump = np.zeros(len(parameter), dtype=dtype)
    sides, traces = [], []
    for cell in neighbors:
        response = system.responses[cell]
        fine = system.local_metadata[cell][0]
        side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
        sign = float(mesh.signs[cell, side])
        prepared = prepare_scalar_trace(mesh, fine, face, degree)
        evaluation = scalar_trace_matrix(mesh, fine, face, degree, parameter, prepared)
        local_ids = np.r_[response.problem.trace_dofs, system.kernel_offsets[cell]]
        derivative = np.column_stack((-response.lifts, response.retained_basis))
        jump[:, np.searchsorted(indices, local_ids)] += sign * (evaluation @ derivative)
        source_jump += sign * (evaluation @ response.source)
        sides.append((int(cell), sign, evaluation))
        traces.append(prepared)
    projection = np.zeros((len(breaks) - 1, degree + 1))
    if len(neighbors) == 1:
        start, end = mesh.points[mesh.faces[face]]
        points = start + parameter[:, None] * (end - start)
        values = scalar_values(dirichlet, points).reshape(-1, len(gauss))
        basis = legendre_values(gauss, degree)
        projection = (values * weights / 2) @ basis * (2 * np.arange(degree + 1) + 1)
        prescribed = (projection @ basis.T).ravel()
    else:
        prescribed = np.zeros(len(parameter))
    return FaceJumpForm(
        face,
        parameter,
        measure,
        alpha * lower / (2 * mesh.lengths[face]),
        indices,
        jump,
        source_jump,
        prescribed,
        breaks,
        projection,
        tuple(sides),
        tuple(traces),
    )
