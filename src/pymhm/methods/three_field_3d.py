"""Three-field MH²M on tetrahedra with independently resolved interface spaces."""

from math import fsum
from typing import Any, cast

import numpy as np
from scipy import sparse

from pymhm.core.validation import positive_int
from pymhm.fem.scalar.tetrahedron import tetra_operators
from pymhm.fem.traces.pairing import interface_pairing
from pymhm.fem.traces.pressure_3d import PressureTraceSpace3D, boundary_rules
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import scalar_values_3d
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.methods.three_field import MH2MLocal, _neumann_maps
from pymhm.postprocessing.solutions import MH2M3DSolution as MH2M3DSolution


def _local(
    mesh: TetraMesh,
    cell: int,
    gamma: PressureTraceSpace3D,
    flux: TriangularSkeleton,
    degree: int,
    refinement: int,
    material: Any,
    source: Any,
    order: int,
    solver: str,
) -> MH2MLocal:
    """Integrate unsigned conormal moments and the two Neumann maps in three dimensions."""
    fine = mesh.submesh(cell, refinement)
    stiffness, mass, load = tetra_operators(
        fine, degree, diffusion=material, source=source, order=order
    )
    coupling = tetra_trace_coupling(mesh, cell, fine, flux, degree)
    trace_ids = gamma.cell_dofs(cell)
    pairing = interface_pairing(flux, gamma, cell, order=order)
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        width = len(flux.dofs(int(face)))
        coupling[:, offset : offset + width] *= mesh.signs[cell, side]
        offset += width
    return _neumann_maps(
        fine,
        trace_ids,
        stiffness,
        mass,
        load,
        coupling,
        pairing,
        np.ones(coupling.shape[1]),
        solver,
    )


def solve_mh2m_3d(
    mesh: TetraMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    pressure_trace: PressureTraceSpace3D | None = None,
    flux_space: TriangularSkeleton | None = None,
    degree: int = 2,
    local_refinement: int = 2,
    quadrature_order: int = 5,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
) -> MH2M3DSolution:
    """Solve the three-dimensional MH²M using equations (25)--(29) of version 3.

    Gamma is continuous nodal Pk on uniformly refined triangular macrofaces.
    Lambda is independently subdivided and uses broken Bernstein polynomials;
    each incident macrocell has its own outward conormal copy. All partitions
    must align the fine tetrahedral boundary. A singular local Neumann map or
    pressure-trace system is rejected; this check is not a uniform inf-sup proof.

    Neumann data prescribe outward physical q.n. Remaining exterior faces
    interpolate pressure at Gamma nodes. Pure Neumann data require integral
    compatibility and mean_pressure fixes the full volume-mean reconstruction.
    Local operators use positive tetrahedral Gaussian integration. Discontinuous
    material requires geometrically adequate local resolution.
    """
    if not isinstance(mesh, TetraMesh):
        raise TypeError("MH2M3D requires a tetrahedral mesh")
    degree = positive_int(degree, "degree")
    refinement = _dyadic(local_refinement, "local_refinement")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    gamma = PressureTraceSpace3D(mesh) if pressure_trace is None else pressure_trace
    flux = TriangularSkeleton(mesh) if flux_space is None else flux_space
    if gamma.mesh is not mesh or flux.mesh is not mesh:
        raise ValueError("Gamma and Lambda must use the supplied mesh")
    if np.any(flux.continuous):
        raise ValueError("Lambda uses discontinuous face polynomials")
    if gamma.subdivisions > refinement or np.any(flux.subdivisions > refinement):
        raise ValueError("local refinement must resolve both interface partitions")
    natural = {} if neumann is None else dict(neumann)
    if any(face not in mesh.boundary_faces for face in natural):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    mean = np.asarray(mean_pressure)
    if mean.shape != () or np.iscomplexobj(mean) or not np.isfinite(mean):
        raise ValueError("mean_pressure must be finite and real")
    pure = len(natural) == len(mesh.boundary_faces)
    if not pure and mean_pressure != 0:
        raise ValueError("mean_pressure applies only to pure Neumann problems")
    local = tuple(
        _local(
            mesh, cell, gamma, flux, degree, refinement, permeability, source, order, local_solver
        )
        for cell in range(len(mesh.cells))
    )
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    rhs = np.zeros(gamma.size)
    for data in local:
        ids = data.trace_dofs
        rows.extend(np.repeat(ids, len(ids)))
        columns.extend(np.tile(ids, len(ids)))
        entries.extend(data.trace_matrix.ravel())
        np.add.at(rhs, ids, data.trace_rhs)
    matrix = sparse.coo_matrix((entries, (rows, columns)), shape=(gamma.size, gamma.size)).tocsc()
    trace = np.zeros(gamma.size)
    fixed: list[int] = []
    boundary_terms: list[float] = []
    for cell, data in enumerate(local):
        for face, _, _, points, weights, bary in boundary_rules(
            mesh, cell, cast(TetraMesh, data.mesh), degree, max(order, gamma.degree + 2)
        ):
            if face not in natural:
                continue
            weighted = weights * scalar_values_3d(natural[face], points)
            boundary_terms.extend(weighted)
            np.add.at(rhs, gamma.face_dofs[face], -(gamma.evaluate(face, bary).T @ weighted))
    for boundary_face in mesh.boundary_faces:
        if boundary_face not in natural:
            ids = gamma.face_dofs[boundary_face]
            trace[ids] = scalar_values_3d(dirichlet, gamma.nodes[ids])
            fixed.extend(ids)
    if pure:
        sources = np.concatenate([data.load for data in local])
        total_source, total_boundary = fsum(sources), fsum(boundary_terms)
        units = (len(sources) + len(boundary_terms)) * np.finfo(float).eps
        rounding = (
            units / (1 - units) * (fsum(abs(sources)) + fsum(abs(np.asarray(boundary_terms))))
        )
        if abs(total_source - total_boundary) > (
            1e-10 * max(abs(total_source), abs(total_boundary), np.finfo(float).tiny) + rounding
        ):
            raise ValueError("incompatible Neumann data: source and outward flux integrals differ")
    free = np.setdiff1d(np.arange(gamma.size), fixed)
    forcing = (rhs - matrix @ trace)[free]
    if len(free):
        reduced = matrix[free][:, free]
        if pure:
            moments = np.zeros(gamma.size)
            offset = 0.0
            for data in local:
                np.add.at(moments, data.trace_dofs, data.volume_moments @ data.pressure_lift)
                offset += float(data.volume_moments @ data.pressure_source)
            target = mean_pressure * mesh.volumes.sum() - offset
            augmented = sparse.bmat(
                [[reduced, moments[:, None]], [moments[None], None]], format="csc"
            )
            trace[free] = solve_linear(augmented, np.r_[forcing, target], solver=solver)[:-1]
        else:
            trace[free] = solve_linear(reduced, forcing, solver=solver)
    defect = (matrix @ trace - rhs)[free]
    scale = max(
        float(np.linalg.norm(rhs[free])),
        float(np.linalg.norm((abs(matrix) @ abs(trace))[free])),
        np.finfo(float).tiny,
    )
    residual = float(np.linalg.norm(defect) / scale)
    if residual > 1e-10:
        raise ValueError("MH2M original trace equations fail after boundary elimination")
    pressure = tuple(
        data.pressure_lift @ trace[data.trace_dofs] + data.pressure_source for data in local
    )
    conormal = tuple(
        data.conormal_lift @ trace[data.trace_dofs] + data.conormal_source for data in local
    )
    return MH2M3DSolution(
        gamma,
        flux,
        local,
        trace,
        pressure,
        conormal,
        matrix,
        rhs,
        free,
        degree,
        permeability,
        source,
        order,
        residual,
    )
