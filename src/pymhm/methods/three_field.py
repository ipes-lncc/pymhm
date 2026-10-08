"""Three-field multiscale Darcy method of de Barros, Madureira and Valentin.

The pressure trace Gamma is globally continuous, while Lambda has independent
outward conormal coefficients on each macrocell. Equations (25)--(29) of the
2026 preprint define two local Neumann inverses and a global pressure-trace
system. The paper's lambda=A grad(p).n is minus the physical Darcy flux q.n.
"""

from typing import Any, cast

import numpy as np
from scipy import sparse

from pymhm.core.validation import positive_int
from pymhm.fem.scalar.neumann import NeumannMaps as MH2MLocal
from pymhm.fem.scalar.neumann import neumann_maps as _neumann_maps
from pymhm.fem.scalar.triangle import scalar_operators, trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace, interface_pairing
from pymhm.fem.traces.pressure_2d import PressureTraceSpace as PressureTraceSpace
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import scalar_values
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.refinement import validate_submesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import MH2MSolution as MH2MSolution


def _local_problem(
    mesh: TriangleMesh | PolygonMesh,
    cell: int,
    gamma: PressureTraceSpace,
    flux: SkeletonSpace,
    degree: int,
    refinement: int,
    material: Any,
    source: Any,
    order: int,
    solver: str,
    fine: TriangleMesh | None = None,
) -> MH2MLocal:
    """Assemble the two local Neumann maps with an exact boundary-mean constraint."""
    fine = mesh.submesh(cell, refinement) if fine is None else fine
    stiffness, mass, load = scalar_operators(
        fine, degree, diffusion=material, source=source, order=order
    )
    coupling = trace_coupling(cast(TriangleMesh, mesh), cell, fine, flux, degree)
    trace_ids = gamma.cell_dofs(cell)
    pairing = interface_pairing(flux, gamma, cell, order=order)
    constant = np.zeros(coupling.shape[1])
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        lam = flux.faces[face]
        width = lam.size
        coupling[:, offset : offset + width] *= mesh.signs[cell][side]
        constant[offset : offset + width] = lam.constant_coefficients()
        offset += width
    return _neumann_maps(
        fine, trace_ids, stiffness, mass, load, coupling, pairing, constant, solver
    )


def solve_mh2m(
    mesh: TriangleMesh | PolygonMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    pressure_trace: PressureTraceSpace | None = None,
    flux_space: SkeletonSpace | None = None,
    degree: int = 1,
    local_refinement: int = 4,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
    quadrature_order: int = 6,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
) -> MH2MSolution:
    """Solve the 2D three-field MH2M, equations (25)--(29), with independent spaces.

    Gamma is continuous across macro vertices; Lambda is discontinuous and is
    duplicated on each incident macrocell. Default Gamma is P1, Lambda is P0.
    Macroelements may be triangles or simple polygons, including nonconvex
    polygons triangulated by the shared local-mesh builder. Each polygon side
    retains its independent conormal space and continuous pressure endpoints.
    Pressure degree and local refinement specify Vh independently. Insufficient
    Lambda/Vh injectivity or a rank-deficient global trace system is rejected.
    This numerical rank check is not a mesh-uniform inf-sup proof. Sufficient
    mesh conditions (M1),(M2) and the exceptions of Remark 16 remain relevant.
    The optional local_meshes supplies one validated conforming triangular
    partition per macrocell. It replaces the uniform local_refinement
    geometry without changing Gamma, Lambda, or either Neumann map.

    ``neumann`` maps exterior faces to outward physical q.n. Other exterior
    faces interpolate ``dirichlet`` at Gamma nodes. Pure Neumann data require
    compatibility; ``mean_pressure`` then fixes the volume average of the full
    reconstructed pressure. Nonhomogeneous and mixed data are the consistent
    extension of the paper's homogeneous Dirichlet examples.
    """
    if not isinstance(mesh, (TriangleMesh, PolygonMesh)):
        raise TypeError("MH2M requires a TriangleMesh or PolygonMesh")
    degree = positive_int(degree, "degree")
    refinement = positive_int(local_refinement, "local_refinement")
    order = positive_int(quadrature_order, "quadrature_order")
    gamma = PressureTraceSpace(mesh) if pressure_trace is None else pressure_trace
    flux = SkeletonSpace(cast(TriangleMesh, mesh)) if flux_space is None else flux_space
    if gamma.mesh is not mesh or flux.mesh is not mesh or flux.components != 1:
        raise ValueError("Gamma and scalar Lambda must use the supplied mesh")
    if any(face.continuous for face in flux.faces):
        raise ValueError("Lambda uses discontinuous face polynomials")
    natural = {} if neumann is None else dict(neumann)
    if any(face not in mesh.boundary_faces for face in natural):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    if local_meshes is not None:
        if len(local_meshes) != len(mesh.cells):
            raise ValueError("local_meshes must contain one partition per macrocell")
        for cell, fine in enumerate(local_meshes):
            validate_submesh(mesh, cell, fine)
    local = tuple(
        _local_problem(
            mesh,
            cell,
            gamma,
            flux,
            degree,
            refinement,
            permeability,
            source,
            order,
            local_solver,
            None if local_meshes is None else local_meshes[cell],
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
    boundary_flux = 0.0
    compatibility_scale = sum(float(np.sum(abs(data.load))) for data in local)
    for face in mesh.boundary_faces:
        ids, space = gamma.face_dofs[face], gamma.faces[face]
        if face in natural:
            t, w = space.quadrature(max(order, max(space.degrees) + 2))
            a, b = mesh.points[mesh.faces[face]]
            weighted_flux = (
                mesh.lengths[face] * w * scalar_values(natural[face], a + t[:, None] * (b - a))
            )
            boundary_flux += float(np.sum(weighted_flux))
            compatibility_scale += float(np.sum(abs(weighted_flux)))
            np.add.at(rhs, ids, -(space.evaluate(t).T @ weighted_flux))
        else:
            trace[ids] = scalar_values(dirichlet, gamma.nodes[ids])
            fixed.extend(ids)
    if fixed and mean_pressure != 0:
        raise ValueError("mean_pressure applies only to pure Neumann problems")
    if not fixed:
        imbalance = sum(float(np.sum(data.load)) for data in local) - boundary_flux
        if abs(imbalance) > 1e-10 * max(compatibility_scale, np.finfo(float).tiny):
            raise ValueError("incompatible Neumann data: source and outward flux integrals differ")
    free = np.setdiff1d(np.arange(gamma.size), fixed)
    forcing = (rhs - matrix @ trace)[free]
    if len(free):
        reduced = matrix[free][:, free]
        if not fixed:
            moments = np.zeros(gamma.size)
            offset = 0.0
            for data in local:
                np.add.at(moments, data.trace_dofs, data.volume_moments @ data.pressure_lift)
                offset += float(data.volume_moments @ data.pressure_source)
            target = mean_pressure * sum(mesh.areas) - offset
            augmented = sparse.bmat(
                [[reduced, moments[:, None]], [moments[None, :], None]], format="csc"
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
        raise ValueError(
            "MH2M original trace equations fail: incompatible Neumann data or unstable spaces"
        )
    pressure = tuple(
        data.pressure_lift @ trace[data.trace_dofs] + data.pressure_source for data in local
    )
    conormal = tuple(
        data.conormal_lift @ trace[data.trace_dofs] + data.conormal_source for data in local
    )
    return MH2MSolution(
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
        residual,
    )
