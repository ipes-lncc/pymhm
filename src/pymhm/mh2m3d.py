"""Three-field MH²M on tetrahedra with independently resolved interface spaces."""

from dataclasses import dataclass
from math import fsum
from typing import Any, cast

import numpy as np
from scipy import sparse

from pymhm.darcy3d import Darcy3DSolution, TriangularSkeleton, tetra_trace_coupling
from pymhm.hybrid import HybridSolution
from pymhm.mesh import FloatArray, IntArray, positive_int
from pymhm.mh2m import MH2MLocal, _neumann_maps
from pymhm.mh_trace3d import PressureTraceSpace3D, boundary_rules, broken_face_basis
from pymhm.solvers import solve_linear
from pymhm.tetrahedral import TetraMesh, _dyadic, scalar_values_3d, tetra_operators


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
    pairing = np.zeros((coupling.shape[1], len(trace_ids)))
    rows = {}
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        width = len(flux.dofs(int(face)))
        coupling[:, offset : offset + width] *= mesh.signs[cell, side]
        rows[int(face)] = np.arange(offset, offset + width)
        offset += width
    for face, _, _, _, weights, bary in boundary_rules(
        mesh, cell, fine, degree, max(order, gamma.degree + int(flux.degrees.max()) + 2)
    ):
        lam, rho = broken_face_basis(flux, face, bary), gamma.evaluate(face, bary)
        columns = np.searchsorted(trace_ids, gamma.face_dofs[face])
        pairing[np.ix_(rows[face], columns)] += lam.T @ (weights[:, None] * rho)
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


@dataclass(frozen=True)
class MH2M3DSolution:
    """Continuous pressure trace and reconstructed broken tetrahedral fields.

    Each macrocell has independent outward conormal coefficients
    lambda=K grad(p).n. Their negatives are physical normal fluxes. Global
    continuity is tested by Gamma, not pointwise. The local complement has
    boundary mean zero; a global pure-Neumann gauge fixes the volume mean.
    """

    trace_space: PressureTraceSpace3D
    flux_space: TriangularSkeleton
    local: tuple[MH2MLocal, ...]
    trace: FloatArray
    pressure: tuple[FloatArray, ...]
    conormal: tuple[FloatArray, ...]
    matrix: Any
    rhs: FloatArray
    free_dofs: IntArray
    degree: int
    permeability: Any
    source: Any
    quadrature_order: int
    residual: float

    @property
    def local_meshes(self) -> tuple[TetraMesh, ...]:
        """Return each conforming local tetrahedral partition."""
        return tuple(cast(TetraMesh, data.mesh) for data in self.local)

    def _fields(self) -> Darcy3DSolution:
        """Reuse only volume-field evaluation; this trace is not a Darcy flux skeleton."""
        hybrid = HybridSolution(self.trace, (), self.pressure, self.residual, np.empty(0))
        return Darcy3DSolution(
            self.flux_space,
            self.local_meshes,
            self.pressure,
            hybrid,
            self.degree,
            self.permeability,
            self.source,
            self.quadrature_order,
        )

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate complete reconstructed pressure and raw physical flux."""
        return self._fields().evaluate(cell, bary)

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the broken pressure error with independent tetrahedral quadrature."""
        return self._fields().l2_error(exact, order)

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the raw physical flux error, keeping the full permeability tensor."""
        return self._fields().flux_l2_error(exact, order)

    def conservation_residuals(self) -> FloatArray:
        """Return integrated outward physical conormal flux minus source per macrocell."""
        return np.array(
            [
                -data.flux_integrals @ lam - data.load.sum()
                for data, lam in zip(self.local, self.conormal, strict=True)
            ]
        )

    def trace_moment_residuals(self) -> tuple[FloatArray, ...]:
        """Return Lambda-tested differences of the local pressure and Gamma trace."""
        return tuple(
            data.boundary_coupling.T @ p - data.trace_pairing @ self.trace[data.trace_dofs]
            for data, p in zip(self.local, self.pressure, strict=True)
        )

    def local_equation_residuals(self) -> tuple[FloatArray, ...]:
        """Return original nodal equations A p-B lambda-f before condensation."""
        return tuple(
            data.stiffness @ p - data.boundary_coupling @ lam - data.load
            for data, p, lam in zip(self.local, self.pressure, self.conormal, strict=True)
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
