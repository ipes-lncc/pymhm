"""Three-field multiscale Darcy method of de Barros, Madureira and Valentin.

The pressure trace Gamma is globally continuous, while Lambda has independent
outward conormal coefficients on each macrocell. Equations (25)--(29) of the
2026 preprint define two local Neumann inverses and a global pressure-trace
system. The paper's lambda=A grad(p).n is minus the physical Darcy flux q.n.
"""

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy import sparse

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.triangle import element_tabulate, scalar_operators, trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.linalg.linear import LinearSolveError, factorize, solve_linear
from pymhm.materials.evaluation import scalar_values, tensor_values, vector_values
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.refinement import validate_submesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class PressureTraceSpace:
    """Globally continuous nodal polynomials on triangular or polygonal edges.

    ``faces`` may prescribe different positive degrees and segment partitions.
    Endpoints of different macrofaces share the mesh vertex unknown. Interior
    edge nodes belong to that face only. This is Gamma, not the flux skeleton.
    """

    mesh: TriangleMesh | PolygonMesh
    faces: tuple[FaceSpace, ...] = ()
    nodes: FloatArray = field(init=False, repr=False)
    face_dofs: tuple[IntArray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Assign shared vertices and private face-interior interpolation nodes."""
        if not isinstance(self.mesh, (TriangleMesh, PolygonMesh)):
            raise TypeError("pressure traces require a TriangleMesh or PolygonMesh")
        faces = self.faces
        faces = (
            tuple(FaceSpace.uniform(1, continuous=True) for _ in self.mesh.faces)
            if not faces
            else tuple(faces)
        )
        if len(faces) != len(self.mesh.faces) or any(
            not isinstance(f, FaceSpace) or not f.continuous for f in faces
        ):
            raise ValueError(
                "Gamma requires one continuous positive-degree FaceSpace per macroface"
            )
        points, dofs = list(self.mesh.points), []
        for edge, space in zip(self.mesh.faces, faces, strict=True):
            t = np.r_[
                space.breaks,
                np.concatenate(
                    [
                        a + (b - a) * np.arange(1, k) / k
                        for a, b, k in zip(
                            space.breaks[:-1], space.breaks[1:], space.degrees, strict=True
                        )
                    ]
                ),
            ]
            ids = np.empty(space.size, dtype=np.int64)
            ids[0], ids[len(space.breaks) - 1] = edge
            for j in range(space.size):
                if j not in (0, len(space.breaks) - 1):
                    ids[j] = len(points)
                    points.append(
                        (1 - t[j]) * self.mesh.points[edge[0]] + t[j] * self.mesh.points[edge[1]]
                    )
            dofs.append(ids)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "nodes", np.asarray(points))
        object.__setattr__(self, "face_dofs", tuple(dofs))

    @classmethod
    def uniform(
        cls, mesh: TriangleMesh | PolygonMesh, degree: int = 1, segments: int = 1
    ) -> "PressureTraceSpace":
        """Use the same continuous nodal degree and subdivision on every face."""
        face = FaceSpace.uniform(degree, segments, continuous=True)
        return cls(mesh, tuple(face for _ in mesh.faces))

    @property
    def size(self) -> int:
        """Return the number of shared pressure-trace unknowns before boundary elimination."""
        return len(self.nodes)

    def cell_dofs(self, cell: int) -> IntArray:
        """Return unique Gamma indices touching one macrocell, in increasing order."""
        return np.unique(np.concatenate([self.face_dofs[f] for f in self.mesh.cell_faces[cell]]))


@dataclass(frozen=True)
class MH2MLocal:
    """Local operators and Eq. (29) lifts in declared nodal/Legendre coordinates.

    ``conormal_lift`` and ``conormal_source`` use the paper's lambda convention.
    ``neumann_energy`` acts on zero-boundary-average Lambda coordinates.
    Boundary averages, rather than volume averages, define the local complement.
    """

    mesh: TriangleMesh | TetraMesh
    trace_dofs: IntArray
    stiffness: Any
    load: FloatArray
    boundary_coupling: FloatArray
    trace_pairing: FloatArray
    flux_integrals: FloatArray
    zero_mean_basis: FloatArray
    neumann_energy: FloatArray
    source_lift: FloatArray
    pressure_lift: FloatArray
    pressure_source: FloatArray
    conormal_lift: FloatArray
    conormal_source: FloatArray
    volume_moments: FloatArray
    trace_matrix: FloatArray
    trace_rhs: FloatArray


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
    pairing = np.zeros((coupling.shape[1], len(trace_ids)))
    constant = np.zeros(coupling.shape[1])
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        lam, rho = flux.faces[face], gamma.faces[face]
        width = lam.size
        coupling[:, offset : offset + width] *= mesh.signs[cell][side]
        constant[offset : offset + width] = lam.constant_coefficients()
        cuts = tuple(sorted(set(lam.breaks) | set(rho.breaks)))
        t, w = FaceSpace(cuts, (0,) * (len(cuts) - 1)).quadrature(
            max(lam.degrees) + max(rho.degrees) + 2
        )
        block = lam.evaluate(t).T @ (w[:, None] * rho.evaluate(t)) * mesh.lengths[face]
        pairing[
            np.ix_(
                np.arange(offset, offset + width), np.searchsorted(trace_ids, gamma.face_dofs[face])
            )
        ] = block
        offset += width
    return _neumann_maps(
        fine, trace_ids, stiffness, mass, load, coupling, pairing, constant, solver
    )


def _neumann_maps(
    fine: TriangleMesh | TetraMesh,
    trace_ids: IntArray,
    stiffness: Any,
    mass: Any,
    load: FloatArray,
    coupling: FloatArray,
    pairing: FloatArray,
    constant: FloatArray,
    solver: str,
) -> MH2MLocal:
    """Construct the dimension-independent Neumann inverses and Eq. (29) lifts.

    Coupling has the unsigned outward-conormal convention. Boundary integrals
    supply its physical mean; mass supplies the volume mean used only by gauges.
    """
    integrals = coupling.sum(axis=0)
    boundary = coupling @ constant
    perimeter = float(integrals @ constant)
    pivot = int(np.argmax(abs(integrals)))
    others = np.delete(np.arange(len(integrals)), pivot)
    zero_mean = np.eye(len(integrals))[:, others]
    zero_mean[pivot] = -integrals[others] / integrals[pivot]
    rhs = np.column_stack((coupling @ zero_mean, load))
    rhs -= boundary[:, None] * (rhs.sum(axis=0) / perimeter)
    lifts = np.zeros_like(rhs)
    with factorize(stiffness[1:, 1:], solver=solver) as factor:
        lifts[1:] = factor.solve(rhs[1:])
    lifts -= (boundary @ lifts / perimeter)[None, :]
    neumann, eta = lifts[:, :-1], lifts[:, -1]
    energy = (coupling @ zero_mean).T @ neumann
    energy = (energy + energy.T) / 2
    moments = zero_mean.T @ pairing
    source_moments = (coupling @ zero_mean).T @ eta
    try:
        with factorize(energy, solver=solver) as factor:
            inverse = factor.solve(np.column_stack((moments, source_moments)))
    except LinearSolveError as error:
        raise ValueError(
            "Lambda and Vh violate local Neumann injectivity (Assumption A)"
        ) from error
    response, source_response = inverse[:, :-1], inverse[:, -1]
    mean_row = constant @ pairing / perimeter
    pressure_lift = neumann @ response + mean_row[None, :]
    pressure_source = eta - neumann @ source_response
    conormal_lift = zero_mean @ response
    conormal_source = -constant * load.sum() / perimeter - zero_mean @ source_response
    trace_matrix = moments.T @ response
    trace_matrix = (trace_matrix + trace_matrix.T) / 2
    trace_rhs = -pairing.T @ conormal_source
    return MH2MLocal(
        fine,
        trace_ids,
        stiffness,
        load,
        coupling,
        pairing,
        integrals,
        zero_mean,
        energy,
        eta,
        pressure_lift,
        pressure_source,
        conormal_lift,
        conormal_source,
        np.asarray(mass.sum(axis=1)).ravel(),
        trace_matrix,
        trace_rhs,
    )


@dataclass(frozen=True)
class MH2MSolution:
    """Pressure trace, broken local pressure, and outward physical flux moments.

    ``conormal`` stores lambda=A grad(p).n separately on each macrocell; its
    negative is the physical normal flux. Continuity holds against Gamma test
    functions. Neither pointwise normal continuity nor fine-cell equilibrium
    follows from this weak condition. Raw fluxes are -A grad(p_h).
    """

    trace_space: PressureTraceSpace
    flux_space: SkeletonSpace
    local: tuple[MH2MLocal, ...]
    trace: FloatArray
    pressure: tuple[FloatArray, ...]
    conormal: tuple[FloatArray, ...]
    matrix: Any
    rhs: FloatArray
    free_dofs: IntArray
    degree: int
    permeability: Any
    residual: float

    def conservation_residuals(self) -> FloatArray:
        """Return integral(q.n)-integral(f) per macrocell from conormal moments."""
        return np.array(
            [
                -data.flux_integrals @ lam - data.load.sum()
                for data, lam in zip(self.local, self.conormal, strict=True)
            ]
        )

    def trace_moment_residuals(self) -> tuple[FloatArray, ...]:
        """Return Lambda-tested differences between local pressure and Gamma trace."""
        return tuple(
            data.boundary_coupling.T @ p - data.trace_pairing @ self.trace[data.trace_dofs]
            for data, p in zip(self.local, self.pressure, strict=True)
        )

    def local_equation_residuals(self) -> tuple[FloatArray, ...]:
        """Return original nodal equations A p-B lambda-f, without condensation."""
        return tuple(
            data.stiffness @ p - data.boundary_coupling @ lam - data.load
            for data, p, lam in zip(self.local, self.pressure, self.conormal, strict=True)
        )

    def _error(self, exact: Any, order: int, derivative: bool, flux: bool) -> float:
        """Integrate a physical field error, cutting Cartesian coefficient interfaces."""
        total = 0.0
        for data, coefficients in zip(self.local, self.pressure, strict=True):
            bary, weights, material = material_triangle_quadrature(
                cast(TriangleMesh, data.mesh),
                self.permeability,
                positive_int(order, "quadrature order"),
            )
            dofs, _, basis, gradients, _ = element_tabulate(
                cast(TriangleMesh, data.mesh), self.degree, bary
            )
            points = np.einsum("tqi,tia->tqa", bary, data.mesh.points[data.mesh.cells])
            if derivative:
                field_values = np.einsum("tqia,ti->tqa", gradients, coefficients[dofs])
                if flux:
                    tensor = tensor_values(material, points.reshape(-1, 2)).reshape(
                        *weights.shape, 2, 2
                    )
                    field_values = -np.einsum("tqab,tqb->tqa", tensor, field_values)
                defect = field_values - vector_values(exact, points.reshape(-1, 2)).reshape(
                    field_values.shape
                )
                squared = np.sum(defect**2, axis=-1)
            else:
                values = np.einsum("tqi,ti->tq", basis, coefficients[dofs])
                defect = values - scalar_values(exact, points.reshape(-1, 2)).reshape(weights.shape)
                squared = defect**2
            total += float(np.sum(cast(TriangleMesh, data.mesh).areas[:, None] * weights * squared))
        return float(np.sqrt(total))

    def l2_error(self, exact: Any, order: int = 8) -> float:
        """Return the broken pressure L2 error using independent quadrature."""
        return self._error(exact, order, False, False)

    def gradient_l2_error(self, exact: Any, order: int = 8) -> float:
        """Return the unweighted broken H1 seminorm error."""
        return self._error(exact, order, True, False)

    def flux_l2_error(self, exact: Any, order: int = 8) -> float:
        """Return the raw physical Darcy flux L2 error; this is not the trace error."""
        return self._error(exact, order, True, True)


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
