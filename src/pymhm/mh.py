"""Multiscale Hybrid diffusion with coercive Robin local problems (2024).

The multiplier is lambda=(-K grad(p)-p sigma).n, where
sigma=nu*(x-origin)/2. It is not the physical Darcy normal flux.
The local operators and the Dirichlet condensed operator are symmetric
positive definite under the stated ellipticity and trace-compatibility
conditions. Physical Neumann conditions use a symmetric boundary-pressure
extension; its global system is indefinite.
"""

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.darcy import DarcySolution
from pymhm.elements import tensor_values, triangle_quadrature
from pymhm.hybrid import HybridSolution, HybridSystem, LocalAssembly, LocalProblem
from pymhm.lagrange import nodal_space, scalar_operators, tabulate, trace_coupling
from pymhm.mesh import FloatArray, SkeletonSpace, TriangleMesh, positive_int
from pymhm.mh_boundary import RobinBoundaryFace, solve_robin_neumann
from pymhm.polygon import PolygonMesh
from pymhm.reservoir import CartesianCellField
from pymhm.scalar_boundary import edge_basis, scalar_trace
from pymhm.scalar_trace_integration import integrate_dirichlet_trace


def _volume_weights(fine: TriangleMesh, degree: int) -> FloatArray:
    """Integrate every local nodal basis function for the physical pressure mean."""
    bary, weights = triangle_quadrature(degree + 1)
    dofs, points, basis, _, _ = tabulate(fine, degree, bary)
    result = np.zeros(len(points))
    local = np.einsum("q,qi,t->ti", weights, basis, fine.areas)
    np.add.at(result, dofs.ravel(), local.ravel())
    return result


def _physical_neumann_solve(
    system: HybridSystem,
    skeleton: SkeletonSpace,
    local_meshes: tuple[TriangleMesh, ...],
    natural: dict[int, Any],
    degree: int,
    order: int,
    parameter: float,
    origin: FloatArray,
    mean_pressure: float,
    solver: str,
    precision: Literal["double", "extended"],
) -> tuple[HybridSolution, dict[int, FloatArray], Any, FloatArray]:
    """Append boundary pressure moments, preserving symmetry even when sigma.n=0.

    With S=B.T A^-1 B, t=B.T A^-1 f and r=sigma.n on a straight face,
    the equations are S lambda + M rho=t-g_D and M lambda+r M rho=h_N.
    Thus rho is the Lambda-tested pressure, and h_N is the outward physical
    flux functional. This is an indefinite symmetric boundary extension of
    the Dirichlet MH formulation; the local Robin matrices are unchanged.
    """
    forms = []
    for face in sorted(natural):
        space = skeleton.faces[face]
        t, weights = space.quadrature(max(order, max(space.degrees) + 2))
        basis = space.evaluate(t)
        mass = basis.T @ ((weights * skeleton.mesh.lengths[face])[:, None] * basis)
        midpoint = skeleton.mesh.points[skeleton.mesh.faces[face]].mean(axis=0)
        coefficient = parameter * ((midpoint - origin) @ skeleton.mesh.normals[face]) / 2
        load = integrate_dirichlet_trace(
            skeleton, local_meshes, natural[face], degree, order, faces=(face,)
        )[skeleton.dofs(face)]
        forms.append(
            RobinBoundaryFace(
                face,
                skeleton.dofs(face),
                mass,
                float(coefficient),
                load,
                space.constant_coefficients(),
            )
        )
    return solve_robin_neumann(
        system,
        skeleton.size,
        tuple(forms),
        tuple(_volume_weights(fine, degree) for fine in local_meshes),
        float(np.sum(skeleton.mesh.areas)),
        mean_pressure,
        len(natural) == len(skeleton.mesh.boundary_faces),
        solver,
        precision,
    )


def _ellipticity(material: Any, supplied: Any) -> float:
    """Infer a constant-material bound or require a certified callback bound."""
    if supplied is None:
        if isinstance(material, CartesianCellField):
            data = material.values
            supplied = np.min(data) if data.ndim == 2 else np.min(np.linalg.eigvalsh(data))
        elif callable(material):
            raise ValueError("material callbacks require a certified ellipticity_lower_bound")
        else:
            supplied = np.linalg.eigvalsh(tensor_values(material, np.zeros((1, 2)))).min()
    value = np.asarray(supplied)
    if value.shape != () or np.iscomplexobj(value) or not np.isfinite(value) or value <= 0:
        raise ValueError("ellipticity_lower_bound must be a finite positive scalar")
    return float(value)


def _robin_operator(
    fine: TriangleMesh, degree: int, parameter: float, origin: FloatArray, order: int
) -> Any:
    """Assemble integral_boundary (sigma.n) u v exactly for the affine sigma."""
    count = len(nodal_space(fine, degree)[1])
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    t, weights = leggauss(max(order, degree + 1))
    t, weights = (t + 1) / 2, weights / 2
    basis = edge_basis(degree, t)
    for face in fine.boundary_faces:
        start, end = fine.points[fine.faces[face]]
        points = start + t[:, None] * (end - start)
        sigma_n = parameter * ((points - origin) @ fine.normals[face]) / 2
        ids = np.r_[
            fine.faces[face], len(fine.points) + face * (degree - 1) + np.arange(degree - 1)
        ]
        block = basis.T @ ((weights * fine.lengths[face] * sigma_n)[:, None] * basis)
        rows.extend(np.repeat(ids, len(ids)))
        columns.extend(np.tile(ids, len(ids)))
        values.extend(block.ravel())
    return sparse.coo_matrix((values, (rows, columns)), shape=(count, count)).tocsc()


@dataclass(frozen=True)
class _MHFactory:
    """Portable local Robin assembly, preserving physical source and boundary terms."""

    mesh: Any
    skeleton: SkeletonSpace
    permeability: Any
    source: Any
    degree: int
    refinement: int
    order: int
    parameter: float
    origin: FloatArray
    lower: float

    def __call__(self, cell: int) -> LocalAssembly:
        """Build one coercive local operator, without a mean constraint or coarse mode."""
        fine = self.mesh.submesh(cell, self.refinement)
        # This check catches inconsistent declarations; callback certification remains a premise.
        sampled = np.linalg.eigvalsh(tensor_values(self.permeability, fine.points)).min()
        if sampled < self.lower * (1 - 64 * np.finfo(float).eps):
            raise ValueError("ellipticity_lower_bound exceeds a sampled material eigenvalue")
        stiffness, _, load = scalar_operators(
            fine, self.degree, diffusion=self.permeability, source=self.source, order=self.order
        )
        robin = _robin_operator(fine, self.degree, self.parameter, self.origin, self.order)
        coupling = trace_coupling(self.mesh, cell, fine, self.skeleton, self.degree)
        problem = LocalProblem(stiffness + robin, coupling, load, self.skeleton.cell_dofs(cell))
        return LocalAssembly(problem, (fine, robin))


@dataclass(frozen=True)
class MHSolution:
    """Broken pressure and modified Robin multiplier of the elliptic MH method.

    ``hybrid.trace`` represents (-K grad(p)-p sigma).n in the globally oriented
    skeleton basis. Use ``normal_flux`` for the physical one-sided outward
    flux lambda+p sigma.n. Raw volume fluxes are -K grad(p). These distinctions
    prevent using the Robin multiplier as a conservative Darcy trace.
    """

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    system: HybridSystem
    permeability: Any
    source: Any
    degree: int
    quadrature_order: int
    robin_parameter: float
    origin: FloatArray
    ellipticity_lower_bound: float
    boundary_pressure: dict[int, FloatArray]
    global_matrix: Any
    global_rhs: FloatArray

    @property
    def global_coefficients(self) -> FloatArray:
        """Return Robin lambda followed by pressure on sorted Neumann faces.

        These coefficients satisfy global_matrix/global_rhs before any mean
        augmentation. The system attribute retains the underlying Robin
        condensation, whose response objects reconstruct the volume pressure.
        """
        return np.concatenate((self.hybrid.trace, *self.boundary_pressure.values()))

    def _primal_fields(self) -> DarcySolution:
        """Reuse only the established volume-field integration, not Darcy trace diagnostics."""
        return DarcySolution(
            self.skeleton,
            self.local_meshes,
            self.pressure,
            self.flux,
            self.hybrid,
            "primal",
            self.permeability,
            self.source,
            self.quadrature_order,
            self.degree,
        )

    def l2_error(self, exact: Any, order: int = 8) -> float:
        """Integrate pressure error by independent physical quadrature."""
        return self._primal_fields().l2_error(exact, order)

    def flux_l2_error(self, exact: Any, order: int = 8) -> float:
        """Integrate the raw physical Darcy flux error, not the Robin multiplier error."""
        return self._primal_fields().flux_l2_error(exact, order)

    def normal_flux(self, cell: int, face: int, parameter: Any) -> FloatArray:
        """Evaluate outward physical flux on one incident side of a macroface.

        The Robin multiplier is shared with opposite orientations. Its added
        pressure trace is one-sided; pointwise physical-flux continuity is not
        implied when the discrete pressure jumps.
        """
        mesh = self.skeleton.mesh
        if face not in mesh.cell_faces[cell]:
            raise ValueError("face must be incident to the supplied macrocell")
        side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
        sign = mesh.signs[cell, side]
        t = np.asarray(parameter, dtype=float)
        if t.ndim != 1 or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
            raise ValueError("face parameters must be finite points in [0,1]")
        start, end = mesh.points[mesh.faces[face]]
        points = start + t[:, None] * (end - start)
        pressure = scalar_trace(
            mesh, self.local_meshes[cell], face, self.degree, self.pressure[cell], t
        )
        multiplier = (
            self.skeleton.faces[face].evaluate(t) @ self.hybrid.trace[self.skeleton.dofs(face)]
        )
        sigma_n = self.robin_parameter * ((points - self.origin) @ mesh.normals[face]) / 2
        return sign * (multiplier + pressure * sigma_n)

    def conservation_residuals(self) -> FloatArray:
        """Return integral(lambda+p sigma.n)-integral(f) for every macrocell."""
        values = []
        for response, pressure, (_, robin) in zip(
            self.system.responses, self.pressure, self.system.local_metadata, strict=True
        ):
            problem = response.problem
            values.append(
                np.sum(
                    problem.coupling @ self.hybrid.trace[problem.trace_dofs]
                    + robin @ pressure
                    - problem.load
                )
            )
        return np.asarray(values)


def solve_mh(
    mesh: TriangleMesh | PolygonMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    skeleton: SkeletonSpace | None = None,
    degree: int = 2,
    local_refinement: int = 4,
    quadrature_order: int = 6,
    robin_parameter: float | None = None,
    origin: Any = None,
    ellipticity_lower_bound: float | None = None,
    solver: str = "scipy",
    local_solver: str = "scipy",
    local_refinement_precision: Literal["double", "extended"] = "double",
    refinement_precision: Literal["double", "extended"] = "double",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> MHSolution:
    """Solve diffusion with the 2024 Multiscale Hybrid Robin local formulation.

    Define sigma=nu*(x-origin)/2, with positive ``robin_parameter=nu``. The
    default origin is the lower corner of the bounding box. Let
    C_sigma=max_vertex |x-origin|/2. We require nu<=K_min/(4*C_sigma**2),
    the sufficient coercivity estimate using the actual sigma norm. The default
    is half this bound. This sharpens the paper's conservative substitution of
    the bounding-domain diameter; it does not change its bilinear form.
    Literal scalar/tensor and Cartesian materials supply K_min automatically;
    other callbacks require a certified ``ellipticity_lower_bound``.

    Local Pk spaces and signed skeletal spaces are independent. A fine mesh
    resolving each skeletal segment with two boundary edges gives the mesh
    condition used by the article, with the corresponding compatible degrees.
    An algebraically singular trace choice is rejected by the shared solver.
    Neumann keys identify exterior faces and prescribe the outward physical
    flux q.n, not the Robin multiplier. Remaining exterior faces prescribe
    pressure weakly. Auxiliary boundary pressure moments preserve the Robin
    local operators; this mixed-boundary extension has a symmetric indefinite
    global system. Pure Neumann problems require integral compatibility and
    mean_pressure fixes the volume average of the reconstructed field.
    At nu=0 the elliptic condensation is undefined; use solve_darcy for MHM.
    Assembly and condensation occur together inside the selected workers.
    The explicit refinement-precision options reuse the shared residual-checked
    iterative refinement. Extended accumulation requires a wider long-double
    type; neither option changes the local or global residual tolerance.
    """
    if not isinstance(mesh, (TriangleMesh, PolygonMesh)):
        raise TypeError("MH requires a triangular or polygonal two-dimensional mesh")
    degree = positive_int(degree, "degree")
    refinement = positive_int(local_refinement, "local_refinement")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    geometry = cast(TriangleMesh, mesh)
    skeleton = SkeletonSpace(geometry) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("MH requires a scalar skeleton on the supplied mesh")
    natural = {} if neumann is None else dict(neumann)
    if any(face not in mesh.boundary_faces for face in natural):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    mean_value = np.asarray(mean_pressure)
    if mean_value.shape != () or np.iscomplexobj(mean_value) or not np.isfinite(mean_value):
        raise ValueError("mean_pressure must be finite and real")
    if len(natural) != len(mesh.boundary_faces) and mean_pressure != 0:
        raise ValueError("mean_pressure applies only to pure Neumann problems")
    lower = _ellipticity(permeability, ellipticity_lower_bound)
    point = np.min(mesh.points, axis=0) if origin is None else np.asarray(origin)
    if point.shape != (2,) or np.iscomplexobj(point) or not np.isfinite(point).all():
        raise ValueError("origin must be a finite real two-dimensional point")
    point = np.array(point, dtype=float, copy=True)
    radius = float(np.max(np.linalg.norm(mesh.points - point, axis=1))) / 2
    upper = lower / (4 * radius**2)
    parameter = upper / 2 if robin_parameter is None else robin_parameter
    if (
        np.iscomplexobj(parameter)
        or not np.isfinite(parameter)
        or parameter <= 0
        or parameter > upper
    ):
        raise ValueError(
            "robin_parameter must be positive and satisfy the certified coercivity bound"
        )
    factory = _MHFactory(
        mesh,
        skeleton,
        permeability,
        source,
        degree,
        refinement,
        order,
        float(parameter),
        point,
        lower,
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        local_solver=local_solver,
        local_refinement_precision=local_refinement_precision,
        backend=backend,
        workers=workers,
    )
    local_meshes = tuple(data[0] for data in system.local_metadata)
    boundary = integrate_dirichlet_trace(
        skeleton,
        local_meshes,
        dirichlet,
        degree,
        order,
        faces=tuple(int(f) for f in mesh.boundary_faces if f not in natural),
    )
    system.rhs[: skeleton.size] -= boundary
    system.load_scale[: skeleton.size] += abs(boundary)
    if natural:
        hybrid, boundary_pressure, matrix, rhs = _physical_neumann_solve(
            system,
            skeleton,
            local_meshes,
            natural,
            degree,
            order,
            float(parameter),
            point,
            mean_pressure,
            solver,
            refinement_precision,
        )
    else:
        hybrid = system.solve(solver=solver, refinement_precision=refinement_precision)
        boundary_pressure, matrix, rhs = {}, system.matrix, system.rhs
    flux = []
    for fine, coefficients in zip(local_meshes, hybrid.fields, strict=True):
        dofs, _, _, gradients, _ = tabulate(fine, degree, np.array([[1 / 3] * 3]))
        gradient = np.einsum("tia,ti->ta", gradients[:, 0], coefficients[dofs])
        tensors = tensor_values(permeability, fine.points[fine.cells].mean(axis=1))
        flux.append(-np.einsum("tab,tb->ta", tensors, gradient))
    return MHSolution(
        skeleton,
        local_meshes,
        hybrid.fields,
        tuple(flux),
        hybrid,
        system,
        permeability,
        source,
        degree,
        order,
        float(parameter),
        point,
        lower,
        boundary_pressure,
        matrix,
        rhs,
    )
