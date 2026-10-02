"""Two-level residual estimators for Stokes, Brinkman and Oseen velocity-pressure fields."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from pymhm.elements import p1_geometry, triangle_quadrature, vector_values
from pymhm.flow import _resistance_values
from pymhm.lagrange import nodal_space, reference_basis, tabulate
from pymhm.mesh import FloatArray, IntArray, TriangleMesh, positive_int
from pymhm.vector import VectorSolution


def _evaluate(
    solution: VectorSolution,
    macro: int,
    points: FloatArray,
    cells: IntArray,
    tables: tuple[FloatArray, IntArray, IntArray] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate one-sided velocity, its gradient and pressure in specified fine cells."""
    mesh = solution.local_meshes[macro]
    if tables is None:
        tables = (
            p1_geometry(mesh)[0],
            nodal_space(mesh, solution.degree)[0],
            nodal_space(mesh, solution.pressure_degree)[0],
        )
    geometry, velocity_dofs, pressure_dofs = tables
    vertices = mesh.points[mesh.cells[cells]]
    bary = np.einsum("nva,na->nv", geometry[cells], points - vertices[:, 0])
    bary[:, 0] += 1
    basis, derivatives, _ = reference_basis(solution.degree, bary)
    dofs = velocity_dofs[cells]
    coefficients = solution.values[macro][dofs]
    value = np.einsum("ni,nia->na", basis, coefficients)
    gradient = np.einsum("niv,nib,nba->nva", coefficients, derivatives, geometry[cells])
    pressure_basis = reference_basis(solution.pressure_degree, bary)[0]
    pressure = np.einsum("ni,ni->n", pressure_basis, solution.pressure[macro][pressure_dofs[cells]])
    return value, gradient, pressure


def _boundary_intervals(
    coarse: TriangleMesh,
    macro: int,
    fine: TriangleMesh,
    lengths: FloatArray | None = None,
) -> dict[int, list[tuple[float, float, int, int]]]:
    """Map every boundary microedge to an oriented macroface and its adjacent cell."""
    lengths = coarse.lengths if lengths is None else lengths
    result: dict[int, list[tuple[float, float, int, int]]] = {}
    for macroface in coarse.cell_faces[macro]:
        vertices = coarse.points[coarse.faces[macroface]]
        tangent = vertices[1] - vertices[0]
        length = lengths[macroface]
        entries = []
        for face in fine.boundary_faces:
            points = fine.points[fine.faces[face]]
            parameter = (points - vertices[0]) @ tangent / length**2
            distance = np.linalg.norm(points - vertices[0] - parameter[:, None] * tangent, axis=1)
            if np.max(distance) <= 128 * np.finfo(float).eps * length:
                entries.append(
                    (
                        float(parameter.min()),
                        float(parameter.max()),
                        int(fine.face_cells[face, 0]),
                        int(face),
                    )
                )
        result[int(macroface)] = entries
    return result


def _edge_points(
    vertices: FloatArray, breaks: FloatArray, order: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Integrate along specified subintervals, with weights normalized to edge length."""
    x, weights = leggauss(order)
    lo, hi = breaks[:-1, None], breaks[1:, None]
    parameters = (lo + (hi - lo) * (x + 1) / 2).ravel()
    weights = ((hi - lo) * weights / 2).ravel()
    points = vertices[0] + parameters[:, None] * (vertices[1] - vertices[0])
    return parameters, points, weights


@dataclass(frozen=True)
class FlowEstimator:
    """Quadrature-evaluated first/second-level residual contributions.

    ``face_squared`` stores eta_1,Ftilde², with the original macroface length
    in the denominator. Interior faces count twice in eta_1, as in the sum
    over macrocell boundaries. ``local_squared`` stores unscaled eta_2,K².
    ``fine_squared`` partitions each eta_2,K² among its fine cells, splitting
    an interior fine-face contribution equally between its two neighbors.
    Oseen's reported eta_2 includes 2**(-2*ell); the Stokes/Brinkman estimator
    does not. These are estimators up to the constants/higher-order terms in
    their respective theorems, not certified numerical upper bounds.
    """

    solution: VectorSolution
    face_squared: tuple[FloatArray, ...]
    local_squared: FloatArray
    volume_squared: FloatArray
    divergence_squared: FloatArray
    traction_squared: FloatArray
    second_level_scale: float
    quadrature_order: int
    fine_squared: tuple[FloatArray, ...] = ()

    @property
    def eta1(self) -> float:
        """Return the jump/boundary mismatch indicator with macrocell multiplicity."""
        neighbors = self.solution.skeleton.mesh.face_cells
        return float(
            np.sqrt(
                sum(
                    value.sum() * (1 + (pair[1] >= 0))
                    for value, pair in zip(self.face_squared, neighbors, strict=True)
                )
            )
        )

    @property
    def eta2(self) -> float:
        """Return the literature-specific scaled second-level residual indicator."""
        return self.second_level_scale * float(np.sqrt(self.local_squared.sum()))

    @property
    def total(self) -> float:
        """Return eta_1+eta_2, not the root of their squared sum."""
        return self.eta1 + self.eta2

    def mixed_error(
        self, velocity: Any, velocity_gradient: Any, pressure: Any, *, order: int = 8
    ) -> float:
        """Integrate ||(u-uh,p-ph)||_(V×Q), V²=diam(Omega)^(-2)L2²+H1²."""
        bary, weights = triangle_quadrature(order)
        coarse = self.solution.skeleton.mesh
        diameter = float(pdist(coarse.points[ConvexHull(coarse.points).vertices]).max())
        total = 0.0
        from pymhm.elements import scalar_values

        for macro, fine in enumerate(self.solution.local_meshes):
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            flat = points.reshape(-1, 2)
            cells = np.repeat(np.arange(len(fine.cells)), len(weights))
            values, gradient, p = _evaluate(self.solution, macro, flat, cells)
            exact_gradient = (
                np.asarray(velocity_gradient(flat), dtype=float)
                if callable(velocity_gradient)
                else np.asarray(velocity_gradient, dtype=float)
            )
            difference = (
                np.sum((values - vector_values(velocity, flat)) ** 2, axis=1) / diameter**2
                + np.sum((gradient - exact_gradient) ** 2, axis=(1, 2))
                + (p - scalar_values(pressure, flat)) ** 2
            )
            total += float(fine.areas @ (difference.reshape(len(fine.cells), -1) @ weights))
        return float(np.sqrt(total))


def estimate_flow_error(
    solution: VectorSolution,
    *,
    viscosity: float = 1.0,
    drag: Any = 0.0,
    advection: Any = (0.0, 0.0),
    source: Any = (0.0, 0.0),
    dirichlet: Any = (0.0, 0.0),
    full_dirichlet: bool,
    variant: Literal["oseen-2021", "stokes-brinkman-2021"] = "oseen-2021",
    quadrature_order: int = 8,
) -> FlowEstimator:
    """Evaluate L15 Eqs. (33)--(34) or L14 Eqs. (4.1)--(4.7).

    Supply the same physical coefficients and Dirichlet data used in the
    solve. The formula assumes full Dirichlet boundaries, constant positive
    viscosity, and uniform skeletal polynomial degree ell. Piecewise trace
    partitions and nonmatching local submesh refinements are supported; edge
    integration is cut at every fine and skeletal breakpoint. The supplied
    full_dirichlet declaration is explicit because VectorSolution does not
    retain boundary data. Volume integration is Gaussian; a discontinuous
    material must therefore be aligned with local cells for this estimator.
    The Oseen theorem additionally assumes gamma-div(beta)/2>0. A finite
    estimator for a case outside that hypothesis is not a reliability proof.
    """
    if full_dirichlet is not True:
        raise ValueError("the published flow estimator requires full_dirichlet=True")
    if not np.isfinite(viscosity) or viscosity <= 0:
        raise ValueError("viscosity must be finite and positive")
    if variant not in ("oseen-2021", "stokes-brinkman-2021"):
        raise ValueError("unknown flow estimator variant")
    if variant == "stokes-brinkman-2021" and (callable(advection) or np.any(advection)):
        raise ValueError("the Stokes-Brinkman estimator excludes advection")
    skeleton, coarse = solution.skeleton, solution.skeleton.mesh
    degrees = {degree for face in skeleton.faces for degree in face.degrees}
    if len(degrees) != 1:
        raise ValueError("the published estimator requires a uniform skeletal degree")
    if not solution.pressure:
        raise ValueError("the flow estimator requires a velocity-pressure solution")
    order = positive_int(quadrature_order, "quadrature_order", solution.degree + 1)
    bary, weights = triangle_quadrature(order)
    volume, divergence, traction = np.zeros((3, len(coarse.cells)))
    fine_squared = []
    # Geometry and topology depend on the local mesh, not on the edge points.
    # Reuse them only within this estimator call; no mutable/global cache is kept.
    evaluation = [
        (
            p1_geometry(fine)[0],
            nodal_space(fine, solution.degree)[0],
            nodal_space(fine, solution.pressure_degree)[0],
        )
        for fine in solution.local_meshes
    ]
    for tables in evaluation:
        for table in tables:
            table.setflags(write=False)
    coarse_lengths = coarse.lengths
    coarse_lengths.setflags(write=False)
    boundaries = [
        _boundary_intervals(coarse, cell, fine, coarse_lengths)
        for cell, fine in enumerate(solution.local_meshes)
    ]
    for macro, fine in enumerate(solution.local_meshes):
        fine_areas, fine_lengths, fine_normals = fine.areas, fine.lengths, fine.normals
        for geometry in (fine_areas, fine_lengths, fine_normals):
            geometry.setflags(write=False)
        dofs, _, basis, grad_basis, hessian = tabulate(fine, solution.degree, bary)
        pdofs, _, _, pgrad_basis, _ = tabulate(fine, solution.pressure_degree, bary)
        coefficients = solution.values[macro][dofs]
        u = np.einsum("qi,tia->tqa", basis, coefficients)
        grad = np.einsum("tqib,tia->tqab", grad_basis, coefficients)
        laplacian = np.einsum("tqi,tia->tqa", np.trace(hessian, axis1=-2, axis2=-1), coefficients)
        gradp = np.einsum("tqia,ti->tqa", pgrad_basis, solution.pressure[macro][pdofs])
        physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        points = physical.reshape(-1, 2)
        beta = vector_values(advection, points).reshape(u.shape)
        gamma = _resistance_values(drag, points).reshape(*u.shape, 2)
        residual = (
            viscosity * laplacian
            - np.einsum("tqab,tqb->tqa", grad, beta)
            - np.einsum("tqab,tqb->tqa", gamma, u)
            - gradp
            + vector_values(source, points).reshape(u.shape)
        )
        h = fine_lengths[fine.cell_faces].max(axis=1)
        volume[macro] = float((fine_areas * h**2) @ (np.sum(residual**2, axis=2) @ weights))
        divergence[macro] = float(fine_areas @ (np.trace(grad, axis1=-2, axis2=-1) ** 2 @ weights))
        cell_squared = fine_areas * (
            h**2 * (np.sum(residual**2, axis=2) @ weights)
            + np.trace(grad, axis1=-2, axis2=-1) ** 2 @ weights
        )
        boundary_lookup = {
            entry[3]: (face, entry)
            for face, entries in boundaries[macro].items()
            for entry in entries
        }
        for face, neighbors in enumerate(fine.face_cells):
            vertices = fine.points[fine.faces[face]]
            breaks = np.array([0.0, 1.0])
            if neighbors[1] < 0:
                macroface, interval = boundary_lookup[face]
                origin, tangent = coarse.points[coarse.faces[macroface]]
                tangent = tangent - origin
                parameters = (vertices - origin) @ tangent / coarse_lengths[macroface] ** 2
                cuts = (np.array(skeleton.faces[macroface].breaks) - parameters[0]) / (
                    parameters[1] - parameters[0]
                )
                breaks = np.unique(np.r_[breaks, cuts[(cuts > 0) & (cuts < 1)]])
            _, points, edge_weights = _edge_points(vertices, breaks, order)
            normal = fine_normals[face]
            beta_normal = vector_values(advection, points) @ normal
            u, grad, p = _evaluate(
                solution, macro, points, np.full(len(points), neighbors[0]), evaluation[macro]
            )
            stress = (
                viscosity * (grad @ normal) - p[:, None] * normal - u * beta_normal[:, None] / 2
            )
            if neighbors[1] >= 0:
                u, grad, p = _evaluate(
                    solution, macro, points, np.full(len(points), neighbors[1]), evaluation[macro]
                )
                stress -= (
                    viscosity * (grad @ normal) - p[:, None] * normal - u * beta_normal[:, None] / 2
                )
            else:
                parameter = np.clip(
                    (points - origin) @ tangent / coarse_lengths[macroface] ** 2, 0, 1
                )
                trace = skeleton.faces[macroface].evaluate(parameter) @ solution.hybrid.trace[
                    skeleton.dofs(macroface)
                ].reshape(-1, 2)
                sign = 1 if coarse.face_cells[macroface, 0] == macro else -1
                stress += sign * trace
            contribution = fine_lengths[face] ** 2 * float(edge_weights @ np.sum(stress**2, axis=1))
            traction[macro] += contribution
            owners = neighbors[neighbors >= 0]
            cell_squared[owners] += contribution / len(owners)
        fine_squared.append(cell_squared)
    first = []
    for face, neighbors in enumerate(coarse.face_cells):
        space = skeleton.faces[face]
        cuts = list(space.breaks)
        for macro in neighbors[neighbors >= 0]:
            cuts.extend(value for entry in boundaries[macro][face] for value in entry[:2])
        cuts = np.unique(np.clip(cuts, 0, 1))
        parameters, points, weights = _edge_points(coarse.points[coarse.faces[face]], cuts, order)
        values = []
        for macro in neighbors[neighbors >= 0]:
            cells = np.empty(len(points), dtype=int)
            for lo, hi, cell, _ in boundaries[macro][face]:
                cells[(parameters >= lo - 1e-13) & (parameters <= hi + 1e-13)] = cell
            values.append(_evaluate(solution, int(macro), points, cells, evaluation[macro])[0])
        residual = (
            (values[0] - values[1]) / 2
            if len(values) == 2
            else vector_values(dirichlet, points) - values[0]
        )
        segments = np.minimum(
            np.searchsorted(space.breaks, parameters, side="right") - 1, len(space.degrees) - 1
        )
        # ds/H_F cancels the original face length, including on refined segments.
        first.append(
            np.bincount(
                segments,
                weights=weights * np.sum(residual**2, axis=1),
                minlength=len(space.degrees),
            )
        )
    scale = 2.0 ** (-2 * degrees.pop()) if variant == "oseen-2021" else 1.0
    return FlowEstimator(
        solution,
        tuple(first),
        volume + divergence + traction,
        volume,
        divergence,
        traction,
        scale,
        order,
        tuple(fine_squared),
    )
