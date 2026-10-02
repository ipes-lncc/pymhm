"""Classical conforming Cartesian Qk diffusion for independently refined baselines."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.elements import scalar_values, tensor_values
from pymhm.mesh import FloatArray, positive_int
from pymhm.quadrilateral import CartesianMacroMesh, qk_basis, qk_space, quadrilateral_operators
from pymhm.solvers import solve_linear


@dataclass(frozen=True)
class ConformingQuadrilateralSolution:
    """One global continuous Qk field, with physical evaluation at arbitrary points."""

    mesh: CartesianMacroMesh
    degree: int
    pressure: FloatArray
    permeability: Any
    residual: float

    def evaluate(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return pressure and gradient without interpolation or material averaging."""
        raw = np.asarray(points)
        if np.iscomplexobj(raw) or raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
            raise ValueError("evaluation points must be finite real XY pairs")
        points = np.asarray(raw, dtype=float)
        origin = self.mesh.points[0]
        coordinates = (points - origin) / self.mesh.spacing
        counts = np.array([self.mesh.nx, self.mesh.ny])
        tolerance = 64 * np.finfo(float).eps * counts
        if np.any(coordinates < -tolerance) or np.any(coordinates > counts + tolerance):
            raise ValueError("evaluation points lie outside the rectangular mesh")
        indices = np.minimum(np.maximum(coordinates.astype(int), 0), counts - 1)
        reference = np.clip(coordinates - indices, 0, 1)
        width = self.mesh.nx * self.degree + 1
        offsets = np.array(
            [j * width + i for j in range(self.degree + 1) for i in range(self.degree + 1)]
        )
        node_ids = (indices[:, 1] * width + indices[:, 0])[:, None] * self.degree + offsets
        basis, gradients = qk_basis(self.degree, reference)
        values = self.pressure[node_ids]
        return np.einsum("qi,qi->q", basis, values), np.einsum(
            "qi,qia->qa", values, gradients / self.mesh.spacing
        )

    def physical_flux(self, points: FloatArray) -> FloatArray:
        """Evaluate -K grad(p) using the coefficient's declared pointwise convention."""
        gradient = self.evaluate(points)[1]
        return -np.einsum("qab,qb->qa", tensor_values(self.permeability, points), gradient)


def solve_conforming_quadrilateral(
    mesh: CartesianMacroMesh,
    *,
    degree: int = 2,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    quadrature_order: int = 6,
    solver: str = "scipy",
    refinement_precision: Literal["double", "extended"] = "double",
) -> ConformingQuadrilateralSolution:
    """Solve classical conforming diffusion with strong Dirichlet boundary data.

    Neumann entries are outward physical fluxes on boundary face indices of
    ``mesh``. The remaining exterior faces carry Dirichlet data. Pure Neumann
    problems retain the constant kernel through a physical mean constraint and
    reject incompatible total load. This solver has no MHM skeleton or local
    condensation; its entire conforming matrix is assembled and solved globally.
    """
    degree = positive_int(degree, "degree")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 1)
    matrix, mass, load = quadrilateral_operators(
        mesh, degree, permeability=permeability, source=source, order=order
    )
    dofs, nodes = qk_space(mesh, degree)
    natural = {} if neumann is None else neumann
    if any(face not in mesh.boundary_faces for face in natural):
        raise ValueError("Neumann keys must identify boundary faces")
    fixed = np.zeros(len(nodes), dtype=bool)
    gauss, weights = leggauss(order)
    t = (gauss + 1) / 2
    for face in mesh.boundary_faces:
        start, end = mesh.points[mesh.faces[face]]
        tangent = end - start
        cell = mesh.face_cells[face, 0]
        if face in natural:
            physical = start + t[:, None] * tangent
            reference = (physical - mesh.points[mesh.cells[cell, 0]]) / mesh.spacing
            basis = qk_basis(degree, reference)[0]
            load[dofs[cell]] -= mesh.lengths[face] * (
                basis.T @ (weights / 2 * scalar_values(natural[int(face)], physical))
            )
        else:
            side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
            edge_nodes = (
                np.arange(degree + 1),
                np.arange(degree + 1) * (degree + 1) + degree,
                degree * (degree + 1) + np.arange(degree + 1),
                np.arange(degree + 1) * (degree + 1),
            )[side]
            fixed[dofs[cell, edge_nodes]] = True
    pressure = np.zeros(len(nodes))
    if fixed.any():
        pressure[fixed] = scalar_values(dirichlet, nodes[fixed])
        free = np.flatnonzero(~fixed)
        if len(free):
            solved = solve_linear(
                matrix[free][:, free],
                (load - matrix @ pressure)[free],
                solver=solver,
                refinement_precision=refinement_precision,
            )
            pressure = pressure.astype(solved.dtype)
            pressure[free] = solved
    else:
        if not np.isfinite(mean_pressure):
            raise ValueError("mean pressure must be finite")
        if abs(load.sum()) > 1e-10 * max(float(np.sum(abs(load))), np.finfo(float).tiny):
            raise ValueError("pure-Neumann source and physical flux are incompatible")
        mean = np.asarray(mass @ np.ones(len(nodes)))
        augmented = sparse.bmat([[matrix, mean[:, None]], [mean[None], None]], format="csc")
        pressure = solve_linear(
            augmented,
            np.r_[load, mean_pressure * mesh.areas.sum()],
            solver=solver,
            refinement_precision=refinement_precision,
        )[:-1]
        free = np.arange(len(nodes))
    residual = float(
        np.linalg.norm((matrix @ pressure - load)[free])
        / max(
            np.linalg.norm(load[free]),
            np.linalg.norm((abs(matrix) @ abs(pressure))[free]),
            np.finfo(float).tiny,
        )
    )
    return ConformingQuadrilateralSolution(mesh, degree, pressure, permeability, residual)
