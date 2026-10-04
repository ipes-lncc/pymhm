"""Multiscale HHO from constrained local energy minimization on triangles.

The cell and face unknowns are integral moments, an invertible change of
coordinates from the polynomial coefficients in Chaumont-Frelet et al. (2022).
Local reconstructions minimize diffusion energy subject to these moments.
The cell moments are statically condensed, leaving the represented face system.
The exact diffusion form is symmetric; its assembled operator is retained
without a further projection onto its symmetric part.
Both the projected-source formulation (4.6) and reconstructed-source variant
(5.1) are available. The face-only case m=-1 requires the latter variant.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.core.moments import energy_reconstruction
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.reference import legendre_values
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import scalar_operators, tabulate, trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.linalg.moments import solve_moment_system
from pymhm.materials.evaluation import scalar_values, tensor_values
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class MsHHOLocal:
    """One constrained Galerkin lift and its represented local moment operator.

    ``energy`` is R.T A R, retaining the represented assembly's small
    antisymmetry. For the exactly symmetric diffusion form it is the energy
    Hessian. Projecting this matrix to its symmetric part would change the
    original finite equations.
    """

    mesh: Any
    reconstruction: FloatArray
    moments: FloatArray
    energy: FloatArray
    load: FloatArray
    cell_count: int
    integral: FloatArray


@dataclass(frozen=True)
class MsHHOSolution:
    """Broken pressure fields, common face moments and represented condensed operator."""

    skeleton: Any
    local: tuple[MsHHOLocal, ...]
    pressure: tuple[FloatArray, ...]
    face_moments: FloatArray
    cell_moments: tuple[FloatArray, ...]
    matrix: Any
    residual: float
    degree: int
    permeability: Any
    source_variant: str
    local_refinement_precision: str = "double"

    def l2_error(self, exact: Any, order: int = 8) -> float:
        """Integrate broken pressure error with independent quadrature."""
        bary, weights = triangle_quadrature(positive_int(order, "quadrature order"))
        total = 0.0
        for local, values in zip(self.local, self.pressure, strict=True):
            mesh = local.mesh
            dofs, _, basis, _, _ = tabulate(mesh, self.degree, bary)
            points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
            exact_values = scalar_values(exact, points.reshape(-1, 2)).reshape(points.shape[:2])
            error = values[dofs] @ basis.T - exact_values
            total += float(mesh.areas @ (error**2 @ weights))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact: Any, order: int = 8) -> float:
        """Integrate the physical raw flux -A grad(u); no H(div) claim is made."""
        from pymhm.materials.evaluation import vector_values

        bary, weights = triangle_quadrature(positive_int(order, "quadrature order"))
        total = 0.0
        for local, values in zip(self.local, self.pressure, strict=True):
            mesh = local.mesh
            dofs, _, _, gradient, _ = tabulate(mesh, self.degree, bary)
            points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
            flat = points.reshape(-1, 2)
            tensor = tensor_values(self.permeability, flat).reshape(*points.shape[:2], 2, 2)
            flux = -np.einsum("tqab,tqib,ti->tqa", tensor, gradient, values[dofs])
            error = flux - vector_values(exact, flat).reshape(flux.shape)
            total += float(mesh.areas @ (np.sum(error**2, axis=-1) @ weights))
        return float(np.sqrt(total))


def _local_reconstruction(
    mesh: TriangleMesh,
    cell: int,
    skeleton: SkeletonSpace,
    permeability: Any,
    source: Any,
    cell_degree: int,
    degree: int,
    refinement: int,
    order: int,
    variant: str,
    refinement_precision: Literal["double", "extended"],
) -> MsHHOLocal:
    """Solve the constrained energy minimum defining all local moment basis functions."""
    fine = mesh.submesh(cell, refinement)
    matrix, _, force = scalar_operators(
        fine, degree, diffusion=permeability, source=source, order=order
    )
    bary, weights = triangle_quadrature(max(order, degree + 2, cell_degree + 2))
    dofs, nodes, basis, _, _ = tabulate(fine, degree, bary)
    points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
    vertices = mesh.points[mesh.cells[cell]]
    if cell_degree < 0:
        cell_basis = np.empty((*points.shape[:2], 0))
    else:
        coordinates = 2 * (points - vertices.min(axis=0)) / np.ptp(vertices, axis=0) - 1
        px = legendre_values(coordinates[..., 0], cell_degree)
        py = legendre_values(coordinates[..., 1], cell_degree)
        cell_basis = np.stack(
            [
                px[..., i] * py[..., j]
                for j in range(cell_degree + 1)
                for i in range(cell_degree + 1 - j)
            ],
            axis=-1,
        )
    count = cell_basis.shape[-1]
    volume = np.zeros((len(nodes), count))
    local = np.einsum("t,q,qi,tqj->tij", fine.areas, weights, basis, cell_basis)
    if count:
        np.add.at(volume, dofs.ravel(), local.reshape(-1, count))
    face = trace_coupling(mesh, cell, fine, skeleton, degree)
    offset = 0
    for side, face_id in enumerate(mesh.cell_faces[cell]):
        width = skeleton.faces[face_id].size
        face[:, offset : offset + width] *= mesh.signs[cell, side]
        offset += width
    moments = np.column_stack((volume, face))
    reconstruction, energy = energy_reconstruction(matrix, moments, refinement_precision)
    if variant == "reconstructed":
        load = reconstruction.T @ force
    else:
        gram = np.einsum("t,q,tqi,tqj->ij", fine.areas, weights, cell_basis, cell_basis)
        sampled = scalar_values(source, points.reshape(-1, 2)).reshape(points.shape[:2])
        projected = np.linalg.solve(
            gram, np.einsum("t,q,tqi,tq->i", fine.areas, weights, cell_basis, sampled)
        )
        load = np.r_[projected, np.zeros(face.shape[1])]
    nodal_integral = np.bincount(
        dofs.ravel(),
        weights=np.einsum("t,q,qi->ti", fine.areas, weights, basis).ravel(),
        minlength=len(nodes),
    )
    return MsHHOLocal(
        fine, reconstruction, moments, energy, load, count, nodal_integral @ reconstruction
    )


def _condense_moments(
    cells: tuple[MsHHOLocal, ...],
    skeleton: Any,
    fixed: dict[int, float],
    boundary_load: FloatArray,
    constant_moments: FloatArray,
    mean_target: float | None,
    solver: str,
    refinement_precision: Literal["double", "extended"] = "double",
) -> tuple[Any, FloatArray, tuple[FloatArray, ...], tuple[FloatArray, ...], float]:
    """Condense cell moments, impose natural data and recover the broken fields."""
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    dtype = np.longdouble if refinement_precision == "extended" else float
    rhs = np.zeros(skeleton.size, dtype=dtype)
    lifts, sources = [], []
    for cell, data in enumerate(cells):
        n = data.cell_count
        if n:
            lift = solve_moment_system(
                data.energy[:n, :n], data.energy[:n, n:], refinement_precision=refinement_precision
            )
            particular = solve_moment_system(
                data.energy[:n, :n], data.load[:n], refinement_precision=refinement_precision
            )
        else:
            lift, particular = np.empty((0, data.energy.shape[1])), np.empty(0)
        local_matrix = data.energy[n:, n:] - data.energy[n:, :n] @ lift
        local_rhs = data.load[n:] - data.energy[n:, :n] @ particular
        ids = skeleton.cell_dofs(cell)
        rows.extend(np.repeat(ids, len(ids)))
        columns.extend(np.tile(ids, len(ids)))
        entries.extend(local_matrix.ravel())
        np.add.at(rhs, ids, local_rhs)
        lifts.append(lift)
        sources.append(particular)
    matrix = sparse.coo_matrix((entries, (rows, columns)), shape=(skeleton.size,) * 2).tocsc()
    rhs += boundary_load
    values = np.zeros(skeleton.size, dtype=dtype)
    for index, value in fixed.items():
        values[index] = value
    free = np.setdiff1d(np.arange(skeleton.size), list(fixed))
    if mean_target is not None:
        compatibility = float(constant_moments @ rhs)
        scale = float(np.abs(constant_moments) @ np.abs(rhs))
        if abs(compatibility) > 1e-11 * max(scale, np.finfo(float).tiny):
            raise ValueError("incompatible pure Neumann source and outward flux")
        integral = np.zeros(skeleton.size, dtype=dtype)
        offset = dtype(0.0)
        for cell, data in enumerate(cells):
            n = data.cell_count
            np.add.at(
                integral,
                skeleton.cell_dofs(cell),
                data.integral[n:] - data.integral[:n] @ lifts[cell],
            )
            offset += (data.integral[:n] @ sources[cell]).item()
        augmented = sparse.bmat(
            [[matrix, integral[:, None]], [integral[None, :], None]], format="csc"
        )
        values = solve_moment_system(
            augmented,
            np.r_[rhs, mean_target - offset],
            solver=solver,
            refinement_precision=refinement_precision,
        )[:-1]
    elif len(free):
        values[free] = solve_moment_system(
            matrix[free][:, free],
            (rhs - matrix @ values)[free],
            solver=solver,
            refinement_precision=refinement_precision,
        )
    defect = (matrix @ values - rhs)[free]
    residual = float(
        np.linalg.norm(defect)
        / max(
            np.linalg.norm(rhs[free]),
            np.linalg.norm((abs(matrix) @ abs(values))[free]),
            np.finfo(float).tiny,
        )
    )
    coarse, fields = [], []
    for cell, data in enumerate(cells):
        trace = values[skeleton.cell_dofs(cell)]
        interior = sources[cell] - lifts[cell] @ trace
        coarse.append(interior)
        fields.append(data.reconstruction @ np.r_[interior, trace])
    return matrix, values, tuple(coarse), tuple(fields), residual


def solve_mshho(
    mesh: TriangleMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    skeleton: SkeletonSpace | None = None,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    cell_degree: int = 0,
    degree: int = 2,
    local_refinement: int = 4,
    source_variant: Literal["projected", "reconstructed"] = "projected",
    quadrature_order: int = 6,
    solver: str = "scipy",
    local_refinement_precision: Literal["double", "extended"] = "double",
) -> MsHHOSolution:
    """Solve multiscale HHO with local continuous Pk energy reconstructions.

    ``cell_degree=m`` and ``skeleton`` specify the coarse moment spaces;
    ``degree`` controls the local numerical approximation only. For m=-1,
    select ``source_variant='reconstructed'`` as required by Remark 5.4.
    Exterior Dirichlet data are imposed through face moments. ``neumann`` maps
    selected exterior faces to physical outward flux -A grad(u) dot n; remaining
    faces have Dirichlet data. All-Neumann problems require compatible total
    source/flux and fix the volume mean through ``mean_pressure``.
    This finite-dimensional realization uses the same local variational
    spaces as primal MHM, allowing exact discrete equivalence checks for
    polynomial/projected sources. Raw gradients need not be H(div) conforming.
    ``local_refinement_precision='extended'`` retains reconstruction, represented
    Galerkin operators and moment-coordinate digits through defect corrections.
    Factors remain binary64 and the original residual criterion stays 1e-10;
    platforms without a wider long-double type raise ``SolverUnavailableError``.
    """
    cell_degree = positive_int(cell_degree, "cell_degree", -1)
    degree = positive_int(degree, "degree")
    refinement = positive_int(local_refinement, "local_refinement")
    order = positive_int(quadrature_order, "quadrature order")
    if source_variant not in {"projected", "reconstructed"}:
        raise ValueError("source_variant must be projected or reconstructed")
    if local_refinement_precision not in {"double", "extended"}:
        raise ValueError("local_refinement_precision must be double or extended")
    if cell_degree == -1 and source_variant != "reconstructed":
        raise ValueError("m=-1 requires the reconstructed-source variant")
    skeleton = (
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(0) for _ in mesh.faces))
        if skeleton is None
        else skeleton
    )
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("MsHHO requires a scalar skeleton on the supplied mesh")
    cells = tuple(
        _local_reconstruction(
            mesh,
            cell,
            skeleton,
            permeability,
            source,
            cell_degree,
            degree,
            refinement,
            order,
            source_variant,
            local_refinement_precision,
        )
        for cell in range(len(mesh.cells))
    )
    prescribed = {} if neumann is None else neumann
    if not set(prescribed).issubset(set(mesh.boundary_faces)):
        raise ValueError("Neumann data require exterior face indices")
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    pure_neumann = set(prescribed) == set(mesh.boundary_faces)
    if not pure_neumann and mean_pressure != 0:
        raise ValueError("mean_pressure is a gauge only for pure Neumann boundaries")
    fixed: dict[int, float] = {}
    natural = np.zeros(skeleton.size)
    constants = np.zeros(skeleton.size)
    for face_id in range(len(mesh.faces)):
        face = skeleton.faces[face_id]
        t, w = face.quadrature(max(order, max(face.degrees) + 2))
        start, end = mesh.points[mesh.faces[face_id]]
        points = start + t[:, None] * (end - start)
        ids = skeleton.dofs(face_id)
        basis = face.evaluate(t)
        measure = mesh.lengths[face_id] * w
        constants[ids] = basis.T @ measure
        if face_id in prescribed:
            mass = np.einsum("q,qi,qj->ij", measure, basis, basis)
            natural[ids] = -np.linalg.solve(
                mass, basis.T @ (measure * scalar_values(prescribed[face_id], points))
            )
        elif face_id in mesh.boundary_faces:
            moments = basis.T @ (measure * scalar_values(dirichlet, points))
            fixed.update(zip(ids, moments, strict=True))
    matrix, values, coarse, fields, residual = _condense_moments(
        cells,
        skeleton,
        fixed,
        natural,
        constants,
        mean_pressure * float(mesh.areas.sum()) if pure_neumann else None,
        solver,
        local_refinement_precision,
    )
    return MsHHOSolution(
        skeleton,
        cells,
        tuple(fields),
        values,
        tuple(coarse),
        matrix,
        residual,
        degree,
        permeability,
        source_variant,
        local_refinement_precision,
    )
