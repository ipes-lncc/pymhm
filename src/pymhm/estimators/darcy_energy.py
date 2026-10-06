"""Energy-normalized MHM flux estimators for SPD materials and mixed boundaries.

The decomposition is that of Barrenechea et al. (2026), with dual norms taken
in the physical energy inner product. Material-dependent Poincare weights
therefore contain the certified lower ellipticity bound. Boundary traces must
be representable; no unmeasured boundary residual is silently omitted.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm._legacy.models.darcy.primal import DarcySolution
from pymhm.core.validation import FloatArray, positive_int
from pymhm.estimators.darcy import (
    ConformingPotential,
    DarcyEstimator,
    _average_potential,
    _restrict_potential,
)
from pymhm.execution.cpu import map_local
from pymhm.fem.conditions import validate_estimator_spaces
from pymhm.fem.hdiv.rt import rt_evaluate
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.triangle import element_tabulate, nodal_space
from pymhm.linalg.linear import solve_linear
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import scalar_values, tensor_values, vector_values
from pymhm.recovery.moments import MomentFluxSolution, reconstruct_darcy_moments


def recover_dirichlet_potential(
    solution: DarcySolution, *, dirichlet: Any = 0.0, neumann: dict[int, Any] | None = None
) -> ConformingPotential:
    """Oswald-average the broken field and impose only its Dirichlet boundary nodes.

    The boundary function must belong to the continuous local Pk trace space.
    Representability is checked on every fine Dirichlet edge. Nonpolynomial
    boundary approximation requires an additional lifting estimator and is
    rejected here. Neumann nodes remain unconstrained except at Dirichlet corners.
    """
    mesh, dofs, local_dofs, values = _average_potential(solution)
    _, nodes = nodal_space(mesh, solution.degree)
    coarse = solution.skeleton.mesh
    natural = {} if neumann is None else neumann
    if any(face not in coarse.boundary_faces for face in natural):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    t, _ = leggauss(solution.degree + 3)
    t = (t + 1) / 2
    prescribed = set(int(f) for f in coarse.boundary_faces) - set(natural)
    scale = max(float(np.ptp(mesh.points, axis=0).max()), float(np.abs(mesh.points).max()))
    tolerance = 256 * np.finfo(float).eps * scale
    fine_edges = mesh.points[mesh.faces[mesh.boundary_faces]]
    for face in prescribed:
        start, end = coarse.points[coarse.faces[face]]
        tangent = end - start
        parameter = (nodes - start) @ tangent / (tangent @ tangent)
        on_face = np.linalg.norm(nodes - start - parameter[:, None] * tangent, axis=1) <= tolerance
        on_face &= (parameter >= -tolerance) & (parameter <= 1 + tolerance)
        values[on_face] = scalar_values(dirichlet, nodes[on_face])
        midpoints = fine_edges.mean(axis=1)
        edge_t = (midpoints - start) @ tangent / (tangent @ tangent)
        selected = (
            np.linalg.norm(midpoints - start - edge_t[:, None] * tangent, axis=1) <= tolerance
        )
        selected &= (edge_t >= 0) & (edge_t <= 1)
        for a, b in fine_edges[selected]:
            interpolation_t = np.linspace(0, 1, solution.degree + 1)
            interpolation = a + interpolation_t[:, None] * (b - a)
            coefficients = np.polynomial.polynomial.polyfit(
                interpolation_t, scalar_values(dirichlet, interpolation), solution.degree
            )
            actual = scalar_values(dirichlet, a + t[:, None] * (b - a))
            approximation = np.polynomial.polynomial.polyval(t, coefficients)
            if np.max(abs(actual - approximation)) > 1e-10 * max(
                float(np.max(abs(actual))), np.finfo(float).tiny
            ):
                raise ValueError(
                    "Dirichlet data must be represented exactly by the fine trace space"
                )
    return _restrict_potential(solution, mesh, dofs, local_dofs, values)


def _project_cells(
    mesh: Any, degree: int, data: FloatArray, bary: FloatArray, weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Continuous macro-local L2 projection using cell-dependent positive quadrature."""
    if degree == 0:
        dofs = np.zeros((len(mesh.cells), 1), dtype=int)
        basis = np.ones((*bary.shape[:2], 1))
        size = 1
    else:
        dofs, nodes, basis, _, _ = element_tabulate(mesh, degree, bary)
        size = len(nodes)
    width = basis.shape[-1]
    blocks = np.einsum("t,tq,tqi,tqj->tij", mesh.areas, weights, basis, basis)
    mass = sparse.coo_matrix(
        (
            blocks.ravel(),
            (np.repeat(dofs, width, axis=1).ravel(), np.tile(dofs, (1, width)).ravel()),
        ),
        shape=(size, size),
    ).tocsc()
    moments = np.einsum("t,tq,tqi,tq->ti", mesh.areas, weights, basis, data)
    load = np.bincount(dofs.ravel(), weights=moments.ravel(), minlength=size)
    coefficients = solve_linear(mass, load)
    return np.einsum("ti,tqi->tq", coefficients[dofs], basis), load


def _ellipticity(solution: DarcySolution, bound: Any) -> FloatArray:
    """Obtain certified bounds from literal materials, or require them from the caller."""
    material = solution.permeability
    if bound is None:
        lower: Any
        if isinstance(material, CartesianCellField):
            data = material.values
            lower = np.min(np.linalg.eigvalsh(data)) if data.ndim == 4 else np.min(data)
        elif callable(material):
            raise ValueError("variable callbacks require a certified ellipticity_lower_bound")
        else:
            data = np.asarray(material)
            lower = float(data) if data.ndim == 0 else np.min(np.linalg.eigvalsh(data))
        bound = float(lower)
    raw = np.asarray(bound)
    count = len(solution.local_meshes)
    if np.iscomplexobj(raw) or raw.shape not in ((), (count,)):
        raise ValueError("ellipticity bound must be a positive scalar or one value per macrocell")
    values = np.broadcast_to(raw, (count,)).astype(float)
    if not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("ellipticity bounds must be finite and strictly positive")
    return values


@dataclass(frozen=True)
class WeightedDarcyEstimator(DarcyEstimator):
    """Four material-weighted terms and their explicit reliability assumptions.

    The reported total bounds the broken energy error under exact integration,
    exact represented boundary data, continuous-test flux equilibrium and the
    supplied ellipticity lower bounds. Evaluated quadrature is not interval arithmetic.
    """

    ellipticity_lower_bounds: FloatArray

    @property
    def convention(self) -> str:
        """Identify the energy-normalized material convention explicitly."""
        return "energy"

    def energy_error(self, exact_gradient: Any, order: int = 10) -> float:
        """Integrate the broken physical energy error, resolving declared material cuts."""
        total = 0.0
        for fine, pressure in zip(self.solution.local_meshes, self.solution.pressure, strict=True):
            bary, weights, material = material_triangle_quadrature(
                fine, self.solution.permeability, order
            )
            dofs, _, _, gradient, _ = element_tabulate(fine, self.solution.degree, bary)
            points = np.einsum("tqi,tia->tqa", bary, fine.points[fine.cells])
            tensors = tensor_values(material, points.reshape(-1, 2)).reshape(
                *points.shape[:2], 2, 2
            )
            difference = np.einsum("ti,tqia->tqa", pressure[dofs], gradient) - vector_values(
                exact_gradient, points.reshape(-1, 2)
            ).reshape(points.shape)
            total += np.einsum(
                "t,tq,tqa,tqab,tqb->", fine.areas, weights, difference, tensors, difference
            )
        return float(np.sqrt(total))


@dataclass(frozen=True)
class PublishedDarcyIndicator(WeightedDarcyEstimator):
    """Literal equations (5.3)--(5.7), without a general-material energy-bound claim.

    The flux residual uses the ordinary L2 norm and the divergence/oscillation
    terms use diameter/pi, without an ellipticity factor. These terms coincide
    with the energy-normalized estimator for identity diffusion. For general
    SPD material they are the published numerical indicators; the physical
    ``energy_error`` remains available but no coefficient-independent
    reliability guarantee is attached to ``total``.
    ``ellipticity_lower_bounds`` is empty because the printed terms use no
    ellipticity certificate.
    """

    @property
    def convention(self) -> str:
        """Identify the literal printed material convention."""
        return "published"

    @property
    def local_squared(self) -> FloatArray:
        """Return equation (5.6)'s additive squared numerical indicators."""
        return super().local_squared


@dataclass(frozen=True)
class _IndicatorFactory:
    """Compute the independent five diagnostic terms of one macrocell."""

    solution: DarcySolution
    potential: ConformingPotential
    reconstruction: MomentFluxSolution
    degree: int
    order: int
    alpha: FloatArray
    convention: str

    def __call__(self, cell: int) -> FloatArray:
        """Retain exact operator and reduction ordering inside each local calculation."""
        solution, degree, order = self.solution, self.degree, self.order
        alpha, convention = self.alpha, self.convention
        coarse = solution.skeleton.mesh
        fine, pressure = solution.local_meshes[cell], solution.pressure[cell]
        recovered, flux = self.potential.local_values[cell], self.reconstruction.flux[cell]
        terms = np.zeros(5)
        bary, weights, material = material_triangle_quadrature(fine, solution.permeability, order)
        points = np.einsum("tqi,tia->tqa", bary, fine.points[fine.cells])
        tensors = tensor_values(material, points.reshape(-1, 2)).reshape(*points.shape[:2], 2, 2)
        if convention == "energy" and np.min(np.linalg.eigvalsh(tensors)) < alpha[cell] * (
            1 - 1e-12
        ):
            raise ValueError("ellipticity_lower_bound exceeds a sampled material eigenvalue")
        dofs, _, _, gradient, _ = element_tabulate(fine, solution.degree, bary)
        grad = np.einsum("ti,tqia->tqa", pressure[dofs], gradient)
        delta = np.einsum("ti,tqia->tqa", (pressure - recovered)[dofs], gradient)
        q, divergence = rt_evaluate(fine, flux, degree, bary)
        source = scalar_values(solution.source, points.reshape(-1, 2)).reshape(divergence.shape)
        projected_source, source_moments = _project_cells(fine, degree, source, bary, weights)
        projected_div, div_moments = _project_cells(fine, degree, divergence, bary, weights)
        defect = q + np.einsum("tqab,tqb->tqa", tensors, grad)
        terms[0] = np.sqrt(
            np.einsum(
                "t,tq,tqa,tqab,tqb->",
                fine.areas,
                weights,
                defect,
                np.linalg.inv(tensors)
                if convention == "energy"
                else np.broadcast_to(np.eye(2), tensors.shape),
                defect,
            )
        )
        terms[1] = np.sqrt(
            np.einsum("t,tq,tqa,tqab,tqb->", fine.areas, weights, delta, tensors, delta)
        )
        poincare = np.max(coarse.lengths[coarse.cell_faces[cell]]) / (np.pi * np.sqrt(alpha[cell]))
        terms[2] = poincare * np.sqrt(
            np.einsum("t,tq,tq->", fine.areas, weights, (projected_div - divergence) ** 2)
        )
        terms[3] = poincare * np.sqrt(
            np.einsum("t,tq,tq->", fine.areas, weights, (source - projected_source) ** 2)
        )
        terms[4] = np.linalg.norm(source_moments - div_moments)
        scale = max(
            np.linalg.norm(source_moments),
            np.linalg.norm(div_moments),
            np.linalg.norm(q) * fine.areas.sum(),
            np.finfo(float).tiny,
        )
        if terms[4] > 1e-8 * scale:
            raise ValueError(
                "continuous-test equilibrium failed; resolve material and source integration"
            )
        return terms


def estimate_weighted_darcy_error(
    solution: DarcySolution,
    *,
    degree: int = 1,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    ellipticity_lower_bound: Any = None,
    quadrature_order: int = 8,
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> WeightedDarcyEstimator:
    """Evaluate the energy-normalized decomposition for SPD diffusion.

    The decomposition follows [Barrenechea et al. (2026)](https://doi.org/10.1137/24M1673073).

    Macro cells must be convex triangles; fine meshes must be globally conforming. Require k>=ell+2
    and ell<=m<=k. The constant/Cartesian coefficient's minimum eigenvalue provides a certified
    bound automatically. Other callbacks require an explicit positive lower bound, checked at
    quadrature nodes in addition to the caller's certification between those nodes. Neumann data are
    outward physical fluxes and must be exactly represented by the skeletal trace.
    """
    return estimate_darcy_indicator(
        solution,
        convention="energy",
        degree=degree,
        dirichlet=dirichlet,
        neumann=neumann,
        ellipticity_lower_bound=ellipticity_lower_bound,
        quadrature_order=quadrature_order,
        backend=backend,
        workers=workers,
    )


def estimate_darcy_indicator(
    solution: DarcySolution,
    *,
    convention: Literal["published", "energy"] = "published",
    degree: int = 1,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    ellipticity_lower_bound: Any = None,
    quadrature_order: int = 8,
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> WeightedDarcyEstimator:
    """Select published indicators or the physical-energy normalization.

    The published indicators follow [Barrenechea et al. (2026)](https://doi.org/10.1137/24M1673073).

    ``published`` evaluates eta1=||K grad(p)+q||, eta2 in the K energy norm, and eta3/oscillation
    with diameter/pi, exactly as (5.3)--(5.7) are printed. ``energy`` uses K^-1 in eta1 and
    diameter/(pi sqrt(alpha_K)) in eta3 and oscillation. The latter requires certified ellipticity
    lower bounds for callable coefficients. Both conventions retain represented boundary data,
    conforming local partitions, space restrictions and continuous equilibrium checks. The published
    convention is a numerical indicator for general SPD material, not a claim of a
    coefficient-independent physical-energy bound. ``backend`` executes independent reconstruction
    and indicator calculations; spawn workers require picklable coefficient and source callbacks.
    """
    if convention not in {"published", "energy"}:
        raise ValueError("convention must be 'published' or 'energy'")
    potential = recover_dirichlet_potential(solution, dirichlet=dirichlet, neumann=neumann)
    if any(len(part) for part in solution.point_sources):
        raise ValueError("energy estimation requires an L2 source, not point wells")
    coarse, skeleton = solution.skeleton.mesh, solution.skeleton
    if any(len(cell) != 3 for cell in coarse.cells):
        raise ValueError("the diameter/pi reliability constant requires triangular macrocells")
    ell = max(max(face.degrees) for face in skeleton.faces)
    validate_estimator_spaces(solution.degree, ell, degree, 2)
    order = positive_int(quadrature_order, "quadrature_order", max(solution.degree + 2, degree + 2))
    alpha = (
        _ellipticity(solution, ellipticity_lower_bound)
        if convention == "energy"
        else np.ones(len(coarse.cells))
    )
    for face, field in ({} if neumann is None else neumann).items():
        space = skeleton.faces[face]
        t, _ = space.quadrature(max(order, max(space.degrees) + 3))
        a, b = coarse.points[coarse.faces[face]]
        prescribed = scalar_values(field, a + t[:, None] * (b - a))
        represented = space.evaluate(t) @ solution.hybrid.trace[skeleton.dofs(face)]
        scale = max(
            float(np.max(abs(prescribed))), float(np.max(abs(represented))), np.finfo(float).tiny
        )
        if np.max(abs(prescribed - represented)) > 1e-10 * scale:
            raise ValueError("Neumann data must be represented exactly by the reconstructed trace")
    reconstruction = reconstruct_darcy_moments(
        solution, degree=degree, quadrature_order=order, backend=backend, workers=workers
    )
    factory = _IndicatorFactory(
        solution, potential, reconstruction, degree, order, alpha, convention
    )
    terms = np.array(
        map_local(factory, range(len(coarse.cells)), backend=backend, workers=workers)
    ).T
    result_type = WeightedDarcyEstimator if convention == "energy" else PublishedDarcyIndicator
    return result_type(
        solution,
        potential,
        reconstruction,
        terms[0],
        terms[1],
        terms[2],
        terms[3],
        terms[4],
        order,
        alpha if convention == "energy" else np.empty(0),
    )
