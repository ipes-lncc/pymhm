"""Declare tetrahedral mixed vector equations without physical solver dispatch.

The application uses existing tetrahedral tabulation, scalar trace integration,
element scatter, rigid-mode evaluation and physical inverse-bound kernels. Global
forms, boundary moments and gauges remain explicit in the calling notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from pymhm import LocalEquations
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.fem.vector.elasticity_3d import vector_boundary_data_3d
from pymhm.fem.vector.flow_3d import tetra_flow_operators
from pymhm.fem.vector.pressure_3d import tetra_elasticity_pressure_operators
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.solutions import Flow3DSolution, GaLS3DSolution


@dataclass(frozen=True)
class TetrahedralVectorSpace:
    """Declare continuous vector/scalar degrees, dyadic refinement and trace modes."""

    mesh: TetraMesh
    skeleton: TriangularSkeleton
    vector_degree: int = 2
    pressure_degree: int = 1
    refinement: int = 2
    quadrature_order: int = 5


def vector_boundary_moments(
    skeleton: TriangularSkeleton, datum: Any, traction: dict[int, Any], order: int
) -> tuple[Any, dict[int, float]]:
    """Delegate Cartesian boundary moments to the existing scalar-trace projection owner.

    The datum is the weak vector Dirichlet field. Supplied physical outward
    tractions project to negative fixed multiplier coefficients. This operation
    selects no physical local operator or solution method.
    """
    return vector_boundary_data_3d(skeleton, datum, traction, order)


def _trace_blocks(
    cell: int, fine: TetraMesh, size: int, nv: int, data: TetrahedralVectorSpace
) -> tuple[Any, Any]:
    """Lift signed scalar trace integration to interleaved Cartesian vector coordinates."""
    b = np.zeros((size, 3 * len(data.skeleton.cell_dofs(cell))))
    b[: 3 * nv] = np.kron(
        tetra_trace_coupling(data.mesh, cell, fine, data.skeleton, data.vector_degree), np.eye(3)
    )
    indices = (3 * data.skeleton.cell_dofs(cell)[:, None] + np.arange(3)).ravel()
    return b, indices


def tetra_velocity_pressure_equations(
    cell: int,
    *,
    data: TetrahedralVectorSpace,
    source: Any,
    viscosity: float = 1.0,
    drag: Any = 0.0,
    advection: Any = (0.0, 0.0, 0.0),
    advection_divergence: Any = None,
    advection_bound: float | None = None,
    formulation: str = "taylor-hood",
    stabilization: str = "tensor-2025",
    gamma_min: float | None = None,
) -> LocalEquations:
    """Declare grad-grad flow, pressure/divergence and an explicit residual choice.

    Taylor--Hood has Pk/P(k-1); USFEM and Oseen have equal-order Pk/Pk.
    The common volume owner integrates the stated skew transport, resistance,
    inverse bound and residual test/trial operators. B couples negative physical
    pseudo-traction and C=-B.T. Translations are a kernel only for exactly zero
    drag and constant zero advection; otherwise they remain declared coarse modes.
    Pressure moments and all Cartesian field coordinates are literal physical data.
    """
    expected = data.vector_degree - 1 if formulation == "taylor-hood" else data.vector_degree
    if data.pressure_degree != expected:
        raise ValueError("pressure degree must match the declared flow formulation")
    fine = data.mesh.submesh(cell, data.refinement)
    forms = tetra_flow_operators(
        fine,
        viscosity=viscosity,
        drag=drag,
        advection=advection,
        advection_divergence=advection_divergence,
        advection_bound=advection_bound,
        source=source,
        degree=data.vector_degree,
        formulation=formulation,
        stabilization=stabilization,
        gamma_min=gamma_min,
        order=data.quadrature_order,
    )
    nv, size = len(forms.velocity_nodes), len(forms.load)
    b, indices = _trace_blocks(cell, fine, size, nv, data)
    translation = np.zeros((size, 3))
    translation[: 3 * nv] = np.tile(np.eye(3), (nv, 1))
    moments = np.zeros_like(translation)
    moments[: 3 * nv] = (forms.velocity_moments[:, None, None] * np.eye(3)).reshape(3 * nv, 3)
    pure = np.all(forms.zero_columns) and not callable(advection) and not np.any(advection)
    pressure_weights = np.r_[np.zeros(3 * nv), forms.pressure_moments]
    selector = sparse.eye(size, format="csr")
    return LocalEquations(
        forms.matrix,
        forms.load,
        b,
        -b.T,
        indices,
        kernel=translation if pure else None,
        coarse_basis=None if pure else translation,
        moments=moments,
        metadata=(fine, nv, pressure_weights, forms, moments),
        field_data=(
            nodal_field(
                "velocity",
                fine,
                data.vector_degree,
                components=3,
                reconstruction=selector[: 3 * nv],
            ),
            nodal_field("pressure", fine, data.pressure_degree, reconstruction=selector[3 * nv :]),
        ),
    )


def tetra_displacement_pressure_equations(
    cell: int,
    *,
    data: TetrahedralVectorSpace,
    source: Any,
    lame_lambda: Any = np.inf,
    lame_mu: Any = 1.0,
    lame_mu_gradient: Any = None,
    shear_bounds: tuple[float, float, float] | None = None,
    formulation: str = "gals",
    stabilization_alpha: float | None = None,
) -> LocalEquations:
    """Declare isotropic Herrmann elasticity and GaLS or Taylor--Hood volume forms.

    The residual is div(2*mu*epsilon(u))-grad(p), including grad(mu).
    GaLS subtracts its residual product and adds its source functional. The
    sufficient inverse/coefficient bound is computed on the physical tetrahedra;
    caller-supplied variable-shear bounds remain explicit. Six rigid modes and
    their volume moments use the global volume centroid. Lambda may be infinite.
    """
    expected = data.vector_degree if formulation == "gals" else data.vector_degree - 1
    if data.pressure_degree != expected:
        raise ValueError("pressure degree must match the declared elasticity formulation")
    fine = data.mesh.submesh(cell, data.refinement)
    vertices = data.mesh.points[data.mesh.cells[cell]]
    diameter = float(np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1).max())
    center = (
        data.mesh.volumes @ data.mesh.points[data.mesh.cells].mean(axis=1) / data.mesh.volumes.sum()
    )
    forms = tetra_elasticity_pressure_operators(
        fine,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        lame_mu_gradient=lame_mu_gradient,
        shear_bounds=shear_bounds,
        source=source,
        degree=data.vector_degree,
        formulation=formulation,
        stabilization_alpha=stabilization_alpha,
        macro_diameter=diameter,
        rigid_center=center,
        order=data.quadrature_order,
    )
    nv, size = len(forms.displacement_nodes), len(forms.load)
    b, indices = _trace_blocks(cell, fine, size, nv, data)
    selector = sparse.eye(size, format="csr")
    return LocalEquations(
        forms.matrix,
        forms.load,
        b,
        -b.T,
        indices,
        kernel=forms.kernel,
        moments=forms.rigid_moments,
        metadata=(
            fine,
            nv,
            forms.pressure_moments,
            forms.stabilization_alpha,
            forms.rigid_moments,
            forms.compliance_moments,
            forms.compliance_scale,
        ),
        field_data=(
            nodal_field(
                "displacement",
                fine,
                data.vector_degree,
                components=3,
                reconstruction=selector[: 3 * nv],
            ),
            nodal_field("pressure", fine, data.pressure_degree, reconstruction=selector[3 * nv :]),
        ),
    )


def tetra_velocity_fields(
    system: MultiscaleSystem,
    solution: HybridSolution,
    data: TetrahedralVectorSpace,
    viscosity: float = 1.0,
    advection: Any = (0.0, 0.0, 0.0),
    formulation: str = "taylor-hood",
) -> Flow3DSolution:
    """Interpret the declared coefficients in the unchanged grad-grad flow field record."""
    return Flow3DSolution(
        data.skeleton,
        tuple(record[0] for record in system.local_metadata),
        tuple(
            field[: 3 * record[1]].reshape(-1, 3)
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        tuple(
            field[3 * record[1] :]
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        solution,
        data.vector_degree,
        data.pressure_degree,
        viscosity,
        advection,
        formulation,
    )


def tetra_displacement_fields(
    system: MultiscaleSystem,
    solution: HybridSolution,
    data: TetrahedralVectorSpace,
    lame_lambda: float = np.inf,
    lame_mu: Any = 1.0,
    formulation: str = "gals",
) -> GaLS3DSolution:
    """Interpret displacement, Herrmann pressure and Cauchy stress in the executed basis."""
    return GaLS3DSolution(
        data.skeleton,
        tuple(record[0] for record in system.local_metadata),
        tuple(
            field[: 3 * record[1]].reshape(-1, 3)
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        tuple(
            field[3 * record[1] :]
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        solution,
        data.vector_degree,
        data.pressure_degree,
        lame_lambda,
        lame_mu,
        formulation,
        tuple(record[3] for record in system.local_metadata),
    )
