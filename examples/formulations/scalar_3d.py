"""User-defined conservative scalar equations on tetrahedral or polyhedral macrocells.

Geometry supplies oriented trace pairings; the same four-block equations apply
to both macro meshes. Shared numerical kernels integrate the scalar volume form.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np

from pymhm import Equation, LocalEquations, MultiscaleProblem
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.core.validation import dyadic_refinement, positive_int
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space, tetrahedron_quadrature
from pymhm.fem.scalar.transport_3d import coefficient_derivatives_3d, tetra_transport_operators
from pymhm.fem.traces.normal import boundary_tangent_3d
from pymhm.fem.traces.polygon_3d import PolygonalSkeleton3D, polygonal_boundary_data
from pymhm.fem.traces.polygon_3d import polygonal_trace_coupling as polygonal_trace_coupling
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_boundary_data
from pymhm.fem.traces.triangle_3d import tetra_trace_coupling as tetra_trace_coupling
from pymhm.materials.evaluation import scalar_values_3d, vector_values_3d
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.scalar_3d import PolyhedralRADSolution
from pymhm.postprocessing.solutions import RAD3DSolution

polygonal_boundary_pairing = polygonal_boundary_data
tetra_boundary_pairing = tetra_boundary_data


@dataclass(frozen=True)
class Scalar3DDiscretization:
    """Declared volume space and geometry-specific trace integration callbacks.

    A pairing callback returns B, including the global normal orientation.
    Boundary pairing returns weak data and fixed outward half-advection fluxes.
    Neither callback selects a PDE, local solver or global elimination method.
    """

    mesh: Any
    skeleton: Any
    trace_pairing: Callable[..., Any]
    boundary_pairing: Callable[..., Any]
    degree: int = 4
    refinement: int = 2
    order: int = 6


def scalar_retained_3d(
    cell: int,
    *,
    data: Scalar3DDiscretization,
    coefficients: dict[str, Any],
    coarse_space: Literal["constants", "kernel"],
    fine: Any = None,
    pure: bool | None = None,
) -> tuple[bool, bool]:
    """Declare whether the physical constant is a kernel or a retained complement.

    Counts use the volume form's literal coefficient samples, never a numerical
    rank threshold. Constants are retained even for arbitrarily small transport.
    The selective choice keeps a true constant kernel only when reaction and
    divergence vanish and velocity is tangent to the entire local boundary.
    This geometric/sample plan does not assemble an operator on the coordinator.
    """
    if coarse_space not in ("constants", "kernel"):
        raise ValueError("coarse_space must be constants or kernel")
    if pure is None or coarse_space == "kernel":
        fine = data.mesh.submesh(cell, data.refinement) if fine is None else fine
        bary, _ = tetrahedron_quadrature(coefficients["order"])
        points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(-1, 3)
        reaction = scalar_values_3d(coefficients["reaction"], points)
        divergence = scalar_values_3d(coefficients["velocity_divergence"], points)
        if pure is None:
            pure = not np.any(vector_values_3d(coefficients["velocity"], points)) and not np.any(
                reaction + divergence / 2
            )
    if pure:
        return True, False
    if coarse_space == "constants":
        return False, True
    if np.any(reaction) or np.any(divergence):
        return False, False
    if isinstance(data.mesh, PolyhedralMesh):
        tangent = boundary_tangent_3d(
            PolygonalSkeleton3D(data.mesh),
            coefficients["velocity"],
            coefficients["order"],
            faces=data.mesh.cell_faces[cell],
        )
    else:
        tangent = boundary_tangent_3d(
            TriangularSkeleton(fine), coefficients["velocity"], coefficients["order"]
        )
    return tangent, False


def scalar_equations_3d(
    cell: int,
    *,
    data: Scalar3DDiscretization,
    coefficients: dict[str, Any],
    coarse_space: Literal["constants", "kernel"] = "constants",
) -> LocalEquations:
    """Declare A u+B lambda=f and -B.T u=0 with physical constant moments.

    A integrates diffusion, skew conservative transport, c+div(beta)/2 and
    optional full-residual SUPG. Derivatives of variable coefficients are explicit
    inputs. The multiplier is (-K grad(u)+beta*u/2).n, not total physical flux.
    Constant kernel/complement columns use the same declared physical moments.
    """
    fine = data.mesh.submesh(cell, data.refinement)
    a, load, moments, pure = tetra_transport_operators(fine, data.degree, **coefficients)
    b = data.trace_pairing(data.mesh, cell, fine, data.skeleton, data.degree)
    kernel, complement = scalar_retained_3d(
        cell, data=data, coefficients=coefficients, coarse_space=coarse_space, fine=fine, pure=pure
    )
    constant = np.ones((len(load), 1))
    return LocalEquations(
        a,
        load,
        b,
        -b.T,
        data.skeleton.cell_dofs(cell),
        kernel=constant if kernel else None,
        coarse_basis=constant if complement else None,
        moments=moments[:, None] if kernel or complement else np.empty((len(load), 0)),
        metadata=(fine, moments),
        field_data=(nodal_field("scalar", fine, data.degree),),
    )


@dataclass(frozen=True)
class Scalar3DDefinition:
    """Declared volume/interface equations and explicit physical boundary/gauge data."""

    data: Scalar3DDiscretization
    problem: MultiscaleProblem[int]
    coefficients: dict[str, Any]
    natural_faces: frozenset[int]
    mean_value: float


def define_scalar_3d(
    data: Scalar3DDiscretization,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    velocity_divergence: Any = None,
    diffusion_divergence: Any = None,
    stabilization: str = "galerkin",
    mean_value: float = 0.0,
    coarse_space: Literal["constants", "kernel"] = "constants",
) -> Scalar3DDefinition:
    """Declare conservative RAD with weak values and fixed outward half-advection fluxes.

    Natural data fix (-K grad(u)+beta*u/2).n. Every other exterior macroface
    receives weak Dirichlet moments. The face degree and local Pk/refinement are
    chosen by the caller; a polyhedron's internal triangulation adds no traces.
    ``mean_value`` is the physical domain mean, admissible only for the declared
    all-natural zero-reaction, divergence-free, exterior-tangent constant gauge.
    """
    dyadic_refinement(data.refinement, "local_refinement")
    if coarse_space not in ("constants", "kernel"):
        raise ValueError("coarse_space must be constants or kernel")
    if not np.isfinite(mean_value):
        raise ValueError("mean_value must be finite")
    if data.skeleton.mesh is not data.mesh:
        raise ValueError("skeleton must belong to the supplied macro mesh")
    if isinstance(data.skeleton, TriangularSkeleton) and np.any(
        data.skeleton.subdivisions > data.refinement
    ):
        raise ValueError("local refinement must resolve every skeleton subdivision")
    tetra_nodal_space(
        data.mesh.submesh(0) if isinstance(data.mesh, PolyhedralMesh) else data.mesh, data.degree
    )
    order = max(positive_int(data.order, "quadrature_order"), data.degree + 2)
    div_beta, div_tensor = coefficient_derivatives_3d(
        data.mesh.points,
        diffusion,
        velocity,
        velocity_divergence,
        diffusion_divergence,
        stabilization,
    )
    natural = {} if neumann is None else dict(neumann)
    boundary, fixed = data.boundary_pairing(data.skeleton, dirichlet, natural, order)
    coefficients = dict(
        diffusion=diffusion,
        velocity=velocity,
        reaction=reaction,
        source=source,
        velocity_divergence=div_beta,
        diffusion_divergence=div_tensor,
        stabilization=stabilization,
        order=order,
    )
    counts = (
        (1,) * len(data.mesh.cells)
        if coarse_space == "constants"
        else tuple(
            sum(
                scalar_retained_3d(
                    cell, data=data, coefficients=coefficients, coarse_space=coarse_space
                )
            )
            for cell in range(len(data.mesh.cells))
        )
    )
    problem = MultiscaleProblem(
        Equation(0, -np.r_[boundary, np.zeros(sum(counts))]),
        partial(
            scalar_equations_3d, data=data, coefficients=coefficients, coarse_space=coarse_space
        ),
        range(len(data.mesh.cells)),
        data.skeleton.size,
        counts,
        fixed=fixed,
    )
    return Scalar3DDefinition(data, problem, coefficients, frozenset(natural), float(mean_value))


def scalar_problem_3d(data: Scalar3DDiscretization, **coefficients: Any) -> MultiscaleProblem[int]:
    """Return the declared equations; compatible full-Dirichlet defaults remain available.

    Additional natural data, kernel selection and mean specification follow
    :func:`define_scalar_3d`. Apply :func:`scalar_constraints_3d` after assembly
    when the physical problem has a constant gauge.
    """
    return define_scalar_3d(data, **coefficients).problem


def scalar_constraints_3d(
    definition: Scalar3DDefinition, system: MultiscaleSystem
) -> list[tuple[np.ndarray, float]]:
    """Build the physical integral row only for an admissible global constant gauge.

    Tetrahedral macro geometry checks the original macro quadrature samples.
    Polyhedral geometry checks the actual conforming local tetrahedral samples.
    Both use the same declared boundary tangency rule and original exterior faces.
    """
    data, coefficients = definition.data, definition.coefficients
    gauge = definition.natural_faces == frozenset(data.mesh.boundary_faces)
    if gauge:
        bary, _ = tetrahedron_quadrature(coefficients["order"])
        meshes = (
            tuple(record[0] for record in system.local_metadata)
            if isinstance(data.mesh, PolyhedralMesh)
            else (data.mesh,)
        )
        for mesh in meshes:
            points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells]).reshape(-1, 3)
            gauge = gauge and not np.any(scalar_values_3d(coefficients["reaction"], points))
            gauge = gauge and not np.any(
                scalar_values_3d(coefficients["velocity_divergence"], points)
            )
        gauge = gauge and boundary_tangent_3d(
            data.skeleton, coefficients["velocity"], coefficients["order"]
        )
    if gauge:
        return [
            system.mean_constraint(
                [record[1] for record in system.local_metadata],
                definition.mean_value * float(data.mesh.volumes.sum()),
            )
        ]
    if definition.mean_value != 0:
        raise ValueError(
            "mean_value requires all-Robin zero-reaction divergence-free tangential data"
        )
    return []


def recover_scalar_3d(
    data: Scalar3DDiscretization,
    system: MultiscaleSystem,
    result: HybridSolution,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0, 0.0),
) -> RAD3DSolution:
    """Interpret scalar coefficients and physical flux -K grad(u)+beta*u without averaging."""
    return RAD3DSolution(
        tuple(record[0] for record in system.local_metadata),
        result.fields,
        data.degree,
        diffusion,
        velocity,
        result,
        data.skeleton,
        result.residual,
    )


def recover_transport_3d(
    definition: Scalar3DDefinition, system: MultiscaleSystem, result: HybridSolution
) -> RAD3DSolution | PolyhedralRADSolution:
    """Interpret the executed coefficients in their original volume and macroface spaces."""
    field = recover_scalar_3d(
        definition.data,
        system,
        result,
        diffusion=definition.coefficients["diffusion"],
        velocity=definition.coefficients["velocity"],
    )
    if isinstance(definition.data.mesh, PolyhedralMesh):
        return PolyhedralRADSolution(field, definition.data.skeleton, result)
    return field
