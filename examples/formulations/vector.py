"""Global vector equations with explicitly declared local forms and physical gauges.

The local providers spell out A, B and C=-B.T in the chosen nodal spaces. Here
weak Dirichlet data give the global functional, negative physical traction gives
fixed multiplier coordinates, and integral rows identify only physical null
modes. These definitions are usable independently of the execution controllers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from pymhm import Equation, MultiscaleProblem
from pymhm.core.validation import dyadic_refinement, positive_int, real_array
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.normal import boundary_tangent_3d, trace_represents
from pymhm.fem.traces.physical import boundary_normal_integral, require_compatible_displacement_flux
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.fem.traces.vector_boundary_3d import vector_boundary_components_3d
from pymhm.fem.vector.compatibility import require_strain_trace_compatibility
from pymhm.fem.vector.elasticity_3d import vector_boundary_data_3d
from pymhm.fem.vector.flow import advection_contract
from pymhm.fem.vector.flow_3d import flow_contract_3d
from pymhm.materials.evaluation import vector_values, vector_values_3d
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.refinement import validate_submesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh

from .elasticity import (
    DisplacementPressureSpace,
    displacement_pressure_equations,
    displacement_pressure_fields,
)
from .flow import VelocityPressureSpace, velocity_pressure_equations, velocity_pressure_fields
from .tetrahedral_vector import (
    TetrahedralVectorSpace,
    tetra_displacement_fields,
    tetra_displacement_pressure_equations,
    tetra_velocity_fields,
    tetra_velocity_pressure_equations,
)


@dataclass(frozen=True)
class VectorDefinition:
    """Retain the literal spaces, boundary functional, physical data and gauge declarations."""

    problem: MultiscaleProblem[int]
    space: Any
    dimension: int
    family: str
    formulation: str
    material: Any
    secondary_material: Any
    advection: Any
    boundary: Any
    traction: dict[int, Any]
    components: dict[int, dict[int, Any]]
    mean_pressure: float
    mean_velocity: Any
    translation_kernel: Any
    rigid_moments: Any
    order: int


def _negative_vector(points: Any, *, field: Any) -> Any:
    """Convert a physical Cauchy traction to negative multiplier values."""
    return -vector_values(field, points)


def define_flow(
    mesh: TriangleMesh | TetraMesh,
    *,
    skeleton: Any = None,
    degree: int | None = None,
    formulation: str = "taylor-hood",
    local_refinement: Any = None,
    local_meshes: Any = None,
    viscosity: float = 1.0,
    drag: Any = 0.0,
    advection: Any = None,
    advection_divergence: Any = None,
    advection_bound: float | None = None,
    source: Any = None,
    dirichlet: Any = None,
    traction: dict[int, Any] | None = None,
    traction_components: dict[int, dict[int, Any]] | None = None,
    mean_pressure: float = 0.0,
    mean_velocity: Any = None,
    translation_kernel: Any = None,
    stabilization: str = "tensor-2025",
    gamma_min: float | None = None,
    quadrature_order: int = 5,
) -> VectorDefinition:
    """Declare grad-grad Stokes/Brinkman/Oseen in Pk/P(k-1) or stabilized Pk/Pk.

    The multiplier is negative outward pseudotraction
    (nu grad(u)-p I)n-(beta.n)u/2. Unspecified exterior components have weak
    Dirichlet velocity. The global pressure row exists precisely when no normal
    traction is prescribed. Small positive resistance never becomes a kernel.
    A declared rotated translation kernel must be both physically unresisted
    and free on every exterior component. In 3D tangential advection additionally
    needs representable normal traces on all faces before a translation gauge.
    """
    dim = mesh.points.shape[1]
    beta = (0.0,) * dim if advection is None else advection
    source = (0.0,) * dim if source is None else source
    datum = (0.0,) * dim if dirichlet is None else dirichlet
    mean_velocity = (0.0,) * dim if mean_velocity is None else mean_velocity
    mean = real_array(mean_pressure, "mean_pressure")
    if mean.ndim:
        raise ValueError("mean_pressure must be a scalar")
    natural, components = dict(traction or {}), dict(traction_components or {})
    if dim == 3:
        nu, beta, divergence, bound, minimum, k = flow_contract_3d(
            viscosity,
            drag,
            beta,
            advection_divergence,
            advection_bound,
            formulation,
            stabilization,
            gamma_min,
            degree,
        )
        refinement = dyadic_refinement(
            (4 if k == 1 else 2) if local_refinement is None else local_refinement,
            "local_refinement",
        )
        if local_meshes is not None:
            raise ValueError("3D nodal flow declares uniform tetrahedral refinement")
        skeleton = TriangularSkeleton(mesh, degree=1) if skeleton is None else skeleton
        if skeleton.mesh is not mesh or np.any(skeleton.subdivisions > refinement):
            raise ValueError("local refinement must resolve this mesh's trace partitions")
        order = max(positive_int(quadrature_order, "quadrature_order"), k + 2)
        space = TetrahedralVectorSpace(
            mesh, skeleton, k, k - 1 if formulation == "taylor-hood" else k, refinement, order
        )
        provider = partial(
            tetra_velocity_pressure_equations,
            data=space,
            source=source,
            viscosity=nu,
            drag=drag,
            advection=beta,
            advection_divergence=divergence,
            advection_bound=bound,
            formulation=formulation,
            stabilization=stabilization,
            gamma_min=minimum,
        )
        boundary, fixed = vector_boundary_components_3d(skeleton, datum, natural, components, order)
        interface_size = 3 * skeleton.size
    else:
        if not isinstance(mesh, (TriangleMesh, PolygonMesh)):
            raise TypeError("flow requires a triangular or tetrahedral mesh")
        if not np.isfinite(viscosity) or viscosity <= 0:
            raise ValueError("viscosity must be finite and positive")
        if formulation not in ("taylor-hood", "usfem", "oseen"):
            raise ValueError("unknown flow formulation")
        if stabilization not in ("tensor-2025", "minimum-2017", "pointwise-2017"):
            raise ValueError("unknown USFEM stabilization")
        if formulation != "usfem" and stabilization != "tensor-2025":
            raise ValueError("USFEM stabilization choices require formulation=usfem")
        if gamma_min is not None and stabilization != "minimum-2017":
            raise ValueError("gamma_min is only used with minimum-2017")
        k = (
            (2 if formulation == "taylor-hood" else 1)
            if degree is None
            else positive_int(degree, "degree")
        )
        if formulation == "taylor-hood" and k < 2:
            raise ValueError("Taylor-Hood velocity degree must be at least two")
        beta, divergence, bound = advection_contract(
            beta, advection_divergence, advection_bound, stabilized=formulation == "oseen"
        )
        if formulation == "usfem" and (callable(beta) or np.any(beta)):
            raise ValueError("advection requires Taylor-Hood or Oseen")
        refinement = 4 if local_refinement is None else local_refinement
        if np.ndim(refinement):
            levels = tuple(positive_int(r, "local_refinement") for r in refinement)
            if len(levels) != len(mesh.cells):
                raise ValueError("one refinement count is needed per macrocell")
            local_meshes = tuple(mesh.submesh(i, r) for i, r in enumerate(levels))
            refinement = 1
        else:
            refinement = positive_int(refinement, "local_refinement")
        if local_meshes is not None:
            if len(local_meshes) != len(mesh.cells):
                raise ValueError("one fine partition is needed per macrocell")
            for cell, fine in enumerate(local_meshes):
                validate_submesh(mesh, cell, fine)
        skeleton = SkeletonSpace(mesh, components=2) if skeleton is None else skeleton
        if skeleton.mesh is not mesh or skeleton.components != 2:
            raise ValueError("flow requires a two-component trace on this mesh")
        order = max(positive_int(quadrature_order, "quadrature_order"), k + 2)
        space = VelocityPressureSpace(
            mesh, skeleton, k, k - 1 if formulation == "taylor-hood" else k, refinement, order
        )
        provider = partial(
            velocity_pressure_equations,
            space=space,
            source=source,
            viscosity=viscosity,
            drag=drag,
            advection=beta,
            advection_divergence=divergence,
            advection_bound=bound,
            formulation=formulation,
            stabilization=stabilization,
            gamma_min=gamma_min,
            local_meshes=local_meshes,
        )
        boundary, physical = boundary_data(
            skeleton, datum, natural, neumann_components=components, order=order
        )
        fixed = {i: -v for i, v in physical.items()}
        interface_size = skeleton.size
    normal_traction = bool(natural) or any(
        mesh.normals[face, component] != 0
        for face, values in components.items()
        for component in values
    )
    if normal_traction and mean != 0:
        raise ValueError("mean_pressure is only a gauge without normal traction")
    problem = MultiscaleProblem(
        Equation(0, np.r_[-boundary, np.zeros(dim * len(mesh.cells))]),
        provider,
        range(len(mesh.cells)),
        interface_size,
        (dim,) * len(mesh.cells),
        fixed=fixed,
    )
    return VectorDefinition(
        problem,
        space,
        dim,
        "flow",
        formulation,
        viscosity,
        drag,
        beta,
        boundary,
        natural,
        components,
        float(mean),
        mean_velocity,
        translation_kernel,
        None,
        order,
    )


def flow_constraints(definition: VectorDefinition, system: Any) -> list[Any]:
    """Declare pressure/translation integral rows from physical boundary and material data."""
    dim, mesh = definition.dimension, definition.space.mesh
    volume = float(sum(mesh.areas if dim == 2 else mesh.volumes))
    records = system.local_metadata
    natural, components = definition.traction, definition.components
    normal_traction = bool(natural) or any(
        mesh.normals[face, component] != 0
        for face, values in components.items()
        for component in values
    )
    constraints = (
        []
        if normal_traction
        else [system.mean_constraint([r[2] for r in records], definition.mean_pressure * volume)]
    )
    free = np.array(
        [
            all(
                face in natural or component in components.get(int(face), {})
                for face in mesh.boundary_faces
            )
            for component in range(dim)
        ]
    )
    beta = definition.advection
    if dim == 2:
        eligible = np.any(free) and not (callable(beta) or np.any(beta))
        structural = np.all([r[6] for r in records], axis=0)
        resistance = sum(r[4] for r in records)
        resistance_scale = sum(r[5] for r in records)
        moments = [r[3] for r in records]
    else:
        eligible = np.any(free) and boundary_tangent_3d(
            definition.space.skeleton, beta, max(8, definition.order)
        )
        structural = np.all([r[3].zero_columns for r in records], axis=0)
        resistance = sum(r[3].resistance_moment for r in records)
        resistance_scale = sum(r[3].absolute_resistance_moment for r in records)
        moments = [r[4] for r in records]
    declared = definition.translation_kernel
    mean = (vector_values if dim == 2 else vector_values_3d)(
        definition.mean_velocity, np.zeros((1, dim))
    )[0]
    if not eligible:
        if declared is not None:
            raise ValueError("translation_kernel requires free traction and admissible advection")
        if dim == 3 and np.any(mean):
            raise ValueError("mean_velocity requires free unresisted translations")
        return constraints
    if declared is None:
        nullspace = np.eye(dim)[:, structural & free]
    else:
        nullspace = real_array(declared, "translation_kernel")
        if (
            nullspace.ndim != 2
            or nullspace.shape[0] != dim
            or not 1 <= nullspace.shape[1] <= dim
            or np.linalg.matrix_rank(nullspace) != nullspace.shape[1]
        ):
            raise ValueError("translation_kernel must have independent physical columns")
        if np.any(nullspace[~free] != 0):
            raise ValueError("translation_kernel changes a Dirichlet component")
        nullspace = np.linalg.qr(nullspace)[0]
        if np.any(
            abs(resistance @ nullspace)
            > 128 * np.finfo(float).eps * (resistance_scale @ abs(nullspace))
        ):
            raise ValueError("translation_kernel is resisted by the material")
    if np.linalg.norm(mean - nullspace @ (nullspace.T @ mean)) > 1e-12 * max(
        1.0, np.linalg.norm(mean)
    ):
        raise ValueError("mean_velocity may only prescribe unresisted translation directions")
    if dim == 3 and nullspace.shape[1]:
        for face, normal in enumerate(mesh.normals):
            datum = partial(_normal_advection, beta=beta, normal=normal)
            if not trace_represents(
                definition.space.skeleton, face, datum, order=max(8, definition.order)
            ):
                raise ValueError("translation gauge requires representable normal advection")
    constraints.extend(
        system.mean_constraint([m @ direction for m in moments], float(direction @ mean) * volume)
        for direction in nullspace.T
    )
    return constraints


def _normal_advection(points: Any, *, beta: Any, normal: Any) -> Any:
    """Return a physical scalar trace supplied to the generic representation check."""
    return vector_values_3d(beta, points) @ normal


def define_herrmann_elasticity(
    mesh: TriangleMesh | TetraMesh,
    *,
    skeleton: Any = None,
    degree: int | None = None,
    formulation: str = "gals",
    local_refinement: int | None = None,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    lame_mu_gradient: Any = None,
    shear_bounds: Any = None,
    stabilization_alpha: float | None = None,
    source: Any = None,
    dirichlet: Any = None,
    neumann: dict[int, Any] | None = None,
    traction: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    rigid_moments: Any = None,
    quadrature_order: int = 8,
) -> VectorDefinition:
    """Declare Herrmann GaLS Pk/Pk or Taylor--Hood Pk/P(k-1) and physical moment rows.

    Pressure is -lambda div(u). For a full displacement boundary finite lambda
    determines its weighted integral through compliance and the exact assembled
    boundary normal functional. At lambda infinity the integral becomes a true
    pressure gauge and displacement data must have zero net normal flux.
    Pure physical traction has three/six rigid integral targets. Residual bounds,
    variable shear derivatives, and matching trace hypotheses remain explicit.
    """
    dim = mesh.points.shape[1]
    if formulation not in ("gals", "taylor-hood"):
        raise ValueError("elasticity formulation must be gals or taylor-hood")
    k = (1 if formulation == "gals" else 2) if degree is None else positive_int(degree, "degree")
    if formulation == "taylor-hood" and k < 2:
        raise ValueError("Taylor-Hood displacement degree must be at least two")
    if neumann is not None and traction is not None:
        raise ValueError("provide one physical traction declaration")
    natural = dict(traction if traction is not None else (neumann or {}))
    datum = (0.0,) * dim if dirichlet is None else dirichlet
    source = (0.0,) * dim if source is None else source
    mean = real_array(mean_pressure, "mean_pressure")
    if mean.ndim or (natural and mean != 0):
        raise ValueError("mean_pressure is a scalar gauge for full displacement only")
    refinement = (4 if k == 1 else 2) if local_refinement is None else local_refinement
    order = max(positive_int(quadrature_order, "quadrature_order"), k + 2)
    width = 3 if dim == 2 else 6
    targets = (
        np.zeros(width) if rigid_moments is None else real_array(rigid_moments, "rigid_moments")
    )
    if targets.shape != (width,):
        raise ValueError("rigid_moments needs one integrated displacement target per rigid mode")
    if dim == 2:
        refinement = positive_int(refinement, "local_refinement")
        skeleton = SkeletonSpace(mesh, components=2) if skeleton is None else skeleton
        if skeleton.mesh is not mesh or skeleton.components != 2:
            raise ValueError("Herrmann elasticity requires a two-component trace on this mesh")
        if formulation == "gals":
            require_strain_trace_compatibility(skeleton, k, refinement)
        negative = {face: partial(_negative_vector, field=value) for face, value in natural.items()}
        boundary, fixed = boundary_data(skeleton, datum, negative, order=order)
        space = DisplacementPressureSpace(
            mesh, skeleton, refinement, order, k, k if formulation == "gals" else k - 1
        )
        provider = partial(
            displacement_pressure_equations,
            space=space,
            source=source,
            lame_lambda=lame_lambda,
            lame_mu=lame_mu,
            lame_mu_gradient=(0.0, 0.0) if lame_mu_gradient is None else lame_mu_gradient,
            shear_bounds=shear_bounds,
            formulation=formulation,
            stabilization_alpha=stabilization_alpha,
        )
        interface_size = skeleton.size
    else:
        refinement = dyadic_refinement(refinement, "local_refinement")
        skeleton = TriangularSkeleton(mesh, degree=1) if skeleton is None else skeleton
        if skeleton.mesh is not mesh or np.any(skeleton.subdivisions > refinement):
            raise ValueError("local refinement must resolve the trace subdivisions")
        boundary, fixed = vector_boundary_data_3d(skeleton, datum, natural, order)
        space = TetrahedralVectorSpace(
            mesh, skeleton, k, k if formulation == "gals" else k - 1, refinement, order
        )
        provider = partial(
            tetra_displacement_pressure_equations,
            data=space,
            source=source,
            lame_lambda=lame_lambda,
            lame_mu=lame_mu,
            lame_mu_gradient=lame_mu_gradient,
            shear_bounds=shear_bounds,
            formulation=formulation,
            stabilization_alpha=stabilization_alpha,
        )
        interface_size = 3 * skeleton.size
    problem = MultiscaleProblem(
        Equation(0, np.r_[-boundary, np.zeros(width * len(mesh.cells))]),
        provider,
        range(len(mesh.cells)),
        interface_size,
        (width,) * len(mesh.cells),
        fixed=fixed,
    )
    return VectorDefinition(
        problem,
        space,
        dim,
        "elasticity",
        formulation,
        lame_lambda,
        lame_mu,
        None,
        boundary,
        natural,
        {},
        float(mean),
        None,
        None,
        targets,
        order,
    )


def herrmann_constraints(definition: VectorDefinition, system: Any) -> list[Any]:
    """Declare compressibility, incompressible pressure and free rigid-motion rows."""
    records = system.local_metadata
    scale = max(r[6] for r in records)
    if definition.mean_pressure and scale > 0:
        raise ValueError("finite lambda determines pressure through compressibility")
    constraints = []
    mesh = definition.space.mesh
    if not definition.traction:
        flux = boundary_normal_integral(definition.space.skeleton, definition.boundary)
        if scale == 0:
            require_compatible_displacement_flux(
                flux,
                boundary_normal_integral(
                    definition.space.skeleton, definition.boundary, absolute=True
                ),
            )
            moments = [r[2] for r in records]
            volume = float(sum(mesh.areas if definition.dimension == 2 else mesh.volumes))
            target = definition.mean_pressure * volume
        else:
            moments = [r[5] / scale for r in records]
            target = -flux / scale
        constraints.append(system.mean_constraint(moments, target))
    if set(definition.traction) == set(mesh.boundary_faces):
        constraints.extend(
            system.mean_constraint([r[4][:, i] for r in records], float(target))
            for i, target in enumerate(definition.rigid_moments)
        )
    return constraints


def recover_vector(definition: VectorDefinition, system: Any, solution: Any) -> Any:
    """Interpret the actual coefficients without selecting a discretization or changing gauges."""
    if definition.family == "flow":
        return (
            velocity_pressure_fields(system, solution, definition.space)
            if definition.dimension == 2
            else tetra_velocity_fields(
                system,
                solution,
                definition.space,
                definition.material,
                definition.advection,
                definition.formulation,
            )
        )
    return (
        displacement_pressure_fields(
            system,
            solution,
            definition.space,
            definition.material,
            definition.secondary_material,
            definition.formulation,
        )
        if definition.dimension == 2
        else tetra_displacement_fields(
            system,
            solution,
            definition.space,
            definition.material,
            definition.secondary_material,
            definition.formulation,
        )
    )
