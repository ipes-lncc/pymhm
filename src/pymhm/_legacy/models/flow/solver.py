"""Arbitrary-order Taylor--Hood and residual-stabilized Stokes--Brinkman locals."""

from functools import partial
from typing import Any, Literal, cast

import numpy as np

from pymhm._legacy.models.vector import VectorSolution
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.vector.flow import advection_contract as _advection_contract
from pymhm.fem.vector.flow import minimum_resistance as _minimum_resistance
from pymhm.fem.vector.flow import triangle_flow_operators
from pymhm.materials.evaluation import vector_values
from pymhm.materials.resistance import resistance_values as _resistance_values
from pymhm.meshes.refinement import validate_submesh
from pymhm.meshes.triangle import TriangleMesh


def _flow_local(
    cell: int,
    *,
    mesh: TriangleMesh,
    skeleton: SkeletonSpace,
    degree: int,
    formulation: str,
    refinement: int | tuple[int, ...],
    viscosity: float,
    drag: Any,
    beta: Any,
    source: Any,
    order: int,
    gamma_min: float | None = None,
    pointwise: bool = False,
    beta_divergence: Any = 0.0,
    beta_bound: float | None = None,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
) -> LocalAssembly:
    """Assemble local velocity-pressure forms, resolving Cartesian material cuts."""
    fine = (
        mesh.submesh(cell, refinement if isinstance(refinement, int) else refinement[cell])
        if local_meshes is None
        else local_meshes[cell]
    )
    forms = triangle_flow_operators(
        fine,
        degree=degree,
        formulation=formulation,
        viscosity=viscosity,
        drag=drag,
        beta=beta,
        source=source,
        order=order,
        gamma_min=gamma_min,
        pointwise=pointwise,
        beta_divergence=beta_divergence,
        beta_bound=beta_bound,
    )
    nv, size = len(forms.velocity_nodes), len(forms.load)
    coupling = np.zeros((size, len(skeleton.cell_dofs(cell))))
    coupling[: 2 * nv] = np.kron(trace_coupling(mesh, cell, fine, skeleton, degree), np.eye(2))
    retained = {"kernel": forms.kernel} if forms.pure else {"coarse_basis": forms.kernel}
    problem = LocalProblem(
        forms.matrix,
        coupling,
        forms.load,
        skeleton.cell_dofs(cell),
        constraints=forms.translation_moments,
        **retained,
    )
    return LocalAssembly(
        problem,
        (
            fine,
            nv,
            forms.pressure_moments,
            forms.translation_moments,
            forms.resistance_moment,
            forms.absolute_resistance_moment,
            forms.zero_columns,
        ),
    )


def solve_flow(
    mesh: TriangleMesh,
    *,
    viscosity: float = 1.0,
    drag: Any = 0.0,
    advection: Any = (0.0, 0.0),
    advection_divergence: Any = None,
    advection_bound: float | None = None,
    source: Any = (0.0, 0.0),
    dirichlet: Any = (0.0, 0.0),
    traction: dict[int, Any] | None = None,
    traction_components: dict[int, dict[int, Any]] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int | tuple[int, ...] = 2,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
    quadrature_order: int = 5,
    formulation: str = "taylor-hood",
    stabilization: Literal["tensor-2025", "minimum-2017", "pointwise-2017"] = "tensor-2025",
    gamma_min: float | None = None,
    degree: int | None = None,
    mean_pressure: float = 0.0,
    mean_velocity: Any = (0.0, 0.0),
    translation_kernel: Any = None,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> VectorSolution:
    """Solve Stokes, Brinkman or Oseen with arbitrary-order local finite elements.

    ``formulation="oseen"`` uses the equal-order stabilized form of Araya,
    Cárcamo, Poza and Valentin (2021), Eqs. (25)--(27): positive grad-div and
    negative trial/adjoint-test residual products, including their load terms.
    It requires constant scalar nonnegative ``drag``. Its strict coercivity
    theorem assumes drag-div(advection)/2>0; the published zero-drag internal
    layer remains a supported experiment outside that sufficient hypothesis.
    The parameters are delta=h²/[max(drag*h²,4*viscosity/m)
    +max(4*viscosity/m,advection_bound*h)] and
    kappa=advection_bound*h*min(1,m*advection_bound*h/(4*viscosity)).

    ``advection`` accepts a constant vector or a pointwise callable. Callables
    require their analytical ``advection_divergence`` (explicit zero for a
    solenoidal field), so the skew form represents beta.grad(u), including the
    correction -div(beta)*u/2. Stabilized callable advection additionally
    requires a certified global ``advection_bound`` for its Euclidean norm;
    the bound is checked at quadrature points, not inferred from sampling.

    USFEM uses the full residual and source terms of Araya et al. (2017),
    Eqs. (41)--(42), with an explicitly selected stabilization coefficient.
    ``stabilization="tensor-2025"`` uses the elementwise maximum eigenvalue
    bound of Araya, Harder, Poza and Valentin (2025). ``"minimum-2017"`` uses
    kappa=h²/[max(gamma_min*h²,4*viscosity/m)+4*viscosity/m], m=min(1/3,C),
    with C computed from the local polynomial inverse inequality. This formula
    has its continuous Stokes limit at gamma_min=0. ``"pointwise-2017"``
    applies the same formula to the smallest eigenvalue at each integration
    point, so kappa varies inside a finite element. These are explicit spatial
    interpretations of the minimum coefficient in the heterogeneous 2017
    formula; neither convention is identified as its undocumented historical code.

    For the global ``"minimum-2017"`` choice, ``gamma_min`` is an explicit global lower eigenvalue
    bound. It defaults to the exact minimum for constant scalar/tensor drag or
    over every pixel of a CartesianCellField. Variable callbacks require an
    explicit bound, checked at integration points; unsampled callback values
    remain the caller's responsibility. A global minimum does not guarantee
    coercivity of the negatively stabilized velocity form in a high-contrast
    material; a small algebraic residual alone cannot establish stability.
    This declared spatial convention does
    not assume access to the historical article's diagnostic implementation.
    Viscosity is a positive constant; drag accepts a nonnegative scalar,
    symmetric tensor or pointwise callable. CartesianCellField resistance is
    integrated on geometric pixel/triangle intersections. Other coefficient
    callbacks use the declared Gaussian quadrature. ``stabilization`` options
    apply only to USFEM; the Oseen formulation uses its own published parameters.

    ``traction`` prescribes ``(viscosity*grad(u)-p*I)n-u*(advection.n)/2``;
    this is the grad-grad pseudotraction, not symmetric Cauchy traction.
    ``traction_components[face][i]`` prescribes just Cartesian component i of
    that same outward traction, leaving other components Dirichlet. Full and
    componentwise declarations must not overlap. Pressure has a mean gauge
    precisely when no prescribed traction component has a nonzero normal
    component. Thus free tangential traction on axis-aligned slip walls does
    not by itself fix pressure. In a pure-traction Stokes problem, prescribed
    ``mean_velocity`` fixes both translations; componentwise boundaries can
    leave only one translation free.
    With semidefinite drag only unresisted directions are gauges. Structural zero
    columns are detected exactly; a rotated nullspace can be declared with
    ``translation_kernel`` (a 2-by-r basis), then checked componentwise against
    the material. Small positive diagonal resistances are never turned into gauges.
    All local assembly and condensation execute in the selected worker backend.
    Explicit ``local_meshes`` must form conforming partitions of their respective
    macrotriangles; they replace uniform subdivision without changing local
    polynomial spaces, stabilization or skeletal integration.
    """
    if not np.isfinite(viscosity) or viscosity <= 0:
        raise ValueError("viscosity must be finite and positive")
    if np.ndim(local_refinement) == 0:
        local_refinement = positive_int(local_refinement, "local_refinement")
    else:
        local_refinement = tuple(
            positive_int(r, "local_refinement") for r in cast(tuple[int, ...], local_refinement)
        )
        if len(local_refinement) != len(mesh.cells):
            raise ValueError("local_refinement needs one subdivision count per macrocell")
    positive_int(quadrature_order, "quadrature_order")
    if local_meshes is not None:
        if len(local_meshes) != len(mesh.cells):
            raise ValueError("local_meshes needs one mesh per macrocell")
        for cell, fine in enumerate(local_meshes):
            validate_submesh(mesh, cell, fine)
    if formulation not in ("taylor-hood", "usfem", "oseen"):
        raise ValueError("formulation must be taylor-hood, usfem or oseen")
    if stabilization not in ("tensor-2025", "minimum-2017", "pointwise-2017"):
        raise ValueError("stabilization must be tensor-2025, minimum-2017 or pointwise-2017")
    if gamma_min is not None and stabilization != "minimum-2017":
        raise ValueError("gamma_min is only used by minimum-2017 stabilization")
    if formulation != "usfem" and stabilization != "tensor-2025":
        raise ValueError("USFEM stabilization options do not apply to Taylor-Hood or Oseen")
    minimum = _minimum_resistance(drag, gamma_min) if stabilization == "minimum-2017" else None
    degree = (
        (2 if formulation == "taylor-hood" else 1)
        if degree is None
        else positive_int(degree, "degree")
    )
    if formulation == "taylor-hood" and degree < 2:
        raise ValueError("Taylor-Hood velocity degree must be at least two")
    beta, beta_divergence, beta_bound = _advection_contract(
        advection, advection_divergence, advection_bound, stabilized=formulation == "oseen"
    )
    advective = callable(beta) or np.any(beta)
    if advective and formulation == "usfem":
        raise ValueError("Oseen advection requires Taylor-Hood or the oseen formulation")
    if formulation == "oseen":
        raw_drag = np.asarray(drag)
        if callable(drag) or raw_drag.ndim != 0 or np.iscomplexobj(raw_drag):
            raise ValueError("published stabilized Oseen requires constant scalar drag")
        _resistance_values(drag, np.zeros((1, 2)))
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    skeleton = SkeletonSpace(mesh, components=2) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 2:
        raise ValueError("Brinkman requires a two-component skeleton on this mesh")
    traction = {} if traction is None else traction
    traction_components = {} if traction_components is None else traction_components
    boundary, physical_fixed = boundary_data(
        skeleton,
        dirichlet,
        traction,
        neumann_components=traction_components,
        order=max(quadrature_order, degree + 2),
    )
    # The hybrid multiplier is the negative outward physical pseudotraction.
    fixed = {dof: -value for dof, value in physical_fixed.items()}
    normal_traction = bool(traction) or any(
        mesh.normals[face, component] != 0.0
        for face, components in traction_components.items()
        for component in components
    )
    if normal_traction and mean_pressure != 0:
        raise ValueError("mean_pressure is only a gauge when no normal traction is prescribed")
    factory = partial(
        _flow_local,
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        formulation=formulation,
        refinement=local_refinement,
        viscosity=viscosity,
        drag=drag,
        beta=beta,
        source=source,
        order=quadrature_order,
        gamma_min=minimum,
        pointwise=stabilization == "pointwise-2017",
        beta_divergence=beta_divergence,
        beta_bound=beta_bound,
        local_meshes=local_meshes,
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    gauges = []
    if not normal_traction:
        gauges.append(
            system.mean_constraint(
                [data[2] for data in system.local_metadata], mean_pressure * sum(mesh.areas)
            )
        )
    free_translations = np.array(
        [
            all(
                face in traction or component in traction_components.get(face, {})
                for face in mesh.boundary_faces
            )
            for component in range(2)
        ]
    )
    translation_gauge = np.any(free_translations) and not advective
    if translation_kernel is not None and not translation_gauge:
        raise ValueError("translation_kernel requires a pure-traction direction without advection")
    if translation_gauge:
        if translation_kernel is None:
            structural = (
                np.all([data[6] for data in system.local_metadata], axis=0) & free_translations
            )
            nullspace = np.eye(2)[:, structural]
        else:
            nullspace = np.asarray(translation_kernel, dtype=float)
            if (
                nullspace.ndim != 2
                or nullspace.shape[0] != 2
                or not 1 <= nullspace.shape[1] <= 2
                or not np.isfinite(nullspace).all()
                or np.linalg.matrix_rank(nullspace) != nullspace.shape[1]
            ):
                raise ValueError("translation_kernel must be a finite independent 2-by-r basis")
            if np.any(nullspace[~free_translations] != 0):
                raise ValueError(
                    "declared translation_kernel changes a Dirichlet boundary component"
                )
            nullspace = np.linalg.qr(nullspace)[0]
            material = sum(data[4] for data in system.local_metadata)
            scale = sum(data[5] for data in system.local_metadata) @ np.abs(nullspace)
            if np.any(np.abs(material @ nullspace) > 128 * np.finfo(float).eps * scale):
                raise ValueError("declared translation_kernel is resisted by the material")
        mean = vector_values(mean_velocity, np.zeros((1, 2)))[0]
        if np.linalg.norm(mean - nullspace @ (nullspace.T @ mean)) > 1e-12 * max(
            1.0, np.linalg.norm(mean)
        ):
            raise ValueError("mean_velocity can only prescribe unresisted translation directions")
        for direction in nullspace.T:
            gauges.append(
                system.mean_constraint(
                    [data[3] @ direction for data in system.local_metadata],
                    float(direction @ mean) * sum(mesh.areas),
                )
            )
    result = system.solve(solver=solver, fixed=fixed, constraints=gauges)
    sizes = [data[1] for data in system.local_metadata]
    return VectorSolution(
        skeleton,
        tuple(data[0] for data in system.local_metadata),
        tuple(
            field[: 2 * nv].reshape(-1, 2) for field, nv in zip(result.fields, sizes, strict=True)
        ),
        tuple(field[2 * nv :] for field, nv in zip(result.fields, sizes, strict=True)),
        result,
        degree,
        degree - 1 if formulation == "taylor-hood" else degree,
    )
