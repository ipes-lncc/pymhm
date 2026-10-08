"""User-defined planar elasticity forms and subcycled Newmark endpoint equations.

Spatial matrices use common nodal, quadrature, scatter and trace kernels.
The time response uses the shared dimension-independent Newmark primitive;
the generic multiscale core imposes the declared endpoint displacement moments.
Only field storage/evaluation records are reused from the compatibility model.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import numpy as np
from scipy import sparse

from examples.formulations.original import solve_original
from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.multiscale import MultiscaleProblem, assemble
from pymhm.core.online import OfflineMultiscaleSystem
from pymhm.execution.cpu import map_local
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.fem.scalar.triangle import nodal_space, trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.fem.vector.curl import physical_basis, physical_points, quadrature
from pymhm.fem.vector.elasticity_3d import KELVIN_BASIS_3D, vector_boundary_data_3d
from pymhm.linalg.dynamics import newmark_step
from pymhm.linalg.linear import LinearFactorization, LinearSolveError, factorize
from pymhm.materials.elasticity import (
    KELVIN_BASIS_2D,
    constitutive_values,
    constitutive_values_3d,
)
from pymhm.materials.evaluation import scalar_values_3d
from pymhm.materials.sources import SeparableTriangleField, TriangleQuadratureField
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.dynamics import (
    ElastodynamicLocal,
    ElastodynamicSolution,
    PreparedElastodynamicSource,
)
from pymhm.postprocessing.nodal import nodal_field


@dataclass(frozen=True)
class NewmarkOperators:
    """Executed coefficient bases and owner-held factors for one macro time step."""

    skeleton: Any
    locals: tuple[ElastodynamicLocal, ...]
    mass_factors: tuple[LinearFactorization, ...]
    step_factors: tuple[LinearFactorization, ...]
    lifts: tuple[Any, ...]
    velocity_lifts: tuple[Any, ...]
    time_step: float
    substeps: tuple[int, ...]
    boundary: Any
    fixed: dict[int, float]
    endpoint_system: Any = None
    endpoint_factor: Any = None

    @property
    def trace_size(self) -> int:
        """Return interleaved Cartesian traction coordinates in the declared dimension."""
        return self.skeleton.size * (3 if isinstance(self.skeleton, TriangularSkeleton) else 1)


def prepare_source(
    data: NewmarkOperators, source: SeparableTriangleField
) -> PreparedElastodynamicSource:
    """Integrate a declared separable planar source once in the executed local test bases.

    The snapshot retains physical force-density vectors without a density factor.
    Its temporal scalar is evaluated at every original Newmark endpoint/substep;
    no time equation or alternate spatial rule is selected by this operation.
    """
    if data.locals[0].nodes.shape[1] != 2:
        raise ValueError("prepared triangular sources require two dimensions")
    if not isinstance(source, SeparableTriangleField):
        raise TypeError("source must declare an explicit separable triangular field")
    spatial = source.spatial_field()
    if not isinstance(spatial, TriangleQuadratureField):
        raise TypeError("separable spatial source must provide triangular quadrature")
    return PreparedElastodynamicSource(
        data.locals, tuple(local.load(spatial) for local in data.locals), source.time_scale
    )


def spatial_forms(
    mesh: TriangleMesh | TetraMesh,
    skeleton: Any,
    cell: int,
    degree: int,
    refinement: int,
    order: int,
    density: Any,
    lame_lambda: Any,
    lame_mu: Any,
    constitutive: Any = None,
) -> ElastodynamicLocal:
    """Declare rho mass and lambda div/div plus twice mu symmetric-gradient energy.

    Positive density and elliptic constitutive samples define either planar
    strain or three-dimensional elasticity. Nodes/components are interleaved;
    the canonical traction map is the scalar trace map times the identity.
    This displacement energy has no incompressible-limit guarantee.
    """
    fine = mesh.submesh(cell, refinement)
    dimension = mesh.points.shape[1]
    bary, weights, material_data = quadrature(fine, constitutive, order)
    basis, gradients = physical_basis(fine, degree, bary)
    dofs, nodes = (
        nodal_space(cast(TriangleMesh, fine), degree)
        if dimension == 2
        else tetra_nodal_space(cast(TetraMesh, fine), degree)
    )
    kelvin = KELVIN_BASIS_2D if dimension == 2 else KELVIN_BASIS_3D
    strain = np.einsum("aij,tqnj->tqani", kelvin, gradients).reshape(
        *gradients.shape[:2], len(kelvin), -1
    )
    points = physical_points(fine, bary)
    material = (constitutive_values if dimension == 2 else constitutive_values_3d)(
        material_data, points.reshape(-1, dimension), lame_lambda=lame_lambda, lame_mu=lame_mu
    ).reshape(*weights.shape, len(kelvin), len(kelvin))
    width = dimension * basis.shape[-1]
    vector_dofs = (dimension * dofs[:, :, None] + np.arange(dimension)).reshape(
        len(fine.cells), width
    )
    stiffness = assemble_element_blocks(
        np.einsum("tq,tqai,tqab,tqbj->tij", weights, strain, material, strain),
        vector_dofs,
        vector_dofs,
        (dimension * len(nodes), dimension * len(nodes)),
    )
    bary, weights, density_data = quadrature(fine, density, order)
    basis, _ = physical_basis(fine, degree, bary)
    points = physical_points(fine, bary)
    raw = density_data(points.reshape(-1, dimension)) if callable(density_data) else density_data
    raw = np.asarray(raw)
    if raw.shape[-2:] == (dimension, dimension):
        if not np.all(raw == raw[..., :1, :1] * np.eye(dimension)):
            raise ValueError("density requires scalar or isotropic material values")
        raw = raw[..., 0, 0]
    rho = scalar_values_3d(raw, points.reshape(-1, dimension)).reshape(weights.shape)
    if np.any(rho <= 0):
        raise ValueError("density must be positive")
    scalar_mass = assemble_element_blocks(
        np.einsum("tq,tqi,tqj->tij", weights * rho, basis, basis),
        dofs,
        dofs,
        (len(nodes), len(nodes)),
    )
    mass = sparse.kron(scalar_mass, sparse.eye(dimension), format="csc")
    coupling_scalar = (
        trace_coupling(cast(TriangleMesh, mesh), cell, cast(TriangleMesh, fine), skeleton, degree)
        if dimension == 2
        else tetra_trace_coupling(
            cast(TetraMesh, mesh), cell, cast(TetraMesh, fine), skeleton, degree
        )
    )
    coupling = np.asarray(np.kron(coupling_scalar, np.eye(dimension)), dtype=np.float64)
    trace_dofs = skeleton.cell_dofs(cell)
    if dimension == 3:
        trace_dofs = (dimension * trace_dofs[:, None] + np.arange(dimension)).ravel()
    return ElastodynamicLocal(
        fine,
        degree,
        mass,
        stiffness,
        coupling,
        trace_dofs,
        dofs,
        nodes,
        basis,
        points,
        weights,
        rho,
        constitutive,
        lame_lambda,
        lame_mu,
        order,
    )


@contextmanager
def prepare(
    mesh: TriangleMesh | TetraMesh,
    *,
    time_step: float,
    degree: int = 2,
    local_refinement: int = 2,
    local_substeps: Any = 1,
    quadrature_order: int = 8,
    density: Any = 1.0,
    constitutive: Any = None,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    skeleton: Any = None,
    dirichlet: Any = 0.0,
    traction: dict[int, Any] | None = None,
    backend: Any = "serial",
    workers: int | None = None,
) -> Iterator[NewmarkOperators]:
    """Own reusable factors while preparing the declared simplex coefficient response.

    Each macrocell can use a distinct positive substep count, ending at the
    same macro time. Loads remain physical force densities. The multiplier is
    negative physical traction, constant over a macro time slab; arbitrary
    subcycling does not automatically inherit a global energy identity.
    """
    if not isinstance(mesh, (TriangleMesh, TetraMesh)):
        raise TypeError("this tutorial declares triangular or tetrahedral forms")
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be finite and positive")
    raw_counts = np.broadcast_to(np.asarray(local_substeps), (len(mesh.cells),))
    if any(
        isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1
        for value in raw_counts
    ):
        raise ValueError("local_substeps must contain positive integers")
    counts = tuple(int(value) for value in raw_counts)
    order = max(quadrature_order, degree + 2)
    dimension = mesh.points.shape[1]
    if skeleton is None:
        skeleton = (
            SkeletonSpace(
                cast(TriangleMesh, mesh),
                tuple(FaceSpace.uniform(1) for _ in mesh.faces),
                components=2,
            )
            if dimension == 2
            else TriangularSkeleton(cast(TetraMesh, mesh), degree=1)
        )
    if skeleton.mesh is not mesh:
        raise ValueError("traction skeleton must belong to the supplied mesh")
    if dimension == 2 and skeleton.components != 2:
        raise ValueError("planar traction traces require two Cartesian components")
    traction = {} if traction is None else traction
    if not set(traction).issubset(set(mesh.boundary_faces)):
        raise ValueError("traction keys must identify exterior faces")
    if dimension == 2:
        target, physical_tractions = boundary_data(skeleton, dirichlet, traction, order=order)
        fixed = {int(index): -float(value) for index, value in physical_tractions.items()}
    else:
        target, fixed = vector_boundary_data_3d(skeleton, dirichlet, traction, order)
    locals_ = tuple(
        map_local(
            partial(
                spatial_forms,
                mesh,
                skeleton,
                degree=degree,
                refinement=local_refinement,
                order=order,
                density=density,
                lame_lambda=lame_lambda,
                lame_mu=lame_mu,
                constitutive=constitutive,
            ),
            range(len(mesh.cells)),
            backend=backend,
            workers=workers,
        )
    )
    with ExitStack() as resources:
        masses = tuple(resources.enter_context(factorize(local.mass)) for local in locals_)
        steps = tuple(
            resources.enter_context(
                factorize(local.mass + (time_step / count) ** 2 / 4 * local.stiffness)
            )
            for local, count in zip(locals_, counts, strict=True)
        )
        lifts, velocity_lifts = [], []
        for local, count, mass_factor, step_factor in zip(
            locals_, counts, masses, steps, strict=True
        ):
            u, v = np.zeros_like(local.coupling), np.zeros_like(local.coupling)
            for _ in range(count):
                u, v = newmark_step(
                    local.mass,
                    local.stiffness,
                    step_factor,
                    mass_factor,
                    time_step / count,
                    u,
                    v,
                    local.coupling,
                    local.coupling,
                )
            lifts.append(u)
            velocity_lifts.append(v)
        zero_fields = tuple(np.zeros(local.mass.shape[0]) for local in locals_)
        endpoint = MultiscaleProblem(
            endpoint_global(target),
            endpoint_equations,
            tuple(zip(locals_, zero_fields, lifts, strict=True)),
            len(target),
            (0,) * len(locals_),
            fixed=fixed,
        )
        template = assemble(endpoint)
        offline = resources.enter_context(OfflineMultiscaleSystem(template))
        free = np.setdiff1d(np.arange(len(target)), list(fixed))
        endpoint_factor = (
            resources.enter_context(factorize(template.matrix[free][:, free]))
            if len(free)
            else None
        )
        yield NewmarkOperators(
            skeleton,
            locals_,
            masses,
            steps,
            tuple(lifts),
            tuple(velocity_lifts),
            time_step,
            counts,
            target,
            fixed,
            offline,
            endpoint_factor,
        )


def endpoint_equations(item: tuple[ElastodynamicLocal, Any, Any]) -> LocalEquations:
    """Declare u+U lambda=u_free and -Q.T u as the signed endpoint balance.

    U is the actual subcycled constant-force response computed by newmark_step,
    rather than an approximation based on one macro step. The global core
    solves every face once and reconstructs each independent macrocell field.
    """
    local, free_displacement, lift = item
    return LocalEquations(
        a=sparse.eye(local.mass.shape[0]),
        L=free_displacement,
        b=lift,
        c=-local.coupling.T,
        dofs=local.trace_dofs,
        field_data=(
            nodal_field("displacement", local.mesh, local.degree, components=local.nodes.shape[1]),
        ),
    )


def endpoint_global(boundary: Any) -> Equation:
    """Declare minus the prescribed displacement moments in the global equation."""
    return Equation(0, -boundary)


def projection_equations(item: tuple[ElastodynamicLocal, Any]) -> LocalEquations:
    """Declare physical density-weighted mass projection and signed face moments."""
    local, field = item
    return LocalEquations(
        a=local.mass,
        L=local.load(field, density_weighted=True),
        b=local.coupling,
        c=-local.coupling.T,
        dofs=local.trace_dofs,
        field_data=(
            nodal_field("projection", local.mesh, local.degree, components=local.nodes.shape[1]),
        ),
    )


def _state(
    data: NewmarkOperators, u: tuple[Any, ...], v: tuple[Any, ...], trace: Any, time: float
) -> ElastodynamicSolution:
    """Store fields and check the original free-face displacement constraint."""
    moments = np.zeros(data.trace_size)
    for local, values in zip(data.locals, u, strict=True):
        np.add.at(moments, local.trace_dofs, local.coupling.T @ values)
    free = np.setdiff1d(np.arange(data.trace_size), list(data.fixed))
    residual = float(np.linalg.norm((moments - data.boundary)[free]))
    scale = max(
        float(np.linalg.norm(moments)),
        float(np.linalg.norm(data.boundary)),
        sum(
            float(np.linalg.norm(abs(local.coupling).T @ abs(values)))
            for local, values in zip(data.locals, u, strict=True)
        ),
        np.finfo(float).tiny,
    )
    if residual > 1e-10 * scale:
        raise LinearSolveError("declared endpoint equations fail their physical face constraint")
    energy = sum(
        float(values @ (local.stiffness @ values) + velocity @ (local.mass @ velocity)) / 2
        for local, values, velocity in zip(data.locals, u, v, strict=True)
    )
    return ElastodynamicSolution(
        data.skeleton,
        data.locals,
        tuple(values.copy() for values in u),
        tuple(values.copy() for values in v),
        trace.copy(),
        time,
        energy,
        residual,
    )


def initialize(
    data: NewmarkOperators, displacement: Any = 0.0, velocity: Any = 0.0, *, original: bool = False
) -> ElastodynamicSolution:
    """Solve two explicit constrained physical mass projections through the core."""
    fields = []
    for datum, target in ((displacement, data.boundary), (velocity, np.zeros(data.trace_size))):
        items = tuple((local, datum) for local in data.locals)
        problem = MultiscaleProblem(
            endpoint_global(target),
            projection_equations,
            items,
            data.trace_size,
            (0,) * len(items),
            fixed={index: 0.0 for index in data.fixed},
        )
        system = assemble(problem)
        fields.append((solve_original(system) if original else system.solve()).fields)
    trace = np.zeros(data.trace_size)
    trace[list(data.fixed)] = list(data.fixed.values())
    return _state(data, fields[0], fields[1], trace, 0.0)


def advance(
    data: NewmarkOperators,
    previous: ElastodynamicSolution,
    source: Any = 0.0,
    *,
    original: bool = False,
) -> ElastodynamicSolution:
    """Propagate every local substep then impose the declared global endpoint forms.

    Source callbacks are evaluated at every original local endpoint. The same
    endpoint loads replay the physical Newmark equations with the computed
    slab traction, independently checking the response-based reconstruction.
    """
    macro_index = int(round(previous.time / data.time_step))
    macro_time = macro_index * data.time_step
    free_u, free_v, histories = [], [], []
    for index, local in enumerate(data.locals):
        u, v = previous.displacement[index], previous.velocity[index]
        duration = data.time_step / data.substeps[index]
        old = local.load_at_time(source, macro_time)
        loads = [old]
        for step in range(data.substeps[index]):
            new = local.load_at_time(source, macro_time + (step + 1) * duration)
            u, v = newmark_step(
                local.mass,
                local.stiffness,
                data.step_factors[index],
                data.mass_factors[index],
                duration,
                u,
                v,
                old,
                new,
            )
            loads.append(new)
            old = new
        free_u.append(u)
        free_v.append(v)
        histories.append(loads)
    items = tuple(zip(data.locals, free_u, data.lifts, strict=True))
    if data.endpoint_system is None or original:
        problem = MultiscaleProblem(
            endpoint_global(data.boundary),
            endpoint_equations,
            items,
            data.trace_size,
            (0,) * len(items),
            fixed=data.fixed,
        )
        system = assemble(problem)
        solution = solve_original(system) if original else system.solve()
    else:
        solution = data.endpoint_system.solve(
            tuple(free_u),
            global_load=-data.boundary,
            fixed=data.fixed,
            factorization=data.endpoint_factor,
        )
    velocity = tuple(
        values - lift @ solution.trace[local.trace_dofs]
        for local, lift, values in zip(data.locals, data.velocity_lifts, free_v, strict=True)
    )
    for index, local in enumerate(data.locals):
        u, v = previous.displacement[index], previous.velocity[index]
        force = local.coupling @ solution.trace[local.trace_dofs]
        for old, new in zip(histories[index][:-1], histories[index][1:], strict=True):
            u, v = newmark_step(
                local.mass,
                local.stiffness,
                data.step_factors[index],
                data.mass_factors[index],
                data.time_step / data.substeps[index],
                u,
                v,
                old - force,
                new - force,
            )
        for replay, recovered, free_values, lift in (
            (u, solution.fields[index], free_u[index], data.lifts[index]),
            (v, velocity[index], free_v[index], data.velocity_lifts[index]),
        ):
            scale = max(
                float(np.linalg.norm(free_values)),
                float(np.linalg.norm(lift @ solution.trace[local.trace_dofs])),
                np.finfo(float).tiny,
            )
            if np.linalg.norm(replay - recovered) > 1e-10 * scale:
                raise LinearSolveError(
                    "endpoint response disagrees with original local Newmark histories"
                )
    return _state(
        data, solution.fields, velocity, solution.trace, (macro_index + 1) * data.time_step
    )
