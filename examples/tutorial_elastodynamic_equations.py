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
from typing import Any

import numpy as np
from scipy import sparse

from pymhm._legacy.models.waves.elastodynamics import ElastodynamicLocal, ElastodynamicSolution
from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.multiscale import MultiscaleProblem, assemble
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import nodal_space, trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.vector.curl import physical_basis, physical_points, quadrature
from pymhm.fem.vector.operators import _assemble_blocks
from pymhm.linalg.dynamics import newmark_step
from pymhm.linalg.linear import LinearFactorization, LinearSolveError, factorize
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class NewmarkOperators:
    """Executed coefficient bases and owner-held factors for one macro time step."""

    skeleton: SkeletonSpace
    locals: tuple[ElastodynamicLocal, ...]
    mass_factors: tuple[LinearFactorization, ...]
    step_factors: tuple[LinearFactorization, ...]
    lifts: tuple[Any, ...]
    velocity_lifts: tuple[Any, ...]
    time_step: float
    substeps: tuple[int, ...]
    boundary: Any
    fixed: dict[int, float]


def spatial_forms(
    mesh: TriangleMesh,
    skeleton: SkeletonSpace,
    cell: int,
    degree: int,
    refinement: int,
    order: int,
    density: float,
    lame_lambda: float,
    lame_mu: float,
) -> ElastodynamicLocal:
    """Declare rho mass and lambda div/div plus twice mu symmetric-gradient energy.

    The tutorial uses positive constant density, positive mu and nonnegative
    lambda in planar strain. Nodes and vector components are interleaved; the
    canonical signed traction map is the common scalar trace map times I2.
    This displacement energy has no incompressible-limit guarantee.
    """
    fine = mesh.submesh(cell, refinement)
    bary, weights, _ = quadrature(fine, 1.0, order)
    basis, gradients = physical_basis(fine, degree, bary)
    dofs, nodes = nodal_space(fine, degree)
    width = 2 * basis.shape[-1]
    vector_dofs = (2 * dofs[:, :, None] + np.arange(2)).reshape(len(fine.cells), width)
    strain = np.zeros((*weights.shape, 3, width))
    strain[:, :, 0, 0::2] = gradients[..., 0]
    strain[:, :, 1, 1::2] = gradients[..., 1]
    strain[:, :, 2, 0::2] = gradients[..., 1] / np.sqrt(2)
    strain[:, :, 2, 1::2] = gradients[..., 0] / np.sqrt(2)
    material = np.array(
        [
            [lame_lambda + 2 * lame_mu, lame_lambda, 0],
            [lame_lambda, lame_lambda + 2 * lame_mu, 0],
            [0, 0, 2 * lame_mu],
        ]
    )
    stiffness = _assemble_blocks(
        np.einsum("tq,tqai,ab,tqbj->tij", weights, strain, material, strain),
        vector_dofs,
        2 * len(nodes),
    )
    rho = np.full(weights.shape, density)
    scalar_mass = _assemble_blocks(
        np.einsum("tq,tqi,tqj->tij", weights * rho, basis, basis), dofs, len(nodes)
    )
    mass = sparse.kron(scalar_mass, sparse.eye(2), format="csc")
    coupling = np.asarray(
        np.kron(trace_coupling(mesh, cell, fine, skeleton, degree), np.eye(2)), dtype=np.float64
    )
    return ElastodynamicLocal(
        fine,
        degree,
        mass,
        stiffness,
        coupling,
        skeleton.cell_dofs(cell),
        dofs,
        nodes,
        basis,
        physical_points(fine, bary),
        weights,
        rho,
        None,
        lame_lambda,
        lame_mu,
        order,
    )


@contextmanager
def prepare(
    mesh: TriangleMesh,
    *,
    time_step: float,
    degree: int = 2,
    local_refinement: int = 2,
    local_substeps: Any = 1,
    quadrature_order: int = 8,
    density: float = 1.0,
    lame_lambda: float = 1.0,
    lame_mu: float = 1.0,
    dirichlet: Any = 0.0,
    traction: dict[int, Any] | None = None,
) -> Iterator[NewmarkOperators]:
    """Own reusable factors while preparing the declared planar coefficient response.

    Each macrocell can use a distinct positive substep count, ending at the
    same macro time. Loads remain physical force densities. The multiplier is
    negative physical traction, constant over a macro time slab; arbitrary
    subcycling does not automatically inherit a global energy identity.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("this tutorial declares planar triangular forms")
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be finite and positive")
    if (
        not all(np.isfinite([density, lame_lambda, lame_mu]))
        or min(density, lame_mu) <= 0
        or lame_lambda < 0
    ):
        raise ValueError("require positive density/mu and nonnegative lambda")
    raw_counts = np.broadcast_to(np.asarray(local_substeps), (len(mesh.cells),))
    if any(
        isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1
        for value in raw_counts
    ):
        raise ValueError("local_substeps must contain positive integers")
    counts = tuple(int(value) for value in raw_counts)
    order = max(quadrature_order, degree + 2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), components=2)
    traction = {} if traction is None else traction
    if not set(traction).issubset(set(mesh.boundary_faces)):
        raise ValueError("traction keys must identify exterior faces")
    target, physical_tractions = boundary_data(skeleton, dirichlet, traction, order=order)
    fixed = {int(index): -float(value) for index, value in physical_tractions.items()}
    locals_ = tuple(
        spatial_forms(
            mesh, skeleton, cell, degree, local_refinement, order, density, lame_lambda, lame_mu
        )
        for cell in range(len(mesh.cells))
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
    )


def _state(
    data: NewmarkOperators, u: tuple[Any, ...], v: tuple[Any, ...], trace: Any, time: float
) -> ElastodynamicSolution:
    """Store fields and check the original free-face displacement constraint."""
    moments = np.zeros(data.skeleton.size)
    for local, values in zip(data.locals, u, strict=True):
        np.add.at(moments, local.trace_dofs, local.coupling.T @ values)
    free = np.setdiff1d(np.arange(data.skeleton.size), list(data.fixed))
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
    data: NewmarkOperators, displacement: Any = 0.0, velocity: Any = 0.0
) -> ElastodynamicSolution:
    """Solve two explicit constrained physical mass projections through the core."""
    fields = []
    for datum, target in ((displacement, data.boundary), (velocity, np.zeros(data.skeleton.size))):
        items = tuple((local, datum) for local in data.locals)
        problem = MultiscaleProblem(
            endpoint_global(target),
            projection_equations,
            items,
            data.skeleton.size,
            (0,) * len(items),
            fixed={index: 0.0 for index in data.fixed},
        )
        fields.append(assemble(problem).solve().fields)
    trace = np.zeros(data.skeleton.size)
    trace[list(data.fixed)] = list(data.fixed.values())
    return _state(data, fields[0], fields[1], trace, 0.0)


def advance(
    data: NewmarkOperators, previous: ElastodynamicSolution, source: Any = 0.0
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
    problem = MultiscaleProblem(
        endpoint_global(data.boundary),
        endpoint_equations,
        items,
        data.skeleton.size,
        (0,) * len(items),
        fixed=data.fixed,
    )
    solution = assemble(problem).solve()
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
