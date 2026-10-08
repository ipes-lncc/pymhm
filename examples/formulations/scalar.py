"""User-written skew transport and backward-Euler equations on scalar Pk leaves.

The common FEM operators own basis tabulation and integration. These application
functions declare mathematical blocks; they never invoke a physical solver.
The retained constant is a coarse coordinate for coercive operators and an
explicit kernel for pure diffusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from scipy import sparse

from pymhm import Equation, LocalEquations, MultiscaleProblem
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import nodal_space, scalar_operators, trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.scalar import strong_boundary_dofs
from pymhm.materials.evaluation import scalar_values
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.solutions import ScalarSolution


@dataclass(frozen=True)
class ScalarDiscretization:
    """Geometry, material and polynomial spaces of a user-defined scalar application."""

    mesh: TriangleMesh
    skeleton: SkeletonSpace
    diffusion: Any = 1.0
    degree: int = 1
    refinement: int = 4
    order: int = 6

    def local_mesh(self, cell: int) -> TriangleMesh:
        """Return the declared conforming local partition of this macrocell."""
        return self.mesh.submesh(cell, self.refinement)


def transport_equations(
    cell: int,
    *,
    data: ScalarDiscretization,
    velocity: Any,
    reaction: Any,
    source: Any,
    pure_diffusion: bool = False,
) -> LocalEquations:
    """Write skew-advection Galerkin A and the two independently signed trace pairings.

    Velocity is constant and divergence-free here. The multiplier represents
    ``(-K grad(u)+velocity*u/2).n``. Residual stabilization and variable-velocity
    divergence terms must be declared separately by the application.
    """
    fine = data.local_mesh(cell)
    a, mass, load = scalar_operators(
        fine,
        data.degree,
        diffusion=data.diffusion,
        reaction=reaction,
        advection=velocity,
        skew_advection=True,
        source=source,
        order=data.order,
    )
    b = trace_coupling(data.mesh, cell, fine, data.skeleton, data.degree)
    constant = np.ones((a.shape[0], 1))
    return LocalEquations(
        a,
        load,
        b,
        -b.T,
        data.skeleton.cell_dofs(cell),
        kernel=constant if pure_diffusion else None,
        coarse_basis=None if pure_diffusion else constant,
        moments=mass @ constant,
        metadata=fine,
        field_data=(nodal_field("scalar", fine, data.degree),),
    )


def transport_problem(
    data: ScalarDiscretization,
    *,
    velocity: Any = (0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    pure_diffusion: bool = False,
) -> MultiscaleProblem[int]:
    """Declare the global pressure-trace pairing for full weak Dirichlet data."""
    boundary, fixed = boundary_data(
        data.skeleton, dirichlet, order=max(data.order, data.degree + 2)
    )
    return MultiscaleProblem(
        Equation(0, -np.r_[boundary, np.zeros(len(data.mesh.cells))]),
        partial(
            transport_equations,
            data=data,
            velocity=velocity,
            reaction=reaction,
            source=source,
            pure_diffusion=pure_diffusion,
        ),
        range(len(data.mesh.cells)),
        data.skeleton.size,
        (1,) * len(data.mesh.cells),
        fixed=fixed,
    )


def initial_coefficients(data: ScalarDiscretization, initial: Any) -> tuple[Any, ...]:
    """Interpolate initial values in the exact executed continuous Pk nodal spaces."""
    return tuple(
        scalar_values(initial, nodal_space(data.local_mesh(cell), data.degree)[1])
        for cell in range(len(data.mesh.cells))
    )


def heat_equations(
    cell: int,
    *,
    data: ScalarDiscretization,
    old: tuple[Any, ...],
    duration: float,
    source: Any,
) -> LocalEquations:
    """Write ``(K grad u,grad v)+(u,v)/dt=(f,v)+(old,v)/dt`` and its balance."""
    fine = data.local_mesh(cell)
    stiffness, mass, load = scalar_operators(
        fine, data.degree, diffusion=data.diffusion, source=source, order=data.order
    )
    b = trace_coupling(data.mesh, cell, fine, data.skeleton, data.degree)
    constant = np.ones((stiffness.shape[0], 1))
    return LocalEquations(
        stiffness + mass / duration,
        load + mass @ old[cell] / duration,
        b,
        -b.T,
        data.skeleton.cell_dofs(cell),
        coarse_basis=constant,
        moments=mass @ constant,
        metadata=fine,
        field_data=(nodal_field("temperature", fine, data.degree),),
    )


def heat_problem(
    data: ScalarDiscretization,
    old: tuple[Any, ...],
    duration: float,
    *,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
) -> MultiscaleProblem[int]:
    """Declare one positive-duration backward-Euler step with physical source units."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    boundary, _ = boundary_data(data.skeleton, dirichlet, order=max(data.order, data.degree + 2))
    return MultiscaleProblem(
        Equation(0, -np.r_[boundary, np.zeros(len(data.mesh.cells))]),
        partial(heat_equations, data=data, old=old, duration=duration, source=source),
        range(len(data.mesh.cells)),
        data.skeleton.size,
        (1,) * len(data.mesh.cells),
    )


def recover_scalar(
    data: ScalarDiscretization,
    result: HybridSolution,
    local_meshes: tuple[TriangleMesh | tuple[Any, ...], ...],
) -> ScalarSolution:
    """Interpret scalar coefficients using the actual meshes retained by providers.

    A provider may store its fine mesh alone or as the first entry of a tuple
    carrying additional integrated moments. Recovery uses that executed mesh;
    it does not construct another mesh or basis from the refinement parameter.
    """
    meshes = tuple(record[0] if isinstance(record, tuple) else record for record in local_meshes)
    return ScalarSolution(data.skeleton, meshes, result.fields, result, data.degree)


def strong_diffusion_equations(
    cell: int,
    *,
    data: ScalarDiscretization,
    source: Any,
    dirichlet: Any,
    essential: tuple[int, ...],
) -> LocalEquations:
    """Declare diffusion and exact nodal boundary constraints as a local mixed form.

    The auxiliary field enforces E.T*u=g. Boundary skeleton columns vanish on
    essential faces; cells without prescribed nodes retain their constant mode.
    """
    fine = data.local_mesh(cell)
    a, mass, load = scalar_operators(
        fine, data.degree, diffusion=data.diffusion, source=source, order=data.order
    )
    b = trace_coupling(data.mesh, cell, fine, data.skeleton, data.degree)
    faces = tuple(face for face in essential if face in data.mesh.cell_faces[cell])
    ids, values = strong_boundary_dofs(data.mesh, fine, faces, data.degree, dirichlet)
    size = len(load)
    constant = np.ones((size, 1))
    if len(ids):
        e = sparse.csc_matrix(
            (np.ones(len(ids)), (ids, np.arange(len(ids)))), shape=(size, len(ids))
        )
        a = sparse.bmat([[a, e], [e.T, None]], format="csc")
        load = np.r_[load, values]
        b = np.vstack((b, np.zeros((len(ids), b.shape[1]))))
        global_ids = data.skeleton.cell_dofs(cell)
        for face in faces:
            b[:, np.isin(global_ids, data.skeleton.dofs(face))] = 0
        kernel = np.empty((len(load), 0))
        moments = kernel.copy()
    else:
        kernel = constant
        moments = mass @ constant
    return LocalEquations(
        a,
        load,
        b,
        -b.T,
        data.skeleton.cell_dofs(cell),
        kernel=kernel,
        moments=moments,
        metadata=(fine, size),
        field_data=(
            nodal_field(
                "scalar",
                fine,
                data.degree,
                reconstruction=sparse.eye(len(load), format="csr")[:size],
            ),
        ),
    )


def strong_diffusion_problem(
    data: ScalarDiscretization,
    *,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    diffusive_flux: dict[int, Any] | None = None,
) -> MultiscaleProblem[int]:
    """Declare mixed diffusion boundary data for the adaptive face-indicator example.

    Advection and reaction vanish, so the physical diffusive outflow coincides
    with the skeletal multiplier. Pure Neumann data needs a caller-declared mean.
    """
    natural = {} if diffusive_flux is None else diffusive_flux
    essential = tuple(int(f) for f in data.mesh.boundary_faces if f not in natural)
    boundary, fixed = boundary_data(data.skeleton, dirichlet, natural, order=data.order)
    for face in essential:
        boundary[data.skeleton.dofs(face)] = 0
        fixed.update(dict.fromkeys(data.skeleton.dofs(face), 0.0))
    sizes = []
    for cell in range(len(data.mesh.cells)):
        faces = tuple(face for face in essential if face in data.mesh.cell_faces[cell])
        ids, _ = strong_boundary_dofs(
            data.mesh, data.local_mesh(cell), faces, data.degree, dirichlet
        )
        sizes.append(0 if len(ids) else 1)
    return MultiscaleProblem(
        Equation(0, -np.r_[boundary, np.zeros(sum(sizes))]),
        partial(
            strong_diffusion_equations,
            data=data,
            source=source,
            dirichlet=dirichlet,
            essential=essential,
        ),
        range(len(data.mesh.cells)),
        data.skeleton.size,
        tuple(sizes),
        fixed=fixed,
    )


def recover_strong_scalar(
    data: ScalarDiscretization,
    system: MultiscaleSystem,
    result: HybridSolution,
    *,
    natural_faces: tuple[int, ...] = (),
) -> ScalarSolution:
    """Extract the physical nodal field while retaining auxiliary coefficients in result."""
    return ScalarSolution(
        data.skeleton,
        tuple(record[0] for record in system.local_metadata),
        tuple(
            field[: record[1]]
            for field, record in zip(result.fields, system.local_metadata, strict=True)
        ),
        result,
        data.degree,
        True,
        natural_faces,
    )
