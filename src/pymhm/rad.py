"""Scalar reaction-advection-diffusion with Galerkin, SUPG and UNUSUAL locals."""

from functools import partial
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.cut_cells import material_triangle_quadrature
from pymhm.elements import (
    boundary_data,
    scalar_values,
    tensor_values,
    vector_values,
)
from pymhm.hybrid import HybridSystem, LocalAssembly, LocalProblem
from pymhm.lagrange import element_tabulate, trace_coupling
from pymhm.mesh import SkeletonSpace, TriangleMesh, positive_int
from pymhm.reservoir import CartesianCellField
from pymhm.scalar_boundary import diffusive_boundary_matrix, strong_boundary_dofs
from pymhm.transport import ScalarSolution
from pymhm.unusual import UnusualParameters, unusual_scale
from pymhm.vector import _assemble_blocks


def _streamline_scale(fine: TriangleMesh, tensor: Any, beta: Any, strong_reaction: Any) -> Any:
    """Return the common positive residual time scale for spatial and time-step forms."""
    h = np.max(fine.lengths[fine.cell_faces], axis=1)[:, None]
    magnitude = np.linalg.norm(beta, axis=-1)
    diffusivity = np.linalg.eigvalsh(tensor)[..., -1]
    return 1 / np.sqrt(
        (2 * magnitude / h) ** 2 + (4 * diffusivity / h**2) ** 2 + strong_reaction**2
    )


def _rad_local(
    cell: int,
    *,
    mesh: TriangleMesh,
    skeleton: SkeletonSpace,
    degree: int,
    refinement: int,
    diffusion: Any,
    diffusion_divergence: Any,
    velocity: Any,
    velocity_divergence: Any,
    reaction: Any,
    source: Any,
    stabilization: str,
    order: int,
    strong_faces: tuple[int, ...] = (),
    dirichlet: Any = 0.0,
    diffusive_faces: tuple[int, ...] = (),
    coarse_space: Literal["constants", "kernel"] = "constants",
    unusual_parameters: UnusualParameters | None = None,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
) -> LocalAssembly:
    """Assemble the conservative skew form and its complete streamline residual."""
    fine = mesh.submesh(cell, refinement) if local_meshes is None else local_meshes[cell]
    bary, weights, material = material_triangle_quadrature(fine, diffusion, max(order, degree + 2))
    dofs, nodes, basis, gradient, hessian = element_tabulate(fine, degree, bary)
    physical = np.einsum("tqi,tij->tqj", bary, fine.points[fine.cells])
    flat = physical.reshape(-1, 2)
    nt, nq = weights.shape
    tensor = tensor_values(material, flat).reshape(nt, nq, 2, 2)
    beta = vector_values(velocity, flat).reshape(nt, nq, 2)
    normal_pair = hasattr(velocity, "advection_boundary_matrix")
    if normal_pair and stabilization != "galerkin":
        raise ValueError("raw volume/numerical-normal velocity requires Galerkin stabilization")
    div_beta = (
        np.zeros((nt, nq))
        if normal_pair
        else scalar_values(velocity_divergence, flat).reshape(nt, nq)
    )
    c = scalar_values(reaction, flat).reshape(nt, nq)
    effective = c + div_beta / 2
    if np.any(effective < 0):
        raise ValueError("reaction+div(velocity)/2 must be nonnegative")
    force = scalar_values(source, flat).reshape(nt, nq)
    blocks = np.einsum("tq,tqia,tqab,tqjb,t->tij", weights, gradient, tensor, gradient, fine.areas)
    blocks += np.einsum("tq,tqi,tqj,tq,t->tij", weights, basis, basis, effective, fine.areas)
    streamline = np.einsum("tqa,tqia->tqi", beta, gradient)
    advective = np.einsum("tq,tqi,tqj,t->tij", weights, basis, streamline, fine.areas)
    blocks += (
        -advective.swapaxes(1, 2) if normal_pair else (advective - advective.swapaxes(1, 2)) / 2
    )
    element_load = np.einsum("tq,tqi,tq,t->ti", weights, basis, force, fine.areas)
    if stabilization in ("supg", "unusual"):
        div_tensor = vector_values(diffusion_divergence, flat).reshape(nt, nq, 2)
        diffusion_residual = np.einsum("tqab,tqiab->tqi", tensor, hessian) + np.einsum(
            "tqa,tqia->tqi", div_tensor, gradient
        )
        strong = -diffusion_residual + streamline + (c + div_beta)[:, :, None] * basis
        if stabilization == "supg":
            tau = _streamline_scale(fine, tensor, beta, c + div_beta)
            test_residual = streamline
        else:
            if isinstance(diffusion, CartesianCellField) and np.any(tensor != tensor[:, :1]):
                raise ValueError(
                    "UNUSUAL requires material interfaces aligned with local fine cells"
                )
            tau = unusual_scale(
                basis,
                gradient,
                diffusion_residual,
                weights,
                fine.lengths[fine.cell_faces].max(axis=1),
                tensor,
                c,
                fine.points[fine.cells].mean(axis=1),
                unusual_parameters or UnusualParameters(),
            )[:, None]
            test_residual = -strong
        blocks += np.einsum("tq,tqi,tqj,tq,t->tij", weights, test_residual, strong, tau, fine.areas)
        element_load += np.einsum(
            "tq,tqi,tq,tq,t->ti", weights, test_residual, force, tau, fine.areas
        )
    matrix = _assemble_blocks(blocks, dofs, len(nodes))
    if normal_pair:
        matrix += velocity.advection_boundary_matrix(degree, order)
    load = np.bincount(dofs.ravel(), weights=element_load.ravel(), minlength=len(nodes))
    moments = np.zeros(len(nodes))
    np.add.at(moments, dofs, fine.areas[:, None] * np.einsum("tq,tqi->ti", weights, basis))
    constant = np.ones((len(nodes), 1))
    pure_diffusion = not np.any(effective) and not np.any(beta)
    retained = {"kernel": constant} if pure_diffusion else {"coarse_basis": constant}
    constraints = moments[:, None]
    if coarse_space == "kernel" and not pure_diffusion:
        kernel = (
            not np.any(c)
            and not np.any(div_beta)
            and _boundary_tangent(SkeletonSpace(fine), velocity, order)
        )
        retained = {"kernel": constant} if kernel else {}
        if not kernel:
            constraints = np.empty((len(nodes), 0))
    coupling = trace_coupling(mesh, cell, fine, skeleton, degree)
    own_faces = set(mesh.cell_faces[cell])
    natural = tuple(face for face in diffusive_faces if face in own_faces)
    if natural:
        matrix += diffusive_boundary_matrix(mesh, fine, natural, degree, velocity, order)
    essential = tuple(face for face in strong_faces if face in own_faces)
    ids, values = strong_boundary_dofs(mesh, fine, essential, degree, dirichlet)
    if len(ids):
        selection = sparse.csc_matrix(
            (np.ones(len(ids)), (ids, np.arange(len(ids)))), shape=(len(nodes), len(ids))
        )
        matrix = sparse.bmat([[matrix, selection], [selection.T, None]], format="csc")
        load = np.r_[load, values]
        coupling = np.vstack((coupling, np.zeros((len(ids), coupling.shape[1]))))
        global_dofs = skeleton.cell_dofs(cell)
        for face in essential:
            coupling[:, np.isin(global_dofs, skeleton.dofs(face))] = 0
        constraints = np.empty((len(load), 0))
        retained = {"coarse_basis": np.empty((len(load), 0))}
    problem = LocalProblem(
        matrix,
        coupling,
        load,
        skeleton.cell_dofs(cell),
        constraints=constraints,
        **retained,
    )
    return LocalAssembly(
        problem,
        (fine, moments, pure_diffusion, not np.any(c) and not np.any(div_beta), len(nodes), ids),
    )


def solve_rad(
    mesh: TriangleMesh,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    diffusive_flux: dict[int, Any] | None = None,
    dirichlet_enforcement: str = "weak",
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 4,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
    degree: int = 1,
    quadrature_order: int = 6,
    velocity_divergence: Any = None,
    diffusion_divergence: Any = None,
    stabilization: str = "galerkin",
    unusual_parameters: UnusualParameters | None = None,
    mean_value: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    coarse_space: Literal["constants", "kernel"] = "constants",
    global_refinement_precision: Literal["double", "extended"] = "double",
) -> ScalarSolution:
    """Solve ``-div(K grad(u))+div(beta*u)+c*u=f`` with Pk local fields.

    The skeleton variable is ``(-K grad(u)+beta*u/2).n``. ``neumann`` prescribes
    this Robin flux, not the full conservative flux; Dirichlet data are imposed
    on the remaining boundary. The local skew form includes ``c+div(beta)/2``.
    A variable velocity requires its explicitly supplied divergence. Reaction may
    be negative provided ``c+div(beta)/2 >= 0`` at the integration points, as in
    the coercivity assumption of the generalized RAD formulation. This sampled
    check is not a certificate of pointwise positivity between quadrature nodes.
    ``diffusive_flux`` instead prescribes ``-K grad(u).n`` on disjoint exterior
    faces; its half-advection boundary mass is included in the local operator.
    ``dirichlet_enforcement='strong'`` fixes local nodal traces exactly and removes
    those exterior skeleton unknowns. The default imposes Dirichlet data weakly.

    ``stabilization='supg'`` adds ``(tau*(L(u)-f), beta.grad(v))`` with the
    complete conservative strong operator, including ``div(K)`` and ``div(beta)``.
    Its positive algebraic time scale is
    ``[(2|beta|/h)^2+(4 lambda_max(K)/h^2)^2+(c+div(beta))^2]^-1/2``.
    Variable diffusion in SUPG requires ``diffusion_divergence``. This is a
    consistent SUPG option; no discrete maximum-principle claim is made.

    ``stabilization='unusual'`` implements the reaction--diffusion method of
    Santiago, Valentin and Martins (2025), Eqs. (14)--(15): subtract
    ``(tau*L(u), L(v))`` and ``(tau*f, L(v))``, where
    ``L(u)=c*u-div(K*grad(u))``. Advection must vanish. Coefficient callbacks
    require the bounds in ``unusual_parameters``; variable diffusion also
    requires its divergence. Interfaces must be resolved by fine cells for
    this strong-residual method. The inverse parameter is computed from the
    local polynomial operator unless explicitly supplied and checked.
    ``local_meshes`` optionally supplies one validated conforming fine mesh per
    macrocell, for example a mesh fitted to material interfaces. It changes the
    local approximation space explicitly; quadrature cuts alone do not do so.
    Reaction changes the constant-test equation through its stabilization
    term: raw physical flux balance is not asserted from the skeleton alone.

    Local constants are retained even for near-zero reaction or transport.
    Set ``coarse_space='kernel'`` for the selective generalized MHM formulation:
    only zero-reaction, divergence-free, locally boundary-tangent constant modes
    remain. Invertible locals then have no coarse amplitude. This changes the
    condensed coordinates, not the reconstructed discrete field. The default
    constant retention avoids losing nearly singular modes at extreme scales.
    With selective retention, physical diffusive-flux faces require tangent
    advection; use the general retained-constant form for other diffusive data.
    A pure-Neumann diffusion problem imposes the prescribed global mean. This
    also applies to divergence-free advection tangent to the external boundary
    when reaction vanishes. In that case the skeleton must represent beta.n/2
    on internal faces to preserve the global constant mode; the original hybrid
    equations are checked after imposing the mean.
    ``global_refinement_precision`` forwards the explicit correction-storage
    mode of :meth:`pymhm.hybrid.HybridSystem.solve`; local assembly is unchanged.
    """
    positive_int(local_refinement, "local_refinement")
    positive_int(degree, "degree")
    positive_int(quadrature_order, "quadrature_order")
    if stabilization not in ("galerkin", "supg", "unusual"):
        raise ValueError("stabilization must be galerkin, supg or unusual")
    if unusual_parameters is not None and (
        stabilization != "unusual" or not isinstance(unusual_parameters, UnusualParameters)
    ):
        raise ValueError("unusual_parameters requires UNUSUAL stabilization and UnusualParameters")
    if stabilization == "unusual":
        if (
            callable(velocity)
            or np.any(velocity)
            or callable(velocity_divergence)
            or (velocity_divergence is not None and np.any(velocity_divergence))
        ):
            raise ValueError("published scalar UNUSUAL requires zero advection and divergence")
        parameters = unusual_parameters or UnusualParameters()
        if (
            callable(diffusion)
            and not isinstance(diffusion, CartesianCellField)
            and parameters.diffusion_lower is None
        ):
            raise ValueError("UNUSUAL variable diffusion requires diffusion_lower")
        if callable(reaction) and parameters.reaction_upper is None:
            raise ValueError("UNUSUAL variable reaction requires reaction_upper")
    if coarse_space not in ("constants", "kernel"):
        raise ValueError("coarse_space must be constants or kernel")
    if dirichlet_enforcement not in ("weak", "strong"):
        raise ValueError("dirichlet_enforcement must be weak or strong")
    natural = {} if neumann is None else dict(neumann)
    diffusive_flux = {} if diffusive_flux is None else dict(diffusive_flux)
    if set(natural) & set(diffusive_flux):
        raise ValueError("Robin and diffusive flux faces must be disjoint")
    natural.update(diffusive_flux)
    if callable(velocity) and velocity_divergence is None:
        raise ValueError("variable velocity requires velocity_divergence")
    if (
        stabilization in ("supg", "unusual")
        and callable(diffusion)
        and not isinstance(diffusion, CartesianCellField)
        and diffusion_divergence is None
    ):
        raise ValueError(
            "residual stabilization with variable diffusion requires diffusion_divergence"
        )
    velocity_divergence = 0.0 if velocity_divergence is None else velocity_divergence
    diffusion_divergence = (0.0, 0.0) if diffusion_divergence is None else diffusion_divergence
    if not callable(velocity) and np.any(scalar_values(velocity_divergence, mesh.points)):
        raise ValueError("constant velocity must have zero divergence")
    if not callable(diffusion) and np.any(vector_values(diffusion_divergence, mesh.points)):
        raise ValueError("constant diffusion must have zero divergence")
    if not np.isfinite(mean_value):
        raise ValueError("mean_value must be finite")
    skeleton = SkeletonSpace(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("transport requires a scalar skeleton on this mesh")
    if local_meshes is not None:
        from pymhm.refinement import validate_submesh

        if len(local_meshes) != len(mesh.cells):
            raise ValueError("one local mesh is required for each macrocell")
        for cell, fine in enumerate(local_meshes):
            validate_submesh(mesh, cell, fine)
    boundary, fixed = boundary_data(
        skeleton, dirichlet, natural, order=max(quadrature_order, degree + 2)
    )
    if coarse_space == "kernel" and not _boundary_tangent(
        skeleton, velocity, quadrature_order, tuple(diffusive_flux)
    ):
        raise ValueError(
            "selective coarse space requires tangent advection on diffusive-flux faces"
        )
    strong_faces: tuple[int, ...] = ()
    if dirichlet_enforcement == "strong":
        strong_faces = tuple(int(face) for face in mesh.boundary_faces if face not in natural)
        for face in strong_faces:
            boundary[skeleton.dofs(face)] = 0
            fixed.update(dict.fromkeys(skeleton.dofs(face), 0.0))
    factory = partial(
        _rad_local,
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        refinement=local_refinement,
        diffusion=diffusion,
        diffusion_divergence=diffusion_divergence,
        velocity=velocity,
        velocity_divergence=velocity_divergence,
        reaction=reaction,
        source=source,
        stabilization=stabilization,
        order=quadrature_order,
        strong_faces=strong_faces,
        dirichlet=dirichlet,
        diffusive_faces=tuple(diffusive_flux),
        coarse_space=coarse_space,
        unusual_parameters=unusual_parameters,
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
    pure_neumann = set(natural) == set(mesh.boundary_faces)
    gauges = []
    if (
        pure_neumann
        and all(data[3] for data in system.local_metadata)
        and _boundary_tangent(skeleton, velocity, quadrature_order)
    ):
        gauges.append(
            system.mean_constraint(
                [data[1] for data in system.local_metadata], mean_value * sum(mesh.areas)
            )
        )
    elif mean_value != 0:
        raise ValueError(
            "mean_value is only a gauge for zero reaction, divergence-free tangential "
            "velocity and pure Robin data"
        )
    result = system.solve(
        solver=solver,
        fixed=fixed,
        constraints=gauges,
        refinement_precision=global_refinement_precision,
    )
    return ScalarSolution(
        skeleton,
        tuple(data[0] for data in system.local_metadata),
        tuple(
            field[: data[4]]
            for field, data in zip(result.fields, system.local_metadata, strict=True)
        ),
        result,
        degree,
        dirichlet_enforcement == "strong",
        tuple(natural),
    )


def _boundary_tangent(
    skeleton: SkeletonSpace, velocity: Any, order: int, faces: tuple[int, ...] | None = None
) -> bool:
    """Check zero external normal advection with componentwise cancellation scales."""
    for face in skeleton.mesh.boundary_faces if faces is None else faces:
        parameter, _ = skeleton.faces[face].quadrature(max(order, 8))
        start, end = skeleton.mesh.points[skeleton.mesh.faces[face]]
        points = start + parameter[:, None] * (end - start)
        beta = vector_values(velocity, points)
        normal = skeleton.mesh.normals[face]
        scale = float(np.max(np.abs(beta) @ np.abs(normal)))
        if np.any(np.abs(beta @ normal) > 64 * np.finfo(float).eps * scale):
            return False
    return True
