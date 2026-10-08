"""Primal Pk and locally H(div)-conforming RT0/P0 Darcy MHM methods."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.darcy._mixed import normal_flux_blocks
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.refinement import refine_hybrid
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.loads import point_load_vector, split_point_sources
from pymhm.fem.quadrature.orders import nodal_quadrature_order
from pymhm.fem.scalar.operators import (
    boundary_data,
    face_integration,
    rt0_operators,
)
from pymhm.fem.scalar.triangle import scalar_operators, tabulate, trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.materials.evaluation import tensor_values
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import DarcySolution as DarcySolution


@dataclass(frozen=True)
class _DarcyLocalFactory:
    """Assemble one scalar Neumann problem in nodal pressure or RT0 coordinates."""

    mesh: TriangleMesh
    skeleton: SkeletonSpace
    permeability: Any
    source: Any
    point_parts: tuple[FloatArray, ...]
    local_refinement: int
    supplied_meshes: tuple[TriangleMesh, ...] | None
    degree: int
    formulation: str
    quadrature_order: int
    element_backend: Literal["portable", "basix"] = "basix"

    def __call__(self, cell: int) -> LocalAssembly:
        """Use a dimensionless local average and retain physical integral weights for gauges.

        Dividing both auxiliary constraint blocks by the cell volume changes
        only the Lagrange-multiplier coordinate. The local physical responses
        and the globally prescribed pressure integral retain their conventions.
        """
        mesh, skeleton = self.mesh, self.skeleton
        permeability, source, point_parts = self.permeability, self.source, self.point_parts
        local_refinement, supplied_meshes = self.local_refinement, self.supplied_meshes
        degree, formulation, quadrature_order = self.degree, self.formulation, self.quadrature_order
        fine = (
            mesh.submesh(cell, local_refinement)
            if supplied_meshes is None
            else supplied_meshes[cell]
        )
        coupling, flux_map = face_integration(mesh, cell, fine, skeleton)
        if formulation == "primal":
            coupling = trace_coupling(mesh, cell, fine, skeleton, degree)
            matrix, mass, load = scalar_operators(
                fine,
                degree,
                diffusion=permeability,
                source=source,
                order=quadrature_order,
                element_backend=self.element_backend,
            )
            if len(point_parts[cell]):
                load += point_load_vector(fine, degree, point_parts[cell])
            kernel = np.ones((matrix.shape[0], 1))
            physical_mean = mass @ kernel
            constraints = physical_mean / (kernel.T @ physical_mean).item()
            problem = LocalProblem(
                matrix, coupling, load, skeleton.cell_dofs(cell), kernel, constraints
            )
        else:
            for face in mesh.cell_faces[cell]:
                space = skeleton.faces[face]
                if any(space.degrees) or not np.allclose(
                    np.array(space.breaks) * local_refinement,
                    np.round(np.array(space.breaks) * local_refinement),
                    atol=1e-12,
                    rtol=0,
                ):
                    raise ValueError("RT0 needs degree-zero trace segments aligned with fine edges")
            mass, divergence, force = rt0_operators(
                fine, permeability, source, order=quadrature_order
            )
            if len(point_parts[cell]):
                force += np.array(
                    [part[:, 2].sum() for part in split_point_sources(fine, point_parts[cell])]
                )
            nq, npres, nb = len(fine.faces), len(fine.cells), len(fine.boundary_faces)
            matrix, coupling_mixed, load = normal_flux_blocks(
                mass, divergence, force, fine.boundary_faces, flux_map
            )
            kernel = np.r_[np.zeros(nq), np.ones(npres + nb)][:, None]
            physical_mean = np.r_[np.zeros(nq), fine.areas, np.zeros(nb)][:, None]
            constraints = physical_mean / (kernel.T @ physical_mean).item()
            problem = LocalProblem(
                matrix, coupling_mixed, load, skeleton.cell_dofs(cell), kernel, constraints
            )
        return LocalAssembly(problem, (fine, physical_mean[:, 0]))


def darcy_local_provider(
    mesh: TriangleMesh,
    *,
    skeleton: SkeletonSpace | None = None,
    permeability: Any = 1.0,
    source: Any = 0.0,
    formulation: Literal["primal", "mixed"] = "primal",
    degree: int = 1,
    local_refinement: int = 4,
    quadrature_order: int = 4,
    element_backend: Literal["portable", "basix"] = "basix",
) -> Callable[[int], LocalAssembly]:
    """Create a portable callable for primal Pk or flux-prescribing RT0 cells.

    The provider reuses the original Darcy local construction on one refined
    triangle per macrocell. ``primal`` represents nodal pressure and its
    physical constant mode; ``mixed`` represents RT0 flux, cellwise constant
    pressure and auxiliary boundary-pressure multipliers with their joint
    pressure mode. It enforces prescribed skeletal normal flux through the
    augmented local equations. This is not a pressure-trace hybridization.

    ``skeleton`` must belong to ``mesh`` and have one component. Its oriented
    global coefficients represent Darcy normal flux. Primal weak pressure
    boundary moments enter ``GlobalForm.boundary_load`` unchanged; mixed RT0
    uses their negative because its augmented coupling is ``-flux_map``.
    RT0 requires piecewise constant traces aligned with refined boundary edges
    and retains the legacy ``degree=1`` setting. Higher RT/BDM and restricted
    H(div) families have their own discretization contracts and public solvers.

    Return metadata is ``(local_mesh, physical_pressure_mean_weights)`` in the
    original local coefficient order. These weights can form physical global
    gauges; auxiliary multipliers do not contribute to the pressure mean.
    Basix tabulates primal Pk and mixed RT0 in their declared nodal/moment
    coordinates. The portable spelling selects the same implementation.
    All native resources remain invocation-local; callbacks must be picklable
    for process execution.
    """
    refinement = positive_int(local_refinement, "local_refinement")
    degree = positive_int(degree, "degree")
    order = _assembly_quadrature_order(degree, quadrature_order)
    if formulation not in {"primal", "mixed"}:
        raise ValueError("formulation must be primal or mixed")
    if formulation == "mixed" and degree != 1:
        raise ValueError("RT0 uses degree=1; primal Pk has its own degree")
    if element_backend not in {"portable", "basix"}:
        raise ValueError("element_backend must be portable or basix")
    skeleton = SkeletonSpace(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("Darcy requires a scalar skeleton on the supplied mesh")
    return _DarcyLocalFactory(
        mesh,
        skeleton,
        permeability,
        source,
        tuple(np.empty((0, 3)) for _ in mesh.cells),
        refinement,
        None,
        degree,
        formulation,
        order,
        element_backend,
    )


def solve_darcy(
    mesh: TriangleMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    point_sources: Any = (),
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 4,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
    quadrature_order: int = 4,
    formulation: str = "primal",
    degree: int = 1,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    local_refinement_precision: Literal["double", "extended"] = "double",
    hybrid_refinement_steps: int = 0,
    hybrid_refinement_min_steps: int = 0,
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    parallel_assembly: bool = False,
) -> DarcySolution:
    """Solve ``div(q)=f``, ``q=-K grad(p)`` with mixed boundary conditions.

    ``dirichlet`` prescribes pressure on all faces not listed in ``neumann``;
    Neumann entries prescribe the outward physical flux. Coefficients accept
    scalars, constant SPD tensors, or callables taking points of shape (n,2).
    Pure Neumann data require compatibility and use a global mean-pressure gauge.
    ``point_sources`` is an (n,3) array of x, y and signed strengths. These
    discrete Dirac loads are shared geometrically across macro interfaces;
    primal Pk evaluates the point functional exactly, while mixed P0 assigns
    its integrated strength to incident fine cells by angle. Such singular
    problems do not have a finite H1 pressure energy in two dimensions.
    ``degree`` selects the primal local polynomial degree. RT0 supports
    piecewise constant skeletons aligned with fine boundary faces.
    ``quadrature_order`` requests Gauss points per Duffy coordinate and boundary
    integration interval, with an effective minimum of ``degree+2``. The executed
    volume order is retained on the solution. Boundary data additionally apply
    the trace degree+2 floor; polynomial trace coupling uses an exact
    degree-dependent rule. Nonpolynomial data require independent order controls.
    ``local_meshes`` optionally supplies one validated conforming triangular
    partition per macrocell for the primal formulation; it takes precedence over
    uniform ``local_refinement``. Material-fitted local meshes do not change the
    macro geometry or the separately supplied skeletal space.
    ``local_refinement_precision="extended"`` retains direct-solver correction
    digits in the local source/trace lifts and reconstructed fields. Factors
    remain in double precision and the original residual tolerance is unchanged;
    platforms without a wider NumPy long-double type reject this explicit mode.
    ``hybrid_refinement_steps`` optionally corrects the original local, weak-trace
    and physical mean equations after condensation, using the same solver and
    local precision settings. Zero preserves the default solve. A positive value
    limits defect corrections and requires relative original-equation residual
    at most 1e-10; failure raises ``LinearSolveError``. Persist the resulting
    local fields: trace and coarse coefficients alone do not replay these
    additional defect-source responses.
    ``hybrid_refinement_min_steps`` optionally requires a minimum number of
    these same corrections, including after residual acceptance. It cannot
    exceed ``hybrid_refinement_steps``. Its default is zero; the requested
    count and a small residual alone do not establish field accuracy.
    ``parallel_assembly=True`` assembles and condenses each local problem in
    the selected worker backend. Spawn workers require picklable material/source
    callbacks. The default assembles in the parent and retains compatibility
    with closures even when condensation uses processes.
    """
    positive_int(local_refinement, "local_refinement")
    positive_int(degree, "degree")
    positive_int(hybrid_refinement_steps, "hybrid_refinement_steps", 0)
    positive_int(hybrid_refinement_min_steps, "hybrid_refinement_min_steps", 0)
    if hybrid_refinement_min_steps > hybrid_refinement_steps:
        raise ValueError("hybrid_refinement_min_steps must not exceed hybrid_refinement_steps")
    if formulation == "mixed" and degree != 1:
        raise ValueError("RT0 uses degree=1; higher-order primal locals use formulation=primal")
    quadrature_order = _assembly_quadrature_order(degree, quadrature_order)
    if formulation not in ("primal", "mixed"):
        raise ValueError("formulation must be 'primal' or 'mixed'")
    skeleton = SkeletonSpace(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("Darcy requires a scalar skeleton on the supplied mesh")
    boundary, fixed = boundary_data(skeleton, dirichlet, neumann, order=quadrature_order)
    point_parts = split_point_sources(mesh, point_sources)
    supplied_meshes = local_meshes
    if supplied_meshes is not None:
        if formulation != "primal" or len(supplied_meshes) != len(mesh.cells):
            raise ValueError("provide one local mesh per macrocell for primal Darcy only")
        from pymhm.meshes.refinement import validate_submesh

        for cell, fine in enumerate(supplied_meshes):
            validate_submesh(mesh, cell, fine)
    factory = _DarcyLocalFactory(
        mesh,
        skeleton,
        permeability,
        source,
        tuple(point_parts),
        local_refinement,
        supplied_meshes,
        degree,
        formulation,
        quadrature_order,
    )
    if parallel_assembly:
        system = HybridSystem.from_local_factory(
            factory,
            range(len(mesh.cells)),
            boundary_load=boundary if formulation == "primal" else -boundary,
            local_solver=local_solver,
            local_refinement_precision=local_refinement_precision,
            backend=backend,
            workers=workers,
        )
        metadata = system.local_metadata
    else:
        assemblies = [factory(cell) for cell in range(len(mesh.cells))]
        metadata = tuple(item.metadata for item in assemblies)
        system = HybridSystem(
            [item.problem for item in assemblies],
            boundary_load=boundary if formulation == "primal" else -boundary,
            local_solver=local_solver,
            local_refinement_precision=local_refinement_precision,
            backend=backend,
            workers=workers,
        )
    computed_meshes = [item[0] for item in metadata]
    mean_weights = [item[1] for item in metadata]
    pure_neumann = neumann is not None and set(neumann) == set(mesh.boundary_faces)
    gauges = (
        [system.mean_constraint(mean_weights, mean_pressure * float(np.sum(mesh.areas)))]
        if pure_neumann
        else None
    )
    hybrid = system.solve(solver=solver, fixed=fixed, constraints=gauges)
    if hybrid_refinement_steps:
        hybrid = refine_hybrid(
            system,
            hybrid,
            boundary_load=boundary if formulation == "primal" else -boundary,
            fixed=fixed,
            moments=[(mean_weights, mean_pressure * float(np.sum(mesh.areas)))]
            if pure_neumann
            else (),
            max_steps=hybrid_refinement_steps,
            min_steps=hybrid_refinement_min_steps,
            solver=solver,
            local_solver=local_solver,
            refinement_precision=local_refinement_precision,
        ).solution
    pressure, fluxes = [], []
    for fine, field in zip(computed_meshes, hybrid.fields, strict=True):
        if formulation == "primal":
            pressure.append(field)
            dofs, _, _, gradients, _ = tabulate(fine, degree, np.full((1, 3), 1 / 3))
            grad = np.einsum("ti,tia->ta", field[dofs], gradients[:, 0])
            tensors = tensor_values(permeability, fine.points[fine.cells].mean(axis=1))
            fluxes.append(-np.einsum("tab,tb->ta", tensors, grad))
        else:
            pressure.append(field[len(fine.faces) : len(fine.faces) + len(fine.cells)])
            fluxes.append(field[: len(fine.faces)])
    return DarcySolution(
        skeleton,
        tuple(computed_meshes),
        tuple(pressure),
        tuple(fluxes),
        hybrid,
        formulation,
        permeability,
        source,
        quadrature_order,
        degree,
        point_parts if any(len(part) for part in point_parts) else (),
    )


_assembly_quadrature_order = nodal_quadrature_order
