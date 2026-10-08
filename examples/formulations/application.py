"""Executable example compositions of publicly declared mathematical equations.

The form definitions remain in adjacent importable modules. These controllers
select their explicit application data, assemble through the generic API, apply
physical integral constraints and interpret coefficients with public field
records. They never dispatch to PyMHM's prepared physical method solvers.
"""

from typing import Any

from pymhm import ExecutionConfig, SolverConfig, assemble
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.tetrahedron import TetraMesh

from .cartesian_darcy import define_cartesian_darcy, recover_cartesian_darcy
from .darcy import define_darcy, pressure_constraints, recover_darcy
from .mapped_darcy import define_mapped_darcy, recover_mapped_darcy
from .mixed_darcy import define_bdm_darcy, define_rt_darcy, recover_bdm_darcy, recover_rt_darcy
from .mixed_darcy_3d import define_hdiv_darcy, recover_hdiv_darcy
from .moments import define_moment_diffusion, recover_moment_diffusion
from .moments import pressure_constraints as moment_constraints
from .moments_3d import define_moment_diffusion_3d, recover_moment_diffusion_3d
from .penalty import add_jump_form, recover_penalty
from .primal_elasticity import (
    define_primal_elasticity,
    primal_constraints,
    recover_primal_elasticity,
)
from .robin import (
    define_robin,
    define_robin_3d,
    recover_robin,
    recover_robin_3d,
    robin_gauge,
    robin_gauge_3d,
)
from .tensor_darcy import define_tensor_darcy, recover_tensor_darcy
from .tetrahedral_darcy import define_tetrahedral_darcy, recover_tetrahedral_darcy
from .three_field import (
    define_three_field,
    define_three_field_3d,
    recover_three_field,
    recover_three_field_3d,
)
from .transport import define_transport, recover_transport, transport_constraints
from .vector import (
    define_flow,
    define_herrmann_elasticity,
    flow_constraints,
    herrmann_constraints,
    recover_vector,
)
from .weak_stress import define_weak_stress, recover_weak_stress, weak_stress_constraints


def execution_options(options: dict[str, Any]) -> tuple[ExecutionConfig, SolverConfig]:
    """Separate scheduling and algebra choices from the literal mathematical form data.

    The dictionary is this controller's owned copy. Every remaining option is
    passed to its declared mathematical provider, so unsupported inputs fail
    explicitly instead of disappearing or changing the approximation space.
    """
    execution = ExecutionConfig(
        backend=options.pop("backend", "serial"), workers=options.pop("workers", None)
    )
    solvers = SolverConfig(
        local_solver=options.pop("local_solver", "scipy"),
        global_solver=options.pop("solver", "scipy"),
        local_refinement_precision=options.pop("local_refinement_precision", "double"),
        global_refinement_precision=options.pop("global_refinement_precision", "double"),
    )
    return execution, solvers


def execute_definition(
    definition: Any,
    recovery: Any,
    constraints: Any,
    execution: ExecutionConfig,
    solvers: SolverConfig,
) -> Any:
    """Execute generic assembly/solve with separately declared physical integral rows."""
    system = assemble(definition.problem, execution=execution, solvers=solvers)
    gauges = constraints(definition, system)
    result = system.solve(fixed=definition.problem.fixed, constraints=gauges)
    return recovery(definition, system, result)


def darcy(mesh: Any, **options: Any) -> Any:
    """Execute triangular primal or mixed RT0 Darcy equations defined by this example.

    The former parallel_assembly flag is accepted for existing acquisition
    scripts. Generic assemble schedules both provider integration and local
    algebra through the explicit backend/workers execution configuration.
    """
    parallel_assembly = options.pop("parallel_assembly", False)
    if not isinstance(parallel_assembly, bool):
        raise ValueError("parallel_assembly must be a boolean")
    execution, solvers = execution_options(options)
    return execute_definition(
        define_darcy(mesh, **options), recover_darcy, pressure_constraints, execution, solvers
    )


def tetrahedral_darcy(mesh: Any, **options: Any) -> Any:
    """Execute nodal tetrahedral Darcy forms with explicit physical pressure gauges."""
    execution, solvers = execution_options(options)
    return execute_definition(
        define_tetrahedral_darcy(mesh, **options),
        recover_tetrahedral_darcy,
        pressure_constraints,
        execution,
        solvers,
    )


def cartesian_darcy(mesh: Any, **options: Any) -> Any:
    """Execute Cartesian Qk Darcy forms in their original nodal coordinates."""
    execution, solvers = execution_options(options)
    return execute_definition(
        define_cartesian_darcy(mesh, **options),
        recover_cartesian_darcy,
        pressure_constraints,
        execution,
        solvers,
    )


def rt_darcy(mesh: Any, **options: Any) -> Any:
    """Execute RTm/DG Pm with its explicitly prescribed normal moments."""
    execution, solvers = execution_options(options)
    return execute_definition(
        define_rt_darcy(mesh, **options), recover_rt_darcy, pressure_constraints, execution, solvers
    )


def bdm_darcy(mesh: Any, **options: Any) -> Any:
    """Execute independent BDM normal/interior degrees and their DG pressure space."""
    execution, solvers = execution_options(options)
    return execute_definition(
        define_bdm_darcy(mesh, **options),
        recover_bdm_darcy,
        pressure_constraints,
        execution,
        solvers,
    )


def hdiv_darcy(mesh: Any, **options: Any) -> Any:
    """Execute affine tetrahedral/prismatic H(div) forms with literal Piola scaling."""
    execution, solvers = execution_options(options)
    return execute_definition(
        define_hdiv_darcy(mesh, **options),
        recover_hdiv_darcy,
        pressure_constraints,
        execution,
        solvers,
    )


def tensor_darcy(mesh: Any, **options: Any) -> Any:
    """Execute rectangular tensor RT with the declared normal/interior enrichment."""
    execution, solvers = execution_options(options)
    return execute_definition(
        define_tensor_darcy(mesh, **options),
        recover_tensor_darcy,
        pressure_constraints,
        execution,
        solvers,
    )


def robin_diffusion(mesh: Any, **options: Any) -> Any:
    """Execute explicitly declared Robin volume/boundary forms and physical reconstruction."""
    execution, solvers = execution_options(options)
    if isinstance(mesh, TetraMesh):
        return execute_definition(
            define_robin_3d(mesh, **options), recover_robin_3d, robin_gauge_3d, execution, solvers
        )
    return execute_definition(
        define_robin(mesh, **options), recover_robin, robin_gauge, execution, solvers
    )


def three_field_diffusion(mesh: Any, **options: Any) -> Any:
    """Execute independent conormal/pressure trace equations and their Neumann complement."""
    execution, solvers = execution_options(options)
    if isinstance(mesh, TetraMesh):
        return execute_definition(
            define_three_field_3d(mesh, **options),
            recover_three_field_3d,
            pressure_constraints,
            execution,
            solvers,
        )
    return execute_definition(
        define_three_field(mesh, **options),
        recover_three_field,
        pressure_constraints,
        execution,
        solvers,
    )


def moment_diffusion(mesh: Any, **options: Any) -> Any:
    """Execute the declared MsHHO moment/energy blocks and complete physical mean."""
    execution, solvers = execution_options(options)
    if isinstance(mesh, TetraMesh):
        return execute_definition(
            define_moment_diffusion_3d(mesh, **options),
            recover_moment_diffusion_3d,
            moment_constraints,
            execution,
            solvers,
        )
    return execute_definition(
        define_moment_diffusion(mesh, **options),
        recover_moment_diffusion,
        moment_constraints,
        execution,
        solvers,
    )


def petrov_galerkin_diffusion(mesh: Any, *, stabilization_parameter: float, **options: Any) -> Any:
    """Execute mean-constrained diffusion plus its explicitly assembled jump equation.

    Boundary pressure enters once through the jump functional on its exact common
    fine partition. The base global pressure functional is zero. Neumann faces
    have essential physical flux coefficients and receive no penalty/enrichment.
    """
    execution, solvers = execution_options(options)
    datum = options.pop("dirichlet", 0.0)
    lower_bound = options.pop("ellipticity_lower_bound", None)
    neumann_faces = tuple((options.get("neumann") or {}).keys())
    definition = define_darcy(mesh, dirichlet=0.0, **options)
    base = assemble(definition.problem, execution=execution, solvers=solvers)
    system, jumps, lower = add_jump_form(
        definition,
        base,
        dirichlet=datum,
        alpha=stabilization_parameter,
        neumann_faces=neumann_faces,
        lower_bound=lower_bound,
    )
    solution = system.solve(
        fixed=definition.problem.fixed, constraints=pressure_constraints(definition, system)
    )
    return recover_penalty(
        definition, system, solution, jumps, lower, alpha=stabilization_parameter
    )


def primal_elasticity(mesh: Any, **options: Any) -> Any:
    """Execute explicit symmetric-strain energy and moment-fixed rigid coordinates."""
    execution, solvers = execution_options(options)
    return execute_definition(
        define_primal_elasticity(mesh, **options),
        recover_primal_elasticity,
        primal_constraints,
        execution,
        solvers,
    )


def weak_stress_elasticity(mesh: Any, **options: Any) -> Any:
    """Execute weak stress symmetry with original BDM/RT/AFW physical field spaces."""
    execution, solvers = execution_options(options)
    if "degree" in options:
        options["stress_degree"] = options.pop("degree")
    if isinstance(mesh, CartesianMacroMesh):
        options.setdefault("stress_degree", 1)
    return execute_definition(
        define_weak_stress(mesh, **options),
        recover_weak_stress,
        weak_stress_constraints,
        execution,
        solvers,
    )


def flow(mesh: Any, **options: Any) -> Any:
    """Execute explicit Stokes/Brinkman/Oseen forms and physically admissible integral rows."""
    execution, solvers = execution_options(options)
    return execute_definition(
        define_flow(mesh, **options), recover_vector, flow_constraints, execution, solvers
    )


def herrmann_elasticity(mesh: Any, **options: Any) -> Any:
    """Execute user-declared GaLS or Taylor--Hood Herrmann elasticity forms."""
    execution, solvers = execution_options(options)
    return execute_definition(
        define_herrmann_elasticity(mesh, **options),
        recover_vector,
        herrmann_constraints,
        execution,
        solvers,
    )


def elasticity(mesh: Any, *, formulation: str = "primal", **options: Any) -> Any:
    """Select this example's explicit primal or Herrmann mathematical definition."""
    if formulation == "primal":
        return primal_elasticity(mesh, **options)
    return herrmann_elasticity(mesh, formulation=formulation, **options)


def brinkman(mesh: Any, *, formulation: str = "taylor-hood", **options: Any) -> Any:
    """Execute the selected explicit Stokes/Brinkman/Oseen velocity-pressure forms.

    Taylor--Hood is the default. ``formulation`` may select the same USFEM or
    Oseen spaces accepted by :func:`flow`, including their explicit stabilization.
    """
    return flow(mesh, formulation=formulation, **options)


def mapped_darcy(mesh: Any, *, global_rtol: float = 1e-10, **options: Any) -> Any:
    """Execute explicit mapped H(div) equations with the physical block residual criterion."""
    import numpy as np

    if np.iscomplexobj(global_rtol) or not np.isfinite(global_rtol) or global_rtol <= 0:
        raise ValueError("global_rtol must be finite and positive")
    execution, solvers = execution_options(options)
    definition = define_mapped_darcy(mesh, **options)
    system = assemble(definition.problem, execution=execution, solvers=solvers)
    solution = system.solve(
        fixed=definition.problem.fixed,
        constraints=pressure_constraints(definition, system),
        rtol=global_rtol,
    )
    return recover_mapped_darcy(definition, system, solution)


def transport(mesh: Any, **options: Any) -> Any:
    """Execute conservative Galerkin/SUPG/UNUSUAL with explicit physical boundary rows."""
    execution, solvers = execution_options(options)
    return execute_definition(
        define_transport(mesh, **options),
        recover_transport,
        transport_constraints,
        execution,
        solvers,
    )
