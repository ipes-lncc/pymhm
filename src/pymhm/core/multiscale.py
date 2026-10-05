"""Assembly, solution and recursive reconstruction of user-defined hybrid equations.

All numerical elimination delegates to the common constrained condensation and
ordered contribution owners. A recursive local operator is itself another
MultiscaleProblem; the same contracts apply at every scale. No physical model
or method name selects the operator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import copy
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar, cast

import numpy as np
from scipy import sparse

from pymhm.core.assembly import HybridProblem, SolverConfig, assemble_hybrid
from pymhm.core.contracts import HybridSolution, LocalAssembly, LocalResponse
from pymhm.core.contributions import local_global_contribution
from pymhm.core.equations import (
    CompiledLocalEquations,
    Equation,
    FormCompiler,
    LocalEquations,
    compile_form,
    compile_local_equations,
)
from pymhm.core.nested import NestedLocalProblem, nest_hybrid_system
from pymhm.core.system import HybridSystem, solve_hybrid_system
from pymhm.core.validation import FloatArray, IntArray
from pymhm.core.variational import GlobalForm
from pymhm.execution.cpu import ExecutionConfig, _prepare_callable

Item = TypeVar("Item")
_DEFAULT_EXECUTION = ExecutionConfig()
_DEFAULT_SOLVERS = SolverConfig()


@dataclass(frozen=True)
class MultiscaleProblem(Generic[Item]):
    """A global variational equation coupled to independently declared local equations.

    ``global_equation`` supplies additional bilinear and linear global forms;
    each local provider supplies its own explicit ``C u + D lambda = g``
    balance. Global coordinates are trace first, then each cell's retained
    modes in item order. ``coarse_sizes`` declares those counts, including zero.
    A global form can contain facet terms between different cells, Robin terms
    or other user-defined couplings. Its full reduced dimensions must match.

    The provider runs once per item, inside the selected worker, and returns
    LocalEquations or a compiled equation record. Scalar/vector/mixed fields and
    recursive local operators share this API. Boundary coefficients and physical
    gauge rows are explicit; the package does not choose them from a PDE name.
    """

    global_equation: Equation
    local_provider: Callable[[Item], LocalEquations | NestedEquations | CompiledLocalEquations]
    items: Iterable[Item]
    trace_size: int
    coarse_sizes: tuple[int, ...]
    fixed: Mapping[int, float] | None = None
    constraints: tuple[tuple[FloatArray, float], ...] = ()
    compiler: FormCompiler = compile_form

    def __post_init__(self) -> None:
        """Validate coordinates and form/compiler contracts without consuming the items."""
        if not isinstance(self.global_equation, Equation):
            raise TypeError("global_equation must be an Equation")
        if not callable(self.local_provider) or not callable(self.compiler):
            raise TypeError("local_provider and compiler must be callable")
        layout = GlobalForm(
            self.trace_size,
            self.coarse_sizes,
            fixed_trace=self.fixed,
            constraints=self.constraints,
        )
        object.__setattr__(self, "trace_size", layout.trace_size)
        object.__setattr__(self, "coarse_sizes", layout.coarse_sizes)
        object.__setattr__(self, "fixed", layout.fixed_trace)
        object.__setattr__(self, "constraints", layout.constraints)


@dataclass(frozen=True)
class NestedEquations:
    """Declare a child hybrid problem with a parent boundary-trace constraint.

    If the child's reduced equation is ``A x=f``, E selects its boundary trace
    coordinates and M is ``trace_map``, the local equations are
    ``-A x + E mu=-f`` and ``E.T x-M lambda=0``. The global pairing is
    ``+M.T mu`` in the declared DSL balance C. The original physical moment
    is ``-M.T mu``; negating that equation matches the shared reduced saddle
    convention. The local field contains child coordinates followed by boundary
    reactions. This is a generic equality-constrained operator, independent of
    a physical PDE. Explicit ``kernel``/``left_kernel`` are in child coordinates;
    their boundary reactions are lifted by the shared nesting owner.

    Children leave boundary data and physical gauges to their parent. Supplied
    moments have one row per child coordinate and are extended by zero reaction
    weights. M includes every restriction and orientation; it is not inferred.
    """

    problem: MultiscaleProblem[Any]
    boundary_dofs: Any
    trace_map: Any
    dofs: Any
    kernel: Any = None
    left_kernel: Any = None
    moments: Any = None
    test_moments: Any = None
    metadata: Any = None


@dataclass(frozen=True)
class MultiscaleSolution(HybridSolution):
    """Hybrid coefficients and recursively recovered child solutions.

    ``children[cell]`` is None for a leaf local discretization. Otherwise its
    coordinates are exactly that cell's reconstructed field; its own fields
    and children recover the finer scales without another global solve.
    """

    children: tuple[MultiscaleSolution | None, ...] = ()


@dataclass(frozen=True)
class _CellRecord:
    """Picklable direct global terms, application metadata and an assembled child."""

    equations: CompiledLocalEquations
    child: MultiscaleSystem | None = None
    nested: NestedLocalProblem | None = None


@dataclass(frozen=True)
class _Provider(Generic[Item]):
    """Compile a cell and all its finer levels entirely inside its worker."""

    provider: Callable[[Item], LocalEquations | NestedEquations | CompiledLocalEquations]
    compiler: FormCompiler
    solvers: SolverConfig

    def prepare_runtime(self) -> None:
        """Forward explicit provider/compiler library initialization before limits."""
        _prepare_callable(self.provider)
        _prepare_callable(self.compiler)

    def close(self) -> None:
        """Release an application's reusable local workspaces after execution."""
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()

    def __call__(self, item: Item) -> LocalAssembly:
        """Return one checked local operator with coefficient-only reconstruction data."""
        supplied = self.provider(item)
        child = None
        nested = None
        if isinstance(supplied, NestedEquations):
            inner = supplied.problem
            if not isinstance(inner, MultiscaleProblem):
                raise TypeError("a nested problem must be a MultiscaleProblem")
            if inner.fixed or inner.constraints:
                raise ValueError("a nested child must leave boundary data and gauges to its parent")
            child = assemble(inner, solvers=self.solvers)
            nested = nest_hybrid_system(
                child,
                supplied.boundary_dofs,
                supplied.trace_map,
                supplied.dofs,
                kernel=supplied.kernel,
                left_kernel=supplied.left_kernel,
                constraints=supplied.moments,
                test_constraints=supplied.test_moments,
            )
            count = len(nested.problem.trace_dofs)
            supplied = CompiledLocalEquations(
                nested.problem, np.zeros((count, count)), np.zeros(count), supplied.metadata
            )
        if isinstance(supplied, LocalEquations):
            if isinstance(supplied.a, MultiscaleProblem):
                inner = supplied.a
                if inner.fixed or inner.constraints:
                    raise ValueError(
                        "a recursive local operator must leave boundary data and gauges "
                        "to its parent; encode essential constraints as explicit local blocks"
                    )
                # The outer executor owns available workers. Child assembly is serial,
                # preventing nested process pools and native communicator transfer.
                child = assemble(inner, solvers=self.solvers)
                extra_load = self.compiler(supplied.L, child.rhs.shape)
                supplied = replace(supplied, a=child.matrix, L=child.rhs + extra_load)
            equations = compile_local_equations(supplied, self.compiler)
        elif isinstance(supplied, CompiledLocalEquations):
            equations = supplied
        else:
            raise TypeError("local_provider must return LocalEquations or CompiledLocalEquations")
        if child is not None and nested is None:
            # Parent forcing may act on child balance rows. Its retained rows
            # are leaf compatibility equations and cannot change independently.
            if np.any(equations.problem.coupling[child.trace_size :]):
                raise ValueError(
                    "parent coupling must not change child retained compatibility rows"
                )
            if np.any(equations.problem.load[child.trace_size :] != child.rhs[child.trace_size :]):
                raise ValueError("recursive extra loads must not change child retained rows")
        return LocalAssembly(equations.problem, _CellRecord(equations, child, nested))


def _contribution(
    response: LocalResponse, record: _CellRecord, coarse_dofs: IntArray
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Add D and g to this cell's Schur block before ordered shared-face reduction."""
    direct = record.equations
    indices, matrix, load = local_global_contribution(
        response, coarse_dofs, direct_matrix=direct.matrix, direct_load=direct.load
    )
    dtype = np.result_type(matrix, load)
    return indices, matrix.astype(dtype, copy=False), load.astype(dtype, copy=False)


class MultiscaleSystem(HybridSystem):
    """Assembled variational equations with application data and recursive reconstruction.

    Assembly stores numerical operators, executed local bases and child systems,
    without retaining native UFL meshes/forms. The inherited physical moment
    utility can build gauge rows from local field integrals before solving.
    """

    layout: GlobalForm
    solvers: SolverConfig
    cells: tuple[_CellRecord, ...]

    def with_rhs(self, rhs: Any, *, load_scale: Any = None) -> MultiscaleSystem:
        """Delegate a compatible load update to :func:`with_global_load`."""
        return with_global_load(self, rhs, load_scale=load_scale)

    def solve(self, **options: Any) -> MultiscaleSolution:
        """Delegate solution and hierarchical recovery to :func:`solve_multiscale_system`."""
        return solve_multiscale_system(self, **options)

    def reconstruct(self, coordinates: Any, *, rhs: Any = None) -> MultiscaleSolution:
        """Delegate recovery of executed coordinates to :func:`reconstruct_multiscale`."""
        return reconstruct_multiscale(self, coordinates, rhs=rhs)


def with_global_load(
    system: MultiscaleSystem, rhs: Any, *, load_scale: Any = None
) -> MultiscaleSystem:
    """Reuse executed local responses for a new global balance right-hand side.

    Retained rows enforce the existing local source's compatibility and must
    remain unchanged. Changing a volume source requires new local equations
    or explicitly cached local factors. Child operators and basis matrices
    are reused literally, and all local fields remain reconstructible.
    """
    updated = cast(MultiscaleSystem, HybridSystem.with_rhs(system, rhs, load_scale=load_scale))
    if np.any(updated.rhs[system.trace_size :] != system.rhs[system.trace_size :]):
        raise ValueError("a global load update must preserve retained compatibility rows")
    updated.responses = system.responses
    updated.local_metadata = system.local_metadata
    updated.layout, updated.solvers, updated.cells = (system.layout, system.solvers, system.cells)
    return updated


def solve_multiscale_system(system: MultiscaleSystem, **options: Any) -> MultiscaleSolution:
    """Solve the declared boundary/gauge problem and recover every finer field.

    Keyword overrides use the common HybridSystem solver contract. Omitting
    ``fixed`` and ``constraints`` uses those declared by MultiscaleProblem.
    Numerical criteria, original equations and basis coordinates are the
    same as in the shared hybrid solver.
    """
    defaults: dict[str, Any] = {
        "solver": system.solvers.global_solver,
        "fixed": dict(system.layout.fixed_trace or {}),
        "constraints": list(system.layout.constraints),
        "refinement_precision": system.solvers.global_refinement_precision,
    }
    defaults.update(options)
    result = solve_hybrid_system(system, **defaults)
    coordinates = np.concatenate((result.trace, *result.coarse))
    return _solution_tree(system, result, coordinates)


def _solution_tree(
    system: MultiscaleSystem, result: HybridSolution, coordinates: FloatArray
) -> MultiscaleSolution:
    """Recover each child's fields against its actual parent-induced forcing."""
    children: list[MultiscaleSolution | None] = []
    for record, field in zip(system.cells, result.fields, strict=True):
        child = record.child
        if child is None:
            children.append(None)
        else:
            if record.nested is None:
                problem = record.equations.problem
                forcing = problem.load - problem.coupling @ coordinates[problem.trace_dofs]
                child_coordinates = field
            else:
                n = len(child.rhs)
                child_coordinates = field[:n]
                forcing = child.rhs.copy()
                forcing[record.nested.boundary_dofs] += field[n:]
            children.append(child.reconstruct(child_coordinates, rhs=forcing))
    return MultiscaleSolution(
        result.trace,
        result.coarse,
        result.fields,
        result.residual,
        result.gauge_multipliers,
        result.raw_residual,
        result.raw_residual_norm,
        tuple(children),
    )


def reconstruct_multiscale(
    system: MultiscaleSystem, coordinates: Any, *, rhs: Any = None
) -> MultiscaleSolution:
    """Recover finer fields from executed hybrid coordinates without solving again.

    ``rhs`` is the actual load at this scale, including parent coupling.
    This method checks the represented global equations before descending.
    Parent loads must not change retained leaf compatibility equations.
    """
    coordinates = compile_form(coordinates, system.rhs.shape)
    rhs = system.rhs if rhs is None else compile_form(rhs, system.rhs.shape)
    defect = system.matrix @ coordinates - rhs
    scale = max(
        float(np.linalg.norm(rhs)),
        float(np.linalg.norm(system.load_scale)),
        float(np.linalg.norm(abs(system.matrix) @ abs(coordinates))),
        np.finfo(float).tiny,
    )
    absolute = float(np.linalg.norm(defect))
    residual = absolute / scale
    if residual > 1e-08:
        raise ValueError("recursive coordinates do not satisfy their executed global equations")
    coarse = tuple(
        (
            coordinates[first:last].copy()
            for first, last in zip(
                system.kernel_offsets[:-1], system.kernel_offsets[1:], strict=True
            )
        )
    )
    fields = tuple(
        (
            response.reconstruct(coordinates[response.problem.trace_dofs], value)
            for response, value in zip(system.responses, coarse, strict=True)
        )
    )
    result = HybridSolution(
        coordinates[: system.trace_size].copy(),
        coarse,
        fields,
        residual,
        np.empty(0),
        residual,
        absolute,
    )
    return _solution_tree(system, result, coordinates)


def leaf_moment(
    system: MultiscaleSystem, weights: Callable[[Any], Any]
) -> tuple[FloatArray, float]:
    """Represent a physical finest-scale moment as ``row @ coordinates + offset``.

    ``weights(metadata)`` supplies one integration vector in each leaf's
    executed coefficient basis. Vector fields use that basis's component
    ordering. Child source offsets and boundary-reaction padding are included
    recursively. A physical gauge with prescribed moment m uses
    ``(row, m-offset)`` as its global constraint; no field is solved or sampled.
    """
    local_weights: list[Any] = []
    offset = 0.0
    for record in system.cells:
        if record.child is None:
            local_weights.append(weights(record.equations.metadata))
        else:
            row, child_offset = leaf_moment(record.child, weights)
            offset += child_offset
            local_weights.append(
                row
                if record.nested is None
                else np.r_[row, np.zeros(len(record.nested.boundary_dofs))]
            )
    row, target = system.mean_constraint(local_weights)
    return row, offset - target


def with_global_equation(
    system: MultiscaleSystem,
    equation: Equation,
    *,
    compiler: FormCompiler = compile_form,
) -> MultiscaleSystem:
    """Add global forms without rebuilding any local operator or executed basis.

    The forms act on the existing reduced coordinates. They may depend on the
    executed local response maps, for example a jump of reconstructed fields.
    This operation adds both the declared operator and load once, preserving
    child systems and coefficient interpretation. It returns a new system;
    it does not infer jump signs, penalties or stability conditions.
    """
    if not isinstance(equation, Equation):
        raise TypeError("equation must be an Equation")
    a = compiler(equation.a, system.matrix.shape)
    load = compiler(equation.L, system.rhs.shape)
    updated = copy(system)
    updated.matrix = (system.matrix + sparse.csc_matrix(a)).tocsc()
    updated.matrix.eliminate_zeros()
    updated.rhs = system.rhs + load
    updated.load_scale = system.load_scale + abs(load)
    return updated


def assemble(
    problem: MultiscaleProblem[Item],
    *,
    execution: ExecutionConfig = _DEFAULT_EXECUTION,
    solvers: SolverConfig = _DEFAULT_SOLVERS,
) -> MultiscaleSystem:
    """Assemble user-defined local/global equations in serial or bounded parallel batches.

    Workers build and eliminate local equations. Only the coordinator sums shared
    global indices, in item order. The additional global equation is assembled
    once after those contributions. Recursive children run serially within their
    owning worker, so one resource budget controls the hierarchy. Live optional
    native resources never cross the worker boundary.
    """
    if not isinstance(problem, MultiscaleProblem):
        raise TypeError("problem must be a MultiscaleProblem")
    layout = GlobalForm(
        problem.trace_size,
        problem.coarse_sizes,
        fixed_trace=problem.fixed,
        constraints=problem.constraints,
    )
    provider = _Provider(problem.local_provider, problem.compiler, solvers)
    hybrid = assemble_hybrid(
        HybridProblem(
            layout,
            provider,
            problem.items,
            _contribution,
            require_local_trace_coverage=False,
            contribution_execution="worker",
        ),
        execution=execution,
        solvers=solvers,
    )
    system = MultiscaleSystem.__new__(MultiscaleSystem)
    system.__dict__.update(hybrid.__dict__)
    system.layout, system.solvers = layout, solvers
    system.cells = hybrid.local_metadata
    system.local_metadata = tuple(record.equations.metadata for record in system.cells)
    return with_global_equation(system, problem.global_equation, compiler=problem.compiler)


def solve(
    problem: MultiscaleProblem[Item],
    *,
    execution: ExecutionConfig = _DEFAULT_EXECUTION,
    solvers: SolverConfig = _DEFAULT_SOLVERS,
) -> MultiscaleSolution:
    """Assemble, solve and reconstruct a user-defined variational hierarchy."""
    return assemble(problem, execution=execution, solvers=solvers).solve()
