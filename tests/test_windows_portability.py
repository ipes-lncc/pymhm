"""Exercise the portable hybrid API and native solvers under Windows-compatible spawn."""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import pymhm
from pymhm import (
    CartesianMacroMesh,
    Equation,
    LocalContext,
    LocalEquations,
    MeshHierarchy,
    MultiscaleProblem,
    TraceBinding,
    assemble,
    bind_problem,
)
from pymhm.core.assembly import SolverConfig
from pymhm.core.contracts import LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.execution.cpu import ExecutionConfig


def _local_problem(cell: int, *, source: float) -> LocalProblem:
    """Declare P1 diffusion on a half-interval with globally oriented traction coordinates."""
    conductivity = (2.0, 3.0)[cell]
    return LocalProblem(
        2 * conductivity * np.array([[1.0, -1.0], [-1.0, 1.0]]),
        np.diag([-1.0, 1.0]),
        np.full(2, source / 4),
        np.array([cell, cell + 1]),
        kernel=np.ones((2, 1)),
        constraints=np.full((2, 1), 0.25),
    )


def _local_equations(cell: int, *, source: float) -> LocalEquations:
    """Express the same local operator and signed continuity balance through the generic DSL."""
    problem = _local_problem(cell, source=source)
    return LocalEquations(
        a=problem.matrix,
        L=problem.load,
        b=problem.coupling,
        c=problem.coupling.T,
        dofs=problem.trace_dofs,
        kernel=problem.kernel,
        moments=problem.constraints,
        metadata={
            "cell": cell,
            "pid": os.getpid(),
            "start_method": multiprocessing.get_start_method(allow_none=True),
        },
    )


@dataclass(frozen=True)
class _StripInterface:
    """Declare constant normal densities on vertical faces of two y-independent strips."""

    mesh: Any
    size: int = 3

    def binding(self, cell: int) -> TraceBinding:
        """Map global increasing-x density into outward left/right local coordinates."""
        return TraceBinding(
            [cell, cell + 1],
            np.diag([-1.0, 1.0]),
            basis_id="constant vertical-face density, increasing-x global normal",
        )


def _context_equations(local: LocalContext, *, source: float) -> LocalEquations:
    """Reuse the diffusion owner while letting the context supply orientation and layout."""
    explicit = _local_equations(local.cell, source=source)
    return local.equations(
        a=explicit.a,
        L=explicit.L,
        b=np.eye(2),
        c=np.eye(2),
        kernel=explicit.kernel,
        moments=explicit.moments,
        metadata=explicit.metadata,
    )


@pytest.mark.parametrize("solver", ["scipy", "pypardiso"])
@pytest.mark.parametrize(
    "source,boundary",
    [(0.0, (0.2, 1.1)), (0.6, (0.0, 0.0)), (0.6, (0.2, 1.1))],
)
def test_generic_hybrid_and_dsl_preserve_conforming_fields_under_spawn(
    solver: str, source: float, boundary: tuple[float, float]
) -> None:
    """Check original rows, physical moments and the shared face without extended precision."""
    if solver == "pypardiso":
        pytest.importorskip("pypardiso")

    # Independent conforming assembly on nodes x=(0, 1/2, 1), with strong
    # Dirichlet elimination. This is a discrete comparison when source != 0.
    conforming = np.array([[4.0, -4.0, 0.0], [-4.0, 10.0, -6.0], [0.0, -6.0, 6.0]])
    load = source * np.array([0.25, 0.5, 0.25])
    expected = np.array([boundary[0], 0.0, boundary[1]])
    expected[1] = np.linalg.solve(
        conforming[1:2, 1:2], load[1:2] - conforming[1:2, [0, 2]] @ expected[[0, 2]]
    )[0]
    expected_fields = expected[np.array([[0, 1], [1, 2]])]
    boundary_load = np.array([-boundary[0], 0.0, boundary[1]])
    solvers = SolverConfig(local_solver=solver, global_solver=solver)
    solutions = []

    for backend in ("serial", "thread", "process"):
        execution = ExecutionConfig(backend=backend, workers=2, native_threads=1)
        direct = HybridSystem.from_local_factory(
            partial(_local_problem, source=source),
            range(2),
            boundary_load=boundary_load,
            local_solver=solver,
            backend=backend,
            workers=2,
            native_threads=1,
        )
        declared = assemble(
            MultiscaleProblem(
                Equation(0, np.r_[boundary_load, 0.0, 0.0]),
                partial(_local_equations, source=source),
                range(2),
                trace_size=3,
                coarse_sizes=(1, 1),
            ),
            execution=execution,
            solvers=solvers,
        )
        macro = CartesianMacroMesh(2, 1)
        interface = _StripInterface(macro)
        hierarchy = MeshHierarchy(macro, tuple(macro.submesh(cell, 1) for cell in range(2)))
        bound = assemble(
            bind_problem(
                hierarchy,
                interface,
                partial(_context_equations, source=source),
                retained=1,
                global_equation=lambda context: Equation(0, context.trace_load(boundary_load)),
            ),
            execution=execution,
            solvers=solvers,
        )
        for contextual in (declared, bound):
            for record in contextual.cells:
                metadata = record.equations.metadata
                if backend == "process":
                    assert metadata["pid"] != os.getpid()
                    assert metadata["start_method"] == "spawn"
                else:
                    assert metadata["pid"] == os.getpid()

        for system in (direct, declared, bound):
            result = system.solve(solver=solver)
            solutions.append(result)
            assert_allclose(result.fields, expected_fields, rtol=2e-12, atol=2e-12)
            assert_allclose(result.fields[0][-1], result.fields[1][0], atol=2e-12)
            assert result.raw_residual is not None and result.raw_residual < 1e-12
            for response, field in zip(system.responses, result.fields, strict=True):
                problem = response.problem
                assert_allclose(
                    problem.matrix @ field
                    + problem.coupling @ result.trace[problem.trace_dofs]
                    - problem.load,
                    0,
                    atol=2e-12,
                    rtol=0,
                )
                assert_allclose(problem.constraints.T @ response.source, 0, atol=2e-12)
                assert_allclose(problem.constraints.T @ response.lifts, 0, atol=2e-12)
                assert_allclose(response.retained_basis, problem.kernel, atol=2e-12)
                assert field.dtype == np.dtype(float)
            actual_integral = sum(float(np.full(2, 0.25) @ field) for field in result.fields)
            assert_allclose(actual_integral, np.array([0.25, 0.5, 0.25]) @ expected, atol=2e-12)
            if system is bound:
                for cell in range(2):
                    binding = interface.binding(cell)
                    assert result.trace_bindings[cell].basis_digest == binding.basis_digest
                    assert not result.trace_bindings[cell].trial_map.flags.writeable
                    assert_allclose(
                        result.local_trace(cell), binding.trial_map @ result.trace[binding.dofs]
                    )
                    assert_allclose(
                        result.local_trace(cell, test=True),
                        binding.test_map @ result.trace[binding.test_dofs],
                    )

    for result in solutions[1:]:
        assert_allclose(result.trace, solutions[0].trace, rtol=2e-12, atol=2e-12)
        assert_allclose(result.coarse, solutions[0].coarse, rtol=2e-12, atol=2e-12)


def test_installed_wheel_owns_imports_when_requested() -> None:
    """Reject an editable checkout silently replacing the wheel in its isolated CI check."""
    expected_root = os.environ.get("PYMHM_EXPECT_INSTALL_ROOT")
    if expected_root is None:
        pytest.skip("The installed-wheel check supplies its isolated installation root")
    assert pymhm.__file__ is not None
    assert Path(pymhm.__file__).resolve().is_relative_to(Path(expected_root).resolve())


def test_core_and_native_basix_work_without_optional_backends() -> None:
    """Check fresh core imports and native Basix with optional backends blocked."""
    script = """
import importlib.abc
import sys
import numpy as np

class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'dolfinx', 'mpi4py', 'petsc4py', 'cupy', 'nvmath', 'pyamgx',
            'pypardiso', 'gmsh', 'netgen',
        }:
            raise AssertionError('Portable operation imported ' + fullname)

sys.meta_path.insert(0, BlockNative())
from pymhm import (
    CartesianMacroMesh, Equation, LocalEquations, MeshHierarchy, MultiscaleProblem,
    SkeletonSpace, TraceBinding, bind_interface, bind_problem, solve,
)
from pymhm.fem.reference import ReferenceElementSpec, create_reference_element, tabulate_reference

def local(cell):
    return LocalEquations([[2.0]], [1.0], [[1.0]], [[-1.0]], [0], d=[[1.0]])

problem = MultiscaleProblem(Equation(0, 0), local, [0], 1, (0,))
result = solve(problem)
np.testing.assert_allclose(result.trace, [1 / 3], atol=1e-14)
np.testing.assert_allclose(result.fields, [[1 / 3]], atol=1e-14)

macro = CartesianMacroMesh(1)
hierarchy = MeshHierarchy(macro, (macro,))

class CustomInterface:
    size = 1
    def binding(self, cell):
        return TraceBinding([0], [[2.0]], test_map=[[-3.0]], basis_id='scaled face basis')

def context_local(context):
    return context.equations(a=[[2.0]], L=[1.0], b=[[1.0]], c=[[-1.0]], d=[[1.0]])

custom = CustomInterface()
bound = bind_problem(hierarchy, custom, context_local)
result = solve(bound)
np.testing.assert_allclose(result.fields, [[1/3]], atol=1e-14)
np.testing.assert_allclose(result.trace, [1/6], atol=1e-14)
np.testing.assert_allclose(result.local_trace(0), [1/3], atol=1e-14)
np.testing.assert_allclose(result.local_trace(0, test=True), [-1/2], atol=1e-14)
assert result.trace_bindings[0].basis_digest == custom.binding(0).basis_digest

interface = bind_interface(SkeletonSpace(macro), convention='normal')
def face_local(context):
    width = context.binding.trial_size
    return context.equations(a=[[2.0]], L=[1.0], b=np.ones((1,width)),
                             c=-np.ones((width,1)), d=np.eye(width))

result = solve(bind_problem(hierarchy, interface, face_local))
np.testing.assert_allclose(result.fields, [[1/6]], atol=1e-14)
np.testing.assert_allclose(result.trace, np.full(4,1/6), atol=1e-14)
np.testing.assert_allclose(result.local_trace(0), result.trace, atol=1e-14)
element = create_reference_element(
    ReferenceElementSpec('P', 'triangle', 1, lagrange_variant='equispaced')
)
assert element.dimension == 3
np.testing.assert_allclose(tabulate_reference(element, [[0.2, 0.3]])[0].sum(), 1, atol=1e-14)
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, timeout=60)
