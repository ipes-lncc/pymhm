"""Execute real PETSc-free UFL assembly and hybrid solves in a fresh interpreter.

This importable test helper also runs directly for native platform qualification.
Missing DOLFINx or a requested solver is an error; no native control is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.metadata
import json
import multiprocessing
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal


class _ForbidPETSc(importlib.abc.MetaPathFinder):
    """Make the optional PETSc Python modules unavailable without replacing DOLFINx."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        """Reject PETSc imports, including DOLFINx's optional startup probe."""
        if (
            fullname == "petsc4py"
            or fullname.startswith("petsc4py.")
            or fullname == "dolfinx.fem.petsc"
        ):
            raise ModuleNotFoundError(f"PETSc is forbidden by the native control: {fullname}")
        return None


def _assert_no_petsc() -> None:
    """Reject even indirectly loaded PETSc modules in the coordinator or a worker."""
    assert not [
        name
        for name in sys.modules
        if name == "petsc4py" or name.startswith("petsc4py.") or name == "dolfinx.fem.petsc"
    ]


_assert_no_petsc()
sys.meta_path.insert(0, _ForbidPETSc())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402
from scipy import sparse  # noqa: E402

from examples.variational_darcy import (  # noqa: E402
    DarcyProvider,
    NativeLocalContext,
    affine_pressure,
    build_problem,
    native_local_forms,
)
from pymhm.backends.fenics import from_ufl, mixed_darcy_forms  # noqa: E402
from pymhm.backends.forms import assemble_form  # noqa: E402
from pymhm.core.assembly import SolverConfig  # noqa: E402
from pymhm.core.equations import Equation, LocalEquations  # noqa: E402
from pymhm.core.multiscale import assemble  # noqa: E402
from pymhm.execution.cpu import ExecutionConfig  # noqa: E402
from pymhm.fem.scalar.operators import boundary_data  # noqa: E402


def _unit_source_forms(context: NativeLocalContext) -> LocalEquations:
    """Supply source one using the same declared pressure/normal-flux convention."""
    import ufl

    forms = native_local_forms(context)
    test = ufl.TestFunction(context.space)
    return replace(forms, L=test * ufl.dx(domain=context.space.mesh))


@dataclass(frozen=True)
class _NativeDarcyProvider:
    """Keep native assembly and PETSc exclusion inside each spawned local worker."""

    provider: DarcyProvider

    def prepare_runtime(self) -> None:
        """Load real MPI/DOLFINx before numerical thread limits, with PETSc forbidden."""
        from dolfinx import fem  # noqa: F401
        from mpi4py import MPI  # noqa: F401

        _assert_no_petsc()

    def __call__(self, cell: int) -> LocalEquations:
        """Return real forms with portable process and basis metadata."""
        forms = self.provider(cell)
        metadata = dict(forms.metadata)
        metadata["start_method"] = multiprocessing.get_start_method(allow_none=True)
        _assert_no_petsc()
        return replace(forms, metadata=metadata)


@dataclass(frozen=True)
class _PortableDarcyProvider:
    """Reuse independent portable operators and consistent volume moments."""

    provider: DarcyProvider
    source: float

    def __call__(self, cell: int) -> LocalEquations:
        """Set the source through the owner's physical P1 volume moments."""
        forms = self.provider(cell)
        return replace(forms, L=self.source * forms.moments[:, 0])


def _darcy_control(solver: str) -> list[dict[str, Any]]:
    """Compare signed shared-face hybrid equations against independent full assembly.

    Unit permeability and P1/P0 pressure/normal-flux spaces are identical in both
    assemblies. Sources are zero or one; boundary pressure is zero or 1+x+2y.
    A physical volume moment fixes each local Neumann complement, retaining its
    constant in the global equations. Source-one cases are discrete comparisons.
    """
    records = []
    solvers = SolverConfig(local_solver=solver, global_solver=solver)
    for source, nonhomogeneous in ((0.0, True), (1.0, False), (1.0, True)):
        portable_problem = build_problem(provider="portable", subdivisions=2)
        native_problem = build_problem(provider="fenics", subdivisions=2)
        skeleton = portable_problem.local_provider.skeleton
        boundary, _ = boundary_data(skeleton, affine_pressure if nonhomogeneous else 0.0)
        global_equation = Equation(0, -np.r_[boundary, np.zeros(2)])
        portable_problem = replace(
            portable_problem,
            global_equation=global_equation,
            local_provider=_PortableDarcyProvider(portable_problem.local_provider, source),
        )
        native_provider = replace(
            native_problem.local_provider,
            native_forms=_unit_source_forms if source else native_local_forms,
        )
        native_problem = replace(
            native_problem,
            global_equation=global_equation,
            local_provider=_NativeDarcyProvider(native_provider),
        )
        reference = assemble(portable_problem)
        expected = reference.solve()
        operators = [record.equations.problem for record in reference.cells]
        full_a = sparse.block_diag([problem.matrix for problem in operators]).toarray()
        full_b = np.zeros((full_a.shape[0], skeleton.size))
        offset = 0
        for problem in operators:
            count = problem.matrix.shape[0]
            full_b[offset : offset + count, problem.trace_dofs] = problem.coupling
            offset += count
        full = np.block([[full_a, full_b], [full_b.T, np.zeros((skeleton.size, skeleton.size))]])
        rhs = np.r_[np.concatenate([problem.load for problem in operators]), boundary]
        uncondensed = np.linalg.solve(full, rhs)
        assert_allclose(np.r_[*expected.fields, expected.trace], uncondensed, atol=3e-12, rtol=0)
        for backend in ("serial", "thread", "process"):
            system = assemble(
                native_problem,
                execution=ExecutionConfig(backend, workers=2, native_threads=1, batch_size=1),
                solvers=solvers,
            )
            solution = system.solve()
            assert_allclose(system.matrix.toarray(), reference.matrix.toarray(), atol=3e-14)
            assert_allclose(system.rhs, reference.rhs, atol=3e-14)
            assert_allclose(solution.trace, uncondensed[-skeleton.size :], atol=3e-12, rtol=0)
            assert_allclose(solution.coarse, expected.coarse, atol=3e-12, rtol=0)
            fields = []
            for cell, (native, portable) in enumerate(
                zip(system.cells, reference.cells, strict=True)
            ):
                data, reference_data = native.equations.metadata, portable.equations.metadata
                coordinates, reference_coordinates = data["points"], reference_data["points"]
                permutation = np.argmin(
                    np.linalg.norm(coordinates[:, None] - reference_coordinates[None, :], axis=2),
                    axis=1,
                )
                assert len(np.unique(permutation)) == len(permutation)
                actual, original = native.equations.problem, portable.equations.problem
                assert_allclose(
                    actual.matrix.toarray(),
                    original.matrix.toarray()[permutation][:, permutation],
                    atol=3e-14,
                )
                assert_allclose(actual.coupling, original.coupling[permutation], atol=3e-14)
                assert_allclose(actual.constraints, original.constraints[permutation], atol=3e-14)
                assert_allclose(actual.matrix @ actual.kernel, 0, atol=3e-14)
                assert_allclose(actual.constraints.T @ actual.kernel, [[0.5]], atol=3e-14)
                values = solution.fields[cell]
                assert_allclose(values, expected.fields[cell][permutation], atol=3e-12, rtol=0)
                assert_allclose(
                    actual.constraints.T @ values, 0.5 * solution.coarse[cell], atol=3e-12
                )
                fields.append(values[np.argsort(permutation)])
                assert (data["assembly_pid"] != os.getpid()) == (backend == "process")
                if backend == "process":
                    assert data["start_method"] == "spawn"
                if source == 0:
                    assert_allclose(values, affine_pressure(coordinates), atol=3e-12, rtol=0)
            full_values = np.r_[*fields, solution.trace]
            assert_allclose(full @ full_values, rhs, atol=3e-12, rtol=0)
            assert solution.raw_residual is not None and solution.raw_residual < 1e-12
            if source == 0:
                assert_allclose(solution.trace, skeleton.mesh.normals @ [-1.0, -2.0], atol=3e-12)
            _assert_no_petsc()
            records.append(
                {
                    "source": source,
                    "nonhomogeneous": nonhomogeneous,
                    "execution": backend,
                    "original_equation_residual": solution.raw_residual,
                    "maximum_full_equation_error": float(np.max(np.abs(full @ full_values - rhs))),
                    "maximum_reference_field_difference": float(
                        np.max(np.abs(full_values - uncondensed))
                    ),
                }
            )
    return records


def _local_field_controls(solver: str) -> list[str]:
    """Recover affine scalar/vector fields and RT0/DG0 pressure data in 2D and 3D."""
    import basix.ufl
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    verified = []
    for dimension in (2, 3):
        domain = (
            mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
            if dimension == 2
            else mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
        )
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})
        ds = ufl.Measure("ds", domain=domain, metadata={"quadrature_degree": 6})
        x, normal = ufl.SpatialCoordinate(domain), ufl.FacetNormal(domain)
        affine = 1 + sum((axis + 1) * x[axis] for axis in range(dimension))
        for kind in ("scalar", "vector"):
            space = fem.functionspace(
                domain,
                basix.ufl.element(
                    "Lagrange",
                    domain.basix_cell(),
                    1,
                    shape=() if kind == "scalar" else (dimension,),
                ),
            )
            u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
            exact = (
                affine
                if kind == "scalar"
                else ufl.as_vector([affine + axis for axis in range(dimension)])
            )
            a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(u, v)) * dx
            load = ufl.inner(exact, v) * dx + ufl.inner(ufl.dot(ufl.grad(exact), normal), v) * ds
            problem = from_ufl(a, load, (), np.empty(0, dtype=int))
            result = fem.Function(space)
            result.x.array[:] = problem.condense(solver).source
            error = float(
                fem.assemble_scalar(fem.form(ufl.inner(result - exact, result - exact) * dx))
            )
            assert error < 1e-22
            assert_allclose(problem.matrix @ result.x.array, problem.load, atol=3e-13, rtol=0)
            assert_allclose(assemble_form(a).toarray(), problem.matrix.toarray(), atol=0)
            verified.append(f"{kind}{dimension}")
        element = basix.ufl.mixed_element(
            [
                basix.ufl.element("RT", domain.basix_cell(), 1),
                basix.ufl.element("DG", domain.basix_cell(), 0),
            ]
        )
        space = fem.functionspace(domain, element)
        a, load = mixed_darcy_forms(space, 1.0, 0.0)
        velocity_test, _ = ufl.TestFunctions(space)
        for pressure_value in (0.0, 2.0):
            forcing = load - pressure_value * ufl.dot(velocity_test, normal) * ds
            problem = from_ufl(a, forcing, (), np.empty(0, dtype=int))
            result = fem.Function(space)
            result.x.array[:] = problem.condense(solver).source
            velocity, pressure = ufl.split(result)
            error = float(
                fem.assemble_scalar(
                    fem.form(
                        (ufl.inner(velocity, velocity) + (pressure - pressure_value) ** 2) * dx
                    )
                )
            )
            assert error < 1e-22
            assert_allclose(problem.matrix @ result.x.array, problem.load, atol=3e-13, rtol=0)
        verified.append(f"mixed{dimension}")
    _assert_no_petsc()
    return verified


def _source_provenance() -> dict[str, Any]:
    """Identify the checked-out revision and literal current tracked package sources.

    File bytes come from the working tree, including uncommitted modifications.
    The digest visits sorted tracked POSIX paths, appending UTF-8 path, NUL,
    literal file bytes and NUL for each entry. Native controls require that this
    source snapshot remain unchanged throughout their execution.
    """
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    paths = subprocess.run(
        ["git", "ls-files", "-z", "--", "src/pymhm"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    tracked = sorted(path for path in paths if path)
    assert tracked
    digest = hashlib.sha256()
    for relative in tracked:
        digest.update(relative.encode("utf8") + b"\0")
        digest.update((root / relative).read_bytes() + b"\0")
    return {
        "source_revision": revision,
        "executed_source_sha256": digest.hexdigest(),
        "tracked_source_files": len(tracked),
        "source_digest_convention": "sorted tracked POSIX path, NUL, literal file bytes, NUL",
    }


def run_control(solver: Literal["scipy", "pypardiso"]) -> dict[str, Any]:
    """Require the requested native solver and report successfully executed controls."""
    import basix
    import dolfinx
    import scipy
    import ufl
    from mpi4py import MPI

    import pymhm

    if solver == "pypardiso":
        import pypardiso  # noqa: F401

    root = Path(__file__).resolve().parents[1]
    assert Path(pymhm.__file__).resolve().parent == root / "src/pymhm"
    provenance = _source_provenance()
    versions = {"scipy": importlib.metadata.version("scipy")}
    if solver == "pypardiso":
        versions["pypardiso"] = importlib.metadata.version("pypardiso")
    _assert_no_petsc()
    receipt = {
        **provenance,
        "solver": solver,
        "solver_package_versions": versions,
        "platform": sys.platform,
        "dolfinx_version": dolfinx.__version__,
        "dolfinx_has_petsc": dolfinx.has_petsc,
        "dolfinx_has_petsc4py": dolfinx.has_petsc4py,
        "pymhm_version": pymhm.__version__,
        "basix_version": basix.__version__,
        "ufl_version": ufl.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "mpi_library": MPI.Get_library_version().strip(),
        "local_fields": _local_field_controls(solver),
        "shared_face_cases": _darcy_control(solver),
        "petsc_python_modules": [],
    }
    assert _source_provenance() == provenance, "package sources changed during native qualification"
    return receipt


def main() -> None:
    """Run a required SciPy or PARDISO integration without dependency-based skips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("scipy", "pypardiso"), default="scipy")
    parser.add_argument(
        "--report", type=Path, help="Write the successful native qualification receipt"
    )
    arguments = parser.parse_args()
    receipt = json.dumps(run_control(arguments.solver), indent=2)
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(receipt + "\n", encoding="utf8")
    print(receipt)


if __name__ == "__main__":
    main()
