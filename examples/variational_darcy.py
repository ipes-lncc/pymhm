"""Solve the same affine Darcy hybrid form with portable or native UFL assembly.

Run as a module, for example ``python -m examples.variational_darcy --backend
process --workers 2 --batch-size 1 --provider fenics`` in the Pixi fem environment.
The two macrotriangles have independent local meshes and signed shared faces.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from pymhm.core.contracts import HybridSolution
from pymhm.core.equations import Equation, LocalEquations, columns, rows
from pymhm.core.multiscale import MultiscaleProblem, assemble
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.execution.cpu import ExecutionConfig
from pymhm.fem.scalar.operators import boundary_data, face_integration, p1_operators
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


def affine_pressure(points: FloatArray) -> FloatArray:
    """Return the prescribed pressure ``p(x,y)=1+x+2y`` at physical points."""
    return 1 + points[:, 0] + 2 * points[:, 1]


@dataclass(frozen=True)
class DarcyProvider:
    """Build one primal Darcy problem using the declared signed macroface map.

    Permeability is one and source zero on the unit square. Continuous local
    P1 pressure has a constant kernel with an integral moment. ``kind`` selects
    assembly only; it preserves operator, fine mesh, trace space and data.
    The callable is top-level and holds ordinary picklable mesh data. Native
    UFL/DOLFINx objects are constructed inside the invocation on ``COMM_SELF``;
    its forms are compiled inside that worker before numerical responses and
    copied point coordinates reach the coordinator.
    """

    macro_mesh: TriangleMesh
    skeleton: SkeletonSpace
    kind: Literal["portable", "fenics"] = "portable"
    subdivisions: int = 2
    native_forms: Callable[[NativeLocalContext], LocalEquations] | None = None

    def __post_init__(self) -> None:
        """Validate the assembly choice and local refinement parameter."""
        if self.kind not in ("portable", "fenics"):
            raise ValueError("kind must be portable or fenics")
        positive_int(self.subdivisions, "subdivisions")
        if self.skeleton.mesh is not self.macro_mesh or self.skeleton.components != 1:
            raise ValueError("DarcyProvider requires its own scalar skeleton")

    def __call__(self, cell: int) -> LocalEquations:
        """Declare one cell's equations with numerical reconstruction metadata."""
        if self.kind == "fenics":
            provider = NativeFormProvider(
                self.macro_mesh,
                self.skeleton,
                native_local_forms if self.native_forms is None else self.native_forms,
                self.subdivisions,
            )
            return provider(cell)
        fine = self.macro_mesh.submesh(cell, self.subdivisions)
        matrix, mass, _ = p1_operators(fine)
        coupling, _ = face_integration(self.macro_mesh, cell, fine, self.skeleton)
        return LocalEquations(
            matrix,
            np.zeros(len(fine.points)),
            columns(*coupling.T),
            rows(*(-coupling.T)),
            self.skeleton.cell_dofs(cell),
            kernel=np.ones((len(fine.points), 1)),
            moments=np.asarray(mass.sum(axis=1)).reshape(-1, 1),
            metadata={"cell": cell, "points": fine.points.copy(), "assembly_pid": os.getpid()},
        )


@dataclass(frozen=True)
class NativeLocalContext:
    """Worker-owned native integration data for a user-written UFL form callback.

    ``ds(i+1)`` selects macroface ``i`` in cell incidence order. ``signs``
    converts the fixed macroface normal into the cell's outward normal.
    ``points`` uses the native executed coefficient order. The context never
    leaves its worker; returned reconstruction metadata contains only arrays.
    ``face_endpoints`` stores the fixed global orientation used to write a
    trace polynomial, independently of the outward normal sign.
    """

    space: Any
    ds: Any
    signs: IntArray
    trace_dofs: IntArray
    points: FloatArray
    face_endpoints: FloatArray


@dataclass(frozen=True)
class NativeFormProvider:
    """Supply worker-owned mesh integration data to arbitrary user-written forms.

    The P1 scalar context provides mesh geometry, tagged macrofaces and native
    point ordering. A callback can declare another scalar, vector or mixed
    function space on that mesh and supplies its actual coefficient equations.
    A different coefficient layout must provide its own numerical metadata;
    omitted metadata uses the context's scalar P1 point coordinates.
    Process callbacks must be importable; no live native object is pickled.
    """

    macro_mesh: TriangleMesh
    skeleton: SkeletonSpace
    forms: Callable[[NativeLocalContext], LocalEquations]
    subdivisions: int = 2

    def __post_init__(self) -> None:
        """Validate mesh ownership, callback and local subdivisions."""
        if self.skeleton.mesh is not self.macro_mesh:
            raise ValueError("native provider requires its own skeleton mesh")
        if not callable(self.forms):
            raise TypeError("native forms must be callable")
        positive_int(self.subdivisions, "subdivisions")

    def __call__(self, cell: int) -> LocalEquations:
        """Create the local native context and return the user's equations."""
        fine = self.macro_mesh.submesh(cell, self.subdivisions)
        return _fenics_local(self.macro_mesh, self.skeleton, fine, cell, self.forms)


def native_local_forms(context: NativeLocalContext) -> LocalEquations:
    """Write ``(grad(p),grad(v))+<lambda,v>=(0,v)`` and ``-<p,mu>=g``.

    Both trace pairings include the declared global/outward orientation.
    The pressure mean selects the local complement; its constant remains a
    retained global unknown. No PDE-specific form helper supplies these forms.
    """
    import ufl

    p, v = ufl.TrialFunction(context.space), ufl.TestFunction(context.space)
    dx = ufl.dx(domain=context.space.ufl_domain())
    return LocalEquations(
        ufl.inner(ufl.grad(p), ufl.grad(v)) * dx,
        0.0 * v * dx,
        columns(*(float(sign) * v * context.ds(i + 1) for i, sign in enumerate(context.signs))),
        rows(*(-float(sign) * p * context.ds(i + 1) for i, sign in enumerate(context.signs))),
        context.trace_dofs,
        kernel=np.ones((len(context.points), 1)),
        moments=columns(v * dx),
    )


def _fenics_local(
    coarse: TriangleMesh,
    skeleton: SkeletonSpace,
    fine: TriangleMesh,
    cell: int,
    form_builder: Callable[[NativeLocalContext], LocalEquations] | None = None,
) -> LocalEquations:
    """Build worker-owned native forms for compilation before worker return."""
    import basix.ufl
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    local_mesh = mesh.create_mesh(MPI.COMM_SELF, fine.cells, x=fine.points, e=domain)
    space = fem.functionspace(local_mesh, ("Lagrange", 1))
    entities, tags = [], []
    for side, face in enumerate(coarse.cell_faces[cell]):
        start, end = coarse.points[coarse.faces[face]]
        tangent = end - start

        def on_face(points: Any, start: Any = start, tangent: Any = tangent) -> Any:
            """Select fine boundary facets collinear with the declared macroedge."""
            return np.isclose(
                (points[0] - start[0]) * tangent[1] - (points[1] - start[1]) * tangent[0], 0
            )

        facets = mesh.locate_entities_boundary(local_mesh, 1, on_face)
        entities.extend(facets)
        tags.extend([side + 1] * len(facets))
    order = np.argsort(entities)
    markers = mesh.meshtags(
        local_mesh,
        1,
        np.asarray(entities, dtype=np.int32)[order],
        np.asarray(tags, dtype=np.int32)[order],
    )
    ds = ufl.Measure("ds", domain=local_mesh, subdomain_data=markers)
    points = np.array(space.tabulate_dof_coordinates()[:, :2], copy=True)
    context = NativeLocalContext(
        space,
        ds,
        coarse.signs[cell],
        skeleton.cell_dofs(cell),
        points,
        coarse.points[coarse.faces[coarse.cell_faces[cell]]].copy(),
    )
    forms = (native_local_forms if form_builder is None else form_builder)(context)
    if not isinstance(forms, LocalEquations):
        raise TypeError("native form builder must return LocalEquations")
    return replace(
        forms,
        metadata={"cell": cell, "points": points, "assembly_pid": os.getpid()}
        if forms.metadata is None
        else forms.metadata,
    )


def build_problem(
    *, provider: Literal["portable", "fenics"] = "portable", subdivisions: int = 2
) -> MultiscaleProblem[int]:
    """Declare the local Darcy provider and its common global hybrid form."""
    coarse = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(coarse)
    local = DarcyProvider(coarse, skeleton, provider, subdivisions)
    boundary, _ = boundary_data(skeleton, affine_pressure)
    form = Equation(0, -np.r_[boundary, np.zeros(len(coarse.cells))])
    return MultiscaleProblem(
        form, local, range(len(coarse.cells)), skeleton.size, (1,) * len(coarse.cells)
    )


def solve_example(
    *,
    provider: Literal["portable", "fenics"] = "portable",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    batch_size: int | None = None,
    subdivisions: int = 2,
) -> tuple[HybridSystem, HybridSolution]:
    """Assemble and solve the declared form with one native thread per worker."""
    problem = build_problem(provider=provider, subdivisions=subdivisions)
    execution = ExecutionConfig(backend, workers, native_threads=1, batch_size=batch_size)
    system = assemble(problem, execution=execution)
    return system, system.solve()


def main() -> None:
    """Run the example and print independently checked affine field errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("portable", "fenics"), default="portable")
    parser.add_argument("--backend", choices=("serial", "thread", "process"), default="serial")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--subdivisions", type=int, default=2)
    args = parser.parse_args()
    system, solution = solve_example(
        provider=args.provider,
        backend=args.backend,
        workers=args.workers,
        batch_size=args.batch_size,
        subdivisions=args.subdivisions,
    )
    pressure_error = max(
        float(np.max(np.abs(field - affine_pressure(record["points"]))))
        for field, record in zip(solution.fields, system.local_metadata, strict=True)
    )
    coarse = TriangleMesh.unit_square()
    expected_trace = coarse.normals @ np.array([-1.0, -2.0])
    trace_error = float(np.max(np.abs(solution.trace - expected_trace)))
    if max(pressure_error, trace_error) > 1e-10:
        raise RuntimeError("the independently known affine Darcy field was not recovered")
    print(
        json.dumps(
            {
                "provider": args.provider,
                "backend": args.backend,
                "macrocells": len(solution.fields),
                "max_pressure_coefficient_error": pressure_error,
                "max_oriented_darcy_flux_error": trace_error,
                "original_equation_residual": solution.raw_residual,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
