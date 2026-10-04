"""Solve the same affine Darcy hybrid form with portable or native UFL assembly.

Run as a module, for example ``python -m examples.variational_darcy --backend
process --workers 2 --batch-size 1 --provider fenics`` in the Pixi fem environment.
The two macrotriangles have independent local meshes and signed shared faces.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm.assembly import HybridProblem, assemble_hybrid
from pymhm.elements import boundary_data, face_integration, p1_operators
from pymhm.hybrid import HybridSolution, HybridSystem, LocalAssembly, LocalProblem
from pymhm.mesh import FloatArray, SkeletonSpace, TriangleMesh, positive_int
from pymhm.parallel import ExecutionConfig
from pymhm.variational import GlobalForm


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
    only copied coefficient arrays and point coordinates are returned.
    """

    macro_mesh: TriangleMesh
    skeleton: SkeletonSpace
    kind: Literal["portable", "fenics"] = "portable"
    subdivisions: int = 2

    def __post_init__(self) -> None:
        """Validate the assembly choice and local refinement parameter."""
        if self.kind not in ("portable", "fenics"):
            raise ValueError("kind must be portable or fenics")
        positive_int(self.subdivisions, "subdivisions")

    def __call__(self, cell: int) -> LocalAssembly:
        """Assemble one cell and return numerical reconstruction metadata."""
        fine = self.macro_mesh.submesh(cell, self.subdivisions)
        if self.kind == "fenics":
            return _fenics_local(self.macro_mesh, self.skeleton, fine, cell)
        matrix, mass, _ = p1_operators(fine)
        coupling, _ = face_integration(self.macro_mesh, cell, fine, self.skeleton)
        problem = LocalProblem(
            matrix,
            coupling,
            np.zeros(len(fine.points)),
            self.skeleton.cell_dofs(cell),
            kernel=np.ones((len(fine.points), 1)),
            constraints=np.asarray(mass.sum(axis=1)).reshape(-1, 1),
        )
        return LocalAssembly(
            problem, {"cell": cell, "points": fine.points.copy(), "assembly_pid": os.getpid()}
        )


def _fenics_local(
    coarse: TriangleMesh, skeleton: SkeletonSpace, fine: TriangleMesh, cell: int
) -> LocalAssembly:
    """Own native local mesh/forms within this invocation and return only arrays."""
    import basix.ufl
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    from pymhm.fenics import assemble_local_forms, primal_darcy_forms
    from pymhm.variational import LocalForm

    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    local_mesh = mesh.create_mesh(MPI.COMM_SELF, fine.cells, fine.points, domain)
    space = fem.functionspace(local_mesh, ("Lagrange", 1))
    test = ufl.TestFunction(space)
    a, load = primal_darcy_forms(space, permeability=1.0, source=0.0)
    entities, tags = [], []
    for side, face in enumerate(coarse.cell_faces[cell]):
        start, end = coarse.points[coarse.faces[face]]
        tangent = end - start

        def on_face(points: Any, start: Any = start, tangent: Any = tangent) -> Any:
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
    traces = tuple(float(coarse.signs[cell, side]) * test * ds(side + 1) for side in range(3))
    points = np.array(space.tabulate_dof_coordinates()[:, :2], copy=True)
    forms = LocalForm(
        a,
        load,
        traces,
        skeleton.cell_dofs(cell),
        kernel=np.ones((len(points), 1)),
        moment_forms=(test * ufl.dx(domain=local_mesh),),
    )
    problem = assemble_local_forms(forms)
    return LocalAssembly(problem, {"cell": cell, "points": points, "assembly_pid": os.getpid()})


def build_problem(
    *, provider: Literal["portable", "fenics"] = "portable", subdivisions: int = 2
) -> HybridProblem[int]:
    """Declare the local Darcy provider and its common global hybrid form."""
    coarse = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(coarse)
    local = DarcyProvider(coarse, skeleton, provider, subdivisions)
    boundary, _ = boundary_data(skeleton, affine_pressure)
    form = GlobalForm(skeleton.size, (1,) * len(coarse.cells), boundary_load=boundary)
    return HybridProblem(form, local, range(len(coarse.cells)))


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
    system = assemble_hybrid(problem, execution=execution)
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
