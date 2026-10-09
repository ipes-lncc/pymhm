"""Native conforming acoustic reference with exactly represented Cartesian materials.

The real symmetric form changes only the sign of the imaginary test function.
It is equivalent to the complex Helmholtz equation with outgoing impedance.
All material interfaces must be mesh edges; no cell averaging is permitted.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.hpc4e_parallel import (
    checked_solve,
    compact_matrix,
    original_residual,
    symmetric_equilibration,
)
from examples.marmousi_data import load_marmousi_crop
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from pymhm.materials.cartesian import CartesianCellField


def validate_material_grid(
    counts: tuple[int, int], bounds: tuple[float, float, float, float], material: Any
) -> None:
    """Require every material interface to be represented by Cartesian mesh edges."""
    if not isinstance(material, CartesianCellField):
        return
    widths = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2]]) / counts
    ratios = np.asarray(material.spacing) / widths
    start = (np.array([bounds[0], bounds[2]]) - material.origin) / widths
    if not np.allclose(np.r_[ratios, start], np.rint(np.r_[ratios, start]), rtol=0, atol=1e-10):
        raise ValueError("reference mesh must resolve every material pixel without averaging")
    if np.any(ratios < 1):
        raise ValueError("reference cells cannot be larger than the material pixels")


def native_reference(
    counts: tuple[int, int],
    *,
    degree: int,
    bounds: tuple[float, float, float, float],
    omega: float,
    density: Any,
    bulk_modulus: Any,
    point_sources: tuple[tuple[float, float, float], ...] = (),
    volume_source: Any = 0j,
    absorbing_load: Any = 0j,
    impedance_pressure: Any = 0j,
    dirichlet: Any = 0j,
    comm: Any = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Assemble and solve a native CG reference with top Dirichlet and other-side absorption.

    Volume and absorbing loads are interpolated in the selected continuous
    space; the Marmousi calculation uses zero for both and an exact nodal Dirac.
    An additional prescribed impedance pressure contributes
    -i*omega/sqrt(rho*kappa) times that pressure to the absorbing load, allowing
    material jumps to remain exact on the boundary. Verification loads below
    are representable polynomials. Return owned nodal
    coordinates and complex coefficients, plus independent native field norms.
    PETSc matrices, vectors, factors and generated options are explicitly freed.
    """
    import basix
    import basix.ufl
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    comm = MPI.COMM_WORLD if comm is None else comm

    def report(stage: str, **values: Any) -> None:
        """Report completed native stages on rank zero during large acquisitions."""
        if progress and comm.rank == 0:
            print(json.dumps({"stage": stage, **values}), flush=True)

    for material in (density, bulk_modulus):
        validate_material_grid(counts, bounds, material)
    start = time.perf_counter()
    mesh = dolfinx.mesh.create_rectangle(
        comm,
        [np.array([bounds[0], bounds[2]]), np.array([bounds[1], bounds[3]])],
        counts,
        cell_type=dolfinx.mesh.CellType.triangle,
        diagonal=dolfinx.mesh.DiagonalType.right,
    )
    scalar = basix.ufl.element(
        "Lagrange", "triangle", degree, lagrange_variant=basix.LagrangeVariant.equispaced
    )
    space = dolfinx.fem.functionspace(mesh, basix.ufl.blocked_element(scalar, shape=(2,)))
    report("mesh-and-space", complex_dofs=space.dofmap.index_map.size_global)
    dg = dolfinx.fem.functionspace(mesh, ("DG", 0))
    materials = []
    for value in (density, bulk_modulus):
        function = dolfinx.fem.Function(dg)
        function.interpolate(
            lambda x, value=value: (
                np.asarray(value(x[:2].T) if callable(value) else value) + np.zeros(x.shape[1])
            )
        )
        if not np.all(np.isfinite(function.x.array) & (function.x.array > 0)):
            raise ValueError("acoustic coefficients must be finite and positive")
        materials.append(function)
    rho, modulus = materials

    def vector_data(value: Any) -> Any:
        """Interpolate real/imaginary coordinates without changing their physical signs."""
        function = dolfinx.fem.Function(space)

        def values(x: np.ndarray) -> np.ndarray:
            """Evaluate the declared scalar complex data at interpolation points."""
            result = np.asarray(value(x[:2].T) if callable(value) else value, dtype=complex)
            result = result + np.zeros(x.shape[1])
            return np.array([result.real, result.imag])

        function.interpolate(values)
        return function

    forcing_field, boundary_field, impedance_field, imposed_pressure = map(
        vector_data, (volume_source, dirichlet, absorbing_load, impedance_pressure)
    )
    top = dolfinx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: np.isclose(x[1], bounds[2], rtol=0, atol=1e-10)
    )
    exterior = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    absorbing = np.setdiff1d(exterior, top)
    facets = np.r_[top, absorbing].astype(np.int32)
    tags = np.r_[np.ones(len(top)), 2 * np.ones(len(absorbing))].astype(np.int32)
    order = np.argsort(facets)
    markers = dolfinx.mesh.meshtags(mesh, 1, facets[order], tags[order])
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 2 * degree + 2})
    ds = ufl.Measure(
        "ds", domain=mesh, subdomain_data=markers, metadata={"quadrature_degree": 2 * degree + 2}
    )
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    a = (
        (ufl.inner(ufl.grad(u[0]), ufl.grad(v[0])) - ufl.inner(ufl.grad(u[1]), ufl.grad(v[1])))
        / rho
        - omega**2 / modulus * (u[0] * v[0] - u[1] * v[1])
    ) * dx + omega / ufl.sqrt(rho * modulus) * (u[1] * v[0] + u[0] * v[1]) * ds(2)
    load = (
        (forcing_field[0] * v[0] - forcing_field[1] * v[1]) * dx
        + (impedance_field[0] * v[0] - impedance_field[1] * v[1]) * ds(2)
        + omega
        / ufl.sqrt(rho * modulus)
        * (imposed_pressure[1] * v[0] + imposed_pressure[0] * v[1])
        * ds(2)
    )
    with ExitStack() as resources:
        original = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
        resources.callback(original.destroy)
        original.assemble()
        original, storage = compact_matrix(original, resources)
        report("native-assembly", **storage)
        forcing = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(load))
        resources.callback(forcing.destroy)
        forcing.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        owned = space.dofmap.index_map.size_local
        coordinates = space.tabulate_dof_coordinates()[:owned, :2]
        for x, y, strength in point_sources:
            matches = np.flatnonzero(np.max(abs(coordinates - [x, y]), axis=1) <= 1e-10)
            if comm.allreduce(len(matches), op=MPI.SUM) != 1:
                raise ValueError("each Dirac point must be one unique global nodal coordinate")
            if len(matches):
                forcing.array[2 * matches[0]] += strength
        matrix = original.copy()
        resources.callback(matrix.destroy)
        rhs = original.createVecRight()
        resources.callback(rhs.destroy)
        rhs.array[:] = forcing.array[: 2 * owned]
        boundary = original.createVecRight()
        resources.callback(boundary.destroy)
        boundary.array[:] = boundary_field.x.array[: 2 * owned]
        top_blocks = dolfinx.fem.locate_dofs_topological(space, 1, top)
        top_blocks = top_blocks[top_blocks < owned]
        prescribed = (2 * top_blocks[:, None] + np.arange(2)).ravel().astype(np.int32)
        global_blocks = space.dofmap.index_map.local_to_global(top_blocks)
        global_rows = (2 * global_blocks[:, None] + np.arange(2)).ravel().astype(PETSc.IntType)
        matrix.zeroRowsColumns(global_rows, diag=1.0, x=boundary, b=rhs)
        matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        matrix.setOption(PETSc.Mat.Option.SPD, False)
        diagonal = symmetric_equilibration(matrix, resources)
        ksp = PETSc.KSP().create(comm)
        resources.callback(ksp.destroy)
        ksp.setOperators(matrix)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("cholesky")
        pc.setFactorSolverType("mumps")
        ksp.setUp()
        report("native-factorization", elapsed_seconds=time.perf_counter() - start)
        state, history = checked_solve(
            ksp,
            original,
            rhs,
            forcing,
            prescribed,
            boundary,
            diagonal,
            resources,
            refinement_precision="extended",
            rtol=1e-10,
        )
        defect = original_residual(original, forcing.array[: 2 * owned], state.array)
        defect[prescribed] = boundary.array[prescribed] - state.array[prescribed]
        squared = comm.allreduce(np.sum(defect**2, dtype=np.longdouble), op=MPI.SUM)
        final_residual = (
            float(np.sqrt(squared)) / rhs.norm() if rhs.norm() else float(np.sqrt(squared))
        )
        if final_residual > 1e-10:
            raise ValueError("stored reference coefficients fail the original physical residual")
        field = dolfinx.fem.Function(space)
        field.x.array[: 2 * owned] = state.array
        field.x.scatter_forward()
        norm = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(field, field) * dx)),
            op=MPI.SUM,
        )
        return {
            "coordinates": coordinates.copy(),
            "pressure": field.x.array[: 2 * owned : 2].copy()
            + 1j * field.x.array[1 : 2 * owned : 2],
            "complex_dofs": space.dofmap.index_map.size_global,
            "residual_history": history,
            "stored_original_residual": final_residual,
            "pressure_l2": float(np.sqrt(norm)),
            "elapsed_seconds": time.perf_counter() - start,
            "storage": storage,
            "native_versions": {
                "dolfinx": dolfinx.__version__,
                "basix": basix.__version__,
                "ufl": ufl.__version__,
                "petsc": PETSc.Sys.getVersion(),
            },
        }


def main() -> None:
    """Acquire one pixel-conforming polynomial level with MPI native assembly."""
    from mpi4py import MPI

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--degree", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--output", type=Path, default=Path("examples/results/marmousi"))
    args = parser.parse_args()
    material = load_marmousi_crop(args.data)
    comm = MPI.COMM_WORLD
    root = case_workspace()
    sources = [
        Path(__file__),
        source_file("examples/marmousi_data.py", root=root),
        source_file("examples/hpc4e_parallel.py", root=root),
        source_file("examples/campaign_provenance.py", root=root),
        *sorted(source_file("src/pymhm/__init__.py", root=root).parent.rglob("*.py")),
    ]
    before = current_source_manifest(source_identity(root, sources), packages=("pymhm", "examples"))
    with threadpool_limits(1):
        result = native_reference(
            (2048, 512),
            degree=args.degree,
            bounds=(0, 10240, 0, 2560),
            omega=40 * np.pi,
            density=material.density,
            bulk_modulus=material.bulk_modulus,
            point_sources=((5000, 50, 1.0),),
            progress=True,
        )
    after = current_source_manifest(source_identity(root, sources), packages=("pymhm", "examples"))
    if comm.allreduce(before != after, op=MPI.LOR):
        raise RuntimeError("reference acquisition sources changed during execution")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"classical-p{args.degree}-rank{comm.rank}.npz"
    np.savez_compressed(
        archive, coordinates=result.pop("coordinates"), pressure=result.pop("pressure")
    )
    archives = comm.gather(
        {"archive": archive.name, "sha256": file_digest(archive)},
        root=0,
    )
    if comm.rank == 0:
        result.update(
            degree=args.degree,
            method="DOLFINx/UFL conforming triangular CG",
            geometry=[2048, 512],
            bounds=[0, 10240, 0, 2560],
            boundary=(
                "Top conforming Dirichlet zero; other sides outgoing first-order absorption zero"
            ),
            material=material.provenance,
            omega=40 * np.pi,
            point_source=[5000, 50, 1.0],
            mpi_ranks=comm.size,
            archives=archives,
            source_sha256=before,
            source_changed_during_run=False,
            field_coordinates="Owned continuous equispaced Pk nodal values on SW-NE triangles",
            solver="PETSc/MUMPS symmetric-indefinite LDLt; five Ruiz congruences",
        )
        (args.output / f"classical-p{args.degree}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.marmousi_reference").main()
