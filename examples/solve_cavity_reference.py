"""Independent Taylor--Hood refinement for regularized and constant lid-driven flow."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from solve_spe10_taylor_hood import TaylorHoodField, difference

ROOT = Path(__file__).resolve().parents[1]


def solve(n: int, drag: float, lid: str = "regularized") -> tuple[TaylorHoodField, dict[str, Any]]:
    """Solve vector-Laplacian Brinkman with strong velocity data and mean-zero pressure."""
    import basix.ufl
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    started = perf_counter()
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_SELF,
        n,
        n,
        cell_type=dolfinx.mesh.CellType.triangle,
        diagonal=dolfinx.mesh.DiagonalType.right,
    )
    element = basix.ufl.mixed_element(
        [
            basix.ufl.element("Lagrange", "triangle", 2, shape=(2,)),
            basix.ufl.element("Lagrange", "triangle", 1),
        ]
    )
    mixed_space = dolfinx.fem.functionspace(domain, element)
    velocity_space, velocity_map = mixed_space.sub(0).collapse()
    pressure_space, pressure_map = mixed_space.sub(1).collapse()
    velocity_map, pressure_map = np.asarray(velocity_map), np.asarray(pressure_map)
    u, p = ufl.TrialFunctions(mixed_space)
    v, q = ufl.TestFunctions(mixed_space)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})
    form = dolfinx.fem.form(
        (
            ufl.inner(ufl.grad(u), ufl.grad(v))
            + drag * ufl.inner(u, v)
            - p * ufl.div(v)
            - q * ufl.div(u)
        )
        * dx
    )
    coordinates = velocity_space.tabulate_dof_coordinates()[:, :2]
    boundary = np.flatnonzero(
        np.any(
            np.isclose(coordinates, 0.0, rtol=0.0, atol=1e-12)
            | np.isclose(coordinates, 1.0, rtol=0.0, atol=1e-12),
            axis=1,
        )
    )
    top = np.flatnonzero(np.isclose(coordinates[:, 1], 1.0, rtol=0.0, atol=1e-12))
    if len(boundary) != 8 * n or len(top) != 2 * n + 1:
        raise RuntimeError("the complete P2 exterior velocity trace must be constrained")
    velocity_fixed = velocity_map[(2 * boundary[:, None] + [0, 1]).ravel()]
    fixed = np.r_[velocity_fixed, pressure_map[0]].astype(PETSc.IntType)
    with ExitStack() as resources:

        def own(obj: Any) -> Any:
            """Release every PETSc object even when a numerical check fails."""
            resources.callback(obj.destroy)
            return obj

        original = own(dolfinx.fem.petsc.assemble_matrix(form))
        original.assemble()
        matrix = own(original.copy())
        rhs, prescribed = own(original.createVecLeft()), own(original.createVecRight())
        rhs.set(0)
        prescribed.set(0)
        x = coordinates[top, 0]
        if lid == "regularized":
            prescribed.array[velocity_map[2 * top]] = 16 * x**2 * (1 - x) ** 2
        elif lid == "constant":
            prescribed.array[velocity_map[2 * top]] = (x > 1e-12) & (x < 1 - 1e-12)
        else:
            raise ValueError("lid must be regularized or constant")
        matrix.zeroRowsColumns(fixed, diag=1.0, x=prescribed, b=rhs)
        ksp = own(PETSc.KSP().create(MPI.COMM_SELF))
        ksp.setOperators(matrix)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        vector = own(matrix.createVecRight())
        ksp.solve(rhs, vector)
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError(f"MUMPS solve failed: {ksp.getConvergedReason()}")
        field = dolfinx.fem.Function(mixed_space)
        field.x.array[:] = vector.array
        uh, ph = field.split()
        mean = float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ph * dx)))
        vector.array[pressure_map] -= mean
        field.x.array[:] = vector.array
        residual = own(original.createVecLeft())
        original.mult(vector, residual)
        free = np.ones(len(vector.array), dtype=bool)
        free[velocity_fixed] = False
        relative = float(np.linalg.norm(residual.array[free]) / np.linalg.norm(rhs.array[free]))
        if relative > 1e-10 or not np.isfinite(relative):
            raise RuntimeError(f"original physical-equation residual {relative} exceeds 1e-10")
        divergence = float(
            np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.div(uh) ** 2 * dx)))
        )
        pressure_mean = float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ph * dx)))
        nodal_u = np.empty((2 * n + 1, 2 * n + 1, 2))
        indices = np.rint(coordinates * (2 * n)).astype(int)
        nodal_u[indices[:, 1], indices[:, 0]] = vector.array[velocity_map].reshape(-1, 2)
        nodal_p = np.empty((n + 1, n + 1))
        indices = np.rint(pressure_space.tabulate_dof_coordinates()[:, :2] * n).astype(int)
        nodal_p[indices[:, 1], indices[:, 0]] = vector.array[pressure_map]
        canonical = TaylorHoodField(nodal_u, nodal_p, (0.0, 1.0, 0.0, 1.0))
        cells = np.arange(min(127, len(domain.geometry.dofmap)), dtype=np.int32)
        samples = domain.geometry.x[domain.geometry.dofmap[cells]].mean(axis=1)
        evaluated = canonical.evaluate(samples[:, :2])
        np.testing.assert_allclose(evaluated[0], uh.eval(samples, cells), atol=1e-11)
        np.testing.assert_allclose(evaluated[1], ph.eval(samples, cells).ravel(), atol=1e-9)
        expression = dolfinx.fem.Expression(ufl.grad(uh), np.array([[1 / 3, 1 / 3]]))
        gradients = expression.eval(domain, cells).reshape(-1, 2, 2)
        np.testing.assert_allclose(canonical.gradient(samples[:, :2])[0], gradients, atol=2e-10)
        report = dict(
            n=n,
            drag=drag,
            viscosity=1.0,
            method="conforming P2/P1 Taylor-Hood",
            backend=f"DOLFINx {dolfinx.__version__}; PETSc/MUMPS",
            triangles=2 * n * n,
            total_dofs=matrix.getSize()[0],
            free_dofs=int(free.sum()) - 1,
            pressure_gauge="zero volume mean",
            lid=lid,
            corner_nodal_values="zero, continuous P2 interpolation on the boundary",
            mesh_diagonal="SW-NE",
            residual=relative,
            pressure_mean=pressure_mean,
            divergence_l2=divergence,
            elapsed_seconds=perf_counter() - started,
        )
    return canonical, report


def main() -> None:
    """Archive canonical fields, provenance and independent nested refinement differences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--drag", type=float, nargs="+", default=[0.0, 1e4])
    parser.add_argument("--lid", choices=["regularized", "constant"], default="regularized")
    args = parser.parse_args()
    output = ROOT / "examples/results/stokes-adaptive"
    coefficients = ROOT / "build/results/stokes-adaptive"
    output.mkdir(parents=True, exist_ok=True)
    coefficients.mkdir(parents=True, exist_ok=True)
    for drag in args.drag:
        rows, previous = [], None
        prefix = "cavity-constant" if args.lid == "constant" else "cavity"
        for n in args.levels:
            stem = f"{prefix}-classical-gamma{drag:g}-n{n}"
            path, report_path = coefficients / f"{stem}.npz", output / f"{stem}.json"
            if path.exists() and report_path.exists():
                report = json.loads(report_path.read_text())
                if report["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise RuntimeError("classical coefficient archive checksum mismatch")
                with np.load(path) as data:
                    field = TaylorHoodField(
                        data["velocity"], data["pressure"], tuple(data["bounds"])
                    )
            else:
                field, report = solve(n, drag, args.lid)
                np.savez_compressed(
                    path,
                    velocity=field.velocity,
                    pressure=field.pressure,
                    bounds=np.array(field.bounds),
                )
                report.update(
                    archive=path.relative_to(ROOT).as_posix(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            if previous is not None:
                report["difference_from_previous"] = difference(field, previous, gradients=True)
                if args.lid == "constant":
                    report["corner_cutout"] = 1 / 32
                    report["interior_difference_from_previous"] = difference(
                        field, previous, gradients=True, corner_cutout=1 / 32
                    )
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            rows.append(report)
            previous = field
            print(report, flush=True)
            (output / f"{prefix}-classical-gamma{drag:g}.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
