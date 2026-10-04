"""Independent material-fitted DOLFINx CG2 reference for square-annulus inclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import __version__ as scipy_version
from scipy import sparse
from threadpoolctl import threadpool_limits

from examples.cg2_physical_form import CG2DiffusionForm
from examples.compensated_reference import component_residual, refine_components
from examples.inclusion_grading import graded_axis
from examples.pgmhm_inclusion_data import axis, coefficient, material_array
from examples.solve_unusual_spe10_reference import CG2Field, subdivide_axis
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.io.provenance import current_source_manifest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/pgmhm-inclusions"


@dataclass(frozen=True)
class InclusionField(CG2Field):
    """Evaluate principal and correction polynomials before combining physical fields."""

    correction: np.ndarray | None = None
    _correction_field: CG2Field = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate both canonical P2 arrays on the same physical geometry/material."""
        super().__post_init__()
        correction = self.correction
        if correction is None:
            correction = np.zeros_like(self.coefficients)
        if np.asarray(correction).shape != self.coefficients.shape:
            raise ValueError("correction coefficients must have the principal shape")
        object.__setattr__(self, "correction", correction)
        object.__setattr__(
            self,
            "_correction_field",
            CG2Field(
                np.asarray(correction, dtype=np.longdouble),
                self.permeability,
                self.bounds,
                self.x_axis,
                self.y_axis,
            ),
        )

    def evaluate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate each component's pressure, gradient and flux, then add in wide arithmetic."""
        principal = super().evaluate(points)
        correction = self._correction_field.evaluate(points)
        return tuple(a + b for a, b in zip(principal, correction, strict=True))

    def evaluate_cells(
        self, points: np.ndarray, rectangles: np.ndarray, lower: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retain principal/correction polynomials and explicit incident sides in traces."""
        principal = super().evaluate_cells(points, rectangles, lower)
        correction = self._correction_field.evaluate_cells(points, rectangles, lower)
        return tuple(a + b for a, b in zip(principal, correction, strict=True))


def load_field(path: Path) -> InclusionField:
    """Keep both portable components until after their independent P2 evaluations."""
    with np.load(path) as data:
        correction = data["coefficients_correction"].astype(np.longdouble)
        correction += data["coefficients_tail"]
        return InclusionField(
            data["coefficients"].astype(np.longdouble),
            data["permeability"],
            tuple(data["bounds"]),
            data["x_axis"],
            data["y_axis"],
            correction,
        )


def physical_diagnostics(field: InclusionField, source: float = 1.0) -> dict[str, float]:
    """Check persisted components against the unassembled physical weak form.

    This application has Dirichlet conditions on the complete exterior. Boundary
    work uses that trace; the free residual is normalized by the physical source
    load. With zero source, use the physical load induced by the prescribed
    trace. Neither case uses the rounded sparse factorization's RHS.
    """
    x = (field.x_axis[:-1] + field.x_axis[1:]) / 2
    y = (field.y_axis[:-1] + field.y_axis[1:]) / 2
    xx, yy = np.meshgrid(x, y)
    k = field.material(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    form = CG2DiffusionForm(field.x_axis, field.y_axis, k, source)
    action = form.action(field.coefficients, field.correction)
    load = form.load()
    defect = action - load
    value = field.coefficients.astype(np.longdouble) + field.correction
    energy = np.sum(value * action, dtype=np.longdouble)
    boundary_work = np.sum(value * defect, dtype=np.longdouble) - np.sum(
        value[1:-1, 1:-1] * defect[1:-1, 1:-1], dtype=np.longdouble
    )
    work = np.sum(value * load, dtype=np.longdouble) + boundary_work
    scale = np.linalg.norm(load[1:-1, 1:-1])
    if scale == 0:
        prescribed = np.array(field.coefficients, copy=True)
        correction = np.array(field.correction, copy=True)
        prescribed[1:-1, 1:-1] = 0
        correction[1:-1, 1:-1] = 0
        scale = np.linalg.norm(form.action(prescribed, correction)[1:-1, 1:-1])
    return dict(
        residual=float(np.linalg.norm(defect[1:-1, 1:-1]) / max(scale, np.finfo(float).tiny)),
        energy_squared=float(energy),
        source_boundary_work=float(work),
        relative_energy_work_defect=float(
            abs(energy - work) / max(abs(energy), abs(work), np.finfo(float).tiny)
        ),
        outward_boundary_flux=float(-np.sum(defect) + np.sum(defect[1:-1, 1:-1])),
        source_integral=float(np.sum(load)),
    )


def difference(fine: CG2Field, coarse: CG2Field, order: int) -> dict[str, float]:
    """Integrate nested-grid pressure, physical flux and diffusion-energy increments."""
    nx, ny = fine.shape
    old_nx, old_ny = coarse.shape
    if nx % old_nx or ny % old_ny or nx // old_nx != ny // old_ny:
        raise ValueError("difference requires isotropically nested triangular grids")
    factor = nx // old_nx
    for current, previous in ((fine.x_axis, coarse.x_axis), (fine.y_axis, coarse.y_axis)):
        expected = subdivide_axis(previous, factor)
        tolerance = 16 * np.finfo(float).eps * max(1.0, float(np.max(abs(expected))))
        if not np.allclose(current, expected, rtol=0, atol=tolerance):
            raise ValueError("difference requires coincident nested physical axes")
    bary, weight = triangle_quadrature(order)
    sums = np.zeros(6, dtype=np.longdouble)
    for first in range(0, nx * ny, 512):
        ids = np.arange(first, min(first + 512, nx * ny))
        i, j = ids % nx, ids // nx
        corners = np.stack(
            (
                np.column_stack((fine.x_axis[i], fine.y_axis[j])),
                np.column_stack((fine.x_axis[i + 1], fine.y_axis[j])),
                np.column_stack((fine.x_axis[i + 1], fine.y_axis[j + 1])),
                np.column_stack((fine.x_axis[i], fine.y_axis[j + 1])),
            ),
            axis=1,
        )
        for local in ((0, 1, 2), (0, 2, 3)):
            vertices = corners[:, local]
            points = np.einsum("qi,tia->tqa", bary, vertices).reshape(-1, 2)
            area = np.diff(fine.x_axis)[i] * np.diff(fine.y_axis)[j] / 2
            p, gradient, flux = fine.evaluate(points)
            old_p, old_gradient, old_flux = coarse.evaluate(points)
            k = fine.material(points)
            values = np.column_stack(
                (
                    (p - old_p) ** 2,
                    np.sum((flux - old_flux) ** 2, axis=1),
                    k * np.sum((gradient - old_gradient) ** 2, axis=1),
                    p**2,
                    np.sum(flux**2, axis=1),
                    k * np.sum(gradient**2, axis=1),
                )
            )
            sums += np.sum(
                (area[:, None] * weight).ravel()[:, None] * values, axis=0, dtype=np.longdouble
            )
    norms = np.sqrt(sums)
    return {
        key: float(value)
        for name, error, norm in zip(
            ("pressure", "flux", "energy"), norms[:3], norms[3:], strict=True
        )
        for key, value in (
            (f"{name}_absolute", error),
            (f"{name}_reference", norm),
            (f"{name}_relative", error / norm),
        )
    }


def solve(factor: int, *, patch: bool = False, graded: bool = False) -> tuple[CG2Field, dict]:
    """Assemble native CG2 forms, impose exterior values and check original equations."""
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    if patch:
        coordinates = (
            np.array([0.0, 0.01, 0.07, 0.5, 0.93, 1.0]) if graded else np.linspace(0, 1, 5)
        )
    else:
        coordinates = graded_axis(factor) if graded else axis(factor)
    n = len(coordinates) - 1
    started = perf_counter()
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_SELF,
        n,
        n,
        cell_type=dolfinx.mesh.CellType.triangle,
        diagonal=dolfinx.mesh.DiagonalType.right,
    )
    domain.geometry.x[:, :2] = coordinates[np.rint(domain.geometry.x[:, :2] * n).astype(int)]
    space = dolfinx.fem.functionspace(domain, ("Lagrange", 2))
    dg = dolfinx.fem.functionspace(domain, ("DG", 0))
    material = dolfinx.fem.Function(dg)
    material.x.array[:] = 2.0 if patch else coefficient(dg.tabulate_dof_coordinates()[:, :2])
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})
    original = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(material * ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx)
    )
    original.assemble()
    rhs = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form((4.0 if patch else 1.0) * test * dx))
    physical_rhs = rhs.array.copy()
    nodes = space.tabulate_dof_coordinates()[:, :2]
    fixed = np.flatnonzero(np.any((nodes < 1e-13) | (nodes > 1 - 1e-13), axis=1)).astype(
        PETSc.IntType
    )
    boundary = original.createVecRight()
    boundary.set(0)
    if patch:
        boundary.array[fixed] = nodes[fixed, 0] * (1 - nodes[fixed, 0])
    matrix = original.copy()
    matrix.zeroRowsColumns(fixed, diag=1, x=boundary, b=rhs)
    matrix.setOption(PETSc.Mat.Option.SPD, True)
    vector = matrix.createVecRight()
    ksp = PETSc.KSP().create(MPI.COMM_SELF)
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("cholesky")
    pc.setFactorSolverType("mumps")
    effective_rhs = rhs.array.copy()
    indptr, columns, entries = matrix.getValuesCSR()
    csr = sparse.csr_matrix((entries, columns, indptr), shape=matrix.getSize())

    def native_solve(forcing: np.ndarray) -> np.ndarray:
        """Reuse native Cholesky for both the physical RHS and residual corrections."""
        rhs.array[:] = forcing
        ksp.solve(rhs, vector)
        if ksp.getConvergedReason() <= 0:
            raise ArithmeticError("native MUMPS factorization failed")
        return vector.array.copy()

    grid = subdivide_axis(coordinates, 2)
    indices = np.searchsorted(grid, nodes)
    indices = np.minimum(indices, len(grid) - 1)
    left = np.maximum(indices - 1, 0)
    indices = np.where(abs(grid[left] - nodes) < abs(grid[indices] - nodes), left, indices)
    np.testing.assert_allclose(grid[indices], nodes, atol=1e-13, rtol=0)
    canonical = (indices[:, 1], indices[:, 0])
    if len(np.unique(indices[:, 1] * len(grid) + indices[:, 0])) != len(grid) ** 2:
        raise ArithmeticError("native CG2 nodes do not bijectively cover the canonical grid")
    centers = (coordinates[:-1] + coordinates[1:]) / 2
    xx, yy = np.meshgrid(centers, centers)
    material_values = np.full((n, n), 2.0) if patch else coefficient(np.stack((xx, yy), axis=-1))
    form = CG2DiffusionForm(coordinates, coordinates, material_values, 4.0 if patch else 1.0)
    load = form.load()[canonical]
    free = np.ones(len(nodes), dtype=bool)
    free[fixed] = False
    target_scale = float(np.linalg.norm(load[free]))
    canonical_high = np.empty(form.shape)
    canonical_low = np.empty(form.shape)

    def physical_residual(high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Evaluate unassembled physical free equations and the exact boundary values."""
        canonical_high[canonical], canonical_low[canonical] = high, low
        residual = load - form.action(canonical_high, canonical_low)[canonical]
        residual[fixed] = (boundary.array[fixed].astype(np.longdouble) - high[fixed]) - low[fixed]
        return residual

    principal, correction, refinement_history = refine_components(
        csr, effective_rhs, native_solve, residual=physical_residual, residual_scale=target_scale
    )
    solution = principal.astype(np.longdouble) + correction
    vector.array[:] = solution
    indptr, columns, entries = original.getValuesCSR()
    physical = sparse.csr_matrix((entries, columns, indptr), shape=original.getSize())
    csr_defect = -component_residual(physical, physical_rhs, principal, correction)
    csr_relative = float(np.linalg.norm(csr_defect[free]) / np.linalg.norm(physical_rhs[free]))
    canonical_high[canonical], canonical_low[canonical] = principal, correction
    action = form.action(canonical_high, canonical_low)[canonical]
    defect = action - load
    relative = float(np.linalg.norm(defect[free]) / target_scale)
    if relative > 1e-10:
        raise ArithmeticError(f"physical free equations failed: {relative}")
    energy = float(np.dot(solution, action))
    work = float(np.dot(solution, load) + np.dot(boundary.array, defect))
    if abs(energy - work) > 1e-10 * max(abs(energy), abs(work)):
        raise ArithmeticError("physical energy differs from source and boundary work")
    values = np.empty((2 * n + 1, 2 * n + 1), dtype=solution.dtype)
    values[indices[:, 1], indices[:, 0]] = solution
    correction_values = np.empty(values.shape)
    correction_values[indices[:, 1], indices[:, 0]] = correction
    principal_values = np.empty(values.shape, dtype=np.longdouble)
    principal_values[indices[:, 1], indices[:, 0]] = principal
    field = InclusionField(
        principal_values,
        np.full((1, 1), 2.0) if patch else material_array(),
        (0, 1, 0, 1),
        coordinates,
        coordinates,
        correction_values,
    )
    function = dolfinx.fem.Function(space)
    function.x.array[:] = vector.array
    cells = np.linspace(0, 2 * n * n - 1, min(97, 2 * n * n), dtype=np.int32)
    points = domain.geometry.x[domain.geometry.dofmap[cells]].mean(axis=1)
    np.testing.assert_allclose(
        field.evaluate(points[:, :2])[0],
        function.eval(points, cells).ravel(),
        atol=1e-12,
        rtol=1e-12,
    )
    if patch:
        np.testing.assert_allclose(
            vector.array, nodes[:, 0] * (1 - nodes[:, 0]), atol=2e-13, rtol=0
        )
    report = dict(
        method="DOLFINx/UFL conforming CG2",
        factor=factor,
        grading_level=factor if graded else None,
        mesh_family="interface-graded" if graded else "piecewise-uniform",
        grading=("1:2:4:8:4:2:1 per material interval, then uniform bisection" if graded else None),
        rectangles=[n, n],
        triangles=2 * n * n,
        dofs=len(nodes),
        residual=relative,
        csr_residual=csr_relative,
        residual_target="physical CG2 weak form by exact element gradient moments",
        source_integral=float(load.sum()),
        outward_boundary_flux=float(-defect[fixed].sum()),
        energy_squared=energy,
        source_boundary_work=work,
        pressure_min=float(values.min()),
        pressure_max=float(values.max()),
        seconds=perf_counter() - started,
        dolfinx=dolfinx.__version__,
        petsc=list(PETSc.Sys.getVersion()),
        numpy=np.__version__,
        scipy=scipy_version,
        solver=(
            "PETSc/MUMPS unchanged-CSR Cholesky preconditioner; physical-form residual corrections"
        ),
        analytical_patch=patch,
        refinement_precision="two binary64 coefficient components; extended element accumulation",
        refinement_history=refinement_history,
        archive_encoding="float64 principal + correction; zero tail",
    )
    for obj in (ksp, vector, matrix, boundary, rhs, original):
        obj.destroy()
    return field, report


def main() -> None:
    """Acquire independent references and nested polynomial norm checks outside CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factors",
        nargs="+",
        type=int,
        help="uniform subdivision factors (default1,2,4) or graded levels (default1,2)",
    )
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--graded", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.factors is None:
        args.factors = [1, 2] if args.graded else [1, 2, 4]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "examples/solve_pgmhm_inclusions_reference.py",
        "examples/pgmhm_inclusion_data.py",
        "examples/inclusion_grading.py",
        "examples/solve_unusual_spe10_reference.py",
        "examples/compensated_reference.py",
        "examples/cg2_physical_form.py",
        "src/pymhm/fem/scalar/operators.py",
    )
    hashes = current_source_manifest(
        {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    )
    with threadpool_limits(1):
        for factor in args.factors:
            field, row = solve(factor, patch=args.patch, graded=args.graded)
            if args.patch:
                print(json.dumps(row), flush=True)
                return
            prefix = "classical-cg2-graded" if args.graded else "classical-cg2-factor"
            path = output / f"{prefix}{factor}.npz"
            np.savez_compressed(
                path,
                coefficients=np.asarray(field.coefficients, dtype=float),
                coefficients_correction=field.correction,
                coefficients_tail=np.zeros_like(field.correction),
                permeability=field.permeability,
                bounds=field.bounds,
                x_axis=field.x_axis,
                y_axis=field.y_axis,
            )
            replay = load_field(path)
            replay_check = physical_diagnostics(replay)
            if (
                replay_check["residual"] > 1e-10
                or replay_check["relative_energy_work_defect"] > 1e-10
            ):
                raise ArithmeticError("archived components failed the physical weak-form checks")
            sample = np.random.default_rng(417).uniform(size=(97, 2))
            for saved, original_values in zip(
                replay.evaluate(sample), field.evaluate(sample), strict=True
            ):
                np.testing.assert_array_equal(saved, original_values)
            row.update(
                archive=path.name,
                archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_hashes=hashes,
                archive_physical_verification=replay_check,
                archive_field_replay_bitwise=True,
            )
            previous_level = factor - 1 if args.graded else factor // 2
            previous = output / f"{prefix}{previous_level}.npz"
            if previous.exists():
                old = load_field(previous)
                row["norms"] = [
                    dict(order=order, **difference(field, old, order)) for order in (3, 4)
                ]
                row["previous_archive"] = previous.name
                row["previous_sha256"] = hashlib.sha256(previous.read_bytes()).hexdigest()
            if hashes != current_source_manifest(
                {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
            ):
                raise RuntimeError("reference sources changed during acquisition")
            path.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
