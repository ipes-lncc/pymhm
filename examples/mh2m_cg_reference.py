"""Independent continuous P3 reference for the oscillatory three-field example.

The structured mesh splits each square southwest--northeast. DOLFINx/UFL
assembles and solves the classical problem; archived coefficients are physical
nodal values on its equispaced lattice. This does not change the P1 reference
resolution used in the article's printed figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm.io.workspace import case_workspace

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/mh2m-heterogeneous/cg3"
LATTICE = np.array(
    [
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [1, 1, 1],
    ]
)
TRIANGLES = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]])
OFFSETS = np.einsum("ij,kja->kia", LATTICE, TRIANGLES)
BARYCENTRIC_GRADIENTS = np.array(
    [[[-1, 0], [1, -1], [0, 1]], [[0, -1], [1, 0], [-1, 1]]], dtype=float
)


def cubic_basis(barycentric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the ten explicit cubic cardinals and their barycentric derivatives."""
    b = np.asarray(barycentric)
    values = np.empty((len(b), 10))
    gradients = np.zeros((len(b), 10, 3))
    for i in range(3):
        t = b[:, i]
        values[:, i] = 0.5 * t * (3 * t - 1) * (3 * t - 2)
        gradients[:, i, i] = 0.5 * (27 * t**2 - 18 * t + 2)
    for node, (a, c) in enumerate(((0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)), 3):
        x, y = b[:, a], b[:, c]
        values[:, node] = 4.5 * x * y * (3 * x - 1)
        gradients[:, node, a] = 4.5 * y * (6 * x - 1)
        gradients[:, node, c] = 4.5 * x * (3 * x - 1)
    values[:, 9] = 27 * np.prod(b, axis=1)
    for i in range(3):
        gradients[:, 9, i] = 27 * np.prod(np.delete(b, i, axis=1), axis=1)
    return values, gradients


@dataclass(frozen=True)
class CubicTriangularField:
    """Continuous P3 coefficients on the physical lattice, indexed ``[iy, ix]``.

    Fine-cell gradients are evaluated separately. On horizontal grid lines,
    ``y_side`` selects the incident cell without perturbing the coordinates.
    """

    pressure: np.ndarray

    def __post_init__(self) -> None:
        """Require one complete finite real equispaced triangular P3 lattice."""
        p = np.asarray(self.pressure)
        if (
            p.ndim != 2
            or p.shape[0] != p.shape[1]
            or p.shape[0] < 4
            or (p.shape[0] - 1) % 3
            or np.iscomplexobj(p)
            or not np.isfinite(p).all()
        ):
            raise ValueError("a finite real square P3 nodal lattice is required")

    @property
    def resolution(self) -> int:
        """Return the number of Cartesian fine squares in each direction."""
        return (self.pressure.shape[0] - 1) // 3

    def evaluate(self, points: np.ndarray, *, y_side: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate pressure and physical gradient using explicit cubic polynomials."""
        xy = np.asarray(points)
        if (
            y_side not in (-1, 1)
            or xy.ndim != 2
            or xy.shape[1] != 2
            or np.iscomplexobj(xy)
            or not np.isfinite(xy).all()
            or np.any(xy < 0)
            or np.any(xy > 1)
        ):
            raise ValueError("finite unit-square points and incident side -1 or 1 are required")
        n = self.resolution
        scaled = xy * n
        index = np.floor(scaled).astype(int)
        if y_side == -1:
            index[scaled[:, 1] == np.rint(scaled[:, 1]), 1] -= 1
        index = np.clip(index, 0, n - 1)
        local = scaled - index
        half = (local[:, 1] > local[:, 0]).astype(int)
        b = np.where(
            (half == 0)[:, None],
            np.column_stack((1 - local[:, 0], local[:, 0] - local[:, 1], local[:, 1])),
            np.column_stack((1 - local[:, 1], local[:, 0], local[:, 1] - local[:, 0])),
        )
        nodes = 3 * index[:, None, :] + OFFSETS[half]
        coefficients = self.pressure[nodes[:, :, 1], nodes[:, :, 0]]
        values, derivative = cubic_basis(b)
        # Taking nodal differences preserves the constant mode in both fields.
        delta = coefficients[:, 1:] - coefficients[:, :1]
        pressure = coefficients[:, 0] + np.einsum("qi,qi->q", values[:, 1:], delta)
        gradient = n * np.einsum(
            "qib,qba,qi->qa", derivative[:, 1:], BARYCENTRIC_GRADIENTS[half], delta
        )
        return pressure, gradient

    @classmethod
    def load(cls, path: Path) -> CubicTriangularField:
        """Restore the declared lattice without importing a native FEM runtime."""
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["degree"]) != 3 or str(archive["diagonal"]) != "SW-NE":
                raise ValueError("archive is not the declared SW-NE cubic field")
            return cls(archive["pressure"])


def coefficient(points: np.ndarray) -> np.ndarray:
    """Evaluate the specified gamma=1.8, epsilon=1/14 scalar coefficient."""
    x, y = 28 * np.pi * np.asarray(points).T
    return (2 + 1.8 * np.sin(x)) / (2 + 1.8 * np.cos(y)) + (2 + np.sin(y)) / (2 + 1.8 * np.sin(x))


def patch_pressure(points: np.ndarray) -> np.ndarray:
    """Return a represented cubic with nonhomogeneous boundary values."""
    x, y = np.asarray(points).T
    return 1 + x * x + x * y * y + y**3


def patch_gradient(points: np.ndarray) -> np.ndarray:
    """Differentiate the independent cubic patch analytically."""
    x, y = np.asarray(points).T
    return np.column_stack((2 * x + y * y, 2 * x * y + 3 * y * y))


def solve(
    n: int, order: int, *, patch: bool = False
) -> tuple[CubicTriangularField, dict[str, Any]]:
    """Assemble continuous P3 with native UFL, solve by MUMPS and verify field replay."""
    import basix
    import basix.ufl
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc
    from scipy.sparse import csr_matrix

    if (
        isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or n < 1
        or isinstance(order, bool)
        or not isinstance(order, (int, np.integer))
        or order < 6
    ):
        raise ValueError("positive mesh resolution and quadrature degree at least six required")
    started = perf_counter()
    grid = np.linspace(0, 1, n + 1)
    xg, yg = np.meshgrid(grid, grid, indexing="xy")
    points = np.column_stack((xg.ravel(), yg.ravel()))
    ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    sw = (iy * (n + 1) + ix).ravel()
    cells = np.stack(
        (np.column_stack((sw, sw + 1, sw + n + 2)), np.column_stack((sw, sw + n + 2, sw + n + 1))),
        axis=1,
    ).reshape(-1, 3)
    geometry = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, cells, x=points, e=geometry)
    element = basix.ufl.element(
        "Lagrange", "triangle", 3, lagrange_variant=basix.LagrangeVariant.equispaced
    )
    space = dolfinx.fem.functionspace(domain, element)
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        domain, 1, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = dolfinx.fem.locate_dofs_topological(space, 1, boundary_facets)
    boundary = dolfinx.fem.Function(space)
    if patch:
        boundary.interpolate(lambda x: 1 + x[0] ** 2 + x[0] * x[1] ** 2 + x[1] ** 3)
    bc = dolfinx.fem.dirichletbc(boundary, boundary_dofs)
    x = ufl.SpatialCoordinate(domain)
    material = (
        1.0
        if patch
        else (2 + 1.8 * ufl.sin(28 * ufl.pi * x[0])) / (2 + 1.8 * ufl.cos(28 * ufl.pi * x[1]))
        + (2 + ufl.sin(28 * ufl.pi * x[1])) / (2 + 1.8 * ufl.sin(28 * ufl.pi * x[0]))
    )
    source = -(2 + 2 * x[0] + 6 * x[1]) if patch else -2 * x[0] * (x[0] - 1) - 2 * x[1] * (x[1] - 1)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": order})
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    bilinear = dolfinx.fem.form(material * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
    linear = dolfinx.fem.form(source * v * dx)
    matrix = dolfinx.fem.petsc.assemble_matrix(bilinear, bcs=[bc])
    matrix.assemble()
    rhs = dolfinx.fem.petsc.assemble_vector(linear)
    dolfinx.fem.petsc.apply_lifting(rhs, [bilinear], bcs=[[bc]])
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(rhs, [bc])
    solution = dolfinx.fem.Function(space)
    solver = PETSc.KSP().create(MPI.COMM_SELF)
    residual = rhs.duplicate()
    correction = rhs.duplicate()
    try:
        matrix.setOption(PETSc.Mat.Option.SPD, True)
        solver.setOperators(matrix)
        solver.setType("preonly")
        solver.getPC().setType("cholesky")
        solver.getPC().setFactorSolverType("mumps")
        solver.setErrorIfNotConverged(True)
        solver.solve(rhs, solution.x.petsc_vec)
        solution.x.scatter_forward()
        indptr, indices, entries = matrix.getValuesCSR()
        extended = csr_matrix(
            (entries.astype(np.longdouble), indices, indptr), shape=matrix.getSize()
        )
        extended_rhs = np.asarray(rhs.getArray(readonly=True), dtype=np.longdouble)
        rhs_norm = np.sqrt(np.sum(extended_rhs**2))
        refinement_residuals = []
        for step in range(3):
            defect = extended_rhs - extended @ solution.x.array.astype(np.longdouble)
            relative_residual = float(np.sqrt(np.sum(defect**2)) / rhs_norm)
            refinement_residuals.append(relative_residual)
            if step == 2 or relative_residual < 1e-12:
                break
            residual.getArray()[:] = np.asarray(defect, dtype=float)
            solver.solve(residual, correction)
            solution.x.petsc_vec.axpy(1.0, correction)
            solution.x.scatter_forward()
        if relative_residual > 1e-10:
            raise ValueError("independent P3 original-equation residual exceeds 1e-10")
        matrix.mult(solution.x.petsc_vec, residual)
        residual.axpy(-1, rhs)
        binary64_residual = residual.norm() / rhs.norm()
        del extended, entries, extended_rhs, defect
        coordinates = space.tabulate_dof_coordinates()[:, :2]
        keys = np.rint(3 * n * coordinates).astype(int)
        flat = keys[:, 0] + (3 * n + 1) * keys[:, 1]
        if (
            len(np.unique(flat)) != (3 * n + 1) ** 2
            or np.max(abs(keys / (3 * n) - coordinates)) > 1e-13
        ):
            raise ValueError("native P3 coordinates do not form the declared physical lattice")
        pressure = np.empty((3 * n + 1) ** 2)
        pressure[flat] = solution.x.array
        field = CubicTriangularField(pressure.reshape(3 * n + 1, 3 * n + 1))
        sample_cells = np.linspace(0, len(cells) - 1, min(len(cells), 113), dtype=np.int32)
        barycentric = np.array([[0.19, 0.27, 0.54], [0.31, 0.47, 0.22]])
        vertices = domain.geometry.x[domain.geometry.dofmap[sample_cells], :2]
        sample_points = np.einsum("qi,tia->tqa", barycentric, vertices).reshape(-1, 2)
        native_values = solution.eval(
            np.column_stack((sample_points, np.zeros(len(sample_points)))),
            np.repeat(sample_cells, 2),
        ).ravel()
        expression = dolfinx.fem.Expression(ufl.grad(solution), barycentric[:, 1:])
        native_gradients = expression.eval(domain, sample_cells).reshape(-1, 2)
        values, gradients = field.evaluate(sample_points)
        replay_value = float(np.max(abs(values - native_values)))
        replay_gradient = float(np.max(abs(gradients - native_gradients)))
        if replay_value > 2e-12 or replay_gradient > 2e-10:
            raise ValueError("archived physical P3 field differs from native evaluation")
        energy = float(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(material * ufl.inner(ufl.grad(solution), ufl.grad(solution)) * dx)
            )
        )
        work = float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(source * solution * dx)))
        energy_defect = abs(energy - work) / max(abs(energy), abs(work)) if not patch else None
        if energy_defect is not None and energy_defect > 1e-8:
            raise ValueError("physical energy and source work do not agree")
        material_difference = 0.0
        if not patch:
            native_material = dolfinx.fem.Expression(material, barycentric[:, 1:])
            material_difference = float(
                np.max(
                    abs(
                        native_material.eval(domain, sample_cells).ravel()
                        - coefficient(sample_points)
                    )
                )
            )
            if material_difference > 2e-11:
                raise ValueError("native and archived-reader material conventions differ")
        report = dict(
            resolution=n,
            degree=3,
            dofs=len(pressure),
            triangles=2 * n * n,
            quadrature_degree=order,
            relative_equation_residual=relative_residual,
            residual_accumulation="extended CSR action on the stored binary64 coefficients",
            binary64_matrix_action_residual=binary64_residual,
            refinement_residuals=refinement_residuals,
            native_replay_pressure_linf=replay_value,
            native_replay_gradient_linf=replay_gradient,
            energy=energy,
            source_work=work,
            homogeneous_dirichlet=not patch,
            relative_energy_work_defect=energy_defect,
            native_material_difference_linf=material_difference,
            solver="PETSc/MUMPS Cholesky",
            versions=dict(
                dolfinx=dolfinx.__version__, basix=basix.__version__, petsc=PETSc.Sys.getVersion()
            ),
            elapsed_seconds=perf_counter() - started,
        )
        if patch:
            report["patch_pressure_linf"] = float(
                np.max(abs(values - patch_pressure(sample_points)))
            )
            report["patch_gradient_linf"] = float(
                np.max(abs(gradients - patch_gradient(sample_points)))
            )
        return field, report
    finally:
        correction.destroy()
        residual.destroy()
        solver.destroy()
        rhs.destroy()
        matrix.destroy()


def acquire(n: int, order: int, output: Path) -> dict[str, Any]:
    """Archive a current independent reference with immutable source and field digests."""
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    field, report = solve(n, order)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_digest:
        raise RuntimeError("reference acquisition source changed during the solve")
    output.mkdir(parents=True, exist_ok=True)
    name = f"reference-cg3-n{n}-q{order}"
    archive = output / (name + ".npz")
    np.savez_compressed(archive, pressure=field.pressure, degree=3, diagonal="SW-NE")
    report.update(
        method="Classical conforming P3, independently assembled by UFL/DOLFINx",
        gamma=1.8,
        epsilon="1/14",
        source="-2*x*(x-1)-2*y*(y-1)",
        boundary="homogeneous Dirichlet",
        diagonal="SW-NE",
        archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        acquisition_source_sha256=source_digest,
        source_changed_during_run=False,
    )
    (output / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Acquire requested reference resolutions with an explicitly fixed native thread count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(1):
        for n in args.sizes:
            print(json.dumps(acquire(n, args.order, args.output)), flush=True)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.mh2m_cg_reference").main()
