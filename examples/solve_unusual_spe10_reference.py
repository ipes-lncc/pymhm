"""Compute an independently assembled conforming CG2 SPE10 diffusion--reaction reference.

The declared operator is ``-div(Kx grad p) + p = 0`` on the original
1200-by-2200 coordinate domain, with bottom/top values one/zero and homogeneous
natural side fluxes. The explicit zero source defines this control; it is not
an inference about an unspecified historical experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from pymhm.io.workspace import case_workspace, ensure_resource, local_resource, source_file

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/unusual-spe10"
LAYER = ROOT / "examples/results/spe10/layer-36.npz"


@dataclass(frozen=True)
class CG2Field:
    """Canonical P2 nodal coefficients, indexed ``[half-grid y, half-grid x]``.

    Every original rectangle is split southwest--northeast. At a shared edge,
    the evaluator chooses the lower triangle and the right/up rectangle. This
    affects broken gradients, not continuous scalar values. Integrated norms
    use strict-interior quadrature points, so no tie convention enters them.
    """

    coefficients: np.ndarray
    permeability: np.ndarray
    bounds: tuple[float, float, float, float] = (0.0, 1200.0, 0.0, 2200.0)
    x_axis: np.ndarray | None = None
    y_axis: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Reject malformed fields or a nonpositive Cartesian scalar coefficient."""
        values = np.asarray(self.coefficients)
        coefficient = np.asarray(self.permeability)
        if (
            values.ndim != 2
            or min(values.shape) < 3
            or any(size % 2 != 1 for size in values.shape)
            or not np.all(np.isfinite(values))
            or coefficient.ndim != 2
            or not coefficient.size
            or not np.all(np.isfinite(coefficient))
            or np.any(coefficient <= 0)
        ):
            raise ValueError("finite odd-sized P2 grid and positive 2D permeability are required")
        xmin, xmax, ymin, ymax = self.bounds
        if not np.all(np.isfinite(self.bounds)) or xmax <= xmin or ymax <= ymin:
            raise ValueError("bounds must define a finite positive rectangle")
        for name, count, low, high in (
            ("x_axis", self.shape[0], xmin, xmax),
            ("y_axis", self.shape[1], ymin, ymax),
        ):
            axis = getattr(self, name)
            axis = np.linspace(low, high, count + 1) if axis is None else np.asarray(axis)
            if (
                axis.shape != (count + 1,)
                or not np.all(np.isfinite(axis))
                or np.any(np.diff(axis) <= 0)
                or axis[0] != low
                or axis[-1] != high
            ):
                raise ValueError("coordinate axes must strictly increase across the exact bounds")
            object.__setattr__(self, name, axis)

    @property
    def shape(self) -> tuple[int, int]:
        """Return rectangle counts in x and y."""
        return (self.coefficients.shape[1] - 1) // 2, (self.coefficients.shape[0] - 1) // 2

    def material(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the original Cartesian coefficient, retaining its x/y array order."""
        xmin, xmax, ymin, ymax = self.bounds
        indices = np.floor(
            (np.asarray(points) - [xmin, ymin])
            / [xmax - xmin, ymax - ymin]
            * self.permeability.shape
        ).astype(int)
        indices = np.clip(indices, 0, np.array(self.permeability.shape) - 1)
        return self.permeability[indices[:, 0], indices[:, 1]]

    def evaluate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return scalar value, physical gradient and the broken flux ``-K grad p``."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("evaluation points must be a finite (n,2) array")
        nx, ny = self.shape
        xmin, xmax, ymin, ymax = self.bounds
        if np.any(points < np.array([xmin, ymin]) - 1e-10) or np.any(
            points > np.array([xmax, ymax]) + 1e-10
        ):
            raise ValueError("evaluation point lies outside the reference domain")
        origin = np.column_stack(
            [
                np.clip(np.searchsorted(axis, points[:, i], side="right") - 1, 0, count - 1)
                for i, (axis, count) in enumerate(((self.x_axis, nx), (self.y_axis, ny)))
            ]
        )
        corners = np.column_stack((self.x_axis[origin[:, 0]], self.y_axis[origin[:, 1]]))
        lengths = np.column_stack(
            (np.diff(self.x_axis)[origin[:, 0]], np.diff(self.y_axis)[origin[:, 1]])
        )
        x, y = ((points - corners) / lengths).T
        lower = y <= x
        return self._evaluate_coordinates(points, origin, lengths, x, y, lower)

    def evaluate_cells(
        self, points: np.ndarray, rectangles: np.ndarray, lower: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate exact one-sided traces in explicitly supplied incident triangles.

        ``rectangles[:, 0:2]`` contains x/y rectangle indices, while ``lower``
        selects the southwest--southeast--northeast triangle. Coefficients are
        evaluated at the original points, including endpoints; neither points
        nor gradients are averaged or shifted. The material comes from the
        stated rectangle interior, so both flux traces remain distinct at a
        material interface. Material interfaces must align with the grid.
        """
        points = np.asarray(points, dtype=float)
        rectangles, lower = np.asarray(rectangles), np.asarray(lower)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.isfinite(points).all()
            or rectangles.shape != points.shape
            or rectangles.dtype.kind not in "iu"
            or lower.shape != (len(points),)
            or lower.dtype.kind != "b"
            or np.any(rectangles < 0)
            or np.any(rectangles >= np.array(self.shape))
        ):
            raise ValueError(
                "explicit finite points, integer rectangle owners and boolean sides required"
            )
        corners = np.column_stack((self.x_axis[rectangles[:, 0]], self.y_axis[rectangles[:, 1]]))
        lengths = np.column_stack(
            (np.diff(self.x_axis)[rectangles[:, 0]], np.diff(self.y_axis)[rectangles[:, 1]])
        )
        x, y = ((points - corners) / lengths).T
        # Propagate physical-coordinate subtraction rounding to reference coordinates.
        bound = 8 * np.finfo(float).eps * (np.abs(points) + np.abs(corners)) / lengths
        if (
            np.any(x < -bound[:, 0])
            or np.any(x > 1 + bound[:, 0])
            or np.any(y < -bound[:, 1])
            or np.any(y > 1 + bound[:, 1])
            or np.any(np.where(lower, y - x, x - y) > bound.sum(axis=1))
        ):
            raise ValueError("points do not belong to their declared triangles")
        return self._evaluate_coordinates(
            points,
            rectangles,
            lengths,
            x,
            y,
            lower,
            material_values=self.material(corners + lengths / 2),
        )

    def _evaluate_coordinates(
        self,
        points: np.ndarray,
        origin: np.ndarray,
        lengths: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        lower: np.ndarray,
        material_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Share the same P2 polynomial arithmetic between bulk and one-sided evaluation."""
        bary = np.where(
            lower[:, None],
            np.column_stack((1 - x, x - y, y)),
            np.column_stack((1 - y, y - x, x)),
        )
        offsets = np.where(
            lower[:, None, None],
            [[0, 0], [2, 0], [2, 2], [1, 0], [2, 1], [1, 1]],
            [[0, 0], [0, 2], [2, 2], [0, 1], [1, 2], [1, 1]],
        )
        indices = 2 * origin[:, None] + offsets
        values = self.coefficients[indices[..., 1], indices[..., 0]]
        derivatives = (
            np.where(
                lower[:, None, None],
                [[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]],
                [[0.0, -1.0], [-1.0, 1.0], [1.0, 0.0]],
            )
            / lengths[:, None, :]
        )
        edges = ((0, 1), (1, 2), (2, 0))
        basis = np.column_stack(
            (bary * (2 * bary - 1), *[4 * bary[:, i] * bary[:, j] for i, j in edges])
        )
        gradient = np.concatenate(
            (
                (4 * bary - 1)[..., None] * derivatives,
                np.stack(
                    [
                        4
                        * (
                            bary[:, i, None] * derivatives[:, j]
                            + bary[:, j, None] * derivatives[:, i]
                        )
                        for i, j in edges
                    ],
                    axis=1,
                ),
            ),
            axis=1,
        )
        differences = values[:, 1:] - values[:, :1]
        value = values[:, 0] + np.einsum("qi,qi->q", basis[:, 1:], differences)
        derivative = np.einsum("qia,qi->qa", gradient[:, 1:], differences)
        material = self.material(points) if material_values is None else material_values
        return value, derivative, -material[:, None] * derivative

    @classmethod
    def load(cls, path: Path) -> CG2Field:
        """Read complete portable coefficients and their material from an archive."""
        with np.load(local_resource(path)) as archive:
            return cls(
                archive["coefficients"],
                archive["permeability"],
                tuple(archive["bounds"]),
                archive.get("x_axis", None),
                archive.get("y_axis", None),
            )


def integrated_difference(fine: CG2Field, coarse: CG2Field, order: int = 4) -> dict[str, float]:
    """Integrate nested P2 errors exactly by positive triangle quadrature.

    Denominators are the fine reference's norms. The energy contains both
    ``K |grad p|²`` and ``p²``. The permeability must align with both meshes.
    """
    from pymhm.fem.scalar.operators import triangle_quadrature

    nx, ny = fine.shape
    cx, cy = coarse.shape
    px, py = fine.permeability.shape
    if (
        fine.bounds != coarse.bounds
        or not np.array_equal(fine.permeability, coarse.permeability)
        or nx % cx
        or ny % cy
        or nx // cx != ny // cy
    ):
        raise ValueError("comparison requires nested, pixel-aligned meshes with identical material")
    for fine_axis, coarse_axis, pixels in (
        (fine.x_axis, coarse.x_axis, px),
        (fine.y_axis, coarse.y_axis, py),
    ):
        ratio = (len(fine_axis) - 1) // (len(coarse_axis) - 1)
        expected = subdivide_axis(coarse_axis, ratio)
        if not np.allclose(fine_axis, expected, rtol=0, atol=2e-12) or not axis_aligned(
            coarse_axis, pixels
        ):
            raise ValueError(
                "comparison requires nested, pixel-aligned meshes with identical material"
            )
    bary, weights = triangle_quadrature(order)
    vertices = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [0, 1], [1, 1]]])
    reference = np.einsum("qi,tia->tqa", bary, vertices).reshape(-1, 2)
    totals = np.zeros(6, dtype=np.longdouble)
    for start in range(0, nx * ny, 512):
        ids = np.arange(start, min(start + 512, nx * ny))
        origin = np.column_stack((fine.x_axis[ids % nx], fine.y_axis[ids // nx]))
        lengths = np.column_stack((np.diff(fine.x_axis)[ids % nx], np.diff(fine.y_axis)[ids // nx]))
        points = (origin[:, None] + reference * lengths[:, None]).reshape(-1, 2)
        weight = lengths.prod(axis=1)[:, None] * np.tile(weights, 2)[None, :] / 2
        u, gradient, flux = fine.evaluate(points)
        v, coarse_gradient, coarse_flux = coarse.evaluate(points)
        difference = u - v
        gradient_difference = gradient - coarse_gradient
        integrands = np.column_stack(
            (
                difference**2,
                np.sum((flux - coarse_flux) ** 2, axis=1),
                fine.material(points) * np.sum(gradient_difference**2, axis=1) + difference**2,
                u**2,
                np.sum(flux**2, axis=1),
                fine.material(points) * np.sum(gradient**2, axis=1) + u**2,
            )
        )
        totals += np.sum(
            integrands.reshape(len(ids), -1, 6) * weight[:, :, None],
            axis=(0, 1),
            dtype=np.longdouble,
        )
    norms = np.sqrt(totals)
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


def subdivide_axis(axis: np.ndarray, factor: int) -> np.ndarray:
    """Split every existing coordinate interval without changing its endpoints."""
    fractions = np.arange(factor) / factor
    return np.concatenate(
        ((axis[:-1, None] + np.diff(axis)[:, None] * fractions).ravel(), axis[-1:])
    )


def axis_aligned(axis: np.ndarray, pixels: int) -> bool:
    """Check that every material-interface coordinate is a mesh vertex coordinate."""
    target = np.linspace(axis[0], axis[-1], pixels + 1)
    index = np.minimum(np.searchsorted(axis, target), len(axis) - 1)
    return bool(
        np.all(
            np.minimum(abs(axis[index] - target), abs(axis[np.maximum(index - 1, 0)] - target))
            <= 2e-12
        )
    )


def graded_axes(factor: int, *, horizontal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the base reaction length while retaining every original material pixel.

    The first interval is sqrt(minimum bottom-layer K)/8; subsequent intervals
    grow by 1.2 up to y=10. The rest of the domain has two intervals per pixel.
    Optional horizontal grading resolves transitions beside every vertical
    material interface. Isotropic subdivision gives a nested triangular sequence.
    """
    with np.load(ensure_resource("examples/results/spe10/layer-36.npz", ROOT)) as archive:
        step = float(np.sqrt(archive["permeability"][:, 0, 0].min()) / 8)
    near = [0.0]
    while near[-1] + step < 10:
        near.append(near[-1] + step)
        step *= 1.2
    near.append(10.0)
    if horizontal:
        half = np.array([0.0, 0.02, 0.1, 0.5, 2.0, 5.0, 10.0])
        pixel = np.r_[half, 20 - half[-2::-1]]
        x = np.r_[(20 * np.arange(60)[:, None] + pixel[:-1]).ravel(), 1200.0]
        y = np.concatenate((near, np.arange(20.0, 2200.1, 10.0)))
    else:
        x = np.linspace(0, 1200, 121)
        y = np.concatenate((near, np.arange(15.0, 2200.1, 5.0)))
    return subdivide_axis(x, factor), subdivide_axis(y, factor)


def solve(
    nx: int,
    ny: int,
    *,
    threads: int = 4,
    patch: bool = False,
    axes: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[CG2Field, dict[str, Any]]:
    """Assemble the independent native CG2 system and retain its physical residual checks."""
    if min(nx, ny, threads) < 1 or (axes is None and not patch and (nx % 60 or ny % 220)):
        raise ValueError("positive mesh sizes/threads and exact SPE10 pixel alignment are required")
    if axes is None:
        axes = (np.linspace(0, 1200, nx + 1), np.linspace(0, 2200, ny + 1))
    if (
        len(axes[0]) != nx + 1
        or len(axes[1]) != ny + 1
        or any(np.any(np.diff(axis) <= 0) for axis in axes)
    ):
        raise ValueError("coordinate axes do not match the mesh shape")
    if not patch and not all(
        axis_aligned(axis, pixels) for axis, pixels in zip(axes, (60, 220), strict=True)
    ):
        raise ValueError("coordinate axes must retain every material pixel interface")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["BLIS_NUM_THREADS"] = str(threads)
    import basix
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    if not PETSc.Sys.hasExternalPackage("mumps"):
        raise RuntimeError("the independent CG2 reference requires native MUMPS")
    started = perf_counter()
    domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_SELF,
        np.array([[0.0, 0.0], [1200.0, 2200.0]]),
        [nx, ny],
        cell_type=dolfinx.mesh.CellType.triangle,
        diagonal=dolfinx.mesh.DiagonalType.right,
    )
    for i, (count, length) in enumerate(((nx, 1200), (ny, 2200))):
        indices = np.rint(domain.geometry.x[:, i] / length * count).astype(int)
        domain.geometry.x[:, i] = axes[i][indices]
    space = dolfinx.fem.functionspace(domain, ("Lagrange", 2))
    dg = dolfinx.fem.functionspace(domain, ("DG", 0))
    coefficient = dolfinx.fem.Function(dg)
    centers = dg.tabulate_dof_coordinates()[:, :2]
    if patch:
        material = np.full((1, 1), 2.0)
        coefficient.x.array[:] = 2.0
    else:
        with np.load(ensure_resource("examples/results/spe10/layer-36.npz", ROOT)) as archive:
            material = archive["permeability"][..., 0]
        pixel = np.floor(centers / [20.0, 10.0]).astype(int)
        coefficient.x.array[:] = material[pixel[:, 0], pixel[:, 1]]
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})
    original = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(
            (coefficient * ufl.inner(ufl.grad(trial), ufl.grad(test)) + trial * test) * dx
        )
    )
    original.assemble()
    rhs = original.createVecLeft()
    rhs.set(0)
    if patch:
        rhs.destroy()
        x = ufl.SpatialCoordinate(domain)
        rhs = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form((1 + x[1] / 2200) * test * dx))
    physical_rhs = rhs.array.copy()
    coords = space.tabulate_dof_coordinates()[:, :2]
    bottom = np.flatnonzero(np.isclose(coords[:, 1], 0, rtol=0, atol=1e-10))
    top = np.flatnonzero(np.isclose(coords[:, 1], 2200, rtol=0, atol=1e-10))
    fixed = np.concatenate((bottom, top)).astype(PETSc.IntType)
    boundary = original.createVecRight()
    boundary.set(0)
    boundary.array[bottom] = 1.0
    boundary.array[top] = 2.0 if patch else 0.0
    matrix = original.copy()
    matrix.zeroRowsColumns(fixed, diag=1.0, x=boundary, b=rhs)
    matrix.setOption(PETSc.Mat.Option.SPD, True)
    vector = matrix.createVecRight()
    ksp = PETSc.KSP().create(MPI.COMM_SELF)
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("cholesky")
    pc.setFactorSolverType("mumps")
    assembled = perf_counter()
    print(f"Assembled CG2 {nx}x{ny}: {matrix.getSize()[0]:,} DOFs", flush=True)
    with threadpool_limits(limits=threads):
        pools = threadpool_info()
        ksp.solve(rhs, vector)
    solved = perf_counter()
    if ksp.getConvergedReason() <= 0:
        raise ArithmeticError(f"native CG2 factorization failed: {ksp.getConvergedReason()}")
    residual = original.createVecLeft()
    original.mult(vector, residual)
    physical_residual = residual.array - physical_rhs
    free = np.ones(len(physical_residual), dtype=bool)
    free[fixed] = False
    relative = float(np.linalg.norm(physical_residual[free]) / np.linalg.norm(rhs.array[free]))
    if not np.isfinite(relative) or relative > 1e-10:
        raise ArithmeticError(f"physical free-equation residual exceeds 1e-10: {relative}")
    function = dolfinx.fem.Function(space)
    function.x.array[:] = vector.array
    mass = float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(function * dx)))
    energy = float(
        dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(
                (coefficient * ufl.inner(ufl.grad(function), ufl.grad(function)) + function**2) * dx
            )
        )
    )
    work = float(np.dot(vector.array, physical_rhs) + np.dot(boundary.array, physical_residual))
    balance = float(-physical_residual[fixed].sum() + mass - physical_rhs.sum())
    if abs(energy - work) > 1e-10 * max(abs(energy), abs(work)):
        raise ArithmeticError("discrete energy and boundary/source work disagree")
    indices = []
    for i, axis in enumerate(axes):
        nodes = subdivide_axis(axis, 2)
        right = np.minimum(np.searchsorted(nodes, coords[:, i]), len(nodes) - 1)
        left = np.maximum(right - 1, 0)
        index = np.where(
            abs(nodes[left] - coords[:, i]) < abs(nodes[right] - coords[:, i]), left, right
        )
        if np.max(abs(nodes[index] - coords[:, i])) > 2e-12:
            raise ArithmeticError("native P2 nodes do not match the canonical coordinate grid")
        indices.append(index)
    indices = np.column_stack(indices)
    canonical = np.empty((2 * ny + 1, 2 * nx + 1))
    canonical[indices[:, 1], indices[:, 0]] = vector.array
    field = CG2Field(canonical, material, x_axis=axes[0], y_axis=axes[1])
    cells = np.linspace(0, len(domain.geometry.dofmap) - 1, min(101, 2 * nx * ny), dtype=np.int32)
    samples = domain.geometry.x[domain.geometry.dofmap[cells]].mean(axis=1)
    np.testing.assert_allclose(
        field.evaluate(samples[:, :2])[0],
        function.eval(samples, cells).ravel(),
        atol=2e-12,
        rtol=2e-12,
    )
    reference_points = np.array([[0.17, 0.23], [0.41, 0.31]])
    expression = dolfinx.fem.Expression(ufl.grad(function), reference_points)
    native_gradient = expression.eval(domain, cells).reshape(-1, 2)
    vertices = domain.geometry.x[domain.geometry.dofmap[cells], :2]
    physical = np.einsum(
        "qi,tia->tqa",
        np.column_stack((1 - reference_points.sum(axis=1), reference_points)),
        vertices,
    ).reshape(-1, 2)
    np.testing.assert_allclose(field.evaluate(physical)[1], native_gradient, atol=2e-12, rtol=3e-11)
    report = dict(
        method="classical conforming triangular CG2; independently assembled DOLFINx/UFL",
        operator="-div(Kx_layer36 grad p) + p = 0",
        boundary="bottom p=1, top p=0, homogeneous outward flux on both vertical sides",
        source=0.0,
        reaction=1.0,
        mesh_shape=[nx, ny],
        triangles=2 * nx * ny,
        mesh_grading="explicit coordinate axes, with every material interface retained",
        minimum_interval=[float(np.diff(axis).min()) for axis in axes],
        unknowns=matrix.getSize()[0],
        triangle_diagonal="southwest to northeast",
        material="original layer-36 Kx used in both Cartesian directions, exact pixel alignment",
        physical_coordinate_unit="ft source numbers",
        permeability_unit="mD source numbers, no implicit SI conversion",
        native_backend="PETSc/MUMPS SPD Cholesky on COMM_SELF",
        quadrature_degree=4,
        original_free_residual_relative=relative,
        residual_tolerance=1e-10,
        reaction_integral=mass,
        energy_squared=energy,
        boundary_and_source_work=work,
        energy_identity_relative=abs(energy - work) / max(abs(energy), abs(work)),
        weak_outward_flux_bottom_top=[
            float(-physical_residual[bottom].sum()),
            float(-physical_residual[top].sum()),
        ],
        weak_global_balance=balance,
        conservation_scope=(
            "continuous CG2 weak equations and boundary reactions, "
            "not fine-cell H(div) conservation"
        ),
        value_min=float(canonical.min()),
        value_max=float(canonical.max()),
        assembly_seconds=assembled - started,
        factor_and_solve_seconds=solved - assembled,
        timing_scope="concurrent acquisition, not an exclusive performance benchmark",
        threads=threads,
        threadpools=[{k: v for k, v in pool.items() if k not in {"filepath"}} for pool in pools],
        dolfinx=dolfinx.__version__,
        basix=basix.__version__,
        petsc=PETSc.Sys.getVersion(),
        numpy=np.__version__,
        python=platform.python_version(),
        analytical_patch=patch,
        historical_scope=(
            "explicit zero-source diffusion-reaction control; "
            "no claim that an unspecified historical load was zero"
        ),
    )
    for obj in (ksp, residual, vector, rhs, matrix, boundary, original):
        obj.destroy()
    return field, report


def main() -> None:
    """Acquire three aligned references, complete coefficients and nested-mesh physical norms."""
    from examples.cg2_profiles import vertical_traces

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--graded", action="store_true")
    parser.add_argument(
        "--graded-x", action="store_true", help="also grade at vertical material interfaces"
    )
    args = parser.parse_args()
    if args.threads < 1 or not args.levels or any(level < 1 for level in args.levels):
        parser.error("threads and refinement levels must be positive")
    if args.levels != sorted(set(args.levels)):
        parser.error("levels must increase without repetition")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_bytes = Path(__file__).read_bytes()
    driver_sha = hashlib.sha256(source_bytes).hexdigest()
    profile_source = source_file("examples/cg2_profiles.py", root=ROOT)
    profile_bytes = profile_source.read_bytes()
    profile_sha = hashlib.sha256(profile_bytes).hexdigest()
    snapshot = ROOT / "build/results/unusual-spe10/acquisition-sources"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / f"{driver_sha}.py").write_bytes(source_bytes)
    (snapshot / f"{profile_sha}.py").write_bytes(profile_bytes)
    previous = None
    rows = []
    for level in args.levels:
        axes = (
            graded_axes(level, horizontal=args.graded_x) if args.graded or args.graded_x else None
        )
        nx, ny = (
            (len(axes[0]) - 1, len(axes[1]) - 1) if axes is not None else (60 * level, 220 * level)
        )
        field, report = solve(nx, ny, threads=args.threads, axes=axes)
        if previous is not None:
            differences = [integrated_difference(field, previous, order) for order in (3, 4)]
            change = max(
                abs(differences[0][k] - differences[1][k]) / max(abs(differences[1][k]), 1e-300)
                for k in differences[0]
            )
            if change > 1e-9:
                raise ArithmeticError("nested reference norms are not quadrature resolved")
            report.update(
                refinement=differences[1],
                refinement_orders=[3, 4],
                refinement_quadrature_relative_change=change,
            )
        points = np.array(
            [(x, y) for x in np.arange(10, 1200, 20) for y in np.arange(5, 2200, 10)], dtype=float
        )
        values, _, flux = field.evaluate(points)
        profile = np.column_stack((np.full(4401, 33.0), np.linspace(0, 2200, 4401)))
        profile_values, _, profile_flux = field.evaluate(profile)
        traces = vertical_traces(field, 33.0)
        prefix = (
            "classical-cg2-graded-xy"
            if args.graded_x
            else "classical-cg2-graded"
            if args.graded
            else "classical-cg2"
        )
        path = OUTPUT / f"{prefix}-{nx}x{ny}.npz"
        np.savez_compressed(
            path,
            coefficients=field.coefficients,
            permeability=field.permeability,
            bounds=field.bounds,
            x_axis=field.x_axis,
            y_axis=field.y_axis,
            points=points.reshape(60, 220, 2),
            pressure=values.reshape(60, 220),
            flux=flux.reshape(60, 220, 2),
            profile_points=profile,
            profile_pressure=profile_values,
            profile_flux=profile_flux,
            **{f"profile_trace_{name}": value for name, value in traces.items()},
        )
        if (
            Path(__file__).read_bytes() != source_bytes
            or profile_source.read_bytes() != profile_bytes
        ):
            raise RuntimeError("acquisition driver changed during execution")
        report.update(
            archive=path.name,
            archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            driver_sha256=driver_sha,
            sampling_source_hashes={"examples/cg2_profiles.py": profile_sha},
            material_sha256=hashlib.sha256(LAYER.read_bytes()).hexdigest(),
            profile_convention=(
                "profile_trace arrays retain both endpoint traces in explicitly owned triangles; "
                "fixed-point arrays select the lower triangle and right/up rectangle"
            ),
        )
        path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        rows.append(report)
        aggregate = (
            "classical-graded-xy-convergence.json"
            if args.graded_x
            else "classical-graded-convergence.json"
            if args.graded
            else "classical-convergence.json"
        )
        (OUTPUT / aggregate).write_text(
            json.dumps(
                dict(
                    rows=rows,
                    norm_denominator="finer numerical reference",
                    comparison="independently assembled CG2 for the explicitly declared PDE",
                ),
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(report), flush=True)
        previous = field


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_unusual_spe10_reference").main()
