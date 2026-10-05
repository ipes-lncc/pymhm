"""Compute a global conforming Taylor--Hood reference for the SPE10 Brinkman case.

Use the ``fem`` environment. This independently assembled P2/P1 discretization
has continuous velocity and pressure, no skeletal unknowns and no residual
stabilization. Pixel-aligned meshes preserve the original piecewise coefficient.
The top condition is zero grad-grad traction, not prescribed pressure.
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

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/spe10"
COEFFICIENTS = ROOT / "build/results/spe10/taylor-hood"


@dataclass(frozen=True)
class TaylorHoodField:
    """Canonical nodal arrays on a rectangular grid split along SW--NE diagonals."""

    velocity: np.ndarray
    pressure: np.ndarray
    bounds: tuple[float, float, float, float] = (0.0, 1200.0, 0.0, 2200.0)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the number of original rectangles in x and y."""
        return self.pressure.shape[1] - 1, self.pressure.shape[0] - 1

    def _reference_data(self, points: np.ndarray) -> tuple:
        """Locate the structured triangle and its physical barycentric derivatives."""
        points = np.asarray(points, dtype=float)
        nx, ny = self.shape
        xmin, xmax, ymin, ymax = self.bounds
        lengths = np.array([(xmax - xmin) / nx, (ymax - ymin) / ny])
        scaled = (points - [xmin, ymin]) / lengths
        if np.any(scaled < -1e-10) or np.any(scaled > np.array([nx, ny]) + 1e-10):
            raise ValueError("evaluation point lies outside the reference domain")
        origin = np.minimum(np.maximum(np.floor(scaled).astype(int), 0), [nx - 1, ny - 1])
        x, y = (scaled - origin).T
        lower = y <= x
        bary = np.where(
            lower[:, None],
            np.column_stack((1 - x, x - y, y)),
            np.column_stack((1 - y, y - x, x)),
        )
        below = np.array([[0, 0], [2, 0], [2, 2], [1, 0], [2, 1], [1, 1]])
        above = np.array([[0, 0], [0, 2], [2, 2], [0, 1], [1, 2], [1, 1]])
        offsets = np.where(lower[:, None, None], below, above)
        indices = 2 * origin[:, None] + offsets
        vertices = origin[:, None] + offsets[:, :3] // 2
        gradient = (
            np.where(
                lower[:, None, None],
                np.array([[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]]),
                np.array([[0.0, -1.0], [-1.0, 1.0], [1.0, 0.0]]),
            )
            / lengths
        )
        return bary, gradient, indices, vertices

    def evaluate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the actual P2 velocity and P1 pressure without nodal smoothing."""
        bary, _, indices, vertices = self._reference_data(points)
        values = self.velocity[indices[..., 1], indices[..., 0]]
        basis = np.column_stack(
            (
                bary * (2 * bary - 1),
                4 * bary[:, 0] * bary[:, 1],
                4 * bary[:, 1] * bary[:, 2],
                4 * bary[:, 2] * bary[:, 0],
            )
        )
        pressure = self.pressure[vertices[..., 1], vertices[..., 0]]
        return np.einsum("qi,qia->qa", basis, values), np.einsum("qi,qi->q", bary, pressure)

    def gradient(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the physical velocity Jacobian and pressure gradient per triangle."""
        bary, gradient, indices, vertices = self._reference_data(points)
        basis = np.concatenate(
            (
                (4 * bary - 1)[..., None] * gradient,
                np.stack(
                    [
                        4 * (bary[:, i, None] * gradient[:, j] + bary[:, j, None] * gradient[:, i])
                        for i, j in ((0, 1), (1, 2), (2, 0))
                    ],
                    axis=1,
                ),
            ),
            axis=1,
        )
        velocity = self.velocity[indices[..., 1], indices[..., 0]]
        pressure = self.pressure[vertices[..., 1], vertices[..., 0]]
        return np.einsum("qia,qic->qca", basis, velocity), np.einsum(
            "qia,qi->qa", gradient, pressure
        )


def load_field(path: str | Path) -> TaylorHoodField:
    """Reload canonical coefficients independently of DOLFINx numbering."""
    with np.load(path) as data:
        bounds = tuple(data["bounds"]) if "bounds" in data else (0.0, 1200.0, 0.0, 2200.0)
        return TaylorHoodField(data["velocity"], data["pressure"], bounds)


def load_archived_reference(
    shape: tuple[int, int],
) -> tuple[TaylorHoodField, dict[str, Any]]:
    """Verify archived metadata and coefficients before reusing a reference.

    Both numerical archives must match the recorded checksums. The requested
    grid must agree with the metadata and with the P2/P1 coefficient dimensions.
    """
    nx, ny = shape
    stem = f"taylor-hood-{nx}x{ny}"
    report = json.loads((OUTPUT / f"{stem}.json").read_text())
    if report.get("mesh_shape") != [nx, ny] or report.get("archive") != f"{stem}.npz":
        raise ValueError("reference archive metadata does not match the requested shape")
    coefficient_path = COEFFICIENTS / f"{stem}.npz"
    for path, key in (
        (coefficient_path, "coefficient_sha256"),
        (OUTPUT / f"{stem}.npz", "sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != report.get(key):
            raise ValueError(f"reference archive checksum mismatch: {path.name}")
    field = load_field(coefficient_path)
    if field.velocity.shape != (2 * ny + 1, 2 * nx + 1, 2) or field.pressure.shape != (
        ny + 1,
        nx + 1,
    ):
        raise ValueError("reference coefficient array shape does not match the requested grid")
    if not np.all(np.isfinite(field.velocity)) or not np.all(np.isfinite(field.pressure)):
        raise ValueError("reference coefficient archive must contain finite values")
    return field, report


def difference(
    fine: TaylorHoodField,
    coarse: TaylorHoodField,
    *,
    gradients: bool = False,
    corner_cutout: float = 0.0,
) -> dict[str, float]:
    """Integrate nested-grid differences, optionally excluding aligned top-corner boxes.

    ``corner_cutout`` is a fraction of each side length. Nonzero cutouts must
    align with fine-grid lines so that quadrature covers the exact retained domain.
    """
    from pymhm.fem.scalar.operators import triangle_quadrature

    nx, ny = fine.shape
    if fine.bounds != coarse.bounds:
        raise ValueError("reference comparisons require identical physical bounds")
    if not 0 <= corner_cutout < 0.5:
        raise ValueError("corner_cutout must lie in [0, 0.5)")
    cut = corner_cutout * np.array([nx, ny])
    if not np.allclose(cut, np.rint(cut), rtol=0, atol=32 * np.finfo(float).eps):
        raise ValueError("corner boxes must align with fine reference grid lines")
    cut = np.rint(cut).astype(int)
    xmin, xmax, ymin, ymax = fine.bounds
    sizes = np.array([(xmax - xmin) / nx, (ymax - ymin) / ny])
    if (
        nx % coarse.shape[0]
        or ny % coarse.shape[1]
        or nx // coarse.shape[0] != ny // coarse.shape[1]
    ):
        raise ValueError("convergence comparison requires nested aligned reference grids")
    bary, weights = triangle_quadrature(4)
    vertices = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [0, 1], [1, 1]]])
    reference = np.einsum("qi,tia->tqa", bary, vertices).reshape(-1, 2)
    weight = np.tile(weights, 2) * sizes.prod() / 2
    totals = np.zeros(6 if gradients else 4)
    for start in range(0, nx * ny, 2048):
        ids = np.arange(start, min(start + 2048, nx * ny))
        origins = np.column_stack((ids % nx, ids // nx))
        if corner_cutout:
            retained = (origins[:, 1] < ny - cut[1]) | (
                (origins[:, 0] >= cut[0]) & (origins[:, 0] < nx - cut[0])
            )
            origins = origins[retained]
            if not len(origins):
                continue
        points = ((origins[:, None] + reference) * sizes + [xmin, ymin]).reshape(-1, 2)
        u, p = fine.evaluate(points)
        v, q = coarse.evaluate(points)
        integrands = np.column_stack(
            ((u - v) ** 2 @ np.ones(2), (p - q) ** 2, u**2 @ np.ones(2), p**2)
        )
        if gradients:
            grad_u, grad_v = fine.gradient(points)[0], coarse.gradient(points)[0]
            integrands = np.column_stack(
                (
                    integrands,
                    np.sum((grad_u - grad_v) ** 2, axis=(1, 2)),
                    np.sum(grad_u**2, axis=(1, 2)),
                )
            )
        totals += np.einsum("q,cqi->i", weight, integrands.reshape(len(origins), -1, len(totals)))
    norms = np.sqrt(totals)

    def relative(error: float, reference: float) -> float:
        """Keep a nonzero error against the zero field distinct from agreement."""
        return float(error / reference) if reference else (float("inf") if error else 0.0)

    result = dict(
        velocity_l2=float(norms[0]),
        pressure_l2=float(norms[1]),
        velocity_relative=relative(norms[0], norms[2]),
        pressure_relative=relative(norms[1], norms[3]),
    )
    if gradients:
        result.update(
            velocity_h1_seminorm=float(norms[4]),
            velocity_h1_relative=relative(norms[4], norms[5]),
        )
    return result


def mhm_difference(
    reference: TaylorHoodField, archive: str | Path, *, order: int = 8
) -> dict[str, float | int]:
    """Integrate broken MHM/reference differences over every MHM fine triangle.

    The quadrature respects all MHM macro interfaces. Reference finite-element
    interfaces can cross its integration triangles, so repeat with increasing
    ``order`` to assess this integration error. This is a physical area-weighted
    L2 norm, not an RMS of pixel-center display samples.
    """
    from pymhm import TriangleMesh
    from pymhm.fem.scalar.operators import triangle_quadrature
    from pymhm.fem.scalar.triangle import nodal_space, reference_basis

    bary, weights = triangle_quadrature(order)
    basis = reference_basis(3, bary)[0]
    totals = np.zeros(4)
    with np.load(archive) as data:
        local_points = data["local_points"]
        local_cells = data["local_cells"]
        local_velocity = data["local_velocity"]
        local_pressure = data["local_pressure"]
        for cell, points in enumerate(local_points):
            mesh = TriangleMesh(points, local_cells[cell])
            dofs, _ = nodal_space(mesh, 3)
            physical = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
            u = np.einsum("qi,tia->tqa", basis, local_velocity[cell][dofs])
            p = np.einsum("qi,ti->tq", basis, local_pressure[cell][dofs])
            v, q = reference.evaluate(physical.reshape(-1, 2))
            v, q = v.reshape(u.shape), q.reshape(p.shape)
            integrands = np.stack(
                (np.sum((u - v) ** 2, axis=-1), (p - q) ** 2, np.sum(v**2, axis=-1), q**2), axis=-1
            )
            totals += np.einsum("t,q,tqi->i", mesh.areas, weights, integrands)
    norms = np.sqrt(totals)
    return dict(
        quadrature_order=order,
        velocity_l2=float(norms[0]),
        pressure_l2=float(norms[1]),
        velocity_relative=float(norms[0] / norms[2]),
        pressure_relative=float(norms[1] / norms[3]),
    )


def solve(
    nx: int, ny: int, *, threads: int = 8, constant_drag: float | None = None
) -> tuple[TaylorHoodField, dict[str, Any]]:
    """Assemble and solve the conforming mixed system with native PETSc/MUMPS."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["BLIS_NUM_THREADS"] = "1"
    import basix.ufl
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    if not PETSc.Sys.hasExternalPackage("mumps"):
        raise RuntimeError("This reference requires PETSc built with the MUMPS LU solver")
    if constant_drag is None and (nx % 60 or ny % 220):
        raise ValueError("SPE10 reference grids must align with all 60-by-220 material pixels")
    started = perf_counter()
    domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_SELF,
        np.array([[0.0, 0.0], [1200.0, 2200.0]]),
        [nx, ny],
        cell_type=dolfinx.mesh.CellType.triangle,
        diagonal=dolfinx.mesh.DiagonalType.right,
    )
    element_u = basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))
    element_p = basix.ufl.element("Lagrange", "triangle", 1)
    W = dolfinx.fem.functionspace(domain, basix.ufl.mixed_element([element_u, element_p]))
    V, velocity_map = W.sub(0).collapse()
    P, pressure_map = W.sub(1).collapse()
    velocity_map = np.asarray(velocity_map, dtype=np.int32)
    pressure_map = np.asarray(pressure_map, dtype=np.int32)
    DG = dolfinx.fem.functionspace(domain, ("DG", 0))
    gamma = dolfinx.fem.Function(DG)
    centers = DG.tabulate_dof_coordinates()[:, :2]
    if constant_drag is None:
        with np.load(OUTPUT / "layer-1.npz") as data:
            permeability = data["permeability"][..., 0]
        pixel = np.floor(centers / [20.0, 10.0]).astype(int)
        gamma.x.array[:] = 0.3 / permeability[pixel[:, 0], pixel[:, 1]]
    else:
        gamma.x.array[:] = constant_drag
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})
    a = (
        0.3 * ufl.inner(ufl.grad(u), ufl.grad(v))
        + gamma * ufl.inner(u, v)
        - p * ufl.div(v)
        - q * ufl.div(u)
    ) * dx
    original = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
    original.assemble()
    coords = V.tabulate_dof_coordinates()[:, :2]
    bottom = np.flatnonzero(np.isclose(coords[:, 1], 0.0, rtol=0, atol=1e-10))
    sides = np.flatnonzero(
        np.isclose(coords[:, 0], 0.0, rtol=0, atol=1e-10)
        | np.isclose(coords[:, 0], 1200.0, rtol=0, atol=1e-10)
    )
    prescribed = np.unique(
        np.concatenate(
            (velocity_map[(2 * bottom[:, None] + [0, 1]).ravel()], velocity_map[2 * sides])
        )
    ).astype(PETSc.IntType)
    g = original.createVecRight()
    g.set(0)
    g.array[velocity_map[2 * bottom + 1]] = 1.0
    matrix = original.copy()
    rhs = original.createVecLeft()
    rhs.set(0)
    matrix.zeroRowsColumns(prescribed, diag=1.0, x=g, b=rhs)
    vector = matrix.createVecRight()
    ksp = PETSc.KSP().create(MPI.COMM_SELF)
    prefix = "spe10_th_"
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    options = PETSc.Options()
    settings = {"mat_mumps_icntl_7": 5, "mat_mumps_icntl_14": 50, "mat_mumps_icntl_10": 2}
    for key, value in settings.items():
        options[prefix + key] = value
    ksp.setFromOptions()
    assembled = perf_counter()
    print(f"Assembled {nx}x{ny}: {matrix.getSize()[0]:,} mixed DOFs; solving MUMPS", flush=True)
    ksp.solve(rhs, vector)
    solved = perf_counter()
    if ksp.getConvergedReason() <= 0:
        raise RuntimeError(f"MUMPS failed: PETSc reason {ksp.getConvergedReason()}")
    residual = original.createVecLeft()
    original.mult(vector, residual)
    physical_residual = residual.array.copy()
    free = np.ones(len(physical_residual), dtype=bool)
    free[prescribed] = False
    relative = float(np.linalg.norm(physical_residual[free]) / np.linalg.norm(rhs.array[free]))
    if not np.isfinite(relative) or relative > 1e-10:
        raise RuntimeError(f"Original free-equation residual {relative:.6g} exceeds 1e-10")
    mixed = dolfinx.fem.Function(W)
    mixed.x.array[:] = vector.array
    uh, ph = mixed.split()
    div_l2 = float(np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.div(uh) ** 2 * dx))))
    facets, tags = [], []
    for marker, axis, value in [(1, 1, 0.0), (2, 0, 1200.0), (3, 1, 2200.0), (4, 0, 0.0)]:
        found = dolfinx.mesh.locate_entities_boundary(
            domain,
            1,
            lambda x, axis=axis, value=value: np.isclose(x[axis], value, rtol=0, atol=1e-10),
        )
        facets.extend(found)
        tags.extend([marker] * len(found))
    permutation = np.argsort(facets)
    markers = dolfinx.mesh.meshtags(
        domain,
        1,
        np.asarray(facets, dtype=np.int32)[permutation],
        np.asarray(tags, dtype=np.int32)[permutation],
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=markers)
    normal = ufl.FacetNormal(domain)
    flux = [
        float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.dot(uh, normal) * ds(i))))
        for i in range(1, 5)
    ]
    velocity_values = vector.array[velocity_map].reshape(-1, 2)
    velocity_indices = np.rint(coords / [1200 / (2 * nx), 2200 / (2 * ny)]).astype(int)
    canonical_u = np.empty((2 * ny + 1, 2 * nx + 1, 2))
    canonical_u[velocity_indices[:, 1], velocity_indices[:, 0]] = velocity_values
    pressure_coords = P.tabulate_dof_coordinates()[:, :2]
    pressure_indices = np.rint(pressure_coords / [1200 / nx, 2200 / ny]).astype(int)
    canonical_p = np.empty((ny + 1, nx + 1))
    canonical_p[pressure_indices[:, 1], pressure_indices[:, 0]] = vector.array[pressure_map]
    field = TaylorHoodField(canonical_u, canonical_p)
    # Check that the exported evaluator reproduces independent DOLFINx point evaluation.
    cells = np.arange(min(31, len(domain.geometry.dofmap)), dtype=np.int32)
    samples = domain.geometry.x[domain.geometry.dofmap[cells]].mean(axis=1)
    values = field.evaluate(samples[:, :2])
    np.testing.assert_allclose(values[0], uh.eval(samples, cells), atol=2e-10, rtol=2e-12)
    np.testing.assert_allclose(
        values[1], ph.eval(samples, cells).reshape(-1), atol=2e-9, rtol=2e-12
    )
    report = dict(
        method="global conforming Taylor-Hood P2/P1; no residual stabilization",
        reference_backend="DOLFINx/UFL with PETSc/MUMPS",
        mesh_shape=[nx, ny],
        triangles=2 * nx * ny,
        triangle_diagonal="southwest to northeast",
        velocity_degree=2,
        pressure_degree=1,
        unknowns=matrix.getSize()[0],
        prescribed_velocity_unknowns=len(prescribed),
        free_unknowns=int(free.sum()),
        matrix_nonzeros=int(original.getInfo()["nz_used"]),
        original_free_residual_relative=relative,
        residual_tolerance=1e-10,
        weak_pressure_equation_residual_l2=float(np.linalg.norm(physical_residual[pressure_map])),
        weak_pressure_equation_residual_max=float(np.max(np.abs(physical_residual[pressure_map]))),
        divergence_l2=div_l2,
        exterior_flux_bottom_right_top_left=flux,
        exterior_balance=float(sum(flux)),
        velocity_nodal_max=float(np.linalg.norm(canonical_u, axis=-1).max()),
        pressure_nodal_min=float(canonical_p.min()),
        pressure_nodal_max=float(canonical_p.max()),
        boundary=(
            "bottom u=(0,1); sides ux=0 and tangential grad-grad traction=0; "
            "top full grad-grad traction=0"
        ),
        pressure_gauge="none: natural top traction fixes the additive pressure constant",
        material="gamma=0.3/Kx on original layer-1 pixels"
        if constant_drag is None
        else f"constant gamma={constant_drag}",
        viscosity=0.3,
        coordinate_unit="ft source numbers",
        permeability_unit="mD source numbers; no implicit SI conversion",
        material_integration="DG0 exact pixel assignment on fitted triangles",
        quadrature_degree=4,
        assembly_seconds=assembled - started,
        factor_and_solve_seconds=solved - assembled,
        total_seconds=perf_counter() - started,
        timing_scope=(
            "assembly, factorization/solve and in-solve diagnostics; excludes native object "
            "destruction, export and field-comparison norms; observed, not an exclusive "
            "performance benchmark"
        ),
        dolfinx=dolfinx.__version__,
        basix=basix.__version__,
        petsc=PETSc.Sys.getVersion(),
        numpy=np.__version__,
        python=platform.python_version(),
        mumps_settings=settings,
        openmp_threads=threads,
        blis_threads=1,
        conservation_scope=(
            "global boundary flux and weak continuous-P1 pressure equations; "
            "not fine-cell or MHM macro conservation"
        ),
    )
    for obj in [ksp, residual, vector, rhs, matrix, g, original]:
        obj.destroy()
    for key in settings:
        del options[prefix + key]
    return field, report


def archive(field: TaylorHoodField, report: dict[str, Any]) -> Path:
    """Store public samples and complete interpolable coefficients in build outputs."""
    nx, ny = field.shape
    stem = f"taylor-hood-{nx}x{ny}"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    COEFFICIENTS.mkdir(parents=True, exist_ok=True)
    coefficient_path = COEFFICIENTS / f"{stem}.npz"
    np.savez_compressed(coefficient_path, velocity=field.velocity, pressure=field.pressure)
    points = np.array(
        [(x, y) for x in np.arange(10, 1200, 20) for y in np.arange(5, 2200, 10)], dtype=float
    )
    velocity, pressure = field.evaluate(points)
    profile_points = np.column_stack((np.full(4401, 199.0), np.linspace(0, 2200, 4401)))
    profile_u, profile_p = field.evaluate(profile_points)
    path = OUTPUT / f"{stem}.npz"
    np.savez_compressed(
        path,
        points=points.reshape(60, 220, 2),
        velocity=velocity.reshape(60, 220, 2),
        pressure=pressure.reshape(60, 220),
        profile_points=profile_points,
        profile_velocity=profile_u,
        profile_pressure=profile_p,
    )
    report.update(
        archive=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        coefficient_archive=coefficient_path.relative_to(ROOT).as_posix(),
        coefficient_sha256=hashlib.sha256(coefficient_path.read_bytes()).hexdigest(),
        layer_sha256=hashlib.sha256((OUTPUT / "layer-1.npz").read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return path


def main() -> None:
    """Run one aligned reference mesh or an exact constant-drag boundary patch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=2, default=[60, 220])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--mhm", type=Path, help="P3/P3 MHM numerical archive to compare")
    parser.add_argument("--comparison-orders", type=int, nargs="+", default=[24, 32])
    parser.add_argument("--compare-only", action="store_true", help="reuse archived coefficients")
    args = parser.parse_args()
    if min(*args.shape, args.threads, *args.comparison_orders) < 1:
        parser.error("mesh sizes, threads and quadrature orders must be positive")
    if args.compare_only and (args.patch or args.mhm is None):
        parser.error("--compare-only requires --mhm and excludes --patch")
    if args.patch and args.mhm is not None:
        parser.error("--patch cannot be combined with --mhm")
    if args.compare_only:
        field, report = load_archived_reference(tuple(args.shape))
    else:
        field, report = solve(
            *args.shape, threads=args.threads, constant_drag=0.3 if args.patch else None
        )
    if args.patch:
        expected = 0.3 * (2200 - np.linspace(0, 2200, args.shape[1] + 1))[:, None]
        np.testing.assert_allclose(
            field.velocity, np.broadcast_to([0.0, 1.0], field.velocity.shape), atol=2e-9, rtol=0
        )
        np.testing.assert_allclose(
            field.pressure, np.broadcast_to(expected, field.pressure.shape), atol=2e-7, rtol=0
        )
        report["affine_patch"] = "u=(0,1), p=0.3*(2200-y), f=0; exact represented solution"
    elif not args.compare_only:
        if args.previous:
            report["successive_difference"] = difference(field, load_field(args.previous))
            report["previous_coefficients"] = args.previous.resolve().relative_to(ROOT).as_posix()
        archive(field, report)
    print(json.dumps(report, indent=2), flush=True)
    if args.mhm:
        norms = []
        for order in args.comparison_orders:
            row = mhm_difference(field, args.mhm, order=order)
            norms.append(row)
            print(json.dumps(row), flush=True)
        comparison = dict(
            reference=report["archive"],
            reference_sha256=report["sha256"],
            reference_coefficients_sha256=report["coefficient_sha256"],
            mhm=args.mhm.name,
            mhm_sha256=hashlib.sha256(args.mhm.read_bytes()).hexdigest(),
            integration=(
                "physical L2 over each MHM fine triangle; both one-sided macro "
                "fields preserved; no pressure-mean adjustment"
            ),
            comparison_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            norms=norms,
        )
        target = OUTPUT / f"taylor-hood-mhm-{args.shape[0]}x{args.shape[1]}.json"
        target.write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    main()
