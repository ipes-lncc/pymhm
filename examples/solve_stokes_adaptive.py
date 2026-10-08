"""Acquire L14 Stokes convergence and its two distinct adaptive marking strategies."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from threadpoolctl import threadpool_limits

from examples.field_sampling import sample_field
from examples.formulations.application import flow as solve_flow
from examples.solve_oseen import crisscross
from pymhm.adaptivity.flow import adapt_flow
from pymhm.adaptivity.flow_macro import adapt_flow_macros
from pymhm.estimators.flow import estimate_flow_error
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import VectorSolution

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/stokes-adaptive"


def save_coefficients(solution: VectorSolution, path: Path) -> dict:
    """Archive independent local polynomial coefficients for physical reference comparisons."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = dict(
        macro_points=solution.skeleton.mesh.points,
        macro_cells=solution.skeleton.mesh.cells,
        velocity_degree=solution.degree,
        pressure_degree=solution.pressure_degree,
    )
    for cell, local in enumerate(solution.local_meshes):
        raw[f"points_{cell}"] = local.points
        raw[f"cells_{cell}"] = local.cells
        raw[f"velocity_{cell}"] = solution.values[cell]
        raw[f"pressure_{cell}"] = solution.pressure[cell]
    np.savez_compressed(path, **raw)
    return dict(
        path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


@dataclass(frozen=True)
class StokesData:
    """Solenoidal L14 §5.3 data and explicitly selected cavity boundary traces."""

    viscosity: float = 1.0
    drag: float = 0.0
    cavity: bool = False
    lid: str = "regularized"

    def fields(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Differentiate psi=128*x²(1-x)²*y²(1-y)² independently of the FE operators."""
        polynomial = Polynomial([0, 0, 1, -2, 1])
        x, y = points.T
        fx = [polynomial.deriv(i)(x) for i in range(4)]
        fy = [polynomial.deriv(i)(y) for i in range(4)]
        u = 128 * np.column_stack((fx[0] * fy[1], -fx[1] * fy[0]))
        gradient = 128 * np.array(
            [[fx[1] * fy[1], fx[0] * fy[2]], [-fx[2] * fy[0], -fx[1] * fy[1]]]
        ).transpose(2, 0, 1)
        laplacian = 128 * np.column_stack(
            (fx[2] * fy[1] + fx[0] * fy[3], -fx[3] * fy[0] - fx[1] * fy[2])
        )
        return u, gradient, laplacian

    def velocity(self, points: np.ndarray) -> np.ndarray:
        """Return the analytical divergence-free velocity in the manufactured case."""
        return self.fields(points)[0]

    def gradient(self, points: np.ndarray) -> np.ndarray:
        """Return the exact Jacobian, component index followed by derivative index."""
        return self.fields(points)[1]

    def pressure(self, points: np.ndarray) -> np.ndarray:
        """Return the exact zero-mean pressure from L14 §5.3."""
        return 150 * (points[:, 0] - 0.5) * (points[:, 1] - 0.5)

    def source(self, points: np.ndarray) -> np.ndarray:
        """Evaluate -nu*Delta(u)+gamma*u+grad(p), or zero forcing for the cavity."""
        if self.cavity:
            return np.zeros_like(points)
        value, _, laplacian = self.fields(points)
        return -self.viscosity * laplacian + self.drag * value + 150 * (points[:, ::-1] - 0.5)

    def boundary(self, points: np.ndarray) -> np.ndarray:
        """Use a smooth or constant top trace; isolated corner values have zero trace measure."""
        if not self.cavity:
            return self.velocity(points)
        values = np.zeros_like(points)
        lid = np.isclose(points[:, 1], 1, rtol=0, atol=1e-13)
        x = points[lid, 0]
        values[lid, 0] = 1.0 if self.lid == "constant" else 16 * x**2 * (1 - x) ** 2
        return values

    def options(self) -> dict:
        """Return picklable PDE data for solver and estimator."""
        return dict(
            viscosity=self.viscosity, drag=self.drag, source=self.source, dirichlet=self.boundary
        )


def main() -> None:
    """Run explicit uniform or adaptive numerical campaigns outside the lightweight CI suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["uniform", "macro", "face"], default="uniform")
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--drag", type=float, default=0.0)
    parser.add_argument("--trace-degree", type=int, default=0)
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--cavity", action="store_true")
    parser.add_argument("--lid", choices=["regularized", "constant"], default="regularized")
    parser.add_argument("--maximum-cells", type=int, default=20000)
    parser.add_argument("--max-local-refinement", type=int, default=32)
    parser.add_argument("--local-refiner", choices=["uniform", "longest-edge"], default="uniform")
    parser.add_argument("--max-local-cells", type=int, default=65536)
    parser.add_argument("--local-error-marking", choices=["uniform", "maximum"], default="uniform")
    parser.add_argument(
        "--macro-refiner", choices=["red-green", "longest-edge"], default="red-green"
    )
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--save-history", action="store_true")
    parser.add_argument("--resume", help="Recorded macro-adaptive JSON basename, without suffix")
    args = parser.parse_args()
    if args.lid == "constant" and not args.cavity:
        parser.error("constant lid requires --cavity")
    data = StokesData(args.viscosity, args.drag, args.cavity, args.lid)
    case_name = (
        "cavity-constant"
        if args.cavity and args.lid == "constant"
        else ("cavity" if args.cavity else "polynomial")
    )
    name = f"{case_name}-{args.strategy}-nu{args.viscosity:g}-g{args.drag:g}-l{args.trace_degree}"
    if args.strategy == "macro" and args.macro_refiner != "red-green":
        name += "-" + args.macro_refiner
    if args.strategy == "face" and args.local_refiner != "uniform":
        name += "-local-" + args.local_refiner
    if args.strategy == "face" and args.local_error_marking != "uniform":
        name += "-mark-" + args.local_error_marking
    if args.theta != 0.5:
        name += f"-theta{args.theta:g}"
    degree = 2 if args.cavity else 3
    resumed_mesh, resumed_record = None, None
    if args.resume:
        if args.strategy != "macro" or Path(args.resume).name != args.resume:
            parser.error("resume requires macro strategy and a recorded JSON basename")
        recorded_path = OUTPUT / (args.resume + ".json")
        previous = json.loads(recorded_path.read_text())
        required = dict(
            strategy="macro",
            viscosity=args.viscosity,
            drag=args.drag,
            trace_degree=args.trace_degree,
            local_degree=degree,
            marking_theta=args.theta,
            macro_refiner=args.macro_refiner,
            lid=args.lid if args.cavity else None,
        )
        if any(previous.get(key) != value for key, value in required.items()):
            parser.error("resume must preserve the recorded physical and adaptive parameters")
        previous_archive = OUTPUT / previous["archive"]
        if hashlib.sha256(previous_archive.read_bytes()).hexdigest() != previous["sha256"]:
            parser.error("resume mesh archive digest mismatch")
        with np.load(previous_archive) as saved:
            resumed_mesh = TriangleMesh(saved["macro_points"], saved["macro_cells"])
        resumed_record = dict(
            record=recorded_path.name,
            sha256=hashlib.sha256(recorded_path.read_bytes()).hexdigest(),
            field_sha256=previous["sha256"],
        )
        name = args.resume + "-continued"
    options = dict(
        formulation="usfem", degree=degree, quadrature_order=args.order, **data.options()
    )
    paths = [
        Path(__file__),
        ROOT / "src/pymhm/_legacy/models/flow/solver.py",
        ROOT / "src/pymhm/estimators/flow.py",
        ROOT / "src/pymhm/adaptivity/flow.py",
        ROOT / "src/pymhm/adaptivity/flow_macro.py",
        ROOT / "src/pymhm/meshes/refinement.py",
        ROOT / "src/pymhm/adaptivity/flow_local_mesh.py",
        ROOT / "src/pymhm/meshes/longest_edge.py",
    ]
    hashes = current_source_manifest(
        {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    )
    snapshots = ROOT / "build/source-snapshots/stokes-adaptive"
    snapshots.mkdir(parents=True, exist_ok=True)
    for path in paths:
        (snapshots / f"{hashes[path.relative_to(ROOT).as_posix()]}-{path.name}").write_bytes(
            path.read_bytes()
        )
    solutions, estimators = [], []
    with threadpool_limits(1):
        if args.strategy == "uniform":
            for n in args.levels:
                mesh = crisscross(n)
                skeleton = SkeletonSpace(
                    mesh, tuple(FaceSpace.uniform(args.trace_degree) for _ in mesh.faces), 2
                )
                solution = solve_flow(mesh, skeleton=skeleton, local_refinement=1, **options)
                estimator = estimate_flow_error(
                    solution,
                    full_dirichlet=True,
                    variant="stokes-brinkman-2021",
                    quadrature_order=args.order,
                    **data.options(),
                )
                solutions.append(solution)
                estimators.append(estimator)
            stop = "levels"
        else:
            mesh = crisscross(args.levels[0]) if resumed_mesh is None else resumed_mesh
            if args.strategy == "macro":
                result = adapt_flow_macros(
                    mesh,
                    solve_step=solve_flow,
                    trace_degree=args.trace_degree,
                    iterations=args.iterations,
                    theta=args.theta,
                    maximum_cells=args.maximum_cells,
                    macro_refiner=args.macro_refiner,
                    estimator_order=args.order,
                    **options,
                )
            else:
                skeleton = SkeletonSpace(
                    mesh, tuple(FaceSpace.uniform(args.trace_degree) for _ in mesh.faces), 2
                )
                result = adapt_flow(
                    mesh,
                    solve_step=solve_flow,
                    skeleton=skeleton,
                    iterations=args.iterations,
                    theta=args.theta,
                    estimator_variant="stokes-brinkman-2021",
                    estimator_order=args.order,
                    max_local_refinement=args.max_local_refinement,
                    local_refiner=args.local_refiner,
                    max_local_cells=args.max_local_cells,
                    local_error_marking=args.local_error_marking,
                    **options,
                )
            solutions, estimators, stop = result.solutions, result.estimators, result.stop_reason
        rows = []
        for solution, estimator in zip(solutions, estimators, strict=True):
            row = dict(
                macro_cells=len(solution.skeleton.mesh.cells),
                fine_cells=sum(len(m.cells) for m in solution.local_meshes),
                trace_dofs=solution.skeleton.size,
                coarse_dofs=sum(len(value) for value in solution.hybrid.coarse),
                trace_and_coarse_dofs=solution.skeleton.size
                + sum(len(value) for value in solution.hybrid.coarse),
                eta1=estimator.eta1,
                eta2=estimator.eta2,
                total=estimator.total,
                divergence_l2=solution.divergence_l2(),
                algebraic_residual=solution.hybrid.residual,
            )
            vertices = solution.skeleton.mesh.points[solution.skeleton.mesh.cells]
            angles = []
            for corner in range(3):
                first = vertices[:, (corner + 1) % 3] - vertices[:, corner]
                second = vertices[:, (corner + 2) % 3] - vertices[:, corner]
                cosine = np.sum(first * second, axis=1) / (
                    np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
                )
                angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            row["minimum_macro_angle_degrees"] = float(np.min(angles))
            if not args.cavity:
                mixed = estimator.mixed_error(
                    data.velocity, data.gradient, data.pressure, order=args.order
                )
                row.update(
                    velocity_l2=solution.l2_error(data.velocity, args.order),
                    pressure_l2=solution.pressure_l2_error(data.pressure, args.order),
                    mixed_error=mixed,
                    effectivity=estimator.total / mixed,
                )
            rows.append(row)
            print(json.dumps(row), flush=True)
        solution = solutions[-1]
        u = sample_field(solution.local_meshes, solution.values, solution.degree, 4)
        p = sample_field(solution.local_meshes, solution.pressure, solution.pressure_degree, 4)
        arrays = dict(
            points=u["points"],
            cells=u["cells"],
            velocity=u["values"],
            pressure=p["values"],
            macro_points=solution.skeleton.mesh.points,
            macro_cells=solution.skeleton.mesh.cells,
        )
        if not args.cavity:
            arrays.update(
                exact_velocity=data.velocity(u["points"]), exact_pressure=data.pressure(p["points"])
            )
        for i, solved in enumerate(solutions):
            arrays[f"macro_points_{i}"] = solved.skeleton.mesh.points
            arrays[f"macro_cells_{i}"] = solved.skeleton.mesh.cells
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / (name + ".npz")
    np.savez_compressed(archive, **arrays)
    coefficients = ROOT / "build/results/stokes-adaptive" / (name + "-coefficients.npz")
    saved = save_coefficients(solution, coefficients)
    history = (
        [
            save_coefficients(state, coefficients.with_name(f"{name}-state{i}-coefficients.npz"))
            for i, state in enumerate(solutions)
        ]
        if args.save_history
        else []
    )
    report = dict(
        case=f"{args.lid} cavity" if args.cavity else "L14 solenoidal polynomial data",
        lid=args.lid if args.cavity else None,
        strategy=args.strategy,
        viscosity=args.viscosity,
        drag=args.drag,
        trace_degree=args.trace_degree,
        local_degree=degree,
        marking_theta=args.theta,
        maximum_cells=args.maximum_cells,
        max_local_refinement=args.max_local_refinement,
        local_refiner=args.local_refiner if args.strategy == "face" else None,
        max_local_cells=args.max_local_cells,
        local_error_marking=args.local_error_marking if args.strategy == "face" else None,
        macro_refiner=args.macro_refiner if args.strategy == "macro" else None,
        quadrature_order=args.order,
        stop_reason=stop,
        rows=rows,
        archive=archive.name,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        coefficients=saved["path"],
        coefficients_sha256=saved["sha256"],
        coefficient_history=history,
        source_hashes=hashes,
        initial_mesh="structured crisscross, four triangles per square",
        resumed_record=resumed_record,
        refinement=f"{args.macro_refiner} macro closure"
        if args.strategy == "macro"
        else "fixed macro topology",
        limitations=(
            "The solenoidal component convention is stated explicitly; printed Ei is not "
            "a universal numerical target. Historical theta, mesh connectivity and cavity "
            "corner data are unspecified. Constant lid is interpreted as a weak boundary "
            "trace with corner singularities; the regularized lid is a separate variation."
        ),
    )
    (OUTPUT / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p.relative_to(ROOT).as_posix()]
        for p in paths
    ):
        raise RuntimeError("Campaign numerical source changed during acquisition")


if __name__ == "__main__":
    main()
