"""Helmholtz pollution versus the exact-flux face projection of Section 6.1."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import time
from collections.abc import Collection
from dataclasses import replace
from fractions import Fraction
from math import isqrt
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from threadpoolctl import threadpool_limits

from examples.campaign_checkpoint import require_sources, verify_checkpoint
from examples.helmholtz_campaign import AcousticWave, norms, source_hashes
from examples.helmholtz_threshold import sampled_threshold
from examples.tutorial_helmholtz_equations import solve_acoustic
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.helmholtz import complex_vector, real_vector
from pymhm.fem.traces.helmholtz import helmholtz_skeleton
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.linalg.linear import LinearSolveError
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.acoustics import HelmholtzSolution

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(path: Path, record: dict[str, Any]) -> None:
    """Save accepted and rejected attempts under one source and threshold contract."""
    if record["source_sha256"] != {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in record["source_sha256"]
    }:
        raise RuntimeError("stability sources changed during acquisition")
    record["rows"].sort(key=lambda row: row["n"])
    threshold = sampled_threshold(record["rows"], record.get("excluded_settings", ()))
    record["largest_suffix_H_star"] = threshold["sampled_suffix_threshold"]
    record["sampled_threshold"] = threshold
    temporary = path.with_suffix(".json.new")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def projected_trace(
    skeleton: SkeletonSpace,
    wave: AcousticWave,
    absorbing: Collection[int],
    order: int = 24,
) -> np.ndarray:
    """Project oriented physical flux into real interleaved skeletal coordinates.

    The physical flux is ``-grad(u).n`` in the globally oriented macroface
    normal. Absorbing-face multipliers are zero because the analytical
    impedance datum belongs to the local load. Each remaining face uses its
    own L2 Gram matrix and the declared Gauss order.
    """
    mesh = skeleton.mesh
    order = positive_int(order, "projection_order")
    if order <= max(max(face.degrees) for face in skeleton.faces):
        raise ValueError("projection quadrature must integrate the trace Gram matrix")
    t, weights = leggauss(order)
    t, weights = (t + 1) / 2, weights / 2
    trace = np.zeros(skeleton.size)
    normals = mesh.normals
    for face, (vertices, space) in enumerate(zip(mesh.faces, skeleton.faces, strict=True)):
        if face in absorbing:
            continue
        endpoints = mesh.points[vertices]
        points = endpoints[0] + t[:, None] * (endpoints[1] - endpoints[0])
        basis = space.evaluate(t)
        flux = -wave.gradient(points) @ normals[face]
        coefficient = np.linalg.solve(
            basis.T @ (weights[:, None] * basis), basis.T @ (weights * flux)
        )
        trace[skeleton.dofs(face)] = real_vector(coefficient)
    return trace


def projected_solution(
    solution: HelmholtzSolution, wave: Any, order: int = 24
) -> HelmholtzSolution:
    """Lift the oriented physical-flux L2 projection through the actual local inverse.

    On absorbing faces the skeletal trace is zero: the analytical impedance
    datum already belongs to each local load. The returned pressure represents
    T(Pi_H lambda)+Ttilde(g); it is not the best approximation in the local space.
    """
    trace = projected_trace(solution.skeleton, wave, solution.absorbing, order)
    fields = tuple(
        complex_vector(response.reconstruct(trace[response.problem.trace_dofs], np.empty(0)))
        for response in solution.system.responses
    )
    return replace(solution, pressure=fields, trace=complex_vector(trace))


def continuous_local_resonance(n: int, frequency: float) -> tuple[int, int] | None:
    """Identify exact homogeneous-square Neumann eigenfrequencies in this study.

    For square edge H=1/n and omega=2*pi*frequency, the interior Neumann
    eigenvalues are pi**2*(a**2+b**2)/H**2. A decimal frequency is treated as
    its declared exact decimal value, without a near-resonance tolerance.
    For n<3 every macrocell touches an absorbing side, so no pure Neumann
    interior cell exists. Passing this test does not establish an inf-sup bound.
    """
    n = positive_int(n, "n")
    if np.iscomplexobj(frequency) or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency must be finite and positive")
    if n < 3:
        return None
    squared = (2 * Fraction(str(frequency)) / n) ** 2
    if squared.denominator != 1:
        return None
    for a in range(isqrt(squared.numerator) + 1):
        b = isqrt(squared.numerator - a * a)
        if a * a + b * b == squared.numerator:
            return a, b
    return None


def run(
    output: Path,
    ell: int,
    frequency: float,
    sizes: list[int],
    workers: int,
    *,
    assembly_order: int = 12,
    norm_order: int = 16,
    projection_order: int = 24,
) -> None:
    """Checkpoint a fixed-frequency sequence and its suffix-based quasi-optimal threshold.

    An empty sequence validates inputs and any existing checkpoint identity
    without starting a solve or claiming a sampled stability threshold.
    """
    assembly_order = positive_int(assembly_order, "assembly_order")
    norm_order = positive_int(norm_order, "norm_order")
    projection_order = positive_int(projection_order, "projection_order")
    ell = positive_int(ell, "ell", 0)
    if ell > 1:
        raise ValueError("this stability study requires ell=0 or ell=1")
    workers = positive_int(workers, "workers")
    continuous_local_resonance(1, frequency)
    for n in sizes:
        continuous_local_resonance(n, frequency)
    if projection_order <= ell:
        raise ValueError("projection quadrature must integrate the trace Gram matrix")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"ell{ell}-frequency{frequency:g}.json"
    original = source_hashes()
    original[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    helper = ROOT / "examples/helmholtz_threshold.py"
    original[helper.relative_to(ROOT).as_posix()] = hashlib.sha256(helper.read_bytes()).hexdigest()
    record: dict[str, Any] = {
        "reference": "Chaumont-Frelet and Valentin (2020), Section 6.1, Equation (6.3)",
        "ell": ell,
        "omega": 2 * np.pi * frequency,
        "local_space": f"Q{ell + 3} on 2x2 fine cells per macrocell",
        "local_space_status": "explicitly selected; not specified in the article",
        "mesh_parameter": "H=1/n is square edge length; diameter=sqrt(2)/n",
        "requested_assembly_order": assembly_order,
        "norm_order": norm_order,
        "projection_order": projection_order,
        "source_sha256": original,
        "rows": [],
    }
    identity = {
        key: record[key]
        for key in (
            "ell",
            "omega",
            "local_space",
            "requested_assembly_order",
            "norm_order",
            "projection_order",
        )
    }
    if path.exists():
        record = json.loads(path.read_text())
        require_sources(record.get("source_sha256", {}), original)
        verify_checkpoint(record, identity, directory=output)
        for row in record["rows"]:
            verify_checkpoint(
                row,
                {"n": row["n"]},
                directory=output,
                metrics=(
                    "mhm_gradient_l2",
                    "interpolated_gradient_l2",
                    "ratio",
                    "mhm_gradient_relative",
                    "interpolated_gradient_relative",
                    "algebraic_residual",
                ),
            )
    done = {row["n"] for row in record["rows"]}
    done.update(row["n"] for row in record.get("excluded_settings", ()))
    wave = AcousticWave(2 * np.pi * frequency, 0, "hankel")
    for n in sorted(set(sizes) - done):
        mode = continuous_local_resonance(n, frequency)
        if mode is not None:
            record.setdefault("excluded_settings", []).append(
                {
                    "n": n,
                    "failure_type": "ContinuousLocalResonance",
                    "neumann_mode": list(mode),
                    "omega_H_over_pi": 2 * frequency / n,
                    "reason": (
                        "The interior continuous Neumann lifting has a nonzero cosine "
                        "eigenmode at this frequency; a nonsingular finite matrix would "
                        "not make this an admissible one-level MHM point."
                    ),
                }
            )
            checkpoint(path, record)
            print(json.dumps(record["excluded_settings"][-1]), flush=True)
            continue
        start = time.perf_counter()
        mesh = CartesianMacroMesh(n)
        with threadpool_limits(1):
            try:
                result = solve_acoustic(
                    mesh,
                    omega=wave.omega,
                    degree=ell + 3,
                    local_refinement=2,
                    quadrature_order=assembly_order,
                    skeleton=helmholtz_skeleton(mesh, wave.omega, degree=ell),
                    absorbing=wave.absorbing,
                    backend="process" if workers > 1 else "serial",
                    workers=workers,
                )
            except LinearSolveError as error:
                record.setdefault("excluded_settings", []).append(
                    {"n": n, "failure_type": type(error).__name__, "reason": str(error)}
                )
                checkpoint(path, record)
                raise
            actual = norms(result, wave, order=norm_order)
            interpolated = norms(
                projected_solution(result, wave, projection_order), wave, order=norm_order
            )
        row = {
            "n": n,
            "macro_edge_length": 1 / n,
            "macro_diameter": np.sqrt(2) / n,
            "assembly_order": result.quadrature_order,
            "norm_order": norm_order,
            "projection_order": projection_order,
            "mhm_gradient_l2": actual["gradient_l2"],
            "interpolated_gradient_l2": interpolated["gradient_l2"],
            "mhm_gradient_relative": actual["gradient_relative_error"],
            "interpolated_gradient_relative": interpolated["gradient_relative_error"],
            "ratio": actual["gradient_l2"] / interpolated["gradient_l2"],
            "algebraic_residual": result.hybrid.residual,
            "elapsed_seconds": time.perf_counter() - start,
        }
        record["rows"].append(row)
        checkpoint(path, record)
        print(json.dumps(row), flush=True)
        del result


def main() -> None:
    """Compute the paper's separate approximation and pollution diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ell", type=int, choices=(0, 1), default=0)
    parser.add_argument("--frequency", type=float, default=10)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--assembly-order", type=int, default=12)
    parser.add_argument("--norm-order", type=int, default=16)
    parser.add_argument("--projection-order", type=int, default=24)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "examples/results/helmholtz-stability"
    )
    args = parser.parse_args()
    run(
        args.output,
        args.ell,
        args.frequency,
        args.sizes,
        args.workers,
        assembly_order=args.assembly_order,
        norm_order=args.norm_order,
        projection_order=args.projection_order,
    )


if __name__ == "__main__":
    main()
