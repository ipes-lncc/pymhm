"""Helmholtz pollution versus the exact-flux face projection of Section 6.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from threadpoolctl import threadpool_limits

from examples.campaign_checkpoint import require_sources, verify_checkpoint
from examples.helmholtz_campaign import AcousticWave, norms, source_hashes
from examples.helmholtz_threshold import sampled_threshold
from pymhm.helmholtz import HelmholtzSolution, solve_helmholtz
from pymhm.helmholtz_forms import complex_vector, real_vector
from pymhm.helmholtz_spaces import helmholtz_skeleton
from pymhm.quadrilateral import CartesianMacroMesh
from pymhm.solvers import LinearSolveError

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


def projected_solution(solution: HelmholtzSolution, wave: AcousticWave) -> HelmholtzSolution:
    """Lift the oriented physical-flux L2 projection through the actual local inverse.

    On absorbing faces the skeletal trace is zero: the analytical impedance
    datum already belongs to each local load. The returned pressure represents
    T(Pi_H lambda)+Ttilde(g); it is not the best approximation in the local space.
    """
    skeleton = solution.skeleton
    mesh = skeleton.mesh
    t, weights = leggauss(24)
    t, weights = (t + 1) / 2, weights / 2
    trace = np.zeros(skeleton.size)
    normals = mesh.normals
    for face, (vertices, space) in enumerate(zip(mesh.faces, skeleton.faces, strict=True)):
        if face in solution.absorbing:
            continue
        endpoints = mesh.points[vertices]
        points = endpoints[0] + t[:, None] * (endpoints[1] - endpoints[0])
        basis = space.evaluate(t)
        flux = -wave.gradient(points) @ normals[face]
        coefficient = np.linalg.solve(
            basis.T @ (weights[:, None] * basis), basis.T @ (weights * flux)
        )
        trace[skeleton.dofs(face)] = real_vector(coefficient)
    fields = tuple(
        complex_vector(response.reconstruct(trace[response.problem.trace_dofs], np.empty(0)))
        for response in solution.system.responses
    )
    return replace(solution, pressure=fields, trace=complex_vector(trace))


def run(output: Path, ell: int, frequency: float, sizes: list[int], workers: int) -> None:
    """Checkpoint a fixed-frequency sequence and its suffix-based quasi-optimal threshold."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"ell{ell}-frequency{frequency:g}.json"
    original = source_hashes()
    original[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    helper = ROOT / "examples/helmholtz_threshold.py"
    original[str(helper.relative_to(ROOT))] = hashlib.sha256(helper.read_bytes()).hexdigest()
    record: dict[str, Any] = {
        "reference": "Chaumont-Frelet and Valentin (2020), Section 6.1, Equation (6.3)",
        "ell": ell,
        "omega": 2 * np.pi * frequency,
        "local_space": f"Q{ell + 3} on 2x2 fine cells per macrocell",
        "local_space_status": "explicitly selected; not specified in the article",
        "mesh_parameter": "H=1/n is square edge length; diameter=sqrt(2)/n",
        "source_sha256": original,
        "rows": [],
    }
    identity = {key: record[key] for key in ("ell", "omega", "local_space")}
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
        start = time.perf_counter()
        mesh = CartesianMacroMesh(n)
        with threadpool_limits(1):
            try:
                result = solve_helmholtz(
                    mesh,
                    omega=wave.omega,
                    degree=ell + 3,
                    local_refinement=2,
                    quadrature_order=12,
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
            actual = norms(result, wave, order=16)
            interpolated = norms(projected_solution(result, wave), wave, order=16)
        row = {
            "n": n,
            "macro_edge_length": 1 / n,
            "macro_diameter": np.sqrt(2) / n,
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
    parser.add_argument(
        "--output", type=Path, default=ROOT / "examples/results/helmholtz-stability"
    )
    args = parser.parse_args()
    run(args.output, args.ell, args.frequency, args.sizes, args.workers)


if __name__ == "__main__":
    main()
