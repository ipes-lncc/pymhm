"""Acquire the acoustic stability study with exact disk-backed local responses.

The local assembler, condensation, trace solve, physical-flux projection and
error integrals are shared with the in-memory study. Local coefficients are
reconstructed one macrocell at a time after the complete global trace solve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.campaign_checkpoint import archive_path, require_sources, verify_checkpoint
from examples.helmholtz_campaign import AcousticWave, source_hashes
from examples.helmholtz_compact_family import CompactFamily
from examples.helmholtz_response_store import ResponseStore
from examples.helmholtz_stability import (
    checkpoint,
    continuous_local_resonance,
    projected_trace,
)
from examples.tutorial_helmholtz_equations import AcousticAssemblyProvider
from pymhm.core.validation import positive_int
from pymhm.fem.loads import split_point_sources
from pymhm.fem.traces.helmholtz import helmholtz_skeleton
from pymhm.io.provenance import file_digest
from pymhm.io.workspace import case_workspace, source_file
from pymhm.linalg.linear import LinearSolveError
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.acoustics import local_helmholtz_error_squared

ROOT = case_workspace()


def hashes() -> dict[str, str]:
    """Identify numerical, projection, response-store and acquisition owners."""
    sources = source_hashes()
    for name in (
        "examples/helmholtz_stability.py",
        "examples/helmholtz_stability_phases.py",
        "examples/helmholtz_compact_family.py",
        "examples/helmholtz_trace_family.py",
        "examples/helmholtz_response_store.py",
        "examples/local_response_cache.py",
        "examples/helmholtz_threshold.py",
        "examples/campaign_checkpoint.py",
        "examples/campaign_provenance.py",
    ):
        sources[name] = hashlib.sha256((source_file(name, root=ROOT)).read_bytes()).hexdigest()
    return sources


def solve_configuration(
    n: int,
    ell: int,
    frequency: float,
    directory: Path,
    *,
    sources: dict[str, str],
    workers: int = 1,
    batch_size: int = 64,
    assembly_order: int = 12,
    norm_order: int = 16,
    projection_order: int = 24,
    exact_response_cache: bool = False,
) -> dict[str, Any]:
    """Solve the complete square and stream its two broken gradient errors.

    The requested assembly order must already satisfy the shared assembler's
    local/trace polynomial minimum. Q(ell+3) on a 2x2 local mesh and P(ell)
    traces match the in-memory study's explicitly selected finite spaces.
    ``directory`` retains verified, resumable responses for this physical case.
    """
    n, ell = positive_int(n, "n"), positive_int(ell, "ell", 0)
    workers = positive_int(workers, "workers")
    batch_size = positive_int(batch_size, "batch_size")
    assembly_order = positive_int(assembly_order, "assembly_order")
    norm_order = positive_int(norm_order, "norm_order")
    projection_order = positive_int(projection_order, "projection_order")
    if ell > 1 or assembly_order < 2 * ell + 5 or projection_order <= ell:
        raise ValueError("stability phases require ell=0/1 and adequate declared quadrature")
    if continuous_local_resonance(n, frequency) is not None:
        raise ValueError("the continuous interior Neumann lifting is resonant")
    started = time.perf_counter()
    wave = AcousticWave(2 * np.pi * frequency, 0, "hankel")
    mesh = CartesianMacroMesh(n)
    skeleton = helmholtz_skeleton(mesh, wave.omega, degree=ell)
    absorbing = dict.fromkeys(mesh.boundary_faces, wave.absorbing)
    configuration = {
        "n": n,
        "ell": ell,
        "omega": wave.omega,
        "local_degree": ell + 3,
        "local_refinement": 2,
        "assembly_order": assembly_order,
        "density": 1,
        "bulk_modulus": 1,
        "source": 0,
        "bounds": [0, 1, 0, 1],
        "boundary": "grad(u).n-i*omega*u=g on every exterior face",
        "exact_field": "Hankel H0(omega*|x-(1.5,0.5)|)",
        "trace": "Discontinuous polynomial P(ell), physical globally oriented normal flux",
    }
    factory = AcousticAssemblyProvider(
        mesh,
        skeleton,
        wave.omega,
        ell + 3,
        2,
        assembly_order,
        1.0,
        1.0,
        0j,
        split_point_sources(mesh, ()),
        0j,
        absorbing,
        {},
        None,
    )
    with threadpool_limits(1):
        store = ResponseStore.prepare(
            factory,
            directory,
            sources=sources,
            configuration=configuration,
            batch_size=batch_size,
            backend="process" if workers > 1 else "serial",
            workers=workers,
            exact_response_cache=exact_response_cache,
        )
        family = CompactFamily.from_locals(skeleton, store)
        prepared_seconds = time.perf_counter() - started
        solved = family.solve_trace(ell)
        original_trace_residual = family.verify_fields(
            solved, (item[0] for item in family.reconstruct(solved.prepared_coordinates))
        )
        projection = projected_trace(skeleton, wave, absorbing, projection_order)
        trace_seconds = time.perf_counter() - started - prepared_seconds
        actual_squared, projected_squared, exact_squared = 0.0, 0.0, 0.0
        local_residual_max, macro_balance_max = 0.0, 0.0
        count = 0
        for cell, (actual, interpolated) in enumerate(
            zip(
                family.reconstruct(solved.prepared_coordinates),
                family.reconstruct(projection),
                strict=True,
            )
        ):
            fine = mesh.submesh(cell, 2)
            actual_squared += local_helmholtz_error_squared(
                fine, actual[0], ell + 3, wave.gradient, norm_order, derivative=True
            )
            projected_squared += local_helmholtz_error_squared(
                fine, interpolated[0], ell + 3, wave.gradient, norm_order, derivative=True
            )
            exact_squared += local_helmholtz_error_squared(
                fine,
                np.zeros_like(actual[0]),
                ell + 3,
                wave.gradient,
                norm_order,
                derivative=True,
            )
            local_residual_max = max(local_residual_max, actual[2], interpolated[2])
            macro_balance_max = max(macro_balance_max, abs(actual[1]), abs(interpolated[1]))
            count += 1
    if count != n * n or hashes() != sources:
        raise RuntimeError("stability responses or executed sources changed during acquisition")
    row = {
        "n": n,
        "macro_edge_length": 1 / n,
        "macro_diameter": np.sqrt(2) / n,
        "assembly_order": assembly_order,
        "norm_order": norm_order,
        "projection_order": projection_order,
        "mhm_gradient_l2": float(np.sqrt(actual_squared)),
        "interpolated_gradient_l2": float(np.sqrt(projected_squared)),
        "mhm_gradient_relative": float(np.sqrt(actual_squared / exact_squared)),
        "interpolated_gradient_relative": float(np.sqrt(projected_squared / exact_squared)),
        "ratio": float(np.sqrt(actual_squared / projected_squared)),
        "algebraic_residual": solved.residual,
        "original_field_trace_residual": original_trace_residual,
        "local_equation_residual_max": local_residual_max,
        "macro_balance_max": float(macro_balance_max),
        "elapsed_seconds": time.perf_counter() - started,
        "preparation_seconds": prepared_seconds,
        "trace_and_projection_seconds": trace_seconds,
        "local_response_storage_bytes": store.storage_bytes,
        "response_batch_acquisition_seconds_total": store.acquisition_seconds,
        "timing_convention": (
            "elapsed_seconds covers this invocation; batch acquisition time includes "
            "earlier resumed batches. Global assembly, verification and evaluation "
            "remain in this invocation's stage times."
        ),
        "reconstructed_macro_count": count,
        "response_manifest": str((directory / "responses.json").resolve()),
        "response_manifest_sha256": hashlib.sha256(
            (directory / "responses.json").read_bytes()
        ).hexdigest(),
    }
    return row


def run(
    output: Path,
    ell: int,
    frequency: float,
    sizes: list[int],
    workers: int = 1,
    *,
    batch_size: int = 64,
    assembly_order: int = 12,
    norm_order: int = 16,
    projection_order: int = 24,
    exact_response_cache: bool = False,
) -> None:
    """Checkpoint complete phased configurations under one numerical source contract."""
    ell = positive_int(ell, "ell", 0)
    workers = positive_int(workers, "workers")
    batch_size = positive_int(batch_size, "batch_size")
    assembly_order = positive_int(assembly_order, "assembly_order")
    norm_order = positive_int(norm_order, "norm_order")
    projection_order = positive_int(projection_order, "projection_order")
    if not isinstance(exact_response_cache, bool) or (exact_response_cache and workers != 1):
        raise ValueError("exact response caching requires one serial worker")
    if ell > 1 or assembly_order < 2 * ell + 5 or projection_order <= ell:
        raise ValueError("stability phases require ell=0/1 and adequate declared quadrature")
    if not sizes:
        raise ValueError("sizes must not be empty")
    for n in sizes:
        continuous_local_resonance(n, frequency)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"ell{ell}-frequency{frequency:g}.json"
    sources = hashes()
    identity = {
        "ell": ell,
        "omega": 2 * np.pi * frequency,
        "local_space": f"Q{ell + 3} on 2x2 fine cells per macrocell",
        "requested_assembly_order": assembly_order,
        "norm_order": norm_order,
        "projection_order": projection_order,
        "response_batch_size": batch_size,
        "storage": "Exact response batches; trace solve followed by streamed local norms",
    }
    if exact_response_cache:
        identity["exact_response_cache"] = True
    record = {
        **identity,
        "reference": "Chaumont-Frelet and Valentin (2020), Section 6.1, Equation (6.3)",
        "local_space_status": "explicitly selected; not specified in the article",
        "mesh_parameter": "H=1/n is square edge length; diameter=sqrt(2)/n",
        "source_sha256": sources,
        "rows": [],
    }
    if path.exists():
        record = json.loads(path.read_text())
        require_sources(record.get("source_sha256", {}), sources)
        verify_checkpoint(record, identity, directory=output)
        for row in record["rows"]:
            verify_checkpoint(
                row,
                {
                    "assembly_order": assembly_order,
                    "norm_order": norm_order,
                    "projection_order": projection_order,
                    "reconstructed_macro_count": row["n"] ** 2,
                },
                directory=output,
                metrics=(
                    "mhm_gradient_l2",
                    "interpolated_gradient_l2",
                    "ratio",
                    "mhm_gradient_relative",
                    "interpolated_gradient_relative",
                    "algebraic_residual",
                    "original_field_trace_residual",
                    "local_equation_residual_max",
                    "macro_balance_max",
                ),
            )
            manifest = Path(row["response_manifest"])
            if file_digest(manifest) != row["response_manifest_sha256"]:
                raise ValueError("accepted response manifest digest differs")
            saved = json.loads(manifest.read_text())
            for batch in saved["batches"]:
                if file_digest(archive_path(manifest.parent, batch["archive"])) != batch["sha256"]:
                    raise ValueError("accepted response batch digest differs")
    done = {row["n"] for row in record["rows"]}
    done.update(row["n"] for row in record.get("excluded_settings", ()))
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
                        "The continuous interior Neumann lifting has a nonzero cosine kernel."
                    ),
                }
            )
        else:
            directory = output / "responses" / f"ell{ell}-frequency{frequency:g}-n{n}"
            try:
                row = solve_configuration(
                    n,
                    ell,
                    frequency,
                    directory,
                    sources=sources,
                    workers=workers,
                    batch_size=batch_size,
                    assembly_order=assembly_order,
                    norm_order=norm_order,
                    projection_order=projection_order,
                    exact_response_cache=exact_response_cache,
                )
            except LinearSolveError as error:
                record.setdefault("excluded_settings", []).append(
                    {"n": n, "failure_type": type(error).__name__, "reason": str(error)}
                )
                checkpoint(path, record)
                raise
            record["rows"].append(row)
        checkpoint(path, record)
        print(
            json.dumps(record["rows"][-1] if mode is None else record["excluded_settings"][-1]),
            flush=True,
        )


def main() -> None:
    """Acquire a declared stability sequence with bounded local response storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ell", type=int, choices=(0, 1), required=True)
    parser.add_argument("--frequency", type=float, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--assembly-order", type=int, default=12)
    parser.add_argument("--norm-order", type=int, default=16)
    parser.add_argument("--projection-order", type=int, default=24)
    parser.add_argument("--exact-response-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.output,
        args.ell,
        args.frequency,
        args.sizes,
        args.workers,
        batch_size=args.batch_size,
        assembly_order=args.assembly_order,
        norm_order=args.norm_order,
        projection_order=args.projection_order,
        exact_response_cache=args.exact_response_cache,
    )


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.helmholtz_stability_phases").main()
