"""Complete angular grids and refinement sequences of the 2020 acoustic study.

The manuscript specifies the traces and macro meshes for the analytical cases,
but not their local discretization. Those choices are explicit below and receive
separate local-refinement controls. Each completed configuration is checkpointed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.campaign_checkpoint import archive_identity, require_sources, verify_checkpoint
from examples.helmholtz_campaign import AcousticWave, archive, norms, source_hashes
from examples.helmholtz_incident_family import IncidentFamily
from examples.helmholtz_trace_family import verify_helmholtz_solution
from pymhm._legacy.models.waves.helmholtz import HelmholtzSolution, solve_helmholtz
from pymhm.fem.traces.helmholtz import helmholtz_skeleton
from pymhm.meshes.cartesian import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Configuration:
    """All mathematical choices of one complex wave calculation."""

    study: str
    n: int
    ell: int
    oscillatory: bool
    omega: float
    angle: float
    refinement: int = 2
    archive_fields: bool = False

    @property
    def key(self) -> str:
        """Return an exact deterministic configuration key for resume checking."""
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()[:20]


def solve_configuration(configuration: Configuration, output: Path) -> dict[str, Any]:
    """Solve one configuration with one BLAS thread and no nested worker pools."""
    c = configuration
    wave = AcousticWave(c.omega, c.angle)
    mesh = CartesianMacroMesh(c.n)
    skeleton = helmholtz_skeleton(mesh, c.omega, degree=c.ell, oscillatory=c.oscillatory)
    start = time.perf_counter()
    with threadpool_limits(1):
        solution = solve_helmholtz(
            mesh,
            omega=c.omega,
            skeleton=skeleton,
            degree=c.ell + 2,
            local_refinement=c.refinement,
            absorbing=wave.absorbing,
            quadrature_order=10,
        )
        return record_solution(c, solution, wave, start, output)


def record_solution(
    c: Configuration,
    solution: HelmholtzSolution,
    wave: AcousticWave,
    start: float,
    output: Path,
) -> dict[str, Any]:
    """Record the executed physical norms, quadrature and optional field bytes."""
    row: dict[str, Any] = {
        **c.__dict__,
        "key": c.key,
        "local_degree": c.ell + 2,
        "macro_edge_length": 1 / c.n,
        "macro_diameter": np.sqrt(2) / c.n,
        "requested_assembly_order": 10,
        "assembly_order": solution.quadrature_order,
        "error_order": 12,
        "trace_basis": "oscillatory" if c.oscillatory else "polynomial",
        "residual": solution.hybrid.residual,
        "macro_balance_max": float(np.max(abs(solution.conservation_residuals()))),
        **verify_helmholtz_solution(solution),
        **norms(solution, wave, order=12),
        "elapsed_seconds": time.perf_counter() - start,
    }
    if c.archive_fields:
        filename = f"{c.study}-{c.key}.npz"
        archive(solution, output / filename, wave)
        row.update(archive_identity(output / filename))
    return row


def solve_direction_family(cases: list[Configuration], output: Path) -> Iterator[dict[str, Any]]:
    """Reuse fixed harmonic responses/factors while recording each complete direction."""
    if not cases:
        return
    first = cases[0]
    fixed = (first.n, first.ell, first.oscillatory, first.omega, first.refinement)
    if any(
        c.study != "direction" or (c.n, c.ell, c.oscillatory, c.omega, c.refinement) != fixed
        for c in cases
    ):
        raise ValueError("incident reuse requires one fixed angular operator and trace basis")
    mesh = CartesianMacroMesh(first.n)
    wave = AcousticWave(first.omega, first.angle)
    skeleton = helmholtz_skeleton(
        mesh, first.omega, degree=first.ell, oscillatory=first.oscillatory
    )
    start = time.perf_counter()
    with threadpool_limits(1):
        prepared = solve_helmholtz(
            mesh,
            omega=first.omega,
            skeleton=skeleton,
            degree=first.ell + 2,
            local_refinement=first.refinement,
            absorbing=wave.absorbing,
            quadrature_order=10,
        )
        with IncidentFamily(prepared) as family:
            preparation_seconds = time.perf_counter() - start
            for index, c in enumerate(cases):
                angle_started = time.perf_counter()
                wave = AcousticWave(c.omega, c.angle)
                solution = prepared if index == 0 else family.solve(wave.absorbing)
                row = record_solution(c, solution, wave, angle_started, output)
                row["response_reuse"] = (
                    "Original fixed harmonic columns and checked local/global factors"
                )
                row["offline_preparation_seconds_shared"] = preparation_seconds
                yield row


def validate_row(row: dict[str, Any], output: Path) -> None:
    """Check an acquired row against its full configuration, metrics and field bytes."""
    configuration = Configuration(**{key: row[key] for key in Configuration.__dataclass_fields__})
    verify_checkpoint(
        row,
        {
            **configuration.__dict__,
            "key": configuration.key,
            "local_degree": configuration.ell + 2,
            "requested_assembly_order": 10,
            "assembly_order": max(10, 2 * configuration.ell + 4),
            "error_order": 12,
            "trace_basis": "oscillatory" if configuration.oscillatory else "polynomial",
            "macro_edge_length": 1 / configuration.n,
            "macro_diameter": np.sqrt(2) / configuration.n,
        },
        directory=output,
        metrics=(
            "pressure_l2",
            "gradient_l2",
            "pressure_exact_l2",
            "gradient_exact_l2",
            "pressure_relative_error",
            "gradient_relative_error",
            "energy_relative_error",
            "residual",
            "macro_balance_max",
            "original_field_trace_residual",
            "original_local_equation_residual_max",
        ),
        archive_required=configuration.archive_fields,
    )
    if (
        max(row["original_field_trace_residual"], row["original_local_equation_residual_max"])
        > 1e-10
    ):
        raise ValueError("article fields fail original Helmholtz equations")


def article_hashes() -> dict[str, str]:
    """Identify every shared numerical source and the incident acquisition owner."""
    hashes = source_hashes()
    for path in (
        Path(__file__),
        ROOT / "examples/helmholtz_incident_family.py",
        ROOT / "examples/local_response_cache.py",
    ):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def complete_rows(
    record: dict[str, Any], expected: list[Configuration], output: Path
) -> list[dict[str, Any]]:
    """Require the exact acquired case set before presenting a complete study.

    Duplicate cases, omitted directions, unrelated configurations and stale
    quadrature or archive contracts fail before a figure is rendered. The
    caller independently verifies the record's acquisition source identity.
    """
    rows = record["rows"]
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != {case.key for case in expected}:
        raise ValueError("the acquired Helmholtz case set is incomplete or inconsistent")
    for row in rows:
        validate_row(row, output)
    return rows


def publication_rows(output: Path) -> list[dict[str, Any]]:
    """Load all 1024 angles, 22 convergence points and 32 local controls under current sources."""
    record = json.loads((output / "article.json").read_text())
    require_sources(record["source_sha256"], article_hashes())
    verify_checkpoint(record, {"requested_assembly_order": 10, "error_order": 12}, directory=output)
    return complete_rows(
        record, configurations(["direction", "convergence", "local-control"]), output
    )


def configurations(studies: list[str]) -> list[Configuration]:
    """Return the 256 angles and analytical refinement/control configurations."""
    result: list[Configuration] = []
    for ell, n, omega in ((2, 11, 20 * np.pi), (4, 21, 40 * np.pi)):
        if "direction" in studies:
            for angle in np.linspace(0, np.pi / 2, 256):
                for oscillatory in (False, True):
                    result.append(Configuration("direction", n, ell, oscillatory, omega, angle))
        if "local-control" in studies:
            for angle in (0, np.pi / 13, np.pi / 4, np.pi / 2):
                for oscillatory in (False, True):
                    for refinement in (2, 4):
                        result.append(
                            Configuration(
                                "local-control", n, ell, oscillatory, omega, angle, refinement
                            )
                        )
    if "convergence" in studies:
        for ell, sizes in ((2, (12, 16, 24, 32, 48, 64)), (3, (24, 32, 48, 64, 96))):
            for n in sizes:
                for oscillatory in (False, True):
                    result.append(
                        Configuration(
                            "convergence",
                            n,
                            ell,
                            oscillatory,
                            10 * np.pi,
                            np.pi / 13,
                            archive_fields=n == sizes[-1],
                        )
                    )
    return result


def run(
    output: Path, studies: list[str], workers: int, resume: bool, *, reuse_incidents: bool = False
) -> None:
    """Acquire independent solves with source-checked atomic per-case checkpoints."""
    output.mkdir(parents=True, exist_ok=True)
    hashes = article_hashes()
    path = output / "article.json"
    record: dict[str, Any] = {
        "reference": "Chaumont-Frelet and Valentin (2020), DOI 10.1137/19M1255616",
        "analytical_sections": ["6.2", "6.3"],
        "local_space_convention": "Q(ell+2), local refinement 2; separately tested with 4",
        "requested_assembly_order": 10,
        "assembly_order_convention": "Each row records max(10, local degree + trace degree + 2)",
        "error_order": 12,
        "source_sha256": hashes,
        "rows": [],
    }
    if resume and path.exists():
        record = json.loads(path.read_text())
        require_sources(record["source_sha256"], hashes)
        verify_checkpoint(
            record, {"requested_assembly_order": 10, "error_order": 12}, directory=output
        )
        for row in record["rows"]:
            validate_row(row, output)
    done = {row["key"] for row in record["rows"]}
    pending = [c for c in configurations(studies) if c.key not in done]

    def save_row(row: dict[str, Any]) -> None:
        """Commit each completed point only while its numerical sources remain fixed."""
        current = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in hashes}
        if current != hashes:
            raise RuntimeError("runtime sources changed during the wave acquisition")
        record["rows"].append(row)
        record["rows"].sort(key=lambda r: r["key"])
        temporary = path.with_suffix(".json.new")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path)
        print(json.dumps(row), flush=True)

    if reuse_incidents:
        groups: dict[tuple, list[Configuration]] = {}
        for c in pending:
            if c.study == "direction":
                groups.setdefault((c.n, c.ell, c.oscillatory, c.omega, c.refinement), []).append(c)
        for cases in groups.values():
            for row in solve_direction_family(cases, output):
                save_row(row)
        pending = [c for c in pending if c.study != "direction"]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(solve_configuration, c, output): c for c in pending}
        for future in as_completed(futures):
            save_row(future.result())
    current = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in hashes}
    if current != hashes:
        raise RuntimeError("runtime sources changed during the wave acquisition")


def main() -> None:
    """Run full angle studies, convergence sequences and local-space controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/helmholtz-article")
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=("direction", "convergence", "local-control"),
        default=["direction", "convergence", "local-control"],
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-incidents", action="store_true")
    args = parser.parse_args()
    run(args.output, args.studies, args.workers, args.resume, reuse_incidents=args.reuse_incidents)


if __name__ == "__main__":
    main()
