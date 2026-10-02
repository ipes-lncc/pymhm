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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.campaign_checkpoint import archive_identity, require_sources, verify_checkpoint
from examples.helmholtz_campaign import AcousticWave, archive, norms, source_hashes
from pymhm.helmholtz import solve_helmholtz
from pymhm.helmholtz_spaces import helmholtz_skeleton
from pymhm.quadrilateral import CartesianMacroMesh

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
        row: dict[str, Any] = {
            **c.__dict__,
            "key": c.key,
            "local_degree": c.ell + 2,
            "macro_edge_length": 1 / c.n,
            "macro_diameter": np.sqrt(2) / c.n,
            "trace_basis": "oscillatory" if c.oscillatory else "polynomial",
            "residual": solution.hybrid.residual,
            "macro_balance_max": float(np.max(abs(solution.conservation_residuals()))),
            **norms(solution, wave, order=12),
            "elapsed_seconds": time.perf_counter() - start,
        }
        if c.archive_fields:
            filename = f"{c.study}-{c.key}.npz"
            archive(solution, output / filename, wave)
            row.update(archive_identity(output / filename))
    return row


def validate_row(row: dict[str, Any], output: Path) -> None:
    """Check an acquired row against its full configuration, metrics and field bytes."""
    configuration = Configuration(**{key: row[key] for key in Configuration.__dataclass_fields__})
    verify_checkpoint(
        row,
        {**configuration.__dict__, "key": configuration.key, "local_degree": configuration.ell + 2},
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
        ),
        archive_required=configuration.archive_fields,
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


def run(output: Path, studies: list[str], workers: int, resume: bool) -> None:
    """Acquire independent solves with source-checked atomic per-case checkpoints."""
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    for path in (Path(__file__), ROOT / "src/pymhm/loads.py"):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = output / "article.json"
    record: dict[str, Any] = {
        "reference": "Chaumont-Frelet and Valentin (2020), DOI 10.1137/19M1255616",
        "analytical_sections": ["6.2", "6.3"],
        "local_space_convention": "Q(ell+2), local refinement 2; separately tested with 4",
        "assembly_order": 10,
        "error_order": 12,
        "source_sha256": hashes,
        "rows": [],
    }
    if resume and path.exists():
        record = json.loads(path.read_text())
        require_sources(record["source_sha256"], hashes)
        verify_checkpoint(record, {"assembly_order": 10, "error_order": 12}, directory=output)
        for row in record["rows"]:
            validate_row(row, output)
    done = {row["key"] for row in record["rows"]}
    pending = [c for c in configurations(studies) if c.key not in done]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(solve_configuration, c, output): c for c in pending}
        for future in as_completed(futures):
            row = future.result()
            record["rows"].append(row)
            record["rows"].sort(key=lambda r: r["key"])
            temporary = path.with_suffix(".json.new")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(path)
            print(json.dumps(row), flush=True)
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
    args = parser.parse_args()
    run(args.output, args.studies, args.workers, args.resume)


if __name__ == "__main__":
    main()
