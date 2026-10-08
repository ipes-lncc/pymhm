"""Acquire the oscillatory MH2M equation with explicit, identifiable data conventions.

Equation (61) and epsilon=1/14 follow arXiv:2404.16978v3, Section 8.2.
The predecessor dissertation by Barros (2022), Equations4.1/4.4, specifies
gamma=1.8 and the source used here. Its epsilon=1/17 differs from the
article-v3 value1/14; mesh and array identity are not inferred.
"""

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
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.campaign_provenance import (
    index_records,
    positive_integers,
    require_equal,
    verify_archive,
)
from examples.formulations.application import darcy as solve_darcy
from examples.formulations.application import three_field_diffusion as solve_mh2m
from examples.mh2m_campaign import diagnostics, source
from examples.mh2m_campaign_contracts import verify_difference_result as verify_result
from examples.mh2m_heterogeneous_norms import StructuredP1, difference
from pymhm.fem.scalar.triangle import scalar_operators
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.pressure_2d import PressureTraceSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.linalg.linear import solve_linear
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/mh2m-heterogeneous"


@dataclass(frozen=True)
class OscillatoryCoefficient:
    """Evaluate the literal equation (61), with a declared gamma and epsilon."""

    gamma: float = 1.8
    epsilon: float = 1 / 14

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Return the positive scalar permeability on physical unit-square points."""
        x, y = 2 * np.pi * points.T / self.epsilon
        return (2 + self.gamma * np.sin(x)) / (2 + self.gamma * np.cos(y)) + (2 + np.sin(y)) / (
            2 + self.gamma * np.sin(x)
        )


def source_hashes() -> dict[str, str]:
    """Identify all executed numerical sources, including shared norm and archive owners."""
    names = [
        "examples/mh2m_heterogeneous.py",
        "examples/campaign_provenance.py",
        "examples/mh2m_campaign_contracts.py",
        "examples/mh2m_heterogeneous_norms.py",
        "examples/mh2m_campaign.py",
        *(
            f"src/pymhm/{name}.py"
            for name in (
                "methods/three_field",
                "_legacy/models/darcy/primal",
                "fem/scalar/triangle",
                "fem/scalar/operators",
                "meshes/triangle",
                "fem/quadrature/material",
                "core/contracts",
                "linalg/linear",
            )
        ),
    ]
    return current_source_manifest(
        {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    )


def save_field(path: Path, mesh: TriangleMesh, fine: tuple, pressure: tuple) -> StructuredP1:
    """Archive independent physical fine triangles and their local nodal pressures."""
    vertices = np.concatenate([local.points[local.cells] for local in fine])
    values = np.concatenate([p[local.cells] for p, local in zip(pressure, fine, strict=True)])
    np.savez_compressed(
        path, vertices=vertices, pressure=values, macro_points=mesh.points, macro_cells=mesh.cells
    )
    return StructuredP1.from_arrays(vertices, values)


def load_field(path: Path) -> StructuredP1:
    """Restore archived fields using their actual physical triangle coordinates."""
    with np.load(path) as arrays:
        return StructuredP1.from_arrays(arrays["vertices"], arrays["pressure"])


def data_conventions() -> dict[str, Any]:
    """Identify the physical data and their primary sources without inferring historical arrays."""
    return {
        "gamma": 1.8,
        "epsilon": "1/14",
        "source": "-2*x*(x-1)-2*y*(y-1)",
        "boundary": "homogeneous Dirichlet",
        "coefficient": "literal equation (61)",
        "gamma_provenance": "Barros (2022), Section 4.2.2.1, Equation (4.4), gamma=1.8",
        "source_provenance": "Barros (2022), Equation (4.1)",
        "historical_scope": (
            "Dissertation data with article-v3 epsilon=1/14, rather than dissertation "
            "epsilon=1/17; original figure arrays are not supplied."
        ),
        "reference": "classical conforming P1; separate assembly, shared FEM kernels",
    }


def case_configurations(resolutions: list[int]) -> list[tuple[str, int, int, int, int, str]]:
    """Declare the campaign spaces once for both acquisition and resume validation."""
    configurations = []
    for n in resolutions:
        configurations.extend(
            [
                ("MH2M", n, 16, 1, 8, "global-refinement"),
                ("MHM", max(1, n // 2), 16, 1, 1, "global-refinement"),
            ]
        )
    configurations.extend(
        ("MH2M", 16, r, 1, s, "fixed-Gamma") for r in (16, 32) for s in (1, 2, 4, 8)
    )
    configurations.append(("MH2M", 31, 4, 2, 2, "figure-5-discretization"))
    return configurations


def validate_campaign_resume(
    record: dict[str, Any],
    output: Path,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check immutable numerical inputs and all completed archives before any reuse."""
    stored = record.get("mathematical_configuration")
    if stored is None:
        references = record["references"]
        resolutions = [
            row["macro_resolution"]
            for row in record["cases"]
            if row["method"] == "MH2M" and row["family"] == "global-refinement"
        ]
        if not references or not resolutions:
            raise ValueError("legacy campaign lacks a complete mathematical configuration")
        stored = dict(
            references=[row["resolution"] for row in references],
            resolutions=resolutions,
            assembly_order=references[0]["assembly_order"],
            norm_orders=[6, 8],
        )
    configuration = stored if configuration is None else configuration
    require_equal(stored, configuration, label="mathematical configuration")
    positive_integers(configuration["references"], label="references", increasing=True)
    positive_integers(configuration["resolutions"], label="resolutions", increasing=True)
    positive_integers([configuration["assembly_order"]], label="assembly quadrature")
    require_equal(configuration["norm_orders"], [6, 8], label="norm quadratures")
    ordered_references = [row["resolution"] for row in record["references"]]
    require_equal(
        ordered_references,
        configuration["references"][: len(ordered_references)],
        label="acquired reference sequence",
    )
    configured: dict[str, dict[str, Any]] = {}
    for method, n, r, g, flux, family in case_configurations(configuration["resolutions"]):
        name = f"{method.lower()}-n{n}-r{r}-g{g}-l{flux}.npz"
        configured.setdefault(
            name,
            dict(
                method=method,
                family=family,
                macro_resolution=n,
                local_refinement=r,
                Gamma_degree=1 if method == "MH2M" else None,
                Gamma_segments=g if method == "MH2M" else None,
                Lambda_segments=flux,
            ),
        )
    order = configuration["assembly_order"]
    previous = None
    for row in index_records(record["references"], key="archive").values():
        if row["resolution"] not in configuration["references"]:
            raise ValueError("reference is outside the requested resolution sequence")
        require_equal(row["assembly_order"], order, label="reference assembly quadrature")
        verify_archive(output / row["archive"], row["archive_sha256"])
        if previous is not None:
            view = {
                **row,
                "norms": {
                    f"quadrature_{q}": row[key]
                    for q, key in ((6, "increment"), (8, "increment_quadrature_check"))
                    if key in row
                },
            }
            verify_result(
                view,
                {"assembly_order": order},
                archives={
                    output / row["archive"]: row["archive_sha256"],
                    output / previous["archive"]: previous["archive_sha256"],
                },
                norm_orders=[6, 8],
            )
        previous = row
    if record["cases"] and previous is None:
        raise ValueError("completed cases require an archived reference")
    for row in index_records(record["cases"], key="archive").values():
        if row["archive"] not in configured:
            raise ValueError("completed case is outside the declared campaign spaces")
        view = {
            **row,
            "norms": {
                f"quadrature_{q}": row[key]
                for q, key in ((6, "norms"), (8, "norms_quadrature_check"))
                if key in row
            },
        }
        reference = next(
            (
                item
                for item in record["references"]
                if item["resolution"] == configuration["references"][-1]
            ),
            None,
        )
        if reference is None:
            raise ValueError("completed cases require the declared finest reference")
        verify_result(
            view,
            dict(
                **configured[row["archive"]],
                assembly_order=order,
                local_degree=1,
                Lambda_degree=0,
                reference_resolution=reference["resolution"],
            ),
            archives={
                output / row["archive"]: row["archive_sha256"],
                output / reference["archive"]: reference["archive_sha256"],
            },
            norm_orders=[6, 8],
        )

    return configuration


def run(
    output: Path,
    references: list[int],
    resolutions: list[int],
    order: int,
    *,
    new_source_phase: bool = False,
    norm_workers: int = 1,
) -> None:
    """Acquire reference refinement, global refinement and independent Lambda enrichment."""
    configuration = dict(
        references=list(positive_integers(references, label="references", increasing=True)),
        resolutions=list(positive_integers(resolutions, label="resolutions", increasing=True)),
        assembly_order=positive_integers([order], label="assembly quadrature")[0],
        norm_orders=[6, 8],
    )
    output.mkdir(parents=True, exist_ok=True)
    hashes, material = source_hashes(), OscillatoryCoefficient()
    path = output / "comparison.json"
    record: dict[str, Any] = {
        "paper": "de Barros, Madureira, Valentin, arXiv:2404.16978v3, Section 8.2",
        "conventions": data_conventions(),
        "references": [],
        "cases": [],
        "source_sha256": hashes,
        "mathematical_configuration": configuration,
    }
    conventions = record["conventions"]
    if path.exists():
        record = json.loads(path.read_text())
        physical_keys = ("gamma", "epsilon", "source", "boundary", "coefficient")
        require_equal(
            {key: record["conventions"][key] for key in physical_keys},
            {key: conventions[key] for key in physical_keys},
            label="physical conventions",
        )
        validate_campaign_resume(record, output, configuration)
        if record["source_sha256"] != hashes:
            if not new_source_phase:
                raise ValueError(
                    "changed sources require --new-source-phase or --output NEW_DIRECTORY; "
                    "validate archived records with python -m examples.validate_mh2m_campaign"
                )
            prior = record.setdefault(
                "acquisition_sources",
                [
                    {
                        "source_sha256": record["source_sha256"],
                        "source_changed_during_run": record.get("source_changed_during_run", False),
                    }
                ],
            )
            for row in (*record["references"], *record["cases"]):
                row.setdefault("source_phase", 0)
            prior.append({"source_sha256": hashes, "source_changed_during_run": False})
            record["source_sha256"] = hashes
    phases = record.setdefault(
        "acquisition_sources", [{"source_sha256": hashes, "source_changed_during_run": False}]
    )
    phase = len(phases) - 1

    def checkpoint() -> None:
        """Persist completed cases with a source-integrity gate after every solve."""
        validate_campaign_resume(record, output, configuration)
        record["source_changed_during_run"] = hashes != source_hashes()
        phases[phase]["source_changed_during_run"] = record["source_changed_during_run"]
        path.write_text(json.dumps(record, indent=2) + "\n")
        if record["source_changed_during_run"]:
            raise RuntimeError("numerical acquisition sources changed during the campaign")

    previous = None
    for n in references:
        name = f"reference-n{n}.npz"
        existing = next((row for row in record["references"] if row["resolution"] == n), None)
        if existing is None:
            start = perf_counter()
            mesh = TriangleMesh.unit_square(n)
            matrix, _, load = scalar_operators(
                mesh, 1, diffusion=material, source=source, order=order
            )
            fixed = np.unique(mesh.faces[mesh.boundary_faces])
            free = np.setdiff1d(np.arange(len(mesh.points)), fixed)
            values = np.zeros(len(mesh.points))
            values[free] = solve_linear(matrix[free][:, free], load[free])
            field = save_field(output / name, mesh, (mesh,), (values,))
            row = {
                "source_phase": phase,
                "resolution": n,
                "dofs": len(values),
                "assembly_order": order,
                "residual": float(
                    np.linalg.norm((matrix @ values - load)[free]) / np.linalg.norm(load[free])
                ),
                "archive": name,
                "archive_sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                "elapsed_seconds": perf_counter() - start,
            }
            if previous is not None:
                row["increment"] = difference(field, previous, material, 6, workers=norm_workers)
                row["increment_quadrature_check"] = difference(
                    field, previous, material, 8, workers=norm_workers
                )
            record["references"].append(row)
            checkpoint()
            print(json.dumps({"reference": row}), flush=True)
        else:
            field = load_field(output / existing["archive"])
        previous = field
    if previous is None:
        raise ValueError("at least one conforming reference is required")
    reference = previous
    configurations = case_configurations(resolutions)
    for method, n, refinement, gamma_segments, flux_segments, family in configurations:
        name = f"{method.lower()}-n{n}-r{refinement}-g{gamma_segments}-l{flux_segments}.npz"
        if any(row["archive"] == name for row in record["cases"]):
            continue
        start = perf_counter()
        mesh = TriangleMesh.unit_square(n)
        flux_space = SkeletonSpace(
            mesh, tuple(FaceSpace.uniform(0, flux_segments) for _ in mesh.faces)
        )
        common = dict(
            permeability=material,
            source=source,
            degree=1,
            local_refinement=refinement,
            quadrature_order=order,
        )
        if method == "MH2M":
            result = solve_mh2m(
                mesh,
                pressure_trace=PressureTraceSpace.uniform(mesh, 1, gamma_segments),
                flux_space=flux_space,
                **common,
            )
            fine = tuple(local.mesh for local in result.local)
            info = diagnostics(result)
            total = result.trace_space.size
            free = len(result.free_dofs)
        else:
            result = solve_darcy(mesh, skeleton=flux_space, **common)
            fine = result.local_meshes
            total = len(result.hybrid.trace) + len(result.hybrid.coarse)
            free = total
            info = {
                "macro_balance_max": float(np.max(abs(result.conservation_residuals()))),
                "algebraic_residual": result.hybrid.residual,
            }
        solved = perf_counter()
        field = save_field(output / name, mesh, fine, result.pressure)
        row = {
            "source_phase": phase,
            "method": method,
            "family": family,
            "macro_resolution": n,
            "macro_cells": len(mesh.cells),
            "local_degree": 1,
            "local_refinement": refinement,
            "Gamma_degree": 1 if method == "MH2M" else None,
            "Gamma_segments": gamma_segments if method == "MH2M" else None,
            "Lambda_degree": 0,
            "Lambda_segments": flux_segments,
            "global_dofs_total": total,
            "global_dofs_free": free,
            "assembly_order": order,
            "reference_resolution": reference.resolution,
            "norm_workers": norm_workers,
            "diagnostics": info,
            "norms": difference(reference, field, material, 6, workers=norm_workers),
            "norms_quadrature_check": difference(
                reference, field, material, 8, workers=norm_workers
            ),
            "archive": name,
            "archive_sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
            "solve_seconds": solved - start,
            "elapsed_seconds": perf_counter() - start,
        }
        record["cases"].append(row)
        checkpoint()
        print(json.dumps(row), flush=True)


def main() -> None:
    """Run a separate numerical campaign with per-case checkpoints and fixed data conventions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--references", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--resolutions", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--norm-workers", type=int, default=1)
    parser.add_argument(
        "--new-source-phase",
        action="store_true",
        help="retain earlier fields under their recorded sources and start a new source phase",
    )
    args = parser.parse_args()
    with threadpool_limits(1):
        run(
            args.output,
            args.references,
            args.resolutions,
            args.order,
            new_source_phase=args.new_source_phase,
            norm_workers=args.norm_workers,
        )


if __name__ == "__main__":
    main()
