"""Sample accepted smooth Taylor--Hood solutions without coefficient replay.

The unchanged acquisition owner creates and verifies both MHM and independently
assembled native classical solutions. A measurement observer retains those
executed physical fields while delegating every norm call unchanged. Spatial
sampling reuses named nodal evaluators and keeps one-sided macrocell values.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

from examples import tutorial_stokes_taylor_hood_acquisition as acquisition
from examples.plot_mesh import draw_macro_mesh
from examples.solve_stokes_adaptive import StokesData
from pymhm.io.provenance import file_digest
from pymhm.postprocessing.fields import DiscreteField
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.solutions import VectorSolution

matplotlib.use("Agg")
LICENSE = "CC BY 4.0; IPES Research Group; https://creativecommons.org/licenses/by/4.0/"


@contextmanager
def observe_physical_fields() -> Iterator[list[VectorSolution]]:
    """Retain one actual solution and delegate every physical-error integration."""
    solutions: list[VectorSolution] = []
    original = acquisition.physical_errors

    def measure(solution: VectorSolution, order: int) -> dict[str, float]:
        """Perform the original norm operation before observing its same solution."""
        result = original(solution, order)
        if not solutions:
            solutions.append(solution)
        elif solutions[0] is not solution:
            raise ValueError("One physical solution is required per measurement observation")
        return result

    with patch.object(acquisition, "physical_errors", measure):
        yield solutions


def spatial_samples(solution: VectorSolution) -> dict[str, Any]:
    """Evaluate physical fields at every fine-cell centroid without interface merging."""
    data = StokesData()
    points, cells, values = [], [], []
    offset = 0
    for fine, velocity, pressure in zip(
        solution.local_meshes, solution.values, solution.pressure, strict=True
    ):
        centers = fine.points[fine.cells].mean(axis=1)
        ownership = np.arange(len(fine.cells))
        u_field = DiscreteField(
            nodal_field("velocity", fine, solution.degree, components=2), velocity.ravel()
        )
        p_field = DiscreteField(nodal_field("pressure", fine, solution.pressure_degree), pressure)
        actual_u, actual_p = (
            u_field.evaluate(centers, cells=ownership),
            p_field.evaluate(centers, cells=ownership),
        )
        exact_u, exact_p = data.velocity(centers), data.pressure(centers)
        values.append(
            np.column_stack(
                (
                    np.linalg.norm(exact_u, axis=1),
                    np.linalg.norm(actual_u, axis=1),
                    exact_p,
                    actual_p,
                    np.linalg.norm(actual_u - exact_u, axis=1),
                    abs(actual_p - exact_p),
                )
            )
        )
        points.append(fine.points)
        cells.append(fine.cells + offset)
        offset += len(fine.points)
    return {
        "sampling": "independent fine-cell centroid values; no averaging across macrofaces",
        "points": np.vstack(points).tolist(),
        "cells": np.vstack(cells).tolist(),
        "columns": [
            "analytical_velocity_magnitude",
            "velocity_magnitude",
            "analytical_pressure",
            "pressure",
            "velocity_error_magnitude",
            "absolute_pressure_error",
        ],
        "values": np.vstack(values).tolist(),
    }


def export_figures(record: dict[str, Any], output: Path) -> list[dict[str, str]]:
    """Draw exact, MHM and classical fields with actual comparison macro boundaries."""
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    macro = acquisition.TriangleMesh(
        np.asarray(record["macro_points"]), np.asarray(record["macro_cells"])
    )
    samples = [record["mhm_samples"], record["reference_samples"]]
    arrays = [np.asarray(sample["values"]) for sample in samples]
    triangulations = [
        Triangulation(*np.asarray(sample["points"]).T, np.asarray(sample["cells"]))
        for sample in samples
    ]
    figures = []
    fields, axes = plt.subplots(2, 3, figsize=(12, 7.2), layout="constrained")
    for row, (label, exact_column, actual_column) in enumerate(
        (("Velocity magnitude", 0, 1), ("Pressure", 2, 3))
    ):
        values = [
            arrays[0][:, exact_column],
            arrays[0][:, actual_column],
            arrays[1][:, actual_column],
        ]
        lower, upper = min(value.min() for value in values), max(value.max() for value in values)
        for column, (title, value, triangulation) in enumerate(
            zip(
                ("Analytical", "MHM: n=8, r=4", "Classical Taylor–Hood: n=64"),
                values,
                (triangulations[0], triangulations[0], triangulations[1]),
                strict=True,
            )
        ):
            axis = axes[row, column]
            image = axis.tripcolor(
                triangulation,
                facecolors=value,
                shading="flat",
                cmap="viridis",
                vmin=lower,
                vmax=upper,
                rasterized=True,
            )
            draw_macro_mesh(axis, macro)
            axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            fields.colorbar(image, ax=axis, label=label, fraction=0.046, pad=0.035)
    fields.suptitle("Smooth Stokes: ν=1, γ=0; macro boundaries shown on every panel")
    figures.append((fields, "stokes-taylor-hood-fields"))
    errors, axes = plt.subplots(2, 2, figsize=(9.4, 8.3), layout="constrained")
    for row, (label, column) in enumerate(
        (("Velocity error magnitude", 4), ("Absolute pressure error", 5))
    ):
        for method, axis, triangulation, array in zip(
            ("MHM", "Classical Taylor–Hood"), axes[row], triangulations, arrays, strict=True
        ):
            image = axis.tripcolor(
                triangulation,
                facecolors=array[:, column],
                shading="flat",
                cmap="magma",
                vmin=0,
                rasterized=True,
            )
            draw_macro_mesh(axis, macro)
            axis.set(title=method, xlabel="x", ylabel="y", aspect="equal")
            errors.colorbar(image, ax=axis, label=label, fraction=0.046, pad=0.035)
    errors.suptitle("Physical errors against the analytical solution; separate error scales")
    figures.append((errors, "stokes-taylor-hood-errors"))
    outputs = []
    description = (
        "Smooth positive-stream-function Stokes qualification; local P2/P1-r4, vector P1 "
        "trace, n8 MHM; independent native classical P2/P1 n64. Macro boundaries are the "
        "actual n8 MHM mesh; cell-centroid values retain independent macrocell ownership. "
        + LICENSE
    )
    for figure, stem in figures:
        for suffix in ("png", "svg", "pdf"):
            target = output / f"{stem}.{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            metadata = (
                {"Author": "IPES Research Group", "Description": description, "Copyright": LICENSE}
                if suffix == "png"
                else {
                    "Creator": "IPES Research Group",
                    "Description": description,
                    "Rights": LICENSE,
                    "Date": None,
                }
                if suffix == "svg"
                else {
                    "Author": "IPES Research Group",
                    "Subject": description,
                    "CreationDate": None,
                    "ModDate": None,
                }
            )
            figure.savefig(target, dpi=240, bbox_inches="tight", pad_inches=0.06, metadata=metadata)
            if suffix == "svg":
                target.write_text(
                    "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
                )
            outputs.append({"path": target.as_posix(), "sha256": file_digest(target)})
        preview = Path("build/docs-restructure/vector-canonical-previews") / f"{stem}-850.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            preview, dpi=850 / figure.get_figwidth(), bbox_inches="tight", pad_inches=0.06
        )
        plt.close(figure)
    return outputs


def main() -> None:
    """Acquire two unchanged physical solutions, verify sources, then sample and plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("A physical sampling record already exists")
    root = Path(__file__).resolve().parents[1]
    hashes = acquisition.source_manifest(root)
    with threadpool_limits(1):
        with observe_physical_fields() as mhm:
            mhm_row = acquisition.acquire_level(8, 1)
        with observe_physical_fields() as classical:
            classical_row = acquisition.native_reference(64)
        mhm_samples, reference_samples = spatial_samples(mhm[0]), spatial_samples(classical[0])
    if not acquisition.sources_unchanged(root, hashes):
        raise ValueError("A physical field's numerical source changed during execution")
    macro = mhm[0].skeleton.mesh
    report = {
        "schema": "pymhm-sampled-physical-fields-v1",
        "problem": "smooth Stokes with positive polynomial stream function",
        "exact_data": "psi=+128*x²*(1-x)²*y²*(1-y)²; u=(psi_y,-psi_x); p=150*(x-.5)*(y-.5)",
        "literature_scope": (
            "same polynomial degree and regularity as Araya et al. (2017); "
            "opposite velocity sign to section 3.1.1; analytical qualification, "
            "not matched figure reproduction"
        ),
        "source_sha256": hashes,
        "source_unchanged": True,
        "mhm": mhm_row,
        "reference": classical_row,
        "macro_points": macro.points.tolist(),
        "macro_cells": macro.cells.tolist(),
        "mhm_samples": mhm_samples,
        "reference_samples": reference_samples,
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "license": LICENSE,
        "accepted": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    outputs = export_figures(report, root / "docs/assets/tutorials/methods")
    for output in outputs:
        output["path"] = Path(output["path"]).relative_to(root).as_posix()
    receipt = {
        "record": args.output.as_posix(),
        "sha256": file_digest(args.output),
        "figures": outputs,
    }
    (root / "build/docs-restructure/stokes-taylor-hood-field-figures.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
