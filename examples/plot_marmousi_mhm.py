"""Render archived broken Marmousi pressure against a classical reference.

The maps sample material-cell centres without averaging incident macro fields.
Profiles retain both endpoints of every macro interval as separate segments.
All display limits include the complete native nodal ranges of both fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from examples.marmousi_campaign import evaluate_fields
from examples.marmousi_comparison import BrokenQField
from examples.marmousi_fields import PixelCGField, load_reference
from examples.marmousi_records import checked_mhm, checked_reference
from pymhm.io.workspace import case_workspace, read_resource_bytes

ROOT = case_workspace()


def centre_fields(
    candidate: BrokenQField, reference: PixelCGField, *, batch_size: int = 65536
) -> tuple[np.ndarray, np.ndarray]:
    """Sample both fields at original pixel centres, preserving every macro side."""
    dtype = np.result_type(candidate.pressure.dtype, reference.nodes.dtype)
    values = [np.empty(reference.counts, dtype=dtype) for _ in range(2)]
    lower = np.asarray(reference.bounds)[[0, 2]]
    for start in range(0, values[0].size, batch_size):
        index = np.arange(start, min(start + batch_size, values[0].size))
        cells = np.column_stack((index // reference.counts[1], index % reference.counts[1]))
        points = lower + (cells + 0.5) * reference.spacing
        values[0].ravel()[index] = evaluate_fields(
            candidate.pressure,
            candidate.mesh,
            candidate.refinement,
            candidate.degree,
            points,
        )
        values[1].ravel()[index] = reference.sample(points)
    return values[0], values[1]


def profile_segments(
    candidate: BrokenQField, depth: float, *, samples_per_macro: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Return unjoined macro-interval profiles with the proper incident endpoints.

    The horizontal line must avoid horizontal macro interfaces so that only
    its intersections with vertical faces have two incident values. Endpoints
    use exact geometric selectors, without displaced sampling coordinates.
    """
    mesh = candidate.mesh
    scaled_depth = (depth - mesh.bounds[2]) / mesh.spacing[1]
    if (
        not np.isfinite(depth)
        or not 0 < scaled_depth < mesh.ny
        or np.isclose(scaled_depth, round(scaled_depth), rtol=0, atol=1e-12)
        or samples_per_macro < 2
    ):
        raise ValueError("profile must lie strictly between horizontal macro interfaces")
    edges = np.linspace(mesh.bounds[0], mesh.bounds[1], mesh.nx + 1)
    x = edges[:-1, None] + np.linspace(0, 1, samples_per_macro) * mesh.spacing[0]
    points = np.column_stack((x.ravel(), np.full(x.size, depth)))
    values = evaluate_fields(
        candidate.pressure,
        mesh,
        candidate.refinement,
        candidate.degree,
        points,
        side=(1, 1),
    ).reshape(x.shape)
    right = np.column_stack((x[:, -1], np.full(mesh.nx, depth)))
    values[:, -1] = evaluate_fields(
        candidate.pressure,
        mesh,
        candidate.refinement,
        candidate.degree,
        right,
        side=(-1, 1),
    )
    return x, values


def main() -> None:
    """Render signed complex maps and one-sided profiles from verified field archives."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm

    from examples.plot_marmousi import panel, save

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/marmousi")
    args = parser.parse_args()
    candidate_record = checked_mhm(args.candidate)
    reference_record = checked_reference(args.reference)
    candidate = BrokenQField.load(args.candidate)
    reference = load_reference(args.reference)
    if (
        any(
            candidate_record[key] != reference_record[key]
            for key in ("material", "omega", "point_source")
        )
        or candidate.mesh.bounds != reference.bounds
    ):
        raise ValueError("the two fields must share the declared physical problem")
    args.output.mkdir(parents=True, exist_ok=True)
    values, truth = centre_fields(candidate, reference)
    difference = values - truth
    name = f"mhm-H{candidate_record['H_m']}-ell{candidate_record['trace_degree']}"
    title = f"MHM Q{candidate.degree}, ℓ={candidate_record['trace_degree']}"
    figure, axes = plt.subplots(2, 3, figsize=(11.4, 6.8), layout="constrained")
    display: dict[str, Any] = {}
    for row, (component, label) in enumerate(((np.real, "Re p"), (np.imag, "Im p"))):
        parts = [component(truth), component(values), component(difference)]
        common = max(
            float(np.max(abs(component(candidate.pressure)))),
            float(np.max(abs(component(reference.nodes)))),
            float(np.max(abs(parts[0]))),
            float(np.max(abs(parts[1]))),
        )
        display[label] = {
            "native_nodal_range_mhm": [
                float(component(candidate.pressure).min()),
                float(component(candidate.pressure).max()),
            ],
            "native_nodal_range_reference": [
                float(component(reference.nodes).min()),
                float(component(reference.nodes).max()),
            ],
            "shared_symmetric_limit": common,
        }
        for column, (part, heading) in enumerate(
            zip(parts, (f"Classical P{reference.degree}", title, "MHM − classical"), strict=True)
        ):
            extent = common if column < 2 else float(np.max(abs(part)))
            panel(
                axes[row, column],
                part,
                heading,
                f"{label}; asinh display",
                candidate.mesh,
                norm=AsinhNorm(linear_width=1.0, vmin=-extent, vmax=extent),
            )
    figure.suptitle(
        "Declared Marmousi crop; actual macro faces; material-cell-centre samples", fontsize=15
    )
    save(figure, args.output, f"{name}-fields")
    depth = 505.0
    x, pressure = profile_segments(candidate, depth)
    rx = np.linspace(reference.bounds[0], reference.bounds[1], 4097)
    rvalues = reference.sample(np.column_stack((rx, np.full_like(rx, depth))))
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 3.8), layout="constrained")
    for axis, component, label in zip(axes, (np.real, np.imag), ("Re p", "Im p"), strict=True):
        axis.plot(rx / 1000, component(rvalues), lw=1, label=f"Classical P{reference.degree}")
        for cell, (xx, pp) in enumerate(zip(x, pressure, strict=True)):
            axis.plot(
                xx / 1000, component(pp), color="#d55e00", lw=1, label=title if cell == 0 else None
            )
            axis.axvline(xx[0] / 1000, color="#66737a", lw=0.4, alpha=0.22, zorder=0)
        axis.set(
            xlabel="Horizontal distance (km)",
            ylabel=label,
            title=f"Depth {depth:g} m; independent incident macro values",
        )
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    save(figure, args.output, f"{name}-profiles")
    display.update(
        {
            "candidate_record": args.candidate.name,
            "candidate_record_sha256": hashlib.sha256(
                read_resource_bytes(args.candidate)
            ).hexdigest(),
            "reference_record": args.reference.name,
            "reference_record_sha256": hashlib.sha256(
                read_resource_bytes(args.reference)
            ).hexdigest(),
            "map_sampling": "Original material-cell centres; no incident-field averaging",
            "color_scale": "Signed asinh, linear width 1; physical-value ticks; no clipping",
            "profile_depth_m": depth,
            "profile_intersections": "Exact one-sided macro endpoints; no joins across faces",
            "source_sha256": {
                f"examples/{name}.py": hashlib.sha256(
                    read_resource_bytes(Path(__file__).with_name(f"{name}.py"))
                ).hexdigest()
                for name in (
                    "plot_marmousi_mhm",
                    "plot_marmousi",
                    "marmousi_campaign",
                    "marmousi_comparison",
                    "marmousi_fields",
                    "plot_mesh",
                )
            },
        }
    )
    (args.output / f"{name}-display.json").write_text(json.dumps(display, indent=2) + "\n")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_marmousi_mhm").main()
