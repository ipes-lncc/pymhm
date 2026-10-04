"""Render the public initial catalogue from hash-checked scalar JSON data only.

Run ``python examples/plot_initial_convergence.py --output build/initial-figures``
in the locked Pixi notebooks environment. No PDE driver, coefficient archive,
external solver or private acquisition source is imported or read. Errors,
field norms and numerical differences retain separate physical interpretations;
the renderer does not compute convergence rates or certify a reference solution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "examples/results/minimal-convergence/catalogue.json"
FIELDS = {
    "pressure_l2": "Pressure",
    "flux_l2": "Darcy flux",
    "gradient_l2": "Pressure gradient",
    "displacement_l2": "Displacement",
    "velocity_l2": "Velocity",
    "stress_l2": "Cauchy stress",
    "electric_l2": "Electric field",
    "magnetic_l2": "Magnetic field",
    "rotation_l2": "Weak rotation",
    "velocity_h1_seminorm": "Velocity gradient",
    "displacement_h1_seminorm": "Displacement gradient",
}


@dataclass
class Curve:
    """One declared series of scalar observations, without interpolation."""

    label: str
    x: list[float]
    y: list[float]


@dataclass
class Panel:
    """Separate physical metric and its explicitly stated normalization."""

    title: str
    xlabel: str
    ylabel: str
    curves: list[Curve]


def digest(path: Path) -> str:
    """Return SHA256 of the literal catalogue, record or rendered artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object without following its provenance or archive paths."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def explicit_panels(record: dict[str, Any]) -> list[Panel]:
    """Preserve already declared scalar, transport and periodic panel semantics."""
    return [
        Panel(
            title=panel["title"],
            xlabel=panel["xlabel"],
            ylabel=panel["ylabel"],
            curves=[
                Curve(series["label"], list(series["x"]), list(series["y"]))
                for series in panel["series"]
            ],
        )
        for panel in record["panels"]
    ]


def wave_panels(record: dict[str, Any]) -> list[Panel]:
    """Separate exact wave errors, physical norms and finest-field differences.

    Missing finest self-differences are absent observations, never zero errors.
    Recorded increments in these pilots compare each listed coarse field with
    the stated finest numerical field, rather than with a continuum solution.
    """
    panels = []
    reference = (
        f"dt={record['finest_numerical_reference_time_step_s']:g} s"
        if "finest_numerical_reference_time_step_s" in record
        else f"n={record['finest_numerical_reference_resolution']}"
        if "finest_numerical_reference_resolution" in record
        else f"macro {record.get('finest_numerical_reference_macro_shape')}"
    )
    for field in record["plot_fields"]:
        base = field.replace("_increment_l2", "_l2")
        if base not in FIELDS:
            raise ValueError(f"Unsupported physical wave metric: {field}")
        rows = [row for row in record["rows"] if field in row["norms"]]
        if "_increment_l2" in field:
            title = f"{FIELDS[base]}: difference from {reference}"
            ylabel = "L2 difference from finest numerical field"
        elif record["exact_solution_available"]:
            title, ylabel = FIELDS[base], "L2 error against stated exact solution"
        else:
            title, ylabel = FIELDS[base], "Physical L2 field norm"
        panels.append(
            Panel(
                title,
                record["level_label"],
                ylabel,
                [
                    Curve(
                        "Recorded field",
                        [row["level"] for row in rows],
                        [row["norms"][field] for row in rows],
                    )
                ],
            )
        )
    return panels


def flow_elasticity_panels(record: dict[str, Any]) -> list[Panel]:
    """Group fixed formulations and keep each analytical field error separate."""
    group = record["group"]
    fields = (
        ("velocity_l2", "pressure_l2", "velocity_h1_seminorm")
        if group in {"flow3d", "stokes2d", "oseen2d-trace", "oseen2d-viscosity", "oseen2d-data"}
        else ("displacement_l2", "pressure_l2", "stress_l2", "displacement_h1_seminorm")
        if group == "gals3d"
        else ("displacement_l2", "stress_l2", "rotation_l2", "divergence_l2")
        if group in {"initial-elasticity2d", "mixed-elasticity3d"}
        else ("displacement_l2", "stress_l2")
        if group == "primal-elasticity3d"
        else ("displacement_l2", "stress_l2", "rotation_l2", "stress_divergence_l2")
        if group == "elasticity-l18-oscillatory"
        else ()
    )
    if not fields:
        raise ValueError(f"Unsupported flow/elasticity group: {group}")
    panels = []
    for field in fields:
        curves = []
        for case in dict.fromkeys(row["case"] for row in record["rows"]):
            rows = sorted(
                (row for row in record["rows"] if row["case"] == case),
                key=lambda row: row["resolution"],
            )
            curves.append(
                Curve(
                    case,
                    [row["resolution"] for row in rows],
                    [row["terminal_norms"][field] for row in rows],
                )
            )
        title = (
            "Stress divergence"
            if field in {"divergence_l2", "stress_divergence_l2"}
            else FIELDS[field]
        )
        if field == "pressure_l2" and group == "gals3d":
            title = "Herrmann pressure"
        xlabel = (
            "Face segments s (fixed 32 macros; coupled local refinement)"
            if group == "elasticity-l18-oscillatory"
            else "Declared macro resolution n"
        )
        panels.append(Panel(title, xlabel, "L2 analytical error", curves))
    return panels


def darcy_panels(record: dict[str, Any]) -> list[Panel]:
    """Interpret the six public Darcy records using their physical metric keys."""
    schema = record["schema"]
    rows = record.get("rows", [])
    xlabel, ylabel = "Declared macro resolution n", "L2 analytical error"
    if schema == "pymhm-minimal-unfitted-trace-convergence-v1":
        x = [row["segments"] for row in rows]
        xlabel = "Trace segments s (fixed local and macro meshes)"
        p = [row["pressure_l2_error"] for row in rows]
        q = [row["flux_l2_error"] for row in rows]
    elif schema == "pymhm-minimal-quarter-classical-reuse-v1":
        differences = record["refinement_differences"]
        x = [row["fine_refinement"] for row in differences]
        p = [row["pressure_l2_relative"] for row in differences]
        q = [row["flux_l2_relative"] for row in differences]
        xlabel = "Finer classical refinement r"
        ylabel = "Relative L2 successive increment / finer field norm"
    elif record.get("case") == "spe10":
        rows = rows[1:]
        norms = [row["norms"][str(row["norm_orders"][-1])] for row in rows]
        x = [row["level"] for row in rows]
        p = [norm["pressure_relative_increment"] for norm in norms]
        q = [norm["flux_relative_increment"] for norm in norms]
        xlabel = "Finer local refinement r (fixed macro and trace meshes)"
        ylabel = "Relative L2 successive increment / finer field norm"
    else:
        x = [row["level"] for row in rows]
        p = [row["pressure_l2_error"] for row in rows]
        q = [row["flux_l2_error"] for row in rows]
        if record["case"] == "mixedwell":
            xlabel = "Fine refinement F (fixed macro mesh; two accepted levels)"
        elif record["case"] != "pgmhm" and record["case"] != "unusual":
            raise ValueError(f"Unsupported Darcy case: {record['case']}")
    names = (
        ("Scalar", "Diffusion flux")
        if record.get("case") == "unusual"
        else ("Pressure", "Darcy flux")
    )
    if record.get("case") == "pgmhm":
        names = ("Enriched pressure", "Darcy flux from enriched raw gradient")
    return [
        Panel(name, xlabel, ylabel, [Curve("Recorded field", x, values)])
        for name, values in zip(names, (p, q), strict=True)
    ]


def panels_for(record: dict[str, Any]) -> list[Panel]:
    """Dispatch only explicitly supported public scalar-record schemas."""
    schema = record.get("schema")
    if schema in {
        "pymhm-initial-scalar-publication-v1",
        "pymhm-initial-transport-temporal-consumer-v1",
        "pymhm-initial-periodic-reference-consumer-v1",
    }:
        return explicit_panels(record)
    if schema == "pymhm-initial-wave-convergence-v1":
        return wave_panels(record)
    if schema == "pymhm-initial-flow-elasticity-convergence-v1":
        return flow_elasticity_panels(record)
    if schema in {
        "pymhm-minimal-darcy-convergence-v1",
        "pymhm-minimal-unfitted-trace-convergence-v1",
        "pymhm-minimal-quarter-classical-reuse-v1",
    }:
        return darcy_panels(record)
    raise ValueError(f"Unsupported public record schema: {schema}")


def validate_panels(panels: list[Panel]) -> None:
    """Require nonnegative finite physical observations and positive resolution."""
    if not panels:
        raise ValueError("No physical metric panels were declared")
    for panel in panels:
        if not panel.curves:
            raise ValueError(f"No series in panel: {panel.title}")
        for curve in panel.curves:
            if not curve.x or len(curve.x) != len(curve.y):
                raise ValueError(f"Inconsistent observations: {panel.title} / {curve.label}")
            if any(not math.isfinite(x) or x <= 0 for x in curve.x):
                raise ValueError("Resolution observations must be positive and finite")
            if any(not math.isfinite(y) or y < 0 for y in curve.y):
                raise ValueError("Physical norms must be nonnegative and finite")


def render_panels(title: str, panels: list[Panel], output: Path) -> None:
    """Save PNG/SVG with individual metrics, explicit ticks and separated regions."""
    validate_panels(panels)
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    ticker = importlib.import_module("matplotlib.ticker")
    columns = min(2, len(panels))
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.2, 4.2 * rows + 0.8),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(textwrap.fill(title, width=85), fontsize=14)
    for axis, panel in zip(axes.flat, panels, strict=False):
        values = [y for curve in panel.curves for y in curve.y]
        axis.set_xscale("log")
        if all(y > 0 for y in values):
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
            axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.6g}"))
        for curve in panel.curves:
            axis.plot(curve.x, curve.y, "o-", label=curve.label, linewidth=1.8)
        ticks = sorted({x for curve in panel.curves for x in curve.x})
        axis.set_xticks(ticks, [f"{x:g}" for x in ticks])
        axis.xaxis.set_minor_locator(ticker.NullLocator())
        axis.yaxis.set_minor_formatter(ticker.NullFormatter())
        axis.set_title(textwrap.fill(panel.title, width=48), fontsize=11)
        axis.set_xlabel(textwrap.fill(panel.xlabel, width=48), fontsize=10)
        axis.set_ylabel(textwrap.fill(panel.ylabel, width=40), fontsize=10)
        axis.grid(True, which="both", alpha=0.23)
        if len(panel.curves) > 1 or panel.curves[0].label != "Recorded field":
            axis.legend(fontsize=8)
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_visible(False)
    figure.savefig(output, dpi=120)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    """Render every hash-checked catalogue entry into a new output directory.

    Unsupported schemas and digest mismatches are explicit failures. Inputs and
    published figures are never overwritten. The output receipt records plotted
    scalars and identities, without assigning new producer UUIDs or source hashes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalogue = read_json(args.catalogue)
    prepared = []
    for entry in catalogue["entries"]:
        path = ROOT / entry["public_record"]
        if digest(path) != entry["record_sha256"]:
            raise ValueError(f"Catalogue record_sha256 mismatch: {path}")
        panels = panels_for(read_json(path))
        validate_panels(panels)
        prepared.append((entry, panels))
    args.output.mkdir(parents=True, exist_ok=False)
    rendered = []
    for entry, panels in prepared:
        directory = args.output / entry["id"]
        directory.mkdir()
        output = directory / "convergence.png"
        render_panels(entry["title"], panels, output)
        rendered.append(
            {
                "id": entry["id"],
                "public_record": entry["public_record"],
                "record_sha256": entry["record_sha256"],
                "panels": [asdict(panel) for panel in panels],
                "png_sha256": digest(output),
                "svg_sha256": digest(output.with_suffix(".svg")),
            }
        )
    receipt = {
        "schema": "pymhm-initial-convergence-rendering-v1",
        "catalogue_sha256": digest(args.catalogue),
        "renderer_sha256": digest(Path(__file__)),
        "pde_solves": 0,
        "coefficient_archives_read": 0,
        "producer_acquisitions_retagged": False,
        "entries": rendered,
    }
    (args.output / "rendering.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    print(f"Rendered {len(rendered)} hash-checked public records into {args.output}")


if __name__ == "__main__":
    main()
