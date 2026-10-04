"""Render initial Darcy/scalar convergence records without acquiring another PDE."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Identify literal record, field or executed source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_source_copies(record: dict[str, Any], directory: Path) -> int:
    """Validate immutable executed source copies rather than mutable checkout owners."""
    for name, expected in record["source_sha256"].items():
        path = directory / "executed-sources" / name
        if digest(path) != expected:
            raise ValueError(f"Executed source copy differs: {name}")
    return len(record["source_sha256"])


def render(
    directory: Path,
    title: str,
    levels: list[int],
    pressure: list[float],
    flux: list[float],
    xlabel: str,
    relative: bool,
    scalar: bool = False,
) -> dict[str, Any]:
    """Save distinct field panels with reserved title, labels and footer regions."""
    with plt.rc_context({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11}):
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), layout=None)
        fig.subplots_adjust(left=0.10, right=0.975, bottom=0.20, top=0.79, wspace=0.37)
        fig.suptitle(title, y=0.965, fontsize=15)
        fig.text(
            0.5,
            0.89,
            "Initial refinement increments; no converged reference"
            if relative
            else "Initial field errors against the stated analytical solution",
            ha="center",
            fontsize=11,
        )
        names = ("Scalar", "Diffusion flux") if scalar else ("Pressure", "Darcy flux")
        for ax, values, name, color in zip(
            axes, (pressure, flux), names, ("#2563a6", "#b44438"), strict=True
        ):
            ax.loglog(levels, values, "o-", color=color, lw=2, markersize=6)
            ax.set_title(name)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Relative L2 increment" if relative else "L2 error")
            ax.set_xticks(levels, [str(value) for value in levels])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="both", alpha=0.23)
            ax.set_xlim(min(levels) * 0.87, max(levels) * 1.15)
        fig.text(
            0.5,
            0.055,
            "Physical fields integrated separately; residuals and quadrature checks "
            "are in the JSON record.",
            ha="center",
            fontsize=9.4,
        )
        paths = {}
        for suffix in ("png", "svg"):
            path = directory / f"convergence.{suffix}"
            fig.savefig(path, dpi=120)
            paths[suffix] = dict(path=path.relative_to(ROOT).as_posix(), sha256=digest(path))
        plt.close(fig)
    return paths


def summarize(directory: Path, resource: Path, *, allow_incomplete: bool = False) -> dict[str, Any]:
    """Publish accepted norms, qualifying a deliberately selected interrupted study."""
    path = directory / "minimal-convergence.json"
    record = json.loads(path.read_text())
    if (not record["complete"] or len(record["rows"]) != 3) and not allow_incomplete:
        raise ValueError("A completed three-level acquisition is required")
    if not record["rows"]:
        raise ValueError("At least one accepted field with completed norms is required")
    count = checked_source_copies(record, directory)
    budget = json.loads(resource.read_text())
    interrupted = allow_incomplete and not record["complete"]
    if budget["changed_sources"] or not budget["owned_group_empty"]:
        raise ValueError("The supplied acquisition resource receipt did not pass")
    if interrupted:
        if budget["guard_reason"] != "elapsed deadline":
            raise ValueError("The incomplete acquisition must retain its actual deadline receipt")
    elif budget["returncode"] != 0 or budget["guard_reason"]:
        raise ValueError("The supplied acquisition resource receipt did not pass")
    resources = [
        dict(
            path=resource.relative_to(ROOT).as_posix(),
            sha256=digest(resource),
            receipt=budget,
            role="interrupted after accepted levels" if interrupted else "completed acquisition",
        )
    ]
    if record.get("resource_phases"):
        resources = []
        for phase in record["resource_phases"]:
            receipt_path = ROOT / phase["path"]
            if digest(receipt_path) != phase["sha256"]:
                raise ValueError("An original resource phase receipt differs")
            receipt = json.loads(receipt_path.read_text())
            if receipt["changed_sources"] or not receipt["owned_group_empty"]:
                raise ValueError("A resource phase has changed sources or retained processes")
            interrupted = phase["role"] == "interrupted after accepted levels"
            if interrupted:
                if receipt["guard_reason"] != "elapsed deadline" or not phase["accepted_levels"]:
                    raise ValueError("The preserved interrupted phase does not match its record")
            elif receipt["guard_reason"] or receipt["returncode"]:
                raise ValueError("A completed resource phase did not pass")
            resources.append(phase | dict(receipt=receipt))
    rows = record["rows"]
    case = record["case"]
    relative = case == "spe10"
    if relative:
        used = rows[1:]
        values = [row["norms"][str(row["norm_orders"][-1])] for row in used]
        pressure = [value["pressure_relative_increment"] for value in values]
        flux = [value["flux_relative_increment"] for value in values]
    else:
        used = rows
        pressure = [row["pressure_l2_error"] for row in used]
        flux = [row["flux_l2_error"] for row in used]
    titles = {
        "pgmhm": "PGMHM: P3 locals / P1 traces, alpha = 0.1",
        "unusual": "UNUSUAL: P1 locals / P0 traces, local r = 2",
        "spe10": "SPE10 layer 36: Q1 locals / continuous P1 trace, s = 2",
        "mixedwell": "Mixed well: tetrahedral H(div) flux / P1 pressure",
    }
    xlabel = (
        "Fine refinement factor F (fixed macro mesh)"
        if case == "mixedwell"
        else "Local refinement r (fixed macro and trace meshes)"
        if relative
        else "Macro subdivision n per axis"
    )
    plots = render(
        directory,
        titles[case] + (" (two accepted levels)" if interrupted else ""),
        [row["level"] for row in used],
        pressure,
        flux,
        xlabel,
        relative,
        case == "unusual",
    )
    quadrature = []
    names = (
        ("pressure_l2_norm", "flux_l2_norm")
        if relative
        else ("enriched_pressure_l2", "enriched_flux_l2")
        if case == "pgmhm"
        else ("scalar_l2", "flux_l2")
        if case == "unusual"
        else ("pressure_l2", "flux_l2")
    )
    for row in rows:
        low, high = [row["norms"][str(order)] for order in row["norm_orders"]]
        quadrature.append(
            dict(
                level=row["level"],
                orders=row["norm_orders"],
                field_relative_changes={
                    key: abs(high[key] - low[key]) / high[key] for key in names
                },
            )
        )
    rates = {}
    if not relative:
        for key in ("pressure_l2_error", "flux_l2_error"):
            rates[key] = [
                float(np.log(a[key] / b[key]) / np.log(a["h"] / b["h"]))
                for a, b in zip(rows[:-1], rows[1:], strict=True)
            ]
    summary = dict(
        schema="pymhm-minimal-darcy-summary-v1",
        case=case,
        acquisition_id=record["acquisition_id"],
        acquisition_complete=record["complete"],
        accepted_level_count=len(rows),
        requested_levels=record.get("requested_levels", [row["level"] for row in rows]),
        acquisition_record=dict(path=path.relative_to(ROOT).as_posix(), sha256=digest(path)),
        resource_phases=[
            {key: value for key, value in phase.items() if key != "receipt"} for phase in resources
        ],
        executed_source_copies_verified=count,
        elapsed_seconds=sum(phase["receipt"]["elapsed_seconds"] for phase in resources),
        elapsed_scope="Sum of original sequential acquisition phases, including interrupted work",
        peak_owned_rss_bytes=max(phase["receipt"]["peak_owned_rss_bytes"] for phase in resources),
        peak_scope="Maximum of measured sequential owned-process phase peaks",
        native_threads=budget["native_threads"],
        workers=budget["workers"],
        physical_field_norms_separate=True,
        quadrature_checks=quadrature,
        observed_initial_rates=rates,
        original_checked_rhs_relative_max=max(
            check["original_rhs_relative_max"]
            for row in rows
            for check in row["checked_original_solves"]
        ),
        interpretation=(
            "Two accepted levels on the fixed polygonal well geometry; the third level "
            "did not close within its acquisition deadline. These two points are an "
            "initial resolution comparison, not an established three-level convergence rate."
            if interrupted
            else "SPE10 pressure increments decrease, but the relative Darcy-flux increments "
            "exceed unity and increase. These local levels do not establish flux convergence "
            "or a sufficiently refined numerical reference."
            if relative
            else "Initial empirical rates for the stated analytical problem and physical spaces; "
            "no matched historical reproduction or general stability theorem is asserted."
        ),
        historical_reproduction=False,
        converged_numerical_reference=False,
        persisted_coefficients=False,
        figures=plots,
        renderer_source_sha256=digest(Path(__file__)),
    )
    (directory / "study-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def reuse_quarter(directory: Path) -> dict[str, Any]:
    """Reuse four verified NeoPZ classical levels with their original provenance."""
    source = ROOT / "examples/results/quarter-five-spot/reference/classical-convergence.json"
    record = json.loads(source.read_text())
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "original-records"
    snapshot.mkdir()
    (snapshot / source.name).write_bytes(source.read_bytes())
    checked = []
    for row in record["levels"]:
        path = source.parent / row["fields"]
        if digest(path) != row["fields_sha256"]:
            raise ValueError(f"Quarter obstacle field archive differs: {path}")
        metadata = path.with_suffix(".json")
        copied = snapshot / metadata.name
        copied.write_bytes(metadata.read_bytes())
        checked.append(
            dict(
                refinement=row["refinement"],
                fields=dict(path=path.relative_to(ROOT).as_posix(), sha256=digest(path)),
                original_metadata=dict(
                    path=metadata.relative_to(ROOT).as_posix(), sha256=digest(metadata)
                ),
                basis_sha256=row["executed_native_basis_sha256"],
                algebra=row["algebra"],
                physical_checks=row["physical_checks"],
                provenance=row["provenance"],
            )
        )
    increments = record["refinement_differences"]
    plots = render(
        directory,
        "Quarter obstacle: classical conforming NeoPZ RT0 / P0",
        [row["fine_refinement"] for row in increments],
        [row["pressure_l2_relative"] for row in increments],
        [row["flux_l2_relative"] for row in increments],
        "Finer local refinement r (fixed physical problem)",
        True,
    )
    summary = dict(
        schema="pymhm-minimal-quarter-classical-reuse-v1",
        case="quarter-obstacle",
        new_pde_solves=0,
        original_record=dict(path=source.relative_to(ROOT).as_posix(), sha256=digest(source)),
        original_geometry=record["geometry"],
        method=record["method"],
        levels=checked,
        refinement_differences=increments,
        interpretation=(
            "Four existing classical conforming reference levels show decreasing physical "
            "pressure and Darcy-flux increments. The finest field remains a numerical "
            "reference with nonzero refinement error; this initial study does not certify "
            "a converged baseline or an exact solution. Original source and native basis "
            "provenance are retained without retagging to the present checkout."
        ),
        converged_numerical_reference=False,
        historical_reproduction=False,
        figures=plots,
        renderer_source_sha256=digest(Path(__file__)),
    )
    (directory / "minimal-convergence.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main() -> None:
    """Render one closed acquisition or the existing four-level classical study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--resource", type=Path)
    parser.add_argument("--reuse-quarter", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.reuse_quarter:
        reuse_quarter(args.directory.resolve())
    elif args.resource is None:
        raise ValueError("The original acquisition resource receipt is required")
    else:
        summarize(
            args.directory.resolve(),
            args.resource.resolve(),
            allow_incomplete=args.allow_incomplete,
        )


if __name__ == "__main__":
    main()
