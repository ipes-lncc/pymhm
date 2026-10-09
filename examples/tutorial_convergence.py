"""Read attributed method convergence measurements for the teaching notebooks.

This module performs no PDE solve and does not relabel archived executions as
current acquisitions. Each returned series preserves its physical observable,
independent refinement variable, spaces, source digest and rate provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def refinement_series(
    record: Path,
    rows: list[dict[str, Any]],
    size: str,
    errors: dict[str, str],
    expected: dict[str, float],
    *,
    method: str,
    spaces: str,
    rate_provenance: str,
    refinement: str = "macro diameter",
    root: Path | None = None,
) -> dict[str, Any]:
    """Describe one attributed sequence without reading unrelated method records.

    ``size`` names the independent positive refinement parameter in each row;
    ``errors`` maps physical observable labels to their positive integrated
    errors. ``expected`` contains only independently justified target powers.
    Sorting retains every supplied row and uses the actual size ratios. The
    record digest identifies the supplied measurements, not a new PDE solve.
    """
    ordered = sorted(rows, key=lambda row: row[size], reverse=True)
    h = np.asarray([row[size] for row in ordered], dtype=float)
    if len(h) < 4 or np.any(h <= 0) or not np.all(np.isfinite(h)) or np.any(np.diff(h) >= 0):
        raise ValueError(f"Invalid refinement sequence for {method}")
    measured = {name: np.asarray([row[key] for row in ordered]) for name, key in errors.items()}
    if any(np.any(values <= 0) or not np.all(np.isfinite(values)) for values in measured.values()):
        raise ValueError(f"Invalid physical errors for {method}")
    rates = {
        name: (np.log(values[:-1] / values[1:]) / np.log(h[:-1] / h[1:])).tolist()
        for name, values in measured.items()
    }
    return {
        "method": method,
        "record": record.relative_to(root).as_posix() if root is not None else record.as_posix(),
        "sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
        "refinement": refinement,
        "sizes": h.tolist(),
        "errors": {name: values.tolist() for name, values in measured.items()},
        "rates": rates,
        "expected": expected,
        "spaces": spaces,
        "rate_provenance": rate_provenance,
    }


def method_series(root: Path) -> list[dict[str, Any]]:
    """Load attributed method studies and compute consecutive physical rates.

    Macro diameters are used for macro refinement; the unfitted study instead
    bisects trace segments on one fixed macro mesh. Pressure orders marked
    ``observed`` are measurements, rather than separate claims from the papers.
    Inputs are immutable numerical records, not approximate values read from
    figures. Their original source manifests remain available at ``record``.
    """
    studies: list[dict[str, Any]] = []

    def add(
        method: str,
        filename: str,
        rows: list[dict[str, Any]],
        size: str,
        errors: dict[str, str],
        expected: dict[str, float],
        spaces: str,
        rate_provenance: str,
        refinement: str = "macro diameter",
    ) -> None:
        """Normalize one already selected physical sequence without changing data."""
        studies.append(
            refinement_series(
                root / "examples/results" / filename,
                rows,
                size,
                errors,
                expected,
                method=method,
                spaces=spaces,
                rate_provenance=rate_provenance,
                refinement=refinement,
                root=root,
            )
        )

    def read(filename: str) -> dict[str, Any]:
        """Read an attributed JSON acquisition, preserving its original metadata."""
        return json.loads((root / "examples/results" / filename).read_text())

    name = "darcy_2013_comparison.json"
    data = read(name)
    add(
        "primal-mhm",
        name,
        data["results"],
        "macro_diameter",
        {"pressure L2": "pressure_l2", "physical Darcy flux L2": "flux_l2"},
        {"physical Darcy flux L2": 1.0},
        "P1 local pressure; P0 normal-flux trace",
        "Harder, Paredes and Valentin (2013): first-order energy; second-order pressure observed",
    )

    name = "tutorial-methods/mixed-mhm-current.json"
    data = read(name)
    add(
        "mixed-mhm",
        name,
        data["rows"],
        "macro_diameter",
        {"pressure L2": "pressure_l2", "physical Darcy flux L2": "flux_l2"},
        {"pressure L2": 2.0, "physical Darcy flux L2": 2.0},
        "RT1/P1 locals; P1 normal-flux trace; two local edge subdivisions",
        "Durán, Devloo, Gomes and Valentin (2019): compatible smooth mixed spaces",
    )

    name = "mh/comparison.json"
    data = read(name)
    rows = [
        row for row in data["smooth"] if row["mesh"] == "L-polygons" and row["trace_degree"] == 1
    ]
    add(
        "robin-mh",
        name,
        rows,
        "macro_diameter",
        {"pressure L2": "pressure_error_l2", "physical Darcy flux L2": "flux_error_l2"},
        {"physical Darcy flux L2": 2.0},
        "P3/r2 locals; P1 Robin trace; nu=1/4",
        "Barrenechea, Gomes and Paredes (2024): smooth energy order two; pressure order "
        "three observed",
    )

    name = "tutorial-methods/mh2m-compatible-current.json"
    data = read(name)
    add(
        "mh2m",
        name,
        data["rows"],
        "macro_diameter",
        {"pressure L2": "pressure_l2", "broken gradient L2": "gradient_l2"},
        {"broken gradient L2": 2.0},
        "Gamma P2; two Lambda P1 segments per Gamma edge; local P2/r4; "
        "two fine edges per Lambda segment; M1/M2-compatible",
        "de Barros, Madureira and Valentin (2026), theorem 19: compatible smooth "
        "energy order two; pressure order three observed separately",
    )

    name = "tutorial-methods/mshho-current.json"
    data = read(name)
    add(
        "mshho",
        name,
        data["rows"],
        "macro_diameter",
        {"pressure L2": "pressure_l2", "physical Darcy flux L2": "flux_l2"},
        {"physical Darcy flux L2": 1.0},
        "Matching P0 cell/face moments; P3/r2 finite energy reconstruction; projected source",
        "Chaumont-Frelet et al. (2022), theorem 6.3: first-order energy; pressure order "
        "two observed",
    )

    name = "tutorial-methods/pgmhm-current.json"
    data = read(name)
    add(
        "pgmhm",
        name,
        data["rows"],
        "macro_diameter",
        {
            "enriched pressure L2": "pressure_l2",
            "enriched Darcy flux L2": "flux_l2",
        },
        {"enriched Darcy flux L2": 2.0},
        "P3 locals; P1 trace; residual enrichment; alpha=0.1",
        "Fernando et al. (2023), section 5: smooth energy; ell>=1, k>=ell+d; pressure "
        "order three observed",
    )

    name = "unusual/analytical.json"
    data = read(name)
    rows = [
        dict(H=2**0.5 / row["n"], **row)
        for row in data["rows"]
        if row["case"] == "smooth" and row["method"] == "unusual"
    ]
    add(
        "mhm-usfem",
        name,
        rows,
        "H",
        {"scalar L2": "scalar_l2", "broken gradient L2": "gradient_l2"},
        {"broken gradient L2": 1.0},
        "P1/r2 locals; P0 trace; epsilon=1; negative residual pairing",
        "Santiago, Valentin and Martins (2025): admissible reaction-diffusion "
        "stabilization; smooth scalar order two observed",
    )

    name = "unfitted/convergence/rate-verification.json"
    data = read(name)
    family = next(row for row in data["families"] if row["trace_degree"] == 1)
    rows = [
        dict(H=0.5 / segment, error=error)
        for segment, error in zip(family["segments"], family["errors"], strict=True)
    ]
    add(
        "unfitted",
        name,
        rows,
        "H",
        {"broken gradient L2": "error"},
        {"broken gradient L2": 2.5},
        "P8/r32 locals; segmented P1 traces; 16 fixed macrotriangles",
        "Chaumont-Frelet, Paredes and Valentin (2026), section 6.1: smooth trace reference ell+3/2",
        refinement="trace segment size on a fixed macro mesh",
    )
    name = "stokes-adaptive/polynomial-uniform-nu1-g0-l1.json"
    data = read(name)
    rows = [dict(H=2.0 / np.sqrt(row["macro_cells"]), **row) for row in data["rows"]]
    add(
        "stokes-brinkman",
        name,
        rows,
        "H",
        {"velocity L2": "velocity_l2", "pressure L2": "pressure_l2"},
        {"velocity L2": 3.0, "pressure L2": 2.0},
        "Stabilized P3/P3 USFEM locals; vector P1 traces; viscosity one, zero drag",
        "Araya, Harder, Poza and Valentin (2017): smooth velocity/pressure estimates",
    )

    name = "tutorial-methods/stokes-taylor-hood-asymptotic.json"
    data = read(name)
    add(
        "stokes-galerkin",
        name,
        data["rows"],
        "H",
        {"velocity L2": "velocity_l2", "pressure L2": "pressure_l2"},
        {"velocity L2": 3.0, "pressure L2": 2.0},
        "Continuous P2/P1 Taylor–Hood locals on r4 meshes; unsplit vector P1 "
        "traces; viscosity one, zero drag; retained translations and pressure gauge",
        "Araya, Harder, Poza and Valentin (2017), smooth degree estimates: "
        "analytical qualification on diagonal macrotriangles with positive "
        "streamfunction; not a reproduction of the paper's negative-streamfunction data",
    )

    name = "oseen/smooth-nu1-l1-uniform.json"
    data = read(name)
    rows = [dict(row, H=row.get("H", 1.0 / np.sqrt(row["macro_cells"]))) for row in data["rows"]]
    add(
        "oseen",
        name,
        rows,
        "H",
        {
            "velocity L2": "velocity_l2",
            "pressure L2": "pressure_l2",
            "mixed V × Q error": "mixed_error",
        },
        {"pressure L2": 2.0, "mixed V × Q error": 2.0},
        "Stabilized P3/P3 Oseen locals; vector P1 trace; smooth data, viscosity one",
        "Araya, Cárcamo, Poza and Valentin (2021), section 5.1: mixed V × Q "
        "estimate of order ell+1; velocity L2 order three is observed separately",
    )

    name = "elasticity.json"
    data = read(name)
    rows = [row for row in data["refinement"] if row["method"] == "gals-p1"]
    add(
        "gals-elasticity",
        name,
        rows,
        "skeleton_size",
        {
            "displacement L2": "displacement_l2",
            "pressure L2": "pressure_l2",
            "broken gradient L2": "gradient_l2",
        },
        {"displacement L2": 2.0, "broken gradient L2": 1.0},
        "Equal-order P1 GaLS locals; fixed macro mesh; matched fine/trace refinement",
        "Smooth stabilized first-order energy and second-order displacement; pressure "
        "reported separately",
        refinement="trace/fine size on the fixed macro mesh",
    )

    name = "tutorial-methods/primal-elasticity-asymptotic.json"
    data = read(name)
    add(
        "primal-elasticity",
        name,
        data["rows"],
        "H",
        {"displacement L2": "displacement_l2", "physical stress L2": "stress_l2"},
        {"displacement L2": 3.0, "physical stress L2": 2.0},
        "P3 local displacement; one local subdivision; P1 physical traction; "
        "fixed homogeneous anisotropic tensor",
        "Harder, Madureira and Valentin (2016): smooth traction-driven estimates; "
        "fresh current-source physical stress and displacement norms",
    )

    name = "tutorial-methods/mixed-elasticity-asymptotic.json"
    data = read(name)
    add(
        "mixed-elasticity",
        name,
        [{**row, "H": 1.0 / row["macro_resolution"]} for row in data["rows"]],
        "H",
        {
            "displacement L2": "displacement_l2",
            "stress L2": "stress_l2",
            "rotation L2": "rotation_l2",
        },
        {"stress L2": 2.0, "rotation L2": 2.0},
        "BDM2 stress rows; P1 displacement/rotation; P1 interior and full fine P2 "
        "exterior traction",
        "Smooth compatible mixed spaces; stress/rotation order two; displacement "
        "measured separately",
        refinement="macro grid spacing",
    )

    name = "tutorial-methods/transient-native-study.json"
    data = read(name)
    add(
        "transient-transport-time",
        name,
        data["rows"],
        "dt",
        {"concentration L2": "concentration_L2"},
        {"concentration L2": 1.0},
        "Native UFL P2/r16 locals; P1 trace on eight face segments; fixed spatial "
        "operator; backward Euler",
        "First-order backward Euler time approximation against an analytical "
        "concentration; independently checked norm quadrature and refined classical reference",
        refinement="time step",
    )

    name = "elastodynamics/comparison.json"
    data = read(name)
    rows = [dict(dt=row["dt"], **row["difference_to_time_reference"]) for row in data["temporal"]]
    add(
        "elastodynamics-time",
        name,
        rows,
        "dt",
        {"displacement L2 difference": "displacement_l2", "velocity L2 difference": "velocity_l2"},
        {"displacement L2 difference": 2.0, "velocity L2 difference": 2.0},
        "Fixed spatial space; Newmark beta=1/4, gamma=1/2; independently refined time reference",
        "Second-order Newmark time integration; differences to a numerical time "
        "reference, not exact errors",
        refinement="time step",
    )

    name = "helmholtz-article/published-convergence.json"
    data = read(name)
    rows = []
    for resolution in sorted(
        {row["n"] for row in data["rows"] if row["ell"] == 2 and row["basis"] == "polynomial"}
    ):
        selected = [
            row
            for row in data["rows"]
            if row["ell"] == 2 and row["basis"] == "polynomial" and row["n"] == resolution
        ]
        rows.append(
            dict(
                H=1.0 / resolution,
                **{row["field"]: row["pymhm_relative_error"] for row in selected},
            )
        )
    add(
        "helmholtz",
        name,
        rows,
        "H",
        {"relative pressure L2": "pressure", "relative gradient L2": "gradient"},
        {"relative pressure L2": 4.0, "relative gradient L2": 3.0},
        "Local Q4/r2; polynomial P2 face trace; fixed analytical plane wave and frequency",
        "Chaumont-Frelet and Valentin (2020): resolved fixed-frequency estimates; PyMHM "
        "measurements, not digitized article values",
        refinement="macro grid spacing",
    )

    name = "tutorial-methods/maxwell-time-current.json"
    data = read(name)
    add(
        "maxwell-time",
        name,
        data["rows"],
        "time_step",
        {"electric L2": "electric_error_l2", "magnetic L2": "magnetic_error_l2"},
        {"electric L2": 2.0, "magnetic L2": 2.0},
        "Fixed constrained spatial Maxwell operator; staggered electric/magnetic field times",
        "Second-order leapfrog time integration against the exact constrained semi- "
        "discrete evolution",
        refinement="time step",
    )
    name = "tutorial-methods/transient-spatial-asymptotic.json"
    data = read(name)
    add(
        "transient-transport",
        name,
        data["rows"],
        "H",
        {"concentration L2": "concentration_l2", "broken gradient L2": "broken_gradient_l2"},
        {"broken gradient L2": 2.0},
        "Native local P2/r2; P1 traces; capacity one, diffusion one; smooth "
        "fixed convection/reaction; c=t sin(pi x) sin(pi y)",
        "Araya et al. (2024), theorems 2–3 and assumption A2: smooth energy "
        "order ell+1; concentration order three observed; independent time-step control",
    )
    name = "tutorial-methods/elastodynamics-spatial-asymptotic.json"
    data = read(name)
    add(
        "elastodynamics",
        name,
        data["rows"],
        "H",
        {
            "displacement L2": "displacement_l2",
            "velocity L2": "velocity_l2",
            "physical stress L2": "stress_l2",
        },
        {"displacement L2": 3.0, "velocity L2": 3.0, "physical stress L2": 2.0},
        "Two-dimensional smooth homogeneous isotropic analogue; P3/r1 local "
        "displacement, P1 traction; quadratic time dependence; Newmark",
        "Gomes, Paredes, Pereira, Souto and Valentin (2017): smooth linear-traction spatial "
        "targets; independent time-step control and conforming reference",
    )
    name = "tutorial-methods/maxwell-spatial-asymptotic.json"
    data = read(name)
    rows = [dict(row, **row["max_in_time"]) for row in data["rows"]]
    add(
        "maxwell",
        name,
        rows,
        "H",
        {"combined L2": "combined_l2", "combined broken H(curl)": "combined_hcurl"},
        {"combined L2": 2.0, "combined broken H(curl)": 1.0},
        "Two-dimensional TM cavity; P3 one-element locals, P1 tangential traces; "
        "maxima over staggered field times",
        "Lanteri, Paredes, Scheid and Valentin (2018), section 6.2: combined "
        "L2 order ell+1 and broken H(curl) order ell; fine time-step control",
        refinement="macro grid spacing",
    )
    return studies


def asymptotic_summary(
    series: dict[str, Any], *, tail_intervals: int = 3
) -> dict[str, dict[str, Any]]:
    """Describe a common terminal refinement window without discarding coarse data.

    The window contains ``tail_intervals + 1`` measured levels. The fitted
    exponent uses their actual log sizes, including non-dyadic refinements.
    For a stated target ``q``, the amplitude ratio compares the largest and
    smallest ``E / h**q`` in this window. A ratio approaching one and stable
    consecutive orders are evidence of an asymptotic regime; neither statistic
    verifies the hypotheses of an error theorem or numerical quadrature.
    No threshold changes the measurements or labels a PDE execution accepted.
    """
    h = np.asarray(series["sizes"], dtype=float)
    if tail_intervals < 2 or len(h) <= tail_intervals:
        raise ValueError("A terminal window needs at least two intervals and distinct levels")
    start = len(h) - tail_intervals - 1
    summary: dict[str, dict[str, Any]] = {}
    for field, values in series["errors"].items():
        errors = np.asarray(values, dtype=float)
        target = series["expected"].get(field)
        entry = {
            "start_index": start,
            "levels": len(h) - start,
            "sizes": h[start:].tolist(),
            "orders": series["rates"][field][start:],
            "fitted_order": float(np.polyfit(np.log(h[start:]), np.log(errors[start:]), 1)[0]),
            "target": target,
        }
        if target is not None:
            amplitude = errors[start:] / h[start:] ** target
            entry["amplitude_ratio"] = float(amplitude.max() / amplitude.min())
        summary[field] = entry
    return summary


def plot_method_series(series: dict[str, Any]) -> Any:
    """Plot all errors, three terminal orders and the target-normalized amplitudes.

    Dashes are target slopes anchored to the finest measured error, rather than
    additional numerical results. The shaded region is always the last four
    measured levels; it does not select intervals to make a slope agree. An
    approximately horizontal normalized-amplitude curve supports the rate
    comparison, alongside the method's independent numerical controls.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    from examples.plot_style import set_refinement_ticks

    h = np.asarray(series["sizes"], dtype=float)
    figure = plt.figure(figsize=(10, 6.4), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.05, 1))
    axes = [
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    ]
    plotted_orders: set[float] = set()
    for label, values in series["errors"].items():
        values = np.asarray(values, dtype=float)
        (line,) = axes[0].loglog(h, values, "o-", label=label)
        axes[1].semilogx(h[1:], series["rates"][label], "o-", color=line.get_color())
        if label in series["expected"]:
            order = series["expected"][label]
            guide = values[-1] * (h / h[-1]) ** order
            axes[0].loglog(h, guide, "--", color=line.get_color(), alpha=0.6)
            if order not in plotted_orders:
                axes[1].axhline(order, color="0.4", linestyle=":", label=f"target q = {order:g}")
                plotted_orders.add(order)
            normalized = values / h**order
            axes[2].semilogx(
                h,
                normalized / normalized[-1],
                "o-",
                color=line.get_color(),
                label=f"{label}, q = {order:g}",
            )
    axes[2].axhline(1, color="0.4", linewidth=0.9, linestyle=":")
    for index, axis in enumerate(axes):
        axis.invert_xaxis()
        # Label a sparse subset of the actual refinement sizes. Decade-only
        # ticks can give just one label for a short dyadic refinement sequence.
        samples = h[1:] if index == 1 else h
        set_refinement_ticks(axis, samples, [f"{value:.3g}" for value in samples], max_labels=4)
        axis.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=12))
        axis.grid(True, which="major", alpha=0.25)
        axis.axvspan(h[-4], h[-1], color="0.85", alpha=0.32, zorder=0)
        axis.set_xlabel(series["refinement"])
    axes[0].legend(fontsize=9, loc="best")
    axes[1].legend(fontsize=9, loc="best")
    axes[2].legend(fontsize=8, loc="best")
    axes[0].set(xlabel=series["refinement"], ylabel="Physical error")
    axes[1].set(xlabel=series["refinement"], ylabel="Observed order")
    axes[2].set(ylabel=r"$E / h^q$, relative to finest")
    axes[0].set_title("All measured levels; dashed curves: target slopes", fontsize=10)
    axes[1].set_title("Orders at every refinement interval", fontsize=10)
    axes[2].set_title("Target-normalized errors", fontsize=10)
    titles = {
        "primal-mhm": "Primal MHM",
        "mixed-mhm": "MHM with mixed H(div) locals",
        "robin-mh": "Robin MH",
        "mh2m": "MH²M",
        "mshho": "MsHHO",
        "pgmhm": "Petrov–Galerkin MHM",
        "mhm-usfem": "MHM-USFEM",
        "unfitted": "Unfitted MHM — trace refinement",
        "stokes-brinkman": "Stokes MHM — stabilized USFEM locals",
        "stokes-galerkin": "Stokes MHM — Taylor–Hood locals",
        "oseen": "Oseen MHM",
        "primal-elasticity": "Primal elasticity MHM",
        "gals-elasticity": "GaLS elasticity MHM",
        "mixed-elasticity": "Mixed-stress elasticity MHM",
        "transient-transport": "Transient transport MHM — spatial refinement",
        "transient-transport-time": "Transient transport MHM — time refinement",
        "elastodynamics": "Elastodynamic MHM — spatial refinement",
        "elastodynamics-time": "Elastodynamic MHM — time refinement",
        "helmholtz": "Helmholtz MHM",
        "maxwell": "Maxwell MHM — spatial refinement",
        "maxwell-time": "Maxwell MHM — time refinement",
    }
    figure.suptitle(titles.get(series["method"], series["method"]))
    figure.supxlabel("Shading: final four measured levels", fontsize=9)
    return figure
