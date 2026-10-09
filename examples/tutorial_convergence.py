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
        path = root / "examples/results" / filename
        ordered = sorted(rows, key=lambda row: row[size], reverse=True)
        h = np.asarray([row[size] for row in ordered], dtype=float)
        if len(h) < 4 or np.any(h <= 0) or np.any(np.diff(h) >= 0):
            raise ValueError(f"Invalid refinement sequence for {method}")
        measured = {name: np.asarray([row[key] for row in ordered]) for name, key in errors.items()}
        if any(
            np.any(values <= 0) or not np.all(np.isfinite(values)) for values in measured.values()
        ):
            raise ValueError(f"Invalid physical errors for {method}")
        rates = {
            name: (np.log(values[:-1] / values[1:]) / np.log(h[:-1] / h[1:])).tolist()
            for name, values in measured.items()
        }
        studies.append(
            {
                "method": method,
                "record": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "refinement": refinement,
                "sizes": h.tolist(),
                "errors": {name: values.tolist() for name, values in measured.items()},
                "rates": rates,
                "expected": expected,
                "spaces": spaces,
                "rate_provenance": rate_provenance,
            }
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

    name = "darcy-rt.json"
    data = read(name)
    rows = [dict(H=2**0.5 / row["n"], **row["mhm"]) for row in data["rows"] if row["degree"] == 1]
    add(
        "mixed-mhm",
        name,
        rows,
        "H",
        {"pressure L2": "pressure_l2_error", "physical Darcy flux L2": "flux_l2_error"},
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

    name = "mh2m/comparison.json"
    data = read(name)
    rows = [row for row in data["smooth"] if row["k"] == 1]
    add(
        "mh2m",
        name,
        rows,
        "macro_diameter",
        {"pressure L2": "pressure_error_l2", "broken gradient L2": "gradient_error_l2"},
        {"broken gradient L2": 2.0},
        "Gamma P2; Lambda P1; local P2/r2",
        "de Barros, Madureira and Valentin (2026), section 8.1: gradient k+1; pressure "
        "k+2 observed",
    )

    name = "mshho.json"
    data = read(name)
    rows = [
        dict(H=2**0.5 / row["n"], **row) for row in data["convergence"] if row["face_degree"] == 0
    ]
    add(
        "mshho",
        name,
        rows,
        "H",
        {"pressure L2": "pressure_l2", "physical Darcy flux L2": "flux_l2"},
        {"physical Darcy flux L2": 1.0},
        "P0 cell and face moments; P3/r2 energy reconstruction",
        "Chaumont-Frelet et al. (2022), theorem 6.3: first-order energy; pressure order "
        "two observed",
    )

    name = "pgmhm/comparison.json"
    data = read(name)
    rows = [
        row
        for row in data["rows"]
        if row["study"] == "macro" and row["mesh"] == "triangles" and row["trace_degree"] == 1
    ]
    add(
        "pgmhm",
        name,
        rows,
        "macro_diameter",
        {
            "enriched pressure L2": "enriched_pressure_l2",
            "enriched Darcy flux L2": "enriched_flux_l2",
        },
        {"enriched Darcy flux L2": 2.0},
        "P3 locals; P1 trace; residual enrichment; alpha=0.1",
        "Fernando et al. (2023), theorem 6: smooth energy; ell>=1, k>=ell+d; pressure "
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

    name = "oseen/smooth-nu1-l1-uniform.json"
    data = read(name)
    rows = [dict(row, H=row.get("H", 1.0 / np.sqrt(row["macro_cells"]))) for row in data["rows"]]
    add(
        "oseen",
        name,
        rows,
        "H",
        {"velocity L2": "velocity_l2", "pressure L2": "pressure_l2"},
        {"velocity L2": 3.0, "pressure L2": 2.0},
        "Stabilized P3/P3 Oseen locals; vector P1 trace; smooth data, viscosity one",
        "Stable smooth Oseen family; fixed physical convection and viscosity",
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

    name = "tutorial-methods/primal-elasticity-current.json"
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

    name = "mixed-elasticity.json"
    data = read(name)
    rows = [dict(H=1.0 / row["macro_resolution"], **row) for row in data["convergence"]]
    add(
        "mixed-elasticity",
        name,
        rows,
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
    )

    name = "tutorial-methods/transient-native-study.json"
    data = read(name)
    add(
        "transient-transport",
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
        "elastodynamics",
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
    )

    name = "tutorial-methods/maxwell-current-refinement.json"
    data = read(name)
    rows = [row for row in data["rows"] if row["study"] == "time"]
    add(
        "maxwell",
        name,
        rows,
        "time_step",
        {"electric L2": "electric_error_l2", "magnetic L2": "magnetic_error_l2"},
        {"electric L2": 2.0, "magnetic L2": 2.0},
        "Fixed constrained spatial Maxwell operator; staggered electric/magnetic field times",
        "Second-order leapfrog time integration against the exact constrained semi- "
        "discrete evolution",
        refinement="time step",
    )
    return studies


def plot_method_series(series: dict[str, Any]) -> Any:
    """Plot measured errors, successive rates and independently stated guide orders."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    from examples.plot_style import set_refinement_ticks

    h = np.asarray(series["sizes"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for label, values in series["errors"].items():
        (line,) = axes[0].loglog(h, values, "o-", label=label)
        axes[1].semilogx(h[1:], series["rates"][label], "o-", color=line.get_color(), label=label)
        if label in series["expected"]:
            order = series["expected"][label]
            axes[1].axhline(
                order, color=line.get_color(), linestyle=":", label=f"literature: {order:g}"
            )
    for index, axis in enumerate(axes):
        axis.invert_xaxis()
        # Label a sparse subset of the actual refinement sizes. Decade-only
        # ticks can give just one label for a short dyadic refinement sequence.
        samples = h if index == 0 else h[1:]
        set_refinement_ticks(axis, samples, [f"{value:.3g}" for value in samples], max_labels=4)
        axis.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=12))
        axis.grid(True, which="major", alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set(xlabel=series["refinement"], ylabel="Physical error")
    axes[1].set(xlabel=series["refinement"], ylabel="Successive observed order")
    titles = {
        "primal-mhm": "Primal MHM",
        "mixed-mhm": "MHM with mixed H(div) locals",
        "robin-mh": "Robin MH",
        "mh2m": "MH²M",
        "mshho": "MsHHO",
        "pgmhm": "Petrov–Galerkin MHM",
        "mhm-usfem": "MHM-USFEM",
        "unfitted": "Unfitted MHM — trace refinement",
        "stokes-brinkman": "Stokes–Brinkman MHM",
        "oseen": "Oseen MHM",
        "primal-elasticity": "Primal elasticity MHM",
        "gals-elasticity": "GaLS elasticity MHM",
        "mixed-elasticity": "Mixed-stress elasticity MHM",
        "transient-transport": "Transient transport MHM — time refinement",
        "elastodynamics": "Elastodynamic MHM — time refinement",
        "helmholtz": "Helmholtz MHM",
        "maxwell": "Maxwell MHM — time refinement",
    }
    figure.suptitle(titles[series["method"]])
    return figure
