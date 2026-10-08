"""Plot measured mixed 3D well errors and the common faceted macro geometry."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, SymLogNorm, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from examples.plot_style import set_refinement_ticks
from examples.solve_mapped_well import WellData
from pymhm.fem.hdiv.family_3d import HDiv3DFamily, reference_faces
from pymhm.meshes.hexahedron import HexMesh
from pymhm.meshes.mixed import AffineMixedMesh, hdiv3d_dofs, hdiv3d_transform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures/mixed-well-geometries"
DATA = ROOT / "examples/results/mixed-well-geometries"


def read_records() -> list[dict]:
    """Load only records whose public physical-field archive matches its acquisition digest."""
    records = []
    for path in sorted(DATA.glob("*-fine*-macro*.json")):
        row = json.loads(path.read_text())
        if row.get("archive_schema") != 2:
            raise ValueError(
                f"field archive must include its executed reference basis: {path.name}"
            )
        if hashlib.sha256((DATA / row["archive"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"field archive digest mismatch: {path.name}")
        with np.load(DATA / row["archive"]) as archive:
            basis_digest = hashlib.sha256(archive["flux_basis_coefficients"].tobytes()).hexdigest()
        if basis_digest != row["flux_basis_sha256"]:
            raise ValueError(f"reference basis digest mismatch: {path.name}")
        records.append(row)
    return records


def boundary_rates(records: list[dict]) -> list[dict]:
    """Integrate archived constant normal moments on the physical exterior faces."""
    rates = []
    for row in records:
        with np.load(DATA / row["archive"]) as archive:
            macro = AffineMixedMesh(archive["macro_points"], archive["macro_cells"], row["kind"])
            trace = archive["trace"]
            values = {"inner": 0.0, "outer": 0.0, "caps": 0.0}
            for face in macro.boundary_faces:
                nodes = macro.points[macro.faces[face]]
                if np.ptp(nodes[:, 2]) < 1e-12:
                    key = "caps"
                elif np.max(np.linalg.norm(nodes[:, :2], axis=1)) <= 0.2 * (1 + 1e-12):
                    key = "inner"
                else:
                    if (
                        np.min(np.linalg.norm(nodes[:, :2], axis=1))
                        < 50 * np.cos(np.pi / 8) - 1e-10
                    ):
                        raise ValueError("an exterior well face is outside the declared boundary")
                    key = "outer"
                # Degree-one trace, one subface: its first dual moment is
                # exactly the physical integrated flux, with outward sign.
                values[key] += float(trace[macro.face_offsets[face]])
        rates.append(
            dict(
                kind=row["kind"],
                pressure_degree=row["pressure_degree"],
                fine_factor=row["fine_factor"],
                macro_factor=row["macro_factor"],
                archive=row["archive"],
                sha256=row["sha256"],
                flux_rates_m3_per_s=values,
                net_outward_rate=sum(values.values()),
                exact_rates_m3_per_s=dict(inner=0.01, outer=-0.01, caps=0.0),
            )
        )
    return rates


def _save(fig: plt.Figure, name: str) -> None:
    """Save standalone raster/vector figures with space reserved for labels."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=180)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def geometry() -> None:
    """Show actual macro edges of three partitions of exactly the same octagonal reservoir."""
    hexa = HexMesh.annular_prism(np.geomspace(0.2, 50, 5), 10, 8)
    prism = AffineMixedMesh.from_extruded_hexahedra(hexa.points, hexa.cells, "prism")
    tetra = AffineMixedMesh.from_extruded_hexahedra(hexa.points, hexa.cells, "tetrahedron")
    fig = plt.figure(figsize=(14, 5), layout="constrained")
    for index, (name, mesh) in enumerate(
        [("Hexahedra", hexa), ("Prisms", prism), ("Tetrahedra", tetra)]
    ):
        ax = fig.add_subplot(1, 3, index + 1, projection="3d")
        if name == "Hexahedra":
            corners = np.array(list(product((0, 1), repeat=3)))
            edges = [
                (i, j)
                for i in range(8)
                for j in range(i + 1, 8)
                if np.sum(abs(corners[i] - corners[j])) == 1
            ]
        else:
            edges = set()
            for face in reference_faces(mesh.kind):
                cycle = list(face) if len(face) == 3 else [face[i] for i in (0, 1, 3, 2)]
                edges.update(
                    tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))
                    for i in range(len(cycle))
                )
        keys = {
            tuple(sorted((int(cell[a]), int(cell[b])))) for cell in mesh.cells for a, b in edges
        }
        lines = np.array([mesh.points[list(edge)] for edge in keys])
        ax.add_collection3d(Line3DCollection(lines, colors="#164a63", linewidths=0.65))
        ax.set(
            xlim=(-52, 52),
            ylim=(-52, 52),
            zlim=(-5, 5),
            xlabel="x [m]",
            ylabel="y [m]",
            zlabel="z [m]",
        )
        ax.set_title(f"{name}: {len(mesh.cells)} macrocells", pad=15)
        ax.set_box_aspect((1, 1, 0.42), zoom=0.77)
        ax.view_init(26, -56)
    fig.suptitle("Common faceted domain; lines show the actual macro partitions", fontsize=15)
    _save(fig, "geometry")


def convergence(records: list[dict]) -> None:
    """Compare classical refinement and macro resolution at equal fine refinement factors."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    configurations = [
        ("prism", 1, "Prism W11", "#007f78"),
        ("tetrahedron", 1, "Tetra P1", "#c25908"),
        ("tetrahedron", 2, "Tetra P2 enriched", "#67419c"),
    ]
    hexahedral = []
    for fine in (1, 2, 4):
        directory = ROOT / "examples/results/mapped-well"
        row = json.loads((directory / f"fine{fine}-macro{fine}-q6.json").read_text())
        if hashlib.sha256((directory / row["archive"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("hexahedral well field archive digest mismatch")
        hexahedral.append(row)
    for j, key in enumerate(("pressure_relative", "flux_relative")):
        axes[0, j].loglog(
            [r["fine_factor"] for r in hexahedral],
            [r[key] for r in hexahedral],
            "s--",
            label="Hexa RT1 / Q1",
            color="#245bb0",
        )
    for kind, degree, label, color in configurations:
        own = [r for r in records if r["kind"] == kind and r["pressure_degree"] == degree]
        classical = sorted(
            [r for r in own if r["fine_factor"] == r["macro_factor"]],
            key=lambda r: r["fine_factor"],
        )
        if not classical:
            continue
        finest = max(r["fine_factor"] for r in own)
        upscaled = sorted(
            [r for r in own if r["fine_factor"] == finest], key=lambda r: r["macro_factor"]
        )
        full = next((r for r in upscaled if r["macro_factor"] == finest), None)
        if full is None:
            raise ValueError("the fixed-grid comparison requires its full-trace reference")
        total = full["skeleton_dofs"] + full["retained_dofs"]
        for j, key in enumerate(("pressure_l2", "flux_l2")):
            axes[0, j].loglog(
                [r["fine_factor"] for r in classical],
                [r["relative_errors"][key] for r in classical],
                "o-",
                label=label,
                color=color,
            )
            axes[1, j].semilogy(
                [100 * (r["skeleton_dofs"] + r["retained_dofs"]) / total for r in upscaled],
                [r["relative_errors"][key] for r in upscaled],
                "o-",
                label=f"{label}, fine {finest}",
                color=color,
            )
    for j, title in enumerate(("Pressure", "Vector flux")):
        axes[0, j].set(
            title=f"{title}: classical fine-grid convergence",
            xlabel="Uniform refinement factor",
            ylabel="Relative physical L² error",
        )
        axes[1, j].set(
            title=f"{title}: MHM skeletal resolution",
            xlabel="Retained DOFs / full-trace DOFs [%]",
            ylabel="Relative physical L² error",
        )
        set_refinement_ticks(axes[0, j], sorted({r["fine_factor"] for r in records}))
        for ax in axes[:, j]:
            ax.grid(True, alpha=0.22)
            ax.legend(fontsize=9)
    fig.suptitle(
        "Common physical domain; fine factor four in the lower panels\n"
        "Tetrahedral midpoint: different fine connectivity; coarse/full endpoints coincide",
        fontsize=12,
    )
    _save(fig, "convergence")


@dataclass(frozen=True)
class TopFaceSamples:
    """One-sided physical fields at top-face centroids and their actual macro edges."""

    polygons: list[np.ndarray]
    locations: np.ndarray
    pressure: np.ndarray
    flux: np.ndarray
    edges: list[np.ndarray]


def top_face_samples(row: dict) -> TopFaceSamples:
    """Evaluate archived mixed coefficients with physical Piola and no field averaging."""
    with np.load(DATA / row["archive"]) as archive:
        macro = AffineMixedMesh(archive["macro_points"], archive["macro_cells"], row["kind"])
        local_points = archive["local_points"]
        local_cells = archive["local_cells"]
        coefficients = archive["flux"]
        pressure_coefficients = archive["pressure"]
        basis_coefficients = archive["flux_basis_coefficients"]
    family = HDiv3DFamily(row["kind"], row["pressure_degree"])
    polygons, locations, pressures, fluxes = [], [], [], []
    edges = []
    for face in macro.boundary_faces:
        x = macro.points[macro.faces[face]]
        if not np.allclose(x[:, 2], 5, rtol=0, atol=1e-12):
            continue
        cycle = list(range(len(x))) if len(x) == 3 else [0, 1, 3, 2]
        edges.extend([x[[cycle[i], cycle[(i + 1) % len(cycle)]], :2] for i in range(len(cycle))])
    for cell in range(len(macro.cells)):
        if np.max(macro.points[macro.cells[cell], 2]) < 5 - 1e-12:
            continue
        mesh = AffineMixedMesh(local_points[cell], local_cells[cell], row["kind"])
        transform = hdiv3d_transform(mesh, family, coefficients=basis_coefficients)
        local = coefficients[cell][hdiv3d_dofs(mesh, family)]
        canonical = np.einsum("tij,tj->ti", transform, local)
        for face in mesh.boundary_faces:
            x = mesh.points[mesh.faces[face]]
            if not np.allclose(x[:, 2], 5, rtol=0, atol=1e-12):
                continue
            owner = mesh.incidence[face][0][0]
            center = x.mean(axis=0)
            reference = (center - mesh.points[mesh.cells[owner, 0]]) @ mesh.inverse[owner].T
            v, _, p = family.tabulate(reference[None], coefficients=basis_coefficients)
            q = mesh.jacobian[owner] @ (canonical[owner] @ v[0]) / mesh.determinants[owner]
            polygons.append(x[:, :2])
            locations.append(center)
            pressures.append(pressure_coefficients[cell, owner] @ p[0])
            fluxes.append(q)
    return TopFaceSamples(
        polygons, np.array(locations), np.array(pressures), np.array(fluxes), edges
    )


def fields(row: dict) -> None:
    """Show one-sided top-face centroid fields and signed flux, with actual macro edges."""
    samples = top_face_samples(row)
    xy = samples.locations
    exactp = WellData().pressure(xy) / 1e6
    exactq = WellData().flux(xy)
    # Matplotlib color normalization requires float64; archived coefficients
    # and all physical norm calculations retain their acquisition precision.
    numerical = [
        np.asarray(samples.pressure, dtype=float) / 1e6,
        np.asarray(samples.flux, dtype=float)[:, 0],
        np.asarray(samples.flux, dtype=float)[:, 1],
    ]
    expected = [exactp, exactq[:, 0], exactq[:, 1]]
    labels = ["Pressure [MPa]", "Flux x [m/s]", "Flux y [m/s]"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), layout="constrained")
    for i, (actual, truth, label) in enumerate(zip(numerical, expected, labels, strict=True)):
        limit = max(float(np.max(abs(truth))), float(np.max(abs(actual))))
        norm = (
            Normalize(min(actual.min(), truth.min()), max(actual.max(), truth.max()))
            if i == 0
            else SymLogNorm(linthresh=0.01 * limit, vmin=-limit, vmax=limit, base=10)
        )
        difference = actual - truth
        error_limit = max(float(np.max(abs(difference))), np.finfo(float).tiny)
        error_norm = (
            TwoSlopeNorm(vmin=-error_limit, vcenter=0, vmax=error_limit)
            if i == 0
            else SymLogNorm(
                linthresh=0.01 * error_limit, vmin=-error_limit, vmax=error_limit, base=10
            )
        )
        for j, (title, values) in enumerate(
            zip(("Exact", "Mixed MHM", "Difference"), (truth, actual, difference), strict=True)
        ):
            ax = axes[i, j]
            collection = PolyCollection(
                samples.polygons,
                array=values,
                cmap="viridis" if i == 0 and j < 2 else "RdBu_r",
                norm=norm if j < 2 else error_norm,
                edgecolors="none",
                rasterized=True,
            )
            ax.add_collection(collection)
            ax.add_collection(LineCollection(samples.edges, colors="#212121", linewidths=0.55))
            ax.set(
                xlim=(-4, 4),
                ylim=(-4, 4),
                aspect="equal",
                xlabel="x [m]",
                ylabel="y [m]",
                title=f"{title}: {label}",
            )
            colorbar = fig.colorbar(collection, ax=ax, fraction=0.05, pad=0.03)
            if i > 0:
                bound = limit if j < 2 else error_limit
                ticks = bound * np.array([-1, -0.1, -0.01, 0, 0.01, 0.1, 1])
                colorbar.set_ticks(
                    ticks, labels=[f"{value:.1e}" if value else "0" for value in ticks]
                )
                colorbar.minorticks_off()
    fig.suptitle(
        f"{row['kind']}, pressure degree {row['pressure_degree']}; "
        f"fine factor {row['fine_factor']} / macro factor {row['macro_factor']}\n"
        "Top-face centroid samples; macro edges; unsmoothed fields; well-region zoom\n"
        "Flux: symmetric log scale, linear core = 1% of range; pressure: linear scale",
        fontsize=14,
    )
    _save(fig, f"{row['kind']}-p{row['pressure_degree']}-fields")


def trace_comparison(coarse: dict, classical: dict) -> None:
    """Compare signed flux components for restricted and complete fine-face trace spaces."""
    if coarse["fine_factor"] != classical["fine_factor"]:
        raise ValueError("the trace comparison requires the same fine refinement factor")
    limited, full = top_face_samples(coarse), top_face_samples(classical)
    exact = WellData().flux(limited.locations)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9.3), layout="constrained")
    titles = (
        "Exact Darcy flux",
        "MHM: coarse face partition\n"
        f"Volume flux error {100 * coarse['relative_errors']['flux_l2']:.2f}%",
        "Complete fine-face traces\n"
        f"Volume flux error {100 * classical['relative_errors']['flux_l2']:.2f}%",
    )
    for component in range(2):
        arrays = [exact[:, component], limited.flux[:, component], full.flux[:, component]]
        bound = max(float(np.max(abs(values))) for values in arrays)
        norm = SymLogNorm(linthresh=0.01 * bound, vmin=-bound, vmax=bound, base=10)
        for column, (sample, values, title) in enumerate(
            zip((limited, limited, full), arrays, titles, strict=True)
        ):
            ax = axes[component, column]
            collection = PolyCollection(
                sample.polygons,
                array=np.asarray(values, dtype=float),
                cmap="RdBu_r",
                norm=norm,
                edgecolors="none",
                rasterized=True,
            )
            ax.add_collection(collection)
            ax.add_collection(LineCollection(sample.edges, colors="#212121", linewidths=0.4))
            ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", xlabel="x [m]", ylabel="y [m]")
            if component == 0:
                ax.set_title(title, fontsize=11, pad=12)
        colorbar = fig.colorbar(collection, ax=axes[component, :], fraction=0.025, pad=0.015)
        ticks = bound * np.array([-1, -0.1, -0.01, 0, 0.01, 0.1, 1])
        colorbar.set_ticks(ticks, labels=[f"{value:.1e}" if value else "0" for value in ticks])
        colorbar.minorticks_off()
        colorbar.set_label(f"Flux {'xy'[component]} [m/s]")
    fig.suptitle(
        f"{coarse['kind']}, pressure degree {coarse['pressure_degree']}; "
        f"fine refinement factor {coarse['fine_factor']}\n"
        "One-sided top-face samples; actual macro edges; common symmetric log scales",
        fontsize=13,
    )
    _save(fig, f"{coarse['kind']}-p{coarse['pressure_degree']}-trace-comparison")


def main() -> None:
    """Regenerate geometry and measured-error figures from validated acquisition records."""
    records = read_records()
    expected = {
        (kind, degree, fine, macro)
        for kind, degree in [("prism", 1), ("tetrahedron", 1), ("tetrahedron", 2)]
        for fine, macro in [(1, 1), (2, 1), (2, 2), (4, 1), (4, 2), (4, 4)]
    }
    present = {
        (r["kind"], r["pressure_degree"], r["fine_factor"], r["macro_factor"]) for r in records
    }
    if not expected <= present:
        raise ValueError("the figure gallery requires the complete 18-case mixed-well campaign")
    geometry()
    convergence(records)
    rates = boundary_rates(records)
    (OUT / "boundary-rates.json").write_text(json.dumps(rates, indent=2) + "\n")
    for name in (
        "native-basis-verification.json",
        "native-solution-verification.json",
        "fine-space-separation.json",
        "replay-verification.json",
    ):
        if (DATA / name).exists():
            (OUT / name).write_bytes((DATA / name).read_bytes())
    for kind, degree in [("prism", 1), ("tetrahedron", 1), ("tetrahedron", 2)]:
        rows = [
            r
            for r in records
            if r["kind"] == kind and r["pressure_degree"] == degree and r["macro_factor"] == 1
        ]
        if rows:
            coarse = max(rows, key=lambda r: r["fine_factor"])
            fields(coarse)
            classical = next(
                row
                for row in records
                if row["kind"] == kind
                and row["pressure_degree"] == degree
                and row["fine_factor"] == coarse["fine_factor"]
                and row["macro_factor"] == row["fine_factor"]
            )
            trace_comparison(coarse, classical)


if __name__ == "__main__":
    main()
