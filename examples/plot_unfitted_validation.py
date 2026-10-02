"""Replay published gradient comparisons and one-sided physical flux samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from examples.layered_poisson import LayeredPoissonSeries
from examples.plot_mesh import draw_macro_mesh, macro_profile_breaks
from examples.unfitted_campaign import macro_mesh, save
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.cut_cells import cartesian_trace_values, fit_material_faces
from pymhm.reservoir import CartesianCellField

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/unfitted"


def geometry_verification(perturbations: list[float]) -> dict:
    """Record the explicit macro map and independent material-subsegment degrees."""
    field = CartesianCellField(np.array([[10.0, 1.0]]), (1.0, 0.5))
    rows = []
    for delta in perturbations:
        mesh = macro_mesh(delta)
        original = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
        fitted = fit_material_faces(original, field)
        points, face_ids = [], []
        for index, (nodes, space) in enumerate(zip(mesh.faces, fitted.faces, strict=True)):
            if len(space.breaks) > 2:
                start, end = mesh.points[nodes]
                points.extend(
                    (start + np.asarray(space.breaks[1:-1])[:, None] * (end - start)).tolist()
                )
                face_ids.append(index)
        rows.append(
            {
                "delta": delta,
                "macro_triangles": len(mesh.cells),
                "macro_edges": len(mesh.faces),
                "trace_dofs_S1": original.size,
                "trace_dofs_S2": fitted.size,
                "retained_constants": len(mesh.cells),
                "centers_vertical_shift": (
                    mesh.points[9:, 1] - macro_mesh(0).points[9:, 1]
                ).tolist(),
                "new_breakpoints": points,
                "split_faces": face_ids,
                "split_boundary_faces": int(np.count_nonzero(mesh.face_cells[face_ids, 1] == -1)),
                "degrees_per_face": [list(space.degrees) for space in fitted.faces],
                "continuous_within_face": [space.continuous for space in fitted.faces],
            }
        )
    return {
        "comparison": (
            "L10 Sections 3.2 and 5.2, Figure 4; DOF counts inferred from the stated spaces, "
            "not tabulated by the article"
        ),
        "macro_map": (
            "Bottom/top rows fixed; middle row moves by delta; "
            "the four crisscross centers move by delta/2"
        ),
        "skeletal_space": (
            "Independent degree-2 polynomial per material subsegment, "
            "without endpoint-continuity constraints"
        ),
        "rows": rows,
    }


def read_flux(setting: str) -> dict[str, np.ndarray]:
    """Recover physical flux with the material side of each archived fine cell.

    The display archive has six P2 sampling nodes for each original fine
    triangle. Its independent P4 gradients are retained at those nodes.
    Samples on a material boundary use the incident triangle, without averaging.
    """
    with np.load(DATA / f"{setting}-fitted-r16.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    points = data["points"]
    groups = points.reshape(-1, 6, 2)
    centers = groups.mean(axis=1)
    if np.any(
        (groups[:, :, 1].min(axis=1) < 0.5 - 1e-14) & (groups[:, :, 1].max(axis=1) > 0.5 + 1e-14)
    ):
        raise ValueError("display cell crosses an unresolved material interface")
    field = CartesianCellField(np.array([[10.0, 1.0]]), (1.0, 0.5))
    material = cartesian_trace_values(field, points, np.repeat(centers, 6, axis=0))
    data["material"] = material
    data["flux"] = -material[:, None] * data["gradient"]
    data["exact_flux"] = -material[:, None] * data["exact_gradient"]
    # The analytical normal flux is continuous. Its archived gradient uses the
    # point's material side; tangential flux instead uses the incident-cell side.
    point_material = np.where(points[:, 1] < 0.5, 10.0, 1.0)
    data["exact_flux"][:, 1] = -point_material * data["exact_gradient"][:, 1]
    return data


def published_comparison(rows: list[dict], published: dict) -> None:
    """Compare only explicitly printed paper values, retaining reference differences."""
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    x = np.asarray(published["perturbations_assumed"])
    for setting, marker in (("S1", "x"), ("S2", "o")):
        values = [row["gradient_l2"] for row in rows if row["setting"] == setting]
        axes[0].semilogx(x, values, marker + "-", label=f"PyMHM {setting}")
    axes[0].semilogx(x, published["S1"]["errors"], "k+--", label="L10 Fig. 5, S1 ticks")
    axes[0].axhline(rows[0]["gradient_l2"], color="C2", label="PyMHM S0")
    axes[0].axhline(published["S0"]["error"], color="black", ls=":", label="L10 Fig. 5, S0 tick")
    axes[0].set(xlabel="Macro perturbation δ", ylabel="Absolute broken-gradient L² error")
    fitted = json.loads((DATA / "mhm-fitted-r16.json").read_text())
    unfitted = json.loads((DATA / "mhm-r16.json").read_text())
    for table, label, style in (
        (fitted, "Material-fitted local P4", "o-"),
        (unfitted, "Unfitted local P4 + exact cuts", "x--"),
    ):
        values = [row["gradient_l2"] for row in table if row["setting"] == "S2"]
        axes[1].semilogx(x, values, style, label=label)
    axes[1].set(
        xlabel="Macro perturbation δ",
        ylabel="Absolute broken-gradient L² error",
        title="S2: separate local-approximation effect",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("PyMHM P4 / trace P2 · printed L10 values are rounded, not solver data")
    save(figure, "publication-comparison")


def physical_flux(fields: list[dict[str, np.ndarray]]) -> None:
    """Display signed raw flux and analytical differences without joining fine cells."""
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 8), layout="constrained")
    bound = max(float(np.abs(data["flux"][:, 1]).max()) for data in fields)
    error_bound = max(
        float(np.abs(data["flux"][:, 1] - data["exact_flux"][:, 1]).max()) for data in fields
    )
    for column, (setting, data) in enumerate(zip(("S0", "S1", "S2"), fields, strict=True)):
        grid = mtri.Triangulation(*data["points"].T, triangles=data["cells"])
        macro = TriangleMesh(data["macro_points"], data["macro_cells"])
        for row, (values, limit, title) in enumerate(
            (
                (data["flux"][:, 1], bound, "Raw physical qy"),
                (data["flux"][:, 1] - data["exact_flux"][:, 1], error_bound, "qy − analytical qy"),
            )
        ):
            axis = axes[row, column]
            artist = axis.tripcolor(
                grid,
                values,
                shading="gouraud",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                rasterized=True,
            )
            draw_macro_mesh(axis, macro)
            axis.axhline(0.5, color="green", lw=1)
            axis.set(title=f"PyMHM {setting} · {title}", xlabel="x", ylabel="y", aspect="equal")
            figure.colorbar(artist, ax=axis, orientation="horizontal", pad=0.1, label=title)
    save(figure, "normal-flux-validation")


def transmission(fields: list[dict[str, np.ndarray]]) -> None:
    """Show both material traces at y=1/2, including all raw normal-flux discrepancies."""
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    x = np.linspace(0, 1, 1001)
    _, gradient = LayeredPoissonSeries(10, 1023).evaluate(np.column_stack((x, x * 0 + 0.5)))
    for column, (setting, data) in enumerate(zip(("S0", "S1", "S2"), fields, strict=True)):
        points, flux = data["points"].reshape(-1, 6, 2), data["flux"].reshape(-1, 6, 2)
        coefficient = data["material"].reshape(-1, 6)[:, 0]
        labeled = set()
        for cell, group in enumerate(points):
            selected = np.flatnonzero(abs(group[:, 1] - 0.5) <= 1e-14)
            if len(selected) < 2 or np.ptp(group[selected, 0]) < 1e-14:
                continue
            selected = selected[np.argsort(group[selected, 0])]
            below = coefficient[cell] == 10
            color, name = (
                ("C0", "Lower material trace") if below else ("C1", "Upper material trace")
            )
            for component in range(2):
                axes[component, column].plot(
                    group[selected, 0],
                    flux[cell, selected, component],
                    color=color,
                    lw=0.8,
                    label=name if below not in labeled else None,
                )
            labeled.add(below)
        axes[0, column].plot(x, -10 * gradient[:, 0], "k--", lw=1, label="Analytical lower")
        axes[0, column].plot(x, -gradient[:, 0], "k:", lw=1, label="Analytical upper")
        axes[1, column].plot(x, -gradient[:, 1], "k--", lw=1, label="Analytical both sides")
        macro = TriangleMesh(data["macro_points"], data["macro_cells"])
        breaks = macro_profile_breaks(macro, np.array([0.0, 0.5]), np.array([1.0, 0.5]))
        for component in range(2):
            axis = axes[component, column]
            axis.plot(
                breaks, np.full(len(breaks), 0.02), "k|", transform=axis.get_xaxis_transform()
            )
            axis.set(
                xlabel="x at material interface y=1/2",
                ylabel=f"q{'xy'[component]}",
                title=f"PyMHM {setting} · {'tangential' if component == 0 else 'normal'} flux",
            )
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
    save(figure, "interface-flux-traces")


def main() -> None:
    """Validate archived samples and regenerate figures plus their physical diagnostics."""
    rows = json.loads((DATA / "mhm-fitted-r16.json").read_text())
    published = json.loads((DATA / "published-figure5.json").read_text())
    fields = [read_flux(setting) for setting in ("S0", "S1", "S2")]
    metrics = []
    for setting, data in zip(("S0", "S1", "S2"), fields, strict=True):
        path = DATA / f"{setting}-fitted-r16.npz"
        metrics.append(
            {
                "setting": setting,
                "archive": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sampled_flux_component_error_linf": np.abs(data["flux"] - data["exact_flux"])
                .max(axis=0)
                .tolist(),
                "pressure_sample_min": float(data["values"].min()),
                "pressure_sample_max": float(data["values"].max()),
            }
        )
    report = {
        "quantity": "Pointwise archived display samples; these are not integrated norms",
        "coefficient_trace": "Incident material side, without coordinate displacement or averaging",
        "raw_flux_contract": (
            "-a grad(p_h), not an H(div) reconstruction or the conservative skeleton multiplier"
        ),
        "rows": metrics,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_record_sha256": {
            name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
            for name in ("published-figure5.json", "mhm-fitted-r16.json", "mhm-r16.json")
        },
    }
    (DATA / "flux-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    geometry = geometry_verification(published["perturbations_assumed"])
    geometry["source_sha256"] = report["source_sha256"]
    (DATA / "geometry-verification.json").write_text(json.dumps(geometry, indent=2) + "\n")
    published_comparison(rows, published)
    physical_flux(fields)
    transmission(fields)


if __name__ == "__main__":
    main()
