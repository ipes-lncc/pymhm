"""Plot archived independent MHM fields without importing reference solver code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import AsinhNorm
from threadpoolctl import threadpool_limits

from examples.archive_precision import restore_precision
from examples.plot_mesh import draw_macro_mesh
from examples.spe10_adaptive_norms import BrokenP2
from pymhm.lagrange import reference_basis

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/spe10-adaptive/published"
FIGURES = ROOT / "docs/figures/spe10-adaptive"


def digest(path: Path) -> str:
    """Verify each archived field against its acquisition record."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PressureView:
    """Expose only physical geometry, material and an independently identified P2 pressure.

    Reconstructed RT2 flux and residual indicators have separate acquisition
    contracts and are never inferred from an overridden pressure vector.
    """

    def __init__(self, geometry: BrokenP2, pressure: tuple[np.ndarray, ...]) -> None:
        """Keep deterministic geometry maps and one-sided coefficients in their executed basis."""
        self.macro = geometry.macro
        self.meshes = geometry.meshes
        self.material = geometry.material
        self.dofs = geometry.dofs
        self.geometry = geometry.geometry
        self.pressure = pressure
        self._geometry_owner = geometry

    def locate(self, points: np.ndarray) -> np.ndarray:
        """Locate samples on the unchanged explicit macro geometry."""
        return self._geometry_owner.locate(points)


def pressure_view(candidate: BrokenP2, archive: Path, expected_sha256: str) -> PressureView:
    """Load a separately archived current pressure after digest, coordinate and shape checks."""
    if digest(archive) != expected_sha256:
        raise ValueError("current pressure archive differs from its acquisition record")
    count = len(candidate.meshes)
    with np.load(archive, allow_pickle=False) as values:
        if not np.array_equal(values["macro_points"], candidate.macro.points) or not np.array_equal(
            values["macro_cells"], candidate.macro.cells
        ):
            raise ValueError("current pressure macro coordinates or connectivity differ")
        if values["local_points"].shape[0] != count or values["local_cells"].shape[0] != count:
            raise ValueError("current pressure archive has an incompatible macro count")
        if any(
            not np.array_equal(values["local_points"][i], mesh.points)
            or not np.array_equal(values["local_cells"][i], mesh.cells)
            for i, mesh in enumerate(candidate.meshes)
        ):
            raise ValueError("current pressure local coordinates or connectivity differ")
        pressure = tuple(
            restore_precision(
                values[f"pressure{i}"],
                values[f"pressure{i}_correction"],
                values[f"pressure{i}_tail"],
            )
            for i in range(count)
        )
        if any(
            p.shape != old.shape or not np.isfinite(p).all()
            for p, old in zip(pressure, candidate.pressure, strict=True)
        ):
            raise ValueError("current pressure coefficient shape or values are invalid")
    return PressureView(candidate, pressure)


def samples(
    candidate: BrokenP2 | PressureView, native: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate pressure and signed raw flux on explicit incident fine triangles.

    The native archive has already been mapped to the same equispaced P2 nodal
    basis. Display points are independent raster-cell centers, without nodal
    averaging or field interpolation across macro or material interfaces.
    """
    owners = np.concatenate(
        [candidate.locate(points[first : first + 1024]) for first in range(0, len(points), 1024)]
    )
    order = np.argsort(owners, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(owners[order])) + 1)
    result = [np.empty((len(points), 3), dtype=np.longdouble) for _ in range(2)]
    for indices in groups:
        cell = owners[indices[0]]
        mesh = candidate.meshes[cell]
        vertices = mesh.points[mesh.cells]
        inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        coordinate = np.einsum(
            "tab,ntb->nta", inverse, points[indices, None] - vertices[None, :, 0]
        )
        score = np.minimum(coordinate.min(axis=2), 1 - coordinate.sum(axis=2))
        fine = np.argmax(score, axis=1)
        if np.any(score[np.arange(len(indices)), fine] < -1e-10):
            raise ValueError("display sample is outside its incident triangle")
        local = coordinate[np.arange(len(indices)), fine]
        bary = np.column_stack((1 - local.sum(axis=1), local))
        basis, derivative, _ = reference_basis(2, bary)
        material = candidate.material(points[indices])
        for output, coefficients in zip(result, (native, candidate.pressure), strict=True):
            values = coefficients[cell][candidate.dofs[cell][fine]]
            output[indices, 0] = np.einsum("qi,qi->q", basis, values)
            gradient = np.einsum(
                "qin,qna,qi->qa", derivative, candidate.geometry[cell][fine], values
            )
            output[indices, 1:] = -np.einsum("qab,qb->qa", material, gradient)
    return result[0], result[1]


def plot(level: int = 6, output: Path = FIGURES) -> None:
    """Compare accepted native MHM pressure/flux components on common physical scales."""
    record = json.loads((DATA / "native-system-verification.json").read_text())
    row = next(item for item in record["rows"] if item["level"] == level)
    path = DATA / f"mhm-level{level}.npz"
    native_path = DATA / row["native_pressure_archive"]
    if (
        not row["accepted"]
        or digest(path) != row["candidate_archive_sha256"]
        or digest(native_path) != row["native_pressure_archive_sha256"]
    ):
        raise ValueError("the field data differ from the accepted comparison")
    candidate: BrokenP2 | PressureView = BrokenP2(path)
    current = row.get("candidate_pressure_archive")
    if current is not None:
        candidate = pressure_view(
            candidate, DATA / current, row["candidate_pressure_archive_sha256"]
        )
    with np.load(native_path) as arrays:
        native = restore_precision(
            arrays["pressure"], arrays["pressure_correction"], arrays["pressure_tail"]
        )
    nx, ny = 240, 440
    x = (np.arange(nx) + 0.5) * 1200 / nx
    y = (np.arange(ny) + 0.5) * 2200 / ny
    xx, yy = np.meshgrid(x, y)
    reference, current = samples(candidate, native, np.column_stack((xx.ravel(), yy.ravel())))
    plt.rcParams.update({"font.size": 10})
    figure, axes = plt.subplots(3, 3, figsize=(11.5, 15.8), layout="constrained")
    names = (r"Pressure $p$", r"Flux $q_x$", r"Flux $q_y$")
    for component, name in enumerate(names):
        field = (reference[:, component], current[:, component])
        common = (float(min(v.min() for v in field)), float(max(v.max() for v in field)))
        if component:
            maximum = max(abs(v) for v in common)
            common = (-maximum, maximum)
        difference = current[:, component] - reference[:, component]
        bound = float(np.max(abs(difference)))
        for column, values in enumerate((*field, difference)):
            axis = axes[component, column]
            vmin, vmax = (-bound, bound) if column == 2 else common
            signed = bool(component or column == 2)
            options = dict(cmap="seismic" if signed else "viridis")
            if component and column != 2:
                options["norm"] = AsinhNorm(linear_width=common[1] / 1000, vmin=vmin, vmax=vmax)
            else:
                options.update(vmin=vmin, vmax=vmax)
            artist = axis.imshow(
                np.asarray(values.reshape(ny, nx), dtype=float),
                origin="lower",
                extent=(0, 1200, 0, 2200),
                interpolation="nearest",
                aspect="equal",
                **options,
            )
            macro = draw_macro_mesh(axis, candidate.macro)
            macro.set_linewidth(0.15)
            macro.set_alpha(0.3)
            macro.set_path_effects(
                [
                    patheffects.Stroke(linewidth=0.35, foreground="white", alpha=0.2),
                    patheffects.Normal(),
                ]
            )
            macro.set_rasterized(True)
            title = ("DOLFINx/UFL MHM", "PyMHM", "PyMHM − DOLFINx/UFL")[column]
            axis.set(title=f"{title}\n{name}", xlabel="x (ft)", ylabel="y (ft)")
            axis.set_xticks([0, 600, 1200])
            axis.set_yticks([0, 1100, 2200])
            bar = figure.colorbar(
                artist, ax=axis, orientation="horizontal", pad=0.035, fraction=0.04
            )
            if component and column != 2:
                bar.set_ticks([-common[1], -common[1] / 100, 0, common[1] / 100, common[1]])
                bar.ax.xaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1g}")
                )
            else:
                bar.locator = matplotlib.ticker.MaxNLocator(3)
                bar.update_ticks()
            bar.ax.tick_params(labelsize=8)
    figure.suptitle(
        f"SPE10 layer 36 · {len(candidate.macro.cells)} macrotriangles\n"
        "Same MHM discretization: local P2 / four triangles / P0 trace",
        fontsize=13,
    )
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(output / f"native-components-level{level}.{extension}", dpi=180)
    plt.close(figure)


def main() -> None:
    """Render either accepted adaptive state from portable numerical archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, choices=(0, 6), default=6)
    args = parser.parse_args()
    with threadpool_limits(1):
        plot(args.level)


if __name__ == "__main__":
    main()
