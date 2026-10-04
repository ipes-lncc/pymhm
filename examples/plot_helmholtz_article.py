"""Complete published angular sampling and analytical convergence at matched trace degrees."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.helmholtz_article import publication_rows as article_rows
from examples.helmholtz_local_control import publication_rows as local_rows
from examples.helmholtz_publication import support_rows
from examples.plot_helmholtz import save
from examples.plot_style import set_refinement_ticks

ROOT = Path(__file__).resolve().parents[1]


def main(data: Path | None = None, output: Path | None = None) -> None:
    """Render all acquired directions and spatial levels with explicit local controls."""
    data = ROOT / "examples/results/helmholtz-article" if data is None else data
    rows = article_rows(data)
    fine_rows = local_rows(data)
    published, native_rows = support_rows(data, rows)
    output = ROOT / "docs/figures/helmholtz" if output is None else output
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), layout="constrained")
    for ax, ell, n, omega in zip(axes, (2, 4), (11, 21), (20, 40), strict=True):
        for basis, style in (("polynomial", "-"), ("oscillatory", "--")):
            part = sorted(
                (
                    r
                    for r in rows
                    if r["study"] == "direction" and r["ell"] == ell and r["trace_basis"] == basis
                ),
                key=lambda r: r["angle"],
            )
            ax.plot(
                [r["angle"] for r in part],
                [r["gradient_relative_error"] for r in part],
                style,
                label=basis,
            )
        ax.set(
            xlabel=r"Propagation angle $\theta$",
            ylabel=r"$|u-u_H|_{1,\mathcal{T}_H}/|u|_{1,\Omega}$",
            title=rf"$\ell={ell}$, $\omega={omega}\pi$, $H=1/{n}$",
        )
        ax.set_xticks(
            [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2],
            ["0", r"$\pi/6$", r"$\pi/4$", r"$\pi/3$", r"$\pi/2$"],
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.grid(alpha=0.25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.suptitle("256 directions per configuration; local Qℓ₊₂ on 2×2 cells")
    save(fig, output, "article-directions")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), layout="constrained")
    for row, ell in zip(axes, (2, 3), strict=True):
        for ax, key, label, power in zip(
            row,
            ("pressure_relative_error", "gradient_relative_error"),
            (r"$\|u-u_H\|_0/\|u\|_0$", r"$|u-u_H|_{1,\mathcal{T}_H}/|u|_{1,\Omega}$"),
            (ell + 2, ell + 1),
            strict=True,
        ):
            for basis, style, color in (
                ("polynomial", "o-", "#0072B2"),
                ("oscillatory", "s--", "#D55E00"),
            ):
                part = sorted(
                    (
                        r
                        for r in rows
                        if r["study"] == "convergence"
                        and r["ell"] == ell
                        and r["trace_basis"] == basis
                    ),
                    key=lambda r: r["n"],
                )
                h = np.array([r["macro_edge_length"] for r in part])
                e = np.array([r[key] for r in part])
                rate = np.log(e[-2] / e[-1]) / np.log(h[-2] / h[-1])
                ax.loglog(
                    h,
                    e,
                    style,
                    color=color,
                    ms=4,
                    label=f"PyMHM {basis}; rate {rate:.2f}",
                )
                printed = sorted(
                    (
                        r
                        for r in published
                        if r["ell"] == ell
                        and r["basis"] == basis
                        and r["field"] == key.split("_")[0]
                    ),
                    key=lambda r: r["n"],
                )
                ordinate = np.array([r["published_graph_value"] for r in printed])
                bounds = np.array([r["published_graph_interval"] for r in printed])
                ax.errorbar(
                    [r["H"] for r in printed],
                    ordinate,
                    yerr=np.array([ordinate - bounds[:, 0], bounds[:, 1] - ordinate]),
                    fmt="o:",
                    color=color,
                    mfc="white",
                    ms=4,
                    capsize=2,
                    label=f"Article {basis}",
                )
                native = sorted(
                    (
                        r
                        for r in native_rows
                        if r["ell"] == ell and r["oscillatory"] == (basis == "oscillatory")
                    ),
                    key=lambda r: r["n"],
                )
                ax.loglog(
                    [1 / r["n"] for r in native],
                    [r[f"native_relative_{key.split('_')[0]}_l2"] for r in native],
                    "x",
                    color=color,
                    ms=7,
                    mew=1.1,
                    label=f"DOLFINx {basis}",
                )
            ax.loglog(h, e[-1] * (h / h[-1]) ** power, ":", color=".3", label=rf"$H^{power}$")
            set_refinement_ticks(ax, h, [f"1/{r['n']}" for r in part])
            ax.set(xlabel="Macro edge length H", ylabel=label, title=rf"$\ell={ell}$")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2)
    fig.suptitle(r"Plane wave: $\omega=10\pi$, $\theta=\pi/13$")
    save(fig, output, "article-convergence")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6), layout="constrained")
    for ax, ell in zip(axes, (2, 4), strict=True):
        for basis, marker, color in (
            ("polynomial", "o", "#0072B2"),
            ("oscillatory", "s", "#D55E00"),
        ):
            for refinement, linestyle in ((2, "-"), (4, "--")):
                part = sorted(
                    (
                        r
                        for r in rows
                        if r["study"] == "local-control"
                        and r["ell"] == ell
                        and r["trace_basis"] == basis
                        and r["refinement"] == refinement
                    ),
                    key=lambda r: r["angle"],
                )
                ax.semilogy(
                    [r["angle"] for r in part],
                    [r["gradient_relative_error"] for r in part],
                    marker=marker,
                    ls=linestyle,
                    color=color,
                    label=f"{basis}; r={refinement}",
                )
            fine = next(r for r in fine_rows if r["ell"] == ell and r["trace_basis"] == basis)
            ax.semilogy(
                [fine["angle"]],
                [fine["gradient_relative_error"]],
                marker="*",
                ms=11,
                ls="none",
                color=color,
                label=f"{basis}; r=8",
            )
        ax.set(
            xlabel=r"$\theta$",
            ylabel="Relative gradient L2 error",
            title=rf"Local-resolution control: $\ell={ell}$",
        )
        ax.set_xticks(
            [0, np.pi / 13, np.pi / 4, np.pi / 2], ["0", r"$\pi/13$", r"$\pi/4$", r"$\pi/2$"]
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, output, "article-local-control")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.data, args.output)
