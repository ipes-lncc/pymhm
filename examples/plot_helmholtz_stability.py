"""Published Helmholtz pollution diagnostics with independently projected exact traces."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.helmholtz_threshold import sampled_threshold
from examples.plot_helmholtz import save
from examples.plot_style import set_refinement_ticks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/helmholtz-stability"


def main(data: Path = DATA, output: Path | None = None) -> None:
    """Plot all four published frequencies and distinguish sampled thresholds from bounds."""
    configurations = ((0, 10, 128), (0, 20, 256), (1, 15, 128), (1, 75, 512))
    records = []
    for ell, frequency, finest in configurations:
        record = json.loads((data / f"ell{ell}-frequency{frequency}.json").read_text())
        if record["rows"][-1]["n"] < finest:
            raise ValueError(f"the ell={ell}, frequency={frequency} sequence is incomplete")
        records.append(record)
    output = ROOT / "docs/figures/helmholtz" if output is None else output
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for ax, record in zip(axes.ravel(), records, strict=True):
        rows = record["rows"]
        ell = record["ell"]
        h = np.array([row["macro_edge_length"] for row in rows])
        suffix = "relative" if ell == 0 else "l2"
        actual = np.array([row[f"mhm_gradient_{suffix}"] for row in rows])
        interpolated = np.array([row[f"interpolated_gradient_{suffix}"] for row in rows])
        ax.loglog(h, actual, "o-", color="#0072B2", ms=4, label="MHM error")
        ax.loglog(h, interpolated, "s--", color="#D55E00", ms=4, label="Exact-flux projection")
        ax.loglog(
            h,
            interpolated[-1] * (h / h[-1]) ** (ell + 1),
            ":",
            color=".3",
            label=rf"$H^{ell + 1}$",
        )
        ax.set(
            xlabel="Macro edge length H",
            ylabel="Relative gradient L2 error" if ell == 0 else "Absolute gradient L2 error",
            title=rf"$\ell={ell}$, $\omega={record['omega'] / np.pi:g}\pi$",
        )
        ax.invert_xaxis()
        set_refinement_ticks(ax, h, [f"1/{row['n']}" for row in rows], max_labels=4)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    save(fig, output, "stability-errors")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    summary = []
    for ax, record in zip(axes.ravel(), records, strict=True):
        rows = record["rows"]
        h = np.array([row["macro_edge_length"] for row in rows])
        ratio = np.array([row["ratio"] for row in rows])
        ax.semilogx(h, ratio, "o-", color="#0072B2", ms=4, label="Measured ratio")
        ax.axhline(3, ls="--", color="#D55E00", label="Published factor 3")
        threshold = sampled_threshold(rows, record.get("excluded_settings", ()))
        bracket = threshold["transition_bracket"]
        if bracket is not None:
            ax.axvspan(*bracket, color=".8", alpha=0.5, label="Sampled transition interval")
        for index, rejected in enumerate(record.get("excluded_settings", ())):
            ax.axvline(
                1 / rejected["n"],
                color="#CC79A7",
                ls=":",
                label="Rejected local inverse" if index == 0 else None,
            )
        ax.set(
            xlabel="Macro edge length H",
            ylabel=r"$E_{\mathrm{MHM}}/E_{\mathrm{INT}}$",
            title=rf"$\ell={record['ell']}$, $\omega={record['omega'] / np.pi:g}\pi$",
        )
        ax.invert_xaxis()
        set_refinement_ticks(ax, h, [f"1/{row['n']}" for row in rows], max_labels=4)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        summary.append(
            {
                "ell": record["ell"],
                "omega": record["omega"],
                "levels": len(rows),
                **threshold,
                "maximum_ratio": float(ratio.max()),
                "finest_ratio": float(ratio[-1]),
                "local_space": record["local_space"],
            }
        )
    save(fig, output, "stability-ratio")
    (data / "comparison.json").write_text(
        json.dumps(
            {
                "definition": (
                    "Equation6.3 tested on the recorded finite sequence; no unsampled guarantee"
                ),
                "sequences": summary,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "stability-comparison.json").write_bytes((data / "comparison.json").read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.data, args.output)
