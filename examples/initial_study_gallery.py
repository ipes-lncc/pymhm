"""Render immutable scalar observations and the initial-study catalogue.

This consumer does not assemble operators or solve field equations. Physical
errors and numerical increments use separate labelled panels. Input digests
identify the original acquisitions; consumer metadata does not retag them.
Matplotlib is imported only when rendering a scientific figure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "pressure_l2": "Pressure L2 error",
    "flux_l2": "Darcy flux L2 error",
    "divergence_l2": "Darcy flux divergence L2 error",
    "gradient_l2": "Raw pressure gradient L2 error",
    "gradient_broken_l2": "Broken pressure gradient L2 error",
    "pressure_h1_seminorm": "Pressure H1 seminorm error",
    "error_l2": "Scalar L2 error",
    "scalar_l2": "Scalar L2 error",
    "error_energy": "Diffusion-weighted gradient error",
    "error_v": "RAD V norm error",
}
REFERENCES = {
    "mh": "Robin hybrid method, doi:10.1137/22M1542556, Section 4.1",
    "mh2m": "de Barros, Madureira and Valentin, arXiv:2404.16978v3, Section 8.1",
    "tensor-rt": "Duran et al. (2019), analytical tensor RT family",
}


def digest(path: Path) -> str:
    """Return the SHA256 identity of complete immutable input bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_input(path: Path, expected: str) -> dict[str, Any]:
    """Read a numerical record only when its supplied acquisition digest matches."""
    if digest(path) != expected:
        raise ValueError(f"Numerical input digest differs: {path}")
    record: dict[str, Any] = json.loads(path.read_text())
    return record


def plot_panels(record: dict[str, Any], output: Path) -> None:
    """Render stated scalar series without interpolating or reacquiring fields.

    Each panel declares its metric and each curve its acquisition series.
    Values must be positive finite observations for logarithmic axes.
    """
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    panels = record["panels"]
    columns = min(2, len(panels))
    figure, axes = plt.subplots(
        (len(panels) + columns - 1) // columns,
        columns,
        figsize=(8.8, 3.4 * ((len(panels) + columns - 1) // columns)),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, panel in zip(axes.flat, panels, strict=False):
        for series in panel["series"]:
            if any(value <= 0 for value in (*series["x"], *series["y"])):
                raise ValueError("Logarithmic panels require positive observations")
            axis.loglog(series["x"], series["y"], "o-", label=series["label"])
        axis.set_xlabel(panel["xlabel"])
        axis.set_ylabel(panel["ylabel"])
        axis.set_title(panel["title"], fontsize=10)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
        ticks = sorted({x for series in panel["series"] for x in series["x"]})
        axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
        axis.minorticks_off()
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_visible(False)
    figure.suptitle(record["title"], fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def collect_scalar(specification: dict[str, Any]) -> dict[str, Any]:
    """Preserve accepted scalar rows, original provenance and corrected attribution.

    The full original-equation gate remains 1e-10 and the physical quadrature
    sensitivity gate remains 1e-6. This format contains norms, not replayable
    coefficient vectors. Three observations do not establish uniform stability.
    """
    inputs, studies = [], []
    for item in specification["inputs"]:
        path = ROOT / item["path"]
        record = checked_input(path, item["sha256"])
        if not record["complete"] or len(record["rows"]) < 3:
            raise ValueError("A complete initial scalar series requires three accepted levels")
        for source, expected in record["source_sha256"].items():
            if source.startswith("src/pymhm/") and digest(ROOT / source) != expected:
                raise ValueError("The accepted numerical core differs from the current core")
        for row in record["rows"]:
            original = row["original_equations"]
            full = original["full_uncondensed_relative_to_physical_rhs"]["relative"]
            if not 0 <= full <= 1e-10 or not 0 <= row["quadrature_relative_change"] <= 1e-6:
                raise ValueError("Original-equation or physical quadrature gate failed")
        resource = checked_input(ROOT / item["resource"], item["resource_sha256"])
        if not resource["execution_accepted"] or resource["owned_processes_remaining"]:
            raise ValueError("The owned acquisition did not close successfully")
        conventions = dict(record["conventions"])
        if record["case"] in REFERENCES:
            problem = conventions["problem"].split(" ", 1)[1]
            conventions["problem"] = problem
            conventions["literature_attribution"] = REFERENCES[record["case"]]
        studies.append(
            {
                "case": record["case"],
                "variant": record["variant"],
                "acquisition_uuid": record["acquisition_uuid"],
                "producer_source_sha256": record["source_sha256"],
                "conventions": conventions,
                "norm_orders": record["norm_orders"],
                "rows": record["rows"],
                "owned_elapsed_seconds": resource["elapsed_seconds"],
                "owned_peak_rss_bytes": resource["peak_owned_rss_bytes"],
            }
        )
        inputs.append(item)
    metrics = sorted({metric for study in studies for metric in study["rows"][0]["norms"]})
    panels = []
    for metric in metrics:
        series = []
        for study in studies:
            if metric not in study["rows"][0]["norms"]:
                continue
            series.append(
                {
                    "label": f"{study['case']} / {study['variant']}",
                    "x": [row["level"] for row in study["rows"]],
                    "y": [row["norms"][metric] for row in study["rows"]],
                }
            )
        panels.append(
            {
                "title": LABELS.get(metric, metric),
                "xlabel": "Declared macro resolution n",
                "ylabel": LABELS.get(metric, metric),
                "series": series,
            }
        )
    return {
        "schema": "pymhm-initial-scalar-publication-v1",
        "title": specification["title"],
        "id": specification["id"],
        "studies": studies,
        "inputs": inputs,
        "panels": panels,
        "consumer_sha256": digest(Path(__file__)),
        "pde_solves_during_collection": 0,
        "coefficient_vectors_persisted": False,
        "historical_literature_reproduction": False,
        "uniform_stability_claimed": False,
        "limitations": specification["limitations"],
        "norm_convention": "Separate physical field errors and original Euclidean coefficient rows",
    }


def render_catalogue(entries: list[dict[str, Any]], path: Path) -> None:
    """Write a Markdown catalogue with immutable public-record and figure identities."""
    lines = [
        "## Numerical catalogue",
        "",
        "| Study | Refinement | Observed conclusion |",
        "| --- | --- | --- |",
    ]
    for entry in entries:
        record = ROOT / entry["public_record"]
        figure = ROOT / entry["figure"]
        if digest(record) != entry["record_sha256"] or digest(figure) != entry["figure_sha256"]:
            raise ValueError("Public numerical record or inspected figure changed")
        lines.append(
            f"| [{entry['title']}](#{entry['id']}) | {entry['refinement']} | {entry['summary']} |"
        )
    for entry in entries:
        lines.extend(
            [
                "",
                f'<a id="{entry["id"]}"></a>',
                "",
                f"### {entry['title']}",
                "",
                entry["summary"],
                "",
                entry["limitations"],
                "",
                f"![{entry['title']}](../{Path(entry['figure']).relative_to('docs').as_posix()})",
                "",
                f"[Numerical record](https://github.com/volpatto/pymhm/blob/main/{entry['public_record']})",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Render one scalar specification or a hash-checked complete catalogue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalogue", action="store_true")
    args = parser.parse_args()
    specification = json.loads(args.manifest.read_text())
    if args.catalogue:
        render_catalogue(specification["entries"], args.output)
    else:
        record = collect_scalar(specification)
        args.output.mkdir(parents=True, exist_ok=True)
        plot_panels(
            record, ROOT / "docs/figures/minimal-convergence" / record["id"] / "convergence.png"
        )
        (args.output / "study.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
